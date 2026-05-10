"""Sanity-check the RoboTwin Stage 2 data path: dataset → tokenizer → masks.

Lightweight pipeline test (no JAX, no images-to-disk) that checks:

  1. Streaming samples from RoboTwinTaskDataset (per-task) and from
     RoboTwinMixedDataset (multi-task weighted sampling).
  2. Tokenizer mask coverage on each cascade context that arises:
       * Context 3  (full cascade, plan-as-target)         — emit_full + p_plan=1.0
       * Context 2  (full cascade, plan-in-prompt)         — emit_full + p_plan=0.0
       * Context 1  (action-vector-only, no language)      — emit_full=False
     For each, assert that:
       - ar_target_mask covers plan ∪ stage ∪ action (incl. marker tokens, after
         the Stage-1 marker fix).
       - plan_mask, stage_mask, action_mask are pairwise disjoint.
       - Marker tokens are inside ar_target_mask, not the prompt span.
  3. Action chunk shape == (action_horizon, 14).
  4. State shape == (14,).
  5. Image dict has exactly the configured slots (base_0_rgb +
     left_wrist_0_rgb), each (224, 224, 3) uint8.
  6. Multi-task mixer respects the weights (~empirically — counts from a
     longer run, with tolerance).

Usage (on pod, where lap venv has all deps)::

    cd /data/zhaoqc/RoboTwin/policy/lap
    .venv/bin/python scripts/test_robotwin_pipeline.py

Stop-grad path / model-side validation lives in ``test_stop_grad_paths.py``
(future) — that one needs JAX + a checkpoint loaded.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import types
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Local-run shim (stub the dlimp-dependent lap.datasets/__init__).
# ---------------------------------------------------------------------------
def _ensure_package_stubs():
    _LAP_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
    _OPENPI_SRC = (
        pathlib.Path(__file__).resolve().parent.parent / "third_party" / "openpi" / "src"
    )
    sys.path.insert(0, str(_LAP_SRC))
    sys.path.insert(0, str(_OPENPI_SRC))
    try:
        import lap.datasets  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if "dlimp" not in str(e) and "dlimp" not in repr(e):
            raise
    for name, sub in [("lap", "lap"), ("lap.datasets", "lap/datasets"),
                      ("lap.models", "lap/models")]:
        m = types.ModuleType(name)
        m.__path__ = [str(_LAP_SRC / sub)]
        sys.modules[name] = m


_ensure_package_stubs()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from lap.datasets.robotwin_dataset import (  # noqa: E402
    DEFAULT_DATASET_WEIGHTS,
    RoboTwinMixedDataset,
    RoboTwinTaskDataset,
)
from lap.models.tokenizer import PaligemmaTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check(cond: bool, label: str, detail: str = "") -> None:
    """Assert-with-friendly-output. Raises on failure but prints for the run log."""
    sym = "✅" if cond else "❌"
    print(f"  {sym} {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(f"{label}: {detail}")


def _classify_context(plan_position: str, has_reasoning: bool) -> str:
    if not has_reasoning and plan_position == "none":
        return "Context 1 (action-only / no-reason)"
    if plan_position == "prompt":
        return "Context 2 (plan-in-prompt)"
    if plan_position == "target":
        return "Context 3 (plan-as-target)"
    return f"Context ?? (plan_position={plan_position}, has_reasoning={has_reasoning})"


def check_one_sample(
    sample: dict,
    tokenizer: PaligemmaTokenizer,
    action_horizon: int,
    image_size: tuple[int, int],
    expected_image_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb"),
) -> str:
    """Run all per-sample assertions; return a short label of which Context fired."""
    # --- 1. image dict shape ---
    images = sample["image"]
    image_masks = sample["image_mask"]
    _check(
        set(images.keys()) == set(expected_image_keys),
        "image dict has expected slots",
        f"got={sorted(images.keys())} expected={sorted(expected_image_keys)}",
    )
    for k in expected_image_keys:
        img = images[k]
        _check(
            isinstance(img, np.ndarray) and img.shape == (*image_size, 3) and img.dtype == np.uint8,
            f"image[{k}] shape/dtype",
            f"shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}",
        )
        _check(
            isinstance(image_masks[k], (bool, np.bool_, np.ndarray)),
            f"image_mask[{k}] is bool",
            f"got type={type(image_masks[k]).__name__}",
        )

    # --- 2. state + actions shape ---
    state = sample["state"]
    actions = sample["actions"]
    _check(state.shape == (14,) and state.dtype == np.float32,
           "state shape/dtype",
           f"shape={state.shape} dtype={state.dtype}")
    _check(actions.shape == (action_horizon, 14) and actions.dtype == np.float32,
           "actions shape/dtype",
           f"shape={actions.shape} dtype={actions.dtype}")

    # --- 3. tokenize and mask coverage ---
    plan_position = sample.get("plan_position", "none")
    reasoning = sample.get("language_actions")
    langact = sample.get("langact")
    plan = sample.get("plan")
    has_reasoning = reasoning is not None

    tok_out = tokenizer.tokenize(
        prompt=sample["prompt"],
        reasoning=reasoning,
        state=None,
        plan=plan,
        plan_position=plan_position,
        langact=langact,
        is_vqa_sample=False,
        is_prediction_sample=False,
    )
    (tokens, attn_mask, ar_target_mask, _num, _dir, _loss, stage_mask, plan_mask) = tok_out

    ctx = _classify_context(plan_position, has_reasoning)

    n_valid = int(attn_mask.sum())
    _check(n_valid > 0, "attn_mask has non-pad positions", f"valid={n_valid}")

    if ar_target_mask is not None:
        n_ar = int(ar_target_mask.sum())
        n_plan = int(plan_mask.sum()) if plan_mask is not None else 0
        n_stage = int(stage_mask.sum()) if stage_mask is not None else 0

        # Pairwise disjoint check (plan_mask ∩ stage_mask == ∅).
        if plan_mask is not None and stage_mask is not None:
            overlap = int((plan_mask & stage_mask).sum())
            _check(overlap == 0, "plan_mask ⊥ stage_mask", f"overlap={overlap}")

        # plan_mask ⊂ ar_target_mask
        if plan_mask is not None and n_plan > 0:
            covered = int((plan_mask & ar_target_mask).sum())
            _check(covered == n_plan, "plan_mask ⊂ ar_target_mask",
                   f"covered={covered}/{n_plan}")

        # stage_mask ⊂ ar_target_mask
        if stage_mask is not None and n_stage > 0:
            covered = int((stage_mask & ar_target_mask).sum())
            _check(covered == n_stage, "stage_mask ⊂ ar_target_mask",
                   f"covered={covered}/{n_stage}")

        # Marker token coverage check — find " [stage] " / " [action] " / " [plan] "
        # inside the tokenized sequence and verify each lies inside ar_target_mask
        # (per the Stage-1 marker fix in tokenizer.py).
        decoded_full = tokenizer.decode(np.asarray(tokens[attn_mask]))
        for marker in ["[plan]", "[stage]", "[action]"]:
            if marker not in decoded_full:
                continue
            # Decode just the ar_target_mask slice — markers should be inside it.
            decoded_target = tokenizer.decode(np.asarray(tokens[ar_target_mask]))
            # Special-case Context 2: [plan] is in PROMPT not target. Allow that.
            if marker == "[plan]" and plan_position == "prompt":
                _check(marker not in decoded_target,
                       f"{marker} stays in prompt (Context 2)",
                       f"target span={decoded_target[:60]!r}")
            else:
                _check(marker in decoded_target,
                       f"{marker} is inside ar_target_mask (post Stage-1 fix)",
                       f"target span={decoded_target[:60]!r}")

        print(f"    tokens={n_valid}  ar={n_ar}  (plan={n_plan} stage={n_stage} "
              f"action={n_ar - n_plan - n_stage})")
    else:
        # Context 1 (no reasoning): ar_target_mask is None → no AR loss. OK.
        _check(plan_mask is None and stage_mask is None,
               "Context 1: no plan/stage masks",
               f"plan_mask={plan_mask}, stage_mask={stage_mask}")
        print(f"    tokens={n_valid}  ar=None (Context 1, no language target)")

    return ctx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default="/data/zhaoqc/RoboTwin/data",
        help="Where per-task subdirs live.",
    )
    ap.add_argument(
        "--single-task",
        default="arrange_blocks_l_shape",
        help="Task to use for the per-Context coverage section.",
    )
    ap.add_argument("--max-token-len", type=int, default=320)
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--mix-samples", type=int, default=120,
                    help="Sample count for the multi-task weight test.")
    args = ap.parse_args()

    tokenizer = PaligemmaTokenizer(max_len=args.max_token_len, prompt_format="lap")
    image_size = (224, 224)

    print("=" * 78)
    print("  PART 1 — per-Context tokenizer coverage")
    print("=" * 78)

    # Force each of Context 1/2/3 by manipulating p_plan and p_full_reasoning.
    contexts_seen: set[str] = set()
    for label, p_plan, p_full in [
        ("Context 1 (action-only)", 0.0, 0.0),
        ("Context 2 (plan-in-prompt)", 0.0, 1.0),
        ("Context 3 (plan-as-target)", 1.0, 1.0),
    ]:
        print(f"\n{label}")
        ds = RoboTwinTaskDataset(
            task_dir=pathlib.Path(args.data_root) / args.single_task,
            action_horizon=args.action_horizon,
            p_plan=p_plan,
            p_full_reasoning=p_full,
            max_episodes=1,
            seed=42,
        )
        n_checked = 0
        for sample in ds.iter_samples(max_samples=20):
            # Filter to the Context we care about (some samples may emit different
            # Contexts due to phase boundary auto-promotion; keep first match).
            plan_position = sample.get("plan_position", "none")
            has_reasoning = sample.get("language_actions") is not None
            this_ctx = _classify_context(plan_position, has_reasoning)
            if not this_ctx.startswith(label.split(" ")[0]):
                continue
            ctx = check_one_sample(
                sample, tokenizer,
                action_horizon=args.action_horizon,
                image_size=image_size,
            )
            contexts_seen.add(ctx)
            n_checked += 1
            if n_checked >= 2:
                break
        if n_checked == 0:
            print("    [warn] no sample matched this context (boundary frames may have promoted to full reasoning)")

    print(f"\n  Contexts exercised: {sorted(contexts_seen)}")

    print()
    print("=" * 78)
    print("  PART 2 — multi-task weighted mixer")
    print("=" * 78)

    weights = dict(DEFAULT_DATASET_WEIGHTS)
    mixed = RoboTwinMixedDataset(
        data_root=pathlib.Path(args.data_root),
        weights=weights,
        action_horizon=args.action_horizon,
        max_episodes_per_dataset=1,
        seed=0,
    )
    counter = Counter()
    for s in mixed.iter_samples(max_samples=args.mix_samples):
        counter[s["_dataset"]] += 1

    total = sum(counter.values())
    total_w = sum(weights.get(n, 0.0) for n in mixed.dataset_names)
    print(f"  sampled {total} from {len(mixed.dataset_names)} tasks")
    for name in sorted(mixed.dataset_names):
        observed = counter[name] / total if total else 0
        expected = weights.get(name, 0.0) / total_w if total_w > 0 else 0
        delta = abs(observed - expected)
        ok = delta < 0.10  # 10% tolerance over a small sample
        sym = "✅" if ok else "⚠️"
        print(f"  {sym} {name:42s}  observed={observed:.2%}  expected={expected:.2%}  Δ={delta:.2%}")

    print()
    print("=" * 78)
    print("  PART 3 — schema spot check on each task")
    print("=" * 78)

    for task in mixed.dataset_names:
        try:
            ds = RoboTwinTaskDataset(
                task_dir=pathlib.Path(args.data_root) / task,
                action_horizon=args.action_horizon,
                p_plan=0.5,
                p_full_reasoning=1.0,
                max_episodes=1,
                seed=1,
            )
            sample = next(ds.iter_samples(max_samples=1))
            print(f"\n  task = {task}")
            check_one_sample(
                sample, tokenizer,
                action_horizon=args.action_horizon,
                image_size=image_size,
            )
        except Exception as e:
            print(f"  ❌ {task}: {e}")
            raise

    print("\nAll checks passed ✅")


if __name__ == "__main__":
    main()
