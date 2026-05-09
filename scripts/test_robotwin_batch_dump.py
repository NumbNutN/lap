"""End-to-end sanity check for the RoboTwin Stage 2 (action-expert) data path.

Loads samples from one or more RoboTwin task directories using the full
pipeline:
  1. ``RoboTwinTaskDataset`` / ``RoboTwinMixedDataset`` (HDF5 + metadata)
  2. Cascade extraction (task_prompt / synth-plan / subgoal_reasoning / phase_prompt)
  3. ``PaligemmaTokenizer.tokenize`` with the marker-fix (Stage 1 patch)
  4. Dump: head_camera + active wrist as PNG, mask coverage as TXT, action vector as JSON

Usage::

    cd /home/numbnut/worksapce/RoboTwin
    .venv/bin/python policy/lap/scripts/test_robotwin_batch_dump.py \\
        --data-root /home/numbnut/worksapce/RoboTwin/data \\
        --num-samples 4 \\
        --out-dir /tmp/robotwin_dump

Inspect ``/tmp/robotwin_dump/sample_NN.{base.png,wrist.png,txt,actions.json}``.

Notes:
* Locally the lap.datasets.__init__ depends on ``dlimp`` (RLDS), which is only
  installed in the lap pod venv. This script works around that with a stub
  to keep local smoke tests cheap.
* Action / state are emitted as float32 numpy. Pure data path test — no JAX.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import types

import numpy as np


# ---------------------------------------------------------------------------
# Path-stub gymnastics so we can run locally w/o the dlimp-dependent package init.
# When running on the pod (where the lap venv has all deps), this is a no-op.
# ---------------------------------------------------------------------------
def _ensure_package_stubs():
    _LAP_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
    _OPENPI_SRC = (
        pathlib.Path(__file__).resolve().parent.parent / "third_party" / "openpi" / "src"
    )
    sys.path.insert(0, str(_LAP_SRC))
    sys.path.insert(0, str(_OPENPI_SRC))
    # Try real import first; if it fails on dlimp, fall back to namespace stub.
    try:
        import lap.datasets  # noqa: F401
        return
    except ModuleNotFoundError as e:
        if "dlimp" not in str(e) and "dlimp" not in repr(e):
            raise
    lap_mod = types.ModuleType("lap")
    lap_mod.__path__ = [str(_LAP_SRC / "lap")]
    sys.modules["lap"] = lap_mod
    lap_datasets_mod = types.ModuleType("lap.datasets")
    lap_datasets_mod.__path__ = [str(_LAP_SRC / "lap" / "datasets")]
    sys.modules["lap.datasets"] = lap_datasets_mod
    lap_models_mod = types.ModuleType("lap.models")
    lap_models_mod.__path__ = [str(_LAP_SRC / "lap" / "models")]
    sys.modules["lap.models"] = lap_models_mod


_ensure_package_stubs()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from lap.datasets.robotwin_dataset import (  # noqa: E402
    DEFAULT_DATASET_WEIGHTS,
    RoboTwinMixedDataset,
    RoboTwinTaskDataset,
)
from lap.models.tokenizer import PaligemmaTokenizer  # noqa: E402


def _save_png(arr: np.ndarray, path: pathlib.Path) -> bool:
    try:
        from PIL import Image as PILImage
    except ImportError:
        try:
            import cv2
            cv2.imwrite(str(path), arr[..., ::-1])  # RGB→BGR
            return True
        except ImportError:
            return False
    PILImage.fromarray(arr).save(path)
    return True


def _decode_span(tokenizer: PaligemmaTokenizer, tokens: np.ndarray, mask: np.ndarray | None) -> str:
    if mask is None or int(mask.sum()) == 0:
        return "(empty)"
    return tokenizer.decode(np.asarray(tokens[mask], dtype=np.int32))


def _format_action_summary(actions: np.ndarray) -> str:
    """7-stat summary per dim of (H, D) action chunk."""
    lines = [f"shape={actions.shape}  dtype={actions.dtype}"]
    for d in range(actions.shape[-1]):
        col = actions[..., d]
        lines.append(
            f"  dim {d:2d}: min={col.min():+.4f} max={col.max():+.4f} "
            f"mean={col.mean():+.4f} std={col.std():.4f}"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-root", default="/home/numbnut/worksapce/RoboTwin/data",
        help="Where per-task subdirs live.",
    )
    ap.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional subset of task names. Default: package DEFAULT_DATASET_WEIGHTS keys.",
    )
    ap.add_argument(
        "--single-task", default=None,
        help="If set, bypass mixing and stream from this one task only "
             "(useful for per-task introspection).",
    )
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-token-len", type=int, default=320)
    ap.add_argument("--p-plan", type=float, default=0.5,
                    help="Plan-as-target probability (0.5 to dump both modes).")
    ap.add_argument("--p-full-reasoning", type=float, default=1.0,
                    help="Mid-phase frame full-cascade probability. Default 1.0 to "
                         "ensure every dumped sample has language to inspect.")
    ap.add_argument("--action-horizon", type=int, default=8)
    ap.add_argument("--max-episodes-per-dataset", type=int, default=2)
    ap.add_argument("--out-dir", default="/tmp/robotwin_dump")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] output dir: {out_dir}")

    # 1. Build dataset.
    if args.single_task is not None:
        ds = RoboTwinTaskDataset(
            task_dir=pathlib.Path(args.data_root) / args.single_task,
            action_horizon=args.action_horizon,
            p_plan=args.p_plan,
            p_full_reasoning=args.p_full_reasoning,
            max_episodes=args.max_episodes_per_dataset,
            seed=args.seed,
        )
        sample_iter = ds.iter_samples(max_samples=args.num_samples * 4)
        print(f"[2/4] single-task dataset = {ds.dataset_name} "
              f"(episodes={len(ds._episode_ids)})")
    else:
        weights = (
            {k: 1.0 for k in args.datasets} if args.datasets else dict(DEFAULT_DATASET_WEIGHTS)
        )
        mixed = RoboTwinMixedDataset(
            data_root=pathlib.Path(args.data_root),
            weights=weights,
            action_horizon=args.action_horizon,
            p_plan=args.p_plan,
            p_full_reasoning=args.p_full_reasoning,
            max_episodes_per_dataset=args.max_episodes_per_dataset,
            seed=args.seed,
        )
        sample_iter = mixed.iter_samples(max_samples=args.num_samples * 4)
        print(f"[2/4] mixed dataset = {mixed.dataset_names}  weights={weights}")

    # 2. Build tokenizer (with the Stage 1 marker fix already applied).
    tokenizer = PaligemmaTokenizer(max_len=args.max_token_len, prompt_format="lap")
    print(f"[3/4] tokenizer max_len = {args.max_token_len}")
    print()

    # 3. Stream + dump.
    print(f"[4/4] dumping {args.num_samples} samples ...")
    n = 0
    for sample in sample_iter:
        if n >= args.num_samples:
            break

        # Tokenize. RoboTwin samples can be in two modes:
        #   a) emit_full=True: prompt + plan + stage + action (Context 2 or 3)
        #   b) emit_full=False: prompt only (Context 1 no-reason)
        # We pass the optional fields as-is; tokenizer handles None correctly.
        (
            tokens,
            attn_mask,
            ar_target_mask,
            _number_mask,
            _direction_mask,
            _token_loss_mask,
            stage_mask,
            plan_mask,
        ) = tokenizer.tokenize(
            prompt=sample["prompt"],
            reasoning=sample.get("language_actions"),
            state=None,
            plan=sample.get("plan"),
            plan_position=sample.get("plan_position", "none"),
            langact=sample.get("langact"),
            is_vqa_sample=False,
            is_prediction_sample=False,
        )

        # Mask stats (handle Context 1 where some masks are None).
        n_valid = int(attn_mask.sum())
        n_ar = int(ar_target_mask.sum()) if ar_target_mask is not None else 0
        n_plan = int(plan_mask.sum()) if plan_mask is not None else 0
        n_stage = int(stage_mask.sum()) if stage_mask is not None else 0
        n_action = max(0, n_ar - n_stage - n_plan)

        action_only_mask = None
        if ar_target_mask is not None:
            action_only_mask = ar_target_mask.copy()
            if stage_mask is not None:
                action_only_mask &= np.logical_not(stage_mask)
            if plan_mask is not None:
                action_only_mask &= np.logical_not(plan_mask)

        # Decode AR spans.
        decoded_plan = _decode_span(tokenizer, tokens, plan_mask)
        decoded_stage = _decode_span(tokenizer, tokens, stage_mask)
        decoded_action = _decode_span(tokenizer, tokens, action_only_mask)

        # 4. Save images.
        base_img = sample["image"]["base_0_rgb"]
        wrist_img = sample["image"]["left_wrist_0_rgb"]
        base_png = out_dir / f"sample_{n:02d}.base.png"
        wrist_png = out_dir / f"sample_{n:02d}.wrist.png"
        _save_png(base_img, base_png)
        _save_png(wrist_img, wrist_png)

        # 5. Save action vector + state as JSON for downstream inspection.
        actions = sample["actions"]
        state = sample["state"]
        actions_json_path = out_dir / f"sample_{n:02d}.actions.json"
        actions_json_path.write_text(json.dumps({
            "dataset": sample.get("_dataset"),
            "episode": sample.get("_episode"),
            "frame_idx": sample.get("_frame_idx"),
            "phase_idx": sample.get("_phase_idx"),
            "arm_tag": sample.get("_arm_tag"),
            "emit_full_reasoning": bool(sample.get("_emit_full_reasoning", False)),
            "state_14d": state.tolist(),
            "actions_HxD": actions.tolist(),
        }, indent=2))

        # 6. Save text inspection.
        txt_path = out_dir / f"sample_{n:02d}.txt"
        lines = [
            f"=== Sample {n} ===",
            "",
            "[PROVENANCE]",
            f"  dataset           : {sample.get('_dataset')}",
            f"  episode / frame   : {sample.get('_episode')} / {sample.get('_frame_idx')}",
            f"  phase_idx         : {sample.get('_phase_idx')}",
            f"  arm_tag           : {sample.get('_arm_tag')}  (active arm — drives wrist cam choice)",
            f"  emit_full_reasoning: {bool(sample.get('_emit_full_reasoning', False))}",
            "",
            "[INPUT]",
            f"  base image (head)   : shape={base_img.shape} dtype={base_img.dtype}  → {base_png.name}",
            f"  wrist image (active): shape={wrist_img.shape} dtype={wrist_img.dtype} → {wrist_png.name}",
            f"  prompt              : {sample['prompt']!r}",
            f"  plan_position       : {sample.get('plan_position', 'none')}",
            f"  plan (raw)          : {sample.get('plan')!r}",
            "",
            "[AR TARGET — what the model is trained to predict]",
            f"  language_actions  : {sample.get('language_actions')!r}  (→ [stage])",
            f"  langact           : {sample.get('langact')!r}  (→ [action])",
            "",
            "[TOKENIZER OUTPUT]",
            f"  total tokens      : {len(tokens)}",
            f"  valid (non-pad)   : {n_valid}",
            f"  ar_target_mask    : {n_ar} positions  (CE loss covers plan ∪ stage ∪ action)",
            f"    plan_mask       : {n_plan} positions  ([plan] when plan_position='target')",
            f"    stage_mask      : {n_stage} positions  ([stage] segment incl. marker)",
            f"    action_only     : {n_action} positions  ([action] span incl. marker)",
            "",
            "[DECODED AR SPANS]",
            f"  decoded[plan]     : {decoded_plan!r}",
            f"  decoded[stage]    : {decoded_stage!r}",
            f"  decoded[action]   : {decoded_action!r}",
            "",
            "[ACTION + STATE]",
            f"  state (14d)       : {np.array2string(state, precision=3, max_line_width=200)}",
            f"  actions (H={actions.shape[0]}):",
            *[f"    {ln}" for ln in _format_action_summary(actions).splitlines()],
            "",
            "[FULL TOKEN SEQUENCE (decoded)]",
            f"  {tokenizer.decode(np.asarray(tokens[attn_mask]))!r}",
        ]
        txt_path.write_text("\n".join(lines))

        # Stdout summary (succinct).
        print(f"--- sample {n} ({sample.get('_dataset')} ep{sample.get('_episode')} "
              f"f{sample.get('_frame_idx')}, arm={sample.get('_arm_tag')}, "
              f"emit_full={bool(sample.get('_emit_full_reasoning', False))}) ---")
        print(f"   plan_position={sample.get('plan_position', 'none')}  "
              f"valid_tokens={n_valid}  ar={n_ar}  (plan={n_plan} stage={n_stage} action={n_action})")
        print(f"   pred[plan]   : {decoded_plan[:100]!r}")
        print(f"   pred[stage]  : {decoded_stage[:100]!r}")
        print(f"   pred[action] : {decoded_action[:100]!r}")
        print(f"   actions: shape={actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")
        print()
        n += 1

    if n < args.num_samples:
        print(f"[warn] requested {args.num_samples} samples but only {n} emitted.")
    print(f"\nDone. Inspect {out_dir}/sample_*.{{base.png,wrist.png,txt,actions.json}}")


if __name__ == "__main__":
    main()
