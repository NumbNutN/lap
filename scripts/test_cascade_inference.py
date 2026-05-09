"""Cascade-VLA inference smoke test for a Bridge ECoT pretrained checkpoint.

Loads a checkpoint, picks one Bridge ECoT sample, runs the VLM autoregressively
(``model.sample_tokens``) from a prompt-only prefix, and prints the decoded
text plus the parsed ``[plan]`` / ``[stage]`` / ``[action]`` segments alongside
the ground truth from the dataset.

Usage on pod (after first save lands at step 2000+):

    cd /data/zhaoqc/RoboTwin/policy/lap
    .venv/bin/python scripts/test_cascade_inference.py \\
        --checkpoint-dir checkpoints/lap_bridge_pretrain/lap_bridge_pretrain_run1/10000 \\
        --sample-idx 0 --max-decoding-steps 200

Notes
-----
* For the Bridge pretraining config, ``enable_action_training=False`` —
  ``sample_tokens`` invokes only the VLM expert. No action expert is loaded.
* The script exercises the *full cascade* path: prompt = task only, AR target
  is the model's own ``[plan] ... [stage] ... [action] ...`` continuation. To
  feed an external plan and only generate stage+action, pass ``--given-plan``.
* Bridge has only a base camera and no proprio state. The wrist slot is zero-
  filled and masked out (matching the training-time ``image_mask=False`` for
  ``left_wrist_0_rgb``); ``state`` is zeros.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import sys

# Parse --gpu before importing JAX (JAX latches CUDA_VISIBLE_DEVICES at import).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()
if _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre_args.gpu

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_LAP_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_LAP_ROOT / "src"))
sys.path.insert(0, str(_LAP_ROOT / "third_party" / "openpi" / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from openpi.models import model as _model  # noqa: E402

from lap.datasets.bridge_ecot_dataset import (  # noqa: E402
    BridgeECoTDataset,
    DEFAULT_ECOT_BRIDGE_JSON,
)
from lap.datasets.utils.bridge_lerobot_loader import (  # noqa: E402
    DEFAULT_MAPPING_CACHE,
    DEFAULT_SCRIPTED_SNAP_PARENT,
    DEFAULT_TELEOP_SNAP,
    LeRobotBridgeImageLoader,
    build_ecot_to_lerobot_mapping,
)
from lap.models.model_adapter import CoTObservation  # noqa: E402
from lap.models.tokenizer import PaligemmaTokenizer  # noqa: E402
from lap.training import config as _config  # noqa: E402


def resolve_params_path(ckpt_dir: pathlib.Path) -> pathlib.Path:
    """Accept either ``<step>/`` or ``<step>/params/`` (or a flat-format dir)."""
    if ckpt_dir.name == "params":
        return ckpt_dir
    if (ckpt_dir / "params").exists():
        return ckpt_dir / "params"
    if (ckpt_dir / "_METADATA").exists() or (ckpt_dir / "default").exists():
        return ckpt_dir
    raise FileNotFoundError(
        f"Could not find a params dir at {ckpt_dir}. "
        f"Expected one of: {ckpt_dir}/params, {ckpt_dir}/_METADATA, {ckpt_dir}/default."
    )


def fetch_bridge_sample(sample_idx: int, max_episodes: int | None) -> dict:
    """Stream the Bridge ECoT dataset, return the Nth sample with ground truth.

    Mirrors the loader-construction logic in
    ``bridge_data_loader.create_bridge_data_loader`` so we hit the same image
    pipeline (LeRobot mp4 → 224x224 RGB) used at training time.
    """
    if not DEFAULT_TELEOP_SNAP.exists():
        raise FileNotFoundError(
            f"LeRobot Bridge teleop snapshot not found at {DEFAULT_TELEOP_SNAP}. "
            "Inference requires the same image source training used."
        )
    scripted_snap = None
    if DEFAULT_SCRIPTED_SNAP_PARENT.exists():
        snaps = sorted(p for p in DEFAULT_SCRIPTED_SNAP_PARENT.iterdir() if p.is_dir())
        scripted_snap = snaps[0] if snaps else None

    mapping = build_ecot_to_lerobot_mapping(
        ecot_json_path=DEFAULT_ECOT_BRIDGE_JSON,
        teleop_snap=DEFAULT_TELEOP_SNAP,
        scripted_snap=scripted_snap,
        cache_path=DEFAULT_MAPPING_CACHE,
        force_rebuild=False,
    )
    image_loader = LeRobotBridgeImageLoader(
        teleop_snap=DEFAULT_TELEOP_SNAP,
        scripted_snap=scripted_snap,
        mapping=mapping,
    )
    ds = BridgeECoTDataset(
        ecot_json_path=DEFAULT_ECOT_BRIDGE_JSON,
        image_loader=image_loader,
        include_plan=True,
        p_plan=0.0,            # inference: keep plan as ground truth in the sample dict only
        skip_steps_without_change=False,
        max_episodes=max_episodes,
    )
    for i, sample in enumerate(ds.iter_samples()):
        if i == sample_idx:
            return sample
    raise IndexError(
        f"Bridge dataset exhausted at i={i}; --sample-idx={sample_idx} unreachable."
    )


def resize_to_224(image_uint8: np.ndarray) -> np.ndarray:
    """HWC uint8 → HWC uint8 at 224x224. Uses cv2 if available, else nearest."""
    target = (224, 224)
    if image_uint8.shape[:2] == target:
        return image_uint8
    try:
        import cv2
        return cv2.resize(image_uint8, (target[1], target[0]), interpolation=cv2.INTER_AREA)
    except ImportError:
        h_old, w_old = image_uint8.shape[:2]
        h_new, w_new = target
        row_idx = (np.arange(h_new) * h_old // h_new).astype(np.int32)
        col_idx = (np.arange(w_new) * w_old // w_new).astype(np.int32)
        return image_uint8[row_idx[:, None], col_idx[None, :]]


def build_inference_observation(
    image_uint8: np.ndarray,
    tokens: np.ndarray,
    mask: np.ndarray,
    state: np.ndarray,
    image_keys: tuple[str, ...],
) -> CoTObservation:
    """Assemble batch-1 ``CoTObservation`` for sample_tokens.

    Following the training-time Bridge layout: base image goes to
    ``image_keys[0]``; remaining slots get zero images with mask=False.
    """
    base_uint8 = resize_to_224(image_uint8)
    base_f = (base_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0  # → [-1, 1]
    zero_f = np.zeros((224, 224, 3), dtype=np.float32)

    images_b: dict[str, jnp.ndarray] = {}
    image_masks_b: dict[str, jnp.ndarray] = {}
    primary = image_keys[0]
    for k in image_keys:
        if k == primary:
            images_b[k] = jnp.asarray(base_f)[None, ...]
            image_masks_b[k] = jnp.ones((1,), dtype=jnp.bool_)
        else:
            images_b[k] = jnp.asarray(zero_f)[None, ...]
            image_masks_b[k] = jnp.zeros((1,), dtype=jnp.bool_)

    return CoTObservation(
        images=images_b,
        image_masks=image_masks_b,
        state=jnp.asarray(state, dtype=jnp.float32)[None, ...],
        tokenized_prompt=jnp.asarray(tokens, dtype=jnp.int32)[None, ...],
        tokenized_prompt_mask=jnp.asarray(mask, dtype=jnp.bool_)[None, ...],
        # Inference-time: no AR target spans.
        token_ar_mask=None,
        token_loss_mask=None,
        tokenized_ar_target_mask=None,
        tokenized_stage_mask=None,
        tokenized_plan_mask=None,
    )


def first_eos_index(tokens: np.ndarray, eos_id: int) -> int | None:
    hits = np.where(tokens == eos_id)[0]
    return int(hits[0]) if len(hits) else None


def parse_cascade_segments(text: str) -> dict[str, str]:
    """Split a generated cascade-VLA string into [plan]/[stage]/[action] parts.

    Marker convention used during training (PaligemmaTokenizer.tokenize):
      ``... [plan] <plan_text> [stage] <stage_text> [action] <langact> [EOS]``
    Markers may be missing if the model didn't produce them (early training).
    Anything before the first marker is reported as ``"prefix"``.
    """
    segments = {"prefix": "", "plan": "", "stage": "", "action": ""}

    # Walk markers in left-to-right occurrence order.
    markers = ["[plan]", "[stage]", "[action]"]
    cursor = 0
    spans: list[tuple[str, int, int]] = []  # (segment_name, content_start, content_end)
    last_label = "prefix"
    last_content_start = 0
    while cursor < len(text):
        next_marker = None
        next_marker_pos = -1
        for m in markers:
            idx = text.find(m, cursor)
            if idx >= 0 and (next_marker_pos == -1 or idx < next_marker_pos):
                next_marker = m
                next_marker_pos = idx
        if next_marker is None:
            spans.append((last_label, last_content_start, len(text)))
            break
        spans.append((last_label, last_content_start, next_marker_pos))
        last_label = next_marker.strip("[]")
        last_content_start = next_marker_pos + len(next_marker)
        cursor = last_content_start

    for label, s, e in spans:
        if label in segments:
            segments[label] = (segments[label] + " " + text[s:e].strip()).strip()
    return segments


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-name", default="lap_bridge_pretrain",
                   help="TrainConfig name (default lap_bridge_pretrain).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to step ckpt (e.g. checkpoints/.../run1/10000), "
                        "or directly to its params/ subdir.")
    p.add_argument("--sample-idx", type=int, default=0,
                   help="Which Bridge sample to grab (0-indexed; default 0).")
    p.add_argument("--max-bridge-episodes", type=int, default=10,
                   help="Cap on episodes streamed before giving up locating sample-idx.")
    p.add_argument("--given-plan", action="store_true",
                   help="Feed ground-truth plan in the prompt (Context 2 mode); "
                        "model only generates [stage]+[action]. "
                        "Default: full cascade (Context 3).")
    p.add_argument("--max-decoding-steps", type=int, default=200,
                   help="Cap on AR decoding length (default 200).")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 → greedy argmax (recommended). >0 → sampling.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES (parsed before JAX import).")
    args = p.parse_args()

    print(f"[1/5] Loading config '{args.config_name}'...")
    config = _config.get_config(args.config_name)
    # stop_action_to_vlm_grad has no inference-time meaning.
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, stop_action_to_vlm_grad=False),
    )
    image_keys = tuple(config.model.image_keys)
    print(f"      image_keys={image_keys}, max_token_len={config.model.max_token_len}, "
          f"action_dim={config.model.action_dim}, paligemma={config.model.paligemma_variant}")

    print(f"[2/5] Restoring params from {args.checkpoint_dir} ...")
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    params_path = resolve_params_path(ckpt_dir)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print(f"      restored. model={type(model).__name__}, EOS={model.EOS_TOKEN}")

    print(f"[3/5] Fetching Bridge sample idx={args.sample_idx} "
          f"(scanning up to {args.max_bridge_episodes} episodes) ...")
    sample = fetch_bridge_sample(args.sample_idx, args.max_bridge_episodes)
    task_text: str = sample["prompt"]
    gt_plan: str | None = sample.get("plan")
    gt_stage: str = sample["language_actions"]
    gt_action: str = sample["langact"]
    print(f"      task = {task_text!r}")
    print(f"      gt[plan]   = {gt_plan!r}")
    print(f"      gt[stage]  = {gt_stage!r}")
    print(f"      gt[action] = {gt_action!r}")

    print("[4/5] Tokenizing inference prompt ...")
    tokenizer = PaligemmaTokenizer(
        max_len=config.model.max_token_len,
        prompt_format=config.model.prompt_format,
        prediction_format=getattr(config.model, "prediction_format", "default"),
        reasoning_mask_prob=0.0,
    )
    state_for_prompt = None  # Bridge sample carries no proprio
    state_vec = np.zeros((config.model.action_dim,), dtype=np.float32)

    if args.given_plan and gt_plan:
        # Context 2: plan_position="prompt" with reasoning=None at inference.
        # The tokenizer's plan-in-prompt path inserts ``[plan] <plan_text>``
        # into the prompt; AR target span stays empty (since reasoning=None /
        # langact=None). Model AR-generates [stage] ... [action] ...
        tok_out = tokenizer.tokenize(
            task_text,
            reasoning=None,
            langact=None,
            state=state_for_prompt,
            plan=gt_plan,
            plan_position="prompt",
        )
        mode = "Context 2 (plan-in-prompt → generate [stage] [action])"
    else:
        # Context 3: plan_position="none", reasoning=None. Prompt is just
        # ``[BOS] <formatted_prompt>``; the model AR-generates the entire
        # cascade ``[plan] ... [stage] ... [action] ...``.
        tok_out = tokenizer.tokenize(
            task_text,
            reasoning=None,
            langact=None,
            state=state_for_prompt,
            plan=None,
            plan_position="none",
        )
        mode = "Context 3 (full cascade → generate [plan] [stage] [action])"
    tokens, mask = tok_out[0], tok_out[1]
    print(f"      mode = {mode}")
    print(f"      tokens used = {int(mask.sum())}/{len(tokens)}")

    obs = build_inference_observation(
        sample["image"], tokens, mask, state_vec, image_keys
    )

    print(f"[5/5] Sampling tokens (max_decoding_steps={args.max_decoding_steps}, "
          f"temperature={args.temperature}) ...")
    rng = jax.random.PRNGKey(args.seed)
    output_tokens = model.sample_tokens(
        rng, obs,
        max_decoding_steps=args.max_decoding_steps,
        temperature=args.temperature,
    )
    output_np = np.asarray(output_tokens[0]).astype(np.int32)
    eos_idx = first_eos_index(output_np, int(model.EOS_TOKEN))
    valid = output_np[:eos_idx] if eos_idx is not None else output_np
    print(f"      generated {len(valid)} tokens "
          f"({'EOS reached' if eos_idx is not None else 'no EOS within budget'})")

    decoded = tokenizer.decode(valid)
    parsed = parse_cascade_segments(decoded)

    print("\n" + "=" * 78)
    print("DECODED RAW")
    print("=" * 78)
    print(decoded)
    print("\n" + "=" * 78)
    print("PARSED SEGMENTS  (model output  vs  ground truth)")
    print("=" * 78)
    for label, gt in [("plan", gt_plan or ""), ("stage", gt_stage), ("action", gt_action)]:
        print(f"\n--- [{label}] ---")
        print(f"  pred: {parsed.get(label, '')!r}")
        print(f"  gt:   {gt!r}")
    if parsed.get("prefix"):
        print(f"\n[orphan prefix before first marker] {parsed['prefix']!r}")
    print("=" * 78)


if __name__ == "__main__":
    main()
