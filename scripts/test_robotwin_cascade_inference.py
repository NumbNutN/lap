"""Cascade-VLA inference smoke test on a Stage 2 (RoboTwin) checkpoint.

Adapts ``test_cascade_inference.py`` (Bridge V2) for RoboTwin Stage 2:
  * Image input  : head_camera (base) + active arm wrist (left or right)
  * State input  : 14-DoF bimanual endpose (zeros allowed)
  * Task prompt  : task_prompt from RoboTwin metadata
  * Ground truth : per-phase ``subgoal_reasoning`` (stage) + ``phase_prompts[0]``
                   (action) + synthesized multi-step plan from unique subgoals

Goal: confirm Stage 2 didn't break the cascade output ability inherited from
Stage 1 (i.e. with ``stop_grad_mode="full"`` the VLM should still emit
``[plan]`` / ``[stage]`` / ``[action]`` markers in semantically correct text).

Usage on pod (CPU mode runs alongside training; GPU mode requires GPU free):

    cd /data/zhaoqc/RoboTwin/policy/lap
    JAX_PLATFORMS=cpu .venv/bin/python scripts/test_robotwin_cascade_inference.py \\
        --checkpoint-dir checkpoints/lap_robotwin_finetune/lap_robotwin_run0/30000 \\
        --task arrange_blocks_l_shape \\
        --episode 0 \\
        --frame-stride 80
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

from lap.datasets.robotwin_dataset import (  # noqa: E402
    DEFAULT_ROBOTWIN_DATA_ROOT,
    RoboTwinTaskDataset,
    cascade_extract,
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


def build_inference_observation(
    base_uint8: np.ndarray,
    wrist_uint8: np.ndarray,
    tokens: np.ndarray,
    mask: np.ndarray,
    state: np.ndarray,
    image_keys: tuple[str, ...],
) -> CoTObservation:
    """Build a batch-1 ``CoTObservation`` for ``model.sample_tokens``.

    Layout matches Stage 2 training: image_keys[0] = head_camera (base),
    image_keys[1] = active arm's wrist camera. Both are real frames (mask=True),
    in contrast to Stage 1 where the wrist slot was zero-filled.
    """
    base_f = (base_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0   # → [-1, 1]
    wrist_f = (wrist_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0

    images_b: dict[str, jnp.ndarray] = {}
    image_masks_b: dict[str, jnp.ndarray] = {}
    primary, secondary = image_keys[0], image_keys[1]
    images_b[primary] = jnp.asarray(base_f)[None, ...]
    image_masks_b[primary] = jnp.ones((1,), dtype=jnp.bool_)
    images_b[secondary] = jnp.asarray(wrist_f)[None, ...]
    image_masks_b[secondary] = jnp.ones((1,), dtype=jnp.bool_)

    return CoTObservation(
        images=images_b,
        image_masks=image_masks_b,
        state=jnp.asarray(state, dtype=jnp.float32)[None, ...],
        tokenized_prompt=jnp.asarray(tokens, dtype=jnp.int32)[None, ...],
        tokenized_prompt_mask=jnp.asarray(mask, dtype=jnp.bool_)[None, ...],
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
    """Split a generated cascade-VLA string into [plan]/[stage]/[action] parts."""
    segments = {"prefix": "", "plan": "", "stage": "", "action": ""}
    markers = ["[plan]", "[stage]", "[action]"]
    cursor = 0
    spans: list[tuple[str, int, int]] = []
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


def fetch_robotwin_sample(
    data_root: pathlib.Path,
    task: str,
    episode: int,
    frame_idx: int,
    action_horizon: int,
):
    """Build one RoboTwin sample at a specific (task, episode, frame_idx)."""
    import json

    ds = RoboTwinTaskDataset(
        task_dir=data_root / task,
        action_horizon=action_horizon,
        p_plan=0.0,           # always Context 2 (plan-in-prompt) for inference visibility
        p_full_reasoning=1.0, # always emit full cascade so we can compare GT
        max_episodes=None,
        seed=0,
    )

    meta_path = ds._meta_path(episode)
    hdf5_path = ds._hdf5_path(episode)
    if not meta_path.exists() or not hdf5_path.exists():
        raise FileNotFoundError(f"episode {episode} of {task} missing on disk")

    with open(meta_path) as f:
        meta = json.load(f)
    phases = meta.get("phases", [])
    n_frames = phases[-1]["end_frame"] if phases else 0
    if frame_idx >= n_frames or frame_idx < 0:
        raise IndexError(f"frame_idx={frame_idx} out of range [0, {n_frames})")
    phase_idx, phase = ds._phase_for_frame(phases, frame_idx)
    if phase is None:
        raise IndexError(f"frame_idx={frame_idx} not covered by any phase")

    # Use the deterministic extraction path with our local rng (no plan
    # randomization since p_plan=0.0).
    import random as _random
    cas = cascade_extract(task, meta, phase_idx, phase, _random.Random(0))
    arm_tag = phase.get("arm_tag", "left")
    wrist_camera = "left_camera" if arm_tag == "left" else "right_camera"
    head_img = ds._reader.decode_frame(hdf5_path, "head_camera", frame_idx)
    wrist_img = ds._reader.decode_frame(hdf5_path, wrist_camera, frame_idx)
    state = ds._reader.read_state(hdf5_path, frame_idx)
    return {
        "task": task,
        "episode": episode,
        "frame_idx": frame_idx,
        "phase_idx": phase_idx,
        "n_phases": len(phases),
        "arm_tag": arm_tag,
        "head_image": head_img,
        "wrist_image": wrist_img,
        "state": state,
        "task_prompt": cas["task_prompt"],
        "plan": cas["plan"],
        "stage": cas["stage"],
        "action_lang": cas["action_lang"],
    }


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-name", default="lap_robotwin_finetune",
                   help="TrainConfig name (default lap_robotwin_finetune).")
    p.add_argument("--checkpoint-dir", required=True,
                   help="Path to step ckpt (e.g. checkpoints/.../run0/30000), "
                        "or directly to its params/ subdir.")
    p.add_argument("--data-root", default=str(DEFAULT_ROBOTWIN_DATA_ROOT),
                   help="RoboTwin data root (default: pod path /data/.cache/RoboTwin).")
    p.add_argument("--task", default="arrange_blocks_l_shape",
                   help="Task name (subdir under data_root).")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="Specific frame indices. Default: 1 frame mid-episode.")
    p.add_argument("--frame-stride", type=int, default=80,
                   help="Used when --frames not set: pick frames at "
                        "[stride, 2*stride, 3*stride, ...] until end of episode.")
    p.add_argument("--max-frames", type=int, default=3,
                   help="Cap number of frames sampled (default 3).")
    p.add_argument("--max-decoding-steps", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default=None,
                   help="CUDA_VISIBLE_DEVICES (parsed before JAX import).")
    args = p.parse_args()

    print(f"[1/5] Loading config '{args.config_name}'...")
    config = _config.get_config(args.config_name)
    # For pure-language AR sampling we disable the action-expert forward path:
    # ``sample_tokens`` only invokes the VLM (expert 0). With
    # ``enable_action_training=True`` the prefix attention mask reserves
    # ``max_decoding_steps`` slots for action-expert suffix tokens that
    # ``sample_tokens`` never actually pads, producing a key/query length
    # mismatch (TypeError: mul broadcast (..., 832, 1032) vs (..., 832, 832)).
    # Setting ``enable_action_training=False`` at config-load time makes the
    # model skip the suffix-token plumbing — the loaded action-expert weights
    # are simply unused for this AR-text test.
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            stop_action_to_vlm_grad=False,
            enable_action_training=False,
        ),
    )
    image_keys = tuple(config.model.image_keys)
    print(f"      image_keys={image_keys}, max_token_len={config.model.max_token_len}, "
          f"action_dim={config.model.action_dim}, "
          f"enable_action_training=False (forced for AR-text sampling)")

    print(f"[2/5] Restoring params from {args.checkpoint_dir} ...")
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    params_path = resolve_params_path(ckpt_dir)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print(f"      restored. model={type(model).__name__}, EOS={model.EOS_TOKEN}")

    data_root = pathlib.Path(args.data_root).expanduser()
    if not (data_root / args.task).is_dir():
        raise FileNotFoundError(f"task dir not found: {data_root / args.task}")

    # Decide which frames to sample
    if args.frames:
        frames_to_run = list(args.frames)[: args.max_frames]
    else:
        # Open metadata to find episode length, pick stride'd frames.
        import json
        meta_path = data_root / args.task / "demo_clean" / "metadata" / f"episode{args.episode}.json"
        with open(meta_path) as f:
            meta = json.load(f)
        n_frames = meta["phases"][-1]["end_frame"] if meta.get("phases") else 0
        frames_to_run = list(range(args.frame_stride, n_frames, args.frame_stride))[: args.max_frames]
        if not frames_to_run:
            frames_to_run = [n_frames // 2]

    print(f"[3/5] Will run inference on task={args.task} ep={args.episode} "
          f"frames={frames_to_run}")

    print("[4/5] Tokenizer + sampling setup...")
    tokenizer = PaligemmaTokenizer(
        max_len=config.model.max_token_len,
        prompt_format=config.model.prompt_format,
        prediction_format=getattr(config.model, "prediction_format", "default"),
        reasoning_mask_prob=0.0,
    )
    rng_root = jax.random.PRNGKey(args.seed)

    print(f"[5/5] Sampling cascade text on {len(frames_to_run)} frames "
          f"(max_decoding_steps={args.max_decoding_steps}, temperature={args.temperature})...")

    for i, frame_idx in enumerate(frames_to_run):
        sample = fetch_robotwin_sample(
            data_root=data_root,
            task=args.task,
            episode=args.episode,
            frame_idx=frame_idx,
            action_horizon=config.model.action_horizon,
        )

        # Inference prompt: only the task_prompt (no reasoning / langact / plan
        # → model AR-generates the entire cascade [plan][stage][action]).
        tok_out = tokenizer.tokenize(
            sample["task_prompt"],
            reasoning=None,
            langact=None,
            state=None,
            plan=None,
            plan_position="none",
        )
        tokens, mask = tok_out[0], tok_out[1]

        obs = build_inference_observation(
            base_uint8=sample["head_image"],
            wrist_uint8=sample["wrist_image"],
            tokens=tokens,
            mask=mask,
            state=sample["state"],
            image_keys=image_keys,
        )

        rng, rng_root = jax.random.split(rng_root)
        output_tokens = model.sample_tokens(
            rng, obs,
            max_decoding_steps=args.max_decoding_steps,
            temperature=args.temperature,
        )
        output_np = np.asarray(output_tokens[0]).astype(np.int32)
        eos_idx = first_eos_index(output_np, int(model.EOS_TOKEN))
        valid = output_np[:eos_idx] if eos_idx is not None else output_np
        decoded = tokenizer.decode(valid)
        parsed = parse_cascade_segments(decoded)

        print()
        print("=" * 78)
        print(f"  Frame {frame_idx}  (phase {sample['phase_idx']}/{sample['n_phases']}, "
              f"arm={sample['arm_tag']})")
        print("=" * 78)
        print(f"  task_prompt : {sample['task_prompt']!r}")
        print(f"  generated {len(valid)} tokens "
              f"({'EOS reached' if eos_idx is not None else 'no EOS within budget'})")
        print()
        print("  DECODED RAW:")
        print(f"  {decoded}")
        print()
        print("  PARSED  (model output  vs  ground truth)")
        for label, gt in [("plan", sample["plan"]), ("stage", sample["stage"]),
                          ("action", sample["action_lang"])]:
            print(f"  --- [{label}] ---")
            print(f"    pred: {parsed.get(label, '')!r}")
            print(f"    gt:   {gt!r}")
        if parsed.get("prefix"):
            print(f"  [orphan prefix before any marker]: {parsed['prefix']!r}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
