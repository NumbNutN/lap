"""Dry-run action-expert quality check on a Stage 2 (RoboTwin) checkpoint.

Loads N RoboTwin sample frames, runs ``model.sample_actions`` (the action
expert flow-matching forward pass) on each, and compares the predicted
``(action_horizon, 14)`` chunks against the ground-truth bimanual
``joint_action/vector`` slice. Reports per-dim MSE / L1 / max-abs error and
whether the predictions are likely mode-collapsed (very low variance).

Use this to triage action quality WITHOUT spinning up the RoboTwin
simulator. Useful early signal before the full sim eval pipeline (server +
port-forward + sim driver).

Usage on pod (CPU is feasible since action_expert is small):

    cd /data/zhaoqc/RoboTwin/policy/lap
    JAX_PLATFORMS=cpu .venv/bin/python scripts/test_robotwin_action_dryrun.py \\
        --checkpoint-dir checkpoints/lap_robotwin_finetune/lap_robotwin_run0/30000 \\
        --task arrange_blocks_l_shape \\
        --episode 0 \\
        --num-samples 8

For higher-fidelity sampling (more diffusion steps), pass
``--num-flow-steps 32``. Defaults to 10 (matches ``LAPModel.sample_actions``
default in lap.py).
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
    if ckpt_dir.name == "params":
        return ckpt_dir
    if (ckpt_dir / "params").exists():
        return ckpt_dir / "params"
    if (ckpt_dir / "_METADATA").exists() or (ckpt_dir / "default").exists():
        return ckpt_dir
    raise FileNotFoundError(f"Could not find params under {ckpt_dir}")


def build_observation(
    base_uint8: np.ndarray,
    wrist_uint8: np.ndarray,
    state: np.ndarray,
    tokens: np.ndarray,
    mask: np.ndarray,
    image_keys: tuple[str, ...],
) -> CoTObservation:
    """Build a batch-1 observation with both real cameras (head + active wrist)."""
    base_f = (base_uint8.astype(np.float32) / 255.0) * 2.0 - 1.0
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


def fetch_one_sample(
    ds: RoboTwinTaskDataset,
    episode: int,
    frame_idx: int,
    action_horizon: int,
):
    """Pull one (image, state, prompt, gt_actions) tuple for inference."""
    import json
    import random as _random

    meta_path = ds._meta_path(episode)
    hdf5_path = ds._hdf5_path(episode)
    with open(meta_path) as f:
        meta = json.load(f)
    phases = meta.get("phases", [])
    n_frames = phases[-1]["end_frame"] if phases else 0
    if frame_idx >= n_frames:
        raise IndexError(f"frame_idx={frame_idx} ≥ n_frames={n_frames}")
    phase_idx, phase = ds._phase_for_frame(phases, frame_idx)
    if phase is None:
        raise IndexError(f"frame_idx={frame_idx} not covered by any phase")

    cas = cascade_extract(ds.dataset_name, meta, phase_idx, phase, _random.Random(0))
    arm_tag = phase.get("arm_tag", "left")
    wrist_camera = "left_camera" if arm_tag == "left" else "right_camera"
    head_img = ds._reader.decode_frame(hdf5_path, "head_camera", frame_idx)
    wrist_img = ds._reader.decode_frame(hdf5_path, wrist_camera, frame_idx)
    state = ds._reader.read_state(hdf5_path, frame_idx)
    gt_actions = ds._reader.read_actions(hdf5_path, frame_idx, action_horizon)
    return {
        "episode": episode,
        "frame_idx": frame_idx,
        "phase_idx": phase_idx,
        "n_phases": len(phases),
        "arm_tag": arm_tag,
        "head_image": head_img,
        "wrist_image": wrist_img,
        "state": state,
        "task_prompt": cas["task_prompt"],
        "gt_actions": gt_actions,        # (action_horizon, 14)
    }


def per_dim_summary(arr: np.ndarray, label: str = "") -> str:
    """7-stat summary per dim of (H, D) array."""
    lines = [f"{label} shape={arr.shape}  dtype={arr.dtype}"]
    for d in range(arr.shape[-1]):
        col = arr[..., d]
        lines.append(
            f"    dim {d:2d}: min={col.min():+.4f} max={col.max():+.4f} "
            f"mean={col.mean():+.4f} std={col.std():.4f}"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config-name", default="lap_robotwin_finetune")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-root", default=str(DEFAULT_ROBOTWIN_DATA_ROOT))
    p.add_argument("--task", default="arrange_blocks_l_shape")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=None,
                   help="Specific frame indices. Default: evenly spaced over episode.")
    p.add_argument("--num-samples", type=int, default=8,
                   help="When --frames not set: how many evenly spaced frames to test.")
    p.add_argument("--num-flow-steps", type=int, default=10,
                   help="Diffusion sampling steps (default 10; LAP default).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", default=None)
    args = p.parse_args()

    print(f"[1/4] Loading config '{args.config_name}'...")
    config = _config.get_config(args.config_name)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, stop_action_to_vlm_grad=False),
    )
    image_keys = tuple(config.model.image_keys)
    print(f"      action_dim={config.model.action_dim}  "
          f"action_horizon={config.model.action_horizon}  "
          f"image_keys={image_keys}")

    print(f"[2/4] Restoring params from {args.checkpoint_dir}")
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    params_path = resolve_params_path(ckpt_dir)
    params = _model.restore_params(params_path, dtype=jnp.bfloat16)
    model = config.model.load(params)
    model.eval()
    print(f"      restored. model={type(model).__name__}")

    data_root = pathlib.Path(args.data_root).expanduser()
    if not (data_root / args.task).is_dir():
        raise FileNotFoundError(f"task dir not found: {data_root / args.task}")

    ds = RoboTwinTaskDataset(
        task_dir=data_root / args.task,
        action_horizon=config.model.action_horizon,
        p_plan=0.0,
        p_full_reasoning=0.0,
        max_episodes=None,
        seed=0,
    )

    # Decide which frames to sample.
    if args.frames:
        frames_to_run = list(args.frames)
    else:
        import json
        meta_path = ds._meta_path(args.episode)
        with open(meta_path) as f:
            meta = json.load(f)
        n_frames = meta["phases"][-1]["end_frame"] if meta.get("phases") else 0
        # Reserve the last action_horizon frames so gt_actions doesn't pad.
        usable = max(0, n_frames - config.model.action_horizon)
        if args.num_samples >= usable:
            frames_to_run = list(range(usable))
        else:
            stride = usable // args.num_samples
            frames_to_run = [i * stride for i in range(args.num_samples)]

    print(f"[3/4] Will sample {len(frames_to_run)} frames "
          f"(task={args.task}, ep={args.episode}, frames={frames_to_run})")

    tokenizer = PaligemmaTokenizer(
        max_len=config.model.max_token_len,
        prompt_format=config.model.prompt_format,
        prediction_format=getattr(config.model, "prediction_format", "default"),
        reasoning_mask_prob=0.0,
    )

    print(f"[4/4] Sampling actions (num_flow_steps={args.num_flow_steps}) ...")
    rng_root = jax.random.PRNGKey(args.seed)

    all_pred: list[np.ndarray] = []   # (H, 14) per frame
    all_gt: list[np.ndarray] = []
    rows: list[dict] = []

    for i, frame_idx in enumerate(frames_to_run):
        sample = fetch_one_sample(ds, args.episode, frame_idx, config.model.action_horizon)

        # Build prompt: task only (model has no language target during action expert call)
        tok_out = tokenizer.tokenize(
            sample["task_prompt"],
            reasoning=None, langact=None,
            state=None, plan=None, plan_position="none",
        )
        tokens, mask = tok_out[0], tok_out[1]

        obs = build_observation(
            base_uint8=sample["head_image"],
            wrist_uint8=sample["wrist_image"],
            state=sample["state"],
            tokens=tokens,
            mask=mask,
            image_keys=image_keys,
        )

        rng, rng_root = jax.random.split(rng_root)
        # sample_actions returns (B, H, action_dim)
        pred_actions = model.sample_actions(rng, obs, num_steps=args.num_flow_steps)
        pred_np = np.asarray(pred_actions[0]).astype(np.float32)   # (H, 14)
        gt_np = sample["gt_actions"].astype(np.float32)            # (H, 14)

        # Per-frame metrics
        diff = pred_np - gt_np
        mse = float(np.mean(diff ** 2))
        l1 = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        per_dim_l1 = np.mean(np.abs(diff), axis=0)            # (14,)
        per_dim_pred_std = pred_np.std(axis=0)                # (14,)
        per_dim_gt_std = gt_np.std(axis=0)
        rows.append({
            "frame_idx": frame_idx,
            "phase_idx": sample["phase_idx"],
            "arm": sample["arm_tag"],
            "mse": mse, "l1": l1, "max_abs": max_abs,
            "per_dim_l1": per_dim_l1,
            "per_dim_pred_std": per_dim_pred_std,
            "per_dim_gt_std": per_dim_gt_std,
        })
        all_pred.append(pred_np)
        all_gt.append(gt_np)

        print(f"  [{i+1}/{len(frames_to_run)}] frame={frame_idx:4d} "
              f"phase={sample['phase_idx']:2d}/{sample['n_phases']:2d} arm={sample['arm_tag']:5s} "
              f"  mse={mse:.5f}  l1={l1:.4f}  max_abs={max_abs:.4f}")

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)
    pred_all = np.stack(all_pred, axis=0)   # (N, H, 14)
    gt_all = np.stack(all_gt, axis=0)
    diff_all = pred_all - gt_all
    print(f"  total frames:   {len(rows)}")
    print(f"  mean MSE:       {np.mean(diff_all ** 2):.5f}")
    print(f"  mean L1:        {np.mean(np.abs(diff_all)):.4f}")
    print(f"  max abs error:  {np.max(np.abs(diff_all)):.4f}")
    print()
    print("  Per-dim mean L1 (averaged over all frames):")
    per_dim_l1_all = np.mean(np.abs(diff_all), axis=(0, 1))
    for d in range(per_dim_l1_all.shape[0]):
        print(f"    dim {d:2d}: L1={per_dim_l1_all[d]:.4f}  "
              f"pred_std={pred_all[..., d].std():.4f}  gt_std={gt_all[..., d].std():.4f}")
    print()
    print("  Mode-collapse heuristic:")
    pred_std_all = pred_all.std(axis=(0, 1))
    gt_std_all = gt_all.std(axis=(0, 1))
    ratio = pred_std_all / np.maximum(gt_std_all, 1e-6)
    suspect = [d for d in range(len(ratio)) if gt_std_all[d] > 0.01 and ratio[d] < 0.1]
    print(f"  pred std / gt std per dim: {np.array2string(ratio, precision=3)}")
    if suspect:
        print(f"  ⚠️  Dims with pred_std < 10% of gt_std (likely collapsed): {suspect}")
    else:
        print("  ✅ No dim shows obvious collapse (pred_std ≥ 10% of gt_std for all active dims).")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
