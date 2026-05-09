"""View training samples from RoboTwin task datasets + Bridge V2 ECoT side-by-side.

Goal: surface the cascade-VLA mapping (`[plan]` / `[stage]` / `[action]`) for
each dataset so we can pick a unified strategy for Stage 2 action-expert
training. Runs locally on the host (data is not on the pod).

Each dataset has its own metadata format. This script normalizes them into a
common per-step record:

    {
      "task_prompt": str,           # episode-level task description
      "plan":         str,          # multi-step plan (cascade-VLA [plan])
      "stage":        str,          # current sub-goal reasoning (cascade-VLA [stage])
      "action_lang":  str,          # atomic step description (cascade-VLA [action])
      "phase_kind":   str,          # approach/grasp/transport/lift_down/...
      "arm_tag":      str,          # left/right
      "frame_idx":    int,
      "phase_idx":    int,
      "n_phases":     int,
    }

Usage:

    cd /home/numbnut/worksapce/RoboTwin
    .venv/bin/python policy/lap/scripts/view_robotwin_dataset.py \
        --out-dir /tmp/robotwin_dataset_view \
        --frames-per-ep 3

Add `--bridge` to also dump 1 Bridge V2 ECoT episode for direct comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

DATA_ROOT = Path("/home/numbnut/worksapce/RoboTwin/data")
BRIDGE_JSON = Path(
    "/home/numbnut/.cache/huggingface/hub/"
    "datasets--Embodied-CoT--embodied_features_bridge/snapshots/"
    "854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
)

# Datasets to inspect, in display order.
ROBOTWIN_DATASETS = [
    "pick_place_primitive",
    "arrange_blocks_line",
    "arrange_blocks_l_shape",
    "arrange_blocks_u_shape",
    "stack_blocks_n_stack_n_v1_open_K3",
]


def find_phase_for_frame(phases: list[dict], frame_idx: int) -> tuple[int, dict] | tuple[None, None]:
    """Return (phase_idx, phase_dict) covering ``frame_idx`` (start_frame ≤ idx < end_frame)."""
    for i, ph in enumerate(phases):
        if ph["start_frame"] <= frame_idx < ph["end_frame"]:
            return i, ph
    return None, None


# ----------------------------------------------------------------------------
# Per-dataset cascade extraction
# ----------------------------------------------------------------------------

def _synth_pickplace_plan(task_spec: dict) -> str:
    pick_obj = task_spec["pick"]["descriptor"]
    place_loc = task_spec["place"]["loc_at_descriptor"]
    return f"Pick up {pick_obj} and place it {place_loc}."


def _synth_pickplace_stage(phase: dict) -> str:
    kind_words = {
        "approach": "approach the target above",
        "grasp": "grasp the target",
        "transport": "transport the target to the destination",
        "lift_down": "place the target at the destination",
    }
    arm = phase["arm_tag"]
    desc = kind_words.get(phase["kind"], phase["kind"])
    return f"The {arm} arm will {desc} as part of the pick-and-place."


def cascade_from_pick_place(meta: dict, phase_idx: int, phase: dict) -> dict[str, str]:
    return {
        "task_prompt": _synth_pickplace_plan(meta["task_spec"]),  # use as both task & plan
        "plan": _synth_pickplace_plan(meta["task_spec"]),
        "stage": _synth_pickplace_stage(phase),
        "action_lang": phase["phase_prompts"][0],
    }


def cascade_from_arrange(meta: dict, phase_idx: int, phase: dict) -> dict[str, str]:
    cot = meta.get("cot", {})
    per_phase_text = cot.get("per_phase_text", []) if isinstance(cot, dict) else []
    text = per_phase_text[phase_idx] if phase_idx < len(per_phase_text) else ""
    # Split "[Subgoal] X. [Step] Y." → stage = X, action = Y.
    stage_part = ""
    action_part = phase["phase_prompts"][0] if phase.get("phase_prompts") else ""
    if "[Subgoal]" in text and "[Step]" in text:
        try:
            sub_idx = text.index("[Subgoal]") + len("[Subgoal]")
            step_idx = text.index("[Step]")
            stage_part = text[sub_idx:step_idx].strip(". ").strip()
            action_part = text[step_idx + len("[Step]"):].strip(". ").strip()
        except ValueError:
            pass
    if not stage_part:
        # Fallback: use phase kind + subgoal-ish summary.
        stage_part = phase.get("subgoal_prompt") or f"Phase {phase_idx} ({phase['kind']})"
    return {
        "task_prompt": meta.get("task_prompt", ""),
        "plan": meta.get("task_prompt", ""),  # arrange has only task-level prompt
        "stage": stage_part,
        "action_lang": action_part,
    }


def cascade_from_stack(meta: dict, phase_idx: int, phase: dict) -> dict[str, str]:
    return {
        "task_prompt": meta.get("task_prompt", ""),
        "plan": meta.get("task_prompt", ""),
        "stage": phase.get("subgoal_prompt", ""),
        "action_lang": phase["phase_prompts"][0] if phase.get("phase_prompts") else "",
    }


def cascade_extractor_for(dataset_name: str):
    """Return the cascade-VLA extraction function appropriate for a dataset name."""
    if dataset_name == "pick_place_primitive":
        return cascade_from_pick_place
    if dataset_name.startswith("arrange_blocks_"):
        return cascade_from_arrange
    if dataset_name.startswith("stack_blocks_"):
        return cascade_from_stack
    raise ValueError(f"No cascade extractor for {dataset_name!r}")


# ----------------------------------------------------------------------------
# HDF5 loading
# ----------------------------------------------------------------------------

def load_episode_record(dataset: str, episode_idx: int = 0) -> tuple[dict, dict]:
    """Return (metadata_json, hdf5_summary)."""
    import h5py

    base = DATA_ROOT / dataset / "demo_clean"
    meta_path = base / "metadata" / f"episode{episode_idx}.json"
    hdf5_path = base / "data" / f"episode{episode_idx}.hdf5"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata not found: {meta_path}")
    if not hdf5_path.exists():
        raise FileNotFoundError(f"hdf5 not found: {hdf5_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    summary = {"path": str(hdf5_path), "fields": {}}
    with h5py.File(hdf5_path, "r") as f:
        n_frames = None

        def collect(name, obj):
            nonlocal n_frames
            if isinstance(obj, h5py.Dataset) and obj.shape:
                summary["fields"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if n_frames is None and len(obj.shape) >= 1 and obj.shape[0] > 0:
                    n_frames = obj.shape[0]
        f.visititems(collect)
        summary["n_frames"] = n_frames

    return meta, summary


def decode_jpeg_frame(hdf5_path: str, camera_key: str, frame_idx: int) -> np.ndarray:
    """Return RGB HWC uint8 image from the JPEG-encoded hdf5 dataset."""
    import cv2
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        ds_path = f"observation/{camera_key}/rgb"
        if ds_path not in f:
            raise KeyError(ds_path)
        raw = f[ds_path][frame_idx]
    if hasattr(raw, "tobytes"):
        raw_bytes = raw.tobytes()
    else:
        raw_bytes = bytes(raw)
    bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"cv2.imdecode failed for {camera_key} frame {frame_idx}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_action_at(hdf5_path: str, frame_idx: int) -> dict[str, Any]:
    """Pull action vector and gripper state at one frame."""
    import h5py

    out = {}
    with h5py.File(hdf5_path, "r") as f:
        if "joint_action/vector" in f:
            out["joint_vector_14d"] = f["joint_action/vector"][frame_idx].tolist()
        if "endpose/left_endpose" in f:
            out["left_endpose_7d"] = f["endpose/left_endpose"][frame_idx].tolist()
        if "endpose/right_endpose" in f:
            out["right_endpose_7d"] = f["endpose/right_endpose"][frame_idx].tolist()
        if "endpose/left_gripper" in f:
            out["left_gripper"] = float(f["endpose/left_gripper"][frame_idx])
        if "endpose/right_gripper" in f:
            out["right_gripper"] = float(f["endpose/right_gripper"][frame_idx])
    return out


# ----------------------------------------------------------------------------
# Bridge V2 ECoT (compare reference)
# ----------------------------------------------------------------------------

def load_bridge_sample(num_episodes: int = 1, frames_per_ep: int = 3) -> list[dict]:
    """Stream 1 episode from Bridge V2 ECoT JSON; return per-frame cascade dicts."""
    try:
        import ijson  # type: ignore
    except ImportError:
        sys.stderr.write("[warn] ijson not installed; skipping Bridge V2 sample\n")
        return []
    if not BRIDGE_JSON.exists():
        sys.stderr.write(f"[warn] Bridge JSON not found: {BRIDGE_JSON}; skipping\n")
        return []

    out: list[dict] = []
    eps_seen = 0
    with open(BRIDGE_JSON, "rb") as f:
        top = ijson.kvitems(f, "", use_float=True)
        for file_path, episodes in top:
            if not isinstance(episodes, dict):
                continue
            for ep_id, ep in episodes.items():
                if eps_seen >= num_episodes:
                    return out
                reasoning = ep.get("reasoning", {})
                if not isinstance(reasoning, dict):
                    continue
                step_keys = sorted(reasoning.keys(), key=lambda s: int(s) if s.isdigit() else 0)
                # Pick frames evenly distributed across the episode.
                if not step_keys:
                    continue
                stride = max(1, len(step_keys) // frames_per_ep)
                picks = step_keys[::stride][:frames_per_ep]
                for k in picks:
                    r = reasoning[k]
                    out.append({
                        "dataset": "bridge_v2_ecot",
                        "episode": f"{file_path}::{ep_id}",
                        "frame_idx": int(k),
                        "task_prompt": r.get("task", ""),
                        "plan": r.get("plan", ""),
                        "stage": r.get("subtask_reason", ""),
                        "action_lang": r.get("subtask", ""),
                        "phase_kind": "(none)",
                        "arm_tag": "(none)",
                        "phase_idx": -1,
                        "n_phases": -1,
                    })
                eps_seen += 1
                if eps_seen >= num_episodes:
                    return out
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def fmt_action_short(act: dict[str, Any]) -> str:
    parts = []
    for k in ["left_endpose_7d", "right_endpose_7d", "left_gripper", "right_gripper"]:
        v = act.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            v_str = "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"
        else:
            v_str = f"{v:+.3f}"
        parts.append(f"{k}={v_str}")
    return "  |  ".join(parts)


def dump_one_dataset(name: str, frames_per_ep: int, out_dir: Path, save_images: bool) -> list[dict]:
    print(f"\n{'='*88}\n  DATASET: {name}\n{'='*88}")
    meta, hdf5_summary = load_episode_record(name, episode_idx=0)
    phases = meta.get("phases", [])
    n_frames = hdf5_summary["n_frames"]
    cameras = sorted({k.split("/")[1] for k in hdf5_summary["fields"] if k.startswith("observation/") and k.endswith("/rgb")})

    print(f"  hdf5 episode 0     : {hdf5_summary['path']}")
    print(f"  total frames       : {n_frames}")
    print(f"  cameras present    : {cameras}")
    print(f"  task_prompt        : {meta.get('task_prompt') or '(none — synthesized below)'}")
    print(f"  task_family        : {meta.get('task_family')}")
    print(f"  num phases         : {len(phases)}")
    print(f"  phase kinds        : {[p['kind'] for p in phases][:10]}{' ...' if len(phases) > 10 else ''}")
    print(f"  episode_success    : {meta.get('episode_success')}")

    extractor = cascade_extractor_for(name)
    # Pick frames distributed across phases (one per phase if few phases, else evenly).
    chosen_frames: list[int] = []
    if len(phases) <= frames_per_ep:
        # one per phase, mid-phase
        for ph in phases:
            mid = (ph["start_frame"] + ph["end_frame"]) // 2
            chosen_frames.append(mid)
    else:
        # evenly distributed
        step = len(phases) // frames_per_ep
        for i in range(frames_per_ep):
            ph = phases[i * step]
            mid = (ph["start_frame"] + ph["end_frame"]) // 2
            chosen_frames.append(mid)

    records: list[dict] = []
    out_subdir = out_dir / name
    if save_images:
        out_subdir.mkdir(parents=True, exist_ok=True)

    for frame_idx in chosen_frames:
        phase_idx, phase = find_phase_for_frame(phases, frame_idx)
        if phase is None:
            continue
        cas = extractor(meta, phase_idx, phase)
        action = get_action_at(hdf5_summary["path"], frame_idx)

        rec = {
            "dataset": name,
            "episode": 0,
            "frame_idx": frame_idx,
            "phase_idx": phase_idx,
            "n_phases": len(phases),
            "phase_kind": phase["kind"],
            "arm_tag": phase.get("arm_tag", "?"),
            **cas,
            "action": action,
        }
        records.append(rec)

        print(f"\n  --- frame {frame_idx} (phase {phase_idx}/{len(phases)}: {phase['kind']}, arm={phase.get('arm_tag')}) ---")
        print(f"    task_prompt: {textwrap.shorten(cas['task_prompt'], 110)}")
        print(f"    [plan]     : {textwrap.shorten(cas['plan'], 110)}")
        print(f"    [stage]    : {textwrap.shorten(cas['stage'], 110)}")
        print(f"    [action]   : {textwrap.shorten(cas['action_lang'], 110)}")
        if action:
            short = fmt_action_short(action)
            print(f"    action     : {short[:300]}")

        if save_images:
            for cam in ["head_camera", "left_camera", "right_camera"]:
                if cam in cameras:
                    try:
                        img = decode_jpeg_frame(hdf5_summary["path"], cam, frame_idx)
                        out_path = out_subdir / f"f{frame_idx:04d}_{cam}.png"
                        try:
                            import cv2
                            cv2.imwrite(str(out_path), img[..., ::-1])  # RGB→BGR for cv2
                        except ImportError:
                            from PIL import Image
                            Image.fromarray(img).save(out_path)
                    except Exception as e:
                        print(f"    [warn] failed to dump {cam} frame {frame_idx}: {e}")
            print(f"    images     : saved {len([c for c in ['head_camera','left_camera','right_camera'] if c in cameras])} cams to {out_subdir.relative_to(out_dir)}/")

    return records


def dump_bridge(num_episodes: int, frames_per_ep: int) -> list[dict]:
    print(f"\n{'='*88}\n  DATASET: bridge_v2_ecot  (reference; from Stage 1 pretraining)\n{'='*88}")
    samples = load_bridge_sample(num_episodes=num_episodes, frames_per_ep=frames_per_ep)
    if not samples:
        return []
    for s in samples:
        print(f"\n  --- episode {s['episode']} frame {s['frame_idx']} ---")
        print(f"    task_prompt: {textwrap.shorten(s['task_prompt'], 110)}")
        print(f"    [plan]     : {textwrap.shorten(s['plan'], 110)}")
        print(f"    [stage]    : {textwrap.shorten(s['stage'], 110)}")
        print(f"    [action]   : {textwrap.shorten(s['action_lang'], 110)}")
    return samples


def write_comparison_table(all_records: list[dict], out_path: Path) -> None:
    """Emit a markdown comparison table for visual side-by-side review."""
    rows: list[str] = [
        "| Dataset | Frame | Phase | Arm | task_prompt | [plan] | [stage] | [action] |",
        "|---------|------:|------:|:---:|-------------|--------|---------|----------|",
    ]
    for r in all_records:
        rows.append(
            "| {ds} | {frame} | {ph}/{nph} | {arm} | {task} | {plan} | {stage} | {act} |".format(
                ds=r["dataset"],
                frame=r["frame_idx"],
                ph=r.get("phase_idx", "-"),
                nph=r.get("n_phases", "-"),
                arm=r.get("arm_tag", "-"),
                task=textwrap.shorten(r.get("task_prompt", ""), 60).replace("|", "\\|"),
                plan=textwrap.shorten(r.get("plan", ""), 60).replace("|", "\\|"),
                stage=textwrap.shorten(r.get("stage", ""), 60).replace("|", "\\|"),
                act=textwrap.shorten(r.get("action_lang", ""), 60).replace("|", "\\|"),
            )
        )
    out_path.write_text("\n".join(rows) + "\n")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", nargs="*", default=ROBOTWIN_DATASETS,
                   help=f"RoboTwin datasets to inspect. Default: {ROBOTWIN_DATASETS}.")
    p.add_argument("--frames-per-ep", type=int, default=3,
                   help="How many frames to dump per dataset episode (default 3).")
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/robotwin_dataset_view"),
                   help="Where to drop PNG dumps and the markdown table.")
    p.add_argument("--no-images", action="store_true",
                   help="Skip JPEG decode + PNG save (faster).")
    p.add_argument("--bridge", action="store_true",
                   help="Also dump Bridge V2 ECoT samples for direct comparison.")
    p.add_argument("--bridge-episodes", type=int, default=1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for ds in args.datasets:
        try:
            recs = dump_one_dataset(
                name=ds,
                frames_per_ep=args.frames_per_ep,
                out_dir=args.out_dir,
                save_images=not args.no_images,
            )
            all_records.extend(recs)
        except FileNotFoundError as e:
            print(f"\n[skip] {ds}: {e}")

    if args.bridge:
        bridge_recs = dump_bridge(num_episodes=args.bridge_episodes, frames_per_ep=args.frames_per_ep)
        all_records.extend(bridge_recs)

    table_path = args.out_dir / "comparison.md"
    write_comparison_table(all_records, table_path)
    print(f"\n{'='*88}")
    print(f"  Wrote markdown comparison: {table_path}")
    if not args.no_images:
        print(f"  Images dumped under     : {args.out_dir}")
    print(f"{'='*88}")


if __name__ == "__main__":
    main()
