"""Extract keyframes + meta.json from DROID raw format (HDF5 + MP4).

Uses `iter_droid_raw`, threads T_ee_wrist through pose_delta so Δee is
expressed in the actual wrist camera frame (left-handed visual labels:
forward/left/up matching wrist image axes).

Output per episode: meta.json + ext1/wrist keyframe JPEGs.

Usage:
    python /tmp/extract_raw.py \
        --root  ~/datasets/droid_raw/1.0.1  \
        --out   /tmp/raw_eps  \
        --max-episodes 5  \
        --labs AUTOLab  \
        [--include-failure]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

# Allow running from either local checkout or pod path.
for _p in (
    "/home/numbnut/worksapce/RoboTwin/policy/lap/scripts",
    "/data/zhaoqc/RoboTwin/policy/lap/scripts",
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from annotate_droid.droid_reader import iter_droid_raw
from annotate_droid.keyframe import detect_keyframes
from annotate_droid.pose_utils import pose_delta
from PIL import Image

_INTERACTION_TYPES = {"grasp", "release", "retry"}
_TRANSITION_WINDOW = 2
_NEAR_RADIUS = 2


def compute_pose_delta_strs(keyframes, ee_pose, T_ee_wrist):
    interaction_positions = [
        i for i, kf in enumerate(keyframes) if kf.type in _INTERACTION_TYPES
    ]

    def gap_target(i):
        for past_idx in interaction_positions:
            if past_idx <= i and (i - past_idx) <= _TRANSITION_WINDOW:
                return None
        for future_idx in interaction_positions:
            if future_idx > i:
                return future_idx
        return None

    out = []
    for i, kf in enumerate(keyframes):
        cur_pose = ee_pose[kf.t]
        if i + 1 < len(keyframes):
            next_pose = ee_pose[keyframes[i + 1].t]
        else:
            next_pose = cur_pose
        fwd = pose_delta(next_pose, cur_pose, T_ee_wrist=T_ee_wrist)
        target_idx = gap_target(i)
        if target_idx is not None:
            tkf = keyframes[target_idx]
            gap = pose_delta(ee_pose[tkf.t], cur_pose, T_ee_wrist=T_ee_wrist)
            out.append(f"gap-to-{tkf.type}: {gap}\n    next-step: {fwd}")
        else:
            out.append(str(fwd))
    return out


def compute_interaction_context(keyframes):
    near = [False] * len(keyframes)
    ctx = [None] * len(keyframes)
    for i, kf in enumerate(keyframes):
        if kf.type in _INTERACTION_TYPES:
            for j in range(max(0, i - _NEAR_RADIUS),
                           min(len(keyframes), i + _NEAR_RADIUS + 1)):
                near[j] = True
    for i, kf in enumerate(keyframes):
        if not near[i] or kf.type in _INTERACTION_TYPES:
            continue
        nearest = None
        nearest_d = 999
        for j, kf2 in enumerate(keyframes):
            if kf2.type in _INTERACTION_TYPES and abs(i - j) < nearest_d:
                nearest_d = abs(i - j)
                nearest = (j, kf2.type)
        if nearest:
            j, itype = nearest
            ctx[i] = f"pre_{itype}" if i < j else f"post_{itype}"
    return near, ctx


def _ep_dir_name(episode_id: str, idx: int) -> str:
    """Make a stable per-episode dir name from the episode_id (h5 path)."""
    # episode_id is like "AUTOLab/failure/2023-07-07/Fri_Jul__7.../trajectory.h5"
    tail = episode_id.replace("/", "_").replace(":", "-")
    if tail.endswith("_trajectory.h5"):
        tail = tail[: -len("_trajectory.h5")]
    return f"ep{idx:03d}__{tail[-80:]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="raw dataset root (e.g. ~/datasets/droid_raw/1.0.1)")
    ap.add_argument("--out", required=True, help="output dir for per-ep folders")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--labs", nargs="*", default=None, help="lab whitelist")
    ap.add_argument("--include-failure", action="store_true",
                    help="include failure/ subtree (default: success only)")
    ap.add_argument("--skipped-log", default=None,
                    help="path to append skipped-episode log")
    ap.add_argument("--whitelist", default=None,
                    help="CSV from screen_episodes.py — only process eps "
                         "whose classification matches --whitelist-class")
    ap.add_argument("--whitelist-class", default="good",
                    help="comma-separated class list (default: good)")
    args = ap.parse_args()

    # Load whitelist if provided
    whitelist_ids: set[str] | None = None
    if args.whitelist:
        import csv as _csv
        allowed = set(c.strip() for c in args.whitelist_class.split(","))
        whitelist_ids = set()
        with open(args.whitelist) as _f:
            for row in _csv.DictReader(_f):
                if row.get("classification", "") in allowed:
                    whitelist_ids.add(row["ep_id"])
        print(f"  whitelist loaded: {len(whitelist_ids)} eps (classes {allowed})")

    root = os.path.expanduser(args.root)
    out_root = os.path.expanduser(args.out)
    os.makedirs(out_root, exist_ok=True)
    skipped_log = args.skipped_log or os.path.join(out_root, "skipped_eps.txt")
    open(skipped_log, "w").close()  # truncate

    n_done = 0
    n_skipped_wl = 0
    for idx, bundle in enumerate(iter_droid_raw(
        root,
        success_only=not args.include_failure,
        labs=args.labs,
        max_episodes=None,  # we do our own max_episodes after whitelist filter
        skipped_log_path=skipped_log,
    )):
        if whitelist_ids is not None and bundle.episode_id not in whitelist_ids:
            n_skipped_wl += 1
            continue
        if args.max_episodes is not None and n_done >= args.max_episodes:
            break
        kfs = detect_keyframes(
            gripper_width=bundle.gripper_width,
            ee_pos=bundle.ee_pos,
            fps=bundle.fps,
        )
        if not kfs:
            print(f"  [skip] ep{idx} {bundle.episode_id}: no keyframes")
            continue

        ep_dir = os.path.join(out_root, _ep_dir_name(bundle.episode_id, idx))
        os.makedirs(ep_dir, exist_ok=True)

        delta_strs = compute_pose_delta_strs(kfs, bundle.ee_pose, bundle.T_ee_wrist)
        near, ctx = compute_interaction_context(kfs)

        kf_records = []
        for i, kf in enumerate(kfs):
            img = Image.fromarray(bundle.frame_loader(kf.t))
            img_name = f"kf{i:02d}_f{kf.t:04d}.jpg"
            img.save(f"{ep_dir}/{img_name}", quality=85)
            wrist_name = None
            if bundle.wrist_loader is not None:
                w = Image.fromarray(bundle.wrist_loader(kf.t))
                wrist_name = f"kf{i:02d}_f{kf.t:04d}_wrist.jpg"
                w.save(f"{ep_dir}/{wrist_name}", quality=85)
            kf_records.append({
                "idx": i,
                "frame_idx": kf.t,
                "type": kf.type,
                "gripper_state": kf.gripper_state or "unknown",
                "near_interaction": near[i],
                "interaction_context": ctx[i],
                "pose_delta_str": delta_strs[i],
                "image_file": img_name,
                "wrist_image_file": wrist_name,
            })

        meta = {
            "episode_id": bundle.episode_id,
            "task_instruction": bundle.task_instruction or "",
            "n_frames": int(bundle.n_frames),
            "fps": bundle.fps,
            "T_ee_wrist": bundle.T_ee_wrist.tolist()
                          if bundle.T_ee_wrist is not None else None,
            "calibrated_wrist_frame": bundle.T_ee_wrist is not None,
            "keyframes": kf_records,
        }
        with open(f"{ep_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        n_done += 1
        print(f"  [ok] ep{idx} {len(kfs):>3} kf → {ep_dir}  "
              f"task: {bundle.task_instruction[:50]!r}")

    print(f"\nDone. {n_done} episodes extracted. "
          f"Skipped (whitelist): {n_skipped_wl}. Skipped log: {skipped_log}")


if __name__ == "__main__":
    main()
