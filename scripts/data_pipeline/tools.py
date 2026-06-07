"""Tool functions exposed to the VLM annotation subagent.

The VLM uses these to pull data on demand, in contrast to v2's push
paradigm where all keyframe pose deltas were placed in the prompt.

Three tools:

- ``get_pose_delta(ep_path, idx1, idx2)``: pose delta between frames
  idx1 and idx2 in both robot frame and idx1's wrist camera frame
- ``get_image(ep_path, frame_idx, view)``: arbitrary-frame image
  access (not limited to keyframes)
- ``get_keyframe_list(ep_path)``: rule-detector keyframe structure for
  planning

All tools take ``ep_path`` — the absolute path to the episode directory
under ``local_data/raw_eps/<ep_dir>/``. Episode metadata (h5, MP4
paths) is resolved relative to this directory.
"""
from __future__ import annotations
import base64
import json
import math
import os
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers (rotation math reused from pose_utils)
# ---------------------------------------------------------------------------

def _euler_xyz_to_matrix(rpy):
    """DROID stores all 6-vec poses (cartesian_position, camera_extrinsics,
    gripper_offset) as EXTRINSIC euler 'xyz' angles, NOT axis-angle. Verified:
    ee_pose ∘ T_ee_wrist == wrist_cam_extrinsic to 0.00cm/0.00deg under euler
    'xyz' (vs 8cm/108deg under axis-angle). Extrinsic xyz = Rz(c)·Ry(b)·Rx(a).
    Matches scipy Rotation.from_euler('xyz', rpy).as_matrix()."""
    a, b, c = (float(x) for x in np.asarray(rpy, dtype=np.float64))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# Back-compat alias: every DROID 6-vec is euler 'xyz'. (The name is historical;
# it does NOT interpret the input as an axis-angle rotvec.)
_rotvec_to_matrix = _euler_xyz_to_matrix


def _matrix_to_rotvec(R):
    cos_a = (np.trace(R) - 1.0) * 0.5
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.acos(cos_a)
    if abs(angle) < 1e-6:
        return np.zeros(3, dtype=np.float64)
    sin_a = math.sin(angle)
    if abs(sin_a) < 1e-6:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * sin_a)
    return axis * angle


def _decompose_rot(angle_deg: float, axis_unit: np.ndarray) -> str:
    """Pretty-print a rotation as decomposed axes (top 2 contributing)."""
    if angle_deg < 0.5:
        return "≈0°"
    proj = {
        "roll":  angle_deg * axis_unit[0],
        "pitch": angle_deg * axis_unit[1],
        "yaw":   angle_deg * axis_unit[2],
    }
    parts = sorted(proj.items(), key=lambda x: abs(x[1]), reverse=True)
    out = []
    for name, deg in parts[:2]:
        if abs(deg) < 0.5:
            continue
        out.append(f"{abs(deg):.0f}° {'-' if deg < 0 else ''}{name}")
    return "≈" + "+".join(out) if out else f"≈{angle_deg:.0f}° mixed"


# ---------------------------------------------------------------------------
# Episode data loaders (cached per ep_path)
# ---------------------------------------------------------------------------

_EP_CACHE: dict[str, dict] = {}


def _load_episode_state(ep_path: str) -> dict:
    """Load ee_pose, T_world_wrist, T_ee_wrist, and keyframe list once per ep."""
    ep_path = os.path.abspath(ep_path)
    if ep_path in _EP_CACHE:
        return _EP_CACHE[ep_path]

    meta_path = os.path.join(ep_path, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json missing in {ep_path}")
    with open(meta_path) as f:
        meta = json.load(f)

    # Self-contained teleop episodes carry their own DROID-shaped trajectory.h5
    # + ep-local mp4s, so read them in place (no DROID_RAW_ROOT / serials).
    local_h5 = os.path.join(ep_path, "trajectory.h5")
    if meta.get("source") == "teleop" and os.path.exists(local_h5):
        import h5py
        with h5py.File(local_h5, "r") as f:
            ee_pose = f["observation/robot_state/cartesian_position"][:].astype(np.float64)
            gripper = f["observation/robot_state/gripper_position"][:].astype(np.float64)
        state = {
            "meta": meta,
            "ee_pose": ee_pose,
            "gripper_width": gripper,
            "T_ee_wrist": None,
            "ext_mp4": os.path.join(ep_path, meta.get("ext_video", "ext.mp4")),
            "wrist_mp4": os.path.join(ep_path, meta.get("wrist_video", "wrist.mp4")),
            "n_frames": ee_pose.shape[0],
        }
        _EP_CACHE[ep_path] = state
        return state

    # ee_pose, T_world_wrist live in the source h5; meta.json caches what
    # we needed for v2. For v3 we need full ee_pose array, so we read h5.
    episode_id = meta["episode_id"]
    # The raw root is derivable from episode_id but is also configurable.
    raw_root = os.environ.get(
        "DROID_RAW_ROOT",
        "/home/numbnut/datasets/droid_raw/1.0.1",
    )
    h5_path = os.path.join(raw_root, episode_id)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"raw h5 not found: {h5_path} (set DROID_RAW_ROOT)"
        )

    import h5py
    md_files = list(Path(os.path.dirname(h5_path)).glob("metadata_*.json"))
    raw_meta = json.load(open(md_files[0])) if md_files else {}
    wrist_serial = raw_meta.get("wrist_cam_serial", "")
    with h5py.File(h5_path, "r") as f:
        ee_pose = f["observation/robot_state/cartesian_position"][:].astype(np.float64)
        gripper = f["observation/robot_state/gripper_position"][:].astype(np.float64)
        offset_key = f"observation/camera_extrinsics/{wrist_serial}_left_gripper_offset"
        T_ee_wrist = (
            f[offset_key][0].astype(np.float64) if offset_key in f else None
        )

    # MP4 paths for image lookup
    ext1_serial = raw_meta.get("ext1_cam_serial", "")
    ext2_serial = raw_meta.get("ext2_cam_serial", "")
    primary_ext = ext1_serial or ext2_serial
    rec_dir = os.path.join(os.path.dirname(h5_path), "recordings", "MP4")
    state = {
        "meta": meta,
        "ee_pose": ee_pose,
        "gripper_width": gripper,
        "T_ee_wrist": T_ee_wrist,
        "ext_mp4": os.path.join(rec_dir, f"{primary_ext}.mp4") if primary_ext else None,
        "wrist_mp4": os.path.join(rec_dir, f"{wrist_serial}.mp4") if wrist_serial else None,
        "n_frames": ee_pose.shape[0],
    }
    _EP_CACHE[ep_path] = state
    return state


def _interaction_events_between(meta: dict, start: int, end: int) -> list[dict]:
    """Return grasp/release/retry keyframes with start < frame_idx < end."""
    out = []
    for kf in meta.get("keyframes", []):
        if start < kf["frame_idx"] < end and kf["type"] in ("grasp", "release", "retry"):
            out.append({"frame_idx": kf["frame_idx"], "type": kf["type"]})
    return out


def _first_event_after(meta: dict, frame: int, kind: str) -> int | None:
    for kf in meta.get("keyframes", []):
        if kf["frame_idx"] > frame and kf["type"] == kind:
            return kf["frame_idx"]
    return None


# ---------------------------------------------------------------------------
# Tool 1: get_pose_delta
# ---------------------------------------------------------------------------

def _compute_delta(state: dict, idx1: int, idx2: int) -> dict:
    """Internal: compute delta without recursion (used by get_pose_delta + _gap_to).
    Returns the 4 delta fields only (no n_frames/events/gap)."""
    p1 = state["ee_pose"][idx1]
    p2 = state["ee_pose"][idx2]
    dxyz_world = p2[:3] - p1[:3]
    R1 = _rotvec_to_matrix(p1[3:6])
    R2 = _rotvec_to_matrix(p2[3:6])
    R_rel = R2 @ R1.T
    rv = _matrix_to_rotvec(R_rel)
    angle = math.degrees(float(np.linalg.norm(rv)))
    axis_world = rv / (np.linalg.norm(rv) + 1e-12)
    delta_robot = {
        "forward": float(dxyz_world[0] * 100.0),
        "left":    float(dxyz_world[1] * 100.0),
        "up":      float(dxyz_world[2] * 100.0),
    }
    rot_world_str = _decompose_rot(angle, axis_world)
    delta_ee = None
    rot_ee_str = None
    if state["T_ee_wrist"] is not None:
        R_ee_w = _rotvec_to_matrix(state["T_ee_wrist"][3:])
        R_world_wrist = R1 @ R_ee_w
        dxyz_wrist = R_world_wrist.T @ dxyz_world
        delta_ee = {
            "forward": float(dxyz_wrist[2] * 100.0),
            "left":    float(-dxyz_wrist[0] * 100.0),
            "up":      float(dxyz_wrist[1] * 100.0),
        }
        axis_wrist = R_world_wrist.T @ axis_world
        visual_axis = np.array([axis_wrist[2], -axis_wrist[0], axis_wrist[1]])
        rot_ee_str = _decompose_rot(angle, visual_axis)
    return {
        "delta_robot": delta_robot,
        "delta_ee": delta_ee,
        "delta_rot_world": rot_world_str,
        "delta_rot_ee": rot_ee_str,
    }


def get_pose_delta(ep_path: str, idx1: int, idx2: int) -> dict:
    """Return pose delta from idx1 → idx2 in robot frame and idx1's wrist frame.

    Wrist-frame delta uses the source pose's EE rotation × hand-eye
    calibration (T_ee_wrist from DROID raw). Falls back to EE frame
    (no wrist projection) when calibration is missing.
    """
    state = _load_episode_state(ep_path)
    n = state["n_frames"]
    if not (0 <= idx1 < n and 0 <= idx2 < n):
        raise IndexError(f"frame out of range: idx1={idx1}, idx2={idx2}, n={n}")
    if idx2 <= idx1:
        raise ValueError(f"idx2 must be > idx1; got {idx1} → {idx2}")

    delta = _compute_delta(state, idx1, idx2)
    g = state.get("gripper_width")
    gripper = None
    if g is not None and idx1 < len(g) and idx2 < len(g):
        # DROID gripper_position: ~0 = open, ~1 = closed.
        def _glabel(v):
            return "open" if v < 0.2 else ("closed" if v > 0.6 else "partial")
        gripper = (f"{_glabel(g[idx1])}({g[idx1]:.2f}) → "
                   f"{_glabel(g[idx2])}({g[idx2]:.2f})")
    return {
        **delta,
        "gripper": gripper,
        "n_frames": idx2 - idx1,
        "interaction_events_in_range": _interaction_events_between(
            state["meta"], idx1, idx2),
        "gap_to_grasp":   _gap_to(state, idx2, "grasp"),
        "gap_to_release": _gap_to(state, idx2, "release"),
    }


def _gap_to(state: dict, from_frame: int, kind: str) -> dict | None:
    target = _first_event_after(state["meta"], from_frame, kind)
    if target is None or target <= from_frame:
        return None
    return {"target_frame": target, **_compute_delta(state, from_frame, target)}


# ---------------------------------------------------------------------------
# Tool 2: get_image
# ---------------------------------------------------------------------------

def get_image(
    ep_path: str,
    frame_idx: int,
    view: Literal["ext", "wrist"] = "ext",
) -> bytes:
    """Return JPEG bytes of `view` camera at `frame_idx` (any frame, not only kf)."""
    import cv2
    state = _load_episode_state(ep_path)
    mp4 = state["ext_mp4"] if view == "ext" else state["wrist_mp4"]
    if not mp4 or not os.path.exists(mp4):
        raise FileNotFoundError(f"MP4 not found for view={view}: {mp4}")
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"failed to read {view} frame {frame_idx} from {mp4}")
    # Encode JPEG
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)


def get_image_base64(ep_path: str, frame_idx: int, view: str = "ext") -> str:
    """Convenience: get_image + base64. For passing in API call bodies."""
    return base64.b64encode(get_image(ep_path, frame_idx, view)).decode("ascii")


# ---------------------------------------------------------------------------
# Tool 3: get_keyframe_list
# ---------------------------------------------------------------------------

def get_keyframe_list(ep_path: str) -> list[dict]:
    """Rule-detector keyframes, structural reference for VLM planning."""
    state = _load_episode_state(ep_path)
    out = []
    for kf in state["meta"].get("keyframes", []):
        out.append({
            "kf_idx": kf["idx"],
            "frame_idx": kf["frame_idx"],
            "type": kf["type"],
            "gripper_state": kf["gripper_state"],
            "near_interaction": kf.get("near_interaction", False),
            "interaction_context": kf.get("interaction_context"),
        })
    return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python tools.py <ep_path> [smoke]")
        sys.exit(0)
    ep = sys.argv[1]
    print(f"=== keyframes for {os.path.basename(ep)} ===")
    kfs = get_keyframe_list(ep)
    for kf in kfs[:5]:
        print(f"  {kf}")
    print(f"  ... ({len(kfs)} total)")
    if len(kfs) >= 2:
        i1, i2 = kfs[0]["frame_idx"], kfs[1]["frame_idx"]
        print(f"\n=== get_pose_delta({i1}, {i2}) ===")
        d = get_pose_delta(ep, i1, i2)
        for k, v in d.items():
            print(f"  {k}: {v}")
        # Image test
        print(f"\n=== get_image({i1}, ext) ===")
        img = get_image(ep, i1, "ext")
        print(f"  {len(img)} bytes JPEG")
