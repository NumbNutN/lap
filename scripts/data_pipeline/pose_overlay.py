"""Project the wrist-camera coordinate frame into the EXTERNAL camera image.

Used by the v3 viewer's optional "pose-axis overlay": at any frame it draws a
small RGB triad (X=red, Y=green, Z=blue) showing the wrist frame's orientation
*as seen from the external camera*, plus the same triad at the keyframe's
`chunk_end_frame`, with faint connectors to read off the rotation sweep.

Calibration notes (be honest — this is a visualization aid, not metrology):
- External-camera extrinsics come from the raw DROID h5 (`camera_extrinsics/
  <ext_serial>_left`, a per-frame 6-vec [xyz, rotvec] = camera pose in robot
  base frame).
- DROID does NOT ship intrinsics in the h5/metadata (they live in the unparsed
  ZED SVO), so we assume a pinhole with `fx=fy≈700, cx,cy=center` at 1280x720.
- The optical frame is recovered empirically: with `c = R_cam @ (p_world - t_cam)`
  (no transpose), the OpenCV optical axes (x-right, y-down, z-fwd) are
  `(x,y,z) = (c0, -c2, c1)` — a proper rotation (det +1) that centres the
  gripper across the trajectory (~26 deg mean off-axis).
- The triad ORIENTATION is exact (driven only by the extrinsic rotation). The
  triad ANCHOR is approximate: it sits at the projected EE *flange*, which is
  offset ~10-15cm from the visible gripper/cup, and uses a guessed focal length.
  So read the rotation, not the exact pixel.

Output (precomputed, h5-free on HF Spaces): `axis_overlay.json` =
  {"view": "ext", "image_w": W, "image_h": H, "axis_len_cm": L,
   "frames": [{"o":[u,v], "x":[u,v], "y":[u,v], "z":[u,v], "valid":bool}, ...]}
The viewer reads this; it never needs the h5.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path

import numpy as np

# Optical-frame remap: c = R @ (p - t); optical = M @ c, M proper rotation.
_M_OPT = np.array([[1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0],
                   [0.0, 1.0, 0.0]], dtype=np.float64)
_DEFAULT_FX = 700.0           # ZED2 720p guess (no intrinsics in DROID raw)
_IMG_W, _IMG_H = 1280, 720
_AXIS_LEN_CM = 8.0            # physical length of drawn axes


def _rv2R(rv):
    rv = np.asarray(rv, dtype=np.float64)
    a = float(np.linalg.norm(rv))
    if a < 1e-9:
        return np.eye(3, dtype=np.float64)
    k = rv / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def _project(p_world, t_cam, R_cam, fx, w, h):
    """World point -> (u,v,depth) in the external image. depth<=0 => behind."""
    c = _M_OPT @ (R_cam @ (np.asarray(p_world, float) - t_cam))
    depth = c[2]
    if depth <= 1e-4:
        return None, None, depth
    u = fx * c[0] / depth + w / 2.0
    v = fx * c[1] / depth + h / 2.0
    return u, v, depth


def compute_overlay(ep_dir: str, ext_serial: str | None = None,
                    fx: float = _DEFAULT_FX,
                    axis_len_cm: float = _AXIS_LEN_CM,
                    w: int = _IMG_W, h: int = _IMG_H) -> dict:
    """Compute the per-frame wrist-triad projection into the ext image.

    Reads the raw DROID h5 (ee_pose, ext extrinsics, hand-eye T_ee_wrist).
    Returns the dict written to axis_overlay.json.
    """
    import h5py
    meta = json.load(open(os.path.join(ep_dir, "meta.json")))
    raw_root = os.environ.get("DROID_RAW_ROOT",
                              "/home/numbnut/datasets/droid_raw/1.0.1")
    h5_path = os.path.join(raw_root, meta["episode_id"])
    ep_root = os.path.dirname(h5_path)
    md_files = list(Path(ep_root).glob("metadata_*.json"))
    raw_meta = json.load(open(md_files[0])) if md_files else {}
    if ext_serial is None:
        ext_serial = raw_meta.get("ext1_cam_serial") or raw_meta.get("ext2_cam_serial")
    wrist_serial = raw_meta.get("wrist_cam_serial", "")

    with h5py.File(h5_path, "r") as f:
        ee = f["observation/robot_state/cartesian_position"][:].astype(np.float64)
        ext = f[f"observation/camera_extrinsics/{ext_serial}_left"][:].astype(np.float64)
        off_key = f"observation/camera_extrinsics/{wrist_serial}_left_gripper_offset"
        T_ee_wrist = f[off_key][0].astype(np.float64) if off_key in f else None

    n = ee.shape[0]
    L = axis_len_cm / 100.0
    frames = []
    for i in range(n):
        t_cam = ext[i][:3]
        R_cam = _rv2R(ext[i][3:6])
        R_ee = _rv2R(ee[i][3:6])
        # wrist origin + axes in world frame
        if T_ee_wrist is not None:
            origin = ee[i][:3] + R_ee @ T_ee_wrist[:3]
            R_ww = R_ee @ _rv2R(T_ee_wrist[3:6])
        else:
            origin = ee[i][:3]
            R_ww = R_ee
        ou, ov, od = _project(origin, t_cam, R_cam, fx, w, h)
        rec = {"valid": ou is not None}
        # Orthographic axis directions in image space (x-right, y-down),
        # driven ONLY by the extrinsic+wrist rotation — no intrinsics, no
        # anchor. This is the trustworthy part: the gizmo's angles are exact.
        # depth (3rd optical comp) is dropped, then 2D-normalised.
        ortho = {}
        for ax, name in ((R_ww[:, 0], "x"), (R_ww[:, 1], "y"), (R_ww[:, 2], "z")):
            c = _M_OPT @ (R_cam @ ax)
            ortho[name] = [round(float(c[0]), 4), round(float(c[1]), 4),
                           round(float(c[2]), 4)]  # [img_x, img_y, depth(z)]
        rec["ortho"] = ortho
        if ou is not None:
            rec["o"] = [round(ou, 1), round(ov, 1)]
            for ax, name in ((R_ww[:, 0], "x"), (R_ww[:, 1], "y"), (R_ww[:, 2], "z")):
                tu, tv, td = _project(origin + ax * L, t_cam, R_cam, fx, w, h)
                rec[name] = [round(tu, 1), round(tv, 1)] if tu is not None else None
        frames.append(rec)

    return {"view": "ext", "ext_serial": ext_serial,
            "image_w": w, "image_h": h, "fx": fx,
            "axis_len_cm": axis_len_cm, "n_frames": n, "frames": frames}


def write_overlay(ep_dir: str, out_path: str | None = None, **kw) -> str:
    out = compute_overlay(ep_dir, **kw)
    out_path = out_path or os.path.join(ep_dir, "axis_overlay.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ep_dir")
    ap.add_argument("--fx", type=float, default=_DEFAULT_FX)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = write_overlay(a.ep_dir, out_path=a.out, fx=a.fx)
    d = json.load(open(p))
    nv = sum(1 for fr in d["frames"] if fr.get("valid"))
    print(f"wrote {p}: {nv}/{d['n_frames']} frames valid")
