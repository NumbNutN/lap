"""Project the wrist-camera coordinate frame into the EXTERNAL camera image.

Used by the v3 viewer's optional "pose-axis overlay": at any frame it draws a
small RGB triad (X=red, Y=green, Z=blue) showing the wrist frame's orientation
*as seen from the external camera*, plus the same triad at the keyframe's
`chunk_end_frame`, with faint connectors to read off the rotation sweep.

Calibration notes (be honest — this is a visualization aid, not metrology):
- External-camera extrinsics come from the raw DROID h5 (`camera_extrinsics/
  <ext_serial>_left`, a per-frame 6-vec [xyz, euler_xyz] = camera pose in robot
  base frame). ALL DROID 6-vecs are EXTRINSIC euler 'xyz', NOT axis-angle —
  verified: ee_pose ∘ T_ee_wrist == wrist_cam_extrinsic to 0.00cm/0.00deg.
- Standard pinhole projection: `p_cam = R_cam^T @ (p_world - t_cam)`, then
  `u = fx*x/z + cx, v = fy*y/z + cy` with OpenCV optical axes (x-right, y-down,
  z-fwd). This lands the projected EE on the visible gripper across the
  trajectory (validated on frame 178), so the triad is genuinely SCENE-ANCHORED.
- DROID does NOT ship intrinsics in the h5/metadata, and the ZED SVO stores
  them compressed (needs the ZED SDK). We use `fx=fy≈700, cx,cy=center` at
  1280x720 — empirically lands on the gripper. Exact focal/principal-point is
  approximate, so treat sub-pixel placement as indicative, not metrology.

Output (precomputed, h5-free on HF Spaces): `axis_overlay.json` =
  {"view": "ext", "image_w": W, "image_h": H, "axis_len_cm": L,
   "frames": [{"o":[u,v], "x":[u,v], "y":[u,v], "z":[u,v], "valid":bool}, ...]}
The viewer reads this; it never needs the h5.
"""
from __future__ import annotations
import json, math, os
from pathlib import Path

import numpy as np

_DEFAULT_FX = 700.0           # ZED2 720p guess (no intrinsics in DROID raw)
_IMG_W, _IMG_H = 1280, 720
_AXIS_LEN_CM = 14.0           # physical length of drawn axes (longer = clearer)


def _rv2R(rpy):
    """DROID 6-vec rotation = EXTRINSIC euler 'xyz' (Rz·Ry·Rx), NOT axis-angle."""
    a, b, c = (float(x) for x in np.asarray(rpy, dtype=np.float64))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _project(p_world, t_cam, R_cam, fx, w, h):
    """World point -> (u,v,depth) in the external image (standard pinhole).
    `R_cam` is camera-to-world; p_cam = R_cam^T @ (p_world - t). depth<=0 = behind."""
    c = R_cam.T @ (np.asarray(p_world, float) - t_cam)
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
        ou, ov, _ = _project(origin, t_cam, R_cam, fx, w, h)
        rec = {"valid": ou is not None}
        # Orthographic axis directions in image space (x-right, y-down),
        # driven ONLY by the extrinsic+wrist rotation — no intrinsics, no
        # anchor. This is the trustworthy part: the gizmo's angles are exact.
        # depth (3rd optical comp) is dropped, then 2D-normalised.
        ortho = {}
        for ax, name in ((R_ww[:, 0], "x"), (R_ww[:, 1], "y"), (R_ww[:, 2], "z")):
            c = R_cam.T @ ax   # world axis dir -> camera frame (x-right,y-down,z-fwd)
            ortho[name] = [round(float(c[0]), 4), round(float(c[1]), 4),
                           round(float(c[2]), 4)]  # [img_x, img_y, depth(z)]
        rec["ortho"] = ortho
        if ou is not None:
            rec["o"] = [round(ou, 1), round(ov, 1)]
            for ax, name in ((R_ww[:, 0], "x"), (R_ww[:, 1], "y"), (R_ww[:, 2], "z")):
                tu, tv, _ = _project(origin + ax * L, t_cam, R_cam, fx, w, h)
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
