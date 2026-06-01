"""Screen DROID raw episodes for wrist camera calibration quality.

For each episode under ``--root``, we score the calibration confidence
by projecting the gripper tip (EE + 10 cm along EE +z) onto the wrist
image at several sampled frames. A well-calibrated episode has the
projected tip landing close to image center (where the gripper jaws
actually are in the wrist view).

Output: CSV with one row per episode (used by `extract_raw.py
--ep-whitelist`).

Columns:
  ep_id                      relative h5 path
  has_calibration            T_ee_wrist exists in h5?
  has_wrist_mp4              wrist video file present?
  n_frames_sampled           how many frames we tested
  mean_tip_dist_to_center_px mean L2 distance (pixel) from projected
                             tip to image center; smaller = better
  frac_in_image              fraction of sampled frames where tip
                             projected inside [0,W]×[0,H]
  fwd_depth_mean_m           mean z-depth of projected tip (m); should
                             be positive and small (wrist is right next
                             to gripper)
  classification             good | marginal | bad | no_calibration
  reason                     short string explaining classification

Thresholds (default; tunable via --good/--marginal):
  good:     dist<150 AND frac_in_image>=0.8  AND fwd_depth_mean<0.4
  marginal: dist<250 OR  frac_in_image>=0.5
  bad:      everything else (or missing data)

Usage:
    python screen_episodes.py \\
        --root ~/datasets/droid_raw/1.0.1 \\
        --out /tmp/calib_audit.csv \\
        [--labs AUTOLab]
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, glob
from pathlib import Path
import numpy as np
import h5py
import cv2


# ----- camera projection ----------------------------------------------------

def rv2m(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-9:
        return np.eye(3)
    k = rv / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def project_world_to_wrist_pixel(
    p_world: np.ndarray,
    T_world_wrist: np.ndarray,
    fx: float = 380.0, fy: float = 380.0,
    cx: float = 640.0, cy: float = 360.0,
    W: int = 1280, H: int = 720,
):
    """Project world point to wrist-image pixel under convention A + y-flip.

    Returns (u, v, depth_z, in_image_bool). Z<=0 ⇒ behind camera ⇒
    in_image=False, u/v will be None.
    """
    R = rv2m(T_world_wrist[3:])
    p_cam = R.T @ (np.asarray(p_world) - T_world_wrist[:3])
    z = float(p_cam[2])
    if z <= 0.01:
        return None, None, z, False
    u = fx * p_cam[0] / z + cx
    v = -fy * p_cam[1] / z + cy
    in_image = (0 <= u <= W) and (0 <= v <= H)
    return float(u), float(v), z, bool(in_image)


# ----- per-episode screening ------------------------------------------------

def screen_episode(ep_dir: Path, n_sample_frames: int = 8, tip_offset_m: float = 0.10) -> dict:
    """Score one episode. Returns a row dict (defaults filled for missing data)."""
    row = {
        "ep_id": "",
        "has_calibration": False,
        "has_wrist_mp4": False,
        "n_frames_sampled": 0,
        "mean_tip_dist_to_center_px": float("nan"),
        "frac_in_image": float("nan"),
        "fwd_depth_mean_m": float("nan"),
        "classification": "bad",
        "reason": "",
    }

    h5p = ep_dir / "trajectory.h5"
    if not h5p.exists():
        row["reason"] = "no_trajectory.h5"
        return row

    metas = list(ep_dir.glob("metadata_*.json"))
    if not metas:
        row["reason"] = "no_metadata"
        return row
    md = json.load(open(metas[0]))
    wrist_serial = md.get("wrist_cam_serial", "")
    if not wrist_serial:
        row["reason"] = "metadata_missing_wrist_serial"
        return row

    wrist_mp4 = ep_dir / "recordings" / "MP4" / f"{wrist_serial}.mp4"
    row["has_wrist_mp4"] = wrist_mp4.exists()

    with h5py.File(h5p, "r") as f:
        if "observation/robot_state/cartesian_position" not in f:
            row["reason"] = "h5_no_cartesian_position"
            return row
        offset_key = f"observation/camera_extrinsics/{wrist_serial}_left_gripper_offset"
        extrinsics_key = f"observation/camera_extrinsics/{wrist_serial}_left"
        row["has_calibration"] = (offset_key in f) and (extrinsics_key in f)
        if not row["has_calibration"]:
            row["classification"] = "no_calibration"
            row["reason"] = "h5_missing_extrinsics"
            return row
        ee_pose = f["observation/robot_state/cartesian_position"][:]
        T_world_wrist = f[extrinsics_key][:]

    T = ee_pose.shape[0]
    if T < n_sample_frames:
        row["reason"] = f"too_few_frames T={T}"
        return row

    sample_idx = np.linspace(0, T - 1, n_sample_frames, dtype=int)
    dists, depths, in_image_flags = [], [], []
    for t in sample_idx:
        ee_t = ee_pose[t]
        R_we = rv2m(ee_t[3:])
        tip_world = ee_t[:3] + R_we @ np.array([0, 0, tip_offset_m])
        u, v, z, in_img = project_world_to_wrist_pixel(tip_world, T_world_wrist[t])
        depths.append(z)
        in_image_flags.append(in_img)
        if u is not None and v is not None:
            d = math.sqrt((u - 640) ** 2 + (v - 360) ** 2)
            dists.append(d)

    row["n_frames_sampled"] = int(n_sample_frames)
    row["mean_tip_dist_to_center_px"] = float(np.mean(dists)) if dists else float("nan")
    row["frac_in_image"] = float(np.mean(in_image_flags))
    row["fwd_depth_mean_m"] = float(np.mean(depths))
    return row


def classify(row: dict, good_dist: float, marginal_dist: float) -> tuple[str, str]:
    if not row["has_calibration"]:
        return "no_calibration", "missing T_ee_wrist or T_world_wrist"
    if not row["has_wrist_mp4"]:
        return "bad", "missing wrist MP4"
    d = row["mean_tip_dist_to_center_px"]
    fi = row["frac_in_image"]
    dep = row["fwd_depth_mean_m"]
    if math.isnan(d):
        return "bad", "projection failed on all sampled frames"
    if d < good_dist and fi >= 0.8 and dep < 0.4:
        return "good", "tip near center on most sampled frames"
    if d < marginal_dist or fi >= 0.5:
        return "marginal", "projection off-center but partially usable"
    return "bad", f"tip projects far from image (mean dist={d:.0f}px, frac_in={fi:.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--labs", nargs="*", default=None)
    ap.add_argument("--include-failure", action="store_true")
    ap.add_argument("--n-frames-sampled", type=int, default=8)
    ap.add_argument("--good-dist-px", type=float, default=150.0)
    ap.add_argument("--marginal-dist-px", type=float, default=300.0)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    root = Path(os.path.expanduser(args.root))
    if not root.exists():
        sys.exit(f"root not found: {root}")

    labs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.labs:
        lab_set = set(args.labs)
        labs = [p for p in labs if p.name in lab_set]
    outcomes = ["success"] if not args.include_failure else ["success", "failure"]

    rows = []
    n = 0
    for lab in labs:
        for outcome in outcomes:
            outcome_dir = lab / outcome
            if not outcome_dir.exists():
                continue
            for date_dir in sorted(outcome_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                for ep_dir in sorted(date_dir.iterdir()):
                    if not ep_dir.is_dir():
                        continue
                    if args.max_episodes is not None and n >= args.max_episodes:
                        break
                    n += 1
                    row = screen_episode(ep_dir, args.n_frames_sampled)
                    row["ep_id"] = str(ep_dir.relative_to(root) / "trajectory.h5")
                    cls, reason = classify(row, args.good_dist_px, args.marginal_dist_px)
                    row["classification"] = cls
                    row["reason"] = reason
                    rows.append(row)
                    if n % 10 == 0:
                        print(f"  scored {n} eps...")

    # Write CSV
    cols = ["ep_id", "classification", "reason", "has_calibration",
            "has_wrist_mp4", "n_frames_sampled",
            "mean_tip_dist_to_center_px", "frac_in_image",
            "fwd_depth_mean_m"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # Summary
    from collections import Counter
    counts = Counter(r["classification"] for r in rows)
    print(f"\nTotal scanned: {len(rows)}")
    for k, v in sorted(counts.items()):
        pct = 100 * v / len(rows) if rows else 0
        print(f"  {k:<16}: {v:>4}  ({pct:.1f}%)")
    print(f"\nCSV written to {args.out}")


if __name__ == "__main__":
    main()
