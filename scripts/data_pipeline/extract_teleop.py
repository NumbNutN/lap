#!/usr/bin/env python3
"""Extract self-contained SSAA-v3 episode dirs from RoboTwin sim teleop hdf5,
optionally SPLITTING a long-horizon episode into short segments (one per
sub-task, e.g. "one cube placed") so each is annotated independently.

Disguises teleop data as the DROID contract → existing tools / viewer /
annotation / audit pipeline runs UNCHANGED. Each segment becomes a
self-contained dir (named by uuid) holding:

  trajectory.h5   observation/robot_state/cartesian_position (T,6) [x,y,z,euler 'xyz']
                  observation/robot_state/gripper_position    (T,)  DROID 0=open,1=closed
  meta.json       DROID fields + source="teleop" + segment provenance + keyframes
  ext.mp4         bird_eye (static external)  — cropped to the segment
  wrist.mp4       left_camera (wrist)         — cropped to the segment
  kfNN_fFFFF.jpg / kfNN_fFFFF_wrist.jpg   keyframe stills (segment-local 0-based)

Boundaries (A: human-given) come from --boundaries / --segments-file; later a
viewer "set segment boundary" button (C) writes the same file. With no
boundaries an episode stays one segment.

Signals need NO forward kinematics: endpose pos+quat(wxyz) + gripper from h5;
cameras already rendered to mp4. euler 'xyz' matches tools._euler_xyz_to_matrix.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
LAP_ROOT = os.path.dirname(SCRIPTS)                       # policy/lap
REPO = os.path.dirname(os.path.dirname(LAP_ROOT))         # RoboTwin (repo root)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from annotate_droid.keyframe import detect_keyframes  # noqa: E402

EXT_CAM = "bird_eye"
WRIST_CAM = "left_camera"
FPS = 6.0
NEAR_RADIUS = 2
INTERACTION = {"grasp", "release", "retry"}
MIN_KF_GAP = 3          # frames @6fps (~0.5s): merge micro-chunks closer than this
_KF_PRIO = {"begin": 5, "end": 5, "grasp": 4, "release": 4, "retry": 4,
            "motion": 2, "filler": 1}


def _merge_close_keyframes(kfs, min_gap):
    """Global NMS: drop keyframes closer than min_gap to the kept one, keeping
    the higher-priority type (begin/end/grasp/release > motion > filler)."""
    if len(kfs) <= 2:
        return kfs
    kept = [kfs[0]]
    for k in kfs[1:]:
        if k.type == "end":                     # always keep the end bracket
            kept.append(k)
            continue
        prev = kept[-1]
        if (k.t - prev.t) < min_gap and prev.type != "begin":
            if _KF_PRIO.get(k.type, 0) > _KF_PRIO.get(prev.type, 0):
                kept[-1] = k                     # replace lower-priority neighbour
            # else: drop k (keep prev)
        else:
            kept.append(k)
    return kept


def _active_arm(f) -> str:
    lg = f["endpose/left_gripper"][:]
    rg = f["endpose/right_gripper"][:]
    return "right" if (rg.max() - rg.min()) >= (lg.max() - lg.min()) else "left"


def _endpose_to_cartesian(endpose: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    pos = endpose[:, :3]
    q = endpose[:, 3:7]                                   # wxyz
    quat_xyzw = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    eul = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    return np.column_stack([pos, eul]).astype(np.float64)


def _crop_mp4(src: str, dst: str, s: int, e: int) -> None:
    """Write frames [s, e) of src to dst (mp4v)."""
    import cv2
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    for _ in range(s, e):
        ok, fr = cap.read()
        if not ok:
            break
        out.write(fr)
    out.release()
    cap.release()


def _mp4_frame(cap, idx: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, img = cap.read()
    return img if ok else None


def _ranges(n: int, boundaries: list[int], min_seg: int = 6) -> list[tuple[int, int]]:
    """Boundaries → [start,end) ranges, dropping cuts that would make a segment
    shorter than min_seg frames (e.g. a boundary placed ≈ the episode end)."""
    pts = [0]
    for b in sorted({int(x) for x in boundaries if 0 < int(x) < n}):
        if b - pts[-1] >= min_seg:
            pts.append(b)
    if len(pts) > 1 and n - pts[-1] < min_seg:
        pts.pop()                       # merge a too-short tail into previous seg
    pts.append(n)
    return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


def extract(h5_path: str, video_dir: str, ep_name: str, out_root: str,
            task: str, boundaries: list[int]) -> list[tuple[str, int]]:
    import cv2, h5py
    with h5py.File(h5_path, "r") as f:
        arm = _active_arm(f)
        endpose = f[f"endpose/{arm}_endpose"][:]
        grip_raw = f[f"endpose/{arm}_gripper"][:].astype(np.float64)
    n = int(endpose.shape[0])
    cart = _endpose_to_cartesian(endpose)
    grip_droid = 1.0 - grip_raw

    from data_pipeline.tools import _euler_xyz_to_matrix
    from scipy.spatial.transform import Rotation
    for i in (0, n // 2, n - 1):
        R_q = Rotation.from_quat([endpose[i, 4], endpose[i, 5], endpose[i, 6],
                                  endpose[i, 3]]).as_matrix()
        assert np.allclose(R_q, _euler_xyz_to_matrix(cart[i, 3:]), atol=1e-6)

    ext_src = os.path.join(video_dir, f"{ep_name}_{EXT_CAM}.mp4")
    wrist_src = os.path.join(video_dir, f"{ep_name}_{WRIST_CAM}.mp4")
    for p in (ext_src, wrist_src):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing camera mp4: {p}")

    segs = _ranges(n, boundaries)
    multi = len(segs) > 1
    results = []
    for si, (s, e) in enumerate(segs):
        seg_cart = cart[s:e]
        seg_grip = grip_droid[s:e]
        seg_n = e - s
        kfs = detect_keyframes(gripper_width=seg_grip, ee_pos=seg_cart[:, :3], fps=FPS)
        # A segment cut mid-action can leave frame 0 / last tagged grasp/motion;
        # SSAA-v3 needs begin/end S-only brackets, so force them.
        if kfs and kfs[0].t == 0:
            kfs[0].type = "begin"
        if kfs and kfs[-1].t == seg_n - 1:
            kfs[-1].type = "end"
        kfs = _merge_close_keyframes(kfs, MIN_KF_GAP)   # drop 1-frame micro-chunks

        uuid = f"teleop_playground_{ep_name}" + (f"_seg{si:02d}" if multi else "")
        ep_dir = os.path.join(out_root, uuid)
        os.makedirs(ep_dir, exist_ok=True)

        with h5py.File(os.path.join(ep_dir, "trajectory.h5"), "w") as g:
            g.create_dataset("observation/robot_state/cartesian_position", data=seg_cart)
            g.create_dataset("observation/robot_state/gripper_position", data=seg_grip)

        _crop_mp4(ext_src, os.path.join(ep_dir, "ext.mp4"), s, e)
        _crop_mp4(wrist_src, os.path.join(ep_dir, "wrist.mp4"), s, e)

        inter = [i for i, k in enumerate(kfs) if k.type in INTERACTION]
        near = [any(abs(i - j) <= NEAR_RADIUS for j in inter) for i in range(len(kfs))]

        cap_e = cv2.VideoCapture(os.path.join(ep_dir, "ext.mp4"))
        cap_w = cv2.VideoCapture(os.path.join(ep_dir, "wrist.mp4"))
        kf_meta = []
        for i, k in enumerate(kfs):
            fi = int(k.t)                       # segment-local 0-based frame
            fn_e = f"kf{i:02d}_f{fi:04d}.jpg"
            fn_w = f"kf{i:02d}_f{fi:04d}_wrist.jpg"
            ie, iw = _mp4_frame(cap_e, fi), _mp4_frame(cap_w, fi)
            if ie is not None:
                cv2.imwrite(os.path.join(ep_dir, fn_e), ie, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if iw is not None:
                cv2.imwrite(os.path.join(ep_dir, fn_w), iw, [cv2.IMWRITE_JPEG_QUALITY, 85])
            kf_meta.append({
                "idx": i, "frame_idx": fi, "type": k.type,
                "gripper_state": k.gripper_state,
                "near_interaction": bool(near[i]), "interaction_context": None,
                "image_file": fn_e, "wrist_image_file": fn_w,
            })
        cap_e.release()
        cap_w.release()

        meta = {
            "episode_id": f"teleop_playground/{uuid}",
            "uuid": uuid,
            "source": "teleop",
            "outcome": "teleop",
            "task_instruction": task,
            "n_frames": seg_n,
            "fps": FPS,
            "calibrated_wrist_frame": False,
            "ext_video": "ext.mp4",
            "wrist_video": "wrist.mp4",
            "ext_camera": EXT_CAM,
            "wrist_camera": WRIST_CAM,
            "active_arm": arm,
            # segment provenance (for ordering / re-stitching / prior-context hints)
            "source_episode": ep_name,
            "segment_idx": si,
            "total_segments": len(segs),
            "orig_frame_range": [s, e],
            "keyframes": kf_meta,
        }
        json.dump(meta, open(os.path.join(ep_dir, "meta.json"), "w"),
                  indent=2, ensure_ascii=False)
        results.append((ep_dir, len(kf_meta)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default=os.path.join(REPO, "data/teleop_playground/teleop_playground"))
    ap.add_argument("--out", default=os.path.join(LAP_ROOT, "local_data/teleop_eps"))
    ap.add_argument("--episodes", nargs="*", help="episode names; default all")
    ap.add_argument("--boundaries", help="comma frames to split at (single episode), e.g. 350,700")
    ap.add_argument("--segments-file", help="json {episode_name: [boundary_frames]}")
    args = ap.parse_args()

    h5dir = os.path.join(args.data_root, "data")
    viddir = os.path.join(args.data_root, "video")
    cotdir = os.path.join(args.data_root, "cot_annotations")
    os.makedirs(args.out, exist_ok=True)

    segfile = {}
    if args.segments_file and os.path.exists(args.segments_file):
        segfile = json.load(open(args.segments_file))
    cli_bounds = [int(x) for x in args.boundaries.split(",")] if args.boundaries else []

    h5s = sorted(glob.glob(os.path.join(h5dir, "*.hdf5")))
    if not h5s:
        sys.exit(f"no .hdf5 under {h5dir}")
    n_segs = 0
    for h5 in h5s:
        ep_name = os.path.splitext(os.path.basename(h5))[0]
        if args.episodes and ep_name not in args.episodes:
            continue
        bounds = segfile.get(ep_name, cli_bounds)
        task = ""
        cotp = os.path.join(cotdir, ep_name + ".json")
        if os.path.exists(cotp):
            try:
                task = json.load(open(cotp)).get("task_instruction", "")
            except Exception:
                pass
        for ep_dir, nkf in extract(h5, viddir, ep_name, args.out, task, bounds):
            print(f"  {os.path.basename(ep_dir)}  ({nkf} kf)")
            n_segs += 1
    print(f"extracted {n_segs} segment(s) → {args.out}")


if __name__ == "__main__":
    main()
