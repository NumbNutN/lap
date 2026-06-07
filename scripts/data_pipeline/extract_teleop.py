#!/usr/bin/env python3
"""Extract self-contained SSAA-v3 episode dirs from RoboTwin sim teleop hdf5.

Disguises teleop data as the DROID contract so the existing tools / viewer /
annotation / audit pipeline works UNCHANGED. Per episode it writes a
self-contained dir (named by uuid) holding:

  trajectory.h5   observation/robot_state/cartesian_position  (T,6) = [x,y,z, euler 'xyz']
                  observation/robot_state/gripper_position     (T,)  DROID semantics 0=open,1=closed
  meta.json       DROID fields + source="teleop" + ext_video/wrist_video + keyframes
  ext.mp4         the bird_eye (static external) camera
  wrist.mp4       the left_camera (wrist) camera
  kfNN_fFFFF.jpg / kfNN_fFFFF_wrist.jpg   keyframe stills

Signals (NO forward kinematics needed):
  - endpose/<arm>_endpose  = [x,y,z, qw,qx,qy,qz] (sapien wxyz) → cartesian euler
  - endpose/<arm>_gripper  teleop 0=closed,1=open  → flipped to DROID 0=open,1=closed
  - cameras already rendered to mp4; extrinsic/intrinsic also live in the h5

The euler 'xyz' produced here matches tools.py `_euler_xyz_to_matrix`
(== scipy Rotation.from_euler('xyz')), so get_pose_delta works verbatim.
"""
from __future__ import annotations
import argparse, glob, json, os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)
from annotate_droid.keyframe import detect_keyframes  # noqa: E402

EXT_CAM = "bird_eye"        # static external (top-down-ish) view
WRIST_CAM = "left_camera"   # wrist camera (moves with the arm)
FPS = 6.0                   # teleop_playground save_freq=5 → 6 fps
NEAR_RADIUS = 2
INTERACTION = {"grasp", "release", "retry"}


def _active_arm(f) -> str:
    """The arm whose gripper actually varies (the one performing the task)."""
    lg = f["endpose/left_gripper"][:]
    rg = f["endpose/right_gripper"][:]
    lr = float(lg.max() - lg.min())
    rr = float(rg.max() - rg.min())
    return "right" if rr >= lr else "left"


def _endpose_to_cartesian(endpose: np.ndarray) -> np.ndarray:
    """(T,7) [x,y,z, qw,qx,qy,qz] (sapien wxyz) → (T,6) [x,y,z, euler 'xyz']."""
    from scipy.spatial.transform import Rotation
    pos = endpose[:, :3]
    q = endpose[:, 3:7]                                   # wxyz
    quat_xyzw = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    eul = Rotation.from_quat(quat_xyzw).as_euler("xyz")   # matches _euler_xyz_to_matrix
    return np.column_stack([pos, eul]).astype(np.float64)


def _mp4_frame(cap, idx: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, img = cap.read()
    return img if ok else None


def extract_episode(h5_path: str, video_dir: str, ep_name: str,
                    out_root: str, task: str = "") -> tuple[str, int]:
    import cv2, h5py
    with h5py.File(h5_path, "r") as f:
        arm = _active_arm(f)
        endpose = f[f"endpose/{arm}_endpose"][:]
        grip_raw = f[f"endpose/{arm}_gripper"][:].astype(np.float64)
    n = int(endpose.shape[0])
    cart = _endpose_to_cartesian(endpose)
    grip_droid = 1.0 - grip_raw                          # teleop→DROID gripper semantics

    # self-consistency: our euler must reproduce the quaternion rotation under
    # the exact convention tools.py uses (fail loud if scipy/convention drifts)
    from data_pipeline.tools import _euler_xyz_to_matrix
    from scipy.spatial.transform import Rotation
    for i in (0, n // 2, n - 1):
        R_q = Rotation.from_quat([endpose[i, 4], endpose[i, 5], endpose[i, 6],
                                  endpose[i, 3]]).as_matrix()
        R_e = _euler_xyz_to_matrix(cart[i, 3:])
        assert np.allclose(R_q, R_e, atol=1e-6), f"euler roundtrip mismatch @{i}"

    kfs = detect_keyframes(gripper_width=grip_droid, ee_pos=cart[:, :3], fps=FPS)

    uuid = f"teleop_playground_{ep_name}"
    ep_dir = os.path.join(out_root, uuid)
    os.makedirs(ep_dir, exist_ok=True)

    with h5py.File(os.path.join(ep_dir, "trajectory.h5"), "w") as g:
        g.create_dataset("observation/robot_state/cartesian_position", data=cart)
        g.create_dataset("observation/robot_state/gripper_position", data=grip_droid)

    ext_src = os.path.join(video_dir, f"{ep_name}_{EXT_CAM}.mp4")
    wrist_src = os.path.join(video_dir, f"{ep_name}_{WRIST_CAM}.mp4")
    for src, dst in ((ext_src, "ext.mp4"), (wrist_src, "wrist.mp4")):
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing camera mp4: {src}")
        shutil.copy(src, os.path.join(ep_dir, dst))

    # near_interaction within ±NEAR_RADIUS keyframes of a grasp/release/retry
    inter = [i for i, k in enumerate(kfs) if k.type in INTERACTION]
    near = [any(abs(i - j) <= NEAR_RADIUS for j in inter) for i in range(len(kfs))]

    cap_e = cv2.VideoCapture(ext_src)
    cap_w = cv2.VideoCapture(wrist_src)
    kf_meta = []
    for i, k in enumerate(kfs):
        fi = int(k.t)
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
        "episode_id": f"teleop_playground/{ep_name}",
        "uuid": uuid,
        "source": "teleop",
        "outcome": "teleop",                 # sim teleop has no success label
        "task_instruction": task,
        "n_frames": n,
        "fps": FPS,
        "calibrated_wrist_frame": False,     # no hand-eye T_ee_wrist → delta_ee=None
        "ext_video": "ext.mp4",
        "wrist_video": "wrist.mp4",
        "ext_camera": EXT_CAM,
        "wrist_camera": WRIST_CAM,
        "active_arm": arm,
        "keyframes": kf_meta,
    }
    json.dump(meta, open(os.path.join(ep_dir, "meta.json"), "w"),
              indent=2, ensure_ascii=False)
    return ep_dir, len(kf_meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default="data/teleop_playground/teleop_playground",
                    help="dir holding data/<ep>.hdf5, video/, cot_annotations/")
    ap.add_argument("--out", default="policy/lap/local_data/teleop_eps")
    ap.add_argument("--episodes", nargs="*",
                    help="episode names to extract (e.g. episode0); default all")
    args = ap.parse_args()

    h5dir = os.path.join(args.data_root, "data")
    viddir = os.path.join(args.data_root, "video")
    cotdir = os.path.join(args.data_root, "cot_annotations")
    os.makedirs(args.out, exist_ok=True)

    h5s = sorted(glob.glob(os.path.join(h5dir, "*.hdf5")))
    if not h5s:
        sys.exit(f"no .hdf5 under {h5dir}")
    done = 0
    for h5 in h5s:
        ep_name = os.path.splitext(os.path.basename(h5))[0]   # episode0
        if args.episodes and ep_name not in args.episodes:
            continue
        task = ""
        cotp = os.path.join(cotdir, ep_name + ".json")
        if os.path.exists(cotp):
            try:
                task = json.load(open(cotp)).get("task_instruction", "")
            except Exception:
                pass
        ep_dir, nkf = extract_episode(h5, viddir, ep_name, args.out, task)
        print(f"  {ep_name} → {ep_dir}  ({nkf} keyframes, task: {task[:50]!r})")
        done += 1
    print(f"extracted {done} episode(s) → {args.out}")


if __name__ == "__main__":
    main()
