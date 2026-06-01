"""Project the wrist camera frame (origin + 3 axes) onto the external camera video.

Validation logic:
  - wrist camera is rigidly attached to the gripper
  - external cameras (Zed 2) see the gripper and the wrist camera mount
  - if T_world_wrist and T_world_ext are both correctly calibrated, the projected
    wrist origin should land EXACTLY on the visible wrist camera body in the
    external view, and the wrist +z arrow should point in the gripper's approach
    direction (= where the wrist camera is actually looking).

Output: per-episode mp4 file with the overlay.
"""
from __future__ import annotations
import os, sys, math, json, glob
import numpy as np, h5py, cv2

# Zed 2 typical intrinsics at HD720 (ext1/ext2); Zed-Mini fx≈380 only matters
# when WE are the wrist camera. Here we project to ext, so use Zed 2 wide.
K_EXT = np.array([[700, 0, 640], [0, 700, 360], [0, 0, 1]], dtype=np.float64)

AXIS_LEN_M = 0.08  # 8 cm axes (big enough to see)

def rv2m(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-9: return np.eye(3)
    k = rv/a; KK = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(a)*KK + (1-math.cos(a))*(KK@KK)

def world_to_pixel(p_world, T_world_cam, K=K_EXT):
    """Convention A + y-flip (camera +y is image up)."""
    R = rv2m(T_world_cam[3:])
    p_cam = R.T @ (np.asarray(p_world) - T_world_cam[:3])
    if p_cam[2] <= 0.02: return None  # behind / too close
    u = K[0,0] * p_cam[0]/p_cam[2] + K[0,2]
    v = -K[1,1] * p_cam[1]/p_cam[2] + K[1,2]
    if not (-100 <= u <= 1380 and -100 <= v <= 820):  # huge margin
        return None
    return int(u), int(v)

def draw_wrist_frame_on_ext(img, T_world_wrist_t, T_world_ext_t):
    """Render wrist origin + 3 axes (R/G/B = X/Y/Z) on the external image."""
    out = img.copy()
    wrist_origin = T_world_wrist_t[:3]
    R_world_wrist = rv2m(T_world_wrist_t[3:])
    o_pt = world_to_pixel(wrist_origin, T_world_ext_t)
    if o_pt is None: return out
    # Three wrist axes' tips, all in world coords
    axes = {
        'x': wrist_origin + R_world_wrist @ np.array([AXIS_LEN_M, 0, 0]),
        'y': wrist_origin + R_world_wrist @ np.array([0, AXIS_LEN_M, 0]),
        'z': wrist_origin + R_world_wrist @ np.array([0, 0, AXIS_LEN_M]),
    }
    cv2.circle(out, o_pt, 10, (0, 255, 255), -1)   # yellow wrist origin
    cv2.circle(out, o_pt, 10, (0, 0, 0), 2)
    cv2.putText(out, "wrist", (o_pt[0]+12, o_pt[1]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    colors = {'x': (0, 0, 255), 'y': (0, 255, 0), 'z': (255, 0, 0)}  # BGR
    labels = {'x': 'X (img right)', 'y': 'Y (img up)', 'z': 'Z (forward)'}
    for ax_name, tip_world in axes.items():
        t_pt = world_to_pixel(tip_world, T_world_ext_t)
        if t_pt is None: continue
        cv2.arrowedLine(out, o_pt, t_pt, colors[ax_name], 3, tipLength=0.25)
        cv2.putText(out, ax_name.upper(), t_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[ax_name], 2)
    # Legend
    for i, (ax, lbl) in enumerate(labels.items()):
        cv2.putText(out, lbl, (15, 60 + 22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[ax], 2)
    return out

def make_video_for_episode(ep_dir, out_path, ext_choice="ext2"):
    md_path = glob.glob(os.path.join(ep_dir, "metadata_*.json"))[0]
    md = json.load(open(md_path))
    wrist_serial = md["wrist_cam_serial"]
    ext_serial = md["ext2_cam_serial"] if ext_choice == "ext2" else md["ext2_cam_serial"]
    ext_mp4 = os.path.join(ep_dir, "recordings/MP4", f"{ext_serial}.mp4")
    h5p = os.path.join(ep_dir, "trajectory.h5")
    if not (os.path.exists(ext_mp4) and os.path.exists(h5p)):
        print(f"  skip {ep_dir}: missing files"); return False

    with h5py.File(h5p, "r") as f:
        cam_x = f["observation/camera_extrinsics"]
        wrist_key = f"{wrist_serial}_left"
        ext_key = f"{ext_serial}_left"
        if wrist_key not in cam_x or ext_key not in cam_x:
            print(f"  skip {ep_dir}: missing extrinsics keys")
            return False
        T_world_wrist = cam_x[wrist_key][:]
        T_world_ext = cam_x[ext_key][:]
        T_h5 = T_world_wrist.shape[0]

    cap = cv2.VideoCapture(ext_mp4)
    T_mp4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = min(T_h5, T_mp4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 15.0, (w, h))  # force 15fps for display

    print(f"  rendering {T} frames → {out_path}")
    for t in range(T):
        ok, img = cap.read()
        if not ok: break
        ov = draw_wrist_frame_on_ext(img, T_world_wrist[t], T_world_ext[t])
        cv2.putText(ov, f"f={t}/{T}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(ov)
    cap.release(); writer.release()
    return True

if __name__ == "__main__":
    ROOT = os.path.expanduser("~/datasets/droid_raw/1.0.1")
    OUT = "/tmp/wrist_validate/videos"
    os.makedirs(OUT, exist_ok=True)

    eps = sorted(glob.glob(f"{ROOT}/*/*/*/*/"))
    n_ok = 0
    for ed in eps:
        if n_ok >= 3: break
        name = "_".join(ed.rstrip("/").split("/")[-3:])
        out_path = os.path.join(OUT, f"{name}.mp4")
        if make_video_for_episode(ed, out_path):
            print(f"  ✓ {name}")
            n_ok += 1
    print(f"\nWrote {n_ok} videos to {OUT}")
