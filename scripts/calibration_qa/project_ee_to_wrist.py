"""v2: convention A + y-flip + Zed-Mini wide-angle intrinsics."""
import sys, os, math, json, glob
import numpy as np, h5py, cv2

K = np.array([[380, 0, 640], [0, 380, 360], [0, 0, 1]], dtype=np.float64)
AXIS_LEN = 0.05  # 5 cm

def rv2m(rv):
    a = float(np.linalg.norm(rv));
    if a < 1e-9: return np.eye(3)
    k = rv/a; KK = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3) + math.sin(a)*KK + (1-math.cos(a))*(KK@KK)

def world_to_pixel(p_world, T_world_cam):
    R = rv2m(T_world_cam[3:])
    p_cam = R.T @ (np.asarray(p_world) - T_world_cam[:3])
    if p_cam[2] <= 0: return None
    u = K[0,0] * p_cam[0]/p_cam[2] + K[0,2]
    v = -K[1,1] * p_cam[1]/p_cam[2] + K[1,2]   # NOTE: -y (image up = camera +y)
    return int(u), int(v)

def draw_axes(img, ee_pose, T_world_cam, gripper_tip_offset_m=0.10):
    p_ee = ee_pose[:3]
    R_we = rv2m(ee_pose[3:])
    # Project EE origin AND gripper-tip (EE +z extended)
    out = img.copy()
    o_pt = world_to_pixel(p_ee, T_world_cam)
    g_world = p_ee + R_we @ np.array([0,0,gripper_tip_offset_m])
    g_pt = world_to_pixel(g_world, T_world_cam)
    if o_pt:
        cv2.circle(out, o_pt, 8, (0,255,255), -1)  # yellow EE origin
        cv2.circle(out, o_pt, 8, (0,0,0), 2)
        cv2.putText(out, "EE", (o_pt[0]+10, o_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    if g_pt:
        cv2.circle(out, g_pt, 10, (255,0,255), -1)  # magenta gripper tip
        cv2.circle(out, g_pt, 10, (0,0,0), 2)
        cv2.putText(out, "tip", (g_pt[0]+12, g_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
    if o_pt and g_pt:
        cv2.arrowedLine(out, o_pt, g_pt, (255,255,255), 2)
    # Three EE axes at EE origin
    colors = {'x':(0,0,255), 'y':(0,255,0), 'z':(255,0,0)}  # BGR
    if o_pt:
        for ax_i, ax_name in enumerate(['x','y','z']):
            ax_unit = np.zeros(3); ax_unit[ax_i] = AXIS_LEN
            tip = p_ee + R_we @ ax_unit
            t_pt = world_to_pixel(tip, T_world_cam)
            if t_pt:
                cv2.arrowedLine(out, o_pt, t_pt, colors[ax_name], 3, tipLength=0.3)
                cv2.putText(out, ax_name.upper(), (t_pt[0]-5, t_pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[ax_name], 2)
    return out

if __name__ == "__main__":
    ed = "/home/numbnut/datasets/droid_raw/1.0.1/AUTOLab/failure/2023-07-07/Fri_Jul__7_09:45:39_2023/"
    wrist_serial = json.load(open(glob.glob(ed + "metadata_*.json")[0]))["wrist_cam_serial"]
    mp4 = os.path.join(ed, "recordings/MP4", f"{wrist_serial}.mp4")
    with h5py.File(ed + "trajectory.h5", "r") as f:
        ee_pose_all = f["observation/robot_state/cartesian_position"][:]
        Twc_all = f[f"observation/camera_extrinsics/{wrist_serial}_left"][:]
    cap = cv2.VideoCapture(mp4)
    T = min(ee_pose_all.shape[0], int(cap.get(7)))
    out_dir = "/tmp/wrist_validate/v2_out"
    os.makedirs(out_dir, exist_ok=True)
    for fi in [0, 50, 100, 150, 200]:
        if fi >= T: continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok: continue
        ov = draw_axes(img, ee_pose_all[fi], Twc_all[fi])
        cv2.putText(ov, f"f={fi}  fx={int(K[0,0])}  conv=A+yFlip", (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(ov, "yellow=EE  magenta=tip(10cm)  RGB=EE x/y/z axes", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(f"{out_dir}/v2_f{fi:04d}.jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
    cap.release()
    print(f"wrote to {out_dir}")
