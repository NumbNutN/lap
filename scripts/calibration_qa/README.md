# Wrist camera calibration QA

Tools for **validating per-episode wrist camera extrinsics** in the DROID
raw release (`/data/datasets/droid_data_raw/1.0.1/`). Used to filter out
episodes whose `T_world_wrist` or `T_ee_wrist` is mis-calibrated before
they enter the annotation / training pipeline.

## Scripts

### `project_ee_to_wrist.py`
Projects the EE origin + the 3 EE axes (x/y/z) onto the **wrist camera
image**. The EE +z axis is the gripper approach direction; if calibration
is correct, the projected origin lands near the gripper hardware visible
in the wrist view, and the +z arrow points along the gripper.

Run on the local sample dataset:
```
python project_ee_to_wrist.py    # outputs to /tmp/wrist_validate/v2_out/
```

### `project_wrist_to_ext.py`
Projects the **wrist camera frame** (origin + 3 axes) onto the
**external camera video** (ext1 or ext2 — pick whichever has the robot
in its field of view; ext1 is often the workspace-only view).

If calibration is correct, the projected wrist origin lands on the
visible wrist mount (Robotiq + Zed-Mini bracket) in the external video,
and the wrist +z axis points in the direction the wrist camera is
actually looking.

```
python project_wrist_to_ext.py   # outputs mp4s to /tmp/wrist_validate/videos/
```

## Convention summary (verified empirically)

- **`wrist_cam_extrinsics[t]` in h5** = `T_world_cam` (convention A):
  - `t` = camera position in world
  - `R = rv2m(rotvec)` maps camera-frame vectors to world
  - To project: `p_cam = R.T @ (p_world - t)`
- **`<wrist>_left_gripper_offset[t]`** = `T_ee_wrist` (constant per episode,
  hand-eye calibration). Compose with EE pose to get T_world_wrist for any
  frame.
- **Camera frame convention** (Zed): +x = image right, +y = image up
  (REP-103, NOT OpenCV +y down), +z = optical axis into scene. Pixel
  projection: `u = fx * px/pz + cx`, `v = -fy * py/pz + cy`.
- **Default Zed-Mini intrinsics** (wide HD720): fx ≈ 380, cx=640, cy=360.
- **Default Zed 2 intrinsics** (HD720): fx ≈ 600–700, cx=640, cy=360.
  (Both are approximate — exact intrinsics are inside SVO containers,
  not in the h5/json metadata.)
- **EE reference point** = panda_hand link (not gripper tip). Gripper
  tip is approximately at `ee_xyz + R_world_ee @ [0, 0, 0.10]` (10 cm
  along EE +z).

## Filtering policy

If at scale we find episodes whose projected EE / wrist position is too
far from the visible gripper / wrist mount (e.g. > 100 px in 1280×720),
those episodes should be flagged with low calibration confidence and
either:
- excluded from training,
- annotated with the fallback empirical EE-frame instead of the
  calibrated wrist frame (see `pose_utils.pose_delta` fallback path).
