"""Pose / rotation helpers for VLM annotation prompts.

DROID's ``observation/cartesian_position`` is a (T, 6) array of
``[x, y, z, rx, ry, rz]`` where ``(rx, ry, rz)`` is a **Rodrigues
rotation vector** (axis × angle in radians; magnitude = rotation angle,
direction = unit rotation axis).

For VLM consumption we want:

- per-keyframe **delta xyz** in centimetres from the prior keyframe
- per-keyframe **delta rotation** as (angle_deg, dominant_axis_name)
  where the axis name is "yaw" / "pitch" / "roll" / "compound" depending
  on which world-frame axis the relative rotation aligns with.

Convention used here (Franka in DROID, base mounted on table edge):

- World +x: forward (away from base)
- World +y: left (looking from robot)
- World +z: up
- Roll  axis = world x  → rotation about robot's forward axis
- Pitch axis = world y  → rotation about robot's left/right axis (gripper tilts up/down)
- Yaw   axis = world z  → rotation about vertical (gripper opening rotates in-plane)

The 7-DoF Franka end-effector reference frame is not strictly world-aligned,
but for the purposes of a coarse description ("rotated 12° around yaw") the
world-axis classification is close enough to be useful to the VLM. Refine
later if confused outputs warrant it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseDelta:
    """Difference between two end-effector poses, formatted for prompts."""
    dx_cm: float
    dy_cm: float
    dz_cm: float
    angle_deg: float
    axis_name: str   # 'yaw' | '-yaw' | 'pitch' | '-pitch' | 'roll' | '-roll' | 'mixed-axis'
    axis_unit: tuple[float, float, float]
    # WORLD-frame per-axis rotation decomposition (signed degrees).
    # roll = about world +x (robot forward axis), pitch = about world +y
    # (lateral axis), yaw = about world +z (vertical).
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    # EE-local frame ("visual EE frame", REP-103 / aircraft convention,
    # rigidly attached to the gripper). Axes:
    #   +x = forward — along camera optical axis (= EE approach direction)
    #   +y = left    — camera left in wrist view
    #   +z = up      — camera up in wrist view
    # Mapping from Franka EE axes (z=approach, x/y=lateral) is:
    #   forward = EE +z;  left = EE +x;  up = EE +y
    # (default mount assumption — may differ by 90°/180° depending on
    #  Zed-Mini bolt orientation; validate against wrist images).
    dee_forward_cm: float | None = None  # along visual +x = EE +z (approach)
    dee_left_cm: float | None = None     # along visual +y = EE +x
    dee_up_cm: float | None = None       # along visual +z = EE +y
    # EE-local frame rotation decomposition in the SAME visual frame.
    # Aircraft RPY convention:
    #   roll  = about visual +x (forward axis)  → "twist gripper around approach axis"
    #   pitch = about visual +y (left axis)     → "nod up/down, object moves vertically in wrist view"
    #   yaw   = about visual +z (up axis)       → "pan left/right, object moves horizontally in wrist view"
    roll_ee_deg: float | None = None
    pitch_ee_deg: float | None = None
    yaw_ee_deg: float | None = None
    axis_name_ee: str | None = None  # 'roll_ee' | 'pitch_ee' | 'yaw_ee' | '-roll_ee' | ... | 'mixed-axis-ee'

    def _rot_str_world(self) -> str:
        """Format rotation in world frame (existing behavior)."""
        if self.angle_deg < 0.5:
            return "Δrot_world≈0°"
        if self.axis_name != "mixed-axis":
            return f"Δrot_world={self.angle_deg:.0f}° {self.axis_name}"
        parts = sorted([
            ("roll", self.roll_deg),
            ("pitch", self.pitch_deg),
            ("yaw", self.yaw_deg),
        ], key=lambda x: abs(x[1]), reverse=True)
        out = []
        for name, deg in parts[:2]:
            if abs(deg) < 0.5: continue
            out.append(f"{abs(deg):.0f}° {name}")
        return "Δrot_world≈" + ("+".join(out) if out else f"{self.angle_deg:.0f}° mixed")

    def _rot_str_ee(self) -> str:
        """Format rotation in EE-local frame. Same magnitude as world,
        different per-axis decomposition."""
        if self.pitch_ee_deg is None:
            return ""
        if self.angle_deg < 0.5:
            return "Δrot_ee≈0°"
        if self.axis_name_ee and self.axis_name_ee != "mixed-axis-ee":
            return f"Δrot_ee={self.angle_deg:.0f}° {self.axis_name_ee}"
        parts = sorted([
            ("roll_ee", self.roll_ee_deg),
            ("pitch_ee", self.pitch_ee_deg),
            ("yaw_ee", self.yaw_ee_deg),
        ], key=lambda x: abs(x[1]), reverse=True)
        out = []
        for name, deg in parts[:2]:
            if abs(deg) < 0.5: continue
            out.append(f"{abs(deg):.0f}° {name}")
        return "Δrot_ee≈" + ("+".join(out) if out else f"{self.angle_deg:.0f}° mixed-ee")

    def __str__(self) -> str:
        # Labelled axes in robot base frame (always present): x=forward,
        # y=left, z=up. Used for action supervision (control-frame
        # consistent).
        robot_str = (
            f"Δrobot=(forward={self.dx_cm:+.1f}cm, "
            f"left={self.dy_cm:+.1f}cm, "
            f"up={self.dz_cm:+.1f}cm)"
        )
        # EE-local "visual" projection (when available). Aircraft RPY
        # convention attached to the gripper — closely matches wrist
        # camera view (modulo a fixed mounting rotation).
        if self.dee_forward_cm is not None:
            ee_str = (
                f"  Δee=(forward={self.dee_forward_cm:+.1f}cm, "
                f"left={self.dee_left_cm:+.1f}cm, "
                f"up={self.dee_up_cm:+.1f}cm)"
            )
        else:
            ee_str = ""
        rot_ee = self._rot_str_ee()
        rot_part = self._rot_str_world() + (f"  {rot_ee}" if rot_ee else "")
        return f"{robot_str}{ee_str}  {rot_part}"


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation vector → 3×3 rotation matrix (numpy-only)."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → Rodrigues rotation vector (axis × angle)."""
    # Robust extraction via trace; falls back near singularities to a
    # simpler axis-from-skew formulation. For our deltas (small angles
    # typically < 30°) the trace path is well-behaved.
    cos_a = (np.trace(R) - 1.0) * 0.5
    cos_a = max(-1.0, min(1.0, cos_a))
    angle = math.acos(cos_a)
    if abs(angle) < 1e-6:
        return np.zeros(3, dtype=np.float64)
    if abs(angle - math.pi) < 1e-6:
        # 180° rotation — pick the largest diagonal eigenvector
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]])
        i = int(np.argmax(diag))
        axis = np.zeros(3)
        axis[i] = math.sqrt(max(0.0, 0.5 * (R[i, i] + 1.0)))
        # sign-fix from off-diagonal
        for j in range(3):
            if j != i:
                axis[j] = R[i, j] / (2.0 * axis[i])
        return axis * angle
    sin_a = math.sin(angle)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * sin_a)
    return axis * angle


def relative_rotation(prev_rotvec, cur_rotvec) -> tuple[float, np.ndarray]:
    """Compute the relative rotation from prev → cur as (angle_rad, axis_unit).

    Equivalent to: R_rel = R_cur · R_prev⁻¹, then rotvec(R_rel).
    """
    R_prev = _rotvec_to_matrix(np.asarray(prev_rotvec, dtype=np.float64))
    R_cur = _rotvec_to_matrix(np.asarray(cur_rotvec, dtype=np.float64))
    R_rel = R_cur @ R_prev.T
    rv = _matrix_to_rotvec(R_rel)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-9:
        return 0.0, np.array([0.0, 0.0, 1.0])
    return angle, rv / angle


# Axis name classification (world-frame). 0.80 cosine threshold = within
# ~37° of the principal axis; below that we call it "compound".
# Raised from 0.80 → 0.92 so more multi-axis cases get decomposed.
# With 0.80, "8° pitch + 5° yaw" (dominance=0.85) was hidden as pure
# pitch. At 0.92, only truly single-axis rotations (>90% of magnitude
# on one axis) get the clean single-axis label.
_AXIS_THRESHOLD = 0.92
_AXIS_NAMES_POS = ["roll", "pitch", "yaw"]    # world x / y / z


def classify_axis(axis_unit) -> str:
    """Return 'yaw' / '-yaw' / 'pitch' / '-pitch' / 'roll' / '-roll' / 'mixed-axis'.

    Changed 'compound' → 'mixed-axis' because 'compound' leaked into
    VLM stage output as a meaningless technical term. 'mixed-axis'
    is slightly more self-explanatory but the prompt tells the VLM
    to describe the EFFECT (e.g. 'reorienting to face downward')
    rather than parroting this label.
    """
    axis_unit = np.asarray(axis_unit, dtype=np.float64)
    abs_axis = np.abs(axis_unit)
    dominant = int(np.argmax(abs_axis))
    if abs_axis[dominant] < _AXIS_THRESHOLD:
        return "mixed-axis"
    name = _AXIS_NAMES_POS[dominant]
    return name if axis_unit[dominant] >= 0 else f"-{name}"


def pose_delta(
    cur_pose: np.ndarray,    # shape (6,) — xyz + rotvec
    prev_pose: np.ndarray,   # shape (6,)
    ee_local: bool = True,
) -> PoseDelta:
    """Compute formatted pose delta from prev to current keyframe.

    When ``ee_local`` is True (default), also project the position
    delta into the source pose's EE-local frame. This frame is rigidly
    attached to the gripper, so it approximately matches the wrist
    camera view (modulo a fixed mounting rotation that the VLM can
    learn from the wrist image).
    """
    cur = np.asarray(cur_pose, dtype=np.float64)
    prev = np.asarray(prev_pose, dtype=np.float64)
    dxyz_m = cur[:3] - prev[:3]
    angle_rad, axis = relative_rotation(prev[3:6], cur[3:6])
    angle_deg = float(math.degrees(angle_rad))
    # Per-axis decomposition: project angle*axis onto world xyz.
    # For small angles (<30°) this is a good approximation of Euler.
    roll_deg = float(angle_deg * axis[0])
    pitch_deg = float(angle_deg * axis[1])
    yaw_deg = float(angle_deg * axis[2])

    # EE-local "visual" frame projections (position + rotation).
    # Mapping Franka EE (z=approach, x/y=lateral) to visual REP-103.
    # Empirically calibrated against ep0 wrist images (kf6→kf9: pot
    # moved from upper-left to upper-right in wrist view ↔ camera
    # moved in image-left direction):
    #   visual +x (forward) = +EE +z (approach)         — verified
    #   visual +y (left)    = -EE +x (sign flipped)     — empirical
    #   visual +z (up)      = -EE +y (sign flipped)     — empirical
    # The sign flips on lateral axes still form a right-handed system:
    #   visual_x × visual_y = EE_z × (-EE_x) = -EE_y = visual_z ✓
    dee_forward_cm = None
    dee_left_cm = None
    dee_up_cm = None
    roll_ee_deg = None
    pitch_ee_deg = None
    yaw_ee_deg = None
    axis_name_ee = None
    if ee_local:
        R_world_ee = _rotvec_to_matrix(prev[3:6])
        dxyz_ee = R_world_ee.T @ dxyz_m
        dee_forward_cm = float(dxyz_ee[2] * 100.0)   # visual +x = +EE +z
        dee_left_cm    = float(-dxyz_ee[0] * 100.0)  # visual +y = -EE +x
        dee_up_cm      = float(-dxyz_ee[1] * 100.0)  # visual +z = -EE +y
        # Rotation: project axis onto visual axes (same sign flips).
        # roll  about visual +x (forward) → axis_ee[2]
        # pitch about visual +y (left)    → -axis_ee[0]
        # yaw   about visual +z (up)      → -axis_ee[1]
        axis_ee = R_world_ee.T @ np.asarray(axis, dtype=np.float64)
        roll_ee_deg  = float(angle_deg * axis_ee[2])
        pitch_ee_deg = float(angle_deg * (-axis_ee[0]))
        yaw_ee_deg   = float(angle_deg * (-axis_ee[1]))
        axis_visual = np.array([axis_ee[2], -axis_ee[0], -axis_ee[1]])
        abs_ax = np.abs(axis_visual)
        dom = int(np.argmax(abs_ax))
        if abs_ax[dom] < _AXIS_THRESHOLD:
            axis_name_ee = "mixed-axis-ee"
        else:
            base = _AXIS_NAMES_POS[dom] + "_ee"
            axis_name_ee = base if axis_visual[dom] >= 0 else f"-{base}"

    return PoseDelta(
        dx_cm=float(dxyz_m[0] * 100.0),
        dy_cm=float(dxyz_m[1] * 100.0),
        dz_cm=float(dxyz_m[2] * 100.0),
        angle_deg=angle_deg,
        axis_name=classify_axis(axis),
        axis_unit=tuple(float(x) for x in axis),
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        dee_forward_cm=dee_forward_cm,
        dee_left_cm=dee_left_cm,
        dee_up_cm=dee_up_cm,
        roll_ee_deg=roll_ee_deg,
        pitch_ee_deg=pitch_ee_deg,
        yaw_ee_deg=yaw_ee_deg,
        axis_name_ee=axis_name_ee,
    )


def _smoke():
    # Test 1: zero pose → zero delta
    p0 = np.zeros(6)
    d = pose_delta(p0, p0)
    print(f"zero delta: {d}")
    assert d.dx_cm == 0 and d.angle_deg == 0

    # Test 2: 5 cm forward, no rotation
    p1 = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = pose_delta(p1, p0)
    print(f"+5cm forward: {d}")
    assert abs(d.dx_cm - 5.0) < 0.01

    # Test 3: pure +30° yaw
    yaw30 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.radians(30)])
    d = pose_delta(yaw30, p0)
    print(f"+30° yaw: {d}")
    assert abs(d.angle_deg - 30.0) < 0.1
    assert d.axis_name == "yaw"

    # Test 4: pure -20° pitch
    pitch_n20 = np.array([0.0, 0.0, 0.0, 0.0, math.radians(-20), 0.0])
    d = pose_delta(pitch_n20, p0)
    print(f"-20° pitch: {d}")
    assert abs(d.angle_deg - 20.0) < 0.1
    assert d.axis_name == "-pitch"

    # Test 5: mixed-axis rotation (yaw + pitch) — decomposed in output
    yp = np.array([0.0, 0.0, 0.0, 0.0, math.radians(15), math.radians(15)])
    d = pose_delta(yp, p0)
    print(f"mixed 15°+15°: {d}")
    assert d.axis_name == "mixed-axis"

    print("\nAll smoke tests pass.")


if __name__ == "__main__":
    _smoke()
