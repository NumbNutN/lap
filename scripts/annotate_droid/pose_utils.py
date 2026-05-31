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
    # EE-local frame projection of the POSITION delta. Computed using
    # the source pose's EE orientation (prev_pose's rotvec). Axes
    # follow Franka EE convention (+z = approach direction / out of
    # gripper jaws; +x, +y = perpendicular). For the wrist camera
    # (rigidly mounted on the gripper flange), Δee_approach corresponds
    # to "object getting closer in wrist view", and the perpendicular
    # components map approximately to the wrist camera's in-plane axes
    # up to a fixed mounting rotation.
    dee_approach_cm: float | None = None  # along EE +z
    dee_perp_x_cm: float | None = None    # along EE +x
    dee_perp_y_cm: float | None = None    # along EE +y
    # EE-local frame rotation decomposition. SAME physical rotation as
    # roll/pitch/yaw_deg above, just expressed in the SOURCE EE's local
    # frame instead of world. The total magnitude (angle_deg) is
    # identical — only the per-axis decomposition differs by frame.
    # Naming convention (matches Δrot_world for VLM consistency):
    #   pitch_ee = around EE +y axis
    #   yaw_ee   = around EE +z axis (approach axis — "twist")
    #   roll_ee  = around EE +x axis
    pitch_ee_deg: float | None = None
    yaw_ee_deg: float | None = None
    roll_ee_deg: float | None = None
    axis_name_ee: str | None = None  # 'pitch_ee' | '-pitch_ee' | ... | 'mixed-axis-ee'

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
        # EE-local projection (when available). Used for visual stage
        # description — approximately matches the wrist camera view.
        if self.dee_approach_cm is not None:
            ee_str = (
                f"  Δee=(approach={self.dee_approach_cm:+.1f}cm, "
                f"perp_x={self.dee_perp_x_cm:+.1f}cm, "
                f"perp_y={self.dee_perp_y_cm:+.1f}cm)"
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

    # EE-local frame projections (position + rotation).
    # R_world_ee maps EE-frame vectors to world; so R_world_ee.T maps
    # a world-frame vector into EE-local frame. The source EE rotation
    # (prev[3:6]) is the reference frame at the START of the delta.
    dee_approach_cm = None
    dee_perp_x_cm = None
    dee_perp_y_cm = None
    pitch_ee_deg = None
    yaw_ee_deg = None
    roll_ee_deg = None
    axis_name_ee = None
    if ee_local:
        R_world_ee = _rotvec_to_matrix(prev[3:6])
        # Position delta in EE frame.
        dxyz_ee = R_world_ee.T @ dxyz_m
        dee_perp_x_cm = float(dxyz_ee[0] * 100.0)
        dee_perp_y_cm = float(dxyz_ee[1] * 100.0)
        dee_approach_cm = float(dxyz_ee[2] * 100.0)
        # Rotation axis re-expressed in source EE frame. The angle is
        # invariant (rotations have an invariant magnitude); only the
        # axis vector is rotated into the new frame.
        axis_ee = R_world_ee.T @ np.asarray(axis, dtype=np.float64)
        roll_ee_deg = float(angle_deg * axis_ee[0])
        pitch_ee_deg = float(angle_deg * axis_ee[1])
        yaw_ee_deg = float(angle_deg * axis_ee[2])
        # Reuse classify_axis machinery for EE frame, append "_ee" suffix
        # so the VLM cannot confuse it with the world-frame label.
        abs_ax = np.abs(axis_ee)
        dom = int(np.argmax(abs_ax))
        if abs_ax[dom] < _AXIS_THRESHOLD:
            axis_name_ee = "mixed-axis-ee"
        else:
            base = _AXIS_NAMES_POS[dom] + "_ee"
            axis_name_ee = base if axis_ee[dom] >= 0 else f"-{base}"

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
        dee_approach_cm=dee_approach_cm,
        dee_perp_x_cm=dee_perp_x_cm,
        dee_perp_y_cm=dee_perp_y_cm,
        pitch_ee_deg=pitch_ee_deg,
        yaw_ee_deg=yaw_ee_deg,
        roll_ee_deg=roll_ee_deg,
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
