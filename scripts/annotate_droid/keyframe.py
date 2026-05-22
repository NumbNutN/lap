"""Rule-based keyframe detection for DROID embodied-CoT annotation.

Why rule-based (not VLM-picked):

- Reproducible — same episode always yields the same keyframe set, so prompt
  iteration changes do not perturb keyframe selection.
- Cheap — runs locally, no API cost, no GPU.
- The VLM only needs to reason at fixed inflection points; letting it also
  *choose* them turns each annotation into a two-step task with worse
  consistency.

Inputs (per episode):
    gripper_width: np.ndarray[float], shape (T,)
        Continuous gripper width. For Franka Panda DROID, 0.0 ≈ closed,
        ~0.08 m ≈ fully open. We normalise into {open, partial, closed}
        with hysteresis to suppress jitter.
    ee_pos:        np.ndarray[float], shape (T, 3)
        End-effector translation (world frame). Used only for motion-phase
        change detection (R4), not for primary phase boundaries.
    fps:           float
        Sampling rate of the episode. DROID default is ~15 Hz.

Output:
    list[Keyframe] — sorted by timestep, deduplicated, with type tags
        that the prompt builder uses to bias the VLM's reasoning.

Rule taxonomy (each function below):

    R1  gripper_state(width)              → {open, partial, closed}
    R2  gripper_transitions(width)        → primary phase boundaries
    R3  detect_grasp_retries(transitions) → close→open→close within 1.5s
    R4  detect_motion_changes(ee_pos)     → EE-velocity direction shift
    R5  boundary keyframes                → first + last frame
    R6  fill_long_gaps(keyframes)         → cap stage length at max_gap

The aggregator `detect_keyframes` unions them, NMS-suppresses neighbours,
fills long gaps, and tags each surviving keyframe.

See README_droid_annotation.md §2 for the design rationale + thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Thresholds (Franka Panda DROID, ~15Hz). Tune in __main__ debug mode.
# ---------------------------------------------------------------------------

# Gripper width discretisation (metres). Franka max is ~0.08.
GRIP_CLOSED_MAX = 0.005
GRIP_OPEN_MIN = 0.060

# Hysteresis: how many consecutive frames must a new state hold before we
# accept it as a transition. Suppresses bounce/contact-pulse noise.
GRIP_PERSIST_FRAMES = 3

# Retry window (s): a grasp→release→grasp sequence within this window is
# treated as a failed-grasp retry and the retry start becomes a keyframe.
RETRY_WINDOW_S = 1.5

# Motion-phase change detector (R4).
MOTION_WINDOW = 10           # frames on each side for velocity average
MOTION_COS_MIN = 0.30        # cos(v_before, v_after) below this = direction flip
MOTION_SPEED_DELTA_M = 0.05  # |speed_after - speed_before| > this (m/s) = phase change
MOTION_NMS_FRAMES = 8        # min frame gap between two motion-change keyframes
MOTION_MIN_SPEED_M = 0.02    # ignore direction-flip if either side speed < this (m/s)

# Long-stage gap filler (R6): if two adjacent keyframes are > MAX_GAP apart,
# insert mid-frames so no stage exceeds this many frames without a checkpoint.
MAX_GAP_FRAMES = 60


GripperState = Literal["open", "partial", "closed"]
KeyframeType = Literal[
    "begin",
    "end",
    "grasp",       # open → closed transition
    "release",     # closed → open transition
    "retry",       # close → open → close within RETRY_WINDOW_S
    "motion",      # EE velocity direction / speed change
    "filler",      # inserted to cap stage length (R6)
]


@dataclass
class Keyframe:
    """A single annotation anchor point."""
    t: int
    """Frame index (0-based, into the episode's frame sequence)."""

    type: KeyframeType
    """Which rule fired. Used by the prompt builder to bias the VLM."""

    gripper_state: GripperState | None = None
    """Discretised gripper state at frame `t`. None for filler keyframes."""

    extra: dict = field(default_factory=dict)
    """Optional metadata (e.g. for retry: which earlier frame was the failed
    grasp; for motion: cos-similarity at the change point)."""


# ---------------------------------------------------------------------------
# R1: Gripper state discretisation
# ---------------------------------------------------------------------------


def gripper_state(width: float) -> GripperState:
    if width < GRIP_CLOSED_MAX:
        return "closed"
    if width > GRIP_OPEN_MIN:
        return "open"
    return "partial"


# ---------------------------------------------------------------------------
# R2: Gripper transitions (with hysteresis)
# ---------------------------------------------------------------------------


def gripper_transitions(
    gripper_width: np.ndarray,
    persist_frames: int = GRIP_PERSIST_FRAMES,
) -> list[tuple[int, GripperState, GripperState]]:
    """Return a list of (frame_idx, from_state, to_state) transitions.

    Uses persistence-based hysteresis: a new state is only accepted if it
    holds for `persist_frames` consecutive frames. Without this, finger
    contact pulses on closure create spurious transitions.
    """
    states = [gripper_state(float(w)) for w in gripper_width]
    if len(states) < 2:
        return []
    transitions: list[tuple[int, GripperState, GripperState]] = []
    last_stable: GripperState = states[0]
    t = 1
    n = len(states)
    while t < n:
        if states[t] != last_stable:
            window_end = min(t + persist_frames, n)
            window = states[t:window_end]
            if len(window) >= persist_frames and all(s == states[t] for s in window):
                transitions.append((t, last_stable, states[t]))
                last_stable = states[t]
                t = window_end
                continue
        t += 1
    return transitions


# ---------------------------------------------------------------------------
# R3: Failed-grasp retry detection
# ---------------------------------------------------------------------------


def detect_grasp_retries(
    transitions: list[tuple[int, GripperState, GripperState]],
    fps: float,
) -> list[tuple[int, int, int]]:
    """Detect close → open → close patterns within RETRY_WINDOW_S.

    Returns list of (t_first_close, t_open, t_retry_close) tuples. The
    retry start (t_retry_close) becomes a keyframe; the failure cue
    (t_open) flagged as the "previous attempt failed" context.
    """
    retries: list[tuple[int, int, int]] = []
    window_frames = int(RETRY_WINDOW_S * fps)
    for i in range(2, len(transitions)):
        t_a, s_a_from, s_a_to = transitions[i - 2]
        t_b, s_b_from, s_b_to = transitions[i - 1]
        t_c, s_c_from, s_c_to = transitions[i]
        # Pattern: ?→closed, closed→open, open→closed within window
        if (s_a_to == "closed"
                and s_b_from == "closed" and s_b_to == "open"
                and s_c_from == "open" and s_c_to == "closed"
                and (t_c - t_a) <= window_frames):
            retries.append((t_a, t_b, t_c))
    return retries


# ---------------------------------------------------------------------------
# R4: EE-pose motion phase change
# ---------------------------------------------------------------------------


def detect_motion_changes(
    ee_pos: np.ndarray,
    fps: float,
    window: int = MOTION_WINDOW,
) -> list[tuple[int, float]]:
    """Detect velocity direction-flips or speed jumps in EE motion.

    Useful for tasks where gripper state doesn't change but the manipulation
    phase does (push, slide, hand-off without re-grip). Returns
    (frame_idx, cos_similarity) tuples after NMS.
    """
    if len(ee_pos) < 2 * window + 2:
        return []
    vel = np.diff(ee_pos, axis=0) * fps  # m/s
    candidates: list[tuple[int, float]] = []
    for t in range(window, len(vel) - window):
        v_before = vel[t - window:t].mean(axis=0)
        v_after = vel[t:t + window].mean(axis=0)
        speed_before = float(np.linalg.norm(v_before))
        speed_after = float(np.linalg.norm(v_after))
        if speed_before < 1e-4 and speed_after < 1e-4:
            continue
        # Only flag a *direction flip* if both sides are clearly moving;
        # otherwise we're catching start/stop transients which are better
        # captured by gripper transitions (R2) anyway.
        cos_sim = float(
            np.dot(v_before, v_after)
            / (speed_before * speed_after + 1e-6)
        )
        speed_delta = abs(speed_after - speed_before)
        cos_flip = (
            cos_sim < MOTION_COS_MIN
            and speed_before > MOTION_MIN_SPEED_M
            and speed_after > MOTION_MIN_SPEED_M
        )
        if cos_flip or speed_delta > MOTION_SPEED_DELTA_M:
            # +1 because vel is offset by 1 from ee_pos
            candidates.append((t + 1, cos_sim))

    # NMS: greedily keep peaks, suppress neighbours within MOTION_NMS_FRAMES
    candidates.sort(key=lambda x: x[1])  # lowest cos-sim first = sharpest turn
    kept: list[tuple[int, float]] = []
    used: set[int] = set()
    for t, cs in candidates:
        if any(abs(t - k) < MOTION_NMS_FRAMES for k in used):
            continue
        kept.append((t, cs))
        used.add(t)
    return sorted(kept, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# R6: Long-gap filler
# ---------------------------------------------------------------------------


def fill_long_gaps(
    keyframes: list[int],
    max_gap: int = MAX_GAP_FRAMES,
) -> list[int]:
    """Ensure no two adjacent keyframes are more than `max_gap` apart.

    Inserts evenly spaced fillers when needed. Without this, a very long
    "transport" stage gets no mid-stage anchor and the VLM is asked to
    reason about a 200-frame span from two endpoint images only.
    """
    if len(keyframes) <= 1:
        return list(keyframes)
    out: list[int] = [keyframes[0]]
    for i in range(1, len(keyframes)):
        gap = keyframes[i] - keyframes[i - 1]
        if gap > max_gap:
            n_mids = gap // max_gap
            for j in range(1, n_mids + 1):
                mid = keyframes[i - 1] + (j * gap) // (n_mids + 1)
                if mid not in out:
                    out.append(mid)
        out.append(keyframes[i])
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def detect_keyframes(
    *,
    gripper_width: np.ndarray,
    ee_pos: np.ndarray | None = None,
    fps: float = 15.0,
    enable_motion_change: bool = True,
) -> list[Keyframe]:
    """Run all rules, union the candidates, label and return."""
    T = int(len(gripper_width))
    if T == 0:
        return []

    # R2 — primary signal
    transitions = gripper_transitions(gripper_width)

    # R3 — retries
    retries = detect_grasp_retries(transitions, fps=fps)
    retry_starts = {t_c for (_, _, t_c) in retries}
    retry_failed_grasps = {t_a for (t_a, _, _) in retries}

    # R4 — motion changes (optional)
    motion_changes: list[tuple[int, float]] = []
    if enable_motion_change and ee_pos is not None:
        motion_changes = detect_motion_changes(np.asarray(ee_pos), fps=fps)

    # R5 — boundary
    raw_keyframes: dict[int, KeyframeType] = {0: "begin", T - 1: "end"}

    for t, _, to_state in transitions:
        if t in retry_starts:
            raw_keyframes[t] = "retry"
        elif to_state == "closed":
            raw_keyframes[t] = "grasp"
        elif to_state == "open":
            raw_keyframes[t] = "release"
        else:
            raw_keyframes.setdefault(t, "motion")

    for t, _cs in motion_changes:
        raw_keyframes.setdefault(t, "motion")

    # R6 — fill long gaps
    sorted_indices = sorted(raw_keyframes.keys())
    filled = fill_long_gaps(sorted_indices)
    for t in filled:
        raw_keyframes.setdefault(t, "filler")

    # Build Keyframe objects with state tags
    out: list[Keyframe] = []
    for t in sorted(raw_keyframes.keys()):
        kf_type = raw_keyframes[t]
        kf = Keyframe(
            t=int(t),
            type=kf_type,
            gripper_state=gripper_state(float(gripper_width[t])),
        )
        if t in retry_failed_grasps:
            kf.extra["previous_attempt_frame"] = int(t)  # the failed one
        out.append(kf)
    return out


# ---------------------------------------------------------------------------
# Debug entry — `python -m lap.scripts.annotate_droid.keyframe synthetic`
# ---------------------------------------------------------------------------


def _synthetic_demo() -> None:
    """Print keyframes for a hand-crafted synthetic episode.

    Sanity check that all rules fire correctly; useful when tuning thresholds.
    """
    fps = 15.0
    T = 200
    # gripper: open 0-40, closing 40-50 (failed), opening 50-60 (retry trigger),
    # closing 60-70 (success), holding 70-150, releasing 150-160, open to end
    width = np.full(T, 0.08, dtype=np.float32)
    width[40:50] = 0.0      # failed grasp attempt
    width[50:60] = 0.08     # release
    width[60:150] = 0.0     # successful grasp + hold
    width[150:] = 0.08      # final release

    # EE: stationary, then linear motion, then turn, then return
    ee = np.zeros((T, 3), dtype=np.float32)
    for t in range(70, 110):
        ee[t] = ((t - 70) * 0.01, 0.0, 0.0)  # +x
    for t in range(110, 150):
        ee[t] = (0.4, (t - 110) * 0.01, 0.0)  # +y (90 deg turn)

    kfs = detect_keyframes(gripper_width=width, ee_pos=ee, fps=fps)
    print(f"episode T={T}, fps={fps}, keyframes={len(kfs)}")
    for kf in kfs:
        print(f"  t={kf.t:4d}  type={kf.type:<8}  grip={kf.gripper_state}  extra={kf.extra}")


if __name__ == "__main__":
    _synthetic_demo()
