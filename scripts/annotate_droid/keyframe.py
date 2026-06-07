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
        Gripper command 0..1 (0=open, 1=closed). Unified via %open=(1-g)*100:
        grasp/release are detected as %open TRENDS (sustained runs), and the
        discrete {open,partial,closed} label is DERIVED from %open (display
        only). Works for DROID and sim teleop alike (robust to thin-object
        grips where %open only falls to ~50).
    ee_pos:        np.ndarray[float], shape (T, 3)
        End-effector translation (world frame). Used only for motion-phase
        change detection (R4), not for primary phase boundaries.
    fps:           float
        Sampling rate of the episode. DROID default is ~15 Hz.

Output:
    list[Keyframe] — sorted by timestep, deduplicated, with type tags
        that the prompt builder uses to bias the VLM's reasoning.

Rule taxonomy (each function below):

    R1  gripper_state(width)              → {open, partial, closed} (from %open)
    R2  gripper_events(width)             → grasp/release from %open trends
    R3  detect_grasp_retries(events)      → grasp→release→grasp within 1.5s
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

# DROID gripper convention (confirmed by direct inspection of
# /data/datasets/droid_data_template/droid_100, see analysis 2026-05-24):
#
#   observation.gripper_position is the COMMANDED normalised position
#   in [0, 1] where 0 = fully open command, 1 = fully closed command.
#   Actual physical width depends on what is between the fingers (a thin
#   object stops closure early, so values rarely reach 1.0 — a marker
#   episode tops out at ~0.81).
#
# Earlier we had this inverted (assuming Franka physical width where
# 0 ≈ closed and 0.08 ≈ open) which gave ALL ep0 keyframe types wrong
# direction. See cot_annotation discussion 2026-05-24.
# Unified gripper semantics: %open = (1 - gripper)*100  (0=closed, 100=open).
# Single source of truth — state labels AND grasp/release detection both derive
# from %open, so DROID (command ~0..0.81) and sim teleop share one convention.
GRIP_OPEN_MIN_PCT = 80    # %open > this → "open"
GRIP_CLOSED_MAX_PCT = 20  # %open < this → "closed"  (else "partial")
GRIP_RUN_DELTA_PCT = 20   # a closing/opening RUN must move %open by ≥ this to
                          # count as grasp/release (robust to a thin object where
                          # %open only drops to ~50 and never hits absolute closed)

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
    """Discrete label DERIVED from %open (display/reference only; detection uses
    the %open trend, not this absolute label)."""
    pct = (1.0 - float(width)) * 100.0
    if pct > GRIP_OPEN_MIN_PCT:
        return "open"
    if pct < GRIP_CLOSED_MAX_PCT:
        return "closed"
    return "partial"


# ---------------------------------------------------------------------------
# R2: Gripper transitions (with hysteresis)
# ---------------------------------------------------------------------------


def gripper_events(gripper_width: np.ndarray) -> list[tuple[int, str]]:
    """Detect grasp/release as %open TRENDS (runs), anchored at the run onset.

    %open = (1-gripper)*100. A sustained closing run (%open drops by ≥
    GRIP_RUN_DELTA_PCT) → 'grasp'; an opening run → 'release'. Robust to
    gripping a thin object (where %open only falls to ~50 and never reaches an
    absolute 'closed'), unlike the old open→closed state machine.

    Returns [(onset_frame, 'grasp'|'release'), ...] in time order.
    """
    pct = (1.0 - np.asarray(gripper_width, dtype=np.float64)) * 100.0
    n = len(pct)
    events: list[tuple[int, str]] = []
    i = 1
    while i < n:
        d = pct[i] - pct[i - 1]
        if abs(d) < 1.0:                       # flat → no run starts here
            i += 1
            continue
        direction = 1 if d > 0 else -1
        start = i - 1
        j = i
        while j < n and (pct[j] - pct[j - 1]) * direction >= -2.0:  # tolerate tiny noise
            j += 1
        total = pct[j - 1] - pct[start]
        if abs(total) >= GRIP_RUN_DELTA_PCT:
            events.append((int(start), "release" if total > 0 else "grasp"))
        i = max(j, i + 1)
    return events


# ---------------------------------------------------------------------------
# R3: Failed-grasp retry detection
# ---------------------------------------------------------------------------


def refine_to_motion_start(
    gripper_width: np.ndarray,
    transition_t: int,
    persist_frames: int = GRIP_PERSIST_FRAMES,
    eps: float = 0.05,
) -> int:
    """Walk backward from a state-transition timestamp to find the first
    frame where the gripper VALUE began departing from its prior baseline.

    State-based detection (R2) anchors at the END of a transition (when
    the new state stabilizes). For human-meaningful labels we want the
    START of the motion (when fingers first begin to close / open).
    """
    if transition_t <= 0:
        return 0
    lookback = max(persist_frames * 4, 12)
    start = max(0, transition_t - lookback)
    baseline = float(gripper_width[start])
    for t in range(start + 1, transition_t + 1):
        if abs(float(gripper_width[t]) - baseline) > eps:
            return max(start, t - 1)
    return transition_t


def detect_grasp_retries(
    events: list[tuple[int, str]],
    fps: float,
) -> list[tuple[int, int, int]]:
    """grasp → release → grasp within RETRY_WINDOW_S = a failed-grasp retry.
    Returns (t_first_grasp, t_release, t_retry_grasp); the retry grasp becomes
    the keyframe, the first grasp is flagged as the failed attempt."""
    retries: list[tuple[int, int, int]] = []
    window_frames = int(RETRY_WINDOW_S * fps)
    for i in range(2, len(events)):
        (ta, ka), (tb, kb), (tc, kc) = events[i - 2], events[i - 1], events[i]
        if ka == "grasp" and kb == "release" and kc == "grasp" \
                and (tc - ta) <= window_frames:
            retries.append((ta, tb, tc))
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

    # R2 — grasp/release from %open trends (anchored at run onset already)
    events = gripper_events(gripper_width)

    # R3 — retries (grasp→release→grasp)
    retries = detect_grasp_retries(events, fps=fps)
    retry_starts = {tc for (_, _, tc) in retries}
    retry_failed_grasps = {ta for (ta, _, _) in retries}

    # R4 — motion changes (optional)
    motion_changes: list[tuple[int, float]] = []
    if enable_motion_change and ee_pos is not None:
        motion_changes = detect_motion_changes(np.asarray(ee_pos), fps=fps)

    # R5 — boundary
    raw_keyframes: dict[int, KeyframeType] = {0: "begin", T - 1: "end"}

    for start, kind in events:
        if start in (0, T - 1):                # never override begin/end
            continue
        raw_keyframes[start] = "retry" if start in retry_starts else kind

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
