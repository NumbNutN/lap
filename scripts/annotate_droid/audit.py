"""Automated quality checks on a parsed annotation.

Catches the most common failure modes:

  A1. keyframes count mismatch (VLM dropped or duplicated keyframes)
  A2. frame_idx mismatch (VLM rewrote frame indices)
  A3. gripper / action mismatch on grasp / release keyframes (R2/R3)
  A4. retry keyframe without [think] (R4)
  A5. excessively long stage / plan strings (R8)
  A6. think field populated on routine keyframes (style — warning only)
  A7. empty plan or empty stages (data error)

Reports separate `errors` (will fail acceptance) from `warnings`
(advisory). Used both by the runner (gate per-episode storage) and by
the pilot evaluation script.

A8. Gripper-action consistency at "grasp" and "release" keyframes is
checked via simple keyword match — strict phrasing matters less than
that the VLM didn't confuse direction.
"""

from __future__ import annotations

import re

from .schema import AuditReport
from .schema import EpisodeAnnotation
from .schema import KeyframeAnnotation


# A9 axis-aware coverage threshold: at least this fraction of a v3
# episode's `motion`-type keyframes should mention an axis name (yaw /
# pitch / roll / cm / °). Below this, the episode looks "coarse" and is
# inconsistent with the rest of the dataset. Warning, not error.
_AXIS_AWARE_PAT = re.compile(
    r"\b(yaw|pitch|roll|cm|°|degrees?|translate|tilt|rotate)\b",
    re.IGNORECASE,
)
_AXIS_AWARE_MIN_RATE_MOTION = 0.30   # 30% of motion keyframes
_AXIS_AWARE_MIN_RATE_ANY = 0.20      # 20% of all keyframes


# Phrasing whitelists used for A3 / R2-R3 (positive constraint at grasp /
# release keyframes — these MUST mention grasping / releasing).
GRASP_VERBS = {"close", "grasp", "grip", "clamp", "squeeze", "pinch", "pick"}
RELEASE_VERBS = {"open", "release", "let go", "drop", "place", "set down", "put down"}

# Stricter vocab used by A8 (negative constraint: at motion keyframes the
# model can describe IMMINENT action like "Place X" / "Position to pick Y"
# without contradicting the gripper state — those are stage-purpose verbs,
# not physical-transition verbs. A8 should only fire on direct gripper
# physical-action verbs like "open the gripper" / "release X" that flatly
# contradict the observed gripper state. 5-ep pilot showed the old A8
# triggering on every "Place"/"Pick" at mid-motion frames, all false.)
STRICT_GRASP_PHRASES = {"close the gripper", "close gripper", "clamp", "squeeze"}
# "release" alone is too broad — "approach release height" / "after
# release" use it as noun/adjective and shouldn't trigger A8. Require
# verb-object patterns: "release the X" / "release it" / "release onto".
STRICT_RELEASE_PHRASES = {
    "open the gripper", "open gripper",
    "release the", "release it", "release onto",
}

# Hard length caps (R8 in the system prompt). These map to ~2-3 sentences.
MAX_PLAN_CHARS = 600
MAX_STAGE_CHARS = 350
MAX_ACTION_WORDS = 22  # allow geometric/compound phrases like "rotate yaw to align"

# Keyframe types where think is expected (R4)
TYPES_REQUIRING_THINK = {"retry"}

# Keyframe types where think is usually noise (style warning only)
TYPES_THINK_USUALLY_NOISE = {"begin", "end", "filler"}


def _contains_any(text: str, verbs: set[str]) -> bool:
    """Whole-word / word-prefix match. ``"gripper"`` MUST NOT count as ``"grip"``;
    use ``\\b`` boundaries. Multi-word verbs (e.g. ``"let go"``) match as substrings."""
    t = text.lower()
    for v in verbs:
        if " " in v:
            if v in t:
                return True
        elif re.search(rf"\b{re.escape(v)}\b", t):
            return True
    return False


def audit_episode(
    ann: EpisodeAnnotation,
    *,
    expected_keyframe_indices: list[int] | None = None,
    expected_keyframe_types: list[str] | None = None,
    expected_gripper_states: list[str] | None = None,
) -> AuditReport:
    """Run all checks. Mutates and returns a fresh AuditReport.

    The expected_* lists come from the keyframe detector and must align
    1:1 with ann.keyframes. Pass them when available — they enable A2/A3/A4.
    """
    report = AuditReport(passed=True)

    # A1 / A2: count and index alignment
    expected = expected_keyframe_indices or ann.keyframe_indices
    if len(ann.keyframes) != len(expected):
        report.errors.append(
            f"A1 keyframe count mismatch: got {len(ann.keyframes)}, "
            f"expected {len(expected)}"
        )
    else:
        for i, (kf, exp_idx) in enumerate(zip(ann.keyframes, expected, strict=True)):
            if kf.frame_idx != exp_idx:
                report.errors.append(
                    f"A2 keyframes[{i}] frame_idx mismatch: "
                    f"got {kf.frame_idx}, expected {exp_idx}"
                )

    # A7: empty content
    if not ann.plan.strip():
        report.errors.append("A7 empty plan")
    for i, kf in enumerate(ann.keyframes):
        if not kf.stage.strip():
            report.errors.append(f"A7 keyframes[{i}].stage is empty")
        if not kf.action.strip():
            report.errors.append(f"A7 keyframes[{i}].action is empty")

    # A5: length caps
    if len(ann.plan) > MAX_PLAN_CHARS:
        report.warnings.append(
            f"A5 plan very long: {len(ann.plan)} chars (cap {MAX_PLAN_CHARS})"
        )
    for i, kf in enumerate(ann.keyframes):
        if len(kf.stage) > MAX_STAGE_CHARS:
            report.warnings.append(
                f"A5 keyframes[{i}].stage long: {len(kf.stage)} chars"
            )
        if len(kf.action.split()) > MAX_ACTION_WORDS:
            report.warnings.append(
                f"A5 keyframes[{i}].action verbose: {len(kf.action.split())} words"
            )

    # A3 / R2-R3 / A4: type-conditional checks
    if expected_keyframe_types is not None and len(expected_keyframe_types) == len(ann.keyframes):
        for i, (kf, kf_type) in enumerate(zip(ann.keyframes, expected_keyframe_types, strict=True)):
            # A3: grasp keyframe must have grasping action
            if kf_type == "grasp" and not _contains_any(kf.action, GRASP_VERBS):
                report.errors.append(
                    f"A3 keyframes[{i}] type=grasp but action lacks grasp verb: "
                    f"{kf.action!r}"
                )
            # A3: release keyframe must have releasing action
            if kf_type == "release" and not _contains_any(kf.action, RELEASE_VERBS):
                report.errors.append(
                    f"A3 keyframes[{i}] type=release but action lacks release verb: "
                    f"{kf.action!r}"
                )
            # A4: retry keyframe must have think
            if kf_type in TYPES_REQUIRING_THINK and not kf.think:
                report.errors.append(
                    f"A4 keyframes[{i}] type={kf_type} requires non-null think"
                )
            # A6: think on usually-noise types — warn only
            if kf_type in TYPES_THINK_USUALLY_NOISE and kf.think:
                report.warnings.append(
                    f"A6 keyframes[{i}] type={kf_type} has unexpected think: "
                    f"{kf.think[:60]!r}"
                )

    # A8: gripper-state consistency (uses STRICT vocab — see top of file).
    # NOTE: with motion-START keyframe anchoring (introduced 2026-05-24),
    # grasp keyframes legitimately have gripper_state="open" (fingers
    # about to close) and release keyframes have gripper_state="closed"
    # (fingers about to open). So A8 must SKIP grasp/release types —
    # they're already constrained by A3 above. A8 only catches
    # contradictions at motion/begin/end/filler keyframes where the
    # gripper state is steady.
    if (expected_gripper_states is not None
            and expected_keyframe_types is not None
            and len(expected_gripper_states) == len(ann.keyframes)
            and len(expected_keyframe_types) == len(ann.keyframes)):
        for i, (kf, gs, kf_type) in enumerate(zip(
                ann.keyframes, expected_gripper_states, expected_keyframe_types,
                strict=True)):
            if kf_type in {"grasp", "release", "retry"}:
                continue
            if gs == "closed" and _contains_any(kf.action, STRICT_RELEASE_PHRASES):
                report.errors.append(
                    f"A8 keyframes[{i}] (type={kf_type}) gripper closed but "
                    f"action says it is opening: {kf.action!r}"
                )
            if gs == "open" and _contains_any(kf.action, STRICT_GRASP_PHRASES):
                report.errors.append(
                    f"A8 keyframes[{i}] (type={kf_type}) gripper open but "
                    f"action says it is closing: {kf.action!r}"
                )

    # A9: axis-aware action coverage (v3 memory-augmented mode only).
    # When the input keyframe meta carried pose deltas, the VLM was
    # explicitly asked to use axis-aware vocab. If almost no keyframe
    # mentions an axis, the model defaulted to coarse "adjust position"
    # phrasing — warn the operator so they can decide whether to
    # re-prompt or accept the episode. This is a STYLE check, not a
    # correctness check, so it's a warning even if all motion keyframes
    # lack axis vocab.
    if expected_keyframe_types is not None and len(expected_keyframe_types) == len(ann.keyframes):
        motion_kfs = [
            (kf, t) for kf, t in zip(ann.keyframes, expected_keyframe_types, strict=True)
            if t in {"motion", "filler", "begin", "end"}
        ]
        if motion_kfs:
            n_motion_axis = sum(
                1 for kf, _ in motion_kfs
                if _AXIS_AWARE_PAT.search(kf.action or "")
            )
            rate_motion = n_motion_axis / len(motion_kfs)
            if rate_motion < _AXIS_AWARE_MIN_RATE_MOTION:
                report.warnings.append(
                    f"A9 axis-aware vocab coverage on motion keyframes: "
                    f"{n_motion_axis}/{len(motion_kfs)} = {rate_motion*100:.0f}% "
                    f"(threshold {_AXIS_AWARE_MIN_RATE_MOTION*100:.0f}%). "
                    f"Episode reads as coarse; consider re-prompting."
                )

        n_any_axis = sum(
            1 for kf in ann.keyframes
            if _AXIS_AWARE_PAT.search(kf.action or "")
        )
        rate_any = n_any_axis / max(1, len(ann.keyframes))
        if rate_any < _AXIS_AWARE_MIN_RATE_ANY:
            report.warnings.append(
                f"A9 axis-aware vocab coverage overall: "
                f"{n_any_axis}/{len(ann.keyframes)} = {rate_any*100:.0f}% "
                f"(threshold {_AXIS_AWARE_MIN_RATE_ANY*100:.0f}%)."
            )

    report.passed = not report.errors
    return report


def _smoke() -> None:
    from .schema import EpisodeAnnotation, KeyframeAnnotation

    good = EpisodeAnnotation(
        episode_id="test",
        task_instruction="Put the cup on the saucer",
        fps=15.0,
        n_frames=100,
        keyframe_indices=[0, 30, 60],
        keyframe_types=["begin", "grasp", "release"],
        plan="Move the cup to the saucer.",
        keyframes=[
            KeyframeAnnotation(0, "start", "Approach the cup."),
            KeyframeAnnotation(30, "grasping cup", "Close the gripper on the cup handle."),
            KeyframeAnnotation(60, "releasing", "Open the gripper to place the cup."),
        ],
    )
    rep = audit_episode(
        good,
        expected_keyframe_types=good.keyframe_types,
        expected_gripper_states=["open", "closed", "open"],
    )
    assert rep.passed, rep.errors
    print(f"good: pass={rep.passed} errs={rep.errors} warns={rep.warnings}")

    bad = EpisodeAnnotation(
        episode_id="test2",
        task_instruction="Put the cup on the saucer",
        fps=15.0,
        n_frames=100,
        keyframe_indices=[0, 30, 60],
        keyframe_types=["begin", "grasp", "release"],
        plan="",
        keyframes=[
            KeyframeAnnotation(0, "start", "Approach the cup."),
            KeyframeAnnotation(30, "now grasping", "Open the gripper."),  # WRONG action
            KeyframeAnnotation(60, "now placing", "Close the gripper."),  # WRONG action
        ],
    )
    rep = audit_episode(
        bad,
        expected_keyframe_types=bad.keyframe_types,
        expected_gripper_states=["open", "closed", "open"],
    )
    assert not rep.passed
    print(f"bad: pass={rep.passed}")
    for e in rep.errors:
        print(f"  err: {e}")


if __name__ == "__main__":
    _smoke()
