"""Single-episode + batch orchestrator.

Pulls together keyframe detection + image loading + VLM client + parser
+ audit, and emits one ``EpisodeAnnotation`` per episode.

The runner is provider-agnostic: it accepts any :class:`VlmClient` and
delegates the actual VLM call to it. Both ``annotate_droid_qwen.py`` and
``annotate_droid_gemini.py`` wire the appropriate client and call
:func:`run_batch`.

Output format: JSONL, one annotation per line, ready for downstream
consumption by ``DROIDCoTDataset`` (TBD, sister to ``BridgeECoTDataset``).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
import logging
import time
from pathlib import Path

import numpy as np

from .audit import audit_episode
from .client_base import VlmClient
from .droid_reader import EpisodeBundle
from .keyframe import Keyframe
from .keyframe import detect_keyframes
from .schema import EpisodeAnnotation
from .schema import VlmMeta
from .schema import VlmOutputParseError
from .schema import parse_vlm_output

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v0.1"


def annotate_episode(
    bundle: EpisodeBundle,
    client: VlmClient,
    *,
    max_retries: int = 2,
    save_raw_on_fail: bool = True,
    feed_types: bool = True,
    memory_augmented: bool = False,
) -> EpisodeAnnotation:
    """Annotate one episode end-to-end.

    When ``feed_types`` is False, the VLM is given ONLY the frame index
    of each keyframe (no type/gripper hint from our detector) and is
    expected to derive type+gripper_state from images itself.

    When ``memory_augmented`` is True (v3 prompt mode), per-keyframe pose
    deltas (cm + axis name) are computed from ``bundle.ee_pose`` and
    formatted into the user message so the VLM can emit finer-grained
    axis-aware actions. Implies feed_types=True. The model is also told
    its earlier outputs serve as memory chain for later keyframes.

    Catches all VLM/parse exceptions and returns an EpisodeAnnotation
    with audit.passed=False rather than raising — so a batch run can
    continue past individual failures.
    """
    # 1. Detect keyframes
    keyframes: list[Keyframe] = detect_keyframes(
        gripper_width=bundle.gripper_width,
        ee_pos=bundle.ee_pos,
        fps=bundle.fps,
    )
    if not keyframes:
        return _failure(bundle, "no keyframes detected", "")

    # 2. Compute per-keyframe FORWARD pose deltas (this keyframe → next
    #    keyframe) — only when we have full 6-DoF pose data AND the user
    #    opted into memory mode. Forward direction is intentional: the
    #    `action` field describes the motion the robot is ABOUT TO
    #    execute, so the delta we surface in the prompt should describe
    #    the same upcoming segment. Last keyframe has Δ=zero (nothing
    #    after it).
    pose_deltas_str: list[str] = ["" for _ in keyframes]
    if memory_augmented and bundle.ee_pose is not None:
        from .pose_utils import pose_delta

        # For TIER_B pre_grasp / pre_release keyframes, compute gap to the
        # ACTUAL interaction pose (the grasp/release keyframe's EE pose),
        # not just the next keyframe. This is the real control target —
        # "how much further until the gripper reaches its grasp configuration"
        # — vs the old forward-delta which was just a sampling step.
        # For all other keyframes, keep forward-delta to next keyframe.
        for i, kf in enumerate(keyframes):
            cur_pose = bundle.ee_pose[kf.t]
            ctx = interaction_context[i]

            if ctx and ctx.startswith("pre_"):
                # Find the target interaction keyframe ahead
                target_type = ctx.split("_", 1)[1]  # "grasp" or "release"
                target_pose = None
                for j in range(i + 1, len(keyframes)):
                    if keyframes[j].type == target_type:
                        target_pose = bundle.ee_pose[keyframes[j].t]
                        break
                if target_pose is not None:
                    d = pose_delta(target_pose, cur_pose)
                    pose_deltas_str[i] = f"gap-to-{target_type}: {d}"
                    continue

            # Default: forward delta to next keyframe
            if i + 1 < len(keyframes):
                next_pose = bundle.ee_pose[keyframes[i + 1].t]
            else:
                next_pose = cur_pose
            d = pose_delta(next_pose, cur_pose)
            pose_deltas_str[i] = str(d)

    # 2b. Tag near-interaction keyframes (within ±2 of grasp/release/retry).
    # The prompt tells the VLM: near_interaction=true → TIER B (fine-tune
    # precision with numbers); false → TIER C (qualitative).
    _INTERACTION_TYPES = {"grasp", "release", "retry"}
    _NEAR_RADIUS = 2
    near_interaction = [False] * len(keyframes)
    # Finer sub-tag: "pre_release" / "pre_grasp" / "post_release" / "post_grasp"
    # so the prompt can give release-approach-specific examples.
    interaction_context: list[str | None] = [None] * len(keyframes)
    if memory_augmented:
        for i, kf in enumerate(keyframes):
            if kf.type in _INTERACTION_TYPES:
                for j in range(max(0, i - _NEAR_RADIUS),
                               min(len(keyframes), i + _NEAR_RADIUS + 1)):
                    near_interaction[j] = True
        # Sub-tag: for each TIER_B keyframe, classify WHY it's near interaction.
        for i, kf in enumerate(keyframes):
            if not near_interaction[i] or kf.type in _INTERACTION_TYPES:
                continue  # only tag the motion frames around interactions
            # Look for the nearest interaction keyframe.
            nearest_inter = None
            nearest_dist = float("inf")
            for j, kf2 in enumerate(keyframes):
                if kf2.type in _INTERACTION_TYPES and abs(i - j) < nearest_dist:
                    nearest_dist = abs(i - j)
                    nearest_inter = (j, kf2.type)
            if nearest_inter is not None:
                j, itype = nearest_inter
                if i < j:
                    interaction_context[i] = f"pre_{itype}"
                else:
                    interaction_context[i] = f"post_{itype}"

    # 3. Build VLM-ready metadata + lazy-load images for selected frames
    keyframes_meta = []
    for i, kf in enumerate(keyframes):
        d = {"frame_idx": kf.t}
        if feed_types or memory_augmented:
            d["type"] = kf.type
            d["gripper_state"] = kf.gripper_state or "unknown"
            if "previous_attempt_frame" in kf.extra:
                d["previous_attempt_frame"] = kf.extra["previous_attempt_frame"]
        if memory_augmented:
            if pose_deltas_str[i]:
                d["pose_delta_str"] = pose_deltas_str[i]
            d["near_interaction"] = near_interaction[i]
            if interaction_context[i]:
                d["interaction_context"] = interaction_context[i]
        keyframes_meta.append(d)

    try:
        keyframe_images = [bundle.frame_loader(kf.t) for kf in keyframes]
    except Exception as e:
        return _failure(bundle, f"frame_loader failed: {e}", "", keyframes=keyframes)

    # 4. Call VLM with retries
    raw_text = ""
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            reply = client.annotate(
                task_instruction=bundle.task_instruction,
                keyframes_meta=keyframes_meta,
                keyframe_images=keyframe_images,
                feed_types=feed_types,
                memory_augmented=memory_augmented,
            )
            raw_text = reply.text
            break
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(
                "[%s] VLM attempt %d/%d failed: %s",
                bundle.episode_id, attempt + 1, max_retries + 1, last_err
            )
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    if not raw_text:
        return _failure(bundle, f"VLM failed after retries: {last_err}", "", keyframes=keyframes)

    # 4. Parse VLM output
    try:
        plan, kf_anns = parse_vlm_output(raw_text)
    except VlmOutputParseError as e:
        return _failure(
            bundle, f"parse failed: {e}", raw_text if save_raw_on_fail else "",
            keyframes=keyframes,
        )

    # 5. Backfill type/gripper_state from detector onto VLM-parsed keyframes.
    # v3 prompt R3 tells the VLM NOT to re-emit these (they come from our
    # detector, not the model). But downstream consumers (viewer, audit,
    # training) expect them per-keyframe. Copy from our detector output.
    for i, kf_ann in enumerate(kf_anns):
        if i < len(keyframes):
            if kf_ann.type is None:
                kf_ann.type = keyframes[i].type
            if kf_ann.gripper_state is None:
                kf_ann.gripper_state = keyframes[i].gripper_state

    # 6. Construct annotation + run audit
    ann = EpisodeAnnotation(
        episode_id=bundle.episode_id,
        task_instruction=bundle.task_instruction,
        fps=bundle.fps,
        n_frames=bundle.n_frames,
        keyframe_indices=[kf.t for kf in keyframes],
        keyframe_types=[kf.type for kf in keyframes],
        plan=plan,
        keyframes=kf_anns,
        vlm=VlmMeta(
            model=reply.model,
            prompt_version=PROMPT_VERSION,
            latency_s=reply.latency_s,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
        ),
    )
    ann.audit = audit_episode(
        ann,
        expected_keyframe_indices=[kf.t for kf in keyframes],
        expected_keyframe_types=[kf.type for kf in keyframes],
        expected_gripper_states=[kf.gripper_state or "unknown" for kf in keyframes],
    )
    # Save raw output only when audit fails — for offline debugging.
    if not ann.audit.passed and save_raw_on_fail:
        ann.raw_output = raw_text
    return ann


def _failure(
    bundle: EpisodeBundle,
    reason: str,
    raw_text: str,
    *,
    keyframes: list[Keyframe] | None = None,
) -> EpisodeAnnotation:
    keyframes = keyframes or []
    ann = EpisodeAnnotation(
        episode_id=bundle.episode_id,
        task_instruction=bundle.task_instruction,
        fps=bundle.fps,
        n_frames=bundle.n_frames,
        keyframe_indices=[kf.t for kf in keyframes],
        keyframe_types=[kf.type for kf in keyframes],
        plan="",
        keyframes=[],
        raw_output=raw_text or None,
    )
    ann.audit.passed = False
    ann.audit.errors.append(reason)
    return ann


def run_batch(
    bundles: Iterable[EpisodeBundle],
    client: VlmClient,
    *,
    output_jsonl: str | Path,
    resume: bool = True,
    flush_every: int = 1,
    feed_types: bool = True,
    memory_augmented: bool = False,
) -> dict[str, int]:
    """Run annotation over many bundles, append to output JSONL.

    Resume semantics: if `resume=True` and the output file exists, the
    set of already-annotated episode_ids is loaded into memory and
    matching bundles are skipped.

    Returns counts: {"emitted": N, "skipped": M, "failed": K}.
    """
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if resume and output_path.exists():
        with output_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ann = EpisodeAnnotation.from_jsonl_line(line)
                    if ann.audit.passed:  # only count passing as done — failed gets retried
                        seen.add(ann.episode_id)
                except Exception:
                    continue
        logger.info("Resume: %d episodes already annotated.", len(seen))

    counts = {"emitted": 0, "skipped": 0, "failed": 0}
    with output_path.open("a") as f:
        for bundle in bundles:
            if bundle.episode_id in seen:
                counts["skipped"] += 1
                continue
            t0 = time.monotonic()
            ann = annotate_episode(
                bundle, client,
                feed_types=feed_types,
                memory_augmented=memory_augmented,
            )
            dt = time.monotonic() - t0
            f.write(ann.to_jsonl_line() + "\n")
            if flush_every == 1:
                f.flush()
            if ann.audit.passed:
                counts["emitted"] += 1
                logger.info(
                    "[ok ] %s  T=%d  kf=%d  %.1fs  warns=%d",
                    bundle.episode_id, bundle.n_frames,
                    len(ann.keyframes), dt, len(ann.audit.warnings),
                )
            else:
                counts["failed"] += 1
                logger.warning(
                    "[FAIL] %s  T=%d  errors=%s",
                    bundle.episode_id, bundle.n_frames, ann.audit.errors[:2],
                )
    return counts
