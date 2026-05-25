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
) -> EpisodeAnnotation:
    """Annotate one episode end-to-end.

    When ``feed_types`` is False, the VLM is given ONLY the frame index
    of each keyframe (no type/gripper hint from our detector) and is
    expected to derive type+gripper_state from images itself.

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

    # 2. Build VLM-ready metadata + lazy-load images for selected frames
    keyframes_meta = []
    for kf in keyframes:
        d = {"frame_idx": kf.t}
        if feed_types:
            d["type"] = kf.type
            d["gripper_state"] = kf.gripper_state or "unknown"
            if "previous_attempt_frame" in kf.extra:
                d["previous_attempt_frame"] = kf.extra["previous_attempt_frame"]
        keyframes_meta.append(d)

    try:
        keyframe_images = [bundle.frame_loader(kf.t) for kf in keyframes]
    except Exception as e:
        return _failure(bundle, f"frame_loader failed: {e}", "", keyframes=keyframes)

    # 3. Call VLM with retries
    raw_text = ""
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            reply = client.annotate(
                task_instruction=bundle.task_instruction,
                keyframes_meta=keyframes_meta,
                keyframe_images=keyframe_images,
                feed_types=feed_types,
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

    # 5. Construct annotation + run audit
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
            ann = annotate_episode(bundle, client, feed_types=feed_types)
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
