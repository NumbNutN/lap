"""End-to-end dry run using a mock VLM that returns canned valid JSON.

Confirms the pipeline glue (reader → keyframes → prompt builder → parser
→ audit → JSONL writer) is wired correctly without any API call.

    python policy/lap/scripts/annotate_droid/_dryrun.py /tmp/dryrun.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .client_base import VlmReply
from .droid_reader import make_synthetic_bundle
from .runner import run_batch


class MockVlmClient:
    model = "mock-vlm-v0"

    def annotate(self, *, task_instruction, keyframes_meta, keyframe_images):
        # Build a "perfect" reply that matches the keyframe schema.
        out = {
            "plan": (
                f"Goal: {task_instruction} "
                "Steps: 1) approach the object, 2) grasp it, 3) lift, "
                "4) place at target, 5) release."
            ),
            "keyframes": [],
        }
        for i, kf in enumerate(keyframes_meta):
            t = kf["type"]
            grip = kf["gripper_state"]
            if t == "begin":
                stage = "Episode start."
                act = "Begin moving toward the object."
            elif t == "grasp":
                stage = "Reached the object; closing the gripper."
                act = "Close the gripper on the object."
            elif t == "release":
                stage = "Above the target zone; opening to release."
                act = "Open the gripper to release the object."
            elif t == "retry":
                stage = "Previous grasp slipped. Re-positioning and trying again."
                act = "Close the gripper firmly on the object."
                think = "The previous attempt failed because the contact was off-centre."
                out["keyframes"].append({
                    "frame_idx": kf["frame_idx"],
                    "stage": stage,
                    "think": think,
                    "action": act,
                })
                continue
            elif t == "end":
                stage = "Object placed at target. Episode complete."
                act = "Retract the gripper."
            elif t == "motion":
                stage = "Transporting between phases."
                act = "Move the arm toward the next phase."
            elif t == "filler":
                stage = "Mid-stage anchor between transitions."
                act = "Continue toward the next phase."
            else:
                stage = "(unknown)"
                act = "Continue."
            out["keyframes"].append({
                "frame_idx": kf["frame_idx"],
                "stage": stage,
                "think": None,
                "action": act,
            })
        return VlmReply(
            text=json.dumps(out, ensure_ascii=False, indent=2),
            latency_s=0.001,
            input_tokens=100,
            output_tokens=200,
            model=self.model,
        )


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dryrun.jsonl"
    Path(out_path).unlink(missing_ok=True)
    bundles = [make_synthetic_bundle() for _ in range(3)]
    # Synthetic bundles all share the same episode_id — resume would collapse
    # them; differentiate so we exercise the full pipeline.
    for i, b in enumerate(bundles):
        b.episode_id = f"synthetic_{i:03d}"

    counts = run_batch(bundles, MockVlmClient(), output_jsonl=out_path, resume=False)
    print(f"counts={counts}")

    # Read it back to confirm round-trip works.
    with open(out_path) as f:
        lines = [line for line in f if line.strip()]
    print(f"wrote {len(lines)} lines to {out_path}")
    from .schema import EpisodeAnnotation
    for i, line in enumerate(lines):
        ann = EpisodeAnnotation.from_jsonl_line(line)
        print(
            f"  ep[{i}] id={ann.episode_id} "
            f"kf={len(ann.keyframes)} pass={ann.audit.passed} "
            f"plan='{ann.plan[:60]}...'"
        )
        if not ann.audit.passed:
            print(f"    errors: {ann.audit.errors}")
        if ann.audit.warnings:
            print(f"    warnings: {ann.audit.warnings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
