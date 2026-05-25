"""System prompt + message builder for DROID embodied-CoT annotation.

This file is the **prompt iteration surface**. Edit `SYSTEM_PROMPT` and the
fewshot examples below; both `client_qwen.py` and `client_gemini.py` import
them unchanged.

Style decisions (locked, see README_cot_annotation_strategy.md §0.A):

- 4 markers: ``[plan]`` ``[stage]`` ``[think]`` (optional) ``[action]``.
- ``[think]`` is a plain text marker, not a special token. Only emitted at
  keyframes where the next move is non-obvious (retry, ambiguous goal,
  spatial reasoning required).
- ``[action]`` is emitted **only at keyframes**, not per frame. Between
  keyframes the model relies on cached cascade context + flow continuation.
- No negative reasoning.
- Output is JSON for downstream parsing reliability.

Fewshot examples are kept short (1 example) — we depend on the system
prompt's instructions + the keyframe-type hints in the user message.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# System prompt — the main iteration surface
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V3_MEMORY = """You annotate embodied chain-of-thought reasoning for a \
single-arm Franka Panda robot manipulation episode from the DROID dataset.

You receive:
  1. Natural-language task instruction.
  2. A list of KEYFRAMES — each with: frame index, type label from a
     rule-based detector (begin/grasp/release/retry/motion/filler/end),
     observed gripper state (open/partial/closed), an external camera
     image, an optional wrist camera image, and a POSE DELTA from the
     previous keyframe formatted as Δxyz (cm) + Δrot (deg around
     yaw/pitch/roll/compound).

You emit per-keyframe annotations as a SEQUENCE (in order), so each
annotation may reference earlier annotations in this same response as
"memory" — that is the whole point of the architecture.

OUTPUT — strictly valid JSON, no markdown fence, no commentary:

{
  "plan": "<2-5 sentences. State the overall goal + inline numbered \
sub-goals. If the trajectory does NOT actually complete the task (early \
truncation, abandoned, etc.), describe what actually happens.>",
  "keyframes": [
    {
      "frame_idx": <int — copy from input>,
      "mode_marker": "[think_act]",
      "stage": "<15-40 words. Describe the current state AND any \
image-invisible context: plan-step progress, past failures, counters, \
cross-keyframe causality. Examples of GOOD stage content:\\n\
  - 'Having released the marker into the pot, the gripper retracts \
upward. Plan step 4 (release) just completed; this is the start of \
the recovery / return phase.'\\n\
  - 'This is the second pair in plan step 2; the first attempt missed.'\\n\
NEVER just describe what is visible ('the gripper is open above the \
table'). NEVER restate the plan verbatim.>",
      "think": "<null OR 1-2 sentences. FILL when one of:\\n\
  - type=retry (REQUIRED — explain failure cause + corrective approach)\\n\
  - multi-step planning decision ('picking leftmost first to free space')\\n\
  - obstacle / orientation choice ('lifting higher to clear the bowl')\\n\
  - reasoning about invisible info ('target is the pot because we are \
holding the marker')\\n\
Target 30-40% of keyframes have non-null think. Routine \
approach/transport keyframes should be null.>",
      "action": "<Imperative phrase, 5-12 words, axis-aware vocabulary. \
Use axis names from the input Δrot when applicable:\\n\
  - 'Yaw counterclockwise slightly to align the gripper with the marker'\\n\
  - 'Tilt downward 8 degrees while lowering 3 cm above the cube'\\n\
  - 'Translate forward 5 cm to approach the candy bar'\\n\
  - 'Close the gripper to grasp the marker firmly'\\n\
Atomic primitives are OK when the move is trivially small: 'Open the \
gripper'. AVOID generic 'move forward / move down' when the input \
Δxyz/Δrot indicates a more specific motion direction.>"
    },
    ...
  ]
}

HARD RULES:

  R1. keyframes length = input keyframe count, in same order, same frame_idx.

  R2. mode_marker MUST be "[think_act]" on every keyframe (downstream
      training will sample sub-modes).

  R3. type & gripper_state are provided in the input — do NOT re-emit them.
      You CAN use them as anchors; the detector is reliable for these.

  R4. stage MUST contain at least one piece of image-invisible context
      (plan progress / past event / counter / cross-step causality). If you
      can't add such context, you are doing it wrong — try harder.

  R5. think non-null on every type=retry keyframe (audit will reject).

  R6 (action vocabulary — three tiers; the key is WHEN to use numbers):

    TIER A — grasp / release / retry keyframes:
      The grip verb (close / open / release / re-grasp / pick) takes
      ABSOLUTE PRIORITY over pose delta in `action`. Always start the
      sentence with the grip verb. You MAY add a pose qualifier in a
      subordinate clause IF it fits within 12 words.
        ✓ "Close the gripper to grasp the marker"
        ✓ "Close the gripper to grasp the marker while pitching down 4°"
        ✗ "Yaw slightly while moving closer to the sink"        (release frame)
        ✗ "Pitch downward and move closer to the cube"           (grasp frame)

    TIER B — INTERACTION / FINE-TUNING motion keyframes
      (the gripper is APPROACHING a target object for interaction, OR
      making precise corrections to align with it. This includes:
        - the 1-3 keyframes immediately before a grasp/release
        - any keyframe where the gripper is adjusting its ORIENTATION
          to match the target (e.g. rotating to be perpendicular)
        - any keyframe where a small position correction brings the
          gripper closer to a precise interaction point
      Use AXIS-AWARE vocabulary with numbers. These numbers are
      meaningful because they describe "how much more motion is needed
      to reach the target configuration" — a useful policy signal.
      "Gap-to-target" phrasing is especially valuable:
          ✓ "Pitch downward 39° to orient jaws perpendicular to the counter"
          ✓ "Lower 2 cm and yaw 5° to align with the cube"
          ✓ "Tilt 8° more to face the marker squarely"
          ✓ "Translate forward 1 cm to seat the gripper on the candy bar"

    TIER C — TRANSPORT / RETRACT motion keyframes
      (the gripper is moving FREELY between scene regions — lifting
      from a table to a shelf, carrying an object across the workspace,
      retracting after release. The per-keyframe Δ here is just one
      sample of continuous motion, not a commanded waypoint.):
        Describe direction + scene-landmark relation (qualitative):
          ✓ "Lift the candy bar up to the shelf level"
          ✓ "Carry the cube leftward over the counter toward the towel"
          ✓ "Retract upward away from the pot"
          ✓ "Ascend steeply from counter height to reach the shelf"
        You MAY name an axis without a number ("lift while yawing
        toward the pot"). Avoid bare numbers like "translate 12 cm"
        — they over-claim intent.
        NEVER write only "adjust the arm" — you must say WHICH
        direction and WHAT scene landmark it's relative to.

    Heuristic for deciding TIER B vs C:
      - Is the robot approaching a target to interact with it? → TIER B
      - Is the next keyframe grasp/release/retry? → TIER B
      - Is the robot adjusting orientation (pitch/yaw/roll) for a
        specific target pose (e.g. becoming perpendicular)? → TIER B
      - Is the robot just covering distance between scene regions
        (lift, carry, retract, return)? → TIER C

  R7. No negative reasoning ("the robot did NOT…"). Describe what is.

  R8. Plan ≤ 5 sentences. Stage ≤ 40 words. Action ≤ 12 words.

  R9. Δxyz / Δrot semantics: the values describe the motion that happens
      BETWEEN this keyframe and the NEXT keyframe (forward-looking, in
      the direction of time). For TIER B (fine-tuning) the Δ is close
      to a commanded waypoint and is informative. For TIER C (transport)
      the Δ is incidental sampling — use it only to pick the dominant
      axis name, not to copy specific numbers into `action`.

  R10 (stage style — CRITICAL for uniqueness & spatial grounding):

    R10a. SPATIAL UNIQUENESS: each `stage` MUST be specific enough that
          a reader could identify WHICH keyframe it belongs to without
          seeing the frame index. Use spatial landmarks:
            - height relative to scene ("just above the counter", "at
              shelf level", "risen past the sink rim")
            - horizontal position ("centered over the pot opening",
              "at the left side of the shelf")
            - gripper orientation state ("jaws perpendicular to counter",
              "opening facing down toward the marker")
            - object interaction state ("candy bar secured in gripper",
              "marker resting in the pot")
          ✗ "The robot continues adjusting its position."  (could be any of 5 keyframes)
          ✗ "The arm swings back and pitches upward, gaining significant height."  (vague — "significant" is not a landmark)
          ✓ "The gripper has risen above the sink rim and is now level with the bottom shelf, carrying the candy bar."  (uniquely identifiable)
          ✓ "Just above the counter, the gripper pitches forward to orient its jaws perpendicular for a top-down grasp."

    R10b. NUMBERS IN STAGE — gap-to-target OK, raw deltas NOT OK.
          Numbers that describe "how far from a target configuration"
          are GOOD because they anchor the state uniquely:
            ✓ "The gripper needs 39° more pitch to be perpendicular to the counter."
            ✓ "The gripper hovers 2 cm above the candy bar, ready to close."
          Numbers that describe raw motion (copied from Δxyz/Δrot) are
          NOT useful in stage — they're sampling artifacts, not state:
            ✗ "Carrying the Doritos 22cm sideways toward the sink basin."
            ✗ "The arm has moved 15cm upward from the table."

    R10c. Plan-step references are OPTIONAL. Prefer scene-relational
          descriptions over plan labels. Use plan-step only when it
          carries image-invisible info (retry count, which-attempt).
          ✗ "Plan step 4 transport phase: the robot moves toward the sink."
          ✓ "Having grasped the Doritos, the gripper rises over the counter edge toward the sink."
"""


SYSTEM_PROMPT_NO_TYPES = """You annotate embodied chain-of-thought reasoning for a \
single-arm Franka Panda robot manipulation episode from the DROID dataset.

You receive:
  1. Natural-language task instruction.
  2. A short list of KEYFRAMES, each as just a frame index + an external
     camera image. Wrist image is optional.

You must derive EVERYTHING from the images alone — including what kind
of event the keyframe is (grasp / release / approach / etc.) and what
the gripper is doing.

OUTPUT — strictly valid JSON, no markdown fence, no commentary outside:

{
  "plan": "<2-5 sentences. Overall goal + inline numbered sub-goals \
(e.g. \\"1) approach the cup, 2) grasp it, 3) place it on the saucer\\"). \
Use concrete object names from the images. If the trajectory you see \
does NOT match the task instruction (e.g. trajectory truncated, or robot \
performs only part of the task), describe what actually happens.>",
  "keyframes": [
    {
      "frame_idx": <int — copy from input>,
      "type": "<one of: begin | grasp | release | retry | motion | end. \
You may also use an open-vocabulary verb if the standard types fit \
poorly (e.g. 'approach', 'transport', 'fine-tune', 'idle', 'push').>",
      "gripper_state": "<open | partial | closed — visually determined>",
      "stage": "<1-3 sentences. Why is this stage happening NOW? What is \
the immediate sub-goal? Use concrete object names visible in the frame. \
Do NOT restate the plan.>",
      "think": "<OPTIONAL or null. Only include for retry-type keyframes \
or genuinely non-obvious decisions.>",
      "action": "<Imperative phrase, OPEN VOCABULARY, ≤ 18 words. Style \
ranges from atomic primitives to richly grounded compound actions:\\n\
  - atomic:    'Close the right gripper'\\n\
  - grounded:  'Approach the red cube on the left'\\n\
  - geometric: 'Rotate the yaw angle to align the gripper on top of the block'\\n\
  - compound:  'Close the gripper and push the block on the right side'\\n\
Pick the level that best describes what is happening — prefer concrete \
object names and spatial qualifiers over generic 'move forward / down'.>"
    },
    ...
  ]
}

HARD RULES the grader will check:

  R1. The "keyframes" array length MUST equal the number of input
      keyframes, in the same order, with the same frame_idx values.

  R2. Determine `type` from VISUAL evidence:
      - "grasp"   — gripper is in the act of closing on an object
      - "release" — gripper is in the act of opening to release
      - "retry"   — gripper closing again after a recent failed grasp
      - "motion"  — arm in motion but gripper state is steady
      - "begin"/"end" — first / last frame of the episode
      - Or an open-vocab verb when the above don't fit.

  R3. The `gripper_state` must match what you SEE:
      - "open" if fingers are clearly apart (nothing between them)
      - "closed" if fingers are touching (or grasping an object firmly)
      - "partial" if mid-transition or holding a thin object

  R4. The [action] for grasp/release keyframes must reference the
      object being grasped/released, by name.

  R5. No negative reasoning. Describe what IS happening.

  R6. If the trajectory does not actually complete the task (gripper
      never reaches target, or task abandoned), say so in the plan
      and in any affected keyframe's [stage]. DO NOT pretend the task
      was completed.

  R7. Plan ≤ 5 sentences. Stage ≤ 3 sentences. Action ≤ 18 words.
"""


SYSTEM_PROMPT = """You annotate embodied chain-of-thought reasoning for a \
single-arm Franka Panda robot manipulation episode from the DROID dataset.

You receive:
  1. Natural-language task instruction.
  2. A short list of KEYFRAMES — each with a frame index, an external camera
     image, a wrist camera image (optional), the gripper state (open /
     partial / closed), and a TYPE tag drawn from a rule-based detector.
  3. Keyframe TYPE legend:
       begin    — first frame of the episode
       grasp    — gripper just closed (about to lift / hold something)
       release  — gripper just opened (just placed / let go of something)
       retry    — gripper closed again shortly after a failed grasp;
                  the previous attempt did not succeed
       motion   — EE velocity direction or speed changed sharply
       filler   — inserted to keep stages short; mid-stage anchor
       end      — last frame of the episode

OUTPUT — strictly valid JSON, no markdown fence, no commentary outside the JSON:

{
  "plan": "<2-5 sentences. State the overall goal in 1 sentence, then \
list the sub-goals as a short numbered sequence inline (e.g. \\"1) approach \
the cup, 2) grasp it, 3) place it on the saucer\\"). Be concrete with \
object names from the scene.>",
  "keyframes": [
    {
      "frame_idx": <int — copy from the input>,
      "stage": "<1-3 sentences. Why is this stage happening *now*? What's \
the immediate sub-goal? Use concrete object names. Do NOT restate the plan.>",
      "think": "<OPTIONAL string or null. Only include for retry-type \
keyframes, or when the next move requires non-obvious reasoning (e.g. \
'the cup is occluded so the gripper must approach from the right'). Skip \
this field (write null) for routine grasp/release/motion keyframes.>",
      "action": "<Imperative phrase, OPEN VOCABULARY, ≤ 18 words. Style \
ranges from atomic primitives to richly grounded compound actions:\n\
  - atomic:      'Close the right gripper'\n\
  - grounded:    'Approach the red cube on the left'\n\
  - geometric:   'Rotate the yaw angle to align the gripper on top of the block'\n\
  - compound:    'Close the gripper and push the block on the right side'\n\
Pick the level that best describes what is happening — prefer concrete \
object names and spatial qualifiers over generic 'move forward / down'. \
Compound actions are OK when naturally coupled (e.g. close + push).>"
    },
    ...
  ]
}

HARD RULES (the grader will check):

  R1. The "keyframes" array length MUST equal the number of input keyframes,
      in the same order, with the same frame_idx values.

  R2. If keyframe TYPE == "grasp", the [action] must describe a closing/
      grasping action, AND the gripper_state at that frame must be closed.

  R3. If keyframe TYPE == "release", the [action] must describe an opening/
      placing/releasing action, AND the gripper_state at that frame must
      be open.

  R4. If keyframe TYPE == "retry", the [think] field MUST be non-null and
      should explain why the previous attempt failed (occlusion, slipped,
      wrong angle, etc.) and what's different this time.

  R5. The [action] in a "grasp" keyframe must reference the same object as
      the [action] in the immediately following "release" keyframe of the
      same cycle. (Same object picked up and put down.)

  R6. Do not invent objects not visible in the images. Use generic shape/
      colour descriptors ("the green can", "the red block") if you cannot
      name the object precisely.

  R7. No negative reasoning. Do not write "the robot did not...", "instead
      of grasping X...", etc. Describe what IS happening only.

  R8. Plan and stage must be short. If you write more than 2 sentences for
      a stage or more than 5 sentences for the plan, you are wrong.

If the task is ambiguous from the images, do your best with the language
instruction and note any uncertainty in the plan field only.
"""


# ---------------------------------------------------------------------------
# Fewshot example (ECoT-style condensed) — single example, single keyframe
# ---------------------------------------------------------------------------

FEWSHOT_USER = {
    "task_instruction": "Put the watermelon on the towel",
    "keyframes_meta": [
        {"frame_idx": 0,   "type": "begin",   "gripper_state": "open"},
        {"frame_idx": 38,  "type": "motion",  "gripper_state": "open"},
        {"frame_idx": 52,  "type": "grasp",   "gripper_state": "closed"},
        {"frame_idx": 85,  "type": "motion",  "gripper_state": "closed"},
        {"frame_idx": 110, "type": "release", "gripper_state": "open"},
        {"frame_idx": 124, "type": "end",     "gripper_state": "open"},
    ],
}


# ---------------------------------------------------------------------------
# v3 fewshot — memory-augmented + axis-aware, demonstrates BOTH small and
# large deltas to nudge the model away from "Adjust position" on small ones.
# ---------------------------------------------------------------------------


FEWSHOT_V3_USER = {
    "task_instruction": "Pick up the blue cube and place it on the towel",
    "keyframes_meta": [
        {"frame_idx": 0,   "type": "begin",   "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+3.2cm,-1.1cm,-1.5cm)  Δrot=4° around -yaw"},
        {"frame_idx": 22,  "type": "motion",  "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+1.0cm,-0.4cm,-6.8cm)  Δrot=2° around pitch"},
        {"frame_idx": 41,  "type": "motion",  "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+0.3cm,-0.2cm,-0.4cm)  Δrot=11° around -yaw"},
        {"frame_idx": 55,  "type": "grasp",   "gripper_state": "closed",
         "pose_delta_str": "Δxyz=(+0.2cm,+0.1cm,+0.0cm)  Δrot=3° around pitch"},
        {"frame_idx": 80,  "type": "motion",  "gripper_state": "closed",
         "pose_delta_str": "Δxyz=(-2.4cm,+4.8cm,+5.6cm)  Δrot=6° around yaw"},
        {"frame_idx": 105, "type": "release", "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+0.5cm,-0.1cm,-0.2cm)  Δrot=1° around pitch"},
        {"frame_idx": 124, "type": "end",     "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+0.0cm,+0.0cm,+0.0cm)  Δrot=0° around yaw"},
    ],
}


FEWSHOT_V3_ASSISTANT = {
    "plan": (
        "The robot picks up the blue cube and places it on the towel. "
        "1) Approach the cube. 2) Lower and align the gripper. "
        "3) Close around the cube. 4) Lift and carry it over the towel. "
        "5) Release. 6) Retract."
    ),
    # Style guide for reader (NOT part of JSON):
    #   - TIER A (grasp/release): grip verb only
    #   - TIER B (approach interaction + fine-tune): precise numbers (gap-to-target)
    #   - TIER C (transport/retract): qualitative + landmark relation
    #   - stage: spatially unique, uses landmarks, gap-to-target numbers OK, raw-delta NO
    "keyframes": [
        {
            "frame_idx": 0,
            "mode_marker": "[think_act]",
            # stage: SPATIALLY UNIQUE — ready pose + relative to table
            "stage": "The gripper starts at a ready pose well above the table, with the blue cube visible near the left edge of the workspace.",
            "think": None,
            # TIER C — transport: qualitative direction + landmark
            "action": "Descend from the ready pose toward the table surface.",
        },
        {
            "frame_idx": 22,
            "mode_marker": "[think_act]",
            # stage: height-relative landmark → uniquely identifiable
            "stage": "The gripper has descended to roughly tabletop height, directly above the blue cube, and is beginning lateral corrections to center over it.",
            "think": None,
            # TIER C→B transition: approaching but not yet fine-tuning
            "action": "Lower toward the cube while centering laterally.",
        },
        {
            "frame_idx": 41,
            "mode_marker": "[think_act]",
            # stage: GAP-TO-TARGET number (39° from perpendicular) = OK
            "stage": "Just above the blue cube, the gripper needs about 11° more yaw to align its jaws with the cube's edge orientation before closing.",
            "think": None,
            # TIER B — pre-grasp fine-tune: precise axis + gap-to-target
            "action": "Yaw counterclockwise 11° to align jaws with the cube edge.",
        },
        {
            "frame_idx": 55,
            "mode_marker": "[think_act]",
            # stage: references prior alignment event
            "stage": "Having aligned its jaws with the cube, the gripper hovers 1 cm above the cube surface, ready to close for a top-down grasp.",
            "think": None,
            # TIER A — grasp: grip verb only
            "action": "Close the gripper to grasp the blue cube.",
        },
        {
            "frame_idx": 80,
            "mode_marker": "[think_act]",
            # stage: uses height landmark (above workspace, towel visible)
            "stage": "With the cube secured, the gripper has risen well above the table and is arcing leftward; the towel is visible below and to the left.",
            "think": None,
            # TIER C — transport: qualitative + landmark
            "action": "Arc leftward over the table toward the towel.",
        },
        {
            "frame_idx": 105,
            "mode_marker": "[think_act]",
            # stage: precise spatial anchor (directly above towel)
            "stage": "The gripper is now directly above the center of the towel, holding the cube at a low height ready for placement.",
            "think": None,
            # TIER A — release: grip verb
            "action": "Open the gripper to release the cube onto the towel.",
        },
        {
            "frame_idx": 124,
            "mode_marker": "[think_act]",
            # stage: final state + scene landmarks confirm task outcome
            "stage": "The cube rests on the towel below. The gripper hovers slightly above with jaws open, task complete.",
            "think": None,
            "action": "Remain at the release pose above the towel.",
        },
    ],
}


def build_fewshot_v3_user_text() -> str:
    return build_user_text(
        task_instruction=FEWSHOT_V3_USER["task_instruction"],
        keyframes_meta=FEWSHOT_V3_USER["keyframes_meta"],
        feed_types=True,
        memory_augmented=True,
    )


def build_fewshot_v3_assistant_text() -> str:
    return json.dumps(FEWSHOT_V3_ASSISTANT, ensure_ascii=False, indent=2)

FEWSHOT_ASSISTANT = {
    "plan": (
        "Move the watermelon onto the towel. Steps: 1) approach the "
        "watermelon, 2) grasp it firmly, 3) lift and carry it over the "
        "towel, 4) place it down and release."
    ),
    # Fewshot intentionally covers the four action styles the system prompt
    # describes (atomic / grounded / geometric / compound), so the VLM
    # sees the full vocabulary range in one example.
    "keyframes": [
        {
            "frame_idx": 0,
            "stage": "Episode start. The gripper is open and far from the watermelon.",
            "think": None,
            # grounded
            "action": "Approach the watermelon on the right side of the counter.",
        },
        {
            "frame_idx": 38,
            "stage": "The gripper has reached the watermelon and is preparing to grasp it.",
            "think": None,
            # geometric
            "action": "Lower the gripper and align the fingers with the watermelon's centre.",
        },
        {
            "frame_idx": 52,
            "stage": "The fingers have closed around the watermelon, ready to lift.",
            "think": None,
            # atomic
            "action": "Close the gripper.",
        },
        {
            "frame_idx": 85,
            "stage": "Carrying the watermelon laterally toward the towel.",
            "think": None,
            # compound
            "action": "Lift the watermelon and carry it to the position above the towel.",
        },
        {
            "frame_idx": 110,
            "stage": "The watermelon is positioned above the towel and the gripper is releasing.",
            "think": None,
            # compound — release + intent
            "action": "Open the gripper to place the watermelon on the towel.",
        },
        {
            "frame_idx": 124,
            "stage": "Watermelon placed on the towel. Episode complete.",
            "think": None,
            # atomic
            "action": "Retract the gripper.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Message builders (provider-agnostic)
# ---------------------------------------------------------------------------


def encode_image_b64(arr: np.ndarray, *, max_side: int = 384, quality: int = 80) -> str:
    """RGB uint8 array → base64 JPEG. Downscale to cap max side."""
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_user_text(
    *,
    task_instruction: str,
    keyframes_meta: list[dict],
    feed_types: bool = True,
    memory_augmented: bool = False,
) -> str:
    """User-message text portion (separate from images).

    Per-keyframe meta dict may include:
      - frame_idx (always)
      - type, gripper_state (when feed_types=True)
      - pose_delta_str (when memory_augmented=True; pre-formatted by runner)
      - previous_attempt_frame (when applicable)
    """
    lines = [
        f"TASK: {task_instruction}",
        "",
        f"KEYFRAMES ({len(keyframes_meta)} total, in order):",
    ]
    for i, kf in enumerate(keyframes_meta):
        bits = [f"frame_idx={kf['frame_idx']:>4d}"]
        if feed_types:
            bits.append(f"type={kf.get('type', '?'):<8}")
            bits.append(f"gripper={kf.get('gripper_state', '?')}")
        if memory_augmented and kf.get("pose_delta_str"):
            bits.append(kf["pose_delta_str"])
        if feed_types and kf.get("previous_attempt_frame") is not None:
            bits.append(f"(previous failed grasp at frame {kf['previous_attempt_frame']})")
        lines.append(f"  [{i}]  " + "  ".join(bits))
    lines.append("")
    if memory_augmented:
        lines.append(
            "Annotate every keyframe in order. Each keyframe's `stage` should "
            "reference relevant prior keyframes (plan-step progress, prior "
            "failures, counters) — your earlier outputs in this same response "
            "are your memory chain. Emit valid JSON only."
        )
    else:
        lines.append("Annotate every keyframe. Emit valid JSON only.")
    return "\n".join(lines)


def build_fewshot_user_text() -> str:
    return build_user_text(
        task_instruction=FEWSHOT_USER["task_instruction"],
        keyframes_meta=FEWSHOT_USER["keyframes_meta"],
    )


def build_fewshot_assistant_text() -> str:
    return json.dumps(FEWSHOT_ASSISTANT, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Provider-specific message constructors
# ---------------------------------------------------------------------------


def _select_system_prompt(feed_types: bool, memory_augmented: bool) -> str:
    if memory_augmented:
        # v3 supersedes both v1 (feed_types) and no-types modes when
        # memory + pose-aware fields are wanted.
        return SYSTEM_PROMPT_V3_MEMORY
    return SYSTEM_PROMPT if feed_types else SYSTEM_PROMPT_NO_TYPES


def build_openai_messages(
    *,
    task_instruction: str,
    keyframes_meta: list[dict],
    keyframe_images: list[np.ndarray],
    include_fewshot: bool = True,
    feed_types: bool = True,
    memory_augmented: bool = False,
) -> list[dict[str, Any]]:
    """Build OpenAI / vLLM chat-completion messages (used by Qwen client).

    ``keyframe_images[i]`` corresponds to ``keyframes_meta[i]``.
    When ``memory_augmented`` is True we use the v3 prompt that asks for
    memory-augmented stage + axis-aware actions + ``mode_marker`` field.
    """
    system_prompt = _select_system_prompt(feed_types, memory_augmented)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    if include_fewshot and feed_types and memory_augmented:
        # v3 fewshot: text-only, demonstrates axis-aware action vocab on
        # BOTH small and large pose deltas (so model doesn't default to
        # "Adjust position" on small ones).
        messages.append({"role": "user", "content": build_fewshot_v3_user_text()})
        messages.append({"role": "assistant", "content": build_fewshot_v3_assistant_text()})
    elif include_fewshot and feed_types:
        messages.append({"role": "user", "content": build_fewshot_user_text()})
        messages.append({"role": "assistant", "content": build_fewshot_assistant_text()})

    # Real query — text first, then one image per keyframe in order.
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_user_text(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            feed_types=feed_types,
            memory_augmented=memory_augmented,
        )},
    ]
    for img in keyframe_images:
        b64 = encode_image_b64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    messages.append({"role": "user", "content": content})
    return messages


def build_gemini_contents(
    *,
    task_instruction: str,
    keyframes_meta: list[dict],
    keyframe_images: list[np.ndarray],
    include_fewshot: bool = True,
    feed_types: bool = True,
    memory_augmented: bool = False,
) -> tuple[str, list[Any]]:
    """Build Gemini ``contents`` argument. Returns (system_instruction, contents)."""
    contents: list[Any] = []
    if include_fewshot and feed_types and memory_augmented:
        contents.append({"role": "user", "parts": [{"text": build_fewshot_v3_user_text()}]})
        contents.append({"role": "model", "parts": [{"text": build_fewshot_v3_assistant_text()}]})
    elif include_fewshot and feed_types:
        contents.append({"role": "user", "parts": [{"text": build_fewshot_user_text()}]})
        contents.append({"role": "model", "parts": [{"text": build_fewshot_assistant_text()}]})

    user_parts: list[Any] = [
        {"text": build_user_text(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            feed_types=feed_types,
            memory_augmented=memory_augmented,
        )},
    ]
    for img in keyframe_images:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        user_parts.append(Image.fromarray(img))
    contents.append({"role": "user", "parts": user_parts})
    system_text = _select_system_prompt(feed_types, memory_augmented)
    return system_text, contents
