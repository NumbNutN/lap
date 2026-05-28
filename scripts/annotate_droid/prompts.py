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
  2. A list of KEYFRAMES — each with:
     - frame_idx: integer frame number
     - type: from a rule-based gripper detector (begin/grasp/release/
       retry/motion/filler/end)
     - gripper_state: open / partial / closed
     - an external camera image (+ optional wrist image)
     - **Interaction context tag** (when present): `pre_grasp`,
       `pre_release`, `post_grasp`, `post_release` — indicates this
       keyframe is near a grasp/release event.
     - **Pose deltas** — one or two lines depending on context:

       For keyframes tagged `pre_grasp` or `pre_release`, you get TWO:
         gap-to-grasp: Δxyz=(...) Δrot=...   ← distance from HERE to
             the grasp/release EE pose. Use this in **stage** to
             describe how far the gripper is from the target config.
         next-step:    Δxyz=(...) Δrot=...   ← what the human demo
             does in the next segment. Use this in **action** to
             describe what motion to execute (this IS the demonstration).

       For all other keyframes (transport, retract, begin, end):
         A single forward-step delta (motion to the next keyframe).
         This is a sample of continuous motion, not a commanded
         waypoint — use it to identify the dominant direction and
         approximate magnitude, not to copy exact numbers.

       Rotation format: single-axis rotations show e.g. "11° pitch".
       Multi-axis rotations are decomposed, e.g. "≈8° pitch+5° yaw"
       (the top 2 contributing axes).

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
image-invisible context: past failures, counters, \
cross-keyframe causality. Examples of GOOD stage content:\\n\
  - 'Having released the marker into the pot, the gripper retracts \
upward.'\\n\
  - 'The gripper failed to grasp the marker on the first attempt.'\\n\
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
gripper'.>"
    },
    ...
  ]
}

STAGE STYLE GUIDE:

    Think about what 2-3 things most important contribute the frame's state.
    Some perspectives:
    1. SUBJECT + CURRENT POSITION (relative to scene, landmarks, camera view, etc. ) find the key relative position best describe the state.
    2. TARGET STATE - what configuration the gripper is trying to reach if the gripper in contact-rich fine-tuning phase, could use precise gap-to-target pose deltas here.
    3. Environmental CONSTRAINTS - Carefully find out if the gripper is navigating around an obstacle, or if a nearby object limits the approach angle, name it.

    Some spcific perspectives example:
    - spatial landmarks: height relative to scene ("just above the counter", "at shelf level", "risen past the sink rim")
    - gripper orientation state ("jaws perpendicular to counter",
              "opening facing down toward the marker")
    - object interaction state ("candy bar secured in gripper",
              "marker resting in the pot")
    - state changes ("The bottle next to it was hit by the gripper.")

    A good stage is specific enough that a reader could identify WHICH keyframe it belongs to without seeing the frame index. 

    Reference:
    FAR from target (approach/transport phase):
        Describe the stage with rough direction if there's no precise spatial relationship because:
         - not a contact-rich keyframe (grasp/release/pre/post interaction), OR
         - not aligned with the target yet

        precise pose from gap-to-grasp or next-step are only recomanded when the gripper need to make a precise adjustment during the next action. Otherwise just give a rough direction.

        ✓ "Gripper in the upper-right of the frame, the bottle cluster is located to the left front" (action expert may know what to do next based on this state description)
        ✓ "Gripper moved across the sink, about 5cm to the left front of the brush"
        ✗ "+2.1 cm right, +1.6 cm back, +7.7 cm up with 13° yaw" (raw delta dump — You are not describing the state here)

    NEAR target (contact-rich fine-tune):
        ✓ "The gripper needs 39° more pitch to be perpendicular to the counter."
        ✓ "The gripper hovers 2 cm above the candy bar, ready to close." (gap-to-target is good here, informatively describes the state, and also help the action expert know what to do )
        ✗ "The gripper is almost aligned with the cube" (ambiguous - how close? Too rough for fine-tuning phase, not informative for action expert to know what to do next)

    NOT ZERO FILLER. These phrases are considered to carry no
        positional information if they are the only content describing the stage:
        ✗ "Position fingers around object"
        ✗ "Prepare to re-engage object"
        ✗ "Begin grasp closure"
        Replace with WHAT specifically: direction, distance, object part.


ACTION STYLE GUIDE:

    GRASP / RELEASE / RETRY keyframes:
        The grip verb (close / open / release / re-grasp / pick) takes PRIORITY over other.
        ✓ "Close the gripper to grasp the marker while pitching down 4°"

    transport / retract motion keyframes

    There're two style reference

    1. The movement has a clear relative goal
    It's welcome to use the Δnext-step to describe action, use axis-aware vocabulary with numbers.
    ✓ "leftward 5.6cm, forward 3.6cm and lower 2.1cm, yawing counterclockwise 13° to align the handle of the mug with the gripper jaws"

    2. The gripper is moving FREELY between scene regions - precise pose is not critical, the keyframe is just a sample of continuous motion, you just describe the trend direction, like "move forward and upward with yaw clockwise to approach the bottle cluster"
    ✓ "Lift the candy bar up to the shelf level"
    ✓ "Head towards 9 o'clock in camera view"
    
    Avoid descrption without direction, reference, or numbers:
    You MAY name an axis without a number ("lift while yawing
    toward the pot"). 
    ✗ "Approach the candy on the table" (which direction? how to approach?) It just like repeating the plan without any information about the current state or how to move.


    interaction / fine-tuning motion keyframes

      The gripper is APPROACHING a target object for interaction, OR
      making precise corrections to align with it. This includes:
        - the 1-3 keyframes immediately before a grasp/release
        - any keyframe where the gripper is adjusting its ORIENTATION
          to match the target (e.g. rotating to be perpendicular)
        - any keyframe where a small position correction brings the
          gripper closer to a precise interaction point
    
      For pre_grasp / pre_release keyframes, two deltas are provided:
        - gap-to-interaction: use in stage to describe distance to target
        - next-step: use in action to describe the demo's next motion

      Use AXIS-AWARE vocabulary with numbers from next-step. These
      numbers describe the actual human demonstration motion.

      **Pre-grasp** (tag `pre_grasp`):
          ✓ "Right 1.7 cm, pitch forward 2° to align jaws with bottle neck"
          ✓ "Lower 3 cm to reach the block; approach from the left to clear adjacent piece"
          ✗ "Position fingers around object and begin grasp closure" (ZERO information)
          ✗ "Survey scene and begin approach" (filler phrase)

      **Pre-release** (tag `pre_release`):
          ✓ "Lower 5 cm to seat the marker just above the pot opening"
          ✓ "Forward 2 cm to center above the bowl rim before releasing"
          ✗ "Prepare to release the object" (filler)

      **Post-grasp / post-release** (`post_grasp` / `post_release`):
          ✓ "Lift 2 cm to clear the table with the grasped cube"
          ✓ "Retract 3 cm above the pot rim"

THINK STYLE GUIDANCE:

        [think] only happen in the case you think there's something is non-obvious to perform the next action.
        - Retry: You just failed a attempt and try again, inaccurate alignment before grasp or place the object; place at wrong place; the object slipped during lifting so you should adjust the grip and try again; etc.
        - multi-step planning decision: You should first do something not directly related to the target object to make the subsequent interaction easier, like "picking leftmost first to free space for the right one", "moving the bowl out of the way before placing the block on the table", etc.
        - obstacle / orientation choice: You notice these's risk to collide with something if you approach from one direction, or you need to orient the gripper in a certain way to clear the obstacle.

        DO NOT fill [think] for regular case, it's obvious to do "lower 5 cm to reach the block" when you are 5 cm above the block, no need to say "the gripper is 5 cm above the block so lower 5 cm to reach it". But if there's an obstacle near the block and you need to approach from a certain direction, then you should fill [think] with "approach from the left to clear the adjacent bowl".


ADDITIONAL GUIDANCE:

    AFFORDANCE at grasp/pre-grasp: describe WHICH PART of the
        object the gripper targets:
        ✓ "Align jaws with the narrow neck of the bottle"
        ✓ "Grip the mug handle from the right side"
        ✓ "Approach the front half of the yellow block — rear half
            is blocked by the adjacent blue piece"
        ✗ "Position fingers around target block" (which part? why?)
        ✗ "Gripper surrounding yellow mug body" (where on the body?)

    SPATIAL CONSTRAINTS from nearby objects. If something limits
        the approach angle or gripper orientation, name it:
        ✓ "Approach from the left to clear the adjacent LEGO brick"
        ✓ "Grip the bottle at the 2/3 height mark, avoiding the cap"
        ✗ "Arm hovering near the microwave zone" (is the microwave
            actually constraining the motion? if not, don't mention it)

    WHICH SUBJECTIVE DIRECTION
    When describing direction, you have scene landmarks and bare
    move left/right/forward/backward:
      Scene landmarks style:  "away from the shelf", "over the counter edge"
      Bare move style: "Head towards 9 o'clock"
    When using left/right, clearly tell it refers to the **camera's point
    of view** (what appears left/right in the image)or the main view to avoid ambiguity. This is consistent within one episode but may differ across episodes filmed from
    different angles. 


HARD RULES:
  R1. keyframes length = input keyframe count, in same order, same frame_idx.

  R2. mode_marker MUST be "[think_act]" on every keyframe (downstream
      training will sample sub-modes).

  R3. stage could contain image-invisible context but could infer due to causality across keyframes
      (done plan progress / past event / counter / cross-step causality).

  R4. Reference length: Plan ≤ 5 sentences. Stage ≤ 40 words. Action ≤ 20 words.


    R5. Δ semantics — see "You receive" section above.
        Pre-interaction keyframes get TWO deltas:
            - gap-to-interaction → for stage (state description)
            - next-step → for action (demo motion to imitate)
        Other keyframes get one forward-step delta → for action direction.


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


# ---------------------------------------------------------------------------
# v4.1 fewshot — REAL episode (ep00 "Put the marker in the pot") grounded
# in actual DROID images. This is not fabricated — every stage description
# matches what is visible in the extracted keyframe JPEGs.
#
# Scene: wooden table in a lab, small stainless pot center-right, thin
# cylindrical marker lying on the table surface (barely visible due to
# small size), wooden box as backdrop, lab clutter around.
# ---------------------------------------------------------------------------

FEWSHOT_V3_USER = {
    "task_instruction": "Put the marker in the pot",
    "keyframes_meta": [
        # begin — single forward-step delta
        {"frame_idx": 0,   "type": "begin",   "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+1.3cm,+1.8cm,-3.6cm)  Δrot≈2° pitch"},
        # transport approach — single forward-step
        {"frame_idx": 11,  "type": "motion",  "gripper_state": "open",
         "pose_delta_str": "Δxyz=(+15.3cm,+5.2cm,-20.7cm)  Δrot≈9° pitch+5° yaw"},
        # pre_grasp — DUAL delta: gap-to-grasp + next-step
        {"frame_idx": 27,  "type": "motion",  "gripper_state": "open",
         "near_interaction": True, "interaction_context": "pre_grasp",
         "pose_delta_str": "gap-to-grasp: Δxyz=(+6.7cm,-0.4cm,-16.0cm)  Δrot=14° pitch\n    next-step: Δxyz=(+3.9cm,-0.5cm,-5.3cm)  Δrot≈3° pitch+1° yaw"},
        {"frame_idx": 35,  "type": "motion",  "gripper_state": "open",
         "near_interaction": True, "interaction_context": "pre_grasp",
         "pose_delta_str": "gap-to-grasp: Δxyz=(+2.8cm,+0.1cm,-10.7cm)  Δrot=11° pitch\n    next-step: Δxyz=(+2.8cm,+0.1cm,-10.7cm)  Δrot=11° pitch"},
        # grasp — single delta
        {"frame_idx": 59,  "type": "grasp",   "gripper_state": "open",
         "near_interaction": True,
         "pose_delta_str": "Δxyz=(+0.5cm,+1.0cm,-0.1cm)  Δrot≈0°"},
        # post_grasp — single forward-step
        {"frame_idx": 67,  "type": "motion",  "gripper_state": "closed",
         "near_interaction": True, "interaction_context": "post_grasp",
         "pose_delta_str": "Δxyz=(+0.6cm,-0.2cm,+0.7cm)  Δrot=1° -yaw"},
        # transport — single forward-step
        {"frame_idx": 75,  "type": "motion",  "gripper_state": "closed",
         "pose_delta_str": "Δxyz=(-0.5cm,-1.4cm,+11.2cm)  Δrot=12° -yaw"},
        # pre_release — DUAL delta
        {"frame_idx": 87,  "type": "motion",  "gripper_state": "closed",
         "near_interaction": True, "interaction_context": "pre_release",
         "pose_delta_str": "gap-to-release: Δxyz=(-11.6cm,-9.9cm,-5.1cm)  Δrot≈12° yaw+6° roll\n    next-step: Δxyz=(-5.0cm,-5.9cm,-0.6cm)  Δrot≈8° yaw+4° roll"},
        {"frame_idx": 97,  "type": "motion",  "gripper_state": "closed",
         "near_interaction": True, "interaction_context": "pre_release",
         "pose_delta_str": "gap-to-release: Δxyz=(-6.6cm,-4.0cm,-4.5cm)  Δrot≈7° roll+3° yaw\n    next-step: Δxyz=(-6.6cm,-4.0cm,-4.5cm)  Δrot≈7° roll+3° yaw"},
        # release — single delta
        {"frame_idx": 134, "type": "release", "gripper_state": "closed",
         "near_interaction": True,
         "pose_delta_str": "Δxyz=(-0.0cm,+0.2cm,-0.1cm)  Δrot≈0°"},
        # post_release — single forward-step
        {"frame_idx": 145, "type": "motion",  "gripper_state": "open",
         "near_interaction": True, "interaction_context": "post_release",
         "pose_delta_str": "Δxyz=(-2.6cm,+3.2cm,+5.3cm)  Δrot≈3° pitch+2° yaw"},
        # end — no delta
        {"frame_idx": 165, "type": "end",     "gripper_state": "open",
         "pose_delta_str": ""},
    ],
}


FEWSHOT_V3_ASSISTANT = {
    "plan": (
        "Put the marker in the pot. 1) Approach the marker on the wooden "
        "table from the rear-left. 2) Lower and pitch to align jaws with "
        "the thin cylindrical marker. 3) Grasp the marker barrel. "
        "4) Lift clear of the table and arc toward the pot. "
        "5) Lower into the pot opening and release. 6) Retract."
    ),
    "keyframes": [
        {
            "frame_idx": 0,
            "mode_marker": "[think_act]",
            "stage": "Gripper at home pose, rear-left of scene, well above the wooden table. Pot sits center-right; marker is a thin dark cylinder lying flat on the table near the pot.",
            "think": None,
            "action": "Head toward the table center, descending toward the marker.",
        },
        {
            "frame_idx": 11,
            "mode_marker": "[think_act]",
            "stage": "Gripper entering the frame from upper-left, about 20 cm above the table, sweeping forward-right toward the marker region.",
            "think": None,
            "action": "Continue steep descent toward the table surface near the marker.",
        },
        {
            "frame_idx": 27,
            "mode_marker": "[think_act]",
            # stage uses gap-to-grasp: "16 cm above + 14° pitch remaining"
            "stage": "Gripper at table height, 16 cm above and 14° pitch from the grasp pose on the marker barrel. Jaws need near-vertical orientation for the flat-lying marker.",
            "think": "Marker is thin (~1 cm) and lying flat — top-down barrel grasp along its length requires near-vertical pitch.",
            # action uses next-step: "lower 5 cm, pitch 3°" (not the full gap)
            "action": "Lower 5 cm and pitch 3° toward the marker.",
        },
        {
            "frame_idx": 35,
            "mode_marker": "[think_act]",
            # stage uses gap-to-grasp: "11 cm + 11° remaining"
            "stage": "Gripper just above the marker, 11 cm and 11° pitch remaining to the grasp pose. Jaws open, nearly vertical.",
            "think": None,
            # action uses next-step (here gap=next-step, last pre-grasp frame)
            "action": "Lower 11 cm and pitch 11° to reach the marker.",
        },
        {
            "frame_idx": 59,
            "mode_marker": "[think_act]",
            "stage": "Jaws bracketing the marker barrel at table level. Near-zero gap — ready to close.",
            "think": None,
            "action": "Close the gripper on the marker barrel.",
        },
        {
            "frame_idx": 67,
            "mode_marker": "[think_act]",
            "stage": "Marker secured in the gripper at table height. Beginning to lift clear of the surface.",
            "think": None,
            "action": "Lift 1 cm to clear the table surface with the marker.",
        },
        {
            "frame_idx": 75,
            "mode_marker": "[think_act]",
            "stage": "Gripper has risen about 11 cm above the table, marker in hand. Pot is below and to the right, about 12 cm away laterally.",
            "think": None,
            "action": "Arc rightward and yaw CW toward the pot opening.",
        },
        {
            "frame_idx": 87,
            "mode_marker": "[think_act]",
            # stage uses gap-to-release: "12 cm left, 5 cm above release pose"
            "stage": "Gripper carrying marker above the pot, 12 cm left and 5 cm above the release pose inside the pot opening.",
            "think": None,
            # action uses next-step: "5 cm right + 1 cm lower" (one segment, not full gap)
            "action": "Shift 5 cm right and lower 1 cm, yaw 8° CW toward the pot.",
        },
        {
            "frame_idx": 97,
            "mode_marker": "[think_act]",
            # stage uses gap-to-release: "7 cm left, 5 cm above"
            "stage": "Marker just above the pot rim, 7 cm left and 5 cm above the release point. Nearly centered.",
            "think": None,
            # action uses next-step (here gap=next-step, last pre-release)
            "action": "Lower 5 cm and shift 7 cm right to center in the pot.",
        },
        {
            "frame_idx": 134,
            "mode_marker": "[think_act]",
            "stage": "Marker inside the pot at the release position. Gripper opens to drop it.",
            "think": None,
            "action": "Open the gripper to release the marker into the pot.",
        },
        {
            "frame_idx": 145,
            "mode_marker": "[think_act]",
            "stage": "Marker resting inside the pot. Gripper open, retracting upward and leftward away from the pot.",
            "think": None,
            "action": "Retract upward 5 cm and back away from the pot.",
        },
        {
            "frame_idx": 165,
            "mode_marker": "[think_act]",
            "stage": "Gripper at rest above the table, task complete. Marker visible inside the pot.",
            "think": None,
            "action": "Hold at the rest pose.",
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
            "stage": "Episode start. The gripper is open. A watermelon is on the back and far from the watermelon.",
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
        if memory_augmented:
            if kf.get("near_interaction"):
                ctx = kf.get("interaction_context", "")
                if ctx:
                    bits.append(f"**TIER_B:{ctx}**")
                else:
                    bits.append("**TIER_B**")
            if kf.get("pose_delta_str"):
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
