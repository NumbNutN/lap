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

  R6. action verb / direction should match the Δxyz/Δrot signal when one
      stands out. Example: if Δrot is "12° yaw", action should mention
      yaw rotation; if Δxyz dominates (e.g. -8cm z = lowering), action
      should mention "lower" / "z descent".

  R7. No negative reasoning ("the robot did NOT…"). Describe what is.

  R8. Plan ≤ 5 sentences. Stage ≤ 40 words. Action ≤ 12 words.
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
    # Skip fewshot for v3 memory-augmented + no-types modes (their schema
    # differs from the v1 fewshot and would bias outputs).
    if include_fewshot and feed_types and not memory_augmented:
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
    if include_fewshot and feed_types and not memory_augmented:
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
