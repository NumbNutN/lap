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
      "action": "<1 imperative verb phrase, ≤ 12 words. What is the robot \
doing at this keyframe. Examples: 'Lower the gripper onto the red cube', \
'Open the gripper to release the toy', 'Move the gripper forward over the \
plate'. NOT a full sentence with conditions.>"
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
    "keyframes": [
        {
            "frame_idx": 0,
            "stage": "Episode start. The gripper is open and far from the watermelon.",
            "think": None,
            "action": "Begin moving toward the watermelon.",
        },
        {
            "frame_idx": 38,
            "stage": "The gripper has reached the watermelon's vicinity and is preparing to descend.",
            "think": None,
            "action": "Lower the gripper above the watermelon.",
        },
        {
            "frame_idx": 52,
            "stage": "The fingers have closed around the watermelon, ready to lift.",
            "think": None,
            "action": "Close the gripper on the watermelon.",
        },
        {
            "frame_idx": 85,
            "stage": "Carrying the watermelon laterally toward the towel area.",
            "think": None,
            "action": "Move the watermelon over the towel.",
        },
        {
            "frame_idx": 110,
            "stage": "The watermelon is positioned above the towel and the gripper is releasing.",
            "think": None,
            "action": "Open the gripper to place the watermelon on the towel.",
        },
        {
            "frame_idx": 124,
            "stage": "Watermelon placed on the towel. Episode complete.",
            "think": None,
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


def build_user_text(*, task_instruction: str, keyframes_meta: list[dict]) -> str:
    """User-message text portion (separate from images).

    Each provider attaches images differently; this function returns the
    metadata table the VLM sees alongside the images.
    """
    lines = [
        f"TASK: {task_instruction}",
        "",
        f"KEYFRAMES ({len(keyframes_meta)} total, in order):",
    ]
    for i, kf in enumerate(keyframes_meta):
        extra = ""
        if kf.get("previous_attempt_frame") is not None:
            extra = f" (previous failed grasp at frame {kf['previous_attempt_frame']})"
        lines.append(
            f"  [{i}]  frame_idx={kf['frame_idx']:>4d}  type={kf['type']:<8}"
            f"  gripper={kf['gripper_state']}{extra}"
        )
    lines.append("")
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


def build_openai_messages(
    *,
    task_instruction: str,
    keyframes_meta: list[dict],
    keyframe_images: list[np.ndarray],
    include_fewshot: bool = True,
) -> list[dict[str, Any]]:
    """Build OpenAI / vLLM chat-completion messages (used by Qwen client).

    ``keyframe_images[i]`` corresponds to ``keyframes_meta[i]``.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if include_fewshot:
        messages.append({"role": "user", "content": build_fewshot_user_text()})
        messages.append({"role": "assistant", "content": build_fewshot_assistant_text()})

    # Real query — text first, then one image per keyframe in order.
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_user_text(
            task_instruction=task_instruction, keyframes_meta=keyframes_meta
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
) -> tuple[str, list[Any]]:
    """Build Gemini ``contents`` argument. Returns (system_instruction, contents).

    Gemini takes system separately from the contents list. Contents is a
    flat list of alternating user / model turns, where each turn can mix
    text and PIL Image parts.
    """
    contents: list[Any] = []
    if include_fewshot:
        contents.append({"role": "user", "parts": [{"text": build_fewshot_user_text()}]})
        contents.append({"role": "model", "parts": [{"text": build_fewshot_assistant_text()}]})

    user_parts: list[Any] = [
        {"text": build_user_text(
            task_instruction=task_instruction, keyframes_meta=keyframes_meta
        )},
    ]
    for img in keyframe_images:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        user_parts.append(Image.fromarray(img))
    contents.append({"role": "user", "parts": user_parts})
    return SYSTEM_PROMPT, contents
