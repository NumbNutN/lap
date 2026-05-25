"""Output schema for DROID embodied-CoT annotation.

Two layers:

- :class:`KeyframeAnnotation` / :class:`EpisodeAnnotation` — typed
  in-memory representation we serialise to JSONL on disk.
- :func:`parse_vlm_output` — robust parser of the VLM's free-form JSON
  reply, tolerant of common deviations (markdown fence, trailing prose,
  missing optional fields).

The on-disk JSONL line format (one episode per line)::

    {
      "episode_id": "<droid file_path or hash>",
      "task_instruction": "...",
      "fps": 15.0,
      "n_frames": 173,
      "keyframe_indices": [0, 38, 52, 85, 110, 124],
      "keyframe_types":   ["begin", "motion", "grasp", "motion", "release", "end"],
      "plan": "...",
      "keyframes": [
        {"frame_idx": 0, "stage": "...", "think": null, "action": "..."},
        ...
      ],
      "audit": {
        "pass": true,
        "errors": [],
        "warnings": ["motion change suspiciously dense"]
      },
      "vlm": {
        "model": "gemini-2.5-pro",
        "latency_s": 12.4,
        "prompt_version": "v0.1"
      },
      "raw_output": "<optional, the unparsed VLM string for debugging>"
    }
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass
class KeyframeAnnotation:
    frame_idx: int
    stage: str
    action: str
    think: str | None = None
    # Optionally emitted by the VLM (no-types prompt mode). When the
    # types-fed prompt is used these come back via the episode-level
    # `keyframe_types` array instead. Keep None when source is detector.
    type: str | None = None
    gripper_state: str | None = None


@dataclass
class AuditReport:
    """Result of running audit.py over a parsed annotation."""
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class VlmMeta:
    model: str
    prompt_version: str
    latency_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class EpisodeAnnotation:
    episode_id: str
    task_instruction: str
    fps: float
    n_frames: int
    keyframe_indices: list[int]
    keyframe_types: list[str]
    plan: str
    keyframes: list[KeyframeAnnotation]
    audit: AuditReport = field(default_factory=AuditReport)
    vlm: VlmMeta | None = None
    raw_output: str | None = None

    def to_jsonl_line(self) -> str:
        d = asdict(self)
        # Drop raw_output if None to keep lines smaller; keep when set
        # for debugging the failure cases.
        if self.raw_output is None:
            d.pop("raw_output", None)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl_line(cls, line: str) -> "EpisodeAnnotation":
        d = json.loads(line)
        d["keyframes"] = [KeyframeAnnotation(**kf) for kf in d.get("keyframes", [])]
        audit_d = d.pop("audit", {})
        ann = cls(**d)
        ann.audit = AuditReport(**audit_d)
        return ann


# ---------------------------------------------------------------------------
# Robust parser for VLM output
# ---------------------------------------------------------------------------


_MD_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class VlmOutputParseError(ValueError):
    """The VLM reply could not be parsed into our schema."""


def _strip_markdown_fence(s: str) -> str:
    """If the VLM wrapped the JSON in ```json ... ```, extract the inside."""
    m = _MD_FENCE_RE.search(s)
    if m:
        return m.group(1)
    return s


def _find_outer_object(s: str) -> str:
    """Return the substring from the first '{' to its matching '}' inclusive.

    Handles cases where the VLM appended prose after the JSON (commentary)
    or prefixed it with a header line. Naive brace-counting; trips on
    string-literal braces which is acceptable for our schema (no such braces).
    """
    start = s.find("{")
    if start < 0:
        raise VlmOutputParseError("no '{' found in VLM output")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    raise VlmOutputParseError("unbalanced braces in VLM output")


def parse_vlm_output(text: str) -> tuple[str, list[KeyframeAnnotation]]:
    """Parse the VLM reply into (plan, keyframes).

    Raises VlmOutputParseError with a short reason if the reply can't be
    coerced into the schema. Caller should record the raw output for
    debugging when this raises.
    """
    if not text or not text.strip():
        raise VlmOutputParseError("empty VLM output")

    candidate = _strip_markdown_fence(text)
    candidate = _find_outer_object(candidate)
    try:
        d = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise VlmOutputParseError(f"JSON decode failed: {e.msg}") from e

    if not isinstance(d, dict):
        raise VlmOutputParseError("top-level JSON is not an object")
    if "plan" not in d or "keyframes" not in d:
        raise VlmOutputParseError("missing required keys 'plan' / 'keyframes'")
    if not isinstance(d["keyframes"], list):
        raise VlmOutputParseError("'keyframes' is not a list")

    plan = str(d["plan"]).strip()
    keyframes: list[KeyframeAnnotation] = []
    for i, raw in enumerate(d["keyframes"]):
        if not isinstance(raw, dict):
            raise VlmOutputParseError(f"keyframes[{i}] is not an object")
        try:
            kf = KeyframeAnnotation(
                frame_idx=int(raw["frame_idx"]),
                stage=str(raw["stage"]).strip(),
                action=str(raw["action"]).strip(),
                think=(
                    None
                    if raw.get("think") in (None, "", "null")
                    else str(raw["think"]).strip()
                ),
                # Optional in types-fed mode (None) — populated in no-types mode.
                type=(str(raw["type"]).strip() if raw.get("type") else None),
                gripper_state=(
                    str(raw["gripper_state"]).strip() if raw.get("gripper_state") else None
                ),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise VlmOutputParseError(f"keyframes[{i}] malformed: {e}") from e
        keyframes.append(kf)
    return plan, keyframes


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke() -> None:
    sample = """```json
{
  "plan": "Move the cup onto the saucer.",
  "keyframes": [
    {"frame_idx": 0, "stage": "Start", "think": null, "action": "Approach the cup."},
    {"frame_idx": 50, "stage": "Grasping", "think": "tilted handle requires side approach", "action": "Close on the cup handle."}
  ]
}
```
trailing prose ignored
"""
    plan, kfs = parse_vlm_output(sample)
    assert plan.startswith("Move the cup")
    assert len(kfs) == 2
    assert kfs[1].think and "side approach" in kfs[1].think
    print(f"parse OK — plan={plan!r}, n_kf={len(kfs)}")


if __name__ == "__main__":
    _smoke()
