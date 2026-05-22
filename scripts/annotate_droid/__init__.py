"""DROID embodied-CoT annotation pipeline.

Subpackage layout::

    keyframe.py       — rule-based keyframe detection
    prompts.py        — system prompt + builder + ECoT fewshot
    schema.py         — output JSON schema + dataclasses
    audit.py          — automated QC over annotator output
    droid_reader.py   — DROID RLDS episode → annotator-ready dict
    client_base.py    — abstract VLM client interface
    client_qwen.py    — Qwen2.5-VL-72B via vLLM OpenAI endpoint
    client_gemini.py  — Gemini 2.5 Pro via google-genai SDK
    runner.py         — single-episode + batch orchestrator
"""

from .keyframe import Keyframe
from .keyframe import detect_keyframes
from .schema import EpisodeAnnotation
from .schema import KeyframeAnnotation

__all__ = [
    "Keyframe",
    "detect_keyframes",
    "EpisodeAnnotation",
    "KeyframeAnnotation",
]
