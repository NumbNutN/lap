"""Abstract VLM client.

Both Qwen-VL and Gemini implementations conform to :class:`VlmClient`,
so the runner is provider-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class VlmReply:
    """One VLM round-trip result."""
    text: str
    """The raw model text response. Schema parser is invoked separately."""

    latency_s: float
    """Wall-clock seconds from request submission to last token."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str = ""


class VlmClient(Protocol):
    """Minimal interface every provider client must implement."""

    model: str

    def annotate(
        self,
        *,
        task_instruction: str,
        keyframes_meta: list[dict],
        keyframe_images: list[np.ndarray],
        feed_types: bool = True,
        memory_augmented: bool = False,
    ) -> VlmReply:
        """Annotate one episode. Caller is responsible for retries / parsing.

        When ``feed_types`` is False, the keyframes_meta omits type and
        gripper_state fields and the system prompt asks the VLM to derive
        these from images.

        When ``memory_augmented`` is True (v3 prompt), keyframes_meta
        carries per-keyframe pose_delta_str and the system prompt asks
        for memory-augmented stage + axis-aware actions + mode_marker.
        Defaults preserve previous behaviour.
        """
        ...
