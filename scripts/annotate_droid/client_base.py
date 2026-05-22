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
    ) -> VlmReply:
        """Annotate one episode. Caller is responsible for retries / parsing."""
        ...
