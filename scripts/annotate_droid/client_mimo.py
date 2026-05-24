"""MiMo (Xiaomi) VLM client.

MiMo exposes an OpenAI-compatible chat-completions endpoint at
``https://api.xiaomimimo.com/v1``. We reuse the OpenAI SDK plumbing from
:mod:`client_qwen` (same protocol), changing only:

  - base_url and api_key
  - model name (default "mimo-v2.5")
  - max image count per request (TBD by experiment — start at 12)

Reference example (from MiMo docs)::

    client = OpenAI(api_key=os.environ.get("MIMO_API_KEY"),
                    base_url="https://api.xiaomimimo.com/v1")
    resp = client.chat.completions.create(
        model="mimo-v2.5",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": "..."},
        ]}],
        max_completion_tokens=1024,
    )
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .client_base import VlmClient
from .client_base import VlmReply
from .prompts import build_openai_messages


class MiMoClient:
    """Wrapper around MiMo's OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "mimo-v2.5",
        request_timeout_s: float = 300.0,
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK required for MiMoClient. "
                "Install with: uv pip install openai"
            ) from e
        api_key = api_key or os.environ.get("MIMO_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MIMO_API_KEY env var (or api_key kwarg) is required for MiMoClient."
            )
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout_s,
        )
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p

    def annotate(
        self,
        *,
        task_instruction: str,
        keyframes_meta: list[dict],
        keyframe_images: list[np.ndarray],
    ) -> VlmReply:
        messages = build_openai_messages(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            keyframe_images=keyframe_images,
            include_fewshot=True,
        )

        t0 = time.monotonic()
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            # MiMo example uses max_completion_tokens (newer OpenAI naming).
            # Fall back to max_tokens if MiMo rejects it.
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some OpenAI-compatible servers reject response_format /
            # max_completion_tokens / temperature with 4xx — retry with
            # minimal kwargs.
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            resp = self._client.chat.completions.create(**kwargs)
        latency = time.monotonic() - t0

        choice = resp.choices[0]
        text = choice.message.content or ""

        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None

        return VlmReply(
            text=text,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
        )
