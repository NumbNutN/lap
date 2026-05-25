"""Qwen2.5-VL-72B client.

Connects to a vLLM-hosted Qwen-VL endpoint that exposes the OpenAI
chat-completions API. We use the ``openai`` SDK because it speaks vLLM
out of the box (vLLM advertises an OpenAI-compatible endpoint).

Spin-up on the H200 host (separate from the annotation worker):

    uv run vllm serve Qwen/Qwen2.5-VL-72B-Instruct \\
        --tensor-parallel-size 2 \\
        --max-model-len 32768 \\
        --limit-mm-per-prompt image=20 \\
        --port 8100

Then point this client at ``http://<h200_host>:8100/v1``.

Why 32768 max-model-len: each keyframe image ≈ 1500-2000 tokens after
Qwen-VL patch encoding; 10 keyframes ≈ 15-20k vision tokens + system
prompt + fewshot. 32k context leaves comfortable headroom.

Why limit-mm-per-prompt=20: ceiling on # images per request, sized for
worst-case keyframe count + 1 fewshot image (if we ever add fewshot
imagery; current fewshot is text-only).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .client_base import VlmClient
from .client_base import VlmReply
from .prompts import build_openai_messages


class QwenVLClient:
    """Wrapper around an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8100/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        request_timeout_s: float = 180.0,
        max_completion_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        # Lazy import — keeps the rest of the package importable without
        # the openai SDK installed.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai SDK required for QwenVLClient. "
                "Install with: uv pip install openai"
            ) from e
        self._OpenAI = OpenAI
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
        feed_types: bool = True,
        memory_augmented: bool = False,
    ) -> VlmReply:
        messages = build_openai_messages(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            keyframe_images=keyframe_images,
            include_fewshot=True,
            feed_types=feed_types,
            memory_augmented=memory_augmented,
        )
        t0 = time.monotonic()
        # response_format=json_object asks Qwen-VL to return strict JSON;
        # vLLM honours this on Qwen-family models via guided decoding.
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        try:
            kwargs["response_format"] = {"type": "json_object"}
            resp = self._client.chat.completions.create(**kwargs)
        except Exception:
            # Some vLLM builds don't accept response_format; retry without.
            kwargs.pop("response_format", None)
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
