"""Gemini 2.5 Pro client.

Uses Google's new ``google-genai`` SDK (the successor to
``google-generativeai``). Install with::

    uv pip install google-genai pillow

Auth:
  - GOOGLE_API_KEY env var (preferred), or
  - GOOGLE_APPLICATION_CREDENTIALS for Vertex AI service account.

Pricing notes (2026Q1, subject to change — re-check before bulk run):
  ~ $1.25 / M input tokens, $10 / M output tokens; ~ 258 tokens per image.

Per-episode budget (10 keyframes, JSON output ~ 800 tokens):
  input  ≈ 2.6k tokens (system + images + metadata)
  output ≈ 0.8k tokens
  cost   ≈ $0.012  (input) + $0.008 (output) = ~$0.02
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .client_base import VlmClient
from .client_base import VlmReply
from .prompts import build_gemini_contents


class GeminiClient:
    """Wrapper around google-genai for Gemini 2.5 Pro."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
        request_timeout_s: float = 180.0,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise RuntimeError(
                "google-genai SDK required for GeminiClient. "
                "Install with: uv pip install google-genai"
            ) from e
        self._types = genai_types
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY env var (or api_key kwarg) is required for GeminiClient."
            )
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.request_timeout_s = request_timeout_s
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def annotate(
        self,
        *,
        task_instruction: str,
        keyframes_meta: list[dict],
        keyframe_images: list[np.ndarray],
    ) -> VlmReply:
        system_text, contents = build_gemini_contents(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            keyframe_images=keyframe_images,
            include_fewshot=True,
        )

        # GenerateContentConfig: ask for JSON output (response_mime_type).
        config: dict[str, Any] = dict(
            system_instruction=system_text,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
        )
        gen_config = self._types.GenerateContentConfig(**config)

        t0 = time.monotonic()
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gen_config,
        )
        latency = time.monotonic() - t0

        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", None) if usage else None
        out_tok = getattr(usage, "candidates_token_count", None) if usage else None

        return VlmReply(
            text=text,
            latency_s=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.model,
        )
