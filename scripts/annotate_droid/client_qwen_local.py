"""Qwen2.5-VL-72B local HF transformers client.

Single-process inference, no vLLM server needed. Use this when the Qwen
weights live next to the annotation script and you have enough GPU memory
to load the model directly.

Hardware sizing for Qwen2.5-VL-72B:
  - fp16:  ~145 GB → needs 2× H100/H200 (with device_map="auto" tensor split)
            or 1× H200 (141GB) is borderline-tight.
  - bf16:  ~145 GB (same)
  - int4:  ~40 GB  → fits on 1× A100-80G with bitsandbytes 4bit (TODO)

Reference (Qwen2.5-VL-72B-Instruct HF model card)::

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(path)

We wrap that into the :class:`VlmClient` protocol so the runner stays
provider-agnostic.

Weight class auto-detection: ``AutoModelForVision2Seq.from_pretrained``
reads the directory's config.json and picks the right Qwen2.5-VL /
Qwen2-VL class automatically. If the directory is Qwen2-VL (older
generation), the class will be ``Qwen2VLForConditionalGeneration``;
if Qwen2.5-VL, ``Qwen2_5_VLForConditionalGeneration``. We don't hard-code
the class name.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from .client_base import VlmClient
from .client_base import VlmReply
from .prompts import SYSTEM_PROMPT
from .prompts import build_user_text
from .prompts import build_fewshot_user_text
from .prompts import build_fewshot_assistant_text


# Qwen2.5-VL processor accepts per-image min/max pixel limits to cap the
# vision-token count. Tradeoff: lower max_pixels → fewer tokens → faster
# inference + lower OOM risk, but loses high-frequency visual detail.
#
# DROID exterior cameras are typically 256x256 or 320x256. With Qwen's
# 28×28 patch size, that's already only ~80 patches per image. We default
# to 256-1280 (Qwen's recommended cost-balanced range).
DEFAULT_MIN_PIXELS = 256 * 28 * 28      # = 200,704 px ≈ 448×448
DEFAULT_MAX_PIXELS = 1280 * 28 * 28     # = 1,003,520 px ≈ 1000×1000


class QwenVLLocalClient:
    """Qwen2.5-VL-72B loaded in-process via HF transformers."""

    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: str | None = None,
        min_pixels: int = DEFAULT_MIN_PIXELS,
        max_pixels: int = DEFAULT_MAX_PIXELS,
        max_new_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModelForVision2Seq
            from transformers import AutoProcessor
        except ImportError as e:
            raise RuntimeError(
                "transformers + torch required for QwenVLLocalClient. "
                "Install with: uv pip install transformers torch accelerate"
            ) from e
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as e:
            raise RuntimeError(
                "qwen_vl_utils required. Install with: uv pip install qwen-vl-utils"
            ) from e

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Qwen model directory not found: {model_path}\n"
                f"Expected HF-format checkpoint with config.json + safetensors shards."
            )

        # Decode torch_dtype string → dtype object (HF accepts both, but bf16
        # autocast on H200 wants the dtype object explicitly).
        dtype: Any = torch_dtype
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            dtype = getattr(torch, torch_dtype)

        # If the caller asked for flash_attention_2 but the flash_attn package
        # is missing from THIS venv (a common source of confusion when the
        # system Python and venv Python differ), silently downgrade to SDPA
        # rather than crashing — transformers' own check raises ImportError
        # that aborts the whole pipeline.
        effective_attn = attn_implementation
        if effective_attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print(
                    "[QwenLocal] WARNING: --attn flash_attention_2 requested "
                    "but flash_attn not importable in this venv. Falling back "
                    "to SDPA (transformers default). To install:\n"
                    "  uv pip install --python .venv/bin/python flash-attn --no-build-isolation"
                )
                effective_attn = "sdpa"

        load_kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if effective_attn is not None:
            load_kwargs["attn_implementation"] = effective_attn

        print(f"[QwenLocal] Loading model from {model_path}  "
              f"dtype={torch_dtype}  device_map={device_map}  "
              f"attn={effective_attn or 'default'}")
        t0 = time.monotonic()
        # AutoModelForVision2Seq auto-detects Qwen2.5-VL vs Qwen2-VL from
        # the config.json's architectures field.
        self._model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._process_vision_info = process_vision_info
        self._torch = torch
        self._model.eval()
        load_dt = time.monotonic() - t0
        print(f"[QwenLocal] Loaded in {load_dt:.1f}s. "
              f"dtype={next(self._model.parameters()).dtype}  "
              f"first_device={next(self._model.parameters()).device}")

        self.model = os.path.basename(model_path.rstrip("/"))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    def _build_messages_qwen(
        self,
        *,
        task_instruction: str,
        keyframes_meta: list[dict],
        keyframe_images: list[np.ndarray],
        include_fewshot: bool = True,
        feed_types: bool = True,
        memory_augmented: bool = False,
    ) -> list[dict[str, Any]]:
        """Qwen-VL chat-message format: image refs use {"type": "image", "image": <PIL>}.

        Different from OpenAI's URL-based content; qwen_vl_utils.process_vision_info
        scans this structure to pull out images for the processor.
        """
        from PIL import Image
        from .prompts import _select_system_prompt

        system_prompt = _select_system_prompt(feed_types, memory_augmented)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        if include_fewshot and feed_types and not memory_augmented:
            # Fewshot is built around the v1 types-fed schema; skip it in
            # no-types and v3 modes to avoid biasing the VLM with examples
            # that don't match the requested output shape.
            messages.append({"role": "user", "content": [
                {"type": "text", "text": build_fewshot_user_text()},
            ]})
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": build_fewshot_assistant_text()},
            ]})

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": build_user_text(
                task_instruction=task_instruction,
                keyframes_meta=keyframes_meta,
                feed_types=feed_types,
                memory_augmented=memory_augmented,
            )},
        ]
        for img in keyframe_images:
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            user_content.append({"type": "image", "image": Image.fromarray(img)})
        messages.append({"role": "user", "content": user_content})
        return messages

    def annotate(
        self,
        *,
        task_instruction: str,
        keyframes_meta: list[dict],
        keyframe_images: list[np.ndarray],
        feed_types: bool = True,
        memory_augmented: bool = False,
    ) -> VlmReply:
        messages = self._build_messages_qwen(
            task_instruction=task_instruction,
            keyframes_meta=keyframes_meta,
            keyframe_images=keyframe_images,
            include_fewshot=True,
            feed_types=feed_types,
            memory_augmented=memory_augmented,
        )

        # Build the input tensors via Qwen's processor.
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # device_map="auto" puts the embedding layer on device 0; move inputs there.
        inputs = inputs.to(next(self._model.parameters()).device)

        n_input_tokens = int(inputs["input_ids"].shape[-1])

        t0 = time.monotonic()
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
            )
        latency = time.monotonic() - t0

        # Strip the input prefix, then decode.
        trimmed = generated[:, n_input_tokens:]
        n_output_tokens = int(trimmed.shape[-1])
        output = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text_out = output[0] if output else ""

        return VlmReply(
            text=text_out,
            latency_s=latency,
            input_tokens=n_input_tokens,
            output_tokens=n_output_tokens,
            model=self.model,
        )
