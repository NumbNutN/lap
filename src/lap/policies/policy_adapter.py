import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import openpi.policies.policy as _policy
from openpi.shared import nnx_utils

from lap.models.model_adapter import CoTObservation


class ARPolicy:
    """A policy that supports autoregressive sampling."""

    def __init__(self, base: _policy.Policy, *, sample_kwargs: dict[str, Any] | None = None):
        self._base = base
        assert hasattr(base._model, "sample_tokens"), "Model must have a sample_tokens method"  # noqa: SLF001
        self._sample_tokens = nnx_utils.module_jit(base._model.sample_tokens)
        # self._sample_tokens = base._model.sample_tokens

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def infer_reasoning(self, obs: dict) -> dict:
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        raw_state = inputs["observation"]["state"].copy()
        inputs = self._base._input_transform(inputs)  # noqa: SLF001
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng_or_pytorch_device = jax.random.split(self._base._rng)

        start_time = time.monotonic()
        tokens = self._sample_tokens(sample_rng_or_pytorch_device, CoTObservation.from_dict(inputs))
        outputs = {
            "state": inputs["state"],
            "tokens": tokens,
            "raw_state": raw_state,
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        # outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._base._output_transform(outputs)  # noqa: SLF001
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        return self.infer_reasoning(obs)

    def vqa_infer(self, obs: dict) -> dict:
        """Run VQA inference using the VLM backbone.

        Expects `obs` to contain images and a tokenized prompt compatible with the
        PaliGemma tokenizer. Returns generated text.
        """

        return self.infer_reasoning(obs)


class DualModePolicy:
    """Policy wrapper that emits BOTH flow-sampled actions and cascade text.

    The websocket server calls ``policy.infer(obs)`` and packs the returned
    dict to the client. We piggy-back the cascade-text path on each inference
    so the client (LAP eval driver) can overlay ``[plan]/[stage]/[action]``
    text on rollout videos.

    Internals:
      * ``_flow`` is the regular ``_policy.Policy`` returned by
        ``create_trained_policy`` — produces ``actions``.
      * ``_ar`` is an :class:`ARPolicy` sharing the same underlying model,
        used only for ``infer_reasoning`` (which calls ``sample_tokens``).
      * The result of ``_ar.infer_reasoning`` is passed through
        ``DetokenizeReasoning`` (already in the model_transforms.outputs of the
        train config), so the merged dict carries a ``reasoning`` string.

    Notes:
      * Cost: doubles inference latency. Acceptable for sim eval but should
        be toggled off for high-frequency real-robot use.
      * Both calls share the obs dict; they are independent samples (different
        RNG splits) but driven by the same conditioning.
    """

    def __init__(self, flow: _policy.Policy, ar: "ARPolicy"):
        self._flow = flow
        self._ar = ar

    def __getattr__(self, name: str):
        # Delegate any unknown attribute (e.g. ``metadata``) to the flow
        # policy — that's what the websocket server reads.
        return getattr(self._flow, name)

    @property
    def metadata(self):
        return self._flow.metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # 1) cascade text (AR path).
        try:
            ar_out = self._ar.infer_reasoning(obs)
            reasoning_text = ""
            for key in ("reasoning", "reasoning_text", "text"):
                val = ar_out.get(key)
                if isinstance(val, str) and val.strip():
                    reasoning_text = val.strip()
                    break
        except Exception as e:  # never fail the whole infer because of the text branch
            reasoning_text = f"<reasoning error: {type(e).__name__}: {e}>"

        # 2) action chunk (flow path).
        flow_out = self._flow.infer(obs, noise=noise)
        flow_out["reasoning_text"] = reasoning_text
        return flow_out
