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


class CascadePipelinePolicy:
    """Cascade-VLA inference: AR-generate cascade text, then flow-sample actions
    *conditioned on the generated cascade*.

    Why this exists:
      The model's ``sample_actions`` reads ``tokenized_prompt`` verbatim — it
      does NOT internally run AR to produce the ``[plan]/[stage]/[action]``
      cascade context. At training time the cascade text is part of
      ``tokenized_prompt`` (encoded via ``TokenizePromptAndReasoning`` from the
      dataset's plan/reasoning/langact fields). At eval time the client only
      sends the bare instruction, so ``sample_actions`` would otherwise see a
      prefix it never saw during training — the cascade context is missing.

    This class fixes the train/test gap by running a single pipeline per
    request:

      1. ``_input_transform``  →  tokenized_prompt = ``[BOS]<instruction>``.
      2. ``sample_tokens``     →  AR-generates the cascade continuation
                                  ``[plan]<plan> [stage]<reason> [action]<langact>[EOS]``.
      3. Append the generated tokens to ``tokenized_prompt`` and mark them as
         AR-target so the action expert sees the same attention pattern as
         during training (cascade span = causal; instruction = bidirectional).
      4. ``sample_actions``    →  flow-sample actions on the augmented obs.
      5. Return ``{actions, reasoning_text}``. ``reasoning_text`` is the
         decoded cascade so the eval driver can overlay it on rollout video.

    Cost: ~2× a pure flow call (one AR decode + one flow sample). For sim
    eval that's fine. For real-robot use you could cache the cascade per
    phase and skip step 2 on most ticks; that's a future optimization.
    """

    EOS_TOKEN: int = 1  # both PaliGemma and Gemma3 use 1

    def __init__(
        self,
        base: _policy.Policy,
        *,
        sample_kwargs: dict[str, Any] | None = None,
    ):
        if base._is_pytorch_model:  # noqa: SLF001
            raise NotImplementedError(
                "CascadePipelinePolicy currently only supports the JAX path."
            )
        self._base = base
        # Reuse base's JIT-compiled sample_actions; add a JIT-compiled
        # sample_tokens. The same params/sharding are shared.
        self._sample_actions = base._sample_actions  # noqa: SLF001
        self._sample_tokens = nnx_utils.module_jit(base._model.sample_tokens)
        # Pull the PaligemmaTokenizer instance out of the composed input
        # transforms so we can decode the AR-generated tokens to text for the
        # video overlay.
        self._tokenizer = self._extract_tokenizer()
        self._sample_kwargs = dict(sample_kwargs or {})

    def _extract_tokenizer(self):
        composite = self._base._input_transform  # noqa: SLF001
        for t in getattr(composite, "transforms", []) or []:
            tok = getattr(t, "tokenizer", None)
            if tok is not None:
                return tok
        raise RuntimeError(
            "Could not locate a tokenizer in base._input_transform.transforms — "
            "CascadePipelinePolicy needs one to decode generated cascade tokens."
        )

    # -- delegation so the websocket server sees the same surface as a Policy --

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    @property
    def metadata(self):
        return self._base.metadata

    # -- core inference --

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        # Stage 0: standard input transform + add batch dim, mirroring
        # ``openpi.policies.policy.Policy.infer`` so we land in the same shape
        # the model expects.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._base._input_transform(inputs)  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        # Stage 1: AR-sample the cascade tokens conditioned on the bare prefix.
        observation_ar = CoTObservation.from_dict(inputs)
        self._base._rng, ar_rng = jax.random.split(self._base._rng)  # noqa: SLF001
        ar_start = time.monotonic()
        ar_tokens = self._sample_tokens(ar_rng, observation_ar)  # (1, max_decode)
        ar_tokens.block_until_ready() if hasattr(ar_tokens, "block_until_ready") else None
        ar_ms = (time.monotonic() - ar_start) * 1000.0
        ar_arr = np.asarray(ar_tokens[0])

        # Find the actual length of the generated cascade. ``sample_tokens``
        # pre-allocates ``output_tokens`` as zeros, fills in step by step, and
        # stops when EOS is hit. So the valid span ends at the first EOS
        # (inclusive). If the model never emits EOS we fall back to the full
        # decode length minus any trailing zero padding (zero is the init).
        eos_positions = np.where(ar_arr == self.EOS_TOKEN)[0]
        if eos_positions.size:
            gen_len = int(eos_positions[0]) + 1  # include EOS
        else:
            nonzero = np.nonzero(ar_arr)[0]
            gen_len = int(nonzero[-1]) + 1 if nonzero.size else 0

        reasoning_text = ""
        if gen_len > 0:
            try:
                reasoning_text = self._tokenizer.decode(ar_arr[:gen_len].astype(np.int32))
            except Exception as e:  # never fail the request because of decode
                reasoning_text = f"<cascade decode error: {type(e).__name__}: {e}>"

        # Stage 2: append generated cascade tokens onto ``tokenized_prompt`` and
        # mark the appended span as AR-target so ``embed_prefix`` applies the
        # same causal-attention pattern the cascade got during training.
        orig_tokens = np.asarray(inputs["tokenized_prompt"][0])      # (L,)
        orig_mask = np.asarray(inputs["tokenized_prompt_mask"][0])    # (L,)
        L = orig_tokens.shape[0]
        orig_len = int(orig_mask.sum())

        ar_target_in = inputs.get("tokenized_ar_target_mask")
        if ar_target_in is None:
            orig_ar = np.zeros(L, dtype=bool)
        else:
            orig_ar = np.asarray(ar_target_in[0])

        new_len = min(orig_len + gen_len, L)
        n_append = max(0, new_len - orig_len)

        new_tokens = orig_tokens.copy()
        new_mask = orig_mask.copy()
        new_ar = orig_ar.copy()
        if n_append > 0:
            new_tokens[orig_len:new_len] = ar_arr[:n_append].astype(new_tokens.dtype)
            new_mask[orig_len:new_len] = True
            new_ar[orig_len:new_len] = True

        inputs_aug = dict(inputs)
        inputs_aug["tokenized_prompt"] = jnp.asarray(new_tokens)[None, :]
        inputs_aug["tokenized_prompt_mask"] = jnp.asarray(new_mask)[None, :]
        inputs_aug["tokenized_ar_target_mask"] = jnp.asarray(new_ar)[None, :]

        # Stage 3: flow-sample actions on the augmented observation.
        observation_flow = CoTObservation.from_dict(inputs_aug)
        self._base._rng, flow_rng = jax.random.split(self._base._rng)  # noqa: SLF001
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_arr = jnp.asarray(noise)
            if noise_arr.ndim == 2:
                noise_arr = noise_arr[None, ...]
            sample_kwargs["noise"] = noise_arr

        flow_start = time.monotonic()
        actions = self._sample_actions(flow_rng, observation_flow, **sample_kwargs)
        actions.block_until_ready() if hasattr(actions, "block_until_ready") else None
        flow_ms = (time.monotonic() - flow_start) * 1000.0

        # Stage 4: output transform — match Policy.infer's contract so the
        # client/server packing path stays identical.
        outputs = {
            "state": inputs_aug["state"],
            "actions": actions,
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._base._output_transform(outputs)  # noqa: SLF001
        outputs["reasoning_text"] = reasoning_text
        outputs["policy_timing"] = {
            "ar_ms": ar_ms,
            "flow_ms": flow_ms,
            "infer_ms": ar_ms + flow_ms,
            "cascade_tokens": int(gen_len),
        }
        return outputs
