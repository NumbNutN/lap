"""Bridge V2 + Embodied-CoT pretraining dataset.

This module provides a *non-RLDS* data source for the LAP cascade-VLA pretraining
pipeline, built on top of the ``Embodied-CoT/embodied_features_bridge`` HuggingFace
dataset.

Data flow
---------

1. **ECoT JSON** (``embodied_features_bridge.json``, 1.4 GB) — the only thing
   shipped by the HF dataset. Contains per-step text annotations:
   ``{file_path: {episode_id: {metadata, features, reasoning}}}``.

2. **Bridge V2 images** — referenced by ``file_path`` + ``episode_id`` + step
   index. The ECoT JSON itself does **not** contain pixels. An external image
   source must be plugged in via :class:`BridgeV2ImageLoader`.

This implementation streams the JSON with ``ijson`` (constant memory) and emits
per-step samples conforming to the schema expected by
``lap.transforms.TokenizePromptAndReasoning``::

    {
        "image": np.ndarray  (H, W, 3)  uint8 RGB,
        "image_mask": bool,
        "prompt": str               # task + " [plan] " + plan
        "language_actions": str,    # subtask_reason  (-> [think] segment)
        "langact": str,             # subtask         (-> [action] segment)
        "is_vqa_sample": False, "is_prediction_sample": False,
        "sample_mask": True,
    }

The downstream model and tokenizer changes from
[cascade-gradient-flow-discussion.md](../../../cascade-gradient-flow-discussion.md)
take care of producing both the langact and reasoning masks.

Usage
-----

>>> ds = BridgeECoTDataset(
...     ecot_json_path="~/.cache/.../embodied_features_bridge.json",
...     image_loader=NullImageLoader(image_shape=(224, 224, 3)),
...     include_plan=True,
... )
>>> for sample in ds.iter_samples(max_samples=4):
...     print(sample["prompt"])

NOTE: The default image loader is a placeholder that returns black frames. Real
training requires plugging in a ``BridgeV2ImageLoader`` backed by the actual
Bridge V2 .npy files (see ``cascade-bridge-pretraining-discussion.md`` §4).
"""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses
import logging
import os
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default location of the embodied_features_bridge JSON (HF cache layout).
DEFAULT_ECOT_BRIDGE_JSON = pathlib.Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/"
        "snapshots/854ee59c7c76868d63fac37c33e0f031ed678014/embodied_features_bridge.json"
    )
)


# ---------------------------------------------------------------------------
# Image loaders
# ---------------------------------------------------------------------------


class BridgeV2ImageLoader:
    """Abstract image source. Maps (file_path, episode_id, step_idx) -> RGB array.

    Concrete implementations:
      - :class:`NullImageLoader`  — returns black/random frames; for sanity testing.
      - :class:`LeRobotBridgeImageLoader`  — loads from HF ``IPEC-COMMUNITY/bridge_orig_lerobot``
        (TODO: implement when image data is in place).
      - :class:`RawNpyBridgeImageLoader`  — loads from raw Bridge V2 .npy files.
        (TODO: implement when image data is in place).
    """

    def get(self, file_path: str, episode_id: str, step_idx: int) -> np.ndarray:
        raise NotImplementedError

    def has(self, file_path: str, episode_id: str, step_idx: int) -> bool:
        """Whether this loader can serve the given (file_path, ep, step). Default True."""
        return True


@dataclasses.dataclass
class NullImageLoader(BridgeV2ImageLoader):
    """Returns black / dummy frames. For pipeline plumbing verification only.

    Training with this loader is **not meaningful** — vision features will be
    constant. Use it to validate tokenization + loss computation flow.
    """

    image_shape: tuple[int, int, int] = (224, 224, 3)
    fill_value: int = 0

    def get(self, file_path: str, episode_id: str, step_idx: int) -> np.ndarray:
        return np.full(self.image_shape, self.fill_value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BridgeECoTSampleBuilder:
    """Builds a single training sample dict from a (file, episode, step) triple.

    Two layout modes (controlled by ``plan_as_ar_target``):

    - ``plan_as_ar_target=False`` (legacy / "plan-as-input"):
        prompt = ``"{task} [plan] {plan}"`` (plan shown to model as condition).
        AR target: ``[think]<subtask_reason>[action]<subtask>``.

    - ``plan_as_ar_target=True`` (recommended for pretraining):
        prompt = ``"{task}"`` (task only).
        AR target: ``[think]<plan>\\n<subtask_reason>[action]<subtask>``.
        Plan becomes part of what the model has to predict.
        See cascade-bridge-pretraining-discussion.md §10 for rationale.
    """

    image_loader: BridgeV2ImageLoader
    include_plan: bool = True
    # If True, plan is emitted as part of the AR target (concatenated into reasoning).
    # If False, plan is appended to the prompt text (input-only).
    plan_as_ar_target: bool = True
    # Separator inserted between task and plan in the prompt (only used when
    # plan_as_ar_target=False).
    plan_separator_prompt: str = " [plan] "
    # Separator inserted between plan and subtask_reason in the AR target (only used
    # when plan_as_ar_target=True). Newline keeps them visually distinct in decoded text.
    plan_separator_ar: str = "\n"

    def build(
        self,
        file_path: str,
        episode_id: str,
        step_idx: int,
        reasoning_step: dict[str, str],
    ) -> dict[str, Any] | None:
        """Return one sample dict, or None to skip (e.g., bad data)."""
        task = reasoning_step.get("task")
        plan = reasoning_step.get("plan")
        subtask = reasoning_step.get("subtask")
        subtask_reason = reasoning_step.get("subtask_reason")

        if not task or not subtask or not subtask_reason:
            return None

        task = task.strip()
        plan = plan.strip() if plan else ""
        subtask = subtask.strip()
        subtask_reason = subtask_reason.strip()

        if self.include_plan and plan and not self.plan_as_ar_target:
            # Mode A — plan as input to prompt.
            prompt = f"{task}{self.plan_separator_prompt}{plan}"
            reasoning_for_ar = subtask_reason
        elif self.include_plan and plan and self.plan_as_ar_target:
            # Mode B — plan as AR target (concatenated into reasoning segment).
            prompt = task
            reasoning_for_ar = f"{plan}{self.plan_separator_ar}{subtask_reason}"
        else:
            # No plan available, or plan disabled.
            prompt = task
            reasoning_for_ar = subtask_reason

        try:
            image = self.image_loader.get(file_path, episode_id, step_idx)
        except FileNotFoundError:
            return None
        # Sanity: enforce HWC uint8.
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"BridgeV2ImageLoader returned image with unexpected shape/dtype: "
                f"{image.shape} {image.dtype}; expected (H, W, 3) uint8"
            )

        return {
            "image": image,
            "image_mask": np.bool_(True),
            "prompt": prompt,
            "language_actions": reasoning_for_ar,
            "langact": subtask,
            # Sample-type flags (cascade-VLA uses these to gate losses).
            "is_vqa_sample": np.bool_(False),
            "is_prediction_sample": np.bool_(False),
            "sample_mask": np.bool_(True),
        }


@dataclasses.dataclass
class BridgeECoTDataset:
    """Iterable of per-step samples from Bridge V2 + ECoT annotations.

    Parameters
    ----------
    ecot_json_path :
        Path to ``embodied_features_bridge.json``.
    image_loader :
        How to fetch the actual image for a given step. The default
        ``NullImageLoader`` is a placeholder; production training requires a real
        Bridge V2 image source.
    include_plan :
        Whether to append the episode-level plan to the prompt as
        ``"<task> [plan] <plan>"``. Recommended True; plan is invariant within
        an episode and provides global context across phase-level AR generations.
    skip_steps_without_change :
        If True, only emit one sample per consecutive run of identical
        ``(subtask, subtask_reason)``. Useful to deduplicate the heavy temporal
        repetition in Bridge ECoT (most episodes have ~3-5 phases but ~40 frames).
        Default False for completeness; set True for faster iteration.
    max_episodes :
        Optional cap on number of episodes; useful for smoke tests.
    """

    ecot_json_path: pathlib.Path = DEFAULT_ECOT_BRIDGE_JSON
    image_loader: BridgeV2ImageLoader = dataclasses.field(default_factory=NullImageLoader)
    include_plan: bool = True
    skip_steps_without_change: bool = False
    max_episodes: int | None = None

    def __post_init__(self):
        self._builder = BridgeECoTSampleBuilder(
            image_loader=self.image_loader,
            include_plan=self.include_plan,
        )

    # ----- Streaming iteration -----

    def iter_samples(self, max_samples: int | None = None) -> Iterator[dict[str, Any]]:
        """Yield per-step samples, streaming the ECoT JSON to avoid loading it all."""
        try:
            import ijson  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ijson is required to stream the 1.4 GB Bridge ECoT JSON. "
                "Install with: uv pip install ijson"
            ) from exc

        path = pathlib.Path(self.ecot_json_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Bridge ECoT JSON not found at {path}")

        emitted = 0
        episodes_seen = 0
        with open(path, "rb") as f:
            top = ijson.kvitems(f, "", use_float=True)
            for file_path, episodes in top:
                if not isinstance(episodes, dict):
                    logger.warning(
                        "Unexpected ECoT JSON structure under %s: not a dict; skipping.",
                        file_path,
                    )
                    continue
                for episode_id, ep in episodes.items():
                    if self.max_episodes is not None and episodes_seen >= self.max_episodes:
                        return
                    episodes_seen += 1

                    yield from self._iter_episode(file_path, episode_id, ep)

                    # The samples generator above is async-ish; track emitted only after.
                    if max_samples is not None and emitted >= max_samples:
                        return

    def _iter_episode(
        self, file_path: str, episode_id: str, ep: dict
    ) -> Iterator[dict[str, Any]]:
        reasoning = ep.get("reasoning", {})
        if not isinstance(reasoning, dict):
            logger.warning(
                "Episode %s/%s has no reasoning dict; skipping.", file_path, episode_id
            )
            return

        # Sort steps by integer index.
        try:
            step_keys = sorted(reasoning.keys(), key=lambda s: int(s))
        except (TypeError, ValueError):
            logger.warning(
                "Episode %s/%s has non-integer reasoning keys; skipping.",
                file_path,
                episode_id,
            )
            return

        prev_subtask = None
        for k in step_keys:
            step_idx = int(k)
            r = reasoning[k]
            if not isinstance(r, dict):
                continue

            if self.skip_steps_without_change:
                cur_key = (r.get("subtask"), r.get("subtask_reason"))
                if cur_key == prev_subtask:
                    continue
                prev_subtask = cur_key

            sample = self._builder.build(file_path, episode_id, step_idx, r)
            if sample is not None:
                yield sample

    # ----- Convenience: count samples without yielding (slow but useful) -----

    def count(self) -> int:
        n = 0
        for _ in self.iter_samples():
            n += 1
        return n
