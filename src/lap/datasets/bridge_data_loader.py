"""Data loader integration for Bridge V2 + Embodied-CoT pretraining.

Bridge ECoT is a streaming, non-RLDS, non-LeRobot data source. This module
adapts ``BridgeECoTDataset`` (which yields per-step Python dicts) into the
``DataLoader`` interface expected by ``scripts/train.py``, matching the same
shape as ``RLDSDataLoader`` so the training loop is unchanged.

Pipeline
--------

::

    BridgeECoTDataset                    (streams ECoT JSON + LeRobot images)
       ↓ wrapped as torch.utils.data.IterableDataset
    BridgeIterableTorchDataset
       ↓ in-loop transforms (per-sample)
       ├── inject default image_mask / state / actions placeholders
       ├── fill image dict for both base & wrist cameras (wrist is zero+masked)
       └── apply data_transforms + model_transforms (tokenizer, pad, etc.)
    torch.utils.data.DataLoader          (batching + multiprocessing)
       ↓
    BridgeDataLoader                     (sharding + (CoTObservation, actions) yield)
       ↓
    train.py main loop

Notes
-----

* Bridge ECoT samples have **no real action** and **no proprio state**. We
  emit zeros so the dict layout matches what ``CoTObservation.from_dict``
  expects, but ``LAPConfig.enable_action_training=False`` ensures these
  placeholders never enter the loss computation.

* Wrist images: the LAP default ``image_keys=("base_0_rgb", "left_wrist_0_rgb")``
  does not match Bridge's single external camera. We emit a zero-filled wrist
  image with ``image_mask=False`` so SigLIP simply doesn't attend to it. This
  avoids the need for a Bridge-specific LAPConfig variant.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import logging
import pathlib
import typing
from typing import Any

import jax
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch IterableDataset wrapper
# ---------------------------------------------------------------------------


class _BridgeIterableTorchDataset:
    """Torch ``IterableDataset`` over a streaming ``BridgeECoTDataset``.

    Inlined into the same module to avoid an unconditional torch import at the
    top level (the Bridge path is opt-in).
    """

    def __init__(
        self,
        bridge_ds,
        transforms: Sequence,
        image_keys: tuple[str, ...],
        action_horizon: int,
        action_dim: int,
        state_dim: int,
        image_resolution: tuple[int, int] = (224, 224),
        max_samples: int | None = None,
    ):
        import torch  # local import; only needed when this path is used

        # We deliberately do NOT subclass torch.utils.data.IterableDataset at
        # type-definition time so that simply importing this module works in
        # environments without torch (e.g., the local sanity test environment).
        # Instead, dynamically build the class once torch is available.
        IterableDataset = torch.utils.data.IterableDataset

        class _Wrapped(IterableDataset):
            def __init__(self_inner):
                super().__init__()
                self_inner._bridge_ds = bridge_ds
                self_inner._transforms = list(transforms)
                self_inner._image_keys = image_keys
                self_inner._action_horizon = action_horizon
                self_inner._action_dim = action_dim
                self_inner._state_dim = state_dim
                self_inner._image_resolution = image_resolution
                self_inner._max_samples = max_samples

            def _apply_transforms(self_inner, sample):
                for t in self_inner._transforms:
                    sample = t(sample)
                return sample

            def _expand_to_dataclass_dict(self_inner, raw_sample):
                """Convert raw BridgeECoTDataset sample to the data dict expected
                by ``CoTObservation.from_dict``.

                Adds:
                  - image: dict[camera_key -> uint8 (H,W,3)] (wrist filled with zeros)
                  - image_mask: dict[camera_key -> bool] (wrist=False)
                  - state: zeros[state_dim] (Bridge has no proprio)
                  - actions: zeros[ah, ad] (Bridge ECoT pretraining doesn't use actions)
                """
                base_image = raw_sample["image"]
                # Replicate base image into all configured camera slots; mask out
                # everything except the primary base camera.
                images: dict[str, np.ndarray] = {}
                image_masks: dict[str, np.ndarray] = {}
                primary_key = self_inner._image_keys[0]
                zero_image = np.zeros(
                    (*self_inner._image_resolution, 3), dtype=np.uint8
                )
                for k in self_inner._image_keys:
                    if k == primary_key:
                        images[k] = base_image
                        image_masks[k] = np.bool_(True)
                    else:
                        # Wrist (or auxiliary) camera not provided by Bridge —
                        # emit zeros + mask=False so SigLIP / preprocessing knows.
                        images[k] = zero_image
                        image_masks[k] = np.bool_(False)

                # Build the data dict consumed downstream. Use openpi/CoTObservation
                # naming (image / image_mask, not images / image_masks).
                return {
                    "image": images,
                    "image_mask": image_masks,
                    "state": np.zeros(self_inner._state_dim, dtype=np.float32),
                    "actions": np.zeros(
                        (self_inner._action_horizon, self_inner._action_dim),
                        dtype=np.float32,
                    ),
                    # Cascade-VLA text fields (consumed by TokenizePromptAndReasoning).
                    "prompt": raw_sample["prompt"],
                    "language_actions": raw_sample["language_actions"],
                    "langact": raw_sample["langact"],
                    "plan": raw_sample.get("plan"),
                    "plan_position": raw_sample.get("plan_position", "none"),
                    # Sample-type flags.
                    "is_vqa_sample": raw_sample["is_vqa_sample"],
                    "is_prediction_sample": raw_sample["is_prediction_sample"],
                    "sample_mask": raw_sample["sample_mask"],
                }

            def __iter__(self_inner):
                for raw in self_inner._bridge_ds.iter_samples(
                    max_samples=self_inner._max_samples
                ):
                    sample = self_inner._expand_to_dataclass_dict(raw)
                    sample = self_inner._apply_transforms(sample)
                    yield sample

        self._inner = _Wrapped()

    def torch_dataset(self):
        return self._inner


# ---------------------------------------------------------------------------
# DataLoader (matches RLDSDataLoader's __iter__ shape)
# ---------------------------------------------------------------------------


class BridgeDataLoader:
    """Iterates Bridge ECoT samples and yields ``(CoTObservation, actions)`` tuples.

    Mirrors the public surface of ``lap.datasets.data_loader.RLDSDataLoader``
    so that ``scripts/train.py`` can use it interchangeably.
    """

    def __init__(
        self,
        torch_dataset,
        *,
        batch_size: int,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        data_cfg=None,
    ):
        import torch

        if jax.process_count() > 1:
            raise NotImplementedError(
                "Multi-process Bridge data loading is not supported yet."
            )

        local_batch_size = max(1, batch_size // jax.process_count())
        if sharding is None:
            # Default to NamedSharding over the batch axis. This works correctly
            # for any rank (rank-1 scalars, rank-3 images, rank-4 batches, etc.),
            # whereas PositionalSharding(shape=(N_devices,)) only fits rank-1
            # tensors and trips on multi-GPU pods. Matches upstream
            # `openpi.training.data_loader.TorchDataLoader` default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._sharding = sharding
        self._num_batches = num_batches
        self._data_cfg = data_cfg
        self._batch_size = local_batch_size

        mp_context = None
        if num_workers > 0:
            mp_context = torch.multiprocessing.get_context("spawn")

        self._loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, torch_dataset),
            batch_size=local_batch_size,
            shuffle=False,  # streaming; HF cache layout already random-ish
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            drop_last=True,
        )

    def data_config(self):
        return self._data_cfg

    def _to_device(self, batch):
        def put(x):
            if not (hasattr(x, "shape") and x.shape):
                return x
            if hasattr(x, "dtype") and (
                x.dtype == np.object_ or getattr(x.dtype, "kind", None) in ("U", "S")
            ):
                return x
            if isinstance(self._sharding, jax.sharding.NamedSharding):
                # Skip sharding when batch axis is not divisible by num devices
                # (e.g., smoke tests with batch_size=4 on 2 GPUs is fine, but
                # batch_size=3 on 2 GPUs is not). Falls back to per-device replicate.
                mesh = self._sharding.mesh
                n_dev = int(np.prod(list(mesh.shape.values())))
                if x.shape[0] % n_dev != 0:
                    return jax.device_put(x)
                return jax.make_array_from_process_local_data(self._sharding, x)
            # PositionalSharding path: reshape to match leaf rank so multi-dim
            # tensors (images) are sharded along axis 0 only.
            n_dev = self._sharding.shape[0]
            if x.shape[0] % n_dev != 0:
                return jax.device_put(x)
            shaped_sharding = self._sharding.reshape((n_dev,) + (1,) * (x.ndim - 1))
            return jax.device_put(x, shaped_sharding)

        return jax.tree_util.tree_map(put, batch)

    def __iter__(self) -> Iterator:
        # CoTObservation is imported lazily so module import doesn't pull JAX
        # for non-training use cases (e.g., view scripts).
        from lap.models.model_adapter import CoTObservation

        seen = 0
        while True:
            for batch in self._loader:
                if self._num_batches is not None and seen >= self._num_batches:
                    return
                batch = self._to_device(batch)
                seen += 1
                yield CoTObservation.from_dict(batch), batch["actions"]
            # Streaming dataset exhausted — restart for indefinite iteration.
            logger.info("BridgeDataLoader exhausted underlying stream; restarting.")


def _collate_fn(items):
    """Collate dicts of numpy/python values into batched numpy arrays."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0),
        *items,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def create_bridge_data_loader(
    train_config,
    *,
    sharding: jax.sharding.Sharding | None = None,
    num_batches: int | None = None,
    seed: int = 0,
    max_samples: int | None = None,
) -> BridgeDataLoader:
    """Construct a :class:`BridgeDataLoader` from a TrainConfig.

    Resolves the Bridge ECoT JSON path + LeRobot snapshot dirs, builds (or
    loads from cache) the ECoT↔LeRobot mapping, instantiates the streaming
    dataset, applies the data + model transforms, and wraps everything into
    a torch DataLoader.
    """
    from lap.datasets.bridge_ecot_dataset import (
        BridgeECoTDataset,
        NullImageLoader,
    )
    from lap.datasets.utils.bridge_lerobot_loader import (
        DEFAULT_MAPPING_CACHE,
        DEFAULT_TELEOP_SNAP,
        DEFAULT_SCRIPTED_SNAP_PARENT,
        LeRobotBridgeImageLoader,
        build_ecot_to_lerobot_mapping,
    )

    data_cfg = train_config.data.create(train_config.assets_dirs, train_config.model)
    cfg = train_config.data  # BridgeECoTDataConfig (frozen original)

    ecot_json = pathlib.Path(cfg.ecot_json_path).expanduser()
    if not ecot_json.exists():
        raise FileNotFoundError(
            f"Bridge ECoT JSON not found at {ecot_json}. Run "
            f"`huggingface-cli download Embodied-CoT/embodied_features_bridge "
            f"--repo-type dataset` on the host (or pod)."
        )

    # Resolve scripted snapshot if available.
    scripted_snap = None
    if DEFAULT_SCRIPTED_SNAP_PARENT.exists():
        snaps = sorted(p for p in DEFAULT_SCRIPTED_SNAP_PARENT.iterdir() if p.is_dir())
        scripted_snap = snaps[0] if snaps else None

    # Build the image loader. If LeRobot teleop snapshot is missing, fall back
    # to NullImageLoader (training will run but produce no useful gradients —
    # use only for plumbing tests).
    if not DEFAULT_TELEOP_SNAP.exists():
        logger.warning(
            "LeRobot teleop snapshot not found at %s; falling back to NullImageLoader. "
            "Training with constant black images is NOT meaningful.",
            DEFAULT_TELEOP_SNAP,
        )
        image_loader = NullImageLoader()
    else:
        mapping = build_ecot_to_lerobot_mapping(
            ecot_json_path=ecot_json,
            teleop_snap=DEFAULT_TELEOP_SNAP,
            scripted_snap=scripted_snap,
            cache_path=DEFAULT_MAPPING_CACHE,
            force_rebuild=False,
        )
        logger.info(
            "Bridge ECoT↔LeRobot mapping: %d/%d matched (%.1f%%)",
            mapping["n_matched"],
            mapping["n_ecot_total"],
            100.0 * mapping["n_matched"] / max(mapping["n_ecot_total"], 1),
        )
        image_loader = LeRobotBridgeImageLoader(
            teleop_snap=DEFAULT_TELEOP_SNAP,
            scripted_snap=scripted_snap,
            mapping=mapping,
        )

    bridge_ds = BridgeECoTDataset(
        ecot_json_path=ecot_json,
        image_loader=image_loader,
        include_plan=cfg.include_plan,
        p_plan=cfg.p_plan,
        skip_steps_without_change=cfg.skip_steps_without_change,
        max_episodes=cfg.max_episodes,
    )

    # Compose the transform pipeline that the existing LAP pipeline uses for
    # RLDS data, but we apply per-sample (not per-batch) since BridgeECoTDataset
    # produces single samples.
    transforms = [
        *data_cfg.data_transforms.inputs,
        *data_cfg.model_transforms.inputs,
    ]

    image_keys = train_config.model.image_keys
    image_resolution = train_config.model.image_resolution

    torch_ds_wrapper = _BridgeIterableTorchDataset(
        bridge_ds=bridge_ds,
        transforms=transforms,
        image_keys=image_keys,
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        state_dim=train_config.model.action_dim,  # Bridge has no real state; pad to action_dim
        image_resolution=image_resolution,
        max_samples=max_samples,
    )

    # NOTE: Bridge ECoT uses a streaming IterableDataset that is non-trivial to
    # pickle (file handles, ijson stream state, decord readers). Spawn workers
    # would also each rebuild the ECoT↔LeRobot mapping on import. Force
    # num_workers=0 to keep things in-process — matches the upstream RLDS
    # default (see openpi/.../config.py:850 comment "RLDS DataLoader requires
    # num_workers=0").
    return BridgeDataLoader(
        torch_ds_wrapper.torch_dataset(),
        batch_size=train_config.batch_size,
        sharding=sharding,
        num_batches=num_batches,
        num_workers=0,
        seed=seed,
        data_cfg=data_cfg,
    )
