"""Torch DataLoader integration for RoboTwin Stage 2 (action-expert) training.

Mirrors the structure of ``bridge_data_loader.py`` but adapts to the
RoboTwin-specific differences:

* ``RoboTwinTaskDataset.iter_samples()`` already emits the per-step dict in
  the schema expected by the LAP transforms (image dict, image_mask dict,
  state 14-d, actions (H, 14), cascade text fields), so we don't need a
  ``_expand_to_dataclass_dict`` step like Bridge.
* Multi-task mixing happens at the dataset layer (``RoboTwinMixedDataset``),
  not the loader layer.
* We *reuse* ``BridgeDataLoader`` as the torch→sharded-CoTObservation
  pipeline since its surface (``__iter__`` yielding ``(obs, actions)``) is
  framework-agnostic and matches RoboTwin's needs unchanged.

The factory ``create_robotwin_data_loader`` is wired into
``lap.datasets.data_loader.create_data_loader`` via a ``repo_id == "robotwin"``
dispatch.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Iterator, Sequence

import numpy as np

from lap.datasets.bridge_data_loader import BridgeDataLoader  # public reuse
from lap.datasets.robotwin_dataset import RoboTwinMixedDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Torch IterableDataset wrapper
# ---------------------------------------------------------------------------


class _RoboTwinIterableTorchDataset:
    """Wrap a ``RoboTwinMixedDataset`` as a torch IterableDataset that yields
    fully-formed sample dicts (after applying transforms).
    """

    def __init__(
        self,
        mixed_dataset: RoboTwinMixedDataset,
        transforms: Sequence,
        max_samples: int | None = None,
    ):
        import torch  # local import; only required when this path is used

        IterableDataset = torch.utils.data.IterableDataset

        class _Wrapped(IterableDataset):
            def __init__(self_inner):
                super().__init__()
                self_inner._mixed = mixed_dataset
                self_inner._transforms = list(transforms)
                self_inner._max_samples = max_samples

            def _apply_transforms(self_inner, sample):
                for t in self_inner._transforms:
                    sample = t(sample)
                return sample

            def _strip_debug_keys(self_inner, sample):
                """Remove leading-underscore bookkeeping keys before transforms.

                Transforms / collate_fn don't know about these (they originated
                in ``RoboTwinTaskDataset._build_sample`` for debug purposes),
                and the downstream ``_collate_fn`` would try to stack arbitrary
                python types and crash.
                """
                return {k: v for k, v in sample.items() if not k.startswith("_")}

            def __iter__(self_inner):
                for raw in self_inner._mixed.iter_samples(max_samples=self_inner._max_samples):
                    sample = self_inner._strip_debug_keys(raw)
                    sample = self_inner._apply_transforms(sample)
                    yield sample

        self._inner = _Wrapped()

    def torch_dataset(self):
        return self._inner


# ---------------------------------------------------------------------------
# Factory wired into create_data_loader
# ---------------------------------------------------------------------------


def create_robotwin_data_loader(
    train_config,
    *,
    sharding=None,
    num_batches: int | None = None,
    seed: int = 0,
    max_samples: int | None = None,
) -> BridgeDataLoader:
    """Build a sharded RoboTwin DataLoader yielding ``(CoTObservation, actions)``.

    Reads:
      * ``train_config.data`` — expected to be a ``RoboTwinDataConfig`` (carries
        ``data_root``, ``dataset_weights``, ``p_plan``, ``p_full_reasoning``,
        ``max_episodes_per_dataset``).
      * ``train_config.model.action_horizon``, ``action_dim``, ``image_keys``,
        ``image_resolution``.
      * ``train_config.batch_size``, ``train_config.assets_dirs``.

    Returns: ``BridgeDataLoader`` (reused; the public surface is dataset-agnostic).
    """
    data_cfg = train_config.data.create(train_config.assets_dirs, train_config.model)
    cfg = train_config.data  # RoboTwinDataConfig (frozen original)

    data_root = pathlib.Path(getattr(cfg, "data_root", "/data/zhaoqc/RoboTwin/data")).expanduser()
    if not data_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin data_root not found: {data_root}. "
            f"Expected per-task subdirs under <data_root>/<task_family>/demo_clean/"
        )

    weights = dict(getattr(cfg, "dataset_weights", {}))
    if not weights:
        # Empty weights → use the package-default mix.
        from lap.datasets.robotwin_dataset import DEFAULT_DATASET_WEIGHTS
        weights = dict(DEFAULT_DATASET_WEIGHTS)

    mixed_ds = RoboTwinMixedDataset(
        data_root=data_root,
        weights=weights,
        action_horizon=train_config.model.action_horizon,
        p_plan=getattr(cfg, "p_plan", 0.15),
        p_full_reasoning=getattr(cfg, "p_full_reasoning", 0.20),
        image_size=tuple(train_config.model.image_resolution),
        max_episodes_per_dataset=getattr(cfg, "max_episodes_per_dataset", None),
        seed=seed,
        state_kind=getattr(cfg, "state_kind", "qpos"),
    )
    logger.info(
        "RoboTwinMixedDataset ready: tasks=%s",
        mixed_ds.dataset_names,
    )

    # Same transform pipeline as Bridge: data_transforms (typically empty for
    # cascade-VLA datasets that already emit the right schema) + model_transforms
    # (TokenizePromptAndReasoning + image normalization).
    transforms = [
        *data_cfg.data_transforms.inputs,
        *data_cfg.model_transforms.inputs,
    ]

    torch_ds_wrapper = _RoboTwinIterableTorchDataset(
        mixed_dataset=mixed_ds,
        transforms=transforms,
        max_samples=max_samples,
    )

    # Reuse BridgeDataLoader. It does not bake in any Bridge-specific
    # assumptions — just iterates a torch Dataset, collates, and pushes shards.
    return BridgeDataLoader(
        torch_ds_wrapper.torch_dataset(),
        batch_size=train_config.batch_size,
        sharding=sharding,
        num_batches=num_batches,
        num_workers=0,    # streaming + open HDF5 handles → workers would dup file handles
        seed=seed,
        data_cfg=data_cfg,
    )
