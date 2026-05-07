"""End-to-end smoke test for the Bridge ECoT data loader.

This goes one layer further than ``test_bridge_batch_dump.py``: it exercises
the full ``create_data_loader`` -> torch DataLoader -> sharding pipeline that
the actual training loop runs. Use this to catch:

  - missing fields in the data dict (CoTObservation.from_dict failures)
  - transform composition errors (TokenizePromptAndReasoning, PadStatesAndActions)
  - collate errors when batching across samples
  - sharding / dtype issues

Usage::

    cd policy/lap
    .venv/bin/python scripts/test_bridge_dataloader.py --batch-size 2 --num-batches 2
"""

from __future__ import annotations

import argparse
import logging
import os

# Quiet TF logs before any heavy import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from lap.datasets.data_loader import create_data_loader
from lap.training import config as _config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="lap_bridge_pretrain")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-batches", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    train_config = _config.get_config(args.config)
    # Override batch_size and num_workers for the smoke test.
    import dataclasses

    train_config = dataclasses.replace(
        train_config, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Loaded config: {train_config.name}")
    print(f"  batch_size      = {train_config.batch_size}")
    print(f"  action_horizon  = {train_config.model.action_horizon}")
    print(f"  action_dim      = {train_config.model.action_dim}")
    print(f"  max_token_len   = {train_config.model.max_token_len}")
    print(f"  data.repo_id    = {train_config.data.repo_id}")
    print(f"  data.p_plan     = {train_config.data.p_plan}")
    print()

    print("Building data loader (this builds ECoT↔LeRobot mapping on first run, ~1 min)...")
    loader = create_data_loader(
        train_config,
        sharding=None,
        shuffle=False,
        num_batches=args.num_batches,
        seed=0,
    )

    print(f"Iterating {args.num_batches} batches:\n")
    for i, (obs, actions) in enumerate(loader):
        print(f"=== Batch {i} ===")
        # Image dict.
        for cam, im in obs.images.items():
            mask_val = obs.image_masks[cam]
            print(f"  image[{cam!r}]      : shape={tuple(im.shape)}  dtype={im.dtype}  mask={np.asarray(mask_val).tolist()}")
        # State.
        print(f"  state              : shape={tuple(obs.state.shape)}  dtype={obs.state.dtype}")
        # Tokenized fields.
        if obs.tokenized_prompt is not None:
            tp = np.asarray(obs.tokenized_prompt)
            print(f"  tokenized_prompt   : shape={tuple(tp.shape)}  dtype={tp.dtype}  unique tokens (sample 0)={len(set(tp[0].tolist()))}")
        if obs.tokenized_prompt_mask is not None:
            tpm = np.asarray(obs.tokenized_prompt_mask)
            print(f"  prompt_mask sums   : per-sample valid token counts = {tpm.sum(axis=-1).tolist()}")
        if obs.tokenized_ar_target_mask is not None:
            ar = np.asarray(obs.tokenized_ar_target_mask)
            print(f"  ar_target_mask     : sums = {ar.sum(axis=-1).tolist()}")
        if obs.tokenized_stage_mask is not None:
            sm = np.asarray(obs.tokenized_stage_mask)
            print(f"  stage_mask         : sums = {sm.sum(axis=-1).tolist()}")
        if obs.tokenized_plan_mask is not None:
            pm = np.asarray(obs.tokenized_plan_mask)
            print(f"  plan_mask          : sums = {pm.sum(axis=-1).tolist()}")
        # Actions.
        a = np.asarray(actions)
        print(f"  actions            : shape={tuple(a.shape)}  dtype={a.dtype}  all_zero={(a == 0).all()}")
        print()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
