"""Sanity-check the Bridge ECoT data path: dataset → tokenizer → masks.

Verifies end-to-end without requiring real Bridge V2 images:

  1. Stream a few samples from BridgeECoTDataset (using NullImageLoader stub).
  2. Tokenize each sample via the cascade-VLA two-segment path.
  3. Print a summary of mask shapes / coverage.

Usage::

    uv run python policy/lap/scripts/test_bridge_ecot_pipeline.py --num-samples 3

Run this AFTER you've made the cascade-VLA tokenizer changes and BEFORE you
plumb in real Bridge V2 image data.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lap.datasets.bridge_ecot_dataset import BridgeECoTDataset
from lap.datasets.bridge_ecot_dataset import DEFAULT_ECOT_BRIDGE_JSON
from lap.datasets.bridge_ecot_dataset import NullImageLoader
from lap.models.tokenizer import PaligemmaTokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--max-token-len", type=int, default=256)
    ap.add_argument("--no-plan", action="store_true", help="Disable [plan] in prompt")
    ap.add_argument(
        "--skip-repeat",
        action="store_true",
        help="Skip consecutive frames with same (subtask, subtask_reason)",
    )
    args = ap.parse_args()

    print(f"Loading Bridge ECoT JSON from {DEFAULT_ECOT_BRIDGE_JSON} ...")
    if not DEFAULT_ECOT_BRIDGE_JSON.exists():
        sys.exit(
            f"[error] ECoT JSON not found. Have you run "
            f"`huggingface-cli download Embodied-CoT/embodied_features_bridge --repo-type dataset`?"
        )

    ds = BridgeECoTDataset(
        ecot_json_path=DEFAULT_ECOT_BRIDGE_JSON,
        image_loader=NullImageLoader(),
        include_plan=not args.no_plan,
        skip_steps_without_change=args.skip_repeat,
    )

    print("Initializing PaligemmaTokenizer ...")
    tokenizer = PaligemmaTokenizer(max_len=args.max_token_len, prompt_format="lap")

    print(f"\nStreaming {args.num_samples} samples from Bridge ECoT...\n")
    for i, sample in enumerate(ds.iter_samples(max_samples=args.num_samples)):
        if i >= args.num_samples:
            break

        print(f"=== Sample {i} ===")
        print(f"  prompt          : {sample['prompt']!r}")
        print(f"  language_actions: {sample['language_actions']!r}")
        print(f"  langact         : {sample['langact']!r}")
        print(f"  image.shape     : {sample['image'].shape} {sample['image'].dtype}")

        # Run through the cascade-VLA segmented tokenizer.
        (
            tokens,
            attn_mask,
            ar_target_mask,
            number_mask,
            direction_mask,
            token_loss_mask,
            stage_mask,
            plan_mask,
        ) = tokenizer.tokenize(
            prompt=sample["prompt"],
            reasoning=sample["language_actions"],
            state=None,  # Bridge ECoT has no state vector
            plan=sample.get("plan"),
            plan_position=sample.get("plan_position", "none"),
            langact=sample["langact"],
            is_vqa_sample=False,
            is_prediction_sample=False,
        )

        n_valid = int(attn_mask.sum())
        n_ar = int(ar_target_mask.sum()) if ar_target_mask is not None else 0
        n_plan = int(plan_mask.sum()) if plan_mask is not None else 0
        n_stage = int(stage_mask.sum()) if stage_mask is not None else 0
        n_action = n_ar - n_stage - n_plan
        print(f"  tokens.shape    : {tokens.shape}  valid={n_valid}/{len(tokens)}")
        print(f"  plan_position   : {sample.get('plan_position', 'none')}")
        print(f"  ar_target_mask  : {n_ar} positions  (= plan + stage + action)")
        print(f"    plan_mask     : {n_plan} positions")
        print(f"    stage_mask    : {n_stage} positions")
        print(f"    action_only   : {n_action} positions  (= ar_target & ~stage & ~plan)")

        # Sanity: stage and plan masks must not overlap; both subsets of ar_target.
        if stage_mask is not None:
            assert (stage_mask & np.logical_not(ar_target_mask)).sum() == 0
        if plan_mask is not None:
            assert (plan_mask & np.logical_not(ar_target_mask)).sum() == 0
        if stage_mask is not None and plan_mask is not None:
            assert (stage_mask & plan_mask).sum() == 0
        print(f"  ✓ masks disjoint: plan ∩ stage = ∅,  plan ⊂ ar_target,  stage ⊂ ar_target")

        # Decode each AR span for human inspection.
        if plan_mask is not None and n_plan > 0:
            plan_ids = tokens[plan_mask]
            print(f"  decoded[plan]   : {tokenizer.decode(np.asarray(plan_ids))!r}")
        if stage_mask is not None and n_stage > 0:
            stage_ids = tokens[stage_mask]
            print(f"  decoded[stage]  : {tokenizer.decode(np.asarray(stage_ids))!r}")
        action_only_mask = ar_target_mask
        if stage_mask is not None:
            action_only_mask = action_only_mask & np.logical_not(stage_mask)
        if plan_mask is not None:
            action_only_mask = action_only_mask & np.logical_not(plan_mask)
        if action_only_mask is not None and action_only_mask.sum() > 0:
            print(f"  decoded[action] : {tokenizer.decode(np.asarray(tokens[action_only_mask]))!r}")
        print()

    print("Done. Pipeline plumbing OK.")


if __name__ == "__main__":
    main()
