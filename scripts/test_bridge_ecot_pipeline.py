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

        # Run through the two-segment tokenizer.
        (
            tokens,
            attn_mask,
            langact_mask,
            number_mask,
            direction_mask,
            token_loss_mask,
            reasoning_only_mask,
        ) = tokenizer.tokenize(
            prompt=sample["prompt"],
            reasoning=sample["language_actions"],
            state=None,  # bridge has no state vector
            langact=sample["langact"],
            is_vqa_sample=False,
            is_prediction_sample=False,
        )

        n_valid = int(attn_mask.sum())
        n_lang = int(langact_mask.sum()) if langact_mask is not None else 0
        n_reason = int(reasoning_only_mask.sum()) if reasoning_only_mask is not None else 0
        n_action = n_lang - n_reason
        print(f"  tokens.shape    : {tokens.shape}  valid={n_valid}/{len(tokens)}")
        print(f"  langact_mask    : {n_lang} positions  (= reasoning + langact)")
        print(f"  reasoning_only  : {n_reason} positions")
        print(f"  langact_only    : {n_action} positions  (= langact_mask & ~reasoning_mask)")

        # Sanity: the two segment masks must not overlap.
        if reasoning_only_mask is not None:
            overlap = (reasoning_only_mask & np.logical_not(langact_mask)).sum()
            assert overlap == 0, f"reasoning_mask leaks outside langact_mask ({overlap} pos)"
            # langact_only inside ar_mask
            langact_only = langact_mask & np.logical_not(reasoning_only_mask)
            assert (langact_only & reasoning_only_mask).sum() == 0
            print(f"  ✓ masks disjoint: reasoning ∩ langact_only = empty")

        # Decode the reasoning + langact spans for human inspection.
        reasoning_token_ids = tokens[reasoning_only_mask] if reasoning_only_mask is not None else []
        langact_only_mask = (
            langact_mask & np.logical_not(reasoning_only_mask)
            if reasoning_only_mask is not None
            else langact_mask
        )
        langact_token_ids = tokens[langact_only_mask] if langact_only_mask is not None else []
        if len(reasoning_token_ids) > 0:
            print(f"  decoded[think]  : {tokenizer.decode(np.asarray(reasoning_token_ids))!r}")
        if len(langact_token_ids) > 0:
            print(f"  decoded[action] : {tokenizer.decode(np.asarray(langact_token_ids))!r}")
        print()

    print("Done. Pipeline plumbing OK.")


if __name__ == "__main__":
    main()
