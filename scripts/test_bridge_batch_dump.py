"""End-to-end sanity check for the Bridge ECoT pretraining data path.

Loads ONE sample (or batch) using the full pipeline:
  1. ECoT JSON -> BridgeECoTDataset
  2. Image fetched via LeRobotBridgeImageLoader (real Bridge V2 frames)
  3. Tokenized via the cascade-VLA two-segment tokenizer
  4. Dump human-readable inspection: image saved as PNG, prompt + AR target +
     mask coverage.

Usage::

    # First time: build the ECoT->LeRobot mapping (slow, ~3-5 min, cached).
    python policy/lap/scripts/test_bridge_batch_dump.py --build-mapping

    # Then: dump one sample
    python policy/lap/scripts/test_bridge_batch_dump.py --num-samples 2 --out-dir /tmp/bridge_dump

The dumped folder will contain ``sample_<i>.png`` (RGB image) and
``sample_<i>.txt`` (text inspection: prompt, ar target text, mask stats).
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np

from lap.datasets.bridge_ecot_dataset import (
    BridgeECoTDataset,
    DEFAULT_ECOT_BRIDGE_JSON,
    NullImageLoader,
)
from lap.datasets.utils.bridge_lerobot_loader import (
    DEFAULT_MAPPING_CACHE,
    DEFAULT_TELEOP_SNAP,
    DEFAULT_SCRIPTED_SNAP_PARENT,
    LeRobotBridgeImageLoader,
    build_ecot_to_lerobot_mapping,
)
from lap.models.tokenizer import PaligemmaTokenizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _resolve_scripted_snap() -> pathlib.Path | None:
    """Find a snapshot dir under datasets--jnogga--bridge_data_v2_scripted."""
    if not DEFAULT_SCRIPTED_SNAP_PARENT.exists():
        return None
    snaps = sorted(p for p in DEFAULT_SCRIPTED_SNAP_PARENT.iterdir() if p.is_dir())
    return snaps[0] if snaps else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--build-mapping",
        action="store_true",
        help="Force rebuild the ECoT->LeRobot mapping cache",
    )
    ap.add_argument(
        "--use-null-images",
        action="store_true",
        help="Use NullImageLoader instead of LeRobot images (skip mapping/decord deps)",
    )
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--max-token-len", type=int, default=256)
    ap.add_argument(
        "--out-dir",
        default="/tmp/bridge_dump",
        help="Directory to write inspection PNG/TXT files",
    )
    ap.add_argument(
        "--plan-as-ar-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), plan goes into AR target. If False, plan into prompt.",
    )
    ap.add_argument(
        "--skip-repeat",
        action="store_true",
        help="Dedupe consecutive frames with same (subtask, subtask_reason)",
    )
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # 1. Mapping (skip if NullImageLoader is selected).
    if args.use_null_images:
        print("Using NullImageLoader — black frames, no mapping needed")
        image_loader = NullImageLoader()
    else:
        print(f"Building / loading ECoT->LeRobot mapping (cache={DEFAULT_MAPPING_CACHE}) ...")
        scripted_snap = _resolve_scripted_snap()
        if scripted_snap:
            print(f"  scripted snapshot: {scripted_snap}")
        else:
            print("  scripted snapshot: NOT FOUND (will skip scripted entries)")
        mapping = build_ecot_to_lerobot_mapping(
            ecot_json_path=DEFAULT_ECOT_BRIDGE_JSON,
            teleop_snap=DEFAULT_TELEOP_SNAP,
            scripted_snap=scripted_snap,
            cache_path=DEFAULT_MAPPING_CACHE,
            force_rebuild=args.build_mapping,
        )
        print(
            f"  mapping: {mapping['n_matched']}/{mapping['n_ecot_total']} matched "
            f"({mapping['n_matched']/max(mapping['n_ecot_total'], 1):.1%})"
        )
        image_loader = LeRobotBridgeImageLoader(
            teleop_snap=DEFAULT_TELEOP_SNAP,
            scripted_snap=scripted_snap,
            mapping=mapping,
        )

    # 2. Dataset
    ds = BridgeECoTDataset(
        ecot_json_path=DEFAULT_ECOT_BRIDGE_JSON,
        image_loader=image_loader,
        include_plan=True,
        skip_steps_without_change=args.skip_repeat,
    )
    # Override plan_as_ar_target via the builder.
    ds._builder.plan_as_ar_target = args.plan_as_ar_target

    # 3. Tokenizer
    tokenizer = PaligemmaTokenizer(max_len=args.max_token_len, prompt_format="lap")

    # 4. Stream samples and dump.
    print()
    n = 0
    for sample in ds.iter_samples(max_samples=args.num_samples * 4):
        if n >= args.num_samples:
            break

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
            state=None,
            langact=sample["langact"],
            is_vqa_sample=False,
            is_prediction_sample=False,
        )

        # Mask stats.
        n_valid = int(attn_mask.sum())
        n_lang = int(langact_mask.sum()) if langact_mask is not None else 0
        n_reason = int(reasoning_only_mask.sum()) if reasoning_only_mask is not None else 0
        n_action = n_lang - n_reason
        if reasoning_only_mask is not None:
            langact_only_mask = langact_mask & np.logical_not(reasoning_only_mask)
        else:
            langact_only_mask = langact_mask

        # Decode AR target spans.
        decoded_reason = (
            tokenizer.decode(np.asarray(tokens[reasoning_only_mask]))
            if reasoning_only_mask is not None
            else "(no reasoning span)"
        )
        decoded_langact = (
            tokenizer.decode(np.asarray(tokens[langact_only_mask]))
            if langact_only_mask is not None
            else "(no langact span)"
        )

        # Save image.
        try:
            from PIL import Image as PILImage

            png_path = out_dir / f"sample_{n:02d}.png"
            PILImage.fromarray(sample["image"]).save(png_path)
        except ImportError:
            png_path = None

        # Save text inspection.
        txt_path = out_dir / f"sample_{n:02d}.txt"
        lines = [
            f"=== Sample {n} ===",
            "",
            "[INPUT]",
            f"  image.shape       : {sample['image'].shape} {sample['image'].dtype}",
            f"  image.png         : {png_path}",
            f"  prompt            : {sample['prompt']!r}",
            "",
            "[AR TARGET — what the model is trained to predict]",
            f"  language_actions  : {sample['language_actions']!r}",
            f"  langact           : {sample['langact']!r}",
            "",
            "[TOKENIZER OUTPUT]",
            f"  total tokens      : {len(tokens)}",
            f"  valid (non-pad)   : {n_valid}",
            f"  langact_mask      : {n_lang} positions  (= reasoning + langact, used for CE loss)",
            f"    reasoning_only  : {n_reason} positions  ([think] segment, blocked from action attn in unmask_langact mode)",
            f"    langact_only    : {n_action} positions  ([action] segment)",
            "",
            "[DECODED AR SPANS]",
            f"  decoded[think]    : {decoded_reason!r}",
            f"  decoded[action]   : {decoded_langact!r}",
            "",
            "[FULL TOKEN SEQUENCE (decoded)]",
            f"  {tokenizer.decode(np.asarray(tokens[attn_mask]))!r}",
        ]
        txt_path.write_text("\n".join(lines))

        # Print a short summary to stdout.
        print(f"--- sample {n} ---")
        for ln in lines:
            print(ln)
        print()

        n += 1

    if n < args.num_samples:
        print(f"WARNING: requested {args.num_samples} samples but only {n} were emitted.")
    print(f"Done. Inspect {out_dir}/sample_*.png and sample_*.txt")


if __name__ == "__main__":
    main()
