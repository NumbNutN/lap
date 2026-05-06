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
        "--p-plan",
        type=float,
        default=0.5,  # 0.5 here (vs 0.15 default) so a small dump shows both modes
        help="Probability per sample that plan is emitted as AR target. "
             "Set to 1.0 to force plan-as-target, 0.0 to force plan-as-prompt.",
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
        p_plan=args.p_plan,
        skip_steps_without_change=args.skip_repeat,
    )

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
            ar_target_mask,
            _number_mask,
            _direction_mask,
            _token_loss_mask,
            stage_mask,
            plan_mask,
        ) = tokenizer.tokenize(
            prompt=sample["prompt"],
            reasoning=sample["language_actions"],
            state=None,
            plan=sample.get("plan"),
            plan_position=sample.get("plan_position", "none"),
            langact=sample["langact"],
            is_vqa_sample=False,
            is_prediction_sample=False,
        )

        # Mask stats.
        n_valid = int(attn_mask.sum())
        n_ar = int(ar_target_mask.sum()) if ar_target_mask is not None else 0
        n_plan = int(plan_mask.sum()) if plan_mask is not None else 0
        n_stage = int(stage_mask.sum()) if stage_mask is not None else 0
        n_action = n_ar - n_stage - n_plan

        # Build action-only mask = ar_target & ~stage & ~plan.
        action_only_mask = ar_target_mask
        if stage_mask is not None:
            action_only_mask = action_only_mask & np.logical_not(stage_mask)
        if plan_mask is not None:
            action_only_mask = action_only_mask & np.logical_not(plan_mask)

        # Decode each AR span for human inspection.
        decoded_plan = (
            tokenizer.decode(np.asarray(tokens[plan_mask]))
            if plan_mask is not None and n_plan > 0
            else "(plan in prompt or absent)"
        )
        decoded_stage = (
            tokenizer.decode(np.asarray(tokens[stage_mask]))
            if stage_mask is not None and n_stage > 0
            else "(no stage span)"
        )
        decoded_action = (
            tokenizer.decode(np.asarray(tokens[action_only_mask]))
            if action_only_mask is not None and action_only_mask.sum() > 0
            else "(no action span)"
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
            f"  plan_position     : {sample.get('plan_position', 'none')}",
            f"  plan (raw)        : {sample.get('plan')!r}",
            "",
            "[AR TARGET — what the model is trained to predict]",
            f"  language_actions  : {sample['language_actions']!r}  (→ [stage])",
            f"  langact           : {sample['langact']!r}  (→ [action])",
            "",
            "[TOKENIZER OUTPUT]",
            f"  total tokens      : {len(tokens)}",
            f"  valid (non-pad)   : {n_valid}",
            f"  ar_target_mask    : {n_ar} positions  (= plan + stage + action, used for CE loss)",
            f"    plan_mask       : {n_plan} positions  ([plan] segment; non-empty only when plan_position='target')",
            f"    stage_mask      : {n_stage} positions  ([stage] segment; always blocked from action attn in unmask_langact)",
            f"    action_only     : {n_action} positions  ([action] segment; visible to action attn in unmask_langact)",
            "",
            "[DECODED AR SPANS]",
            f"  decoded[plan]     : {decoded_plan!r}",
            f"  decoded[stage]    : {decoded_stage!r}",
            f"  decoded[action]   : {decoded_action!r}",
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
