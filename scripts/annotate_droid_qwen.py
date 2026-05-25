"""CLI: annotate DROID episodes with Qwen2.5-VL-72B.

Two backend modes:

  ``--mode local``  (default): load the model in-process via HF transformers
                    + qwen_vl_utils. No vLLM server needed. Easiest path
                    when the weights are on the same machine as the dataset.

  ``--mode vllm``   : talk to a separately-launched vLLM OpenAI-compatible
                    endpoint. Better throughput at scale but adds a process
                    + network hop.

Examples::

    # Local HF transformers (single process, on the same pod)
    uv run python policy/lap/scripts/annotate_droid_qwen.py \\
        --mode local \\
        --model-path /data/zhaoqc/RoboTwin/policy/lap/Qwen2.5-VL-72B-Instruct \\
        --rlds-dir /data/datasets/droid_data_template \\
        --output   /data/zhaoqc/droid_cot/qwen_v0.1_pilot.jsonl \\
        --max-episodes 100

    # vLLM HTTP endpoint
    # (first launch separately:
    #   uv run vllm serve <model_path> --tensor-parallel-size 2 \\
    #       --max-model-len 32768 --limit-mm-per-prompt image=20 --port 8100 )
    uv run python policy/lap/scripts/annotate_droid_qwen.py \\
        --mode vllm \\
        --base-url http://<h200_host>:8100/v1 \\
        --rlds-dir /data/datasets/droid_data_template \\
        --output   /data/zhaoqc/droid_cot/qwen_v0.1_pilot.jsonl \\
        --max-episodes 100
"""

from __future__ import annotations

import argparse
import logging
import sys

# Path hack so we can run from the repo root without installing the package.
import os
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from annotate_droid.droid_reader import iter_droid_rlds
from annotate_droid.droid_reader import iter_jsonl
from annotate_droid.runner import run_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rlds-dir", help="DROID TFDS root (path A)")
    src.add_argument("--jsonl", help="pre-decoded JSONL manifest (path B)")
    ap.add_argument("--images", help="image root for --jsonl path B (required if --jsonl)")
    ap.add_argument("--rlds-name", default="droid",
                    help="TFDS builder name (default: droid; use droid_100 for the 100-ep subset)")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)

    ap.add_argument("--mode", choices=["local", "vllm"], default="local",
                    help="local = in-process HF transformers; vllm = OpenAI-compatible HTTP")

    # local mode args
    ap.add_argument("--model-path",
                    default="policy/lap/Qwen2.5-VL-72B-Instruct",
                    help="local HF-format checkpoint directory (mode=local)")
    ap.add_argument("--torch-dtype", default="auto",
                    help="auto | bfloat16 | float16 | float32 (mode=local)")
    ap.add_argument("--attn", default=None, choices=[None, "flash_attention_2", "sdpa", "eager"],
                    help="attention impl (mode=local). flash_attention_2 = fastest on H100/H200")
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)

    # vllm mode args
    ap.add_argument("--base-url", default="http://localhost:8100/v1",
                    help="vLLM endpoint (mode=vllm)")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--vllm-model-name", default="Qwen/Qwen2.5-VL-72B-Instruct",
                    help="model name vLLM advertises (mode=vllm)")

    # shared sampling args
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout-s", type=float, default=180.0)

    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing output file (default: skip already-done episodes)")
    ap.add_argument("--no-feed-types", action="store_true",
                    help=("Don't feed our detector's keyframe type/gripper_state "
                          "to the VLM; let it derive these from images "
                          "(experimental mode B for type-fed bias study)."))
    ap.add_argument("--memory-augmented", action="store_true",
                    help=("Use v3 prompt: memory-augmented stage + axis-aware "
                          "actions (uses pose deltas) + mode_marker field. "
                          "Implies --feed-types (incompatible with --no-feed-types)."))
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.jsonl and not args.images:
        ap.error("--images is required when using --jsonl")

    if args.rlds_dir:
        bundles = iter_droid_rlds(
            args.rlds_dir,
            dataset_name=args.rlds_name,
            max_episodes=args.max_episodes,
            skip=args.skip,
        )
    else:
        bundles = iter_jsonl(
            args.jsonl, args.images,
            max_episodes=args.max_episodes,
            skip=args.skip,
        )

    if args.mode == "local":
        from annotate_droid.client_qwen_local import QwenVLLocalClient
        client = QwenVLLocalClient(
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        from annotate_droid.client_qwen import QwenVLClient
        client = QwenVLClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.vllm_model_name,
            request_timeout_s=args.request_timeout_s,
            temperature=args.temperature,
            max_completion_tokens=args.max_new_tokens,
        )

    if args.memory_augmented and args.no_feed_types:
        ap.error("--memory-augmented requires types fed; cannot combine with --no-feed-types")

    counts = run_batch(
        bundles, client,
        output_jsonl=args.output,
        resume=not args.no_resume,
        feed_types=not args.no_feed_types,
        memory_augmented=args.memory_augmented,
    )
    print(f"\nDone. emitted={counts['emitted']} skipped={counts['skipped']} failed={counts['failed']}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
