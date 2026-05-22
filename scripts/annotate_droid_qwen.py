"""CLI: annotate DROID episodes with self-hosted Qwen2.5-VL-72B.

Spin up vLLM on the H200 host first::

    uv run vllm serve Qwen/Qwen2.5-VL-72B-Instruct \\
        --tensor-parallel-size 2 \\
        --max-model-len 32768 \\
        --limit-mm-per-prompt image=20 \\
        --port 8100

Then run on the annotation worker::

    # Path A — straight from DROID RLDS root
    uv run python policy/lap/scripts/annotate_droid_qwen.py \\
        --rlds-dir /data/droid \\
        --output  /data/droid_cot/qwen_v0.1.jsonl \\
        --max-episodes 100 \\
        --base-url http://<h200_host>:8100/v1

    # Path B — from a pre-decoded JSONL (offline pilot)
    uv run python policy/lap/scripts/annotate_droid_qwen.py \\
        --jsonl  /data/droid_pilot/manifest.jsonl \\
        --images /data/droid_pilot/images \\
        --output /data/droid_cot/qwen_v0.1_pilot.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys

# Path hack so we can run from the repo root without installing the package.
import os
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from annotate_droid.client_qwen import QwenVLClient
from annotate_droid.droid_reader import iter_droid_rlds
from annotate_droid.droid_reader import iter_jsonl
from annotate_droid.runner import run_batch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rlds-dir", help="DROID TFDS root (path A)")
    src.add_argument("--jsonl", help="pre-decoded JSONL manifest (path B)")
    ap.add_argument("--images", help="image root for --jsonl path B (required if --jsonl)")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)

    ap.add_argument("--base-url", default="http://localhost:8100/v1",
                    help="vLLM OpenAI-compatible endpoint")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-completion-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout-s", type=float, default=180.0)

    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing output file (default: skip already-done episodes)")
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
            max_episodes=args.max_episodes,
            skip=args.skip,
        )
    else:
        bundles = iter_jsonl(
            args.jsonl, args.images,
            max_episodes=args.max_episodes,
            skip=args.skip,
        )

    client = QwenVLClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        request_timeout_s=args.request_timeout_s,
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
    )

    counts = run_batch(
        bundles, client,
        output_jsonl=args.output,
        resume=not args.no_resume,
    )
    print(f"\nDone. emitted={counts['emitted']} skipped={counts['skipped']} failed={counts['failed']}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
