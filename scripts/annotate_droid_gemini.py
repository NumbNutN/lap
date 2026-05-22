"""CLI: annotate DROID episodes with Gemini 2.5 Pro.

Requires GOOGLE_API_KEY in the environment.

Usage::

    # Path A — straight from DROID RLDS root
    GOOGLE_API_KEY=... \\
    uv run python policy/lap/scripts/annotate_droid_gemini.py \\
        --rlds-dir /data/droid \\
        --output   /data/droid_cot/gemini_v0.1.jsonl \\
        --max-episodes 100

    # Path B — from a pre-decoded JSONL (offline pilot)
    GOOGLE_API_KEY=... \\
    uv run python policy/lap/scripts/annotate_droid_gemini.py \\
        --jsonl  /data/droid_pilot/manifest.jsonl \\
        --images /data/droid_pilot/images \\
        --output /data/droid_cot/gemini_v0.1_pilot.jsonl

Cost reminder: full DROID 76k ≈ $1500 at gemini-2.5-pro current pricing.
Pilot 100 ≈ $2.

Concurrency: this script is single-threaded by design (vLLM and API rate
limits both serve sequential workers well). If you need parallelism,
shard the input (--skip / --max-episodes) and run multiple processes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from annotate_droid.client_gemini import GeminiClient
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

    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--api-key", default=None, help="overrides GOOGLE_API_KEY env")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--request-timeout-s", type=float, default=180.0)

    ap.add_argument("--no-resume", action="store_true")
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

    client = GeminiClient(
        api_key=args.api_key,
        model=args.model,
        request_timeout_s=args.request_timeout_s,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
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
