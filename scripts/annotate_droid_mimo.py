"""CLI: annotate DROID episodes with Xiaomi MiMo (mimo-v2.5) API.

OpenAI-compatible endpoint at https://api.xiaomimimo.com/v1.

Auth: set ``MIMO_API_KEY`` env var (or pass ``--api-key``).

Examples::

    export MIMO_API_KEY=sk-...
    export https_proxy=http://192.168.3.225:8906  # pod usually needs proxy for outbound
    export http_proxy=$https_proxy

    uv run python policy/lap/scripts/annotate_droid_mimo.py \\
        --rlds-dir   /data/datasets/droid_data_template \\
        --rlds-name  droid_100 \\
        --output     /data/zhaoqc/droid_cot/mimo_pilot_5ep.jsonl \\
        --max-episodes 5 -v
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from annotate_droid.client_mimo import MiMoClient
from annotate_droid.droid_reader import iter_droid_rlds
from annotate_droid.droid_reader import iter_jsonl
from annotate_droid.runner import run_batch


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--rlds-dir", help="DROID TFDS root")
    src.add_argument("--jsonl", help="pre-decoded JSONL manifest")
    ap.add_argument("--images", help="image root for --jsonl path")
    ap.add_argument("--rlds-name", default="droid",
                    help="TFDS builder name (droid or droid_100)")
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0)

    ap.add_argument("--api-key", default=None,
                    help="overrides MIMO_API_KEY env var")
    ap.add_argument("--base-url",
                    default=MiMoClient.DEFAULT_BASE_URL,
                    help="MiMo API base URL")
    ap.add_argument("--model", default="mimo-v2.5")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
    ap.add_argument("--request-timeout-s", type=float, default=300.0)

    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-feed-types", action="store_true",
                    help="don't feed detector types to VLM (mode B)")
    ap.add_argument("--memory-augmented", action="store_true",
                    help="use v3 prompt (memory + axis-aware + mode_marker)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    if args.memory_augmented and args.no_feed_types:
        ap.error("--memory-augmented requires types fed")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.jsonl and not args.images:
        ap.error("--images is required when using --jsonl")

    if args.rlds_dir:
        bundles = iter_droid_rlds(
            args.rlds_dir, dataset_name=args.rlds_name,
            max_episodes=args.max_episodes, skip=args.skip,
        )
    else:
        bundles = iter_jsonl(
            args.jsonl, args.images,
            max_episodes=args.max_episodes, skip=args.skip,
        )

    client = MiMoClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        request_timeout_s=args.request_timeout_s,
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )

    counts = run_batch(
        bundles, client,
        output_jsonl=args.output,
        resume=not args.no_resume,
        feed_types=not args.no_feed_types,
        memory_augmented=args.memory_augmented,
    )
    print(f"\nDone. emitted={counts['emitted']} skipped={counts['skipped']} "
          f"failed={counts['failed']}")
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
