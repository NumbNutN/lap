"""Dump the per-position loss mask coverage for a real Bridge ECoT sample.

Goal: empirically confirm whether the literal segment markers (`[plan]`,
`[stage]`, `[action]`) receive next-token CE loss gradient. The hypothesis,
from reading PaligemmaTokenizer._create_segmented_masks docstring lines
193-199, is that markers are NOT covered by ar_target_mask and therefore
the model is never trained to emit them.

The script tokenizes one Bridge sample three different ways (Context 1/2/3),
prints each token's piece + which masks claim it, and then sums the masks so
you can verify whether marker positions show up in ``ar_target_mask`` or not.

Run on pod (CPU, ~5 seconds, no JAX):

    cd /data/zhaoqc/RoboTwin/policy/lap
    .venv/bin/python scripts/inspect_marker_loss_coverage.py
"""

from __future__ import annotations

import pathlib
import sys

_THIS = pathlib.Path(__file__).resolve().parent
_LAP_ROOT = _THIS.parent
sys.path.insert(0, str(_LAP_ROOT / "src"))
sys.path.insert(0, str(_LAP_ROOT / "third_party" / "openpi" / "src"))

import numpy as np  # noqa: E402

from lap.models.tokenizer import PaligemmaTokenizer  # noqa: E402


# Synthetic but representative Bridge ECoT content (any real sample exercises
# the same code path; this avoids requiring the dataset to be loaded).
TASK = "Move the wooden arch onto the table."
PLAN = "Reach for the wooden arch. Grasp the wooden arch. Move the wooden arch to the table. Drop the wooden arch onto the table."
STAGE = "The wooden arch is the object that needs to be moved."
ACTION = "Reach for the wooden arch."


def find_substring_token_ranges(tokenizer: PaligemmaTokenizer, tokens: list[int], needle_text: str) -> list[tuple[int, int]]:
    """Best-effort: locate sub-ranges in the token list whose decoded piece sequence
    matches the needle text. Used to highlight where the literal markers landed."""
    needle_clean = needle_text.strip()
    sp = tokenizer._tokenizer  # noqa: SLF001
    out = []
    n = len(tokens)
    for start in range(n):
        for end in range(start + 1, min(n, start + 12) + 1):
            piece_seq = "".join(sp.id_to_piece(int(t)) for t in tokens[start:end]).replace("▁", " ").strip()
            if piece_seq == needle_clean:
                out.append((start, end))
                break
    return out


def dump_one(label: str, tok_out, tokenizer: PaligemmaTokenizer):
    tokens, attn_mask, ar_target_mask, _number, _direction, token_loss_mask, stage_mask, plan_mask = tok_out
    n_tokens = int(attn_mask.sum())
    tokens = list(tokens[:n_tokens])
    sp = tokenizer._tokenizer  # noqa: SLF001

    print("=" * 80)
    print(f"  {label}")
    print("=" * 80)
    print(f"  total non-pad tokens         : {n_tokens}")
    print(f"  ar_target_mask sum (AR-loss) : {int(ar_target_mask.sum())}")
    print(f"  plan_mask sum                : {int(plan_mask.sum()) if plan_mask is not None else 'None'}")
    print(f"  stage_mask sum               : {int(stage_mask.sum()) if stage_mask is not None else 'None'}")
    if plan_mask is not None and stage_mask is not None:
        action_mask = ar_target_mask & ~plan_mask & ~stage_mask
        print(f"  action_mask (derived) sum    : {int(action_mask.sum())}")
    print(f"  token_loss_mask sum          : {int(token_loss_mask.sum())}")
    print()

    # Locate marker token ranges so we can call them out explicitly.
    plan_marker_ranges  = find_substring_token_ranges(tokenizer, tokens, "[plan]")
    stage_marker_ranges = find_substring_token_ranges(tokenizer, tokens, "[stage]")
    action_marker_ranges= find_substring_token_ranges(tokenizer, tokens, "[action]")

    def label_role(i: int) -> str:
        if any(s <= i < e for (s, e) in plan_marker_ranges):
            return "MARK[plan]"
        if any(s <= i < e for (s, e) in stage_marker_ranges):
            return "MARK[stage]"
        if any(s <= i < e for (s, e) in action_marker_ranges):
            return "MARK[action]"
        in_plan  = bool(plan_mask is not None  and plan_mask[i])
        in_stage = bool(stage_mask is not None and stage_mask[i])
        in_ar    = bool(ar_target_mask[i])
        if in_plan:
            return "plan_text"
        if in_stage:
            return "stage_text"
        if in_ar:
            return "action_text"
        return "prompt"

    # Dump per-position detail (compact).
    print(f"  {'idx':>4}  {'tok_id':>6}  {'piece':<22}  {'role':<12}  attn ar_tgt loss plan_m stage_m")
    for i in range(n_tokens):
        tid = int(tokens[i])
        piece = sp.id_to_piece(tid).replace("▁", "_")
        role = label_role(i)
        flags = "  ".join([
            "1" if attn_mask[i] else "0",
            "1" if ar_target_mask[i] else "0",
            "1" if token_loss_mask[i] else "0",
            "1" if (plan_mask is not None and plan_mask[i]) else "0",
            "1" if (stage_mask is not None and stage_mask[i]) else "0",
        ])
        print(f"  {i:>4}  {tid:>6}  {piece:<22}  {role:<12}  {flags}")

    # Summarize: among marker positions, how many have ar_target_mask=True?
    marker_ranges_all = [(s, e, "plan") for s, e in plan_marker_ranges] \
                       + [(s, e, "stage") for s, e in stage_marker_ranges] \
                       + [(s, e, "action") for s, e in action_marker_ranges]
    if marker_ranges_all:
        print()
        print("  Marker → AR loss coverage:")
        for s, e, name in marker_ranges_all:
            covered = int(ar_target_mask[s:e].sum())
            print(f"    [{name}] @ tokens [{s}:{e}]  ar_target_mask True count = {covered}/{e - s}")
    print()


def main():
    tok = PaligemmaTokenizer(max_len=320, prompt_format="lap", prediction_format="default")

    # Context 1 — legacy single-segment (no langact, no plan)
    out1 = tok.tokenize(TASK, reasoning=STAGE, langact=None, plan=None, plan_position="none")
    dump_one("Context 1 (legacy single-segment: prompt + reasoning only)", out1, tok)

    # Context 2 — cascade-VLA, plan in prompt
    out2 = tok.tokenize(TASK, reasoning=STAGE, langact=ACTION, plan=PLAN, plan_position="prompt")
    dump_one("Context 2 (cascade-VLA, plan in prompt)", out2, tok)

    # Context 3 — cascade-VLA, plan as AR target
    out3 = tok.tokenize(TASK, reasoning=STAGE, langact=ACTION, plan=PLAN, plan_position="target")
    dump_one("Context 3 (cascade-VLA, plan as AR target)", out3, tok)


if __name__ == "__main__":
    main()
