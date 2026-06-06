"""Audit SSAA-v3 annotation outputs.

Per-episode checks (one CSV row per annotation file):
  - parse_ok          JSON parses and has the expected top-level shape
  - n_kf_match        number of keyframes matches meta.json
  - frame_idx_match   all frame_idx values match meta.json
  - bounds_ok         every kf has frame_idx < chunk_end_frame ≤ frame_idx+60
  - n_recoveries      count of imitation_supervised F→T (legal demo self-recovery)
  - n_imit_true       count of imitation_supervised=true
  - n_imit_false      count of imitation_supervised=false
  - n_overlap_pairs   pairs (i, i+1) where kf[i].chunk_end_frame > kf[i+1].frame_idx
  - mean_phase_len    mean chunk_end_frame − frame_idx over all kfs
  - max_phase_len     max chunk_end_frame − frame_idx
  - desc_ok           description non-empty
  - missing_fields    comma list of kfs missing required fields (S/S_pred/A/A_pred)
  - first_diverge_kf  kf_idx where imitation_supervised first goes false (-1 if never)

Usage:
  python3 audit_v3.py \
      --raw-root /home/numbnut/worksapce/RoboTwin/policy/lap/local_data/raw_eps \
      --pattern  'annotation_*_v3*.json' \
      --out      /tmp/audit_v3.csv
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys
from pathlib import Path

# S + phase_type + structural fields on every keyframe. S_pred/A are present
# on acting keyframes; begin/end brackets are S-only (A/S_pred/A_correct null).
REQUIRED_KF_FIELDS = ["S", "phase_type", "chunk_end_frame", "imitation_supervised"]
NONEMPTY_FIELDS = {"S", "phase_type"}     # always non-empty
ACTING_FIELDS = {"S_pred", "A"}           # non-empty unless begin/end bracket

# Foreshadow: S / S_pred must describe the realized present state, never an
# impending action ("about to release", "will open", "is going to lift").
_FORESHADOW = re.compile(
    r"\babout to\b|\b(?:is|are)\s+going to\b|\bwill\s+"
    r"(?:open|close|shut|lift|raise|lower|grasp|grip|release|let go|move|begin|"
    r"start|descend|rise|pull|push|insert|place|drop|reach|swing|rotate|retract|"
    r"tilt|pour|tip|slide|press)\b", re.I)


def load_meta(ep_dir: str) -> dict | None:
    p = os.path.join(ep_dir, "meta.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def audit_annotation(ann_path: str, meta: dict) -> dict:
    row: dict = {
        "file": os.path.relpath(ann_path,
                                start=os.path.dirname(os.path.dirname(ann_path))),
        "parse_ok": False,
        "n_kf_match": False,
        "frame_idx_match": False,
        "bounds_ok": True,
        "monotone_ok": True,   # deprecated: imit may legally recover (F→T)
        "n_recoveries": 0,     # count of F→T transitions (demo self-recovery)
        "n_imit_true": 0,
        "n_imit_false": 0,
        "n_overlap_pairs": 0,
        "n_tiled": 0,        # boundaries where chunk_end == next kf frame_idx
        "frac_tiled": 0.0,   # n_tiled / acting boundaries (WARNING metric, not gated)
        "mean_phase_len": 0.0,
        "max_phase_len": 0,
        "desc_ok": False,
        "missing_fields": "",
        "gate_ok": True,
        "gate_issues": "",
        "n_spred_echoes_a": 0,
        "n_foreshadow": 0,
        "first_diverge_kf": -1,
        "n_tool_calls": 0,
        "n_image_reads_claimed": 0,
        "audit_self_present": False,
        "phase_types": "",
        "errors": "",
    }
    try:
        ann = json.load(open(ann_path))
    except Exception as e:
        row["errors"] = f"json_parse:{type(e).__name__}:{e}"
        return row
    if not isinstance(ann, dict) or "keyframes" not in ann:
        row["errors"] = "shape:missing-keyframes"
        return row
    row["parse_ok"] = True

    row["desc_ok"] = bool(ann.get("description", "").strip())

    meta_kfs = meta["keyframes"]
    ann_kfs = ann["keyframes"]
    row["n_kf_match"] = (len(ann_kfs) == len(meta_kfs))
    row["frame_idx_match"] = row["n_kf_match"] and all(
        a.get("frame_idx") == m["frame_idx"]
        for a, m in zip(ann_kfs, meta_kfs)
    )

    missing = []
    phase_lens: list[int] = []
    imit_seq: list[bool] = []
    chunk_ends: list[int] = []
    frame_idxs: list[int] = []
    phase_types_seq: list[str] = []

    gate_issues = []
    first_acting_seen = False
    for i, kf in enumerate(ann_kfs):
        is_bracket = str(kf.get("phase_type", "")).strip() in ("begin", "end")
        a_txt = str(kf.get("A") or "")
        corr_txt = str(kf.get("A_correct") or "")
        imit_g = kf.get("imitation_supervised")
        think_in_a = "<think>" in a_txt

        # presence / non-empty
        for f in REQUIRED_KF_FIELDS:
            if f not in kf:
                missing.append(f"kf{i}:{f}")
            elif f in NONEMPTY_FIELDS and not str(kf[f]).strip():
                missing.append(f"kf{i}:{f}=empty")
        for f in ACTING_FIELDS:                      # S_pred / A
            if not is_bracket and not str(kf.get(f) or "").strip():
                missing.append(f"kf{i}:{f}=empty")
        if isinstance(kf.get("phase_type"), str):
            phase_types_seq.append(kf["phase_type"])

        # Foreshadow gate (S on every kf; S_pred on acting kfs): present-only.
        if _FORESHADOW.search(str(kf.get("S") or "")):
            gate_issues.append(f"kf{i}:foreshadow-in-S")
            row["n_foreshadow"] += 1

        # Schema gate.
        if is_bracket:
            # begin/end are S-only: A / S_pred / A_correct must all be null
            if a_txt.strip() or str(kf.get("S_pred") or "").strip() or corr_txt.strip():
                gate_issues.append(f"kf{i}:bracket-not-S-only")
        else:
            #   (1) A_correct present iff imitation_supervised == false
            #   (2) <think> in A only when supervised (a failure's A stays plain)
            #   (3) the FIRST acting keyframe carries the Plan <think> in its
            #       policy-target field (A when supervised, A_correct when not)
            if bool(corr_txt.strip()) != (imit_g is False):
                gate_issues.append(
                    f"kf{i}:A_correct={'set' if corr_txt.strip() else 'null'}!=imit{imit_g}")
            if imit_g is False and think_in_a:
                gate_issues.append(f"kf{i}:think-in-A-on-failure")
            if not first_acting_seen:
                first_acting_seen = True
                target = corr_txt if imit_g is False else a_txt
                if "<think>" not in target:
                    gate_issues.append(f"kf{i}:first-move-no-plan-think")
            # S_pred should not echo A's cm/° (only on acting kfs) → now gated
            sp_txt = str(kf.get("S_pred") or "")
            sp_nums = set(re.findall(r"\d+\s*cm|\d+\s*°", sp_txt))
            if sp_nums and (sp_nums & set(re.findall(r"\d+\s*cm|\d+\s*°", a_txt))):
                row["n_spred_echoes_a"] += 1
                gate_issues.append(f"kf{i}:spred-echo-A")
            if _FORESHADOW.search(sp_txt):
                gate_issues.append(f"kf{i}:foreshadow-in-S_pred")
                row["n_foreshadow"] += 1

        fi = int(kf.get("frame_idx", 0))
        ce = kf.get("chunk_end_frame")
        frame_idxs.append(fi)
        if isinstance(ce, int):
            chunk_ends.append(ce)
            # brackets are S-only: their chunk_end is vacuous, skip the bounds check
            if not is_bracket and not (fi < ce <= fi + 60):
                row["bounds_ok"] = False
            phase_lens.append(ce - fi)
        else:
            chunk_ends.append(-1)

        imit = kf.get("imitation_supervised")
        if isinstance(imit, bool):
            if imit: row["n_imit_true"] += 1
            else:    row["n_imit_false"] += 1
            if not is_bracket:        # brackets' imit is vacuous (S-only)
                imit_seq.append((i, imit))

    # First diverge index + self-recoveries (F→T) over ACTING keyframes only —
    # imit is not monotone (a demo may make a recoverable detour and rejoin),
    # and begin/end brackets' imit doesn't denote an action.
    seen_false = False
    for j, (kf_i, v) in enumerate(imit_seq):
        if v is False and not seen_false:
            seen_false = True
            row["first_diverge_kf"] = kf_i
        if j > 0 and imit_seq[j - 1][1] is False and v is True:
            row["n_recoveries"] += 1

    # Overlap (shared chunk_end, expected) vs tiling (chunk_end == next frame_idx,
    # the degenerate "every kf its own chunk" anti-pattern — warned, not gated:
    # legitimate for truly intent-less episodes, harmful for task episodes).
    n_bound = 0
    for i in range(len(ann_kfs) - 1):
        if chunk_ends[i] > 0:
            n_bound += 1
            if chunk_ends[i] > frame_idxs[i + 1]:
                row["n_overlap_pairs"] += 1
            elif chunk_ends[i] == frame_idxs[i + 1]:
                row["n_tiled"] += 1
    if n_bound:
        row["frac_tiled"] = round(row["n_tiled"] / n_bound, 2)

    if phase_lens:
        row["mean_phase_len"] = round(sum(phase_lens) / len(phase_lens), 1)
        row["max_phase_len"] = max(phase_lens)

    row["missing_fields"] = ",".join(missing) if missing else ""
    row["gate_issues"] = ",".join(gate_issues) if gate_issues else ""
    row["gate_ok"] = not gate_issues
    if phase_types_seq:
        # Compact "type1×3,type2×2" form
        from collections import OrderedDict
        counts: dict[str, int] = OrderedDict()
        for t in phase_types_seq:
            counts[t] = counts.get(t, 0) + 1
        row["phase_types"] = ",".join(f"{t}×{n}" for t, n in counts.items())

    # Cross-check: tool_audit.jsonl (ground truth) vs companion .audit.json (self-report)
    ep_dir = os.path.dirname(ann_path)
    tool_log_path = os.path.join(ep_dir, ".tool_audit.jsonl")
    if os.path.exists(tool_log_path):
        try:
            with open(tool_log_path) as f:
                row["n_tool_calls"] = sum(1 for _ in f if _.strip())
        except Exception:
            pass
    self_audit_path = ann_path + ".audit.json"
    if os.path.exists(self_audit_path):
        row["audit_self_present"] = True
        try:
            sa = json.load(open(self_audit_path))
            ir = sa.get("image_reads") or []
            row["n_image_reads_claimed"] = len(ir) if isinstance(ir, list) else 0
        except Exception:
            pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True,
                    help="root containing per-ep dirs (each with meta.json)")
    ap.add_argument("--pattern", default="annotation_*_v3*.json",
                    help="glob pattern for v3 annotation filenames")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()

    # any subdir that holds a meta.json is an episode dir (ep000… or uuid-named)
    ep_dirs = sorted(d for d in glob.glob(f"{args.raw_root}/*")
                     if os.path.isdir(d) and os.path.exists(os.path.join(d, "meta.json")))
    print(f"Scanning {len(ep_dirs)} episode dirs for {args.pattern!r}")

    rows: list[dict] = []
    for ep_dir in ep_dirs:
        meta = load_meta(ep_dir)
        if meta is None:
            print(f"  [skip] {os.path.basename(ep_dir)}: no meta.json")
            continue
        for ann_path in sorted(glob.glob(os.path.join(ep_dir, args.pattern))):
            row = audit_annotation(ann_path, meta)
            rows.append(row)

    if not rows:
        print("No annotation files matched.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {args.out}")

    # Summary print — "fully clean" now includes the schema gate (gate_ok),
    # which folds in brackets, think-in-A, first-move-plan, S_pred echo, and
    # foreshadow. The gate is the real pass/fail; a clean file fails NOTHING.
    def _clean(r):
        return (r["parse_ok"] and r["n_kf_match"] and r["bounds_ok"]
                and r["monotone_ok"] and not r["missing_fields"] and r["gate_ok"])
    n_ok = sum(1 for r in rows if _clean(r))
    print(f"\n=== Summary ===")
    print(f"  files audited:    {len(rows)}")
    print(f"  fully clean:      {n_ok}")
    print(f"  parse fail:       {sum(1 for r in rows if not r['parse_ok'])}")
    print(f"  kf count mismatch:{sum(1 for r in rows if not r['n_kf_match'])}")
    print(f"  bounds violation: {sum(1 for r in rows if not r['bounds_ok'])}")
    print(f"  gate fail:        {sum(1 for r in rows if not r['gate_ok'])}")
    print(f"  S_pred echo:      {sum(1 for r in rows if r['n_spred_echoes_a'])}"
          f" files ({sum(r['n_spred_echoes_a'] for r in rows)} kfs)")
    print(f"  foreshadow:       {sum(1 for r in rows if r['n_foreshadow'])}"
          f" files ({sum(r['n_foreshadow'] for r in rows)} kfs)")
    print(f"  missing fields:   {sum(1 for r in rows if r['missing_fields'])}")
    print(f"  no description:   {sum(1 for r in rows if not r['desc_ok'])}")
    over_tiled = [r for r in rows if r["frac_tiled"] >= 0.8]
    if over_tiled:
        print(f"  ⚠ over-tiled (≥80% boundaries, NOT gated): {len(over_tiled)} "
              f"— review chunk_end grouping (ok for intent-less eps):")
        for r in over_tiled:
            print(f"      {os.path.basename(os.path.dirname(r['file']))}"
                  f"  frac_tiled={r['frac_tiled']}")
    if rows:
        avg_overlap = sum(r["n_overlap_pairs"] for r in rows) / len(rows)
        avg_phase = sum(r["mean_phase_len"] for r in rows) / len(rows)
        avg_imitf = sum(r["n_imit_false"] for r in rows) / len(rows)
        print(f"  avg overlap pairs/ep: {avg_overlap:.2f}")
        print(f"  avg mean phase len:   {avg_phase:.1f} frames")
        print(f"  avg n_imit_false/ep:  {avg_imitf:.2f}")

    # Surface non-clean files
    bad = [r for r in rows if not _clean(r)]
    if bad:
        print(f"\n=== Files needing attention ({len(bad)}) ===")
        for r in bad[:20]:
            print(f"  {r['file']}")
            if r["errors"]:        print(f"    errors: {r['errors']}")
            if not r["n_kf_match"]: print(f"    n_kf_match=false")
            if not r["bounds_ok"]:  print(f"    bounds violation")
            if not r["monotone_ok"]:print(f"    monotone_ok=false")
            if r["missing_fields"]: print(f"    missing: {r['missing_fields'][:120]}")


if __name__ == "__main__":
    main()
