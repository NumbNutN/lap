"""Audit SSAA-v3 annotation outputs.

Per-episode checks (one CSV row per annotation file):
  - parse_ok          JSON parses and has the expected top-level shape
  - n_kf_match        number of keyframes matches meta.json
  - frame_idx_match   all frame_idx values match meta.json
  - bounds_ok         every kf has frame_idx < chunk_end_frame ≤ frame_idx+60
  - monotone_ok       once imitation_supervised=false, all subsequent are false
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
import argparse, csv, glob, json, os, sys
from pathlib import Path

REQUIRED_KF_FIELDS = ["S", "S_pred", "A", "A_pred", "phase_type",
                      "chunk_end_frame", "imitation_supervised"]


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
        "monotone_ok": True,
        "n_imit_true": 0,
        "n_imit_false": 0,
        "n_overlap_pairs": 0,
        "mean_phase_len": 0.0,
        "max_phase_len": 0,
        "desc_ok": False,
        "missing_fields": "",
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

    for i, kf in enumerate(ann_kfs):
        for f in REQUIRED_KF_FIELDS:
            if f not in kf:
                missing.append(f"kf{i}:{f}")
                continue
            if f in {"S", "S_pred", "A", "A_pred", "phase_type"} and not str(kf[f]).strip():
                missing.append(f"kf{i}:{f}=empty")
        if isinstance(kf.get("phase_type"), str):
            phase_types_seq.append(kf["phase_type"])

        fi = int(kf.get("frame_idx", 0))
        ce = kf.get("chunk_end_frame")
        frame_idxs.append(fi)
        if isinstance(ce, int):
            chunk_ends.append(ce)
            if not (fi < ce <= fi + 60):
                row["bounds_ok"] = False
            phase_lens.append(ce - fi)
        else:
            chunk_ends.append(-1)

        imit = kf.get("imitation_supervised")
        if isinstance(imit, bool):
            imit_seq.append(imit)
            if imit: row["n_imit_true"] += 1
            else:    row["n_imit_false"] += 1

    # Monotone check + first diverge index
    seen_false = False
    for i, v in enumerate(imit_seq):
        if v is False and not seen_false:
            seen_false = True
            row["first_diverge_kf"] = i
        if seen_false and v is True:
            row["monotone_ok"] = False
            break

    # Overlap: kf[i].chunk_end_frame > kf[i+1].frame_idx
    for i in range(len(ann_kfs) - 1):
        if chunk_ends[i] > 0 and chunk_ends[i] > frame_idxs[i + 1]:
            row["n_overlap_pairs"] += 1

    if phase_lens:
        row["mean_phase_len"] = round(sum(phase_lens) / len(phase_lens), 1)
        row["max_phase_len"] = max(phase_lens)

    row["missing_fields"] = ",".join(missing) if missing else ""
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

    ep_dirs = sorted(d for d in glob.glob(f"{args.raw_root}/ep*")
                     if os.path.isdir(d))
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

    # Summary print
    n_ok = sum(1 for r in rows if r["parse_ok"] and r["n_kf_match"]
               and r["bounds_ok"] and r["monotone_ok"]
               and not r["missing_fields"])
    print(f"\n=== Summary ===")
    print(f"  files audited:    {len(rows)}")
    print(f"  fully clean:      {n_ok}")
    print(f"  parse fail:       {sum(1 for r in rows if not r['parse_ok'])}")
    print(f"  kf count mismatch:{sum(1 for r in rows if not r['n_kf_match'])}")
    print(f"  bounds violation: {sum(1 for r in rows if not r['bounds_ok'])}")
    print(f"  monotone fail:    {sum(1 for r in rows if not r['monotone_ok'])}")
    print(f"  missing fields:   {sum(1 for r in rows if r['missing_fields'])}")
    print(f"  no description:   {sum(1 for r in rows if not r['desc_ok'])}")
    if rows:
        avg_overlap = sum(r["n_overlap_pairs"] for r in rows) / len(rows)
        avg_phase = sum(r["mean_phase_len"] for r in rows) / len(rows)
        avg_imitf = sum(r["n_imit_false"] for r in rows) / len(rows)
        print(f"  avg overlap pairs/ep: {avg_overlap:.2f}")
        print(f"  avg mean phase len:   {avg_phase:.1f} frames")
        print(f"  avg n_imit_false/ep:  {avg_imitf:.2f}")

    # Surface non-clean files
    bad = [r for r in rows if not (r["parse_ok"] and r["n_kf_match"]
                                   and r["bounds_ok"] and r["monotone_ok"]
                                   and not r["missing_fields"])]
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
