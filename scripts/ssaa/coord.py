#!/usr/bin/env python3
"""SSAA distributed-annotation coordinator (server-side, stdlib only).

Runs ON the data server. Manages a single locked `state.json` over the DROID
raw dataset and serves atomic claim/submit operations that the local client
drives over SSH. No daemon, no open ports — every call is a short-lived,
flock-guarded command, which is robust on a read-only-home k8s pod.

Layout (under SSAA_DIR, default /localdisk-tmp/ssaa):
  state.json          ep_uuid -> {rel, lab, outcome, task, status, hint,
                                  claimed_by, claimed_at, annotated}
  state.lock          flock target for state.json mutations
  hints/<uuid>.txt    human hint
  annotations/<uuid>/annotation_subagent_v3.json (+ .audit.json)  (scp'd here)

status flow: available -> hint_claimed -> hinted -> annot_claimed -> annotated

Commands (all print JSON to stdout):
  init [--force]                 scan dataset -> state.json
  stats                          counts by status/outcome
  claim --role hint|annot --n K [--outcome success|failure] --worker W
  set-hint --uuid U --hint TEXT   (or --stdin: {uuid: hint, ...})
  mark-annotated --uuid U         (or --stdin: [uuid, ...])
  bulk-import --stdin             {uuid: {hint?, annotated?, status?}}
  release [--role ...] [--older-than SECONDS] [--worker W]
  list --status S [--limit N]
  resolve --rel REL  / --uuid U   helper lookups
"""
from __future__ import annotations
import argparse, fcntl, glob, json, os, sys, time

SSAA_DIR = os.environ.get("SSAA_DIR", "/localdisk-tmp/ssaa")
DATA_ROOT = os.environ.get("SSAA_DATA_ROOT",
                           "/localdisk-tmp/datasets/droid_raw/1.0.1")
STATE = os.path.join(SSAA_DIR, "state.json")
LOCK = os.path.join(SSAA_DIR, "state.lock")
HINTS = os.path.join(SSAA_DIR, "hints")
ANNOTS = os.path.join(SSAA_DIR, "annotations")


def _ensure_dirs():
    for d in (SSAA_DIR, HINTS, ANNOTS):
        os.makedirs(d, exist_ok=True)


class _Locked:
    """flock + load/save state.json as a context manager."""
    def __init__(self, write=True):
        self.write = write
    def __enter__(self):
        _ensure_dirs()
        self.fh = open(LOCK, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX if self.write else fcntl.LOCK_SH)
        self.state = json.load(open(STATE)) if os.path.exists(STATE) else {"episodes": {}}
        return self.state
    def __exit__(self, *a):
        if self.write and a[0] is None:
            tmp = STATE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp, STATE)
        fcntl.flock(self.fh, fcntl.LOCK_UN); self.fh.close()


def _scan():
    """Walk DATA_ROOT, read each metadata json, return {uuid: record}."""
    eps = {}
    for mp in glob.glob(os.path.join(DATA_ROOT, "*", "*", "*", "*",
                                     "metadata_*.json")):
        try:
            m = json.load(open(mp))
        except Exception:
            continue
        uuid = m.get("uuid")
        epdir = os.path.dirname(mp)
        if not (uuid and os.path.exists(os.path.join(epdir, "trajectory.h5"))):
            continue
        rel = os.path.relpath(epdir, DATA_ROOT)
        outcome = "success" if m.get("success") else "failure"
        eps[uuid] = {
            "rel": rel, "lab": m.get("lab", ""), "outcome": outcome,
            "task": m.get("current_task", ""), "status": "available",
            "hint": None, "claimed_by": None, "claimed_at": None,
            "annotated": False,
        }
    return eps


def cmd_init(args):
    _ensure_dirs()
    with _Locked() as st:
        found = _scan()
        if st["episodes"] and not args.force:
            # merge: add new episodes, keep existing status/hint
            added = 0
            for u, rec in found.items():
                if u not in st["episodes"]:
                    st["episodes"][u] = rec; added += 1
            print(json.dumps({"merged": True, "added": added,
                              "total": len(st["episodes"])}))
        else:
            st["episodes"] = found
            print(json.dumps({"initialized": True, "total": len(found)}))


def _counts(st):
    c = {"total": 0, "by_status": {}, "by_outcome": {}}
    hinted_out = {"success": 0, "failure": 0}
    annot_out = {"success": 0, "failure": 0}
    for r in st["episodes"].values():
        c["total"] += 1
        c["by_status"][r["status"]] = c["by_status"].get(r["status"], 0) + 1
        c["by_outcome"][r["outcome"]] = c["by_outcome"].get(r["outcome"], 0) + 1
        if r.get("hint"):
            hinted_out[r["outcome"]] = hinted_out.get(r["outcome"], 0) + 1
        if r.get("annotated"):
            annot_out[r["outcome"]] = annot_out.get(r["outcome"], 0) + 1
    c["hinted"] = hinted_out
    c["annotated"] = annot_out
    return c


def cmd_stats(args):
    with _Locked(write=False) as st:
        print(json.dumps(_counts(st), indent=2))


def cmd_claim(args):
    want_status = "available" if args.role == "hint" else "hinted"
    picked = []
    with _Locked() as st:
        for u, r in st["episodes"].items():
            if len(picked) >= args.n:
                break
            if r["status"] != want_status:
                continue
            if args.outcome and r["outcome"] != args.outcome:
                continue
            r["status"] = "hint_claimed" if args.role == "hint" else "annot_claimed"
            r["claimed_by"] = args.worker
            r["claimed_at"] = int(time.time())
            picked.append({"uuid": u, "rel": r["rel"], "outcome": r["outcome"],
                           "task": r["task"], "hint": r.get("hint")})
    print(json.dumps(picked))


def _set_hint(st, uuid, hint):
    r = st["episodes"].get(uuid)
    if not r:
        return False
    r["hint"] = hint
    # update the hint text but never DOWNGRADE an ep already past hinting
    # (annot_claimed / annotated) — only advance available/hint_claimed.
    if r["status"] in ("available", "hint_claimed"):
        r["status"] = "hinted"
        r["claimed_by"] = None
    try:
        with open(os.path.join(HINTS, uuid + ".txt"), "w") as f:
            f.write(hint)
    except Exception:
        pass
    return True


def cmd_set_hint(args):
    with _Locked() as st:
        if args.stdin:
            data = json.load(sys.stdin)
            ok = sum(1 for u, h in data.items() if _set_hint(st, u, h))
            print(json.dumps({"set": ok, "of": len(data)}))
        else:
            print(json.dumps({"ok": _set_hint(st, args.uuid, args.hint)}))


def cmd_mark_annotated(args):
    with _Locked() as st:
        uuids = json.load(sys.stdin) if args.stdin else [args.uuid]
        ok = 0
        for u in uuids:
            r = st["episodes"].get(u)
            if r:
                r["annotated"] = True
                r["status"] = "annotated"
                r["claimed_by"] = None
                ok += 1
        print(json.dumps({"marked": ok, "of": len(uuids)}))


def cmd_bulk_import(args):
    """{uuid: {hint?, annotated?, status?}} — for seeding already-done eps."""
    data = json.load(sys.stdin)
    with _Locked() as st:
        ok = 0
        for u, fields in data.items():
            r = st["episodes"].get(u)
            if not r:
                continue
            if "hint" in fields and fields["hint"]:
                _set_hint(st, u, fields["hint"])
            if fields.get("annotated"):
                r["annotated"] = True; r["status"] = "annotated"
            elif fields.get("status"):
                r["status"] = fields["status"]
            ok += 1
        print(json.dumps({"imported": ok, "of": len(data)}))


def cmd_release(args):
    now = int(time.time())
    claimed = {"hint": "hint_claimed", "annot": "annot_claimed"}
    targets = [claimed[args.role]] if args.role else list(claimed.values())
    back = {"hint_claimed": "available", "annot_claimed": "hinted"}
    with _Locked() as st:
        n = 0
        for r in st["episodes"].values():
            if r["status"] in targets:
                if args.worker and r.get("claimed_by") != args.worker:
                    continue
                if args.older_than and r.get("claimed_at") and \
                   now - r["claimed_at"] < args.older_than:
                    continue
                r["status"] = back[r["status"]]
                r["claimed_by"] = None; r["claimed_at"] = None
                n += 1
        print(json.dumps({"released": n}))


def cmd_list(args):
    with _Locked(write=False) as st:
        out = [{"uuid": u, "rel": r["rel"], "outcome": r["outcome"]}
               for u, r in st["episodes"].items()
               if (not args.status or r["status"] == args.status)]
        print(json.dumps(out[:args.limit]))


def cmd_resolve(args):
    with _Locked(write=False) as st:
        if args.uuid:
            print(json.dumps(st["episodes"].get(args.uuid)))
        else:
            for u, r in st["episodes"].items():
                if r["rel"] == args.rel or r["rel"].endswith(args.rel):
                    print(json.dumps({"uuid": u, **r})); return
            print(json.dumps(None))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_init)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("claim"); p.add_argument("--role", required=True, choices=["hint", "annot"])
    p.add_argument("--n", type=int, default=1); p.add_argument("--outcome", choices=["success", "failure"])
    p.add_argument("--worker", default="local"); p.set_defaults(fn=cmd_claim)
    p = sub.add_parser("set-hint"); p.add_argument("--uuid"); p.add_argument("--hint")
    p.add_argument("--stdin", action="store_true"); p.set_defaults(fn=cmd_set_hint)
    p = sub.add_parser("mark-annotated"); p.add_argument("--uuid"); p.add_argument("--stdin", action="store_true"); p.set_defaults(fn=cmd_mark_annotated)
    sub.add_parser("bulk-import").set_defaults(fn=cmd_bulk_import)
    p = sub.add_parser("release"); p.add_argument("--role", choices=["hint", "annot"]); p.add_argument("--older-than", type=int); p.add_argument("--worker"); p.set_defaults(fn=cmd_release)
    p = sub.add_parser("list"); p.add_argument("--status"); p.add_argument("--limit", type=int, default=10000); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("resolve"); p.add_argument("--uuid"); p.add_argument("--rel"); p.set_defaults(fn=cmd_resolve)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
