#!/usr/bin/env python3
"""SSAA distributed-annotation client (local side).

Drives the server `coord.py` over an SSH ControlMaster connection (host alias
`bitahub`, set up once with pexpect — see README) and moves per-episode raw
data with rsync. Extraction, viewing, and subagent annotation happen locally
with the existing pipeline.

Workflow:
  ssaa_client.py stats
  ssaa_client.py claim-hint  --n 5 [--outcome failure]   # → pull + extract; view + write hints
  ssaa_client.py push-hints                              # hints.md → server
  ssaa_client.py claim-annot --n 5                        # → pull + extract + hints ready
  # (launch subagents on RAW_EPS to write annotation_subagent_v3.json)
  ssaa_client.py push-annot                               # local jsonl → server, mark done
  ssaa_client.py import-local-10                          # seed server with the original 10 eps

Paths: raw mirror = ~/datasets/droid_raw/1.0.1 (DROID_RAW_ROOT for tools);
       working set = policy/lap/local_data/raw_eps_remote/ (+ hints.md).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile

REPO = "/home/numbnut/worksapce/RoboTwin"
VENV_PY = f"{REPO}/policy/lap/.venv/bin/python3"
SCRIPTS = f"{REPO}/policy/lap/scripts"
EXTRACT = f"{SCRIPTS}/data_pipeline/extract_raw.py"
MIRROR = os.path.expanduser("~/datasets/droid_raw/1.0.1")
# Working set is per-worker so two Claude sessions on one machine don't collide
# (raw mirror above is shared — different episodes land in different rel paths).
RAW_EPS = os.environ.get("SSAA_RAW_EPS", f"{REPO}/policy/lap/local_data/raw_eps_remote")
HINTS_MD = f"{RAW_EPS}/hints.md"
SSH = "bitahub"                       # ~/.ssh/config alias (ControlMaster)
REMOTE_DATA = "/localdisk-tmp/datasets/droid_raw/1.0.1"
REMOTE_COORD = "cd /localdisk-tmp/ssaa && python3 coord.py"
REMOTE_ANNOT = "/localdisk-tmp/ssaa/annotations"


def _sanitize(uuid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", uuid)


def coord(args: list[str], stdin: str | None = None) -> str:
    """Run a coord.py subcommand on the server; return stdout."""
    cmd = ["ssh", SSH, f"{REMOTE_COORD} " + " ".join(
        (f"'{a}'" if (" " in a or '"' in a) else a) for a in args)]
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[coord error] {' '.join(args)}\n{r.stderr}")
    return r.stdout.strip()


def cmd_stats(_):
    print(coord(["stats"]))


def _rsync_ep(rel: str):
    dst = os.path.join(MIRROR, rel)
    os.makedirs(dst, exist_ok=True)
    src = f"{SSH}:{REMOTE_DATA}/{rel}/"
    r = subprocess.run(["rsync", "-a", "--info=stats0", "-e", "ssh", src, dst + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   [rsync fail] {rel}: {r.stderr.strip()[:160]}")
        return False
    return True


def _extract_batch(eps: list[dict]):
    """rsync each ep into the mirror, extract to a temp dir, rename by uuid,
    move into RAW_EPS. eps = [{uuid, rel, outcome, task, hint?}, ...]."""
    os.makedirs(RAW_EPS, exist_ok=True)
    wl = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    wl.write("ep_id,classification\n")
    relmap = {}
    pulled = 0
    for e in eps:
        if _rsync_ep(e["rel"]):
            wl.write(f"{e['rel']}/trajectory.h5,good\n")
            relmap[f"{e['rel']}/trajectory.h5"] = e
            pulled += 1
    wl.close()
    if not pulled:
        return []
    out = tempfile.mkdtemp(prefix="ssaa_out_")
    env = dict(os.environ, DROID_RAW_ROOT=MIRROR)
    subprocess.run([VENV_PY, EXTRACT, "--root", MIRROR, "--out", out,
                    "--whitelist", wl.name, "--include-failure"],
                   env=env, capture_output=True, text=True)
    done = []
    for name in os.listdir(out):
        d = os.path.join(out, name)
        mp = os.path.join(d, "meta.json")
        if not os.path.isdir(d) or not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        e = relmap.get(meta.get("episode_id"))
        if not e:
            continue
        meta["uuid"] = e["uuid"]; meta["outcome"] = e["outcome"]
        if e.get("task"):
            meta["task_instruction"] = meta.get("task_instruction") or e["task"]
        json.dump(meta, open(mp, "w"), ensure_ascii=False, indent=2)
        target = os.path.join(RAW_EPS, _sanitize(e["uuid"]))
        subprocess.run(["rm", "-rf", target])
        subprocess.run(["mv", d, target])
        done.append((e, target))
    os.unlink(wl.name)
    return done


def _seed_hints_md(done, with_hint: bool):
    """Append/update hints.md sections for the claimed eps."""
    os.makedirs(RAW_EPS, exist_ok=True)
    existing = open(HINTS_MD).read() if os.path.exists(HINTS_MD) else \
        "# Per-episode hints for SSAA distributed annotation\n"
    have = set(re.findall(r"^##\s+(\S+)", existing, re.M))
    add = []
    for e, target in done:
        key = os.path.basename(target)
        if key in have:
            continue
        if with_hint and e.get("hint"):
            body = e["hint"]
        else:
            body = f"(task: {e.get('task','')})  # write the hint below"
        add.append(f"\n## {key}\n{body}\n")
    if add:
        with open(HINTS_MD, "a") as f:
            f.write("".join(add))


def cmd_claim_hint(args):
    a = ["claim", "--role", "hint", "--n", str(args.n), "--worker", args.worker]
    if args.outcome:
        a += ["--outcome", args.outcome]
    eps = json.loads(coord(a))
    print(f"claimed {len(eps)} for hinting; pulling + extracting…")
    done = _extract_batch(eps)
    _seed_hints_md(done, with_hint=False)
    print(f"  extracted {len(done)} → {RAW_EPS}")
    for e, t in done:
        print(f"   {os.path.basename(t)}  [{e['outcome']}]  task: {e.get('task','')[:60]}")
    print(f"\nView + write hints:\n  cd {SCRIPTS} && {VENV_PY} view_droid_v3.py "
          f"--images-dir {RAW_EPS} --suffix subagent_v3 --include-unannotated "
          f"--hints {HINTS_MD} --load-video --port 7864\n"
          f"Then edit {HINTS_MD} and run: ssaa_client.py push-hints")


def cmd_push_hints(args):
    if not os.path.exists(HINTS_MD):
        sys.exit("no hints.md")
    text = open(HINTS_MD).read()
    secs = re.findall(r"^##\s+(\S+)\s*\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    payload = {}
    for key, body in secs:
        body = body.strip()
        if not body or body.startswith("(task:"):    # unfilled placeholder
            continue
        mp = os.path.join(RAW_EPS, key, "meta.json")
        if not os.path.exists(mp):
            continue
        uuid = json.load(open(mp)).get("uuid")
        if uuid:
            payload[uuid] = body
    if not payload:
        sys.exit("no filled hints to push")
    print(coord(["set-hint", "--stdin"], stdin=json.dumps(payload)))


def cmd_claim_annot(args):
    eps = json.loads(coord(["claim", "--role", "annot", "--n", str(args.n),
                            "--worker", args.worker]))
    print(f"claimed {len(eps)} for annotation; pulling + extracting…")
    done = _extract_batch(eps)
    _seed_hints_md(done, with_hint=True)
    print(f"  extracted {len(done)} → {RAW_EPS} (hints written to hints.md)")
    for e, t in done:
        print(f"   {os.path.basename(t)}  [{e['outcome']}]")
    print("\nNext: launch subagents to write annotation_subagent_v3.json into each "
          "ep dir (see README), then: ssaa_client.py push-annot")


def cmd_push_annot(args):
    pushed = []
    for name in sorted(os.listdir(RAW_EPS)):
        d = os.path.join(RAW_EPS, name)
        ann = os.path.join(d, "annotation_subagent_v3.json")
        mp = os.path.join(d, "meta.json")
        if not (os.path.isdir(d) and os.path.exists(ann) and os.path.exists(mp)):
            continue
        uuid = json.load(open(mp)).get("uuid")
        if not uuid:
            continue
        rdir = f"{REMOTE_ANNOT}/{uuid}"
        subprocess.run(["ssh", SSH, f"mkdir -p '{rdir}'"], capture_output=True)
        files = [ann]
        aud = ann + ".audit.json"
        if os.path.exists(aud):
            files.append(aud)
        ok = subprocess.run(["scp", "-q", *files, f"{SSH}:{rdir}/"],
                            capture_output=True, text=True).returncode == 0
        if ok:
            pushed.append(uuid)
    if pushed:
        print(coord(["mark-annotated", "--stdin"], stdin=json.dumps(pushed)))
    print(f"pushed {len(pushed)} annotations")


def cmd_import_local_10(args):
    """Seed the server with the original 10 eps (hints + annotations) by
    matching their meta to server uuids (lab+timestamp)."""
    src = f"{REPO}/policy/lap/local_data/raw_eps"
    # build server lookup: uuid -> rel (resolve via 'list')
    rows = json.loads(coord(["list", "--limit", "100000"]))
    by_rel = {r["rel"]: r["uuid"] for r in rows}
    import glob as _g
    payload = {}; ann_map = {}
    for d in sorted(_g.glob(f"{src}/ep*/")):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        eid = meta.get("episode_id", "")            # e.g. AUTOLab/failure/<date>/<time>/trajectory.h5
        rel = eid.rsplit("/trajectory.h5", 1)[0]
        uuid = by_rel.get(rel) or by_rel.get(rel.replace("AUTOLab/", ""))
        if not uuid:
            continue
        hint = ""
        hm = f"{src}/hints.md"
        if os.path.exists(hm):
            t = open(hm).read()
            m = re.search(rf"^##\s+{re.escape(os.path.basename(d.rstrip('/')))}\s*\n(.*?)(?=^##\s|\Z)", t, re.M | re.S)
            if m:
                hint = m.group(1).strip()
        ann = os.path.join(d, "annotation_subagent_v3.json")
        payload[uuid] = {"hint": hint, "annotated": os.path.exists(ann)}
        if os.path.exists(ann):
            ann_map[uuid] = (d, ann)
    if not payload:
        sys.exit("no local eps matched server uuids")
    # push annotation files for the done ones
    for uuid, (d, ann) in ann_map.items():
        rdir = f"{REMOTE_ANNOT}/{uuid}"
        subprocess.run(["ssh", SSH, f"mkdir -p '{rdir}'"], capture_output=True)
        files = [ann] + ([ann + ".audit.json"] if os.path.exists(ann + ".audit.json") else [])
        subprocess.run(["scp", "-q", *files, f"{SSH}:{rdir}/"], capture_output=True)
    print(coord(["bulk-import"], stdin=json.dumps(payload)))
    print(f"imported {len(payload)} eps ({len(ann_map)} with annotations)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("claim-hint"); p.add_argument("--n", type=int, default=5)
    p.add_argument("--outcome", choices=["success", "failure"]); p.add_argument("--worker", default="local")
    p.set_defaults(fn=cmd_claim_hint)
    sub.add_parser("push-hints").set_defaults(fn=cmd_push_hints)
    p = sub.add_parser("claim-annot"); p.add_argument("--n", type=int, default=5); p.add_argument("--worker", default="local")
    p.set_defaults(fn=cmd_claim_annot)
    sub.add_parser("push-annot").set_defaults(fn=cmd_push_annot)
    sub.add_parser("import-local-10").set_defaults(fn=cmd_import_local_10)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
