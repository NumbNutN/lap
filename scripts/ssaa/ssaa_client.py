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
import argparse, json, os, re, subprocess, sys, tempfile, time

# lap repo root = three levels up from this file (lap/scripts/ssaa/ssaa_client.py),
# so paths work whether lap is a standalone clone or vendored at RoboTwin/policy/lap.
LAP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENV_PY = os.environ.get("SSAA_VENV_PY", f"{LAP_ROOT}/.venv/bin/python3")
if not os.path.exists(VENV_PY):          # fall back to the interpreter running us
    VENV_PY = sys.executable
SCRIPTS = f"{LAP_ROOT}/scripts"
EXTRACT = f"{SCRIPTS}/data_pipeline/extract_raw.py"
AUDIT = f"{SCRIPTS}/data_pipeline/audit_v3.py"
LOCAL_DATA = f"{LAP_ROOT}/local_data"
MIRROR = os.path.expanduser("~/datasets/droid_raw/1.0.1")
# Working set is per-worker so two Claude sessions on one machine don't collide
# (raw mirror above is shared — different episodes land in different rel paths).
RAW_EPS = os.environ.get("SSAA_RAW_EPS", f"{LOCAL_DATA}/raw_eps_remote")
HINTS_MD = f"{RAW_EPS}/hints.md"
# Where pull-annot drops cloud annotations for human QA (others' work included).
REVIEW = os.environ.get("SSAA_REVIEW", f"{LOCAL_DATA}/ssaa_review")
SSH = "bitahub"                       # ~/.ssh/config alias (ControlMaster)
REMOTE_DATA = "/localdisk-tmp/datasets/droid_raw/1.0.1"
REMOTE_SSAA = "/localdisk-tmp/ssaa"
REMOTE_COORD = f"cd {REMOTE_SSAA} && python3 coord.py"
REMOTE_ANNOT = f"{REMOTE_SSAA}/annotations"
REMOTE_HINTS = f"{REMOTE_SSAA}/hints"
REMOTE_STATE = f"{REMOTE_SSAA}/state.json"
REMOTE_REPORTS = f"{REMOTE_SSAA}/reports"
REMOTE_TELEOP = "/localdisk-tmp/datasets/teleop_playground"   # self-contained teleop eps
TELEOP_EPS = os.environ.get("SSAA_TELEOP_EPS", f"{LOCAL_DATA}/teleop_eps")


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


def _extract_batch(eps: list[dict], dest: str | None = None):
    """rsync each ep into the mirror, extract to a temp dir, rename by uuid,
    move into `dest` (RAW_EPS by default). eps = [{uuid, rel, outcome, task,
    hint?}, ...]."""
    dest = dest or RAW_EPS
    os.makedirs(dest, exist_ok=True)
    wl = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    wl.write("ep_id,classification\n")
    relmap = {}
    pulled = 0
    n = len(eps)
    for i, e in enumerate(eps, 1):
        short = os.path.basename(e["rel"])
        sys.stdout.write(f"\r  pulling {i}/{n}  {short[:46]:<46}")
        sys.stdout.flush()
        if _rsync_ep(e["rel"]):          # prints its own line on failure
            wl.write(f"{e['rel']}/trajectory.h5,good\n")
            relmap[f"{e['rel']}/trajectory.h5"] = e
            pulled += 1
    wl.close()
    if n:
        sys.stdout.write(f"\r  pulled {pulled}/{n} episode(s){' ' * 46}\n")
        sys.stdout.flush()
    if not pulled:
        return []
    out = tempfile.mkdtemp(prefix="ssaa_out_")
    env = dict(os.environ, DROID_RAW_ROOT=MIRROR)
    print(f"  extracting {pulled} episode(s) (decoding frames)…", flush=True)
    subprocess.run([VENV_PY, EXTRACT, "--root", MIRROR, "--out", out,
                    "--whitelist", wl.name, "--include-failure"],
                   env=env)             # uncaptured → streams [ok] epN … per ep
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
        target = os.path.join(dest, _sanitize(e["uuid"]))
        subprocess.run(["rm", "-rf", target])
        subprocess.run(["mv", d, target])
        done.append((e, target))
    os.unlink(wl.name)
    return done


def _seed_hints_md(done, with_hint: bool, dest: str | None = None):
    """Append/update hints.md sections for the claimed eps."""
    dest = dest or RAW_EPS
    os.makedirs(dest, exist_ok=True)
    hints_md = os.path.join(dest, "hints.md")
    existing = open(hints_md).read() if os.path.exists(hints_md) else \
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
        with open(hints_md, "a") as f:
            f.write("".join(add))


def _auto_prune(workspace_dir: str, rows: dict) -> int:
    """Remove ep dirs in `workspace_dir` whose content is already on the server
    (hint submitted, or annotation pushed) — they're redundant scratch."""
    import glob as _g
    removed = 0
    for mp in _g.glob(f"{workspace_dir}/*/meta.json"):
        epd = os.path.dirname(mp)
        try:
            uuid = json.load(open(mp)).get("uuid")
        except Exception:
            continue
        srv = rows.get(uuid, {})
        sstat = srv.get("status", "")
        has_ann = os.path.exists(os.path.join(epd, "annotation_subagent_v3.json"))
        safe = (has_ann and sstat == "annotated") or \
               (not has_ann and (sstat in ("hinted", "annot_claimed", "annotated")
                                  or srv.get("hint")))
        if safe:
            subprocess.run(["rm", "-rf", epd])
            removed += 1
    return removed


def cmd_claim_hint(args):
    if not args.no_prune:        # clear already-submitted eps before pulling new
        rows = {r["uuid"]: r for r in json.loads(coord(["list", "--limit", "100000"]))}
        n = _auto_prune(RAW_EPS, rows)
        if n:
            print(f"pruned {n} already-submitted ep(s) from {os.path.basename(RAW_EPS)}")
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


def _push_hints() -> int:
    """Push filled hints from RAW_EPS/hints.md to the server; return count.
    Placeholders ((task: …)) are skipped. No exit on empty.

    The ep-dir name (= section header) is the SANITIZED uuid, so we resolve
    uuid from the server list rather than the local meta.json — that way hints
    still push even if the local dir was pruned/deleted (robust to churn)."""
    if not os.path.exists(HINTS_MD):
        return 0
    text = open(HINTS_MD).read()
    secs = re.findall(r"^##\s+(\S+)\s*\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    filled = {k: b.strip() for k, b in secs
              if b.strip() and not b.strip().startswith("(task:")}
    if not filled:
        return 0
    by_key = {_sanitize(r["uuid"]): r["uuid"]
              for r in json.loads(coord(["list", "--limit", "100000"]))}
    payload, excl = {}, {}
    for key, body in filled.items():
        uuid = by_key.get(key)
        if not uuid:                                 # fallback: local meta.json
            mp = os.path.join(RAW_EPS, key, "meta.json")
            if os.path.exists(mp):
                uuid = json.load(open(mp)).get("uuid")
        if not uuid:
            continue
        if body.startswith("[[EXCLUDE]]"):           # viewer / hand-written tag
            excl[uuid] = body[len("[[EXCLUDE]]"):].strip()
        else:
            payload[uuid] = body
    if excl:
        coord(["exclude", "--stdin"], stdin=json.dumps(excl))
        print(f"excluded {len(excl)} unusable ep(s)")
    if payload:
        coord(["set-hint", "--stdin"], stdin=json.dumps(payload))
    return len(payload) + len(excl)


def cmd_push_hints(args):
    n = _push_hints()
    print(json.dumps({"pushed": n}) if n else "no filled hints to push")


def _claim_hint_batch(n: int, outcome: str | None, worker: str) -> list:
    if n <= 0:
        return []
    a = ["claim", "--role", "hint", "--n", str(n), "--worker", worker]
    if outcome:
        a += ["--outcome", outcome]
    done = _extract_batch(json.loads(coord(a)))
    _seed_hints_md(done, with_hint=False)
    return done


def cmd_hint_round(args):
    """One command per hinting round: submit last round's hints → prune the
    now-submitted scratch → claim a fresh success/failure mix → open the viewer
    (write hints in-page, 💾 Save, Ctrl-C, re-run to push + pull the next batch)."""
    pushed = _push_hints()
    if pushed:
        print(f"pushed {pushed} hint(s) from last round")
    rows = {r["uuid"]: r for r in json.loads(coord(["list", "--limit", "100000"]))}
    pr = _auto_prune(RAW_EPS, rows)
    if pr:
        print(f"pruned {pr} submitted ep(s) from {os.path.basename(RAW_EPS)}")
    done = (_claim_hint_batch(args.success, "success", args.worker) +
            _claim_hint_batch(args.failure, "failure", args.worker))
    print(f"claimed {len(done)} new ep(s) → {os.path.basename(RAW_EPS)}")
    for e, t in done:
        print(f"   {os.path.basename(t)}  [{e['outcome']}]  task: {e.get('task','')[:50]}")
    if not done:
        print("  (nothing new — that outcome's queue may be exhausted)")
    if args.no_viewer:
        print(f"\nviewer:\n  cd {SCRIPTS} && {VENV_PY} view_droid_v3.py --images-dir "
              f"{RAW_EPS} --suffix subagent_v3 --include-unannotated --hints {HINTS_MD} "
              f"--load-video --port {args.port}")
        return
    print(f"\n→ opening viewer at http://localhost:{args.port}  "
          f"(write each hint, click 💾 Save; Ctrl-C to close, then run hint-round again)")
    subprocess.run([VENV_PY, f"{SCRIPTS}/view_droid_v3.py", "--images-dir", RAW_EPS,
                    "--suffix", "subagent_v3", "--include-unannotated", "--hints", HINTS_MD,
                    "--load-video", "--port", str(args.port)])


def _pull_teleop_ep(e: dict) -> str | None:
    """Teleop eps are already self-contained (h5+mp4+kf+meta) on the server —
    just rsync the whole dir down, no extraction."""
    os.makedirs(RAW_EPS, exist_ok=True)
    target = os.path.join(RAW_EPS, _sanitize(e["uuid"]))
    src = f"{SSH}:{REMOTE_TELEOP}/{e['rel']}/"
    r = subprocess.run(["rsync", "-a", "--info=stats0", "-e", "ssh", src, target + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   [rsync fail] {e['rel']}: {r.stderr.strip()[:160]}")
        return None
    return target


def cmd_claim_annot(args):
    a = ["claim", "--role", "annot", "--n", str(args.n), "--worker", args.worker]
    if args.outcome:
        a += ["--outcome", args.outcome]
    if getattr(args, "source", None):
        a += ["--source", args.source]
    eps = json.loads(coord(a))
    print(f"claimed {len(eps)} for annotation; pulling…")
    teleop = [e for e in eps if e.get("source") == "teleop"]
    droid = [e for e in eps if e.get("source") != "teleop"]
    done = _extract_batch(droid) if droid else []
    for e in teleop:                       # self-contained: pull dir, no extract
        t = _pull_teleop_ep(e)
        if t:
            done.append((e, t))
    _seed_hints_md(done, with_hint=True)
    print(f"  pulled {len(done)} → {RAW_EPS} (hints written to hints.md)")
    for e, t in done:
        print(f"   {os.path.basename(t)}  [{e.get('source','droid')}/{e['outcome']}]")
    print("\nNext: launch subagents to write annotation_subagent_v3.json into each "
          "ep dir (see README), then: ssaa_client.py push-annot")


def cmd_push_teleop(args):
    """Upload self-contained teleop segment dirs to the server + register them
    with the coordinator (claimable like any other episode)."""
    src = (args.dir or TELEOP_EPS).rstrip("/")
    if not os.path.isdir(src):
        sys.exit(f"no teleop eps dir: {src}")
    subprocess.run(["ssh", SSH, f"mkdir -p {REMOTE_TELEOP}"], capture_output=True)
    r = subprocess.run(["rsync", "-a", "--info=stats0", "-e", "ssh",
                        src + "/", f"{SSH}:{REMOTE_TELEOP}/"], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[rsync fail] {r.stderr.strip()[:200]}")
    n = sum(1 for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    print(f"uploaded {n} teleop ep(s) → {REMOTE_TELEOP}")
    print(coord(["init"]))   # merge: registers new teleop eps as available
    # Seed each teleop seg's task as its initial hint → status hinted (claimable
    # for annotation now; refine the prior-context hint later in the viewer).
    payload = {}
    for d in sorted(os.listdir(src)):
        mp = os.path.join(src, d, "meta.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            if m.get("uuid"):
                payload[m["uuid"]] = m.get("task_instruction") or "(teleop segment)"
    if payload:
        coord(["set-hint", "--stdin"], stdin=json.dumps(payload))
        print(f"  seeded {len(payload)} teleop hint(s) → hinted (claimable for annot)")


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
        files = [ann, mp]                     # meta.json too → lightweight central re-audit
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


def _ep_date(r: dict) -> str:
    """Episode date 'YYYY-MM-DD' parsed from rel (LAB/outcome/DATE/TIME)."""
    parts = r.get("rel", "").split("/")
    return parts[2] if len(parts) > 2 else ""


def cmd_pull_annot(args):
    """Pull ANNOTATED episodes from the server into a FRESH review subfolder for
    human QA, then open the viewer. Works for ANY annotator's work. Filters
    combine (AND): --uuid (substring match on the sanitized uuid; repeatable),
    --outcome, --lab, --after / --before (YYYY-MM-DD on the episode date),
    --limit. No filter = all annotated."""
    rows = json.loads(coord(["list", "--status", "annotated", "--limit", "100000"]))
    if args.outcome:
        rows = [r for r in rows if r["outcome"] == args.outcome]
    if args.lab:
        rows = [r for r in rows if args.lab.lower() in (r.get("lab", "") + r["rel"]).lower()]
    if args.after:
        rows = [r for r in rows if _ep_date(r) >= args.after]
    if args.before:
        rows = [r for r in rows if _ep_date(r) <= args.before]
    if args.uuid:
        pats = [_sanitize(p) for p in args.uuid]
        matched, unmatched = [], set(pats)
        for r in rows:
            key = _sanitize(r["uuid"])
            hit = next((p for p in pats if p in key), None)
            if hit:
                matched.append(r)
                unmatched.discard(hit)
        for p in sorted(unmatched):
            print(f"  [not annotated / no match] {p}")
        rows = matched
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        sys.exit("no annotated episodes match the criteria")
    label = args.name or ("review_" + time.strftime("%Y%m%d_%H%M%S"))
    dest = os.path.join(REVIEW, label)
    eps = [{"uuid": r["uuid"], "rel": r["rel"], "outcome": r["outcome"],
            "task": r.get("task", ""), "hint": r.get("hint")} for r in rows]
    print(f"pulling {len(eps)} annotated ep(s) → {dest}")
    done = _extract_batch(eps, dest=dest)
    _seed_hints_md(done, with_hint=True, dest=dest)
    got = 0
    for e, t in done:
        rdir = f"{REMOTE_ANNOT}/{e['uuid']}"
        ann = subprocess.run(
            ["scp", "-q", f"{SSH}:{rdir}/annotation_subagent_v3.json", t + "/"],
            capture_output=True, text=True)
        subprocess.run(                       # audit file is optional
            ["scp", "-q", f"{SSH}:{rdir}/annotation_subagent_v3.json.audit.json", t + "/"],
            capture_output=True, text=True)
        if ann.returncode == 0:
            got += 1
        print(f"   {os.path.basename(t)}  [{e['outcome']}]"
              f"{'' if ann.returncode == 0 else '  [no annotation on server]'}")
    print(f"\npulled {got}/{len(done)} annotations → {dest}")
    viewer = [VENV_PY, f"{SCRIPTS}/view_droid_v3.py", "--images-dir", dest,
              "--suffix", "subagent_v3", "--hints", f"{dest}/hints.md",
              "--load-video", "--port", str(args.port)]
    if args.no_viewer:
        print("  viewer:\n   " + " ".join(viewer))
        return
    print(f"\n→ opening viewer at http://localhost:{args.port}  (Ctrl-C to close)")
    subprocess.run(viewer)


def _local_roots():
    """Working dirs that hold ep subdirs: raw_eps* + each review_* subfolder."""
    import glob as _g
    return sorted(set(_g.glob(f"{LOCAL_DATA}/raw_eps*") + _g.glob(f"{REVIEW}/*")))


def _scan_local_eps():
    """uuid -> [(dir_label, has_annotation, is_review), ...] across local dirs."""
    import glob as _g
    review_abs = os.path.abspath(REVIEW) + os.sep
    seen: dict[str, list] = {}
    for d in _local_roots():
        if not os.path.isdir(d):
            continue
        label = os.path.relpath(d, LOCAL_DATA)
        is_review = os.path.abspath(d).startswith(review_abs)
        for mp in _g.glob(f"{d}/*/meta.json"):
            epd = os.path.dirname(mp)
            try:
                uuid = json.load(open(mp)).get("uuid")
            except Exception:
                continue
            if not uuid:
                continue
            has_ann = os.path.exists(os.path.join(epd, "annotation_subagent_v3.json"))
            seen.setdefault(uuid, []).append((label, has_ann, is_review))
    return seen


def cmd_local_status(args):
    """Show every episode that touches local disk, with its lifecycle state —
    so you never hand-track which dir holds what."""
    rows = {r["uuid"]: r for r in json.loads(coord(["list", "--limit", "100000"]))}
    local = _scan_local_eps()
    buckets: dict[str, list] = {}
    for uuid, places in local.items():
        srv = rows.get(uuid, {})
        sstat = srv.get("status", "unknown")
        has_local_ann = any(a for _, a, _ in places)
        in_review = any(rv for _, _, rv in places)
        if in_review:
            state = "pulled_for_review"            # 从云端拉取，人工检查
        elif has_local_ann and sstat == "annotated":
            state = "annotated_pushed"             # 已本地标注并提交
        elif has_local_ann:
            state = "annotated_local_unpushed"     # 已本地标注，未提交
        elif sstat in ("hinted", "annot_claimed"):
            state = "hinted_submitted"             # 已提示并提交
        elif sstat == "hint_claimed":
            state = "awaiting_hint"                # 认领，等待提示
        else:
            state = sstat
        dirs = ",".join(sorted({lbl for lbl, _, _ in places}))
        buckets.setdefault(state, []).append((uuid, srv.get("outcome", "?"), dirs))
    order = ["awaiting_hint", "hinted_submitted", "annotated_local_unpushed",
             "annotated_pushed", "pulled_for_review"]
    for state in order + [s for s in buckets if s not in order]:
        if state not in buckets:
            continue
        print(f"\n[{state}]  ({len(buckets[state])})")
        for uuid, outcome, dirs in sorted(buckets[state]):
            print(f"   {uuid:42s} {outcome:8s} {dirs}")
    print()


def cmd_backup(args):
    """Read-only: mirror ALL server annotations + hints + state.json to a local
    backup (remote storage may not be durable). Never deletes locally — the
    backup only accumulates, so episodes dropped on the server survive here."""
    dest = args.dir or f"{LOCAL_DATA}/ssaa_backup"
    os.makedirs(dest, exist_ok=True)
    for sub in ("annotations", "hints"):
        os.makedirs(f"{dest}/{sub}", exist_ok=True)
        r = subprocess.run(["rsync", "-a", "--info=stats0", "-e", "ssh",
                            f"{SSH}:{REMOTE_SSAA}/{sub}/", f"{dest}/{sub}/"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   [rsync {sub} fail] {r.stderr.strip()[:160]}")
    subprocess.run(["scp", "-q", f"{SSH}:{REMOTE_STATE}", f"{dest}/state.json"],
                   capture_output=True, text=True)
    import glob as _g
    n_ann = len(_g.glob(f"{dest}/annotations/*/annotation_subagent_v3.json"))
    n_hint = len(_g.glob(f"{dest}/hints/*.txt"))
    print(f"backup → {dest}\n   {n_ann} annotations, {n_hint} hints, state.json mirrored")


def cmd_prune(args):
    """Delete local ep dirs whose content is safely on the server (re-pullable).
    Dry-run by default; pass --yes to delete. The server is the source of truth.
      - hint workspace (raw_eps_remote): safe once server has the hint.
      - annot workspace (raw_eps_<worker>): safe once server marks it annotated.
      - review dir: only with --review (after you've QA'd it)."""
    rows = {r["uuid"]: r for r in json.loads(coord(["list", "--limit", "100000"]))}
    review_abs = os.path.abspath(REVIEW) + os.sep
    candidates = []   # (path, uuid, reason)
    import glob as _g
    for d in _local_roots():
        if not os.path.isdir(d):
            continue
        is_review = os.path.abspath(d).startswith(review_abs)
        for mp in _g.glob(f"{d}/*/meta.json"):
            epd = os.path.dirname(mp)
            try:
                uuid = json.load(open(mp)).get("uuid")
            except Exception:
                continue
            srv = rows.get(uuid, {})
            sstat = srv.get("status", "")
            has_ann = os.path.exists(os.path.join(epd, "annotation_subagent_v3.json"))
            if is_review:
                if args.review:
                    candidates.append((epd, uuid, "reviewed (re-pull with pull-annot)"))
            elif has_ann:
                if sstat == "annotated":
                    candidates.append((epd, uuid, "annotation on server"))
            else:   # hint workspace copy
                if sstat in ("hinted", "annot_claimed", "annotated") or srv.get("hint"):
                    candidates.append((epd, uuid, "hint on server"))
    if not candidates:
        print("nothing prunable.")
        return
    mb = 0
    for epd, uuid, reason in candidates:
        sz = sum(os.path.getsize(os.path.join(epd, f)) for f in os.listdir(epd)) / 1e6
        mb += sz
        print(f"   {'DELETE' if args.yes else 'would delete'}  "
              f"{os.path.relpath(epd, LOCAL_DATA):60s} ({sz:5.1f}MB)  [{reason}]")
        if args.yes:
            subprocess.run(["rm", "-rf", epd])
    print(f"\n{'pruned' if args.yes else 'prunable'}: {len(candidates)} dirs, ~{mb:.0f}MB"
          + ("" if args.yes else "   (re-run with --yes to delete)"))


def _audit_summary(csv_path: str) -> dict:
    """Parse an audit_v3 CSV → {n, clean, issues:{name:count}} aggregating
    gate_issues (comma-joined `kf{i}:name` tokens) by issue name."""
    import csv as _csv
    rows = list(_csv.DictReader(open(csv_path)))
    issues: dict[str, int] = {}
    clean = 0
    for r in rows:
        if (r.get("gate_ok") == "True" and r.get("bounds_ok") == "True"
                and (r.get("n_spred_echoes_a") or "0") == "0"):
            clean += 1
        for tok in (r.get("gate_issues") or "").split(","):
            tok = tok.strip()
            if tok:
                name = tok.split(":", 1)[1] if ":" in tok else tok
                issues[name] = issues.get(name, 0) + 1
        if r.get("bounds_ok") == "False":
            issues["bounds-violation"] = issues.get("bounds-violation", 0) + 1
        if (r.get("n_spred_echoes_a") or "0") not in ("0", ""):
            issues["spred-echo"] = issues.get("spred-echo", 0) + 1
    return {"n": len(rows), "clean": clean,
            "issues": dict(sorted(issues.items(), key=lambda kv: -kv[1]))}


def cmd_push_report(args):
    """Worker → server: re-audit the local annotated set and upload the audit
    CSV + a friction report (free-text notes on recurring audit issues or
    ambiguous spec rules) to reports/<worker>_<ts>/. This is the channel for
    spec problems — NOT editing the prompt."""
    csv_path = f"/tmp/ssaa_report_{args.worker}.csv"
    subprocess.run([VENV_PY, AUDIT, "--raw-root", RAW_EPS,
                    "--pattern", "annotation_subagent_v3.json", "--out", csv_path])
    if not os.path.exists(csv_path):
        sys.exit("audit produced no CSV (nothing annotated in this workspace?)")
    summ = _audit_summary(csv_path)
    note = args.note or ""
    if args.note_file and os.path.exists(args.note_file):
        note = open(args.note_file).read()
    rid = f"{args.worker}_{time.strftime('%Y%m%d_%H%M%S')}"
    report = {"worker": args.worker, "id": rid, "summary": summ, "note": note}
    local = f"/tmp/{rid}.report.json"
    json.dump(report, open(local, "w"), indent=2, ensure_ascii=False)
    rdir = f"{REMOTE_REPORTS}/{rid}"
    subprocess.run(["ssh", SSH, f"mkdir -p '{rdir}'"], capture_output=True)
    subprocess.run(["scp", "-q", local, csv_path, f"{SSH}:{rdir}/"], capture_output=True)
    print(f"report {rid}: {summ['n']} eps, {summ['clean']} clean")
    if summ["issues"]:
        print("  issues:", ", ".join(f"{k}×{v}" for k, v in summ["issues"].items()))
    if note.strip():
        print(f"  note: {note.strip()[:120]}")
    print(f"  → {rdir}")


def cmd_exclude(args):
    """Mark unusable episodes excluded (claim skips them; export drops them).
    --uuid takes substring matches on the sanitized uuid; --undo restores."""
    rows = json.loads(coord(["list", "--limit", "100000"]))
    by_key = {_sanitize(r["uuid"]): r["uuid"] for r in rows}
    payload, unmatched = {}, []
    for pat in (args.uuid or []):
        s = _sanitize(pat)
        hit = next((u for k, u in by_key.items() if s in k), None)
        if hit:
            payload[hit] = args.reason or ""
        else:
            unmatched.append(pat)
    for p in unmatched:
        print(f"  [no match] {p}")
    if not payload:
        sys.exit("nothing matched")
    a = ["exclude", "--stdin"] + (["--undo"] if args.undo else [])
    print(coord(a, stdin=json.dumps(payload)))


def cmd_reset_hint(args):
    """Clear a wrong hint and send the ep back to `available` (re-hintable).
    --uuid takes substring matches on the sanitized uuid."""
    rows = json.loads(coord(["list", "--limit", "100000"]))
    by_key = {_sanitize(r["uuid"]): r["uuid"] for r in rows}
    picked = []
    for pat in (args.uuid or []):
        s = _sanitize(pat)
        hit = next((u for k, u in by_key.items() if s in k), None)
        if hit:
            picked.append(hit)
        else:
            print(f"  [no match] {pat}")
    if not picked:
        sys.exit("nothing matched")
    a = ["reset-hint", "--stdin"] + (["--force"] if args.force else [])
    print(coord(a, stdin=json.dumps(picked)))


def cmd_audit_all(args):
    """Central re-audit (lead): rsync ALL server annotations (+ meta.json) and
    reports locally, run audit_v3 authoritatively, aggregate gate_issues, and
    print every worker's friction note — the basis for deciding deliberate
    (central) spec changes vs re-annotation."""
    import glob as _g
    dest = f"{LOCAL_DATA}/ssaa_audit_all"
    os.makedirs(f"{dest}/annotations", exist_ok=True)
    os.makedirs(f"{dest}/reports", exist_ok=True)
    # --delete: this is a scratch mirror of the server, so reflect it exactly
    # (removed reports/annotations shouldn't linger and skew the re-audit).
    subprocess.run(["rsync", "-a", "--delete", "-e", "ssh", f"{SSH}:{REMOTE_ANNOT}/",
                    f"{dest}/annotations/"], capture_output=True)
    subprocess.run(["rsync", "-a", "--delete", "-e", "ssh", f"{SSH}:{REMOTE_REPORTS}/",
                    f"{dest}/reports/"], capture_output=True)
    total = len(_g.glob(f"{dest}/annotations/*/annotation_subagent_v3.json"))
    have_meta = len(_g.glob(f"{dest}/annotations/*/meta.json"))
    csv_path = f"{dest}/audit_all.csv"
    subprocess.run([VENV_PY, AUDIT, "--raw-root", f"{dest}/annotations",
                    "--pattern", "annotation_subagent_v3.json", "--out", csv_path])
    summ = _audit_summary(csv_path) if os.path.exists(csv_path) else {"n": 0, "clean": 0, "issues": {}}
    print(f"\n=== central re-audit ===  ({dest}/audit_all.csv)")
    print(f"  annotations on server: {total}   audited (have meta.json): {summ['n']}")
    if total > have_meta:
        print(f"  ⚠️ {total - have_meta} lack meta.json (pushed before meta upload) "
              f"— re-push or pull-annot to audit them")
    print(f"  clean: {summ['clean']}/{summ['n']}")
    if summ["issues"]:
        print("  gate issues (by type):")
        for k, v in summ["issues"].items():
            print(f"     {v:4d}  {k}")
    reps = sorted(_g.glob(f"{dest}/reports/*/*.report.json"))
    notes = []
    for rp in reps:
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        if (r.get("note") or "").strip():
            notes.append((r.get("worker", "?"), r["note"].strip()))
    if notes:
        print(f"\n  worker friction notes ({len(notes)}):")
        for w, n in notes:
            print(f"   [{w}] {n[:300]}")
    print()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    p = sub.add_parser("claim-hint"); p.add_argument("--n", type=int, default=5)
    p.add_argument("--outcome", choices=["success", "failure"]); p.add_argument("--worker", default="local")
    p.add_argument("--no-prune", action="store_true", help="keep already-submitted eps in the workspace")
    p.set_defaults(fn=cmd_claim_hint)
    sub.add_parser("push-hints").set_defaults(fn=cmd_push_hints)
    p = sub.add_parser("hint-round")
    p.add_argument("--success", type=int, default=8); p.add_argument("--failure", type=int, default=2)
    p.add_argument("--port", type=int, default=7864); p.add_argument("--worker", default="local")
    p.add_argument("--no-viewer", action="store_true"); p.set_defaults(fn=cmd_hint_round)
    p = sub.add_parser("claim-annot"); p.add_argument("--n", type=int, default=5); p.add_argument("--worker", default="local")
    p.add_argument("--outcome", choices=["success", "failure"])
    p.add_argument("--source", choices=["droid", "teleop"])
    p.set_defaults(fn=cmd_claim_annot)
    p = sub.add_parser("push-teleop"); p.add_argument("--dir"); p.set_defaults(fn=cmd_push_teleop)
    sub.add_parser("push-annot").set_defaults(fn=cmd_push_annot)
    p = sub.add_parser("pull-annot")
    p.add_argument("--uuid", nargs="*", help="substring match on the sanitized uuid (repeatable)")
    p.add_argument("--outcome", choices=["success", "failure"])
    p.add_argument("--lab", help="filter by lab / rel substring (e.g. TRI, AUTOLab)")
    p.add_argument("--after", help="episode date >= YYYY-MM-DD")
    p.add_argument("--before", help="episode date <= YYYY-MM-DD")
    p.add_argument("--limit", type=int)
    p.add_argument("--name", help="review subfolder name (default review_<timestamp>)")
    p.add_argument("--port", type=int, default=7870, help="viewer port (default 7870, ≠ hint-round 7864)")
    p.add_argument("--no-viewer", action="store_true")
    p.set_defaults(fn=cmd_pull_annot)
    sub.add_parser("local-status").set_defaults(fn=cmd_local_status)
    p = sub.add_parser("push-report"); p.add_argument("--worker", default="local")
    p.add_argument("--note", help="free-text friction notes (recurring audit issues, ambiguous rules)")
    p.add_argument("--note-file", help="read the note from a file instead")
    p.set_defaults(fn=cmd_push_report)
    sub.add_parser("audit-all").set_defaults(fn=cmd_audit_all)
    p = sub.add_parser("exclude"); p.add_argument("--uuid", nargs="+")
    p.add_argument("--reason"); p.add_argument("--undo", action="store_true")
    p.set_defaults(fn=cmd_exclude)
    p = sub.add_parser("reset-hint"); p.add_argument("--uuid", nargs="+")
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_reset_hint)
    p = sub.add_parser("backup"); p.add_argument("--dir"); p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("prune"); p.add_argument("--yes", action="store_true")
    p.add_argument("--review", action="store_true", help="also prune the review dir")
    p.set_defaults(fn=cmd_prune)
    sub.add_parser("import-local-10").set_defaults(fn=cmd_import_local_10)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
