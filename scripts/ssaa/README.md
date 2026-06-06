# SSAA distributed annotation — operator guide

This lets a Claude session (even with **no prior context**) pull DROID
episodes from the data server, write/collect hints, run sub-agent annotation,
and push results back. Read this top-to-bottom before starting.

## Architecture (why it's built this way)
- Data lives on a remote **k8s pod** (`bitahub`): ~2500 episodes at
  `/localdisk-tmp/datasets/droid_raw/1.0.1/<LAB>/<success|failure>/<date>/<time>/`.
- The pod's home is **read-only**, only SSH (port 42034) is externally mapped
  — **port 8888 is NOT reachable** from outside. So instead of an HTTP service,
  coordination is **file-based over SSH**: a stdlib `coord.py` on the server
  manages a flock-guarded `state.json`; the local `ssaa_client.py` drives it
  via an SSH **ControlMaster** (one password login, then password-free) and
  moves per-episode data with `rsync`. No daemon, no open port — robust to
  pod restarts. (If you ever want 8888: `ssh -L 8888:localhost:8888 bitahub`
  works, but isn't needed.)
- Extraction, viewing, and annotation run **locally** with the existing
  pipeline (`policy/lap/scripts/...`, venv `policy/lap/.venv`).

## 0. One-time connection setup (ControlMaster)
`~/.ssh/config` must contain (already written; recreate if missing):
```
Host bitahub
    HostName xj-member.bitahub.com
    Port 42034
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ControlMaster auto
    ControlPath ~/.ssh/cm-bitahub.sock
    ControlPersist 4h
```
The home is read-only so **key auth is impossible** — use password via a
one-shot pexpect login to open the master socket (lasts 4h, reopen as needed):
```
pip3 install --user pexpect   # if missing
SSAA_PW='<temp password>' python3 - <<'PY'
import pexpect, os
c = pexpect.spawn("ssh bitahub 'echo UP'", timeout=45, encoding='utf-8')
while True:
    i = c.expect([r'[Pp]assword:', r'continue connecting', pexpect.EOF, pexpect.TIMEOUT])
    if i==0: c.sendline(os.environ['SSAA_PW'])
    elif i==1: c.sendline('yes')
    else: break
print(c.before)
PY
ssh bitahub 'echo REUSE_OK'    # must work without a password now
```
Ask the user for the current temp password (it rotates). All `ssh`/`scp`/
`rsync` to `bitahub` then reuse the socket.

## 1. The client (`policy/lap/scripts/ssaa/ssaa_client.py`)
Run with the venv python: `policy/lap/.venv/bin/python3 ssaa_client.py <cmd>`.
| command | what it does |
|---|---|
| `stats` | server counts: total / available / hinted / annotated, by outcome |
| `hint-round [--success 8] [--failure 2] [--port 7864] [--no-viewer]` | **one-command hint loop**: push last round's hints → prune submitted → claim a success/failure mix → open the viewer to write the new hints in-page |
| `claim-hint --n K [--outcome …] [--no-prune]` | (lower-level) auto-prune submitted eps, then claim K *unhinted* eps → extract → `raw_eps_remote/<uuid>/`; seed `hints.md`. (`--n 0` = prune only.) |
| `push-hints` | parse `raw_eps_remote/hints.md`, push filled hints to server (status→hinted) |
| `claim-annot --n K` | claim K *hinted, un-annotated* eps → rsync + extract + write their hints into `hints.md` |
| `push-annot` | scp every local `annotation_subagent_v3.json` (+ audit) to the server, mark annotated |
| `pull-annot [--uuid U…] [--outcome …]` | pull **annotated** eps *from* the server into `ssaa_review/` (re-extract images + drop server annotation + hint) — for human QA of **anyone's** work |
| `push-report [--worker W] [--note … / --note-file F]` | worker → server: re-audit local set + upload audit CSV + friction note to `reports/` (the ONLY channel for spec problems) |
| `audit-all` | lead: rsync ALL annotations (+meta) + reports, re-run audit_v3 centrally, aggregate gate_issues by type, print every worker's friction note |
| `local-status` | every local ep with its lifecycle state + which dirs hold it |
| `backup [--dir D]` | read-only mirror of **all** server annotations + hints + state.json → `ssaa_backup/` (remote isn't durable; never deletes locally) |
| `prune [--yes] [--review]` | delete local ep dirs whose content is confirmed on the server (re-pullable); dry-run without `--yes` |
| `import-local-10` | seed the server with the original 10 eps (matched by path) |

Working set: `policy/lap/local_data/raw_eps_remote/` (ep dirs named by uuid +
`hints.md`). Raw h5/MP4 mirror: `~/datasets/droid_raw/1.0.1/` (= `DROID_RAW_ROOT`).

## 2. Writing hints (human step) — use `hint-round`
`hint-round` (above) is the normal path: it claims the batch and opens the
viewer with an editable **✍️ Write hint** box per episode — type the hint, click
**💾 Save** (writes to `hints.md`), Ctrl-C, re-run `hint-round` to push + pull the
next batch. See `HINTING.md` for the full loop. The viewer started with `--hints`
is in **hint-writing mode**; without it, it's read-only. Manual fallback:
hand-edit `raw_eps_remote/hints.md` (one `## <uuid-dir>` section per ep;
`(task: …)` placeholders are ignored until filled) and `push-hints`.

## 3. Annotation (sub-agents)
Prereqs: the episodes are in `raw_eps_remote/<uuid>/` with their hint in
`hints.md`. For **each** ep, launch one sub-agent (Agent tool, Sonnet) that:
1. Reads the finalized system prompt **in full**:
   `policy/lap/scripts/annotate_droid/prompt_ssaa_v3.md` (and skim
   `policy/lap/local_data/raw_eps/DATASET_NOTES.md` for the conventions).
2. Annotates the episode at `EP_PATH = .../raw_eps_remote/<uuid>` using the
   per-episode **hint** from `hints.md` (the task is whatever the hint says —
   the prompt is general, NOT pour-specific).
3. Tools (venv python): `tools_cli.py keyframes|pose_delta|image <EP_PATH> …`
   (`policy/lap/scripts/data_pipeline/tools_cli.py`). `pose_delta` returns the
   `gripper` field; `image <frame> ext|wrist <out.jpg>` then Read it.
4. Writes `<EP_PATH>/annotation_subagent_v3.json` (+ `.json.audit.json`).

The non-negotiable rules (all in the system prompt — do not re-derive):
begin/end = S-only brackets + Plan on first moving keyframe; chunk_end =
intent boundary (coterminous, no tiling); action frame follows the grounding
view (occlusion→wrist); gripper from the tool field, qualitative; S
present-only; S_pred from the chunk_end image, task-critical, no A-echo;
physical-effect `<think>`; imit non-monotone, A_correct only when imit=false;
no leakage (no "the demo"/frame indices/"N frames"/raw 0–1 gripper value).

Validate every ep before pushing:
```
policy/lap/.venv/bin/python3 policy/lap/scripts/data_pipeline/audit_v3.py \
   --raw-root policy/lap/local_data/raw_eps_remote --pattern annotation_subagent_v3.json --out /tmp/a.csv
```
Require: `gate_ok`, `bounds_ok` True; `gate_issues` empty; 0 leakage / S_pred-echo
/ foreshadow / raw-gripper. Then `ssaa_client.py push-annot`. Both the local
jsonl and the server copy (`/localdisk-tmp/ssaa/annotations/<uuid>/`) are kept.

## 3b. Review & local data lifecycle
**The server is the source of truth; local dirs are prunable scratch.** Each of
the four working states maps to one canonical dir, and once its content is on
the server the local copy is redundant (re-pullable on demand):

| state | local dir | filled by | redundant after |
|---|---|---|---|
| awaiting hint | `raw_eps_remote/` | `claim-hint` | `push-hints` |
| hinted (submitted) | *(server only)* | — | — |
| annotated | `raw_eps_<worker>/` | `claim-annot` | `push-annot` |
| pulled for review | `ssaa_review/` | `pull-annot` | QA sign-off |

To review annotations (yours or another worker's), pull them fresh from the
server and open the viewer:
```
ssaa_client.py pull-annot                 # all annotated → ssaa_review/
ssaa_client.py pull-annot --outcome failure   # or a subset
cd policy/lap/scripts && ../.venv/bin/python3 view_droid_v3.py \
   --images-dir ../local_data/ssaa_review --suffix subagent_v3 \
   --hints ../local_data/ssaa_review/hints.md --load-video --port 7864
```
`local-status` shows where everything sits; `prune` (dry-run first) reclaims the
scratch copies whose content the server already has. Re-pull anytime with
`pull-annot`. Run `backup` regularly — it's your only durable copy if the remote
store is wiped. Multi-worker note: give each annotation worker a unique
`SSAA_WORKER` **and** its own `SSAA_RAW_EPS=<dir>` so claims never collide.

**Companion docs:** `HINTING.md` (human hint pass), `ANNOTATING.md` (one-shot
context-free annotation), `AUTO_ANNOTATE.md` (autonomous monitor loop — watch for
hinted eps, pull→annotate→push on its own, ≤50 parallel per batch).

## 3c. The spec is frozen — governance
`prompt_ssaa_v3.md` and `audit_v3.py` are the **shared, curated source of truth**.
Annotation workers (and their sub-agents) must **never edit them to make the
audit pass** — that's teaching-to-the-test and causes silent per-worker drift.
An audit failure means: fix the annotation (worker error), or escalate via
`push-report` (genuine spec gap). The lead runs `audit-all` to re-audit
everything centrally, aggregate the gate-issue patterns, and read the friction
notes; **spec changes are then made deliberately by the user + lead** — never by
a worker. (See `ANNOTATING.md` §"spec is frozen".)

## 4. Server coordinator (`coord.py`, on the pod at `/localdisk-tmp/ssaa/`)
Stdlib only; `state.json` (flock-guarded) is the single source of truth.
`python3 coord.py init|stats|claim|set-hint|mark-annotated|bulk-import|release|list`.
Status flow: available → hint_claimed → hinted → annot_claimed → annotated.
Re-sync the script if changed: `scp coord.py bitahub:/localdisk-tmp/ssaa/`.
Reclaim abandoned work: `ssh bitahub 'cd /localdisk-tmp/ssaa && python3 coord.py release --older-than 3600'`.

## Gotchas
- Episode dirs/times contain colons (`...16:23:33_2023`); rsync/scp handle
  them after the `bitahub:` host prefix — don't add extra escaping.
- The dataset is **task-diverse** (hang towel, pick-place, pour, …), not just
  pours. Hints carry the task; the prompt is general.
- `state.json` lives only on the server. Back it up: `ssh bitahub 'cp /localdisk-tmp/ssaa/state.json /localdisk-tmp/ssaa/state.bak.json'`.
- ~27 MB raw per episode — claim in modest batches.
