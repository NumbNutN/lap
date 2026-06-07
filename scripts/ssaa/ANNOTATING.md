# Annotation worker guide — for a Claude session with NO prior context

You are a Claude annotation worker for the SSAA-v3 DROID dataset. Follow this
top to bottom. You pull hinted episodes from a shared server, launch one
sub-agent per episode to write the annotation, validate, and push back. Many
Claude workers (even on different machines) can run at once — the server hands
each worker **disjoint** episodes (atomic flock claim), so just pick a unique
worker id and go.

**Paths.** All paths are relative to your **lap checkout root** — call it `$LAP`
(in the RoboTwin monorepo that's `policy/lap`; as a standalone clone it's the
repo root). First:
```
cd <your lap checkout>     # the dir containing scripts/ and .venv/
export LAP=$(pwd)
PY="$LAP/.venv/bin/python3"; CLIENT="$LAP/scripts/ssaa/ssaa_client.py"
```
Env not set up yet? See the lap README → "SSAA-v3 Data Annotation" (a minimal
`uv venv` + `scripts/ssaa/requirements-annotate.txt`, no training stack).

## ⚠️ The spec is frozen — do NOT edit it
`scripts/annotate_droid/prompt_ssaa_v3.md` and
`scripts/data_pipeline/audit_v3.py` are the shared, curated source of truth.
**Never edit them — and never let a sub-agent edit them — to make the audit
pass.** A repeated audit failure is a *signal*, not something to silence:
- if it's your annotation's error → fix the annotation and re-run;
- if you believe a rule is genuinely ambiguous, contradictory, or impossible for
  an episode → comply as best you can and **record it in a report** (step 7) for
  central review. Do not add HARD RULEs or patch the prompt yourself.

Per-worker spec edits cause silent drift: different workers end up annotating to
different specs and the dataset becomes inconsistent. Spec changes are made
centrally and deliberately, never by the annotator to satisfy its own auditor.

## 1. Open the SSH connection (shared master socket)
```
$PY $CLIENT stats
```
- If that prints JSON counts, the master socket is alive — **skip to step 2.**
- If it errors (no socket): install pexpect and open the master with the
  SSH password. **Ask the user for the password** (it's persistent — but never
  commit it or write it to a tracked/shared file), then:
  ```
  $LAP/.venv/bin/python3 -m pip install pexpect    # if missing
  SSAA_PW='<password from user>' $PY - <<'PY'
  import pexpect, os
  c = pexpect.spawn("ssh bitahub 'echo UP'", timeout=45, encoding='utf-8')
  while True:
      i = c.expect([r'[Pp]assword:', r'continue connecting', pexpect.EOF, pexpect.TIMEOUT])
      if i==0: c.sendline(os.environ['SSAA_PW'])
      elif i==1: c.sendline('yes')
      else: break
  print(c.before)
  PY
  ```
  (Requires `~/.ssh/config` with a `bitahub` ControlMaster host — see
  `scripts/ssaa/README.md` §0.) Then re-run `stats`.

## 2. Set your worker identity + a private working dir
Pick a unique id so two workers don't collide. Use it on **every** command:
```
export SSAA_WORKER=claude-$$        # or any unique tag
export SSAA_RAW_EPS=$LAP/local_data/raw_eps_$SSAA_WORKER
```
(`SSAA_RAW_EPS` keeps your episodes in your own folder; `push-annot` only
pushes from there.)

## 3. Claim hinted episodes
```
$PY $CLIENT claim-annot --n 5 --worker $SSAA_WORKER
```
This pulls + extracts K hinted, un-annotated episodes into `$SSAA_RAW_EPS/<uuid>/`
and writes each episode's hint into `$SSAA_RAW_EPS/hints.md`. If it returns 0
episodes, there are no hinted-and-unannotated episodes left (check `$PY $CLIENT stats`).

## 4. Annotate — one sub-agent per episode
For EACH claimed episode dir `$SSAA_RAW_EPS/<uuid>/`, read its hint from
`$SSAA_RAW_EPS/hints.md` (the `## <uuid>` section), then launch ONE sub-agent
(Agent tool, model **sonnet**, run_in_background) with this prompt — substitute
`<LAP>` with your absolute checkout path, and fill EP_PATH, OUTCOME
(success/failure from the uuid path or hint), and the HINT:

```
Annotate ONE DROID episode for SSAA-v3. Produce two JSON files. Follow the
system prompt EXACTLY; the task is whatever the hint says (task-diverse
dataset — NOT necessarily a pour).

STEP 1 — Read the system prompt IN FULL and apply ALL of it:
  <LAP>/scripts/annotate_droid/prompt_ssaa_v3.md
This prompt is a FROZEN spec — follow it, do NOT modify it. If a rule seems
impossible for this episode, comply as best you can and FLAG it in your return
(never work around it by editing the prompt).
Conventions (all in the prompt — don't re-derive): begin/end keyframes are
S-only brackets (A=S_pred=A_correct=null) and the Plan rides on the FIRST
MOVING keyframe (its policy-target field leads with <think>Plan…</think>);
chunk_end = intent boundary (consecutive same-intent keyframes SHARE one
chunk_end, no tiling); action frame follows the view that grounds it
(occlusion→wrist, rotation→wrist, gross translation→robot base); gripper from
the tool's `gripper` field described qualitatively (never the raw 0–1 value),
a grasp/release tag marks where the event begins; S present-only (no
foreshadow); S_pred from the chunk_end image, object-centric/task-critical,
no echo of A's cm/°; physical-effect <think> where a motion's point is a
consequence; imit non-monotone, A_correct only when imit=false; failures
diverge at the pre-failure keyframe (imit=false + A_correct); A imperative
first-person; no leakage ("the demo"/frame indices/"N frames"/raw gripper
value). Schema fields: frame_idx, phase_type, S, S_pred, A, A_correct,
chunk_end_frame, imitation_supervised (NO mode_marker).

STEP 2 — Episode:
  EP_PATH = <$SSAA_RAW_EPS/<uuid>>
  Outcome = <success|failure>
  Hint: <the hint text>

STEP 3 — Tools (EXACT venv python):
  PY=<LAP>/.venv/bin/python3
  CLI=<LAP>/scripts/data_pipeline/tools_cli.py
  - $PY $CLI keyframes "<EP_PATH>"
  - $PY $CLI pose_delta "<EP_PATH>" <i> <j>   (returns delta_robot/ee, rot, gripper)
  - $PY $CLI image "<EP_PATH>" <frame> ext|wrist /tmp/_img_<uuid>.jpg  (then Read it)
  Every cm/° in A = pose_delta over THAT keyframe's own [frame_idx,chunk_end_frame] span.

STEP 4 — Keyframe images are on disk in EP_PATH as kfNN_fFFFF.jpg (ext) and
kfNN_fFFFF_wrist.jpg (wrist). Read them for S. For each S_pred, Read the
chunk_end image. Use the wrist view where the external view is occluded.

STEP 5 — keyframe list → phases (group by sub-intent → shared chunk_end) →
per acting keyframe: pose_delta → choose grounding frame → A (imperative, with
<think> per the gate) → image(chunk_end) → S_pred → S (present-only).
begin/end = S-only; Plan on the first moving keyframe. keyframes length =
keyframe-list length, in order, matching frame_idx; frame_idx < chunk_end_frame
<= frame_idx+60.

STEP 6 — Overwrite <EP_PATH>/annotation_subagent_v3.json (schema above) and
<EP_PATH>/annotation_subagent_v3.json.audit.json (one chunk_end_revisions per kf).

Return: n_keyframes; begin/end S-only + Plan on first move; where imit flips
(if any); confirm no tiling/leakage/echo/foreshadow/raw-gripper.
```

Launch them in parallel (one Agent call each, run_in_background). Wait for all
to finish.

## 5. Validate before pushing
```
$PY $LAP/scripts/data_pipeline/audit_v3.py \
   --raw-root $SSAA_RAW_EPS --pattern annotation_subagent_v3.json --out /tmp/a_$SSAA_WORKER.csv
```
Open the CSV. Each row must have `gate_ok=True`, `bounds_ok=True`, empty
`gate_issues`, and `n_spred_echoes_a=0`. Also grep each annotation for
leakage (`kf\d`, `frame \d`, `\d+ frames`, "the demo", `0.\d\d` gripper) and
S-foreshadowing ("about to", "will open/close/lift"). Fix or re-run any
episode that fails before pushing it.

## 6. Push
```
$PY $CLIENT push-annot
```
This scp's every validated annotation in `$SSAA_RAW_EPS` to the server
(`/localdisk-tmp/ssaa/annotations/<uuid>/`), keeps the local copy, and marks
them annotated. `$PY $CLIENT stats` should show the annotated count rise.

## 7. Loop & report
Repeat 3–6 until `stats` shows no hinted-and-unannotated episodes. Then submit a
friction report — the ONLY channel for spec problems (do NOT edit
`prompt_ssaa_v3.md` or `audit_v3.py`):
```
$PY $CLIENT push-report --worker $SSAA_WORKER \
   --note "any audit issues you could not resolve; any rule that felt ambiguous/impossible (verbatim)"
```
This re-audits your workspace and uploads the audit CSV + your note to the
server for central review. Also give the user a short chat summary (how many
annotated, any skipped and why).

## Notes
- The dataset is task-diverse and DROID `current_task` labels are unreliable —
  trust the hint + the images, not the label.
- ~27 MB raw per episode; claim modest batches (5–10).
- If your master socket drops mid-run, re-open it (step 1) and continue.
- Reclaim work an abandoned worker left:
  `ssh bitahub 'cd /localdisk-tmp/ssaa && python3 coord.py release --older-than 3600'`.
