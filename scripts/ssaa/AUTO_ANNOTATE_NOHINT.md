# Autonomous **no-hint** annotation loop — for a Claude session with NO prior context

You are an **autonomous** SSAA-v3 annotation worker running in **no-hint mode**.
The current push is to scale annotation past the human-hint bottleneck: you pull
**un-hinted `success` episodes straight from the server's `available` pool**,
annotate each from the observation alone (no human hint), validate, and push.
Per batch you may run **up to 50 episodes in parallel**.

This builds on `ANNOTATING.md` (read it once — connection setup, the exact
subagent prompt, the validation gate) and mirrors `AUTO_ANNOTATE.md` (the loop).
The ONLY differences here are: how you claim (no-hint), what queue you watch,
and that there is **no hint to read** — so the annotator must infer the task.

## ⚠️ Two things that bite in no-hint mode
1. **There is no hint.** Each ep's `hints.md` section is just a `[[NO HINT]]`
   marker, NOT guidance. Do not invent a hint or treat the marker as one. The
   subagent infers the task from the keyframe images + tools.
2. **The task's built-in description is very likely WRONG.** DROID episodes carry
   a coarse `current_task` category label (e.g. "Hang or unhang object"), and the
   SSAA-v3 system prompt does **not** caution against it. In this dataset that
   label is frequently mislabeled or empty — in the last batch, "Hang/unhang" and
   "Move lid" episodes were actually *open a drawer* and *upright a mug*. **Pass
   the category to the subagent only as a weak prior and tell it explicitly the
   label is often unreliable: trust the images + tools, not the label.** This
   instruction goes in the subagent prompt (Step 2 below) — without it, no-hint
   annotations drift toward the wrong task.

## The spec is frozen — do NOT edit it
`scripts/annotate_droid/prompt_ssaa_v3.md` and `scripts/data_pipeline/audit_v3.py`
are the shared source of truth. Never edit them (and never let a sub-agent edit
them) to make the audit pass. A repeated audit failure is a *signal*: fix the
annotation, or if a rule is genuinely ambiguous/impossible, comply as best you
can and record it in a friction report (step 7) — never patch the spec.

## One-time setup
1. Connection + identity — do `ANNOTATING.md` steps 1–2 once:
   - ensure the SSH master socket (reuse if alive; else `pip3 install --user
     pexpect` and the pexpect login — **ask the user for the SSH password**);
   - pick a UNIQUE worker id and a private working dir:
     ```
     cd <your lap checkout>            # dir containing scripts/ and .venv/
     export LAP=$(pwd)
     export SSAA_WORKER=nohint-$$
     export SSAA_RAW_EPS=$LAP/local_data/raw_eps_$SSAA_WORKER
     PY="$LAP/.venv/bin/python3"; CLIENT="$LAP/scripts/ssaa/ssaa_client.py"
     ```
2. Decide a stop rule with the user. Default target: the dataset goal is
   **200 annotated `success` episodes total**. Track progress via
   `$PY $CLIENT stats` → `annotated_by_mode.nohint.success` (your contribution)
   and `annotated.success` (overall). Stop when the overall success target is
   met, when no `available success` eps remain, or when the user interrupts.

## The loop
Repeat:

1. **Claim a no-hint batch** (this IS the work-check — no separate queue poll):
   ```
   $PY $CLIENT claim-annot --no-hint --outcome success --n 20 --worker $SSAA_WORKER
   ```
   This atomically claims (flock — disjoint from every other worker) up to N
   **un-hinted `success`** episodes straight from `available`, pulls + extracts
   each into `$SSAA_RAW_EPS/<uuid>/`, and writes a `[[NO HINT]]` marker per ep
   into `$SSAA_RAW_EPS/hints.md`. Server-side each ep is recorded
   `annot_mode=nohint`. Keep batches ≤ 50 (≈27 MB raw/ep).
2. **If it returns 0 episodes:** there are no `available success` eps left to
   claim. Don't busy-poll — `ScheduleWakeup` ~900s (or `Monitor` on the count),
   re-check on wake, and apply the stop rule after a few idle wakes.
3. **Annotate — one Sonnet subagent per episode, parallel, `run_in_background`
   (≤ 50 at once).** Use the EXACT subagent prompt from `ANNOTATING.md` step 4,
   with these no-hint substitutions:
   - **Hint:** `NONE — this is a NO-HINT annotation. Infer the task from the
     keyframe images + tools.`
   - **Task label:** include the DROID category (read it from the ep's
     `meta.json` `current_task`, often empty) as a *weak prior* and add verbatim:
     `the DROID category label is often unreliable/mislabeled — do NOT assume it;
     let the observed motion + gripper events tell you what is manipulated and
     the goal.`
   - Add a final instruction: `before finishing, re-read every S field and delete
     any "about to"/"will"/future phrasing (foreshadow is the most common gate
     failure), then run audit_v3.py on this one ep dir until gate_ok=True.`
   Wait for all subagents in the batch to finish.
4. **Validate the whole batch** (independent of subagent self-reports):
   ```
   $PY $LAP/scripts/data_pipeline/audit_v3.py \
      --raw-root $SSAA_RAW_EPS --pattern annotation_subagent_v3.json --out /tmp/a_$SSAA_WORKER.csv
   ```
   Every row must have `gate_ok=True`, `bounds_ok=True`, empty `gate_issues`,
   `n_spred_echoes_a=0`, `n_foreshadow=0`. Re-run (or surgically fix) any failing
   ep before pushing — central audit routinely catches a foreshadow a subagent's
   self-report missed. Never push an ep that fails the gate.
5. **Push + checkpoint.**
   ```
   $PY $CLIENT push-annot     # scp annotations → server, mark annotated (mode preserved = nohint)
   $PY $CLIENT backup         # durable local mirror (remote isn't durable)
   $PY $CLIENT prune --yes    # drop this batch's now-redundant scratch dirs
   ```
   `push-annot` needs no mode flag — the server already stamped `nohint` at claim
   time, so `stats.annotated_by_mode.nohint.success` rises by the batch size.
6. Log a one-line batch summary (claimed uuids, audit pass count, running
   `nohint.success` total). If a rule felt ambiguous/impossible or an audit issue
   recurred, file a friction report (do NOT edit the spec):
   `$PY $CLIENT push-report --worker $SSAA_WORKER --note "…verbatim…"`.
7. Go back to step 1.

## Guardrails
- **Parallel cap 50.** If you want more, drain across iterations.
- **Disjoint by construction.** Each worker uses its own `SSAA_WORKER`/
  `SSAA_RAW_EPS`; `claim-annot --no-hint` is an atomic flock claim, so no two
  workers ever get the same episode. No-hint claims draw only from `available`,
  so they never compete with the hinting workflow's `hinted` queue.
- **Don't poll hot.** Idle waits ≥ ~900s.
- **Gate is non-negotiable.** Only audit-clean episodes get pushed.
- **Spec is frozen.** Friction → report, never a per-worker prompt/audit edit.
- **Backup every batch.** The remote store may be wiped.
- **Socket drops.** Re-open the master (`ANNOTATING.md` step 1) and resume.
- **Reclaim stalls.** A worker that died mid-batch leaves eps in `annot_claimed`;
  free them (no-hint ones return to `available`, hinted ones to `hinted`):
  `ssh bitahub 'cd /localdisk-tmp/ssaa && python3 coord.py release --older-than 3600'`.

## Kickoff prompt (paste into a blank Claude session)
> Read `<LAP>/scripts/ssaa/AUTO_ANNOTATE_NOHINT.md` and run the autonomous
> **no-hint** SSAA-v3 annotation loop. Use a unique worker id. Claim un-hinted
> `success` episodes with `claim-annot --no-hint --outcome success` (≤50/batch),
> annotate one Sonnet subagent per ep — NO hint, and tell each subagent the DROID
> task label is often unreliable so it must infer the task from the images —
> validate with `audit_v3.py` (require gate_ok, bounds_ok, 0 echoes, 0
> foreshadow), then `push-annot`, `backup`, `prune --yes`. Idle ~15 min when no
> `available success` eps remain. Stop when overall `annotated.success` hits the
> 200 target or I stop you; one-line summary after each batch.
