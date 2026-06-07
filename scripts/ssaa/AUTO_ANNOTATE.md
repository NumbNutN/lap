# Autonomous annotation loop — for a Claude session with NO prior context

You are an **autonomous** SSAA-v3 annotation worker. Instead of doing one batch
and stopping, you run a monitor loop: periodically check the server for hinted
episodes, and whenever any exist, pull → annotate → validate → push on your own,
then keep watching. Per batch you may run **up to 50 episodes in parallel**.

This builds on `ANNOTATING.md` (read it once — it has the connection setup, the
exact subagent prompt, and the validation gate). This file adds the loop.

## One-time setup
1. Connection + identity — do `ANNOTATING.md` steps 1–2 once:
   - ensure the SSH master socket (reuse if alive; else `pip3 install --user
     pexpect` and the pexpect login — **ask the user for the SSH password**);
   - pick a UNIQUE worker id and a private working dir:
     ```
     cd <WORKSPACE>
     export SSAA_WORKER=auto-$$
     export SSAA_RAW_EPS=<WORKSPACE>/lap/local_data/raw_eps_$SSAA_WORKER
     PY="<WORKSPACE>/lap/.venv/bin/python3"; CLIENT="<WORKSPACE>/lap/scripts/ssaa/ssaa_client.py"
     ```
2. Decide a stop rule with the user up front (e.g. "run until I stop you", or
   "stop after N idle checks", or a target episode count). Default: run until the
   user interrupts, idling between batches.

## The loop
Repeat:

1. **Check for work.** Run `$PY $CLIENT stats`. Read `by_status.hinted` — that is
   the number of hinted-but-unannotated episodes (your queue).
2. **If `hinted == 0`:** there's nothing to do. Wait, then re-check — do NOT busy-poll.
   Use `ScheduleWakeup` with `delaySeconds` ≈ 900 (or the `Monitor` tool with an
   until-condition on the hinted count). Re-enter the loop on wake. Apply the stop
   rule (e.g. after a few consecutive idle wakes, stop and report).
3. **If `hinted > 0`:** take a batch.
   ```
   N=$(( hinted < 50 ? hinted : 50 ))     # cap 50 per batch
   $PY $CLIENT claim-annot --n $N --worker $SSAA_WORKER
   ```
   This atomically claims (flock — disjoint from every other worker), pulls +
   extracts each ep into `$SSAA_RAW_EPS/<uuid>/`, and writes each hint into
   `$SSAA_RAW_EPS/hints.md`.
4. **Annotate — one Sonnet subagent per claimed episode, launched in parallel
   (up to 50 at once, `run_in_background`).** Use the EXACT subagent prompt from
   `ANNOTATING.md` step 4 for each ep, filling in its `EP_PATH`, outcome (from the
   uuid path / hint), and the hint text from `$SSAA_RAW_EPS/hints.md`. Do not
   re-derive the schema — the prompt and its convention recap are authoritative
   (task-diverse data; the task is whatever the hint says, NOT pour-specific).
   Wait for all subagents in the batch to finish.
5. **Validate the whole batch** before pushing:
   ```
   $PY <WORKSPACE>/lap/scripts/data_pipeline/audit_v3.py \
      --raw-root $SSAA_RAW_EPS --pattern annotation_subagent_v3.json --out /tmp/a_$SSAA_WORKER.csv
   ```
   Every row must have `gate_ok=True`, `bounds_ok=True`, empty `gate_issues`,
   `n_spred_echoes_a=0`. Re-run (re-launch the subagent for) any failing ep before
   pushing it. Never push an ep that fails the gate.
6. **Push + checkpoint.**
   ```
   $PY $CLIENT push-annot            # scp annotations to server, mark annotated
   $PY $CLIENT backup                # read-only local mirror (remote isn't durable)
   $PY $CLIENT prune --yes           # drop this batch's now-redundant scratch dirs
   ```
7. Log a one-line batch summary (claimed uuids, audit pass count, running total),
   then go back to step 1. If any audit issues recurred or a rule felt
   ambiguous/impossible, submit a friction report (do NOT edit the spec):
   `$PY $CLIENT push-report --worker $SSAA_WORKER --note "…verbatim…"`.

## Guardrails
- **Parallel cap 50.** Never launch more than 50 subagents concurrently. If
  `hinted > 50`, the loop naturally drains it 50 at a time across iterations.
- **Disjoint by construction.** Multiple autonomous workers can run at once —
  each uses its own `SSAA_WORKER`/`SSAA_RAW_EPS`, and `claim-annot` is an atomic
  flock claim, so no two workers ever get the same episode.
- **Don't poll hot.** Idle waits are ≥ ~900s (`ScheduleWakeup`), not tight loops.
- **Gate is non-negotiable.** Only audit-clean episodes get pushed; a failing ep
  is re-annotated, not shipped.
- **Spec is frozen.** Never edit `prompt_ssaa_v3.md` or `audit_v3.py` (and don't
  let sub-agents) to make the audit pass — that's teaching to the test and
  causes silent per-worker drift. If a rule is genuinely ambiguous/impossible,
  record it in your batch report for central review; never patch the spec.
- **Backup every batch.** The remote store may be wiped; `backup` is your durable
  copy of all annotations + hints + state.
- **Socket drops.** If any `ssh`/`rsync` step fails because the master died,
  re-open it (`ANNOTATING.md` step 1, asking the user for the current password)
  and resume the loop.
- **Reclaim stalls.** If a worker died mid-batch leaving episodes stuck in
  `annot_claimed`, free them:
  `ssh bitahub 'cd /localdisk-tmp/ssaa && python3 coord.py release --older-than 3600'`.

## Kickoff prompt (paste into a blank Claude session)
> Read `<WORKSPACE>/lap/scripts/ssaa/AUTO_ANNOTATE.md` and run the autonomous
> annotation loop for SSAA-v3. Use worker id `auto-1`. Start a monitor that
> checks the server for hinted episodes; whenever any exist, pull → annotate
> (one Sonnet subagent per episode, up to 50 in parallel) → validate with
> `audit_v3.py` (require gate_ok, bounds_ok, 0 echoes) → `push-annot`, then
> `backup` and `prune --yes`. Idle ~15 min between checks when the queue is
> empty. Keep going until I stop you; report a one-line summary after each batch.
