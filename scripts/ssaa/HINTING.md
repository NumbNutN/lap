# Hinting guide (for you, the human)

How to claim episodes, write hints by watching them in the viewer, and push.
Hints are the human knowledge the annotator can't get from images alone
(what the task is, where/why a failure happens, perception traps). The DROID
`current_task` label is generic and often wrong — **describe what the video
actually shows.**

Run everything from the repo root with the venv:
```
cd /home/numbnut/worksapce/RoboTwin
PY=policy/lap/.venv/bin/python3
CLIENT=policy/lap/scripts/ssaa/ssaa_client.py
```

### 0. Connection (once per ~4h)
If `$PY $CLIENT stats` errors, the SSH master dropped — re-open it (ask
Claude, or run the pexpect one-liner in `README.md` with the current
password). Then `stats` should print server counts.

### 1. Claim episodes to hint
```
$PY $CLIENT claim-hint --n 10                  # 10 of any outcome
$PY $CLIENT claim-hint --n 5 --outcome failure # only failures
```
This pulls + extracts them into `policy/lap/local_data/raw_eps_remote/` and
adds a `## <ep>` section per episode to that folder's `hints.md`, pre-seeded
with the (generic) task label.

### 2. Watch them and write hints
Launch the preview viewer:
```
cd policy/lap/scripts
../.venv/bin/python3 view_droid_v3.py \
   --images-dir ../local_data/raw_eps_remote --suffix subagent_v3 \
   --include-unannotated --hints ../local_data/raw_eps_remote/hints.md \
   --load-video --port 7864
```
Open http://localhost:7864, pick each episode, scrub the video. Then edit
`policy/lap/local_data/raw_eps_remote/hints.md` — replace each placeholder
line (`(task: …)`) under `## <ep>` with 1–4 sentences:
- what the task actually is (object, goal);
- for a **failure**, where it goes wrong and why (frame range + cause);
- perception traps (occlusion stretches, look-alike objects, the yellow
  sensor-housing block on the gripper, etc.).
You may reference frame numbers in the hint (they're stripped from the
trained annotation later). Leave a section unfilled to skip that episode.

### 3. Push the hints
```
$PY $CLIENT push-hints      # only filled sections are sent
$PY $CLIENT stats           # confirm "hinted" counts went up
```
Hinted episodes become claimable for annotation. You can repeat 1–3 in
batches; re-pushing is safe (it never downgrades already-annotated eps).
