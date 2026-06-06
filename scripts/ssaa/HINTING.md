# Hinting guide (for you, the human)

Hints are the human knowledge the annotator can't get from images alone (what
the task is, where/why a failure happens, perception traps). The DROID
`current_task` label is generic and often wrong — **describe what the video
actually shows.**

Run from the repo root with the venv:
```
cd /home/numbnut/worksapce/RoboTwin
PY=policy/lap/.venv/bin/python3
CLIENT=policy/lap/scripts/ssaa/ssaa_client.py
```

## The one-command loop: `hint-round`
Each round is a single command that **(1)** submits the hints you wrote last
round, **(2)** prunes those now-submitted episodes from your workspace, **(3)**
claims a fresh batch (default **8 success + 2 failure**), and **(4)** opens the
viewer so you can write the new hints in-page:
```
$PY $CLIENT hint-round                      # 8 success + 2 failure, viewer on :7864
$PY $CLIENT hint-round --success 7 --failure 3   # adjust the mix
$PY $CLIENT hint-round --no-viewer          # claim only, print the viewer command
```
In the viewer (http://localhost:7864):
1. Pick an episode from the dropdown, scrub the `frame_idx` slider (with
   `--load-video` it plays the real video, not just keyframes). For a closer
   look, expand **🎬 Raw video** — an inline player (click ▶) plus `▶ ext / wrist
   camera` links that open the raw MP4 in a browser tab, and the mirror folder
   path. The badge shows whether this ep is a 🟢 success or 🔴 failure.
2. In the **✍️ Write hint** box, type 1–4 sentences:
   - what the task actually is (object, goal);
   - for a **failure**, where it goes wrong and why (frame range + cause);
   - perception traps (occlusion stretches, look-alike objects, the yellow
     sensor block on the gripper, …). Frame numbers are fine — they're stripped
     from the trained annotation later.
3. Click **💾 Save hint** (writes to `hints.md`). Repeat for each episode; leave
   one blank to skip it.
4. When done, **Ctrl-C** the viewer and run `hint-round` again — it pushes what
   you just wrote and pulls the next 10.

That's the whole loop: `hint-round` → write + save in the viewer → Ctrl-C →
`hint-round` → …

## Connection
If the first command errors, the SSH master dropped — re-open it (ask Claude, or
run the pexpect one-liner in `README.md` with the current rotating password),
then retry.

## Notes
- The success/failure pools are finite (≈2006 / 495). Failures are rarer and
  more informative — keeping ~2 per round is a good default; `hint-round` just
  prints "queue exhausted" for an outcome once it runs out.
- Hinted episodes become claimable for annotation (see `ANNOTATING.md` /
  `AUTO_ANNOTATE.md`). Re-running `hint-round` never downgrades an already
  annotated episode.
- Prefer editing in the viewer, but you can still hand-edit
  `policy/lap/local_data/raw_eps_remote/hints.md` and `push-hints` directly.
