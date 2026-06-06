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
   look, expand **🎬 Raw video** — an inline player with an **ext / wrist**
   camera toggle (click ▶ to play in-page), plus `▶ ext / wrist camera` links
   that open the raw MP4 in a browser tab, and the mirror folder path. The badge
   shows whether this ep is a 🟢 success or 🔴 failure.
2. In the **✍️ Write hint** box, type 1–4 sentences:
   - what the task actually is (object, goal);
   - for a **failure**, where it goes wrong and why (frame range + cause);
   - perception traps (occlusion stretches, look-alike objects, the yellow
     sensor block on the gripper, …). Frame numbers are fine — they're stripped
     from the trained annotation later.
3. (Optional) Type a few words then click **✨ Complete (LLM)** — a fast vision
   model reads the **current slider frame** (ext + wrist) plus your draft and
   fills in a finished hint, which you can edit. Saves repetitive typing.
4. Click **💾 Save hint** (writes to `hints.md`). Repeat for each episode; leave
   one blank to skip it.
   - **✓ Default hint** — for a trivially simple ep, one click fills a generic
     hint (then Save) so it's claimable without typing.
   - **🚫 Mark unusable** — for an ep that can't be annotated (no clear task,
     aimless, etc.): tags it, and `push-hints` then runs `exclude` on the
     server so it's never claimed again (and is dropped on export). Undo with
     `ssaa_client.py exclude --uuid <key> --undo`.
4. When done, **Ctrl-C** the viewer and run `hint-round` again — it pushes what
   you just wrote and pulls the next 10.

That's the whole loop: `hint-round` → write + save in the viewer → Ctrl-C →
`hint-round` → …

## Enabling ✨ Complete (vision-LLM hint completion)
Off by default. Configure one endpoint via env before launching `hint-round`
(the viewer inherits the env). Provider auto-detects from the keys:
```
# Anthropic Haiku (fast, multimodal):
export ANTHROPIC_API_KEY=sk-ant-...
# or any OpenAI-compatible endpoint:
export OPENAI_API_KEY=sk-...                 # default model gpt-4o-mini
# or a LOCAL model via Ollama (no API cost), needs a vision model pulled:
export SSAA_LLM_BASE_URL=http://localhost:11434/v1 SSAA_LLM_MODEL=llava SSAA_LLM_KEY=ollama
# or MiMo (Anthropic-compatible, multimodal reasoning model):
export SSAA_LLM_PROVIDER=anthropic SSAA_LLM_MODEL=mimo-v2.5 \
       SSAA_LLM_BASE_URL=https://token-plan-sgp.xiaomimimo.com/anthropic \
       SSAA_LLM_KEY=<your-mimo-token>
```
Reasoning models (mimo) spend tokens on an internal thinking block; bump
`SSAA_LLM_MAX_TOKENS` (default 600) if completions come back truncated.
Optional overrides: `SSAA_LLM_PROVIDER` (anthropic|openai), `SSAA_LLM_MODEL`,
`SSAA_LLM_KEY`. With nothing set, the button just reports "LLM not configured".

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
