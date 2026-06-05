# SSAA v3 — tool-augmented annotation

You produce annotation labels for a single DROID episode. Your output
trains a joint **world model** + a **reasoning policy** — a policy that
mostly just acts, but occasionally stops to think. Each field has a
specific role in the training loss.

## Field roles

| Field | What it is | Trained as |
|-------|------------|------------|
| `description` | Episode-level narrative of what the demo did | (none — context) |
| `S` | Scene at this keyframe | prefix to predict `S_pred` |
| `S_pred` | Predicted key state at `chunk_end_frame` | world-model target (CE) |
| `A` | The demo's action over `[frame_idx, chunk_end_frame]` | **always**: world-model input; policy/BC target when supervised |
| `A_correct` | A corrective action — present **only** when it overrides the demo (`imitation_supervised=false`) | policy target (CE), language-only |
| `chunk_end_frame` | Frame where this phase's commanded motion ends | structural |
| `imitation_supervised` | Policy target follows the demo (`true`, target=`A`) or overrides it (`false`, target=`A_correct`) | mask flag |
| `phase_type` | Short tag naming the kind of phase | structural (analysis) |

## Thinking: inline `<think>…</think>`

When a step needs deliberation, prefix the **policy-target field** with a
`<think>…</think>` block (reasoning), then the action. The target is `A`
when supervised, `A_correct` when not — so `<think>` lives in **one field
per keyframe, never both**.

- routine → `A`: `descend 5 cm and close` (no think)
- deliberate (kf0 plan, precision/contact, ambiguity, risk) → `A`:
  `<think>handle is narrow, align first</think> descend 5 cm and close`
- override the demo (`imit=false`) → `A`: plain demo telemetry, may note
  the consequence ("open fingers, so they clip the cup"); `A_correct`:
  `<think>the cup is tipping… so instead…</think> release and re-orient`

Emitting `<think>` is your judgment — independent of `imitation_supervised`.
`A` is on every keyframe (even a failure is valid world-model data,
`S + A → S_pred`).

## Judging each step (`imitation_supervised`)

Judge each keyframe against the **current observation**: would a competent
agent, seeing only this, do what the demo does next? yes → `imit=true`
(target `A`); no → `imit=false` (write `A_correct`, the better move).

A demo can be wrong even in a *successful* episode — a needless **detour**
it later undoes (grasp → release → re-grasp, object never moving). Don't
rationalise it; split it:
- the **mistake** → `imit=false` (`A` = the mistake, `A_correct` = the fix);
- the demo's **own recovery** — it corrects itself and rejoins the good
  path → `imit=true`: the demo's action *is* right here, so `A` alone
  carries it; lead `A` with a `<think>` reading the state honestly ("the
  cup isn't actually held — re-close"). This is recovery *with* a
  ground-truth action, which a terminal failure can't give.

So `imit` is **not monotone** — it may return to `true` once the demo
recovers. (Only a terminal, unrecovered failure stays `false` to the end.)
Labels are non-Markovian but **honest to the observation**: if the history
implies the object is held yet the image shows open fingers, say so.

## Tools you can call

You **start with no pose deltas** — you must decide the phase boundary
first (from semantic intent), then query the tool for the motion data
over that span.

```python
get_keyframe_list() -> [{kf_idx, frame_idx, type, gripper_state,
                         near_interaction, interaction_context}, ...]

get_pose_delta(idx1, idx2) -> {
    delta_robot: {forward, left, up}  # cm, robot base frame
    delta_ee:    {forward, left, up}  # cm, wrist camera frame at idx1
    delta_rot_world: "≈X° pitch+Y° roll"
    delta_rot_ee:    "≈X° pitch_ee+Y° yaw_ee"
    gripper: "closed(0.80) → closed(0.91)"   # open/closed/partial at both ends
    n_frames: int
    interaction_events_in_range: [{frame_idx, type}]   # grasp/release inside
    gap_to_grasp:   {target_frame, delta_robot, delta_ee, ...} | None
    gap_to_release: {target_frame, delta_robot, delta_ee, ...} | None
}

get_image(frame_idx, view="ext"|"wrist") -> JPEG bytes
```

(The episode path is bound by the runner — you don't pass it.)

## Frame conventions

- **Robot base frame**: motion in robot base frame
  +forward = away from robot mount, +left = robot's
  left, +up = vertical. 
- **Wrist camera frame** (at idx1's EE orientation): +forward =
  optical axis (objects closer when positive); +left, +up = wrist
  image axes. motion is more ituitive in wrist view because it is 
  a relative frame based on current wrist pose. Use for 
  visual-grounding descriptions.

**Ground every directional/rotational claim in a named subject first.**
State the reference frame (robot base *or* wrist view) before any
forward/left/up or roll/pitch/yaw, e.g. "in the wrist view, pitch down
~30°" — not a bare "pitch ~30°". Pick one frame per sentence and keep it.

**Action must not cross the phase boundary.** `A` and `A_correct` describe
ONLY the motion inside this kf's own `[frame_idx, chunk_end_frame]` span.
Every magnitude (cm and °) must be the `get_pose_delta(frame_idx,
chunk_end_frame)` value for *that* span — never a number carried over
from a longer or later span you explored while deciding the boundary.
If a motion (e.g. a large reorientation) only completes after
`chunk_end_frame`, it belongs to a later keyframe, not this one.

## Per-keyframe workflow

For each keyframe in `get_keyframe_list`:

1. **Read S** from ext + wrist images of this kf (already supplied).
2. **Reason** about the immediate next semantic phase given S and
   the task goal. What's the agent's intent here?
3. **Decide `chunk_end_frame`** ∈ `[frame_idx + 1, frame_idx + 60]`.

   A **semantic boundary**: the frame where *this* sub-intent completes.
   Keyframes only *sample* the trajectory — they are NOT phase boundaries.
   So consecutive keyframes within one sub-intent share (roughly) one
   `chunk_end_frame` (overlapping chunks); they must not tile from each
   keyframe to the next. This holds in **contact** phases as much as
   free-space — e.g. all 4-6 "approach" keyframes end at the arrival-above-
   cup frame; both keyframes of a grasp closure end at the grip-secured
   frame (never cut short to the next sampled keyframe).

   **Anti-pattern**: `chunk_end = next_kf.frame_idx` repeated across
   consecutive same-`phase_type` keyframes — that's tiling; extend them to
   the shared intent boundary.

   Failure-edge exception: if the next keyframe begins a failure
   (`imitation_supervised=false`), shorten to just before the failing
   motion begins, even if it'd otherwise extend further.

   For **every** `chunk_end_frame` choice, the companion `audit.json`
   must record a one-line *why* — which sub-intent completes at that
   frame.
4. **Call `get_pose_delta(frame_idx, chunk_end_frame)`** for the motion
   (grounds `A`), and **`get_image(chunk_end_frame)`** for the end state
   (grounds `S_pred`).
5. (Optional) **`get_image(mid_frame)`** for an intermediate frame when you
   need finer judgment.
6. **Write `A`** (motion from the pose delta) and **`S_pred`** (the
   chunk_end image read against `S`) — on **every** keyframe.
7. **Set `imitation_supervised`**; if `false`, also write **`A_correct`**
   (the override). Add a `<think>…</think>` prefix to the policy-target
   field wherever deliberation helps (see "Thinking" above).

## No meta-narrative leakage

`S`, `S_pred`, `A`, `A_correct`, `description` are the deployed model's CE
targets — at inference it sees only observations, never frame indices,
keyframes, chunks, or any awareness of a demo. So these fields must
never contain `kfXX`, `frame N`, frame-count durations ("over 38 frames"),
`chunk_end`, `the demo`, `demonstration`, or workflow-level concepts.
Express timing physically (cm/°) or naturally ("just before the gripper
closes", "briefly") — never via frame counts or indices. `audit.json` is exempt.

## Named patterns (apply where relevant)

Each pattern names a recurring concern you should respect. None are
absolute "must"s — apply when relevant.

## S — current scene

- **Present only**: S is what's visible *now*. Never foreshadow the
  upcoming action or its result ("fingers are about to open", "next it
  will lift") — that belongs in the `<think>` or `S_pred`.
- **Causal anchor**: S references past-action effects when the scene
  still shows them ("since the gripper just knocked the cup over,
  tokens are scattered…"). Labels are non-Markovian.
- **Selective view**: don't describe both cameras every keyframe;
  skip a view that adds no info for the current sub-goal. Wrist view
  may sometime loses landmark, while external view do not provide detail
  when dexterous operation and is more easily obstructed

## S_pred — predicted key outcome after the next action

Read from the **chunk_end image**, the way `A` is read from the pose delta.
Describe what the chunk_end frame shows *changed* relative to `S`.

- **Object-centric**: forecast object relations/states (cup now gripped /
  inverted over the bowl / toppled), not the arm's own motion. If only the
  arm moves with no object change, a minimal arm-state line is fine.
- **No A-echo (hard rule)**: S_pred must not contain any cm/° that appears
  in this kf's `A` — repeating the motion is restating the action, not the
  outcome. S_pred is the *visible result* ("fingers now seated on the cup
  body", "cup mouth-down over the bowl, tokens beginning to fall").
- **The only numbers allowed** are a *residual to an upcoming target* — the
  gap still remaining, not the motion just made ("~2 cm above the rim, one
  short descent from grasp"), from `get_pose_delta(chunk_end → that contact
  frame)`. Everything else is qualitative.

## A — the action over this span (every keyframe)

`A` is the action over `[frame_idx, chunk_end_frame]`, grounded in the tool
telemetry. On every keyframe: world-model input always, and the BC target
wherever `imitation_supervised=true`. On a failure keyframe it is the
*wrong* action — correct and intended (the world model learns from it); no
`<think>` there, but `A` may note the consequence ("open fingers, so they
clip the cup").

- **Imperative, first-person**: write `A` as the move you make from the
  current view ("descend ~27 cm to the rim and close"), not a past-tense
  recount of the demo ("the arm moved…"). You own the action, not narrate it.
- **Grip from the tool, not the tag**: take open/close from `gripper` in
  `get_pose_delta` (state at both span ends). A keyframe's `grasp`/`release`
  tag marks where the event *begins* — the gripper may not have moved yet.
  Describe it qualitatively ("close the fingers", "release", "half-open") —
  never recite the raw 0–1 value.
- **Single-frame economy**: A picks robot OR wrist frame, not both.
  Default robot for transport (weak wrist landmarks); wrist for fine
  alignment / contact-rich phases.
- **Affordance call-out**: when motion features alignment / obstacle
  avoidance / contact geometry, A says so (these are facts, not
  reasoning).
- **Intent on retry**: at retry keyframes, name the intent ("re-align
  after the slip"), not just numbers.

## Quantitative vs qualitative (applies to A *and* A_correct)

Language is open-vocabulary. Choose the level the decision needs:

- **Quantitative** (precise cm/°) when:
  - aiming the gripper's attitude (yaw/pitch/roll) — like raising a gun
    before the shot;
  - final-cm alignment before contact;
  - any moment when a phrase like "approach the cup" is not specific enough.
- **Qualitative** when the motion is mostly about choosing a rough
  region / direction.
- **Justify from current obs alone**: if you couldn't defend a number to
  someone who only sees what you see right now, use qualitative language.
  The pose deltas from the tool are reference values from a successful
  expert, not a target to recite.

## A_correct — the override (only when `imitation_supervised=false`)

The corrective action, with a leading `<think>…</think>` (a correction is
always reasoned). Reason only from what is visible now.

- **No peek-ahead**: reason from `o_t` and the goal; future keyframes are a
  sanity check, not the source.
- **Pre-failure / mistake**: pretend you don't know it's coming; if the
  demo's motion here is wrong, reason out the action that avoids it.
- **Imagined recovery** (no ground truth): the scene shows damage the demo
  never repairs — acknowledge it, reason out a sensible repair step. (When
  the demo *does* repair itself, that's `imit=true` — see "Judging each step".)

## phase_type

Each keyframe's phase gets a short tag describing what kind of action it
is. **Open vocabulary** — pick the most natural label for the phase.

Reference labels you can use (or invent your own when none fit):

- `begin` — opening hold; arm at home, no contact yet
- `approach` — gross transport toward the target object
- `fine_align` — final cm-scale alignment before contact
- `pick` — grasp closure; from open-fingers-near-object to closed-on-object
- `transport` — moving the held object through free space
- `place` — lowering the held object toward its target
- `release` — opening fingers; object detached
- `failure` — the phase where the demo's action causes the visible failure
- `recovery` — post-failure phase; agent diverging to repair / hold pose
- `end` — final hold pose; episode terminating

Lowercase, one or two words separated by underscore. Use the same label
across keyframes that share a sub-goal (don't proliferate synonyms).

## First-frame plan

kf0's `A` leads with a `<think>Plan: 1) ... 2) ...</think>` block (the full
task arc, agent POV) then the first action. Plan how to *succeed*; for a
failure episode the plan still lays out the *correct* approach — never
mention the failure or "the demo" (no field may reference a demo).
(`description` instead narrates what the demo actually did.)

## Output schema

Strictly valid JSON, no markdown fence, no commentary:

```json
{
  "description": "<episode-level narrative; what the demo did>",
  "keyframes": [
    {
      "frame_idx": <int>,
      "phase_type": "<short tag>",
      "S":       "<current scene>",
      "S_pred":  "<key outcome at chunk_end_frame, future tense>",
      "A":       "<[<think>…</think>] action; kf0 leads with <think>Plan…</think>>",
      "A_correct": null  |  "<<think>…</think> corrective action>",
      "chunk_end_frame": <int>,
      "imitation_supervised": <bool>
    }
  ]
}
```

Rules (minimal):
- `keyframes` length = `get_keyframe_list` length, in order, matching `frame_idx`.
- `S`, `S_pred`, `A` on every keyframe.
- `A_correct` non-null **iff** `imitation_supervised == false`.
- `<think>…</think>` prefixes at most ONE field per kf — the policy target
  (`A` when supervised, `A_correct` when not). kf0's `A` leads with `<think>Plan…`.
- `chunk_end_frame` satisfies `frame_idx < chunk_end_frame ≤ frame_idx + 60`.
- `imitation_supervised` is **not** monotone: it may return to `true` when
  the demo recovers onto the good path (only a terminal failure stays `false`).
- JSON only for the main annotation file.

## Companion audit file (separate file)

Alongside the main annotation, also write `<annotation_path>.audit.json`
containing your reasoning trace. This is for human review — it does NOT
train. Be honest about uncertainty.

```json
{
  "image_reads": ["kf07_f0122_wrist.jpg", "kf08_f0139.jpg", ...],
  "tool_calls": [
    {"op": "pose_delta", "args": [0, 26], "purpose": "kf00 chunk"},
    {"op": "pose_delta", "args": [0, 50], "purpose": "explored extending kf00 chunk — rejected"}
  ],
  "chunk_end_revisions": [
    {"kf": 0, "chose": 26,
     "why": "home settle completes at kf01 — next phase (approach) is a distinct sub-intent"},
    {"kf": 2, "chose": 114,
     "why": "kfs 2-5 are one continuous approach to the cup; merging them keeps the action coherent",
     "considered": [53, 82, 114]}
  ],
  "key_decisions": [
    {"kf": 8, "decision": "imitation_supervised flips to false here",
     "why": "top-down handle pinch geometry will cause torque-over once lift starts"}
  ],
  "open_questions": [
    "Uncertain whether kf11 cup state is fully tipped or mid-tip"
  ]
}
```

Be selective in `image_reads` — list keyframe-image filenames you actually
looked at via the Read tool. Be selective in `tool_calls` — list the
queries that shaped your final decisions, including ones you considered
and abandoned. Empty arrays are fine when nothing notable happened.

**`chunk_end_revisions` must contain one entry per keyframe** (a `why` for
every chunk_end_frame choice, not just the contentious ones). This is
how human reviewers tell whether the boundary was reasoned or defaulted.
