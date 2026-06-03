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
| `A` | What the demo physically did over `[frame_idx, chunk_end_frame]` | **always**: world-model input; and the policy/BC target |
| `A_pred` | intended action when human domonstration fails  | policy target (CE), language-only |
| `mode_marker` | `[act]` (routine) or `[think_act]` (deliberate) — you PREDICT this | the chain-of-thought gate |
| `chunk_end_frame` | Frame where this phase's commanded motion ends | structural |
| `imitation_supervised` | Whether the policy target follows the demo (`true`) or overrides it (`false`) | mask flag |
| `phase_type` | Short tag naming the kind of phase | structural (analysis) |

## The act / think gate

`mode_marker` is your per-keyframe call — does this step need thought?
- **`[act]`** — obvious move; write `A` only.
- **`[think_act]`** — reasoning helps; also write `A_pred` (reason → intended action).

Think when it adds value: kf0 (plan the task), precision/contact (alignment,
attitude, grasp), ambiguous choices, anticipated risk, pre-failure, recovery.
This is your judgment, not a function of `imitation_supervised`.

Rules:
- `A_pred` present ⟺ `[think_act]`.
- `imitation_supervised=false` (`A_pred` overrides the demo) ⟹ `[think_act]`, and is monotone once false.
- `A` is on every keyframe — even a failure is valid world-model data (`S + A → S_pred`).

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

**Action must not cross the phase boundary.** `A` and `A_pred` describe
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

   `chunk_end_frame` is a **semantic boundary, not a structural one.**
   Ask: *when does this sub-intent complete?* (the one you name in `A`,
   or in `A_pred` on a `[think_act]` step). That frame is `chunk_end_frame`.

   Granularity guide:
   - **Free-space motion** (approach, retract, transport over the air)
     → typically ONE phase spanning **multiple keyframes** (e.g. 4-6 kfs
     merged into one "approach the cup" chunk). The rule-detector
     keyframes sample the trajectory; they are NOT phase boundaries.
   - **Contact-rich / fine-motor** (fine_align, grasp closure, place)
     → may be 1-3 kfs.
   - **Atomic events** (the actual grasp moment, the actual release
     moment) → may be ≤ 5 frames.

   **Anti-pattern check** — if you find yourself setting
   `chunk_end_frame = next_kf.frame_idx` for many *consecutive* kfs
   that share the same `phase_type`, you are almost certainly
   under-merging. Re-check whether those kfs all belong to one
   semantic phase.

   Failure-edge exception: if the next keyframe begins a failure
   (`imitation_supervised=false`), shorten to just before the failing
   motion begins, even if it'd otherwise extend further.

   For **every** `chunk_end_frame` choice, the companion `audit.json`
   must record a one-line *why* — which sub-intent completes at that
   frame.
4. **Call `get_pose_delta(frame_idx, chunk_end_frame)`** to
   fetch the motion data for that span.
5. (Optional) **Call `get_image(mid_frame, "wrist")`** to
   inspect an intermediate frame when you need finer judgment.
6. **Write `A`** (the demo's motion over the span, grounded in the tool
   result) and **`S_pred`** (forecast at chunk_end_frame). These two are
   written on **every** keyframe.
7. **Set `mode_marker` + `imitation_supervised`** per the act/think gate
   above — write `A_pred` only on `[think_act]`; set `imitation_supervised
   = false` where your action overrides the demo.

## No meta-narrative leakage

`S`, `S_pred`, `A`, `A_pred`, `description` are the deployed model's CE
targets — at inference it sees only observations, never frame indices,
keyframes, chunks, or any awareness of a demo. So these fields must
never contain `kfXX`, `frame N`, `chunk_end`, `the demo`,
`demonstration`, or workflow-level concepts. Express timing naturally
("just before the gripper closes", "a moment later"), not via indices.
`audit.json` is exempt.

## Named patterns (apply where relevant)

Each pattern names a recurring concern you should respect. None are
absolute "must"s — apply when relevant.

## S — current scene

- **Causal anchor**: S references past-action effects when the scene
  still shows them ("since the gripper just knocked the cup over,
  tokens are scattered…"). Labels are non-Markovian.
- **Selective view**: don't describe both cameras every keyframe;
  skip a view that adds no info for the current sub-goal. Wrist view
  may sometime loses landmark, while external view do not provide detail
  when dexterous operation and is more easily obstructed

## S_pred — predicted key outcome after the next action

- **Object-centric**: forecast object relations/states, not the
  arm's own motion. If only the arm moves and no object relation
  changes, a minimal arm-motion line is fine.

## A — what the human demo physically did (every keyframe)

`A` is grounded in the tool telemetry and is written on every keyframe —
it is the world-model input always, and the BC target wherever
`imitation_supervised=true`. On a failure keyframe `A` is the *wrong* motion
the demo made — that is correct and intended (the world model learns from it).

- **Single-frame economy**: A picks robot OR wrist frame, not both.
  Default robot for transport (weak wrist landmarks); wrist for fine
  alignment / contact-rich phases.
- **Affordance call-out**: when motion features alignment / obstacle
  avoidance / contact geometry, A says so (these are facts, not
  reasoning).
- **Demo intent on retry**: at retry keyframes, A explains the demo's
  intent ("re-aligning after slip"), not just numbers.

## Quantitative vs qualitative (applies to A *and* A_pred)

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

## A_pred — the deliberation (on `[think_act]` steps)

Chain-of-thought fused with the intended action, reasoned only from what
is visible now.

- **No peek-ahead**: reason from `o_t` and the goal; future keyframes are a
  sanity check, not the source.
- **Pre-failure** (`imitation_supervised=false`): pretend you don't know
  the failure is coming; if the demo's motion here causes it, reason out
  the action that avoids it.
- **Recovery** (`imitation_supervised=false`): acknowledge the damage, reason
  out a sensible repair step.

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

kf0 is `[think_act]`: its `A_pred` leads with a `"Plan: 1) ... 2) ..."`
block (the full task arc, agent POV) then the first-kf action. Plan how to
*succeed* — for failure episodes, replace the demo's failing steps.
(`description` instead narrates what the demo actually did.)

## Output schema

Strictly valid JSON, no markdown fence, no commentary:

```json
{
  "description": "<episode-level narrative; what the demo did>",
  "keyframes": [
    {
      "frame_idx": <int>,
      "mode_marker": "[act]"  |  "[think_act]",
      "phase_type": "<short tag>",
      "S":       "<current scene>",
      "S_pred":  "<key outcome at chunk_end_frame, future tense>",
      "A":       "<demo motion over [frame_idx, chunk_end_frame]>",
      "A_pred":  null  |  "<reasoning + intended action; kf0 leads with Plan>",
      "chunk_end_frame": <int>,
      "imitation_supervised": <bool>
    }
  ]
}
```

Rules (minimal):
- `keyframes` length = `get_keyframe_list` length, in order, matching `frame_idx`.
- `S`, `S_pred`, `A` on every keyframe; `A_pred` non-null iff `[think_act]`.
- `mode_marker`, `A_pred`, `imitation_supervised` follow the act/think gate above.
- `chunk_end_frame` satisfies `frame_idx < chunk_end_frame ≤ frame_idx + 60`.
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
     "why": "kfs 2-5 are one continuous approach to the cup; merging them keeps the A_pred coherent",
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
