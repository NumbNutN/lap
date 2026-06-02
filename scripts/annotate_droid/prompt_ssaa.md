# SSAA prompt — chain-of-thought labels for DROID annotation

You are producing chain-of-thought + action labels for a single-arm
Franka Panda manipulation episode. Your labels train two joint
objectives — a **world model** and a **policy**. Each label has a
specific role in the training loss; you'll write better labels if
you understand which role yours plays.

## Roles of the four fields

```
S      — current scene                    (prefix-only, no loss)
S_pred — predicted next-state outcome     (CE target, world model)
A      — what the demo physically did     (prefix-only, no loss)
A_pred — what an agent SHOULD do          (CE target, policy)
```

Two of these are *anchors* you describe from evidence (S, A).
Two are *predictions* the model is trained to produce (S_pred, A_pred).
The styles, levels of abstraction, and information density differ
accordingly — there is no single "right voice" across all four.

## Inputs you receive

- Task instruction (natural language goal ℓ).
- Per keyframe:
  - frame_idx, type (begin/grasp/release/retry/motion/end),
    gripper_state (open/partial/closed)
  - external camera image
  - wrist camera image
  - pose deltas in two frames:
    - **Δrobot** = motion in robot base frame (forward/left/up;
      world-fixed, control-frame truth)
    - **Δee** = same motion in wrist camera frame (calibrated;
      forward = optical axis, left/up = image axes; what a camera
      looking forward from the gripper "sees")
    - when near a grasp/release, also **gap-to-grasp** / **gap-to-release**
      (vector from here to the upcoming interaction pose)
- All earlier keyframes' annotations as in-context memory.

## S — current scene

**Purpose**: anchor the world model in what's actually present right now.
**Loss**: none. Used only as prefix to predict S_pred.

A faithful present-tense description of the scene relevant to the
dynamics — where the gripper is, what's around it, what's in hand,
what just happened that produced this state. Use external view for
layout, wrist view for gripper-proximal detail.

Describe what would matter to a manipulator. Don't catalog every
distractor or fixture.

**Causal anchor** (labels are non-Markovian): when the current scene
reflects effects of past actions, reference them — e.g., "since the
gripper just knocked the cup over, tokens are scattered to the right
of the bowl."

**Selective view**: you have two camera images (ext + wrist) but you
don't have to mention both every keyframe. Only describe a view when
it contributes visual info relevant to the next step. If the wrist
view is just bare table (gripper far from contact) or the external
view is just the back of the robot (uninformative for the current
sub-goal), skip it. Same principle applies to **A** when phrasing the
motion in terms of visual landmarks.

## S_pred — predicted key outcome after the next action

**Purpose**: world-model target. Forecast the most important state
change A will produce.
**Loss**: CE on this text.

This is **not** another scene description — it's a forecast.

A useful mental model: imagine a human supervisor watching the demo
mid-motion. They aren't tracking every object in the scene — their
attention narrows to the thing that's about to change. That's the
voice of S_pred.

- Future tense, usually ≤ 1 sentence
- **Object-centric**: describe object relations/states (interaction,
  approach, alignment), not the arm's own motion. The arm moving by
  itself is uninteresting; only its relation to objects matters.
  When no object relation changes, a minimal arm-motion line is fine
  ("the arm will lift and sweep right").
- Don't re-describe context already in S
- **You may briefly include the reasoning behind which change is "key"**
  — what makes you confident this is what matters. e.g. "Because the
  upcoming grasp needs the rim aligned, the key change is …"

Examples of the right narrowness:
- ✓ "The jaws will be ~0.5 cm above the cup rim, oriented for closure
  — this is the alignment the upcoming grasp depends on."
- ✓ "The marker will have left the table surface, gripped firmly."
- ✗ "The gripper will move forward into the workspace area where the
  cup is positioned, the marker will be approached..." (re-narrating)

## A — what the human demo physically did

**Purpose**: factual telemetry. The world model needs A to be exactly
the motion that produces the next state. Also becomes the past-action
prefix when conditioning later keyframes.
**Loss**: none. Prefix-only.

- Imperative phrasing, numbers directly from next-step Δrobot or Δee
- **Single-frame economy**: pick *one* frame, not both. Default to
  robot-base for transport / free motion (wrist landmarks are weak
  when far from contact); switch to wrist-frame for fine alignment
  and contact-rich phases where landmarks are dense.
- **Affordance call-out**: when motion features alignment, obstacle
  avoidance, or contact geometry (e.g., approaching the rim from
  above to clear the lip), say so — these are not "reasoning", they
  are facts about what the motion is doing.
- **Demo intent on retry/recovery**: when the keyframe is a retry,
  re-grasp, or correction, explain the demo's intent ("re-aligning
  after the first grasp slipped"), not just the cm/degree values.
  For ordinary motions where nothing's remarkable, plain telemetry
  is fine.
- For genuinely free transport where exact numbers don't add
  information (mid-flight samples), qualitative direction is fine —
  but bias toward including demo numbers when available.

## A_pred — what an agent SHOULD do, given the observation

**Purpose**: policy target. Train the model to choose actions from
observation, **not** to clone the demo.
**Loss**: CE on this text.

This is the only field where you write as if you were the agent
itself, reasoning from o and ℓ. The demo information is your check,
not your answer.

**Articulate your reasoning briefly** before / alongside the action —
the policy head benefits from supervised reasoning chains. e.g.
"The cup is 5 cm ahead and the jaws need to clear its rim, so I will
raise the wrist, align over the rim, then descend." A bare imperative
without justification is weaker training signal than one with the
"why".

The pose deltas you see are **reference values from a successful
expert**. Use them when precision is genuinely the right level of
abstraction for the decision you're describing:

- Aiming the gripper's attitude (yaw/pitch/roll) — precision often
  matters even from far away, like "raising a gun and lining up the
  sight". You may want exact degrees here.
- Final-cm alignment before contact.
- Any moment when "approach the cup" is not specific enough to
  explain what the next motion contributes.

Use qualitative language when:
- The motion is mostly about choosing a region/direction, and
  centimeter precision would be inventing precision the obs doesn't
  support
- You'd be unable to defend the specific number from current obs + ℓ
  alone

**Practical rule of thumb**: ask yourself "could I justify this number
to someone who only saw what I see right now?" If yes, use it. If no,
go qualitative.

**About knowing future keyframes**: you can see them. Don't let that
turn A_pred into "describe what the demo did" — that's A's job. Treat
the demo as a sanity check that your reasoning produces a sensible
trajectory, not as the source of the answer.

**Failure stance** — split by whether the failure has already occurred:

- *Pre-failure* (current keyframe is before the visible failure event):
  pretend **you don't know** the demo will fail. Reason as the competent
  agent would from the current obs. If the demo here is the action
  that causes the failure (e.g., sweeping into the cup), your A_pred
  should diverge — propose the action that *avoids* the failure cause.

- *Post-failure* (failure already happened, scene reflects damage):
  enter recovery mode. Acknowledge the changed state ("the cup is
  tipped on its side, tokens are scattered") and propose a sensible
  next step ("retract upward, then re-approach the tipped cup from
  the side to right it"). This distills broader knowledge into the
  policy even when the demo cannot recover.

**First-frame plan**: at kf[0], A_pred must additionally lead with a
*task-completion plan* — the multi-step arc the agent intends to
execute across the whole episode, from the agent's POV. Format:
"Plan: 1) ... 2) ... 3) ..." then the immediate kf[0] action. The
plan always describes how to *complete* the task; for failure
episodes, replace the demo's failing steps with non-failing
alternatives so the plan still leads to success. (The plan is
agent-side intent; the episode-level `description` field instead
narrates what actually happened in the demo, including failures.)

## Action supervision marking

Every keyframe carries two fields that tell the training pipeline
whether and how far this A_pred maps onto the demo's control data:

- `imitation_supervised: bool`
  - **true** when A_pred is essentially the agent doing what the demo
    did. The action regression head will learn from the demo's
    control segment.
  - **false** when A_pred diverges from the demo (pre-failure
    intervention OR post-failure recovery). Action head skips this
    keyframe; only the language CE loss on the A_pred text trains.
  - Monotone rule: once you set false for a keyframe, every
    subsequent keyframe in the episode must also be false.
    Divergence is a one-way transition.

- `chunk_end_frame: int | null`
  - Frame index where this keyframe's commanded motion ends in the
    demo timeline. Must be > current frame_idx and ≤ current +60.
  - Set to `null` (or omit) when `imitation_supervised=false`
    (no supervision applies).
  - **Default heuristic** (you'll see this pre-filled in the meta as
    `chunk_end_default`): the next keyframe's frame_idx. Adopt the
    default unless your A_pred's semantic span suggests otherwise.
  - **Extend** when A_pred describes a phase that spans multiple
    keyframes (typical for "approach", "transport"). The extended
    value should not pass the next imitation boundary or 60-frame cap.
  - **Shorten** when A_pred completes mid-segment (e.g. "close the
    gripper" finishes within a few frames; the rest of the segment
    is its own thing).
  - **Failure-edge truncation**: when the next keyframe is the start
    of the failure (next kf has `imitation_supervised=false`),
    aggressively shorten chunk_end_frame to before the failing
    motion begins.

## Episode-level `description` field

An episode-level free-text field (no loss; sits outside the keyframe
array). Narrates the full episode arc from the human's POV — what
actually happened in the demo, integrating any human hint. Includes
failure modes when present ("the gripper swept laterally into the
cup at kf10, knocking it over"). Distinct from the agent's plan
(which lives in kf[0]'s A_pred and describes how to *succeed*).

## Output

Strictly valid JSON, no markdown fence, no commentary:

```json
{
  "description": "<episode-level narrative; what the demo did>",
  "keyframes": [
    {
      "frame_idx": <int>,
      "mode_marker": "[think_act]",
      "S":       "<current scene>",
      "S_pred":  "<predicted key outcome + brief why>",
      "A":       "<demo telemetry>",
      "A_pred":  "<at kf[0]: plan + first action. Otherwise: agent reasoning + action.>",
      "imitation_supervised": <bool>,
      "chunk_end_frame": <int | null>
    }
  ]
}
```

Rules (minimal):
- keyframes length = input keyframe count, in order, same frame_idx
- mode_marker = "[think_act]" on every keyframe
- JSON only, no commentary
