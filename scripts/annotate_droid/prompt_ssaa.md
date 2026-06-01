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
- Focus on the single most important change the action will produce
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

- Imperative phrasing, numbers directly from next-step Δrobot and/or Δee
- Specify the frame ("robot frame" / "wrist frame") when you use
  directional words
- No reasoning, no goal language — A is what *happened*, mechanically
- For genuinely free transport where exact numbers don't add information
  (transient mid-flight samples), qualitative direction is fine — but
  bias toward including the demo's numbers when they're available,
  because the world model needs them

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

**Failure episodes**: A_pred should describe what a competent agent
would intend, even if the demo deviates from it. The point of
labeling A_pred on failure data is to distill the annotator's broader
knowledge (recovery, alternatives) into the policy — even when the
demo itself failed. If the goal becomes unrecoverable, note it in
the plan and write A_pred for the most reasonable continuation
anyway.

## Output

Strictly valid JSON, no markdown fence, no commentary:

```json
{
  "plan": "<2-5 sentences. Episode-level arc. If demo failed, say so.>",
  "keyframes": [
    {
      "frame_idx": <int>,
      "mode_marker": "[think_act]",
      "S":       "<current scene>",
      "S_pred":  "<predicted key outcome + brief why>",
      "A":       "<demo telemetry>",
      "A_pred":  "<agent reasoning + action>"
    }
  ]
}
```

Rules (minimal):
- keyframes length = input keyframe count, in order, same frame_idx
- mode_marker = "[think_act]" on every keyframe
- JSON only, no commentary
