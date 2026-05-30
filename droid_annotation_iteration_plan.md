# DROID CoT Annotation — Iteration Plan

Living doc tracking open design questions, pending experiments, and unit
tests for the LAP cascade-VLA pretraining pipeline.

Last updated: 2026-05-30

---

## Open Design Questions

### Q1. Action ↔ next-step coordinate convention
- v4.3.2 finding: **dz 98% agreement; dx 21%; dy 20%** between action
  text and `Δxyz` (robot base frame).
- Root cause: VLM uses **image/camera view** for direction words; pose
  deltas are in **robot base frame**. The two systems are not aligned.
- Action taken in v4.3.3 prompt: require explicit declaration when using
  camera view (e.g. "from the camera view, …").
- **NOT YET** changing the input format. Holding off on labelling Δxyz
  as `(forward=, left=, up=)` until we test v4.3.3 first.

### Q2. Keyframe density (tight cluster problem)
- 8% of keyframe gaps are <0.5s (cross-episode). ep34 alone has 7 tight
  clusters.
- Suspected primary cause: R3 retry (close→open→close in <1.5s) produces
  2-3 frame keyframe spacing **by design** (this is the semantics).
- Need to confirm with data: enumerate all <0.5s gaps across 50 eps and
  classify by (type_i, type_j) and which rule fired.

### Q3. Agent-feedback loop for keyframe selection (proposed)
- Concept: give the VLM `add(t, type)`, `remove(t)`, `view(t)` interfaces
  to revise the rule-based keyframe set.
- Pros: catches semantic events rules miss (object pose change,
  distractor entering scene); self-correcting.
- Risks:
  - 2× API cost (review pass)
  - Non-determinism: same episode → different keyframes across versions,
    making prompt iteration harder to A/B
  - Over-/under-correction
- **Phased proposal**:
  - Phase 1 (now): rule-based only
  - Phase 2: agent can ONLY `add` (gap-filler use case)
  - Phase 3: `remove` allowed but marked as "proposal", human review

### Q4. Bi-level architecture (deployment) — DEFERRED
- High-level (LAP / VLM): keyframe-rate (~1 Hz). Outputs `stage_t` +
  `action_t` (text) + `duration_t` (predicted frame span).
- Low-level (action expert): control-rate (15 Hz). Outputs continuous
  EE/joint deltas conditioned on `action_t`.
- Solves receding-horizon stall: low-level keeps emitting control at
  15 Hz regardless of high-level cadence; smooth transition when
  high-level fires earlier than chunk completion.
- **Decision 2026-05-30**: defer. Stay with single-pass pi05-style
  inference (one infer call → predicted chunk → execute open-loop
  prefix → re-infer). Avoids over-engineering before we measure if
  stall is actually a problem in our deploy setup.

See **Appendix A: Bi-level inference flow with pi05** below — kept as
reference for the time we revisit this.

---

## Implementation Tasks

### Sprint 1 — Verify convention fix (action ↔ delta)
- [x] v4.3.3 prompt update: require declaration when using camera view
- [ ] Re-run ep0/32/34 with v4.3.3 prompt
- [ ] Improve action↔delta checker:
  - [ ] Detect "from camera view" / "in camera frame" declarations
  - [ ] Under declared camera view, flip x/y signs before comparison
  - [ ] Report per-axis sign-flip rate + magnitude agreement
- [ ] Decision point: if v4.3.3 still under 50% on dx/dy → revisit
      labelling the input format `(forward=, left=, up=)`

### Sprint 2 — Keyframe rule audit
- [ ] Enumerate all keyframe pairs with gap <0.5s across all 50 eps
- [ ] For each tight pair, log:
  - Episode + keyframe indices
  - (type_i, type_j)
  - Which rule produced each (R2/R3/R4/R6)
  - Inter-frame distance (in frames + seconds)
- [ ] Classify clusters as semantic (retry) vs artifact (NMS miss,
      hysteresis flicker)
- [ ] Visualize: per-episode timeline plot color-coded by rule
- [ ] Report on artifact count + which rules need tightening

### Sprint 3 — Detector fixes (conditional on Sprint 2 findings)
Only do these if Sprint 2 identifies real artifacts (not just R3 retries):
- [ ] R3: enforce a minimum "open phase" duration in the retry pattern
      (e.g. open must persist ≥4 frames before the retry close)
- [ ] R4: revisit `MOTION_NMS_FRAMES=8` — may need axis-wise NMS
- [ ] R2: revisit hysteresis on tight oscillations
- [ ] Re-run keyframe audit after fixes

### Sprint 4 — Agent-feedback loop prototype (Phase 2 only)
- [ ] Implement `add_keyframe(t, type, reason)` interface
- [ ] Implement `view_frame(t)` returning external + wrist images
- [ ] Two-pass annotation pipeline: rule-based first, then VLM gap-fill
- [ ] Test on 5 episodes: rule-only vs rule+agent
- [ ] Compare: cost, runtime, added/removed counts, quality (human eval)

### Sprint 5 — Bi-level architecture
- [ ] Extend annotation schema: add `duration_t` field per keyframe
      (frames to next keyframe — derivable from existing data)
- [ ] Design action expert input/output spec
- [ ] Decide: train low-level from same keyframe dataset (sparse
      supervision) or from raw DROID trajectories (dense)
- [ ] Simulate receding-horizon deployment on 5 trajectories: measure
      stall rate / smoothness with vs without bi-level

---

## Unit Tests to Build

- [ ] `test_axis_convention_parser`
      - Input: "move left 2 cm and forward 3 cm" + declared "robot view"
      - Expect: (dx=+3, dy=+2, dz=0)
      - Same text + declared "camera view" → (dx=?, dy=?) marked
        view-dependent (downstream must consult camera extrinsics)

- [ ] `test_keyframe_retry_no_artifact`
      - Synthetic episode: close→open→close within 1.5s with 5-frame
        open phase
      - Expect: 3 keyframes (grasp1=failed, release=open, grasp2=retry)
      - All semantically meaningful, no NMS removal

- [ ] `test_decomposed_rotation_correctness`
      - Random rotation axis × angle
      - Expect: roll_deg + pitch_deg + yaw_deg projection matches signed
        axis components (within 1° for angles <30°)

- [ ] `test_gap_transition_window`
      - Synthetic episode: grasp at kf[4]
      - Expect: kf[4], kf[5], kf[6] → no gap line in pose_delta_str
      - kf[7] → gap-to-release present
      - Same for release transition window

- [ ] `test_camera_view_declaration_detected`
      - Action text containing "from camera view" / "in camera frame"
      - Expect: flag set, sign-flip logic enabled in checker

- [ ] `test_dual_delta_for_pre_interaction`
      - Synthetic episode with grasp at kf[10]
      - Expect: kf[0..9] each have `gap-to-grasp` + `next-step` lines
      - kf[10..12] have only single delta (transition window)

---

## Decisions Already Made

- Keep camera-view direction words but require explicit declaration
  (v4.3.3) — preserve VLM visual reasoning advantage
- Hold off on rewriting input pose-delta format until v4.3.3 results
  are in
- **Bi-level architecture is deferred** (2026-05-30). Stay with the
  current single-pass receding-horizon (pi05-style) for now; revisit
  only if deploy-side stall is a measured problem
- Keyframe regeneration policy: regenerate when rule changes; do NOT
  re-use across rule versions (acceptance criterion: explicit version
  string in meta)

## Terminology — what we mean by "chunk"

pi05 has two distinct chunk concepts; we use this vocabulary:

| Term                              | Value (DROID)    | What it is                                                                 |
|-----------------------------------|------------------|----------------------------------------------------------------------------|
| **Predicted action chunk**        | 50 (or 16 in DROID training config) | The model's one-shot output. `action_horizon` in `pi0_config.py`. The "action chunking" of ACT / Diffusion Policy literature |
| **Execution chunk** (open-loop)   | 16 (typical)     | How many actions of the predicted chunk are executed before re-inference. Configured at `ActionChunkBroker(action_horizon=…)` |

So "execute 16 of 50" = predict 50 ahead but only commit to the first
16 before re-reading observation. The remaining 34 are discarded each
cycle.

The Bi-level discussion in Appendix A is what you'd build IF the
predicted-chunk vs execution-chunk gap caused real deploy issues. For
now (2026-05-30) the gap is acceptable and we keep single-pass.

---

## Out of Scope (for now)

- Camera extrinsic / pose-delta reprojection — too much engineering for
  marginal gain when explicit declaration solves it
- Multi-camera fusion (wrist + external) for keyframe selection
- Cross-episode keyframe normalization (e.g. always exactly 12 per
  pick-place task)

---

## Appendix A: Bi-level inference flow with pi05

### A.1 What pi05 looks like today (single-pass)

Pi05 is in `policy/pi05/src/openpi/models_pytorch/pi0_pytorch.py`.
Architecture (`PI0Pytorch.__init__`):

- `PaliGemmaWithExpertModel` — **two transformers connected by shared
  cross-attention**:
  - PaliGemma (vision-language backbone) processes images + language
    prompt tokens
  - Action expert (smaller Gemma variant) processes state + noisy
    action tokens + diffusion timestep
- Default `action_horizon = 50` (in `pi0_config.py` line 26), i.e.
  one inference returns **50 future control steps**

The inference entry point is `sample_actions` (line 376):

```python
def sample_actions(self, device, observation, noise=None, num_steps=10):
    # 1. Encode prefix (images + language) ONCE, build KV cache
    prefix_embs = embed_prefix(images, lang_tokens)
    _, past_key_values = paligemma_with_expert.forward(prefix_embs, ...)

    # 2. Flow-matching denoise loop (10 Euler steps by default)
    x_t = noise                          # shape (B, 50, action_dim)
    time = 1.0
    while time >= -dt/2:
        v_t = denoise_step(state, past_key_values, x_t, time)
        x_t = x_t + dt * v_t             # rectified-flow update
        time += dt
    return x_t                           # 50 control steps
```

What `denoise_step` does (line 421): takes the cached prefix KV +
suffix (state, current noisy actions, timestep) → runs action expert
→ returns velocity field `v_t`. The expert sees the language ONCE
(via cached prefix); state and time are re-injected each step.

**Deploy loop (receding horizon, today)** roughly:

```
t=0:    obs_0 → infer() → [a_0, a_1, ..., a_49]   (50 steps ≈ 3.3s @15Hz)
        execute first 16 steps (open-loop chunk) → ~1.1s motion
t=1.1s: obs_1 → infer() → [a_0', ..., a_49']      (re-plan from fresh obs)
        execute first 16 steps
        ...
```

**Where stall comes in**: chunk length is fixed (e.g. 16). If between
two semantic events (e.g. mid-grasp) the demonstration takes only
3 frames, the model still plans 50 ahead and the chunk runs forward
even after the semantic event has changed. The robot doesn't *freeze*
because the chunk keeps emitting, but the next high-level decision
(re-plan with new obs) is delayed until chunk completion.

### A.2 What changes with LAP cascade bi-level

The split: **language understanding** moves up to LAP (refreshed at
keyframe rate); **action denoising** stays in the action expert
(every control tick).

```
┌─────────────────────────────────────────────────────────────────┐
│ HIGH-LEVEL (LAP)              fires at keyframe boundaries (~1Hz)│
│                                                                 │
│ Inputs:  o_t (image at keyframe), a_0..a_{k-1} (past text       │
│          actions), ℓ (task instruction)                         │
│ Outputs: stage_t (text), action_t (text), duration_t (frames    │
│          to next keyframe — for training the action expert's    │
│          attention window)                                      │
│ Compute: one autoregressive VLM decode (~200-500ms on edge)     │
└─────────────────────────────────────────────────────────────────┘
                            │ action_t = "Lower 5 cm and pitch 11°
                            │            to align jaws with marker barrel"
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ LOW-LEVEL (Action Expert, pi05-style)        every control tick │
│                                              (15Hz)             │
│                                                                 │
│ Inputs:  o_τ (current image), state_τ (proprio), action_t       │
│          (the latest text from LAP)                             │
│ Outputs: 50 EE/joint deltas via flow matching                   │
│ Compute: 10-step denoise loop on shared KV cache (~50ms)        │
└─────────────────────────────────────────────────────────────────┘
```

The action expert here is **almost exactly the existing pi05 expert**,
with one change: its language conditioning is `action_t` (the LAP's
text instruction for the current keyframe), not the global task ℓ.

### A.3 Concrete inference timeline

Suppose ep0 ("put marker in pot") with keyframes at frames
{0, 11, 27, 35, 59, 67, 75, 87, 97, 134, 145, 165} (gaps 0.73 / 1.07 /
0.53 / 1.6 / 0.53 / 0.53 / 0.8 / 0.67 / 2.47 / 0.73 / 1.33 s @15Hz).

```
HIGH-LEVEL (LAP)            LOW-LEVEL (Action Expert)
──────────────────────      ────────────────────────────
t=0.00s (kf 0)
  fire LAP:
    inputs: o_0, ℓ
    out: action_0 =
      "Head toward table
       center, descend
       slightly"
    duration_0 = 11
                            t=0.00s..0.73s (frames 0..10):
                            ┌─ every 1/15s, fire expert:
                            │    inputs: o_τ, state_τ, action_0
                            │    out: 50 deltas → execute first
                            │ (action_0 stays as language context
                            │  until next LAP fire)
                            └─

t=0.73s (kf 1)
  fire LAP:
    inputs: o_11,
            [action_0], ℓ
    out: action_1 =
      "Continue forward and
       lower further toward
       marker region"
    duration_1 = 16
                            t=0.73s..1.80s (frames 11..26):
                            ┌─ expert ticks 15Hz, now conditioned
                            │  on action_1 (smooth swap of language
                            │  context; KV cache for new language
                            │  is rebuilt once at the swap)
                            └─

t=1.80s (kf 2, pre_grasp)
  fire LAP:
    inputs: o_27,
            [action_0, action_1], ℓ
    out: action_2 =
      "Lower 5 cm and pitch
       3° toward marker"
    duration_2 = 8
                            t=1.80s..2.33s (frames 27..34):
                            ┌─ expert refines approach with new
                            │  action_2 as fine-grained instruction
                            └─

t=2.33s (kf 3, pre_grasp)
  ...
```

**Key properties**:

1. **Decoupled rates**. LAP fires only at keyframe boundaries
   (irregular: 0.07s to 2.93s in our data); action expert fires
   regularly at 15Hz regardless.

2. **No stall on tight clusters**. Even if the next keyframe is
   only 2 frames away (e.g. retry pattern), the action expert keeps
   producing control. The LAP's swap of `action_t` is just a change
   in the expert's language conditioning — the expert still emits
   smooth deltas.

3. **Receding-horizon at the expert level**. The expert still
   outputs 50-step chunks but **only the first 1-3 chunks are used**
   before a new image arrives and triggers re-denoising. The chunk
   length stops being a deployment constraint; it's just a training
   batch size now.

4. **The 50-step horizon is overkill at deploy**. We can shorten
   `action_horizon` for inference (e.g. predict 8 steps instead of 50)
   without retraining — flow matching is horizon-flexible if state
   conditioning is consistent.

### A.4 Training the action expert under bi-level

Today pi05 trains: `image + state + ℓ → 50 action deltas`. We'd retrain
to: `image + state + action_t → N action deltas` where `N = duration_t`
(the keyframe-to-keyframe gap).

Pros:
- Action_t is more specific than ℓ (so easier supervision)
- The expert learns to map text instructions to short concrete motion
  windows, which is exactly what deploy needs

Cons:
- Need annotated `action_t` + `duration_t` for every frame, not just
  every keyframe — interpolation policy TBD (forward-fill latest LAP
  output is the simplest)
- The expert must remain robust when `action_t` swaps mid-chunk during
  deploy. Mitigation: during training, randomly perturb the language
  conditioning length to simulate swaps.

### A.5 What to build next (in priority order)

1. **Extend annotation schema** with `duration_t` (already derivable
   from keyframe indices)
2. **Frame-level expansion** of LAP outputs: for every control-rate
   frame, assign the latest active `action_t` and `stage_t`
3. **Wrap pi05's action expert** to accept `action_t` as the prompt
   instead of the global task ℓ. Minimal code change — `embed_prefix`
   already takes tokenized prompt
4. **Train LAP separately**: text-only autoregression on
   `(o_t, a_0..a_{k-1}, ℓ) → (stage_t, action_t, duration_t)`
5. **Simulated deploy** on 5 trajectories: measure smoothness
   (jerk, deviation from demo) with vs without bi-level cadence

### A.6 Open design questions for bi-level

- **Joint vs separate training**: train LAP and expert jointly (more
  expressive, harder optimization) or separately (modular, may have
  distribution mismatch at the language interface)?
- **Action_t representation**: free-form text vs constrained vocab?
  Free-form is more natural but expert may overfit to specific phrasing
- **When LAP fails to fire on time**: if LAP inference is slower than
  the keyframe gap, action expert keeps using stale `action_t` — at
  what staleness does this become harmful?
- **Wrist camera**: today's pi05 uses external + wrist images. LAP
  annotation uses external only. Should LAP also see wrist for
  finer keyframes near interaction?

### A.7 Honest answers to two sharp questions

**Q1 (raised 2026-05-30): Does bi-level training-deploy alignment hold
when action_t is stale, e.g. LAP said "lower 10 cm" at kf_k but at
frame τ > kf_k the gripper has already lowered 5 cm?**

The expert is state-conditioned (sees `state_τ` and image `o_τ`), so
the same `action_t="lower 10 cm"` in different states is interpreted
differently — at τ=kf_k+5 with state showing "5 cm remaining", the
expert is trained to output the remaining 5 cm of motion, not another
full 10 cm. This is standard imitation-learning conditioning.

Real risks (not solved by architecture, need engineering):
1. **State drift at deploy** — accumulated errors mean at deploy the
   gripper may be at a position the demo never visited under this
   `action_t`. Mitigation: DAGGER-style state-noise augmentation during
   training.
2. **Stale action_t during LAP latency** — if LAP takes 200 ms and the
   keyframe gap is 100 ms, expert keeps using the old `action_t` past
   its intended duration. Mitigation: train with random "action_t
   freeze" perturbations so the expert learns to gracefully run out the
   instruction without going wild.

→ **Architecturally fine; needs careful training data augmentation.**

**Q2 (raised 2026-05-30): If LAP must fire at every keyframe (including
the tight 3-frame post-grasp cluster), the bi-level architecture has
just transferred the stall problem from "action chunk padding" to "LAP
inference latency". Did we actually solve the deploy stall?**

**Partial yes.** Here's what bi-level solves and what it does not:

| Sub-problem                                | Bi-level fixes? |
|--------------------------------------------|-----------------|
| Action chunk padding (47 of 50 wasted)     | ✅ yes — expert re-decodes at 15 Hz from fresh obs |
| Whole system blocks on slow LAP            | ✅ yes — LAP runs async to control |
| Tight clusters need LAP to fire within 3 frames | ❌ no — just shifts stall up |

The fix to (3) is **not architectural** — it's the design choice of
**when LAP fires**. Currently `keyframe.py` produces many keyframes
including dense clusters around grasp/release. If LAP must fire at
every one of them, tight clusters break LAP latency budget.

**Resolution: separate "annotation keyframes" from "LAP fire timing".**

| Concept                       | Used for                                                                 | Count    |
|-------------------------------|--------------------------------------------------------------------------|----------|
| **Annotation keyframes** (current `keyframe.py` output) | training LAP — every one is a VQA / next-action supervision signal       | many     |
| **LAP fire moments** (subset / coarser)                  | when LAP actually re-plans at deploy                                       | few — only at semantic phase boundaries (approach / grasp+post_grasp window / lift / transport / pre-release / release+post-release window / retract) |

Under this split:
- One `action_t` from LAP covers the **entire grasp + post_grasp window**
  (≈3 frames + 3 frames). The expert keeps the same `action_t`
  ("close the gripper to grasp the marker") for the whole event;
  state-conditioned interpretation handles the fine motion
- LAP only fires again when entering the next semantic phase (lift)
- This is exactly the standard "hierarchical action" pattern in
  hierarchical RL / option-critic / RT-2 chain-of-thought literature

This implies:
1. **Training**: annotate dense keyframes (current setup is fine), but
   group sequences of keyframes into "semantic phases" via a separate
   light-weight rule (or LLM clustering). LAP only sees phase-boundary
   transitions; expert sees frame-rate state.
2. **Deploy**: LAP fires at phase boundaries (estimated ~5–10 times per
   typical episode, much less than 12–50 keyframes). Plenty of headroom
   for 200 ms LAP inference.
3. **Annotation data still useful**: even non-LAP-fire keyframes give
   us VQA supervision (`stage_t` predictions) at the dense rate, which
   is good for training the joint perception backbone.

**TL;DR**: Q2's concern is valid for the "naive bi-level". Real fix
is to make LAP's firing schedule **semantic-phase-aware**, not
keyframe-aware. Tight clusters then become a within-LAP-event detail,
not a re-firing problem.

