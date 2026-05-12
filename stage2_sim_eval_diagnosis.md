# Stage 2 Sim Eval — Failure Diagnosis & Mitigation Roadmap

_Last updated: 2026-05-12 (session that ran the cascade-pipeline + qpos-aligned
LAP RoboTwin Stage 2 ckpt at step 30000)._

## TL;DR

`lap_robotwin_finetune_qpos` run `lap_robotwin_run_qpos1` step 30000 fits the
training distribution extremely well — `dry-run` on episode-0 frames yields
mean MSE = **0.00114**, mean L1 = **0.0180**, no mode collapse (pred_std/gt_std
∈ [0.96, 1.27] over all 14 dims). The model is **not under-trained**.

But sim eval over the snapshotted `arrange_blocks_l_shape` 10-scene set is
**0/N** in every configuration tried:

| Setting | Result |
|---|---|
| run0 (endpose state) — flow-only — `exec_horizon=8` | 0/10 |
| `lap_robotwin_run_qpos1` (qpos state aligned) — flow-only — `exec_horizon=8` | 0/3 (truncated) |
| same — cascade pipeline (`policy.type=cascade`) — `exec_horizon=8` | 0/3 (truncated) |
| same — cascade pipeline — `exec_horizon=1` | 0/1 (truncated) |

The arm makes coarse motion toward the scene but never closes the gripper on a
cube. Cascade text is correctly generated (`[plan]` adapts per episode), and
the deploy_policy.py `actions[0]` no-op bug (executing `chunk[0]` = current
state) is now fixed (commits to `chunk[1:]`). Even after the fix, behavior is
the same.

## Root cause

**Imitation-learning covariate shift / compounding error.**

- Training data is `demo_clean` — TOPP-smoothed expert trajectories with no
  control noise. State at time `t` is exactly what produced the action chunk
  at time `t`.
- At eval, sim's TOPP planning + joint servo introduce small per-step error
  (~0.01–0.05 rad). Over 100s of steps these accumulate to several radians.
- Once the joint state drifts off the expert manifold, the model has never
  seen this configuration, so its predictions degrade to "looks like some
  phase from training" — the arm hovers near plausible grasp poses but never
  closes on the actual cube positions.
- This is consistent with what the rollout videos show across all 4 eval
  runs: a single coarse approach motion, then aimless hover near the table.

The model is **excellent at predicting the next 8 frames** *from* a state
that's exactly on the expert trajectory (this is what dry-run tests; every
frame is reset to the GT state before sampling). It cannot recover *from
its own drift*.

## Mitigation directions

In rough order of compute cost. Plan: try the cheap one first, escalate
only if needed.

### D. Action smoothing (inference-time, no retrain) — TRIED, INSUFFICIENT

EMA / low-pass filter on the executed action stream. Goal: reduce the
high-frequency jumps at chunk boundaries that compound into trajectory
drift.

- Implementation: `smoothed_t = α · raw_t + (1−α) · smoothed_{t−1}`,
  reset on episode start. `action_smooth_alpha` knob in
  `policy/lap/deploy_policy.yml`.
- Cost: a few lines in `lap_model.py` + `deploy_policy.py`. No retrain.
- **Status (2026-05-12)**: tested at α=0.5 → **0/3 step_limit, same as
  unsmoothed**. Videos show identical "hover-near-block-but-never-grasp"
  pattern. The smoothing changes intra-chunk trajectory shape slightly
  but does not pull the policy back onto the training manifold once it
  has drifted off — confirming the diagnosis that the failure is from
  model mis-prediction on OOD states, not from high-frequency
  controller noise.
- **Verdict**: D alone is insufficient. The fix has to widen what the
  *training distribution* covers, not what the inference filter smooths.
- Side observation worth keeping: with `chunk[1:]` (no-op skip), each
  inference produces a chunk of 7 motion waypoints, so `exec_horizon` is
  effectively capped at 7. The overlay shows `chunk=N/7` correctly.

#### Side note: cascade plan drift across inferences

While inspecting rollout videos under cascade pipeline, observed that the
generated `[plan]` text can change between inferences within one episode
(e.g. "Place the blue block at the corner" at step 150 → "Place the
purple block at the corner" at step 300, same scene). This is consistent
with how cascade was trained (cascade emission is only sampled on ~20%
of frames at phase boundaries; mid-phase had no cascade target so the
model has no strong anchor) and is exacerbated by image observations
changing as the arm moves. Not a bug, but a contributor to behavioral
inconsistency. If we want stable plan conditioning at inference we'd
need to cache the plan per episode (or per phase, if we can detect
boundaries) instead of regenerating every call.

### A. State-noise data augmentation (cheapest retrain)

Add small Gaussian noise to `state` during training so the policy is forced
to learn `f(state + ε) ≈ action`. This widens the training distribution to
cover near-trajectory states the eval will actually visit.

- Implementation: 5 lines in `RoboTwinTaskDataset._build_sample` —
  `state = state + np.random.normal(0, σ, state.shape).astype(state.dtype)`.
  Recommend `σ ≈ 0.02 rad` (~1°), apply only to qpos state, leave actions
  untouched.
- Cost: one retrain run (~6h on 5×H200 with the current `qpos` config).
- Expected lift: this is the canonical "robustness for IL" trick — Pi0 / RT-2
  papers see meaningful gains.
- **Status**: deferred — try D first; revisit if D < 1/10 success.

### B. DAgger / on-policy state collection

Roll out the current ckpt in sim, record the states it visits, get expert
labels at those states (in RoboTwin we can call the expert plan_path
function), mix into training data.

- Implementation: nontrivial. Need a script that:
  1. Resets sim, rolls policy with stochastic noise.
  2. At each visited state, calls `ArrangeBlocksDataGen` expert to get the
     "correct" action.
  3. Appends `(state, image, expert_action)` tuples to a new shard.
  4. Mix shard into training at e.g. 30% weight; retrain.
- Cost: ~1 day of infra + multiple retrain iterations.
- Expected lift: large (this is exactly what DAgger is designed for).

### C. Include explicit failure-recovery trajectories

Record demos where the operator deliberately starts the arm in a slightly
off-trajectory pose and recovers. Add to training set.

- Cost: human-in-the-loop data collection. Probably the largest cost of
  any direction here.
- Expected lift: same intuition as A/B but with hand-curated data.

### E. Train longer / on wider task mix

Hypothesis: at 30k steps the model has memorized the training trajectories
but hasn't yet learned the *underlying skill* well enough to generalize.
Train to 100k+ steps and/or add more task variants (pick_place, stack_blocks,
arrange_fortress) so the action expert sees more diverse states.

- Cost: ≥3× current 6h run.
- Risk: pure scale rarely fixes covariate shift — the model can overfit
  harder instead.

### F. Diffusion-policy-style temporal ensembling

Pi0 / Pi0.5 / DiffusionPolicy all use *receding-window prediction with
temporal ensembling*: predict a longer chunk (e.g. 16 actions), execute only
a few, but at each step blend with previously-predicted actions for the same
absolute timestep. Equivalent to a many-step EMA over the action sequence.

- Implementation: more substantial than D but still inference-only — track
  per-timestep predictions, average across overlapping chunks.
- Cost: medium (no retrain, but real code work).
- Expected lift: notably better than naive EMA D because it averages
  predictions from *different observations* — a true ensemble.

If D's simple EMA helps, F is the principled extension.

## Things that ARE fine (verified)

- **Cascade pipeline**: AR generates coherent task plans per episode
  (verified via probe + video overlay).
- **qpos state alignment**: state and actions live in the same joint-space.
  Endpose state was indeed mismatched (run0) but switching to qpos didn't
  unblock sim eval — covariate shift dominates.
- **Action quality on training distribution**: dry-run MSE 0.001.
- **Cascade marker emission**: model emits `[plan]/[stage]/[action]` markers
  in correct positions (verified server-side probe).
- **No mode collapse**: pred_std ≈ gt_std on all 14 dims.
- **Video overlay infra**: `infer@step`, `chunk=N/M`, `wrist`, full
  reasoning_text all render on rollout mp4s.
