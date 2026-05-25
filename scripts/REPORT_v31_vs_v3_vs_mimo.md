# Pilot Report v3.1 (R6 fix + forward delta + v3 fewshot) — vs v3 vs MiMo

> 2026-05-25 — addresses three user questions raised on the v3 pilot:
> 1. *Verify R6 fix (grip verb priority on grasp/release/retry).*
> 2. *Should pose_delta be forward-looking (current → next)? Implemented forward.*
> 3. *Why does style vary across episodes? Compare with MiMo on same v3 prompt.*

## Changes in this iteration (v3 → v3.1)

| Change | File | Why |
|---|---|---|
| **R6 split into TIER A / TIER B** | `prompts.py` `SYSTEM_PROMPT_V3_MEMORY` | grasp/release/retry now ALWAYS grip verb first; motion/begin/end/filler defaults to axis-aware vocab + magnitude |
| **Forward pose delta** (current → next, not previous → current) | `runner.py` | `action` describes next-step intent, so the visible Δxyz/Δrot should describe the upcoming segment, not the past one |
| **R9 new** — explains forward-delta semantics in prompt | `prompts.py` | the model must understand the time arrow |
| **v3 fewshot (text-only)** showing both small-delta and large-delta axis-aware actions | `prompts.py` `FEWSHOT_V3_*` | the v3 prompt skipped fewshot before; ep1/ep4 in v3 defaulted to "Adjust position" on small deltas because they had no example of how to handle small deltas axis-aware |
| **A9 audit warning** — flags episodes with < 30% axis-aware motion keyframes | `audit.py` | catches the style inconsistency the user observed (ep2: 76% vs ep1: 8%) |
| **MiMo client hardening** — falls back to `reasoning_content`, raises explicit error with finish_reason on empty content | `client_mimo.py` | previously empty MiMo response silently became "VLM failed after retries: " |

## Headline metrics — 3-way (5 episodes each, same DROID_100 subset)

| Metric | v3 Qwen (backward Δ, no fewshot) | **v3.1 Qwen** | v3 MiMo (same v3.1 prompt) |
|--------|--------------------|----|---------------------|
| pass rate | 3/5 | **4/5** | 2/5 |
| keyframes total | 69 | 69 | 48 (3 eps failed parse) |
| **action axis-aware** | 43% | **72%** | 77% |
| stage with history reference | **26%** | 16% ↓ | 15% |
| `think` non-null | 0% | 0% | 2% |
| **grip verb correct at grasp/release** | 80% | **100%** | 100% |

### Per-episode axis-aware breakdown — v3.1 spread shrank substantially

| episode | v3 axis% | **v3.1 axis%** |
|---|---|---|
| green sachet | 31% | 77% |
| marker in pot | 8% | **75%** ⭐ |
| candy bar | 76% | 82% |
| doritos in sink | 79% | 79% |
| pen out of bowl | 8% | 46% |

v3 spread: 8-79% (huge inconsistency). v3.1 spread: 46-82% (tight). **A9 audit no longer triggers on any v3.1 ep.**

---

## Concrete improvement: marker-in-pot ep (v3 → v3.1)

| # | type | v3 action (backward Δ, no fewshot) | **v3.1 action** (forward Δ + R6 + v3 fewshot) |
|---|------|--------------|-----------------|
| 0 | begin | Position the gripper above the marker. | **Translate forward 1.3 cm and lower 3.6 cm towards the marker.** |
| 1 | motion | Move slightly forward and downward towards the marker. | **Translate forward 15.3 cm and lower 20.7 cm towards the marker.** |
| 2 | motion | Adjust position further to align with the marker. | **Translate forward 3.9 cm and lower 5.3 cm to fine-tune position.** |
| 3 | motion | Fine-tune position for optimal marker grasp. | **Lower 10.7 cm and tilt downward 11° to align with the marker.** |
| 4 | grasp | Close the gripper to grasp the marker firmly. | Close the gripper to grasp the marker. ✓ (grip verb only — R6 TIER A) |
| 5 | motion | Lift the marker slightly off the surface. | **Translate forward 0.6 cm and lift 0.7 cm to adjust the grip.** |
| 6 | motion | Yaw counterclockwise slightly to align with the cube. | **Lift 11.2 cm and yaw counterclockwise 12° to aim at the pot.** |
| 7 | motion | Move towards the pot while adjusting orientation. | **Translate leftward 5 cm and lower 0.6 cm towards the pot.** |
| 8 | motion | Lower the marker towards the pot opening. | **Translate leftward 6.6 cm and lower 4.5 cm to reach the pot.** |
| 9 | release | Release the marker into the pot by opening the gripper. | Open the gripper to release the marker into the pot. ✓ (grip verb — R6 TIER A) |
| 10 | motion | Make slight adjustments post-release. | **Translate backward 2.6 cm and lift 5.3 cm to retract.** |
| 11 | end | Retract back to the initial position. | Remain stationary at the final pose. |

**Every motion keyframe in v3.1 has a concrete axis + magnitude.** The forward-delta + R6 + fewshot combo turned generic "adjust" into faithful captions of the upcoming robot motion.

---

## Cross-model: v3.1 Qwen vs MiMo (same prompt, same data, doritos ep)

| # | type | v3.1 Qwen | v3 MiMo |
|---|------|-----------|---------|
| 0 | begin | Translate forward 0.2 cm and lower 1.5 cm toward the Doritos. | Lower 1.5 cm while translating forward toward the counter. |
| 1 | motion | Translate leftward 0.8 cm and lower 5.7 cm toward the Doritos. | Lower 6 cm while translating forward to close distance to the Doritos. |
| 2 | motion | Translate forward 11.7 cm and lower 20.8 cm towards the target. | Translate forward 12 cm and lower 21 cm to reach the Doritos. |
| 4 | grasp | Close the gripper to grasp the Doritos. | Close the gripper to grasp the Doritos pack. |
| 6 | motion | Translate leftward 2.2 cm and lower 11.0 cm towards the sink. | Translate sideways 22 cm and lower 11 cm toward the sink. |
| 8 | release | Open the gripper to release the Doritos into the sink. | Open the gripper to release the Doritos into the sink. |
| 10 | motion | Translate leftward 1.4 cm and lift 7.7 cm away from the sink. | Lift 8 cm and tilt backward 18° to clear the sink. |

Both models pick up axis-aware vocab cleanly. **MiMo occasionally adds extra rotation info** (`tilt backward 18°` on kf[10]) that Qwen doesn't notice, suggesting MiMo's pose-data reading is slightly more attentive. But MiMo's overall pass rate is lower (2/5) due to JSON parsing failures unrelated to content quality.

---

## MiMo failure modes (3/5 fails)

| ep | failure | detail |
|---|---|---|
| marker | `parse failed: no '{' found in VLM output` | MiMo wrapped reply in markdown fence + free-text preamble that our parser couldn't unwrap |
| green sachet | `parse failed: unbalanced braces in VLM output` | MiMo produced truncated JSON (max_completion_tokens hit) |
| pen out of bowl | A8 audit fail (release with "Roll 19° and lower 1 cm to position the sharpie") | grip verb missing on release — same R6 TIER A issue, but MiMo didn't follow the rule |

**Fixes for next iteration**:
1. Strip markdown fence + leading prose more aggressively in `parse_vlm_output` (some MiMo replies need both regex strips before `{` lookup).
2. Bump MiMo `max_completion_tokens` from 2048 → 4096 (DROID episodes with 15+ keyframes can exceed 2k JSON tokens).
3. MiMo's R6 TIER A compliance is weaker than Qwen — add an "EXAMPLE: never write 'roll' as a release action" sentence to the prompt.

---

## Trade-off observed: memory ↓ when axis ↑

v3 stages had 26% history-referencing phrases ("Following the initial search...", "Post-release...").
v3.1 stages have 16%.

The v3 fewshot we added is text-only and demonstrates **terse** stages (focus on axis-aware action). The model picked up the terseness and dropped some memory phrases:

| | v3 stage example | v3.1 stage example |
|---|---|---|
| early motion | "Continuing the search, the robot makes further adjustments to its position based on visual feedback." | "Continuing approach. The robot moves significantly closer to the marker, adjusting its position." |
| post-grasp | "With the sachet partially grasped, the robot prepares to lift it off the table." | "Lifting the marker. The robot adjusts its position slightly after securing the marker." |
| post-release | "Post-release, the robot makes minor adjustments to confirm the sachet is properly placed." | "Retracting. The robot moves away from the pot after releasing the marker." |

v3.1 stages still carry **causal references** (continuing, lifting after securing, after releasing) but less explicit. Worth dialing back in next iteration — the v3 fewshot stages can be padded with explicit "Following X..." phrasing to preserve memory while keeping axis-aware actions.

---

## Recommendations

### Immediate (do now)
1. ✅ R6 fix shipped — grip 100% verb compliance achieved.
2. ✅ Forward delta shipped — actions now predict next motion correctly.
3. ✅ v3 fewshot shipped — axis-aware up 43→72%.
4. ✅ A9 audit shipped — catches style outliers.

### Next iteration (v3.2)
1. **Enrich v3 fewshot stages** with explicit memory phrases ("Continuing the approach after the initial search...") so memory% recovers to ~25%+ without losing axis%.
2. **Strip markdown fence + leading prose** in `parse_vlm_output` for MiMo robustness.
3. **Run 50 ep** on each (Qwen + MiMo) to detect rare retry cases for think coverage measurement.

### Pre-decisions for downstream (training side)
- v3.1 output schema is **stable**; downstream consumers (LAP cascade training dataset class) can wire to it now.
- `mode_marker = "[think_act]"` on every keyframe per spec; sampler can mask down to `[stage]/[act]` modes.
- pose data accuracy: forward Δ is what we feed and what the action captions — training the policy on this should generalise to inference (where the same forward Δ is computed from realtime state).

## Files

- v3 Qwen: `/data/zhaoqc/droid_cot/qwen_v3_mem_5ep.jsonl` (3/5 pass)
- **v3.1 Qwen**: `/data/zhaoqc/droid_cot/qwen_v31_5ep.jsonl` (4/5 pass) ⭐ recommended
- v3 MiMo: `/data/zhaoqc/droid_cot/mimo_v3_5ep.jsonl` (2/5 pass; passing eps are higher axis%)
- Spec doc: [`README_prompt_engineering_spec.md`](README_prompt_engineering_spec.md) §13 (sync'd with v3.1)
- Terms doc: [`TERMS_pick_and_place.md`](TERMS_pick_and_place.md) §11 (axis-aware glossary)
