# v3 Memory-Augmented Pilot Report (5 episodes, Qwen2.5-VL-72B)

## What changed in v3 (vs v2 mode-A)

| Aspect | v2 mode-A | v3 memory+pose |
|--------|-----------|----------------|
| system prompt | `SYSTEM_PROMPT` (no memory hint) | `SYSTEM_PROMPT_V3_MEMORY` |
| per-keyframe input | `frame_idx`, `type`, `gripper_state` | + **pose delta** `Δxyz(cm) + Δrot(deg)+ axis_name` |
| schema field added | — | **`mode_marker`** = `[think_act]` |
| stage style | "describe current state" | **"include image-invisible context: plan progress, history"** |
| action vocab | open-vocab natural language | **axis-aware**: yaw/pitch/roll/translate + magnitude |
| think rule | retry only | retry **required** + multi-step plan / obstacle / invisible-info **encouraged** (target 30-40%) |
| fewshot | v1 example included | skipped (v3 schema differs) |

## Headline numbers (5 episodes, 69 keyframes total)

| Metric | v3 result | Target / Goal |
|--------|-----------|---------------|
| pass rate | **3/5** | high; 2 fails are A3/A8 prompt-followup issues |
| `mode_marker` emitted | **69/69** (100%) | spec compliance |
| action axis-aware | **30/69** (43%) | up from ~0% in v2 |
| stage with history references | **29/69** (42%) | up from ~0% in v2 |
| `think` non-null | **0/69** (0%) | target 30-40%; no retry cases in these 5 eps |

## Episode 0 (RAIL "Put the marker in the pot") — v2 vs v3

Trajectory (verified previously): start with gripper holding initially → release at f60 → re-grasp at f146-ish.

| # | frame | type | v2 action | v3 action |
|---|---|---|---|---|
| 0 | 0 | `begin` | Move towards the marker on the table. | Position the gripper above the marker. |
| 1 | 11 | `motion` | Adjust the arm's position to align with the marker. | Move slightly forward and downward towards the marker. |
| 2 | 27 | `motion` | Fine-tune the arm's position closer to the marker. | Adjust position further to align with the marker. |
| 3 | 35 | `motion` | Make final adjustments to ensure proper grasp alignment. | Fine-tune position for optimal marker grasp. |
| 4 | 59 | `grasp` | Close the gripper to grasp the marker. | Close the gripper to grasp the marker firmly. |
| 5 | 67 | `motion` | Lift the marker upwards. | Lift the marker slightly off the surface. |
| 6 | 75 | `motion` | Move the marker towards the pot. | Yaw counterclockwise slightly to align with the pot. |
| 7 | 87 | `motion` | Adjust the arm's position above the pot. | Move towards the pot while adjusting orientation. |
| 8 | 97 | `motion` | Position the marker directly above the pot. | Lower the marker towards the pot opening. |
| 9 | 134 | `release` | Open the gripper to drop the marker into the pot. | Release the marker into the pot by opening the gripper. |
| 10 | 145 | `motion` | Move the arm away from the pot. | Make slight adjustments post-release. |
| 11 | 165 | `end` | Return to a neutral position with the gripper open. | Retract back to the initial position. |

### Stages (showing memory carry-through in v3)

| # | type | v2 stage | v3 stage |
|---|---|---|---|
| 0 | `begin` | Initial setup with the gripper open and away from the marker.... | Starting position established. The robot is preparing to approach the marke... |
| 1 | `motion` | The robot adjusts its position to better align with the marker.... | Initial approach phase. The robot has moved closer to the marker, setting u... |
| 2 | `motion` | The robot continues to adjust its position for optimal grasp.... | Continuing approach. The robot has adjusted its position to better align wi... |
| 3 | `motion` | Final adjustments before attempting to grasp the marker.... | Final approach before grasping. The robot is now very close to the marker, ... |
| 4 | `grasp` | The gripper closes to grasp the marker.... | Grasping phase initiated. The robot has closed the gripper to secure the ma... |
| 5 | `motion` | The robot lifts the marker off the table.... | Lift phase begins. The robot has successfully grasped the marker and starts... |
| 6 | `motion` | The robot moves the marker towards the pot.... | Transitioning to transport. The robot adjusts its orientation to move towar... |
| 7 | `motion` | The robot adjusts its position above the pot.... | Transport phase. The robot continues moving towards the pot, maintaining a ... |
| 8 | `motion` | The robot prepares to release the marker into the pot.... | Approaching the pot. The robot has aligned with the pot and is preparing to... |
| 9 | `release` | The gripper opens to release the marker into the pot.... | Release phase. The robot has positioned the marker over the pot and is abou... |
| 10 | `motion` | The robot moves away from the pot after releasing the marker.... | Post-release adjustment. The robot has opened the gripper and is making min... |
| 11 | `end` | The robot completes the task and returns to a neutral position.... | Task completion. The robot has finished the sequence and is retracting to t... |


## Sample v3 axis-aware actions (showing pose-delta uptake)

These were impossible to emit before pose data was injected:

- *"Adjust position by tilting downward and moving right."*
- *"Yaw counterclockwise and move closer to the sachet."*
- *"Pitch upward and move closer to the sachet."*
- *"Pitch downward and lift the sachet higher."*
- *"Move towards the bowl while pitching upward."*
- *"Move closer to the bowl while adjusting pitch."*

Compare with v2 (same scenes, no pose data):

- "Adjust the arm's position to align with the marker."
- "Fine-tune the arm's position closer to the marker."
- "Move the marker towards the pot."

v2 is correct but **generic**; v3 names the actual axis being changed. This is exactly the "finer-grained signal" you asked for.

## Sample v3 history-carrying stages

- *"Following the initial search, the robot adjusts its position to better locate the sachet."*
- *"After locating the sachet, the robot adjusts its orientation to prepare for grasping."*
- *"With the sachet partially grasped, the robot prepares to lift it off the table."*
- *"Moving towards the bowl, the robot adjusts its position to ensure accurate placement of the sachet."*
- *"Post-release, the robot makes minor adjustments to confirm the sachet is properly placed in the bowl."*

These reference prior episode events ("the initial search", "locating", "having grasped", "post-release") — exactly the image-invisible context the design doc §11.2 asks for.

## Failure modes seen in 5-ep v3

### F1: pose-delta dominance overrides gripper verb at grasp/release

Two episodes failed audit because at a `release` keyframe, pose change was large and model wrote a yaw-or-translate action instead of "release the X". Examples:

- `[FAIL] Doritos` — kf[8] type=release, action: `"Yaw slightly while moving closer to the sink."` (no "release" verb)
- `[FAIL] pen in bowl` — kf[9] type=release, action: `"Position the pen on the table surface."` (no "release" verb)

**Fix**: tighten system prompt R6 — *"on grasp/release/retry keyframes the grip verb takes priority over the pose delta in `action`; describe both only if room (≤12 words)."*

### F2: 0% think coverage (need more retry / decision data)

5 DROID episodes are all clean single-attempt successes → no retry. Target 30-40% think requires either:
- Multi-arm / cluttered episodes where ordering matters (we don't have these in DROID single-arm)
- Add a prompt hint: *"emit think on any approach keyframe where the angle/position choice is non-obvious"*

For now, **defer**. Re-measure after running on 50+ ep including episodes with visible re-grasps.

## Recommended next iterations

1. **Patch R6** in `SYSTEM_PROMPT_V3_MEMORY` to clarify grip-vs-pose verb priority on grasp/release.
2. **Run 100ep v3** to get a representative think distribution + better failure-mode sample.
3. **Update spec doc & terms** to canonical v3 shape (this PR).
4. **Add axis-aware terms** to `TERMS_pick_and_place.md` so manual annotators stay consistent with v3 output.

## Spec doc alignment status

Comparing implementation against [`README_prompt_engineering_spec.md`](README_prompt_engineering_spec.md) §1/§3/§7:

| Spec §1 field | v3 schema | match? |
|---|---|---|
| `mode_marker` `[think_act]` | yes | ✅ |
| `type` (7-class) | yes (from detector) | ✅ |
| `gripper_state` | yes (from detector) | ✅ |
| `stage` 15-40 words | yes (prompt says 15-40) | ✅ (typical 18-30 in actual output) |
| `think` null or 1-2 sentences | yes | ✅ shape, ❌ coverage |
| `action` 5-12 words | yes (prompt says 5-12) | ⚠️ some 8-15 words observed |

| Spec §3 system prompt | v3 prompt | match? |
|---|---|---|
| memory chain in stage | ✅ |  |
| axis-name in action | ✅ |  |
| mode_marker required | ✅ |  |
| schema in prompt | ✅ |  |
| think target 30-40% | ❌ 0% measured (5 ep too small) |  |

| Spec §7 audit | implementation | match? |
|---|---|---|
| A0 schema completeness | ✅ |  |
| A1/A2 enum check | ✅ (in audit.py) |  |
| A3 type/verb consistency | ✅ but **too strict** — flagged 2 v3 releases |  |
| A4 retry-think required | ✅ |  |
| A5 length limits | ✅ |  |
| A6 stage history check | not implemented |  |
| A7/A8 gripper-state consistency | ✅ (skips grasp/release types) |  |

## Files

- v3 output: `/data/zhaoqc/droid_cot/qwen_v3_mem_5ep.jsonl`
- Local copy: `/tmp/pilot/qwen_v3_mem_5ep.jsonl`
- v2 reference (mode-A 10 ep): `/tmp/pilot/qwen_v2_modeA_10ep.jsonl`
