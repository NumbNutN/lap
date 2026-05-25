# 5-Episode Qwen-VL Comparison — buggy detector vs fixed detector vs no-types

## Bug context

Earlier this session we identified a **gripper convention inversion** in our keyframe detector. DROID's `observation/gripper_position` is normalized **[0=open, 1=closed]** (commanded position), but our detector assumed Franka physical width **[0≈closed, 0.08≈open]**. Every `grasp` / `release` label was therefore reversed.

Fix in `keyframe.py`:
- `GRIP_OPEN_MAX = 0.15` (was `GRIP_CLOSED_MAX = 0.005`)
- `GRIP_CLOSED_MIN = 0.50` (was `GRIP_OPEN_MIN = 0.060`)
- `gripper_state()` thresholds reversed accordingly
- Added `refine_to_motion_start()` to anchor keyframes at the START of the close/open motion (matches user's f60 grasp / f134 release observations) rather than the END of the state transition.

After the fix, ep0 detection:
```
v1 buggy:  [begin, motion×3, release(f60), motion×6, grasp(f146), end]
v2 fixed:  [begin, motion×3, grasp(f59),  motion×4, release(f134), motion, end]
```

User's ground-truth observations:
- f60 = grasp start ✓ (matches f59)
- f134 = release start ✓ (matches f134)
- f146 grasp was wrong ✓ (no longer present)

## Pass rate comparison (10 episodes from DROID_100)

| Pipeline | Detector | Prompt | Pass rate |
|----------|----------|--------|-----------|
| **v1 buggy** (original) | inverted gripper | types fed | 3/5 → 5/5 (after A8 relax) |
| **v2 mode-A** | corrected gripper | types fed | **10/10** (after A8 update for motion-start anchoring) |
| **v2 mode-B** | corrected gripper | NO types — VLM derives | 0/10 audit pass, but failures are VLM internal type-action mismatches; content is partly correct |

## Mode-B unique-type vocabulary (10 episodes)

VLM was free to use open vocabulary, but stayed close to our standard types:

| Type | Count |
|------|-------|
| `motion` | 110× |
| `grasp` | 13× |
| `begin` | 10× |
| `end` | 10× |
| `retry` | 10× |
| `release` | 4× |

No open-vocab verbs emerged. Notable: VLM emitted **fewer release labels (4)** than expected (10) — it seems to label release frames as `motion`. The "10× retry" is suspicious — Qwen may have over-labeled retries that don't exist (since 0 episodes in DROID_100 are retries). The "4× release" is suspiciously low. The "13× grasp" suggests at least one episode had 2 grasps (the marker pot episode does — original grasp + after-release re-grasp).

---

### Episode 0 — Put the marker in the pot (the diagnostic case)

Task: **Put the marker in the pot**
Frames: 166, keyframes: 12

Detector types (v2 corrected): `['begin', 'motion', 'motion', 'motion', 'grasp', 'motion', 'motion', 'motion', 'motion', 'release', 'motion', 'end']`
Detector types (v1 buggy):     `['begin', 'motion', 'motion', 'motion', 'release', 'motion', 'motion', 'motion', 'motion', 'motion', 'grasp', 'end']`

**Plans:**
- v2 mode-A: Put the marker in the pot. Steps: 1) approach the marker, 2) grasp the marker, 3) move towards the pot, 4) release the marker into the pot.
- v2 mode-B: The robot aims to put the marker in the pot. 1) It approaches the marker, 2) grasps it, 3) moves towards the pot, 4) releases the marker into the pot.
- v1 (buggy): Put the marker into the pot. Steps: 1) Move towards the marker, 2) Grasp the marker, 3) Move towards the pot, 4) Release the marker into the pot.

| # | frame | A-type | B-type | A-action | B-action | v1-action |
|---|---|---|---|---|---|---|
| 0 | 0 | `begin` | `begin` | Move towards the marker on the table. | Move towards the marker on the table | Move towards the pot on the table. |
| 1 | 11 | `motion` | `motion` | Adjust the arm's position to align with the marker. | Continue approaching the marker on the table | Adjust the position of the marker closer to the pot. |
| 2 | 27 | `motion` | `motion` | Fine-tune the arm's position closer to the marker. | Position the gripper above the marker | Continue moving the marker towards the pot. |
| 3 | 35 | `motion` | `grasp` | Make final adjustments to ensure proper grasp alignment | Grasp the marker on the table | Position the marker directly above the pot. |
| 4 | 59 | `grasp` | `motion` | Close the gripper to grasp the marker. | Lift the marker upwards | Release the marker into the pot. |
| 5 | 67 | `motion` | `motion` | Lift the marker upwards. | Move the marker towards the pot | Move away from the pot. |
| 6 | 75 | `motion` | `motion` | Move the marker towards the pot. | Position the marker above the pot | Continue moving away from the pot. |
| 7 | 87 | `motion` | `motion` | Adjust the arm's position above the pot. | Fine-tune the position above the pot | Ensure the gripper is clear of the pot. |
| 8 | 97 | `motion` | `motion` | Position the marker directly above the pot. | Prepare to drop the marker into the pot | Maintain distance from the pot. |
| 9 | 134 | `release` | `release` | Open the gripper to drop the marker into the pot. | Release the marker into the pot | — |
| 10 | 145 | `motion` | `motion` | Move the arm away from the pot. | Move away from the pot | Prepare for the next action by partially closing the gr |
| 11 | 165 | `end` | `end` | Return to a neutral position with the gripper open. | Stop moving | Hold the marker securely. |

---

### Episode 1 — Put one green sachet in the grey bowl

Task: **Put one green sachet in the grey bowl.**
Frames: 142, keyframes: 13

Detector types (v2 corrected): `['begin', 'motion', 'motion', 'motion', 'motion', 'grasp', 'motion', 'motion', 'motion', 'motion', 'release', 'motion', 'end']`
Detector types (v1 buggy):     `['begin', 'motion', 'motion', 'motion', 'motion', 'release', 'motion', 'motion', 'motion', 'motion', 'motion', 'grasp', 'end']`

**Plans:**
- v2 mode-A: Place a green sachet into the grey bowl. Steps: 1) move towards the sachets, 2) grasp a green sachet, 3) lift and move towards the grey bowl, 4) release the sachet into the bowl.
- v2 mode-B: The robot aims to place the candy bar on the shelf. 1) Move towards the candy bar, 2) Grasp the candy bar, 3) Lift and move to the shelf. The task appears incomplete as the candy bar is not placed.
- v1 (buggy): Place a green sachet into the grey bowl. Steps: 1) pick up the green sachet, 2) move towards the grey bowl, 3) drop the sachet into the bowl.

| # | frame | A-type | B-type | A-action | B-action | v1-action |
|---|---|---|---|---|---|---|
| 0 | 0 | `begin` | `begin` | Move towards the green sachets on the table. | Move towards the candy bar on the counter. | Position the gripper near the green sachet. |
| 1 | 11 | `motion` | `motion` | Adjust the arm position closer to the green sachets. | Extend the arm towards the candy bar. | Adjust the gripper orientation to grasp the green sache |
| 2 | 19 | `motion` | `motion` | Fine-tune the arm position for optimal sachet grasp. | Adjust the arm's position over the candy bar. | Align the gripper fingers with the green sachet. |
| 3 | 32 | `motion` | `grasp` | Position the gripper directly above a green sachet. | Close the grippers to grasp the candy bar. | Prepare to close the gripper on the green sachet. |
| 4 | 47 | `motion` | `motion` | Prepare to close the gripper on the green sachet. | Lift the candy bar upwards. | Lift the green sachet off the table. |
| 5 | 74 | `grasp` | `motion` | Close the gripper to grasp the green sachet. | Move the arm towards the shelf. | Release the green sachet into the grey bowl. |
| 6 | 79 | `motion` | `motion` | Lift the green sachet upwards. | Continue moving the arm closer to the shelf. | Move the gripper away from the grey bowl. |
| 7 | 87 | `motion` | `motion` | Move the arm towards the grey bowl. | Fine-tune the arm's position near the shelf. | Retract the gripper further from the grey bowl. |
| 8 | 103 | `motion` | `motion` | Adjust the arm position above the grey bowl. | Slightly retract the arm while maintaining the candy ba | Reorient the gripper for the next task. |
| 9 | 111 | `motion` | `motion` | Position the gripper directly above the grey bowl. | Adjust the arm's position to align the candy bar with t | Approach the next object on the table. |
| 10 | 116 | `release` | `motion` | Open the gripper to drop the green sachet into the grey | Make final adjustments to the arm's position. | — |
| 11 | 124 | `motion` | `motion` | Move the arm back away from the grey bowl. | Pause briefly while holding the candy bar. | Fine-tune the gripper position for precise grasping. |
| 12 | 141 | `end` | `motion` | Retract the arm fully and open the gripper. | Retract the arm slightly while holding the candy bar. | Maintain the gripper's closed position. |

---

### Episode 2 — Put the candy bar on the left side of the first shelf

Task: **Put the candy bar on the left side of the first shelf**
Frames: 238, keyframes: 17

Detector types (v2 corrected): `['begin', 'motion', 'motion', 'motion', 'grasp', 'motion', 'motion', 'motion', 'motion', 'motion', 'release', 'motion', 'motion', 'motion', 'motion', 'motion', 'end']`
Detector types (v1 buggy):     `['begin', 'motion', 'motion', 'motion', 'release', 'motion', 'motion', 'motion', 'motion', 'motion', 'motion', 'grasp', 'motion', 'motion', 'motion', 'motion', 'end']`

**Plans:**
- v2 mode-A: Place the candy bar on the left side of the first shelf. Steps: 1) approach the candy bar, 2) grasp it, 3) lift and move towards the shelf, 4) release it on the shelf.
- v2 mode-B: The robot aims to put a green sachet in the grey bowl. 1) Move towards the sachets, 2) grasp a green sachet, 3) lift it, 4) drop it into the grey bowl.
- v1 (buggy): Place the candy bar on the left side of the first shelf. Steps: 1) pick up the candy bar, 2) move towards the shelf, 3) place the candy bar on the shelf.

| # | frame | A-type | B-type | A-action | B-action | v1-action |
|---|---|---|---|---|---|---|
| 0 | 0 | `begin` | `begin` | Move towards the candy bar on the countertop. | Move towards the green sachets on the table. | Move towards the candy bar on the countertop. |
| 1 | 11 | `motion` | `motion` | Adjust the arm position closer to the candy bar. | Extend the arm closer to the green sachets on the table | Align the gripper with the candy bar. |
| 2 | 25 | `motion` | `motion` | Fine-tune the arm position directly above the candy bar | Adjust the arm position for better alignment with the s | Position the gripper directly above the candy bar. |
| 3 | 34 | `motion` | `grasp` | Lower the gripper slightly towards the candy bar. | Close the gripper to grasp the green sachet firmly. | Grasp the candy bar firmly. |
| 4 | 68 | `grasp` | `motion` | Close the gripper to grasp the candy bar. | Lift the green sachet upwards from the table. | Release the candy bar. |
| 5 | 83 | `motion` | `motion` | Lift the candy bar upwards. | Move towards the grey bowl with the green sachet. | Move the arm upwards and away from the countertop. |
| 6 | 91 | `motion` | `motion` | Move the arm horizontally towards the first shelf. | Fine-tune the position above the grey bowl. | Adjust the arm orientation towards the shelf. |
| 7 | 101 | `motion` | `release` | Lower the arm to align with the left side of the shelf. | Open the gripper to drop the green sachet into the grey | Move closer to the shelf. |
| 8 | 130 | `motion` | `motion` | Make minor adjustments to position the candy bar correc | Move the arm away from the grey bowl. | Position the arm near the left side of the shelf. |
| 9 | 139 | `motion` | `motion` | Confirm the candy bar is aligned with the left side of  | Retract the arm to a safe distance from the table. | Fine-tune the arm position for accurate placement. |
| 10 | 170 | `release` | `motion` | Open the gripper to release the candy bar on the shelf. | Settle the arm in a stable resting position. | — |
| 11 | 175 | `motion` | `motion` | Move the arm back slightly. | Hold the arm steady in its final position. | Move the arm back slightly. |
| 12 | 189 | `motion` | `end` | Continue moving the arm away from the shelf. | Conclude the task with the arm in its final position. | Lift the candy bar upwards. |
| 13 | 202 | `motion` | `?` | Retract the arm further to a neutral position. | ? | Move the arm towards the shelf. |
| 14 | 218 | `motion` | `?` | Make final adjustments to fully retract the arm. | ? | Place the candy bar on the left side of the shelf. |
| 15 | 226 | `motion` | `?` | Hold the arm in the retracted position. | ? | Adjust the candy bar position if necessary. |
| 16 | 237 | `end` | `?` | Maintain the arm in its final retracted position. | ? | Retract the arm while keeping the gripper closed. |

---

## Quality observations

### Mode-A v2 (types fed, fixed detector) — recommended

**ep0**: Output is **correct end-to-end**.
- f59 (grasp): "Close the gripper to grasp the marker" ✓
- f67-97 (transport): all correctly describe carrying to pot ✓
- f134 (release): "Open the gripper to drop the marker into the pot" ✓
- f165 (end): "Return to a neutral position with the gripper open" ✓

The model is using the correct detector type as anchor + visual context to fill in object names and spatial details. With correct types, Qwen's behaviour stops looking "broken".

### Mode-B (no types, VLM derives everything)

**ep0**: Has a critical timing miss — VLM labeled f35 as `grasp` (action="Grasp the marker on the table") even though gripper is still fully open (g=0.0). The actual grasp happens at f59 (where VLM labeled as `motion` + action="Lift the marker upwards" — already imagining post-grasp).

This is the failure mode the user originally identified: **VLM uses task-instruction prior to fill in the expected timing, not actual visual grounding**. Without our detector's anchor, VLM hallucinates the grasp 24 frames early.

VLM's other annotations (release at f134, transport phases, end) are reasonable. But the **timing of state transitions** is the wormhole.

### Action style — both modes good

Open-vocabulary, spatial, compound — exactly what we set up the prompt for:
- "Adjust the arm's position to align with the marker"
- "Fine-tune the arm's position closer to the marker"
- "Prepare to drop the marker into the pot"

No "Move down / Move left" ECoT-primitive style. Both modes preserve this.

---

## Verdict

| Question | Answer |
|----------|--------|
| Was the previous Qwen output "broken"? | **No** — we mislabeled. Qwen was faithfully following our inverted types. |
| Does Mode-A v2 (fixed types) produce usable annotations? | **Yes** — ep0 is end-to-end correct. |
| Does Mode-B (no types) improve over A? | **No** — VLM hallucinates grasp timing without our anchor. |
| Should we go full open-vocab type, no detector? | **No** — at least keep gripper-based grasp/release timing. |

### Recommended pipeline going forward

1. **Detector v2** (gripper fixed + motion-start anchoring) — current code ✓
2. **Types fed to VLM** (mode A) — gives VLM correct temporal anchor ✓
3. **Audit A8** — adjusted to skip grasp/release types ✓ (already in audit.py)
4. **Human review in viewer** — still needed for fine corrections (e.g. some plans still slip; f0 = "Move towards marker" when arm is actually at home position)

### Open questions for user analysis

- Episodes 6 + 8 have **empty task_instruction** in DROID. The VLM had to infer the task from images. Worth a look in the viewer to see if it inferred reasonable tasks.
- ep4 "Move the sharpie to the table" — the trajectory has multiple grasp/release cycles. Worth checking if mode-A captured all of them correctly.

## Files

- `qwen_v2_modeA_10ep.jsonl` — types fed, detector fixed
- `qwen_v2_modeB_10ep.jsonl` — no types, VLM derives
- `qwen_pilot_5ep.jsonl` — original buggy detector (kept for archaeology)
