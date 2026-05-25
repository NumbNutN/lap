# v3.3 Annotation Pilot Report — Cross-Version Final Comparison

## Evolution summary (all 5 episodes × same DROID_100 subset)

| Version | pass | axis-aware action | landmark in stage | nums in stage | nums in action | think% |
|---------|------|-------------------|-------------------|---------------|----------------|--------|
| v3 Qwen | 3/5 | **43%** | 4% | 0 | 13 | 0% |
| v3.1 Qwen | 4/5 | **72%** ← peak axis | 4% | 0 | **49** ← too many | 0% |
| v3.2 Qwen | **5/5** | 6% ← over-corrected | 9% | 0 | 0 | 0% |
| **v3.3 Qwen** | 3/5 | 9% | **30%** ✅ | 0 ✅ | 0 ✅ | 0% |
| v3.2 MiMo | 4/5 | 9% | 52% | 0 | 0 | 1% |
| **v3.3 MiMo** | 3/5 | 14% | **70%** ⭐ | 6 (gap-to-target) ✅ | 0 ✅ | 4% |

### What improved v3.2 → v3.3

| Metric | v3.2 Qwen | v3.3 Qwen | v3.2 MiMo | v3.3 MiMo |
|---|---|---|---|---|
| **stage landmark rate** | 9% | **30%** ↑↑ | 52% | **70%** ↑ |
| **stage with gap-to-target nums** | 0 | 0 | 0 | **6** ✅ |
| stage plan-step coupling | medium | low ✅ | low | low ✅ |
| stage raw-delta nums | 0 | 0 | 0 | 0 ✅ |

### What still needs work

| Gap | Detail |
|---|---|
| **TIER B precision in action** | axis-aware 9-14% — model defaults to qualitative even at fine-tune keyframes. v3.1 had 72% but with raw numbers everywhere; v3.3 swung too far the other way. |
| **pass rate** | v3.3 regressed to 3/5 from v3.2's 5/5 — two new audit failures on tricky episodes (sharpie with double grasp/release, sachet release timing) |
| **think coverage** | still 0-4% vs target 30-40% — no retry cases in test set |

---

## Concrete stage quality improvement (marker-in-pot, Qwen v3.2 → v3.3)

| # | f | type | v3.2 stage | v3.3 stage |
|---|---|---|---|---|
| 0 | 0 | begin | "The robot begins its movement towards the marker, preparing to initiate the pick-and-place sequence." | "The robot begins with the gripper open, **positioned above the table**, ready to approach the marker visible near the left edge." |
| 3 | 35 | motion | "The robot makes final adjustments to its position, ensuring optimal alignment before closing the grippers." | "The gripper is **positioned directly above** the marker, needing a slight tilt to align its jaws for a top-down grasp." |
| 8 | 97 | motion | "The robot makes final adjustments to its roll, ensuring the marker is..." | "The gripper is now **directly above the pot**, with the marker positioned for release." |
| 9 | 134 | release | "The robot releases the marker into the pot, completing the primary task." | "The marker is released into the pot, completing the primary task. **The gripper opens above the pot opening.**" |

v3.2 stages were generic ("makes final adjustments", "ensures optimal alignment") — v3.3 uses spatial landmarks ("positioned directly above the marker", "directly above the pot"). **Each stage in v3.3 is identifiable without seeing the frame index.**

## MiMo v3.3 — best stage quality observed

MiMo's candy-bar ep (from earlier v3.2 run, same prompt family):
```
[3] "Very close to the candy bar now; the gripper pitches forward 39°
     to orient its jaws perpendicular to the counter for a top-down grasp."
```

This is the ideal: **gap-to-target number** (39° from perpendicular) + **spatial landmark** (close to candy bar, on the counter) + **target-pose reasoning** (orienting for top-down grasp). MiMo produces this style naturally; Qwen needs more fewshot reinforcement.

## Design insights to carry forward

### Insight 1: Three tiers work, but the model can't auto-detect TIER B

The prompt describes when TIER B (fine-tune precision) applies, but the VLM doesn't reliably identify those moments. **Fix**: tag keyframes in the metadata with `near_interaction: true` for the 1-3 keyframes immediately before/after grasp/release. Then R6 simplifies to "if near_interaction=true → TIER B, else TIER C."

### Insight 2: "Gap-to-target" phrasing > "raw delta" phrasing

Both produce numbers, but the semantics are different:
- Raw delta: "moved 22 cm sideways" — describes past motion, not intent
- Gap-to-target: "needs 39° more pitch to be perpendicular" — describes intent relative to a target state

The latter is vastly more useful for a policy: it tells the model WHAT configuration to reach, not just how far it moved. MiMo naturally generates gap-to-target phrasing; Qwen tends toward either raw deltas (v3.1) or no numbers at all (v3.2-v3.3).

### Insight 3: Stage landmark uniqueness is the strongest quality signal

The landmark rate (4% → 30% → 70%) correlates with perceived annotation quality better than any other metric. A stage that says "the gripper is directly above the pot, holding the marker" is instantly verifiable from the image — it's both a state description AND a visual grounding test. **Recommendation**: make stage-landmark-rate a primary quality metric, ahead of axis-aware-action-rate.

### Insight 4: MiMo produces better stages, Qwen is more audit-compliant

MiMo's spatial/temporal reasoning is richer (70% landmark, gap-to-target numbers, "Having secured..." memory phrases). But MiMo fails audit more often (JSON compliance, R6 TIER A adherence). For production:
- Use MiMo for **initial labeling** (richer content)
- Run Qwen **audit pass** + human review on MiMo output (catch structural issues)
- Or: iterate MiMo's JSON formatting compliance separately

---

## Files

| File | Description |
|---|---|
| `qwen_v33_5ep.jsonl` | v3.3 Qwen output (3/5 pass) |
| `mimo_v33_5ep.jsonl` | v3.3 MiMo output (3/5 pass) |
| Previous versions: `qwen_v3_mem_5ep.jsonl`, `qwen_v31_5ep.jsonl`, `qwen_v32_5ep.jsonl`, etc. | All in `/data/zhaoqc/droid_cot/` |

## Updated docs

| Doc | Section updated |
|---|---|
| `TERMS_pick_and_place.md` | §11 axis-aware + §11.3 grip-verb priority |
| `README_prompt_engineering_spec.md` | §13 implementation status |
| `README_annotation_design.md` | (to be updated with insights 1-4 above) |
