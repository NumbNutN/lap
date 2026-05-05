# Cascade-VLA Configuration & Experiment Guide

This document describes the configuration toggles introduced for the Cascade-VLA
ablation matrix (Variants 0–3) on top of the LAP codebase, including how the
two-segment `[think] / [action]` data flow propagates through tokenizer, model,
and gemma backbone.

---

## 1. Configuration toggles

All toggles live on `lap.models.lap_config.LAPConfig`.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `action_attention_mode` | `Literal["lap_original", "unmask_langact"]` | `"lap_original"` | What the action expert can attend to in the prefix. |
| `stop_grad_mode` | `Literal["off", "full", "partial"]` | `"full"` | How action_expert → VLM gradient is masked at cross-attention boundary. |
| `stop_action_to_vlm_grad` | `bool` | `False` | Legacy binary switch. **Auto-derived** from `stop_grad_mode` (forced True when mode != "off", forced False when mode == "off"). Do not set manually unless you know what you're doing. |

### `action_attention_mode`

- **`"lap_original"`** — Action attends to image + prompt; `tokenized_langact_mask`
  positions are blocked. This is the baseline LAP behaviour. Works with both
  legacy single-segment and new two-segment data.
- **`"unmask_langact"`** — Action attends to image + prompt + langact; only the
  reasoning ([think]) span is blocked via `tokenized_reasoning_mask`. **Requires**
  the data pipeline to provide `tokenized_reasoning_mask` (i.e., the two-segment
  tokenizer path). If the mask is missing, the model falls back to `"lap_original"`
  with a logged warning.

### `stop_grad_mode`

Controls the per-position gradient flow when the action expert reads VLM K/V.

| Mode | Effect | Variant |
|------|--------|---------|
| `"off"` | No stop_gradient. Action loss flows back into the entire VLM trunk. | Variant 3 (Cascade-FullGrad) |
| `"full"` | stop_gradient on **all** VLM K/V. Equivalent to LAP's original `stop_action_to_vlm_grad=True`. | Variant 0 / 1 |
| `"partial"` | stop_gradient on **image + prompt + reasoning** K/V, but allow gradient through **langact** K/V. Requires `tokenized_reasoning_mask`. | Variant 2 (Cascade-Partial) |

Implementation: the gemma backbone receives a per-position `vlm_no_stop_mask`
constructed by `LAP._build_vlm_no_stop_mask` from
`tokenized_langact_mask AND NOT tokenized_reasoning_mask`. Inside
`Attention.__call__` ([gemma.py:207+](src/lap/models/backbones/gemma.py#L207)) the
mask is consulted when building the cross-expert K/V tensors used by
action-expert queries.

---

## 2. Experiment matrix

| Variant | `action_attention_mode` | `stop_grad_mode` | $L_{action}$ affects VLM trunk | Notes |
|---------|------------------------|-----------------|-------------------------------|-------|
| **0. LAP baseline**      | `"lap_original"`   | `"full"`    | ❌ | Original LAP. Reasoning is purely auxiliary. |
| **1. LAP-Unmask-Stop**   | `"unmask_langact"` | `"full"`    | ❌ | Action sees langact, but VLM not reshaped by action loss. |
| **2. LAP-Unmask-Partial**| `"unmask_langact"` | `"partial"` | ✅ langact only | Action shapes only langact hidden states, leaves prompt/reasoning untouched. |
| **3. LAP-Unmask-Free**   | `"unmask_langact"` | `"off"`     | ✅ everywhere | Cascade full gradient. |

### Recommended progression

1. **Variant 0 vs Variant 2** — most informative single comparison. Tests whether
   the langact information bottleneck is genuinely useful when both attention and
   gradient pathways exist.
2. **Variant 1** — isolates "information flow without gradient" (information
   value of langact alone).
3. **Variant 3** — isolates "full gradient" (does global VLM reshaping help or
   hurt vs. surgical langact-only reshaping?).

---

## 3. Data format — two-segment `[think] / [action]`

The two-segment cascade-VLA mode is activated when the data pipeline emits a
`langact` field alongside `language_actions`. Tokenizer layout becomes:

```
[BOS] <formatted_prompt> ; <reasoning> [action] <langact> [EOS] [PAD]...
        prompt span         [think] span    sep      [action] span
```

- `<reasoning>` corresponds to RoboTwin metadata `subgoal_prompt`
  (e.g. `Place the red block at the leftmost slot of the line.`).
- `<langact>` corresponds to one of the `phase_prompts[]` paraphrases
  (e.g. `Move the left gripper above the red block.`).
- The literal `[action]` separator is encoded but does **not** belong to either
  the reasoning or the langact mask.

Mask outputs from `PaligemmaTokenizer.tokenize`:

| Mask | Spans | Used for |
|------|-------|---------|
| `tokenized_langact_mask` (returned as `reasoning_mask` 3rd-position) | reasoning ∪ langact | next-token CE loss target; ar_mask for causal attention |
| `tokenized_reasoning_mask` (new 7th return value) | reasoning only | Action attention block (when `unmask_langact`); partial stop_grad selector |
| `number_mask` / `direction_mask` / `token_loss_mask` | unchanged | reasoning-dropout and verbose metrics |

### Backwards compatibility

If the pipeline does **not** pass `langact`, the tokenizer behaves identically to
the original LAP (single-segment) mode and returns `reasoning_only_mask=None`.
Downstream code falls back to `lap_original` attention behaviour.

### Pipeline wiring

`lap.transforms.TokenizePromptAndReasoning`:
- Pops optional `langact` from the input dict.
- Forwards it to the tokenizer.
- Emits `tokenized_reasoning_mask` only when the tokenizer returns a non-None
  mask (i.e., two-segment mode was active).

To activate two-segment mode in your dataset transform, populate `data["langact"]`
from the metadata's `phase_prompts[]` (pick one paraphrase per sample, possibly
randomly for diversity). The existing `language_actions` field becomes the
`subgoal_prompt`.

---

## 4. End-to-end gradient summary

For the Variant 2 (partial) configuration on a two-segment input:

```
                                attention block?      stop_gradient on K/V?
Image                                  no                    yes
Prompt                                 no                    yes
Reasoning ([think] segment)            YES                   yes
Langact   ([action] segment)           no                    NO  ← gradient flows
```

So the action expert can attend to image + prompt + langact; gradient from
$L_{action}$ flows back into the VLM trunk only through langact positions, leaving
prompt and reasoning representations purely shaped by $L_{lang}$.

---

## 5. Quick-start config snippets

```python
# Variant 0 — LAP baseline (default)
LAPConfig(
    enable_action_training=True,
    enable_langact_training=True,
    action_attention_mode="lap_original",
    stop_grad_mode="full",
)

# Variant 1 — Unmask langact, full stop_grad
LAPConfig(
    enable_action_training=True,
    enable_langact_training=True,
    action_attention_mode="unmask_langact",
    stop_grad_mode="full",
)

# Variant 2 — Unmask langact, partial stop_grad (recommended primary experiment)
LAPConfig(
    enable_action_training=True,
    enable_langact_training=True,
    action_attention_mode="unmask_langact",
    stop_grad_mode="partial",
)

# Variant 3 — Unmask langact, no stop_grad (cascade full gradient)
LAPConfig(
    enable_action_training=True,
    enable_langact_training=True,
    action_attention_mode="unmask_langact",
    stop_grad_mode="off",
)
```

---

## 6. Files touched

- [src/lap/models/lap_config.py](src/lap/models/lap_config.py) — new toggles, derivation logic, inputs_spec.
- [src/lap/models/model_adapter.py](src/lap/models/model_adapter.py) — `tokenized_reasoning_mask` field on `CoTObservation`.
- [src/lap/models/tokenizer.py](src/lap/models/tokenizer.py) — `_create_segmented_masks` helper; `PaligemmaTokenizer.tokenize` returns 7-tuple including `reasoning_only_mask`; `Gemma3Tokenizer.tokenize` updated to match arity.
- [src/lap/transforms.py](src/lap/transforms.py) — `TokenizePromptAndReasoning` accepts `langact` input field and emits `tokenized_reasoning_mask`.
- [src/lap/models/lap.py](src/lap/models/lap.py) — `_build_prefix_action_mask` switches on `action_attention_mode`; `_build_vlm_no_stop_mask` produces partial-mode mask; `compute_loss` threads it through the gemma forward.
- [src/lap/models/backbones/gemma.py](src/lap/models/backbones/gemma.py) — `Attention`, `Block`, `Module` accept `vlm_no_stop_mask` and apply per-position gradient passthrough at cross-expert K/V.

`gemma3.py` is **not** modified; LAP-Gemma3 still uses the original full-only
stop_gradient logic.

---

## 7. Sanity checks before training

1. Confirm your data pipeline now produces `tokenized_reasoning_mask` for at least
   one sample. Run a single batch through the transform and assert
   `batch["tokenized_reasoning_mask"].any(axis=1).any()`.
2. Confirm the masks do not overlap: `(reasoning_mask & langact_only).any() == False`,
   where `langact_only = langact_mask & ~reasoning_mask`.
3. With `stop_grad_mode="partial"` and an all-zero `tokenized_reasoning_mask` (i.e.,
   no reasoning content), partial mode should degrade to "off" behaviour. Verify
   loss values match those of `stop_grad_mode="off"` on the same batch.
4. With `action_attention_mode="unmask_langact"` and an all-zero
   `tokenized_reasoning_mask`, action attention should match `"lap_original"` with
   an empty langact span. Useful baseline alignment check.
