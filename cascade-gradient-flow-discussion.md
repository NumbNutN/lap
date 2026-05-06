# Cascade VLA 设计：梯度流向与训练变体讨论

本文档梳理在 LAP 基础上设计 cascade reasoning → action 架构时，几种训练变体之间的精确差异，以及背后的梯度流向分析。

---

## 1. 名词约定

- **VLM trunk / Expert 0**：处理 image + prompt + reasoning + langact 的共享 transformer 主干（PaliGemma）
- **Action Expert / Expert 1**：处理 action token 的独立专家（同一 attention 层，不同 Q/K/V/MLP 权重）
- **shared attention**：MoE 多专家架构下，两个 expert 在 attention 层混合 token，互相能 attend
- **TF (Teacher Forcing)**：训练时把 ground-truth langact token 喂入 prefix
- **LM head**：把 hidden state 投影到词表空间的输出层
- **Hidden state $h$**：trunk 在 langact 位置输出的连续向量
- **prompt**：任务级描述，例 `Arrange the blocks into a horizontal line.`
- **reasoning** ([stage] / Subgoal)：高层规划，例 `Place the red block at the leftmost slot of the line.`
- **langact** ([action] / Step)：低层语言动作描述，例 `Move the left gripper above the red block.`
- 数据格式参考：[data/arrange_blocks_line/demo_clean/metadata/episode0.json](data/arrange_blocks_line/demo_clean/metadata/episode0.json)

---

## 2. Variant 2 (LAP-Unmask) 理解确认

### 2.1 核心改动
- 取消 [`_build_prefix_action_mask`](policy/lap/src/lap/models/lap.py#L303-L325) 对 langact 的屏蔽
- action 通过 cross-attention 读 langact 位置的 hidden state

### 2.2 LAP 原文中的 stop_gradient 是什么

LAP 论文：

> Following [25], we further block gradients from the action expert propagating back into the VLM backbone. While not required by LAP, this knowledge insulation helps preserve pre-trained VLM representations and stabilizes joint training.

这对应代码里的 `stop_action_to_vlm_grad` flag（[lap.py:47](policy/lap/src/lap/models/lap.py#L47)），实际实现位于 [gemma.py:242-269](policy/lap/src/lap/models/backbones/gemma.py#L242-L269)。

### 2.3 stop_gradient 的精确位置：cross-expert attention 的 K 和 V

```python
# gemma.py:250-255
k0 = k[:, :expert0_len, ...]              # VLM expert 的 K
q_i = q[:, expert0_len:, ...]             # action expert 的 Q
logits0_i = jnp.einsum(
    "BTKGH,BSKH->BKGTS",
    q_i,
    jax.lax.stop_gradient(k0),           # ← 关键：detach 掉 VLM 的 K
    ...
)
```

```python
# gemma.py:265-269
probs_cross = probs * cross_to_expert0.astype(probs.dtype)   # 跨专家 attention 概率
encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_self, v) + jnp.einsum(
    "BKGTS,BSKH->BTKGH",
    probs_cross,
    jax.lax.stop_gradient(v)              # ← 关键：detach 掉 VLM 的 V
)
```

### 2.4 这个 stop_gradient 在做什么

**保留**：
- Action expert 自己的 Q/K/V/MLP 权重 — 接收 $L_{action}$ 梯度
- VLM 自己的 self-attention（VLM 内部 token 互相 attend）— 接收 $L_{lang}$ 梯度
- VLM 的 K/V 仍然正常前向计算，参与 attention

**屏蔽**：
- 当 action expert 的 Q 跨专家读取 VLM 的 K/V 时，反向传播**不会**通过 K/V 流回 VLM expert 的 K/V 投影矩阵
- 即 $L_{action}$ 不会通过"读取 langact"这条路径影响 VLM trunk 的权重

### 2.5 几种粒度的 stop_gradient 对比

| 位置 | 控制粒度 | 影响 |
|------|---------|------|
| **A. Cross-attention K/V**（LAP 当前） | 全局 — 任何来自 expert ≥1 对 expert 0 的 attention 都被屏蔽 | $L_{action}$ 完全无法影响 VLM 主干，但保留 VLM 内部 self-attention 上的 $L_{lang}$ 梯度 |
| **B. Langact hidden states 整体** | 仅 langact 位置 | $L_{action}$ 不影响 langact 的 trunk 计算路径，但仍可以影响 image/prompt 的 trunk 计算（如果 action 也读它们） |
| **C. 整个 prefix hidden** | 所有 prefix token | 等效于 A，但实现更粗暴 |
| **D. 不加 stop_gradient** | 无 | $L_{action}$ 能通过 langact / image / prompt hidden 全面影响 VLM trunk |

LAP 选择 A：保留 image+prompt 在 VLM 内部的 self-attention 梯度（$L_{lang}$ 仍训练 VLM），但把跨专家方向上的耦合切断。

### 2.6 Variant 2 在 LAP-Unmask 下的两种子配置

| 子配置 | langact 屏蔽 | stop_action_to_vlm_grad | $L_{action}$ 能否影响 VLM trunk |
|--------|-------------|-------------------------|-------------------------------|
| **2a (Unmask + stop_grad)** | 关 | 开（默认 LAP 设置） | 否 |
| **2b (Unmask + no stop_grad)** | 关 | 关 | 是（通过 langact + image + prompt hidden） |

**注意**：2a 下虽然 action 能"看见" langact，但因为 cross-expert K/V 被 detach，VLM 不会被 $L_{action}$ "调教成更适合 action 读取"的形态。所以 2a 实际上和原版 LAP 在训练动力学上**很接近**，只是推理时 action 多了一个信息源。

---

## 3. Variant 3 (Cascade-FullGrad) 理解确认

### 3.1 你的描述（正确）

> Teacher forcing 下的半连接：用 ground-truth langact tokens 拼接到 prefix，$L_{action}$ 的梯度能流回 trunk + embedding（流不到 LM Head）

✅ 完全正确。前提是关掉 `stop_action_to_vlm_grad`，否则就退化成 Variant 2a。

### 3.2 三种 reasoning → action 信息传递配置对比

#### 配置 A：Action expert 重新输入 reasoning token（独立编码）

```
[Pass 1] VLM trunk → 生成 reasoning_token_ids
                              ↓ stop_gradient (sampling 不可微)
[Pass 2] Action expert 内部独立的 embedding lookup → action expert transformer
              ↑ 这一支与 VLM trunk 无任何参数共享或梯度连接
```

**梯度流向**：
- $L_{action}$ → action expert 权重 + action expert 自己的 embedding 表
- ❌ 无法影响 VLM trunk 权重
- ❌ 无法影响 VLM 的 embedding 表

**适用场景**：action expert 是真正独立的小模型（如一个外挂在 VLM 后的 controller）

#### 配置 B：Shared trunk + TF + 不加 stop_gradient（你的 Variant 3 / Cascade-FullGrad）

```
shared trunk 处理 [imgs, prompt, langact_GT]
   ↓
hidden states: [H_img, H_prompt, H_langact]
   ↓
action queries 通过 cross-attention 读 H_langact 作为 K/V
```

**梯度流向**：
- $L_{action}$ → action expert 权重
- $L_{action}$ → 通过 cross-attention 反向 → H_langact → trunk 所有 attention/MLP 层 → embedding 表（langact 行）
- ❌ 不到达 LM head（因为前向时 GT token id 不经过 LM head 计算）
- $L_{lang}$ → LM head + trunk 所有层 + embedding 表

**双 loss 共享**：trunk 和 embedding 表（不同行被不同 loss 触达）

#### 配置 C：Shared trunk + TF + stop_gradient on cross-attention（即 Variant 2a / LAP 风格）

```
和配置 B 完全相同的前向，但跨专家 attention 时 detach VLM 的 K/V
```

**梯度流向**：
- $L_{action}$ → action expert 权重（仅）
- $L_{lang}$ → LM head + trunk 所有层 + embedding 表
- ❌ $L_{action}$ 不影响 VLM trunk

**特点**：行为上接近配置 A，但实现更轻——不需要两次前向，所有 token 在同一前向中处理

### 3.3 三种配置的本质差异（一张图）

```
                  L_action 能影响什么
配置          action expert 权重 │ VLM trunk │ VLM embedding │ LM head
─────────────────────────────────┼───────────┼──────────────┼────────
A (独立编码)      ✓               │    ✗      │     ✗        │   ✗
B (Cascade-Full)  ✓               │    ✓      │     ✓ (部分)  │   ✗
C (Cascade-Stop)  ✓               │    ✗      │     ✗        │   ✗
─────────────────────────────────┴───────────┴──────────────┴────────
            （LM head 永远收不到 L_action 因为 TF 切断了 logits 计算图）
```

注：B 中 embedding 表只有"langact 用到的行"被 $L_{action}$ 触达；A 中 action expert 自己有独立的 embedding 表（与 VLM 无关）。

---

## 4. CE Loss 的梯度直觉（数学化确认）

### 4.1 你的直觉表述

> 为了让预测更加准确，模型会将对应的真实目标词的 embedded 向量推向该隐藏状态的方向，而将其他词汇的 embedded 推向远离该隐藏状态的方向（这是梯度流到 LM Head 时发生的），而 hidden states 也会被推向一个让 loss 更小的方向，而 action expert 为了让 hidden state 真的有帮助也会塑造 hidden state

✅ 完全正确。下面是数学化精确版本。

### 4.2 推导

设 logits $z = W h$，其中 $W \in \mathbb{R}^{V \times D}$ 是 LM head（$W_y$ 表示第 $y$ 行），$h \in \mathbb{R}^{D}$ 是 hidden state。softmax 概率 $p_y = \text{softmax}(z)_y$，CE loss $L = -\log p_{y^*}$，$y^*$ 是真实目标词。

**(a) 对 $W_y$ 的梯度**（LM head 行）：

$$
\frac{\partial L}{\partial W_y} = (p_y - \mathbb{1}[y = y^*]) \cdot h
$$

- 当 $y = y^*$：$p_y - 1 < 0$，梯度方向是 $-h$，即 $W_{y^*}$ 更新方向是 $+h$ → **真实词的行向 $h$ 靠近**
- 当 $y \neq y^*$：$p_y > 0$，梯度方向是 $+h$，即 $W_y$ 更新方向是 $-h$ → **其他词的行远离 $h$**（且远离力度正比于该词当前的预测概率 $p_y$）

**(b) 对 $h$ 的梯度**（hidden state）：

$$
\frac{\partial L}{\partial h} = \sum_y (p_y - \mathbb{1}[y = y^*]) W_y = \left( \sum_y p_y W_y \right) - W_{y^*} = \mathbb{E}_p[W_y] - W_{y^*}
$$

更新方向是其负数，即 $W_{y^*} - \mathbb{E}_p[W_y]$。

直觉：**hidden state 朝"真实词的行减去当前期望词的行"方向移动**。如果当前模型把概率均匀分布在多个词上，$h$ 就被强力拉向 $W_{y^*}$；如果模型已经几乎确定预测对了，$h$ 几乎不动。

### 4.3 在 Cascade-FullGrad（配置 B）下的 hidden state 拔河

同一个 $h_{\text{langact}}$ 同时承受两个梯度：

1. **$L_{lang}$ 通过 LM head**：$\partial L_{lang} / \partial h \propto \mathbb{E}_p[W_y] - W_{y^*}$
   - 把 $h$ 推向 "真实下一 token 的 LM head 行"
2. **$L_{action}$ 通过 cross-attention**：action expert 的 query 读 $h$ 作为 K/V，$h$ 被推向 "让 action expert 算出对的 v_t 的方向"

**协调**：如果真实 langact 文本和真实 action 强相关（"move gripper to leftmost block" 真的对应一个具体的 action），两个梯度方向**大致一致**——$h$ 既要预测对下一个语言 token 又要支持 action 计算。

**冲突**：如果 langact 文本里有 action 不需要的细节（如冗长描述），$L_{lang}$ 会把 $h$ 推向"还原文字"的方向，而 $L_{action}$ 推向"提取动作语义"的方向，两者拔河。极端情况下 $h$ 会变成"文字预测变差但 action 变好"或反之。

这也是为什么 [LAP 式 stop_action_to_vlm_grad] 是个保守选择：避免 $L_{action}$ 把 VLM 的语言能力拉走。但代价是放弃了"reasoning 真正成为 action 的瓶颈表征"的可能性。

---

## 5. 训练变体最终矩阵（v2，含 reasoning + langact 双层）

### 5.0 Prompt 结构

```
[BOS] [prompt] task_prompt
      [stage]  reasoning (Subgoal: 高层规划)
      [action] langact   (Step: 低层语言动作描述)
      [EOS] [PAD]...
```

例（来自 `data/arrange_blocks_line/demo_clean/metadata/episode0.json`）：
- `task_prompt`：Arrange the blocks into a horizontal line.
- `reasoning`（subgoal）：Place the red block at the leftmost slot of the line.
- `langact`（step）：Move the left gripper above the red block.

数据已现成可用，`cot.per_phase_text` 字段已预格式化为 `[Subgoal] ... [Step] ...`。

### 5.1 Action expert 的 attention 范围（统一约束）

**所有非基线 Variant 都遵循**：action 能 attend `image + prompt + langact`，但**屏蔽 reasoning**。
- 理由：reasoning 是"思考过程"的中间产物，langact 才是 action 的语言接口
- 实现：需要新增 `_build_prefix_action_mask` 的逻辑分支，按 token role 选择性屏蔽

### 5.2 Stop Gradient 的精细化（token 级）

现有 `stop_action_to_vlm_grad` 是 expert 级二元开关。要支持下表的 Variant 1/2/3 区分，需改造成 **token role-aware** 的版本：根据每个 prefix token 的角色（image / prompt / reasoning / langact）决定是否对其 K/V 施加 `stop_gradient`。

### 5.3 实验配置矩阵

| Variant | action attention | stop_gradient（cross-expert K/V）| langact 来源 | $L_{action}$ 影响范围 | 备注 |
|---------|-----------------|-------------------------------|-------------|--------------------|------|
| **0. LAP** | image + prompt（屏蔽 langact + reasoning） | image, prompt | TF | 仅 action expert | LAP 原版基线，reasoning 仅辅助任务 |
| **1. LAP-Unmask-Stop** | image + prompt + langact（屏蔽 reasoning） | image, prompt, reasoning, **langact** | TF / 自生成混合 | 仅 action expert | action 能读 langact，但 VLM 完全不被反向塑造 |
| **2. LAP-Unmask-Partial** | image + prompt + langact（屏蔽 reasoning） | image, prompt, reasoning（**不 stop langact**） | TF / 自生成混合 | action expert + langact 位置的 trunk hidden | 只塑造 langact 表征，不动 prompt/reasoning 上游 |
| **3. LAP-Unmask-Free** | image + prompt + langact（屏蔽 reasoning） | 无 | TF / 自生成混合 | 全 trunk + 全 embedding | Cascade full gradient |

### 5.4 关键 ablation 对比

| 对比 | 检验问题 |
|------|---------|
| **0 vs 1** | reasoning/langact 暴露给 action 是否有用？（无梯度回流时）→ 信息流入价值 |
| **1 vs 2/3** | 让 $L_{action}$ 反向塑造 VLM 是否有用？→ 梯度回流价值 |
| **2 vs 3** | 部分塑造（只 langact）vs 全塑造（含 prompt/reasoning）→ 哪种粒度最优 |
| **TF vs 混合** | 在每个 Variant 内做调度采样消融 → exposure bias 实际影响 |

### 5.5 Variant 2 的设计巧思

Variant 2 是这套设计的关键创新点。它的逻辑是：

- $L_{action}$ 应该让 langact hidden 变得对 action 有用 — 允许（不 stop）
- $L_{action}$ 不应该污染 prompt/reasoning 的语义表征 — 屏蔽（stop）
- 上游（image/prompt/reasoning）由 $L_{lang}$ 单独塑造，下游接口（langact）受双 loss 共塑

直觉上像是在 prompt/reasoning 和 action 之间放了一个"信息瓶颈层"——**langact 是这个瓶颈，唯一允许双向梯度流的位置**。

---

## 6. argmax + stop_gradient vs GT：训练时梯度流向有何不同

这是 Scheduled Sampling 实现的核心机制。下面把两种方式从梯度路径上彻底拆开。

### 6.1 两种 langact 来源的前向

**方式 A — Teacher Forcing (GT)**
```
GT_token_ids ─lookup─► W_emb[GT_ids] ─trunk─► H_langact ─attn─► action expert
                            ↑
                       embedding 表
```

**方式 B — 自生成 (argmax + stop_gradient)**
```
Phase 1: trunk 跑一次 → logits at langact 位置 → argmax → predicted_token_ids
                                                            │
                                                      stop_gradient
                                                            │
Phase 2: predicted_ids ─lookup─► W_emb[pred_ids] ─trunk─► H'_langact ─attn─► action expert
```

### 6.2 梯度路径对比

| 反向链路 | TF (GT) | 自生成 (argmax + stop_grad) |
|---------|---------|---------------------------|
| $L_{action} \to$ action expert 权重 | ✓ | ✓ |
| $L_{action} \to$ H_langact $\to$ trunk 层 (Variant 3) | ✓ 更新 trunk 权重 | ✓ 更新 trunk 权重（Phase 2 那次前向） |
| $L_{action} \to$ embedding 表 row | ✓ 更新 W_emb[GT_ids] 行 | ✓ 更新 W_emb[**predicted**_ids] 行 |
| $L_{action} \to$ Phase 1 LM head 的 logits | — | ❌ argmax 不可微 + stop_gradient 双重切断 |
| $L_{action} \to$ Phase 1 trunk（生成 reasoning 那次） | — | ❌ 同上，被切断 |
| $L_{lang} \to$ LM head + trunk + embedding | ✓（在 Phase 1 / 唯一前向上） | ✓（在 Phase 1 上） |

### 6.3 关键差异：argmax 本身就阻止梯度

**`argmax` 是非可微操作**（导数处处为 0 或未定义）。所以即使你不显式写 `stop_gradient`，反向传播也不可能通过 `predicted_ids = argmax(logits)` 这一步流回 logits。

那为什么还要写 `stop_gradient`？两个原因：
1. **明示性**：让代码读者一眼看出"这里梯度断了"
2. **避免数值意外**：某些 framework 的 `argmax` 自定义 backward 可能给一个"伪梯度"（如 straight-through），加 `stop_gradient` 强制截断

### 6.4 区别的本质：embedding 表更新的"哪一行"

- **TF**：每次更新 `W_emb` 的 GT 行 → 模型学到"对正确的 langact 表征做出正确的 action"
- **自生成**：每次更新 `W_emb` 的预测行（可能是错的）→ 模型学到"对自己产生的 langact 表征（包括错的）做出尽可能正确的 action"

**这是 exposure bias 的本质修复**：训练时让模型看到自己会犯的错误模式，并学会鲁棒应对。

### 6.5 一个微妙但重要的点：自生成训练**不直接改善 reasoning 质量**

很多人误以为 cascade + scheduled sampling 会让 $L_{action}$ 反过来"教 reasoning 怎么生成更对"。**事实并非如此**：

- 因为 `argmax + stop_gradient` 切断了从 $L_{action}$ 回到 Phase 1 LM head / Phase 1 trunk 的路径
- 所以 reasoning 的生成质量**仍然只由 $L_{lang}$（next-token CE）驱动**
- $L_{action}$ 在自生成模式下学到的是："**接受**模型已有的 reasoning（无论好坏），让 action 尽量做对"

如果你想让 $L_{action}$ 真的反向塑造 reasoning 生成，需要可微采样（Gumbel-softmax / straight-through estimator / REINFORCE）。但代价是显著更复杂、更不稳定。

**实务建议**：先做 argmax + stop_gradient 版本（简单 + 稳定），把它当作"鲁棒性正则化"而不是"reasoning 质量改进器"。

### 6.6 实现伪代码

```python
def langact_for_action_training(model, batch, p_self_gen, train_step, total_steps):
    # 调度: linear warmup
    p = min(p_self_gen, p_self_gen * train_step / (0.5 * total_steps))

    use_self = jax.random.bernoulli(rng, p, shape=(batch_size,))
    if use_self.any():
        # Phase 1: 自生成 langact (with KV cache, AR 解码)
        with jax.disable_jit():  # 或者用 lax.while_loop
            generated_ids = sample_langact_argmax(model, batch.prompt, batch.image)
        generated_ids = jax.lax.stop_gradient(generated_ids)
        langact_input = jnp.where(use_self[:, None], generated_ids, batch.gt_langact_ids)
    else:
        langact_input = batch.gt_langact_ids

    # Phase 2: 正常前向 + 双 loss 训练
    ...
```

---

## 7. 实现备忘录

### 7.1 修改 `_build_prefix_action_mask`：屏蔽 reasoning，保留 langact

```python
def _build_prefix_action_mask(self, prefix_mask, observation):
    """新行为：action 屏蔽 reasoning，但能 attend langact"""
    if observation.tokenized_stage_mask is None:
        return prefix_mask
    img_seq_len = prefix_mask.shape[1] - observation.tokenized_stage_mask.shape[1]
    reasoning_mask_full = jnp.concatenate([
        jnp.zeros((observation.tokenized_stage_mask.shape[0], img_seq_len), dtype=bool),
        observation.tokenized_stage_mask,
    ], axis=1)
    return jnp.logical_and(prefix_mask, jnp.logical_not(reasoning_mask_full))
```

注意：tokenizer 输出现在需要**两个 mask**：`tokenized_stage_mask`（[stage] 段）和 `tokenized_ar_target_mask`（[action] 段）。

### 7.2 改造 `stop_action_to_vlm_grad` 为 token-role 版本

[gemma.py:242-269](policy/lap/src/lap/models/backbones/gemma.py#L242-L269) 现在的 `cross_to_expert0` 是粗粒度的（"任何 expert ≥1 → expert 0 都断"）。需要改成：

```python
# 原: cross_to_expert0 = (q_owner != 0) & (k_owner == 0)
# 新: 加上 token role mask
# stop_grad_role_mask: (B, S) bool — True 表示该位置的 K/V 应被 detach
cross_to_expert0_stop = (q_owner != 0) & (k_owner == 0) & stop_grad_role_mask[:, None, None, None, :]
```

`stop_grad_role_mask` 由 config 决定：
- Variant 1: image + prompt + reasoning + langact 全 True
- Variant 2: image + prompt + reasoning True，langact False
- Variant 3: 全 False

### 7.3 Tokenizer 改造

在 [tokenizer.py](policy/lap/src/lap/models/tokenizer.py) 中需要：
- 解析 `[stage] ... [action] ...` 双段结构
- 输出 `reasoning_start/end` 和 `langact_start/end` 两组 index
- 派生 `reasoning_mask` 和 `langact_mask`（互斥）
- $L_{lang}$ 在两段都计算（reasoning 和 langact 都要 next-token CE 监督）

### 7.4 Scheduled Sampling 调度

```
p_ss(epoch) = clamp(epoch / 30 * 0.5, 0.0, 0.5)
```

epoch 0-30 线性 ramp，之后维持 0.5。

### 7.5 推理流程（cascade，单序列 + KV cache）

```python
# Phase 1: prefill [imgs, prompt]
kv_cache = trunk_prefill([img, prompt])

# Phase 2: AR 生成 reasoning + langact，token 一直写 cache
# 注意：reasoning 段也要生成（因为 L_lang 训过它），但 action 不会 attend 它
for step in range(max_cot_len):
    token = argmax(trunk_step(last_token, kv_cache))
    kv_cache.append(K_step, V_step)
    if token == EOS: break

# Phase 3: action diffusion，attend cache
# 注意构造 attention mask：屏蔽 cache 中 reasoning 段的位置
for t in range(num_diffusion_steps):
    x_t = trunk_step_action(x_t, time, kv_cache, action_attn_mask)
```

---

## 8. 待解决（先记着）

1. **LM head 与 input embedding tied 问题**：PaliGemma 是否 tied weights？如果 tied，$L_{action}$ 通过 embedding 行的更新会间接影响 LM head 输出层。需要查 [gemma.py](policy/lap/src/lap/models/backbones/gemma.py) 中 embedder 实现。
2. **Langact 长度方差**：若 batch 内 langact 长度差异大，cross-attention 有效 K/V 数变化 → 梯度尺度不一致。可能需要 layer norm 或长度归一化。
3. **Langact GT 抽样**：每个 phase 有多个 paraphrase（`phase_prompts[]` 4 个）。建议每个 epoch / 每个 batch 随机抽一个作 GT。可选：加 paraphrase consistency loss。
4. **Reasoning 段是否也要随机抽 paraphrase**：当前数据每个 phase 只有一个 `subgoal_prompt`，无变体。是否需要 LLM 生成 paraphrase 增强？
5. **推理时 reasoning 生成的成本**：reasoning 段是 AR 生成，会引入额外 latency。是否需要给 reasoning 设个 max_len 限制？

---

## 9. 优先级实现建议

按风险递增：

| 阶段 | 工作 | 预期收益 |
|------|------|---------|
| **P0** | Tokenizer 改造支持 `[stage]/[action]` 双段；attention mask 改为屏蔽 reasoning | 数据通路打通 |
| **P0** | 实现 Variant 0 (LAP baseline，不变) + Variant 2 (Partial) 两端对比 | 验证最关键的 hypothesis：langact 信息瓶颈是否有用 |
| **P1** | 实现 Variant 1 + Variant 3，做完整四点对比 | 隔离梯度流向各部分的贡献 |
| **P1** | 推理流程实现 + rollout gap 实验 | 验证 exposure bias 强度 |
| **P2** | Scheduled Sampling | 缓解 exposure bias |
| **P2** | Paraphrase consistency loss | 数据增强 |

---

## 10. DataLoader / Training / Inference 设计分析

针对 action chunking 与多 phase reasoning 时序的几个关键设计问题。以
`data/pick_place_primitive/demo_clean/metadata/episode0.json` 为参考（3 phases：
approach / grasp / transport，phase 长度 36 / 36 / **14** 帧——注意 transport
比典型 `action_horizon=16` 还短）。

### 10.1 一次性输出 vs 边动边 reasoning

```
方案 A — 一次性 plan-then-execute
  forward → [reasoning_全episode, langact_1, langact_2, ..., langact_N]
                                  ↓ 全部执行

方案 B — phase 级闭环 (Per-phase replanning)
  for each phase:
      forward(current_obs) → reasoning_i, langact_i
      execute(langact_i, until phase 结束)
```

| 维度 | A（开环） | B（闭环） |
|------|---------|---------|
| 训练复杂度 | 高（要预测全 chain） | 低（一次只预测一个 phase） |
| 推理复杂度 | 一次大 AR | 多次小 AR |
| 失败鲁棒性 | 差（中间出错没救） | 好（每 phase 都能修正） |
| 是否需要 image 更新 | 否 | 是（每 phase 用最新 obs） |
| 适用任务 | 静态规划任务（搭积木结构图固定） | 动态 / 接触富 / 失败可恢复任务 |

**推荐方案 B**。RoboTwin 的 phases 天然按 `start_frame/end_frame` 切片，
phase 边界明确；闭环架构更稳。pick_place_primitive 的 transport 只有 14 帧
（远小于一个 chunk），方案 A 的"全长 langact 序列"训练时长度方差太大，更容易爆。

**实现路径**：训练时每个样本对应**一个 phase 内的一个时刻**：
- prompt = task_prompt
- reasoning = 该 phase 的 subgoal_prompt（注：pick_place_primitive 数据没有
  subgoal_prompt，需要从 `task_spec.pick.descriptor` / `task_spec.place.relation`
  组合生成；arrange_blocks_line 已有 `subgoal_prompt`）
- langact = 该 phase `phase_prompts[]` 的随机一条（多变体增强）
- action = 从该帧起的 `action_horizon` 长度 chunk

### 10.2 是否 condition 之前的 reasoning 历史

```
方案 B1 — Memoryless（每 phase 重新 prompt）
  prompt = task_prompt
  reasoning = current_phase_subgoal
  → 模型看不到之前 phase 的 reasoning

方案 B2 — Conversational history
  prompt = task_prompt + [完成的 phases 历史]
  reasoning = current_phase_subgoal
  → 模型看到全历史
```

| 维度 | B1 | B2 |
|------|-----|-----|
| 序列长度 | 短，固定 | 增长，方差大 |
| 信息冗余 | 无 | 多（image 已编码状态） |
| 长期一致性 | 弱 | 强 |
| 训练成本 | 低 | 高（attention $O(L^2)$）|

**推荐 B1（memoryless）**，理由：
- **图像本身就编码了"已完成 phase 的物理后果"**（红块已在左槽）
- pick/place/transport/lift_down 这类原语本质是"看图说话"，每 phase 独立
- 多 phase 累积会让 KV cache 膨胀，不利于推理实时性

例外：如果任务需要"记住未表达的意图"（例如"把红色和黄色互换位置"，中间状态对图像
ambiguous），可以加一个**简短的 phase-history 摘要 token** 而不是整段历史。

### 10.3 action chunk 短于 phase 长度怎么办

这是最棘手的工程问题。pick_place_primitive 的 transport phase = 14 帧，
若 `action_horizon=16` 则**任何一个 chunk 都会跨越 phase 边界**。

**几种处理方式：**

| 方案 | 描述 | 问题 |
|------|------|------|
| (a) **截断** | 只用完全落在单 phase 内的 chunk | 丢失 phase 末尾训练数据；transport=14 时根本没数据 |
| (b) **末帧填充** | 用 chunk 内最后一帧的 action 重复填充 | ⚠️ **正是你担心的"模型学到停顿"** —— 强烈不推荐 |
| (c) **缩小 horizon** | 把 `action_horizon` 设为最短 phase 的长度 | 损失长视野规划能力 |
| (d) **跨 phase chunk** ⭐ | 允许 chunk 跨 phase；langact 取 chunk 起始帧的 phase langact | 标准做法（OpenVLA / pi0 均如此） |
| (e) **动作 mask** | chunk 内每帧附带 `valid_mask`，超出 phase 的帧 loss=0 | 训练时不强迫预测，但浪费 capacity |

**推荐 (d) + (e) 组合**：
- 默认允许 chunk 跨 phase
- chunk 起始帧的 phase 决定 reasoning + langact 内容
- chunk 内最后几帧若进入下一 phase，**仍然计算 action loss**（因为 GT 动作是真实的、
  仿真器记录的、平滑过渡的）—— 这教模型学到"phase 之间的衔接"，是 feature 不是 bug
- 仅在 chunk 完全超出 episode 末尾时用 mask 屏蔽 loss

**为什么 (b) 末帧填充会让模型学到停顿**：
- 末帧 action 通常是 phase 末尾的"稳定姿态"（夹爪刚 close 完）
- 反复填充让模型把这个姿态学成"该 langact 完成后默认行为"
- 推理时模型在执行完 phase 后会被这个 prior 拉回稳定姿态，**不会自动衔接下一 phase**
- 若再加上"下一次自回归生成新 langact"的延迟，会出现明显的"卡顿"

**工程检查**：在 dataloader 里加个 assert：每个 chunk 的真实 GT action 帧
都在 episode 范围内（使用 hdf5 的 actions 数组实际长度），不要做任何"末帧重复"。

### 10.4 推理时何时触发新一轮 AR 生成 langact

四种典型策略：

```
(I) Per-step regeneration
    每帧都 AR 一次 langact，看是否变化
    → 计算量大，但最响应

(II) Per-chunk regeneration  ⭐ 推荐起步
    每个 action chunk 执行完后 AR 一次
    → 自然节拍，与训练数据对齐

(III) Phase-end trigger
    用 hand-engineered signal（夹爪状态、TCP 速度归零）触发
    → 需要每任务调参，不通用

(IV) Model-emitted [next] token
    模型在 langact 末尾追加特殊 token，模型自己宣告"该 langact 完成"
    → 优雅但需要 GT 标注切换时刻
```

**推荐流程（方案 II）**：
```python
while not done:
    obs = env.observe()                              # 当前 image
    # AR 生成 reasoning + langact + 第一个 action chunk
    reasoning, langact = model.sample_tokens(obs)
    action_chunk = model.sample_actions(obs, reasoning, langact)  # H 帧
    for a in action_chunk:
        env.step(a)
    # loop back, regenerate at next iteration
```

### 10.5 你提的核心问题：训练时 langact 没执行完，AR 目标是什么？

**仔细想会发现这其实不是问题**——下面是关键观察：

#### 观察一：训练样本是 (frame_t, langact_for_phase_containing_t)

如果 phase k 占据 `[start_k, end_k)` 帧，那么对**这 phase 内的每一帧 t**，训练数据
都会标注同一个 langact。所以 dataloader 输出：

```
sample_at_frame_t:
    image     = obs[t]
    reasoning = subgoal[phase_of(t)]
    langact   = langact[phase_of(t)]   # 注意：t 取 phase 内任意值时这个都一样
    action    = actions[t : t+H]
```

#### 观察二：模型学到的是"给定当前观察，emit 该 phase 的 langact"

由于训练目标在 phase 内是恒定的，模型自然学到：
- 在 phase 进行中的各种中间观察 → 都该 emit 同一个 langact
- 一旦观察跨入下一 phase（比如夹爪已经合上） → 该 emit 新 langact

**所以 AR 的目标既不是 padding 也不是 [keep_going]，就是当前 phase 对应的 langact 字符串本身（再 emit 一次）**。

#### 观察三：你担心的"模型还没执行完就再 AR"——其实是 feature

推理时如果在 phase 中途触发 AR：
- 模型看的图还在 phase 内
- 模型 emit 的 langact 应该和上次相同（因为训练它就是这么学的）
- **这意味着模型自然支持"幂等地确认 langact"**

如果模型 emit 了**不同**的 langact，恰恰说明它认为该切换 phase 了——这是一个
免费的 phase 切换信号。

#### VLM 领域的类比

类似的问题在 VLM 视频理解里被称为 **"action segmentation 的边界识别"**。常见做法：
- 训练时 frame-wise 标注 → 模型自然连续输出同一标签直到边界
- 推理时按 frame 级输出 + 后处理（去抖动、smoothing）

不需要特殊 token（如 `[keep_going]` 或 `[done]`），除非你想**缩短训练序列**
（"只在 phase 边界 emit"）—— 但这要求标注每个时刻"是否 phase boundary"，更复杂。

### 10.6 完整的 DataLoader 输出 schema 提案

```python
{
    # === 已有字段 ===
    "image": (3, 224, 224),            # current frame
    "state": (action_dim,),
    "actions": (action_horizon, action_dim),

    # === 文本 ===
    "prompt": str,                     # task_prompt 例 "Pick the red cube..."
    "language_actions": str,           # subgoal_prompt（reasoning）
    "langact": str,                    # 一条随机 phase_prompts paraphrase

    # === Loss 控制 ===
    "action_valid_mask": (action_horizon,) bool,  # episode 结尾被截断的位置 = False
    "is_vqa_sample": bool, "is_prediction_sample": bool, "sample_mask": bool,
}
```

`langact` 通过 `transforms.TokenizePromptAndReasoning` 自动激活双段 tokenizer 路径，
产出 `tokenized_stage_mask` 和 `tokenized_ar_target_mask`。

### 10.7 推理流程伪代码

```python
def cascade_rollout(env, model, max_steps=500, max_phases=20):
    obs = env.reset()
    done = False
    last_langact = None

    while not done and step_count < max_steps:
        # Phase 1: AR 生成 reasoning + langact
        reasoning, langact = model.sample_tokens(
            image=obs["image"], prompt=task_prompt,
            max_decoding_steps=64,
        )

        # 可选：检测 langact 是否变化（"phase 切换"信号）
        phase_changed = (langact != last_langact)
        last_langact = langact
        # 用于 logging/debug

        # Phase 2: 扩散生成 action chunk（在已有 KV cache 上继续）
        action_chunk = model.sample_actions(
            image=obs["image"],
            reasoning=reasoning,
            langact=langact,
            num_steps=10,
        )

        # Phase 3: 执行 chunk
        for a in action_chunk:
            obs, _, done, _ = env.step(a)
            if done:
                break
```

可选的工程优化：
- **保留 KV cache 跨 chunk**：如果连续两次 langact 相同，prefix 的 KV 不变，
  只需重新跑 action 扩散，不必重新生成 reasoning。
- **异步 AR**：在执行 action chunk 时同时 AR 下一个 langact，掩盖 reasoning latency。

### 10.8 实现的 P0 工作清单

| # | 任务 | 备注 |
|---|------|------|
| 1 | DataLoader：增加按 phase 索引帧的逻辑 | 扫描 metadata 的 phases 数组，生成 `(frame, phase_id)` 映射 |
| 2 | DataLoader：从 `phase_prompts[]` 随机采一条作为 `langact` | 多变体增强 |
| 3 | DataLoader：从 `subgoal_prompt` 取 `language_actions`（reasoning） | arrange_blocks 已有；pick_place_primitive 需要从 task_spec 合成 |
| 4 | DataLoader：跨 phase chunk 的 `action_valid_mask` 处理 | 仅 episode 结尾的越界位置 mask=False |
| 5 | 推理脚本：实现 per-chunk 闭环 rollout | 参考 10.7 伪代码 |
| 6 | 推理脚本：langact 变化日志 | 监控"phase 切换"频率，验证模型行为 |

### 10.9 待解决的设计问题

1. **pick_place_primitive 没有 subgoal_prompt 字段** —— 需要从 `task_spec.pick`
   和 `task_spec.place` 合成。建议格式：
   `"Pick {pick.descriptor}, then place it {place.relation_at_descriptor}."`
2. **多 phase 的 subgoal 是否共享一个 reasoning？** 比如 approach/grasp 都属于
   "pick {color} cube" 阶段，是否合并为同一 reasoning，让 langact 区分
   approach vs grasp？
3. **失败 phase 是否纳入训练？** episode0 的 transport `success: false,
   failure_reason: "plan_fail"` —— 这种数据训练 action 会引入坏 demo，
   但训练 reasoning 反而可能学到"该说 transport 但失败了"的有用信号。
   建议：失败 phase 的 action loss 屏蔽，但 langact loss 保留。
4. **action chunk 跨 phase 时，langact 该用 chunk 起始帧的还是中点的？** 起始帧
   是最自然选择（模型看的就是起始帧的 obs），但 chunk 末段执行的实际是下一 phase
   的动作 —— 模型可能会被这种"前后不一致"的对应训糊。可消融。
