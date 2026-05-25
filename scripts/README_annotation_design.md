# CoT 标注设计规范讨论

> 这是一个**设计论证 doc**，不是接口规范。它讨论 "stage / action / think 各字段语义应该是什么"、"标注是不是 Markovian"、"CoT 长度对推理成本的影响" 等开放问题，最后落到具体的推荐规范。

实现细节参考：
- 手动标注 GUI: [`scripts/annotate_teleop_episodes.py`](../../../scripts/annotate_teleop_episodes.py)
- VLM 自动标注流程 (DROID): [`README_droid_annotation.md`](README_droid_annotation.md)
- VLM 部署 (teleop): [`README_qwen_vlm_for_teleop_annotation.md`](README_qwen_vlm_for_teleop_annotation.md)
- pick-and-place 术语表: [`TERMS_pick_and_place.md`](TERMS_pick_and_place.md)

---

## 1. 现状对比：3 种 `stage` 风格

同一个场景（机械臂放下方块），三种标注风格：

| 风格 | 例子 (stage 字段) | 语义 |
|---|---|---|
| **A. 观测中心** (MimoVL / 商业 VLM) | _"Gripper opens to release any held object near the pot."_ | 描述这帧**看见**什么 |
| **B. 目标中心** (用户原始手标) | _"Stack the rightmost cyan cube on the orange cube behind it"_ | 描述**意图 / 子任务** |
| **C. 混合 + 解释** (Claude 修订版 / ECoT) | _"The gripper opens and the cyan cube is placed on top of the rightmost red cube; the gripper then lifts away."_ | **状态 + 因果**，含动作的语义闭包 |

### 为什么差这么大？三种风格背后的训练目标不同

| 风格 | 训练目标 | 适合下游 |
|---|---|---|
| A | VLM "看图说话" / VQA → 提升视觉 grounding | 通用 VLM 微调；caption / grounding 评测 |
| B | 子任务规划 / hierarchical RL 监督 | 高层规划 policy；options framework |
| C | full Embodied-CoT → 教模型 chain reasoning | VLA cascade 推理；OpenVLA / ECoT-style 模型 |

**没有绝对最佳**。选哪种取决于**你打算用这套数据训练哪种 policy**。

---

## 2. 字段语义讨论：现在该怎么定义？

具体到 LAP 项目，我们要训的是 **cascade-VLA**（高频 action policy + 低频 reasoning），那么：

### 2.1 `plan` —— 整集只填一次的高层全局规划

✅ 已达共识：2-5 句，数字化 sub-goals，整 episode 共享。

```text
"1. Pick up the nearest cyan cube. 2. Stack it on the rightmost red cube.
 3. Repeat for two more pairs."
```

### 2.2 `stage` —— 段级（多帧共享），描述**当前发生的事**

**推荐定义**：以"The robot ..."开头的**陈述句**，描述这段帧里机械臂正在做什么 + (可选) 一句话点明这是对应 plan 的哪一步。

**长度目标**：15-30 词（1-2 句）。

为什么这样定义？
- **可 ground 性**：陈述当前状态，可以直接用 VQA loss 训练 VLM grounding
- **段内一致**：一个段内每帧 stage 相同（人不会每帧改写）
- **避免 plan 复读**：不要直接复制 plan 字段；要用 plan 的子句加 "this is the X step" 收尾
- **半 Markovian**：只引用本段视觉 + 全局 plan，不引用前段细节 → 训练时可以随机采样段

✅ 推荐写法：
```text
"The gripper closes firmly around the rightmost cyan cube. This is the
 grasp step of the second pair."
```

❌ 不要这样写：
```text
"Stack the rightmost cyan cube on the orange cube behind it"
```
（这是 plan 复读，没有描述本段在做什么）

❌ 也不要这样写：
```text
"As mentioned in the previous keyframe, after grasping the cube, the
 robot now ..."
```
（引用前段 → 破坏 Markovian 性 → 训练时随机采样这段就 broken）

### 2.3 `action` —— 帧级（段头），描述**下一步要做什么**

✅ 已达共识：≤12 词，祈使句。

`action` 是 policy 的**直接监督信号**。stage 是观测描述，action 是 next step。两者分工：
- stage 回答 "where are we?"
- action 回答 "what to do next?"

### 2.4 `think` —— 可选，因果推理

✅ 已达共识：仅在 retry / 失败 / 反直觉决策时填。1-2 句，说明**原因 + 修正方向**。

### 2.5 各字段的"时间锚点"

| 字段 | 时间含义 |
|---|---|
| `task_instruction` | 整集 const |
| `plan` | 整集 const |
| `stage` | 段内 const，**描述本段(state)** |
| `gripper_state` | 段末 state |
| `action` | 段内意图：**接下来要做什么**（next-step intent） |
| `think` | 段头反思：**为什么走到这一步**（causal） |

### 2.6 用户给的例子分析（playground episode0 frame 133）

用户给出三种可能：

| 维度 | 用户表述 | 应放哪个字段 |
|---|---|---|
| **obs** | 夹爪悬挂在上方正准备下降去抓 | → `stage`（陈述本段状态）|
| **goal (近)** | 在方块边关闭夹爪并 lift up | → `action`（next-step intent） |
| **goal (远)** | Stack the rightmost cyan cube on the orange cube behind it | → `plan`（整集 const，本段不复读）|
| **action** | 在方块边关闭夹爪并 lift up | → `action` |

推荐填法：
```json
{
  "frame_start": 133,
  "frame_end": 147,
  "type": "grasp",
  "gripper_state": "closed",
  "stage": "The gripper hovers above the rightmost cyan cube with its jaws aligned with the cube's edges, ready to descend and close.",
  "think": null,
  "action": "Lower onto the cube, close the gripper, and lift"
}
```

### 2.7 帧 147 同样分析

| 维度 | 用户 |
|---|---|
| obs | 夹爪已经抓起了 cube，再往前就是目标的橙色方块 |
| action | 把方块 carry 到橙色方块上方 |

推荐填法：
```json
{
  "frame_start": 147,
  "frame_end": 165,
  "type": "motion",
  "gripper_state": "closed",
  "stage": "The robot has the cyan cube secured in its gripper and is positioned just before the target orange cube.",
  "think": null,
  "action": "Carry the cube above the orange cube"
}
```

---

## 3. 标注是不是 Markovian？

### 3.1 严格定义

**Markov 性**：当前 step 的输出只依赖**当前状态**，不依赖历史。

对 VLA 来说：
- 输入：`(image_t, instruction, plan, stage_t)`
- 输出：`action_t`

**如果 stage_t 只描述 image_t 看到的状态**，那 (image_t, plan) → stage_t 是 Markovian 的。
**如果 stage_t 包含"continuing from previous step"**，就不是。

### 3.2 为什么要 Markovian？

**训练时**：随机采样段进 batch。如果 stage_t 依赖 stage_{t-1}，单段采进 batch 就 broken：没有上下文。

**推理时**：cascade 缓存。如果 stage 是 Markovian 的，可以独立预测；非 Markovian 则必须按顺序生成。

### 3.3 当前推荐设计的 Markov 性

| 字段 | Markovian? | 为什么 |
|---|---|---|
| `plan` | ✅（全局 const） | 不依赖时刻 |
| `stage` | ✅（按本规范写）| 只引用本段视觉 + 全局 plan |
| `action` | ✅ | 只描述 next step intent |
| `think` (retry) | ❌ | **明确引用过去**（"previous attempt failed"）|
| `gripper_state` | ✅（段末瞬时态）| 只看本段 |

`think` 是唯一非 Markovian 字段。这是 feature：retry 类的语义本就是"对过去的反思"。

训练上的处理：**think 不出现在 grasp/release/motion 段**（默认 null），只出现在 retry 段。所以非 Markov 性只在 retry 上引入，约占 keyframe 的 5-15%。

---

## 4. CoT 长度 / 推理成本

### 4.1 ECoT 论文实测

| 项 | 数字 |
|---|---|
| ECoT 全链平均输出 token | ~400 token / 帧 |
| 推理时间增加 (vs vanilla VLA) | 5-10× |
| LAP 论文 cascade 平均 | ~100 token / 帧（plan 不重复）|

**核心结论**：CoT 越长，推理越慢。每帧重复 plan / stage 是巨大浪费。

### 4.2 cascade 策略

按变化频率分层：

| 字段 | 推理频率 | 平均 token |
|---|---|---|
| `task_instruction` | 整集 1 次 | ~20 |
| `plan` | 整集 1 次 | ~100 |
| `stage` | 段 1 次（5-10 帧一次）| ~30 |
| `action` | 帧 1 次 | ~15 |
| `think` | 仅 retry，5-15% 频率 | ~50 |

**推荐 inference 路径**（cascade-VLA / LAP）：
1. episode start：plan + task_instruction 一次，全程缓存
2. 进入新段 (与上段类型变化)：重新预测 stage
3. 每帧：image + plan + stage 缓存 → 预测 action
4. 检测到 retry-like 状态：预测 think + 触发 stage 重生

这是 LAP 论文的 "cascade" 思路。比 ECoT 全链每帧重新生成快 5-10×。

### 4.3 反向作用于标注规范

为了让 cascade 可行：
- ✅ stage 只在**段头一次** (本规范已满足)
- ✅ action 短而具体 (本规范 ≤12 词)
- ✅ think 默认 null (本规范已满足)
- ⚠️ **不要在 stage 里复述 plan** —— 增加 token，cascade 没收益
- ⚠️ **不要在 action 里描述 state** —— 重复 stage 的信息

### 4.4 长度参考表

| 字段 | 推荐字符数 | 推荐词数 |
|---|---|---|
| `plan` | 300-500 chars | 50-80 words |
| `stage` | 80-200 chars | 15-30 words |
| `action` | 30-80 chars | **5-12 words** |
| `think` | 100-250 chars | 20-40 words |

超过上限：训练 batch token budget 容易爆 + 推理慢。

---

## 5. 数据质量 tag

### 5.1 用户场景

playground episode0 frames 82-99：操作员调位置 / 卡壳 / 误操作 — **数据本身没错，但对训练 policy 是 noise**。

当前 schema 有 `type=filler` 表示"长 segment 间的填充"。语义不完全匹配 — filler 含义偏中性"无聊但 valid"，用户想要的是"质量低，训练时跳过"。

### 5.2 建议加一个 quality flag

加 KeyframeAnno 字段 `quality: str` ∈ {`clean`, `noisy`, `corrupted`}：

| 值 | 含义 | 训练 sampler 行为 |
|---|---|---|
| `clean` (default) | 正常标注 | 正常 sample |
| `noisy` | 数据采集有质量问题 (操作员卡顿、轻微误操作但任务完成) | **降采样**（保留极少量做 robust 训练）或**跳过** |
| `corrupted` | 数据错误 (sim crash 帧、传感器异常、抓取完全失败但用户没 retry 也没 discard) | **永远跳过** |

实现要点（待做）：
- GUI 里加一个 quality dropdown（默认 clean）
- 训练 dataset class 里 filter `quality != "corrupted"`，并对 `quality == "noisy"` 降权
- audit 流程：标注 `quality=noisy` 的段不强制 think / action 完整性，仅警告

### 5.3 与 `type=filler` 的区别

| | type | quality |
|---|---|---|
| 语义维度 | 动作语义 (begin/grasp/release/...) | 数据质量 (clean/noisy/corrupted) |
| 是否影响训练 sample | 否 (filler 也参与训练) | 是 (filter 掉 corrupted, 降权 noisy) |
| 默认值 | 不能默认 (必填) | clean |

两者**正交**，可以同时存在。例如：`type=motion + quality=noisy` 表示"有效的运动段，但操作员调位置很烂，训练别太看重"。

---

## 6. 训练 / 推理流程对标注的反向影响

### 6.1 训练目标决定 stage 风格

| 目标 | stage 应该是 |
|---|---|
| **VLM grounding 微调** (caption / VQA) | 纯描述 (Option A) |
| **Hierarchical RL 监督** | sub-task label (Option B) |
| **Cascade-VLA (LAP)** | 描述 + 简短意图 (Option C/D，本规范推荐) |
| **OpenVLA-ECoT-style** | full chain (Option C，含 reasoning) |

**LAP 项目用 Cascade-VLA**，所以本规范推荐的混合风格是对的。

### 6.2 推理时间决定 stage 长度

| 推理预算 | stage 长度 |
|---|---|
| 实时 (>20Hz) | ≤ 20 词 |
| 准实时 (5-20Hz) | ≤ 30 词 ← 本规范 |
| 离线 / batched | ≤ 60 词 |

实时 cascade 用 vLLM 这种 throughput-optimized 推理，30 词的 stage 段头预测 ~30ms (Qwen-VL-7B)，可接受。

### 6.3 数据集规模决定标注严格度

| 规模 | 严格度 |
|---|---|
| < 100 集 | 人工严标 + 多次审阅 |
| 100-1000 集 (我们当前) | 人 selects + VLM auto + 人 review 关键段 ← 当前路径 |
| > 10000 集 | VLM full auto + spot check + audit drop |

---

## 7. 推荐规范（落地）

汇总以上讨论，下面是手标 + VLM 自动标都遵守的规范：

### 7.1 必填字段

```json
{
  "task_instruction": "<20-50 words>",
  "plan": "<50-80 words, numbered sub-goals>",
  "keyframes": [
    {
      "frame_start": int,
      "frame_end": int,
      "frame_idx": int,                  // = frame_start
      "type": "begin|grasp|release|retry|motion|filler|end",
      "gripper_state": "open|partial|closed",
      "stage": "<15-30 words, declarative, describes THIS segment's state>",
      "think": null,                     // or "<20-40 words>" for retry
      "action": "<5-12 words, imperative, NEXT step intent>"
    }
  ]
}
```

### 7.2 推荐增量字段（待加）

```json
{
  "keyframes": [
    {
      ...,
      "quality": "clean"   // | "noisy" | "corrupted"
    }
  ]
}
```

### 7.3 写法 do / don't

**stage**

| ✅ DO | ❌ DON'T |
|---|---|
| "The gripper closes around the cyan cube and lifts it off the table." | "Stack the cyan cube on the orange cube" (这是 plan) |
| "The robot hovers above the orange cube, ready to release." | "Continuing from the previous step, ..." (非 Markovian) |
| 1-2 句 | 5+ 句 |

**action**

| ✅ DO | ❌ DON'T |
|---|---|
| "Close the gripper to grasp the cyan cube" | "The robot is now closing the gripper" (这是 stage, 陈述句) |
| "Lift and arc over the middle cube" | "Move forward 5cm then lift 3cm" (过度具体) |
| ≤ 12 词 | 20+ 词 |

**think**

| ✅ DO | ❌ DON'T |
|---|---|
| "The previous grasp closed empty; the gripper undershot the cube's centre. Realign and retry." | (在非 retry 段填) |
| 失败原因 + 修正方向 | "The robot is now trying again" (没信息) |

### 7.4 物体 / 朝向命名

详见 [`TERMS_pick_and_place.md`](TERMS_pick_and_place.md)。要点：

- 物体：`<color> <type>` (e.g. `cyan cube`, `leftmost red cube`)
- 朝向：先轴名后方向 (`yaw counterclockwise`, `tilt up`)
- 避障：`clear the [obstacle]`, `arc around the [obstacle]`

---

## 8. 与 ECoT 论文的对照

| 维度 | ECoT 论文 | 本规范 |
|---|---|---|
| 字段数 | 8 (TASK / PLAN / SUBTASK_REASONING / SUBTASK / MOVE_REASONING / MOVE / GRIPPER POS / VISIBLE OBJECTS) | 5 (type / gripper_state / stage / think / action) |
| 触发节奏 | 每帧全 8 段 | 每段 stage 一次，每帧 action 一次 |
| GRIPPER POS / VISIBLE OBJECTS | 必有 (核心 grounding) | 暂无 (v2 加 GroundingDINO bbox) |
| MOVE 词汇 | 729 模板 + 实际 54 种 | 自由 (≤12 词) |
| 失败语义 | 无显式 | `think` + `type=retry` |
| 输出格式 | XML-tag dict | JSON |
| 推理成本 | 5-10× vanilla | 1.5-3× vanilla (cascade) |

我们的精简版本是为 LAP cascade 优化的，**牺牲 ECoT 的 GRIPPER POS 精确 grounding 换 token 减少**。如果 grounding 不够强可以 v2 加 bbox。

---

## 9. 开放问题 / 待决策

1. **`stage` 是否需要细分"obs vs intent"**：可以加 `stage_obs` + `stage_intent` 双字段，但增加标注成本。**当前推荐：合并到一个 stage**（既描述 state 又点明 sub-goal），如训出来 grounding 弱再拆
2. **`quality` flag 加不加**：用户提议，**强建议加**。下次手标 GUI 升级时加 dropdown
3. **`stage` 是不是真的需要每段重新写**：相邻段可能 stage 一样（都在 "approach phase"）。如果允许 dedup（多个段共享同一 stage 字符串），可以节省人力 + 增强 cascade 缓存效率。**待 pilot 100 集后看相邻 stage 重复率**
4. **是否引入 GROUNDING_OBJECTS**：参 ECoT。VLM 输出每帧检测到的物体 bbox 列表。**LAP cascade 用不太到，可放 v2**
5. **think 是否扩展到非 retry 段**：例如 "为什么选这个朝向" / "为什么先抓左边的方块"。**可选 enrichment**，对训练有 marginal gain 但显著增 token

---

## 10. 落地 action items

- [ ] **更新手标 GUI 字段说明**：把本 doc §2 / §7.3 的 do/don't 加进 GUI 的字段 placeholder 文本
- [ ] **加 quality dropdown**：GUI + schema 升级；训练 dataset filter
- [ ] **mimo 标注转 LAP 风格脚本**：mimo 的 stage 是纯 obs，需要拼接 plan 上下文 → 用 Qwen-VL 跑一次 rewrite (input: 原 stage + plan, output: 混合风格 stage)
- [ ] **人标 30 集后**回头审计：stage 在相邻段的重复率？action 是否真的 ≤12 词？think 出现在多少 %？据此调整 §7 规范
- [ ] **本 doc 跟 VLM prompt 同步**：[`README_qwen_vlm_for_teleop_annotation.md`](README_qwen_vlm_for_teleop_annotation.md) 的 system prompt 应该 reference 本 doc §7 而不是各自维护一套规则
