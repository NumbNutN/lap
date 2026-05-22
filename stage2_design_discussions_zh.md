# Stage 2 关键设计讨论（中文）

> 这个文档记录 Stage 2 推理 / 训练管线里几个**长期需要保持注意力**的设计权衡。每次涉及到这些点的修改，回来读一下相应小节，确认不要破坏既定决策或忘掉未决项。
>
> 配套文档：
> - `stage2_sim_eval_diagnosis.md` ：6 个 mitigation 方向 (A/B/C/D/E/F) 的细节
> - `cascade-bridge-pretraining-discussion.md` ：Stage 1 cascade 预训练讨论
>
> 状态图例：✅ 已实现 / ⚠️ 部分实现 / ❌ 未实现 / 🔬 待实验

---

## 0. 决策快照（2026-05-12 讨论后）

| 问题 | 短期决策（不重训） | 长期决策（要重训） |
|---|---|---|
| **1. last_arm stickiness** | ✅ 客户端解析 `[action]` 段；首帧用默认 wrist（任意，仅一帧）；arm-switch 时降级 exec_horizon=1 一步 | ⏸ 暂不动训练侧（per-phase 切已能用） |
| **2. cascade 重生** | ✅ 方案 A：Plan 首帧 cache，stage/action 每次重生 | 🎯 **C2 + stage_completion_reasoning**：训练时构造"两 stage + 完成 reasoning + 新 stage"样本，模型自学何时切 stage（要重 Stage 1） |
| **3. teacher-forcing gap** | ⏸ 暂缓 | ⏸ 等 C2 + A 结果再说 |
| **4. overlay 排版** | ✅ 多行 + 字号 0.32 | — |
| **5. 推理 profiling** | ✅ AR/flow 分项 + tokens 计数 | — |

---

## 0.A. 2026-05-13 关键诊断升级：脆弱状态机 + grasp gripper 不闭合

### 0.A.1 数据本质 = 脆弱状态机

用户 sharp 诊断：当前所有 demo_clean 数据由 **motion-planning 算法生成**，每个 task 被拆解为 `approach → grasp → transport → put`。其中：

- **每个 phase 的轨迹是 deterministic 最优解**（TOPP planner 给定起止 qpos 输出固定路径）
- **Grasp 阶段尤其单一**：末端垂直下落、夹爪从 1.0 闭合到 0.0
- 整个 task **像一个 4-state FSM**，每个 state 的"邻域"（covariance）极窄

→ 训练分布 = 一组**很瘦的** trajectory tube。任何执行偏离 → 立刻出 tube → action expert 不知道怎么走 → 你视频里看到的"定在原地"

### 0.A.2 实证：模型推理时夹爪几乎不闭合

对比 demo_clean expert 和 `qpos_noise1_step30000` combo eval 三个 episode 的 gripper 值：

| 数据源 | 夹爪 < 0.5 帧占比（L / R）| 平均（L / R）|
|---|---|---|
| **专家 demo (ep0)** | 23% / 21% | 0.77 / 0.79 |
| **模型 ep0 / ep1 / ep2** | 0% / 0-1% | **0.99 / 0.99** |

**模型推理时夹爪几乎从不闭合**，但 dry-run（专家轨迹精确 state 上）的 gripper 维度 `pred_std/gt_std ≈ 1.0` 没有 mode collapse → 模型有能力闭合，**只是推理时永远到不了"专家闭合时的精确 state"**。

### 0.A.3 间歇恢复模式（用户观察）

- 夹爪偏左 1cm 悬在方块上 → 不动了（OOD）
- 偶然晃到方块正上方 → cascade 立刻变成 "Squeeze the fingers of the right gripper..." → 但 action expert 执行不完美又漂出去 → 循环

这个观察支持："**covariate shift 瓶颈具体在 action expert，不在 VLM**"。VLM 看图给出的 reasoning 是稳定的（只看图，不看 state），action expert 看 state 主导 → 它才是受 shift 拖累的部件。

### 0.A.4 解决方案优先级（重排）

| 方案 | 工程量 | 预期 | 备注 |
|---|---|---|---|
| **遥操作数据混入** | 大（要 hardware + 标注） | 高 | 业界公认；多样性天然，覆盖 motion-planner 进不去的 state |
| **DAgger** | 中-大（脚本+迭代） | 高 | 标准解；我们有 expert 可用 (mplib) |
| **Atomic skill data + cascade-decoupling** ★ | 大但有新意 | 中-高 | 见 §0.A.5 |
| **Error-correction reasoning** | 中 | 中 | 需要失败数据，依赖前两项 |
| **state-noise σ=0.1** | 极小 | 低（"轻量 DAgger"） | 用户认为没新意，证实过 0.02 不够 |
| **客户端 gripper override** | 极小（救火） | 低 | 不治本 |

### 0.A.5 用户提议：Atomic skill data + cascade-decoupling（★ 值得专门挖）

**思路**：训练时混入 **task-agnostic motion primitive** 数据，如 "move left gripper 5cm in +x"。cascade 文本指代具体 motion 量级（"the gripper is at the left side of the block, so move 5cm right"），action expert 学到的是 "看到 cascade 提到 X 量级移动，就移 X"。

**为什么有意思**：
1. **解耦**：高层 VLM 出 motion 级 reasoning + 低层 action expert 执行 primitive → hierarchical-VLA
2. **State coverage 高**：5cm-right primitive 在**任何起始 state** 都有效 → 训练时天然采集多样 state
3. **新颖度**：当前 cascade-VLA 工作（π0 / RT-2）的 reasoning 都在"语义级"（pick red cube），把 reasoning 拉到 "motion primitive 级" 没人做过
4. **直击当前瓶颈**：当前 action expert 学的是 "given state s in approach phase, output 8 future qpos" — state-specific 才是脆弱根源；改成 primitive-specific 直接绕过

**实施粗框架**：
- 数据采集：用 motion-planner 在 sim 里生成简单原子动作（move +x/−x/+y/.../rotate wrist/.../open-close gripper）的 short trajectories，起始 state 随机化
- Cascade text 用模板：`"[action] move the {arm} gripper {dx}cm in the {direction} direction."`
- 训练时把这些 episodes 混入 demo_clean，权重 0.2-0.5 之间

**风险**：
- 单 episode 长度短（一个 primitive 几帧）→ action_horizon=8 可能跨多个 primitive；要么短 horizon，要么 padding
- Cascade 怎么 emit "5cm" 这种数字？模型对几何数字的精度有限（这是 VLA 的已知弱点）

### 0.A.6 关于 work positioning（用户视角）

如果**目标 = 强调 cascade reasoning 解决长程任务**：
- 简单 pick-place 任务（带 grasp 成功的）的 demo 就足够 → VLM reasoning 是 selling point，action expert 不需要超精细
- 解决方法：把现有 pick-place 任务跑通（DAgger 或 teleoperation 数据）→ 用 cascade 处理 sequencing → 论文卖点是 long-horizon reasoning

如果**目标 = 通用精细操作**：
- 必须把 action expert 的 state-covariate-shift 彻底解决
- Atomic skill data + cascade-decoupling 是值得投入的研究方向

### 0.A.7 几个被否定的备选

| 想法 | 否定理由 |
|---|---|
| 扰动 state + 恢复 训练 | 等价于轻量 DAgger，缺新颖度（用户判断） |
| state-only retrain with σ=0.1 | 还是 ε-tube 内扩散，shape 不匹配实际 policy 行为 |
| 完全 vision-only 重训 | π0/RT-2 风格，需重新训练 stage 1+2，工程量大且不属于"cascade-VLA"研究范围 |

---

## 1. A1.b last_arm stickiness（哪只手腕摄像头）

### 1.1 背景

模型一次只接收**一个 wrist camera 输入槽** (`left_wrist_0_rgb`)，但 RoboTwin 任务是**双臂**机械手。每个 phase 在 metadata 里有 `arm_tag ∈ {"left", "right"}` 指明本 phase 哪只手主动。

设计目标：训练和推理时让 wrist 槽都装"当前主动那只手"的相机图像。

### 1.2 训练侧：**当前是 A1.a（per-phase 切换）**

代码：`policy/lap/src/lap/datasets/robotwin_dataset.py:495-500`

```python
arm_tag = phase.get("arm_tag", "left")
wrist_camera = "left_camera" if arm_tag == "left" else "right_camera"
wrist_img = self._reader.decode_frame(self._hdf5_path(episode), wrist_camera, frame_idx)
```

每个**训练样本帧**根据它所属 phase 的 `arm_tag` 直接选取对应物理 camera 的图像，塞进 `WRIST_SLOT_KEY = "left_wrist_0_rgb"` 槽位。

**实质**：训练时 wrist 槽**逐 phase 切换**。一个 episode 内，前 3 phase 是 left → wrist 槽都是 left_camera；中间 2 phase 切到 right → wrist 槽全部换成 right_camera；之后切回 left → 再切。

**没有 stickiness 信号**进入模型：模型只看到"wrist 槽显示的物理视角"，并没有显式知道这是"我在用的那只手"。Stage/action 文本里有 `{arm}` 字段（如 "right gripper"），是隐式的语言侧线索。

### 1.3 推理侧：**完全没实现切换** ❌

代码：`policy/lap/lap_model.py:153`

```python
wrist = _ensure_224(left)   # always feed left wrist for now (A1.b stickiness deferred)
```

`update_observation_window` **永远**取 `img_arr[2]` (= `left_camera`) 放进 wrist 槽，无视当前 phase 应该用哪只手。

**train/test gap 巨大**：训练时一半 phase 给模型看右手视角，推理时永远只给左手。如果当前是 right-arm phase，模型看到的 wrist 图像和它训练时见到的同 phase wrist 图像是**完全不同的相机**。

### 1.4 这个问题对 sim eval 失败的影响

之前 `lap_robotwin_finetune_qpos` ep0/ep1/ep2 全 step_limit 失败。视频里都是左臂（或图像底部偏左的暗色臂）伸入场景。考虑到 arrange_blocks 任务中很多 phase 用右臂，而推理时 wrist 槽永远是 left_camera：

- 当 sim 状态进入 "应该用右臂" 的 phase 时，wrist 槽给的是 left_camera（看不到右臂在哪），模型对右臂状态的视觉判断完全错误
- 这**很可能是 sim eval 全失败的一个独立原因**，与 covariate-shift 是叠加的

### 1.5 修复方案讨论

#### 方案 1：客户端解析 cascade text 维护 last_arm

```python
class LAP:
    def __init__(...):
        self.last_arm = "right"  # 默认右臂（pick_place 多数从右开始）

    def update_observation_window(self, img_arr, state):
        head, right, left = img_arr
        wrist_src = right if self.last_arm == "right" else left
        self.last_arm = self._parse_arm_from_cascade(self.last_reasoning_text, self.last_arm)
        wrist = _ensure_224(wrist_src)
        ...

    def _parse_arm_from_cascade(self, text, prev):
        if not text: return prev
        t = text.lower()
        # 优先检查 [action] 段（描述当前要做的动作）
        action_seg = text.split("[action]")[-1] if "[action]" in text else ""
        if "right gripper" in action_seg or "right arm" in action_seg:
            return "right"
        if "left gripper" in action_seg or "left arm" in action_seg:
            return "left"
        return prev  # 没明确信号就保持
```

**优点**：训练时 cascade 已有 `{arm}` 字段，**推理时解析就能拿到信号**。
**缺点**：解析鲁棒性需要测试；如果 cascade 生成质量不稳定（参见问题 2），arm 可能反复切换。

#### 方案 2：训练侧改为 A1.b stickiness

让训练时 wrist 槽不每 phase 切，而是**只在 cascade 提到换手时才切**。要求：
- 数据集 build_sample 时维护跨 phase 的 `prev_arm` 状态
- 当 `phase.arm_tag != prev_arm` 时，cascade 文本必须包含明确的"切换手"信号
- 推理时模型隐式学会"看到换手信号才更新 last_arm"

**优点**：训练/推理一致，模型自己学决策。
**缺点**：需要重训；要修改 cascade text 模板加上换手 marker；流式 iteration 难做（当前 dataloader 按帧随机采样，无跨 phase 状态）。

#### 决策

**短期**：实现方案 1（客户端解析）。低成本，可立即测。
**长期**：如果方案 1 不彻底解决问题，考虑方案 2（A1.b 训练侧）。

### 1.6 实现状态（2026-05-12）

**客户端解析已实现** (`lap_model.py::_parse_arm_from_cascade`)：
- 优先级 `[action]` → `[stage]` → `[plan]`，每段取**最后**出现的 left/right 信号
- 匹配模式：`{left,right} {gripper,arm,hand}`
- 6/6 smoke 测试通过
- `reset_obsrvationwindows` 把 `last_arm` 复位回默认（目前是 `right`，可考虑改 `left`，第一帧实际任意都行）

**arm-switch 紧急 1 步 exec 已实现** (`deploy_policy.py::eval`)：
- 推理前 snapshot `wrist_arm_before_infer`
- 推理后 `wrist_arm_after_infer = model.last_arm`
- 不一致 → `exec_n = 1`（只走 chunk[1] 一帧）+ 打 log
- 下一帧 obs 用更新后的 last_arm 重新 pack wrist → 正常 chunk 执行

**疑点 1**（action chunk 跨 phase）：确认会跨。`_safe_end_frame` 只阻 success→failure 跨界，相邻 success phase 间无限制。
- chunk 内可能含双臂动作；wrist 槽固定为 frame_idx 处 phase 的 arm_tag
- 暂不修。先看 last_arm 修正效果，必要时再做 chunk 截断到 phase 内

**疑点 2**（VLM 发现切换要不要重新 infer）：
- 简化版（采纳）：检测 arm 切换 → 只执行 1 帧 → 下次 infer 时 wrist 已正确，等价于"切换瞬间紧急 1-step 闭环"
- 完整版（成本翻倍）：arm 切换帧重新跑一次完整 cascade+flow — 暂不做

### 1.7 待办

- [x] 客户端 cascade 解析 + last_arm 切换
- [x] arm-switch 紧急 1 步 exec
- [x] 视频 overlay 显示 `wrist=left/right`（反映 last_arm 实时值）
- [ ] 等 state-noise ckpt + 跑 eval 验证

---

## 2. 推理 cascade 重生 / 复用策略

### 2.1 训练时 cascade 文本结构

代码：`robotwin_dataset.py:74-189`

| 段 | 内容 | 频度 |
|---|---|---|
| `[plan]` | `_arrange_plan_from_phases` 拼出的多句子大目标，**整个 episode 共享** | 每 episode 不变 |
| `[stage]` | 从该 phase 的 `subgoal_reasoning` 模板里随机选一句，含 `{arm}/{color}` | 每 **phase** 内随机选变体 |
| `[action]` | 从该 phase 的 `phase_prompts` 列表里随机选一句 | 每 **phase** 内随机选变体 |

**关键**：训练中 plan 是"episode 级常量"，stage/action 是"phase 级 + 模板随机"。**模型从未见过 plan 在 episode 内变化**。

### 2.2 当前 `CascadePipelinePolicy` 行为

代码：`policy_adapter.py::CascadePipelinePolicy.infer`

**每次客户端调用 `infer`** 都执行：
1. AR `sample_tokens` 生成新的 `[plan][stage][action]`
2. 拼到 prefix
3. flow `sample_actions`

也就是说，**每次推理（每个 chunk）都重生整个 cascade**。后果：
- Plan 可能在 episode 内随观察图像变化而切换（视频里 t=15s "blue" → t=30s "purple"）
- Stage/action 也随机漂移

这与训练分布不符（训练里 plan 是 episode-static）。

### 2.3 切换时机的几个方案

#### 方案 A：Plan episode-cached + Stage/action 每次重生

- Plan 在 episode 首次 infer 时生成 + 缓存，后续都复用
- Stage 和 action 每次推理都重新 AR 采样

**优点**：plan 稳定，避免视频里看到的"块颜色反复横跳"
**缺点**：plan 一旦生成有误就锁死 episode（比如初始视角没看清，plan 误选了不存在的颜色）

#### 方案 B：固定 K=16 frames 重生 + gripper 触发双保险

- Stage 默认每 K 步（如 16）重生一次
- 监测 gripper 闭合/张开（state 后两维），触发额外重生（phase 切换信号）
- Plan 仍然 episode-cached

**优点**：减少 cascade 噪声，明确"重新规划"的触发条件
**缺点**：靠规则；如果 gripper signal 不准（比如多次抓握失败），会乱触发

#### 方案 C：模型自己决定何时重生（智能化）

更激进的设计：训练时让模型显式输出"finished_stage"或"need_replan" 信号。两种思路：

**C1. 输出 EOS 后等推理**：
- 训练时 cascade text 末尾加 `[stage_done]` 或类似 marker
- 推理时检测 AR 输出含 `[stage_done]` → 触发 stage 重生
- 这要求训练数据有"完成 stage"信号 → 需要数据预处理脚本能从 phase boundary 推断 stage 完成

**C2. 二选一 head：每帧模型预测"do_replan_now" bool**：
- 训练时给 phase boundary 帧 label=1，其它=0
- 推理时 forward → 看 do_replan_now 概率 → > 0.5 就跑 sample_tokens
- 这是额外的二分类头，需要重训

### 2.4 用户提议的更智能版本

> "当前的 action chunk 已经完成了这个 stage。CE 优化目标会要求输出 `finished stage`"
> "模型通过观察图像已经完成了一个 stage，就会生成新的 [stage]，否则直接生成 [action]"

这是 C1 思路的具体落地。可行性分析：

- **怎么在训练中标注**：每个 frame 属于哪个 phase 我们知道；phase 最后一帧的 cascade 加 `[stage_done]` 后缀就能给模型 supervision
- **推理时如何用**：sample_tokens 看到 `[stage_done]` → 当前 stage 用过的，需重生新 stage；否则继续用旧 stage
- **plan 是否动**：用户建议 plan 一开始生成后直接复用（"直接添加到当前 prompt 的后缀，不像现在一样重新推理"）

这是合理的设计。我建议：

#### 最终决策（短期 + 长期）

**短期（不重训）**：
1. **Plan 缓存**：episode 开始时 sample 一次 plan，之后所有 infer 调用复用同一个 plan
2. **Stage/action 每次重生**：减少噪声
3. **K=16 frames 重生 stage**（暂时）+ gripper signal

**长期（重训）**：
1. 在 dataloader 给 phase 最后一帧加 `[stage_done]` 后缀
2. 推理时检测该 marker 触发 stage 重生
3. 训练时**混合 plan-prompt 和 plan-target**：保留现在的 p_plan=0.15 设置，让模型既能 prompt 时使用 plan，又能生成 plan

### 2.5 C1 / C2 详细分析（2026-05-12 讨论）

#### C1 缺陷已识别

> "VLM 不会看到 action 的注意力，实际上没有任何信号在提示模型要学输出 `[stage_done]`"

确认。`stop_action_to_vlm_grad=True` 时 VLM expert 看不到 action 的反向梯度，所以**用 action chunk 是否完成 stage 当 supervision 信号** 不可行。

**C1 修正**：信号改成 **frame-level，纯图像驱动**。Phase 的最后 K=1-2 帧打 `[stage_done]` 标签 → VLM 学"看到这种图像→ emit `[stage_done]`"。但已被 C2 覆盖，**不再单独推进 C1**。

#### C2 完整方案（采纳为长期路径）

**核心**：模型在 cascade 推理时自动决定何时切 stage —— 看到上一个 stage 文本 + 当前图像 → 二选一：继续 emit action，或 emit 新 stage。

##### C2.1 训练数据改造

dataloader 给每个采样帧生成两种类型之一的 cascade prefix+target：

**类型 A（mid-phase 帧，约 80%）**：
```
prefix: [BOS] task_prompt [plan] <plan> [stage] <current_stage>
target: [action] <action_lang>
```

**类型 B（phase 边界帧，约 20%）**：
```
prefix: [BOS] task_prompt [plan] <plan> [stage] <prev_stage>
target: <stage_completion_reasoning> [stage] <new_stage> [action] <new_action_lang>
```

`<stage_completion_reasoning>` 是新加的字段（**用户提议**，下面 2.5.4 详述）。

##### C2.2 推理时累积 prefix

每 emit 一个 stage，prefix 追加该 stage。N 个 phase 后 prefix 含：
```
[plan] <plan> [stage] <p1_stage> <p1_done_reasoning>
            [stage] <p2_stage> <p2_done_reasoning>
            ...
            [stage] <pN_stage> [action] <action>
```

**Train-test gap**：训练里 prefix 最多 2 个 stage（一前一后），推理时 prefix 含全部历史 stage。

用户决策：**先保留 full prefix，让 attention 自己学**（期望图像 cross-attn 总是 ground 在最新 stage 上）。**超过 token budget (320) 时再上滑动窗口**。

不做 attention mask / 物理删除（设计太繁琐）。

##### C2.3 LLM 训练 prefix 问题（你的提问）

> "LLM 训练的时候只会把之前回归的所有内容作为 prefix 吗？"

**当前 cascade-VLA**：单 phase 一个训练样本。模型一次 forward 看到整个 ` [BOS] prompt [plan] plan_text [stage] stage_text [action] action_text [EOS]` 序列，对 AR-target span 内每位置计算 CE loss。**没有"上一个 stage"概念**。

要做 C2 **必须改 dataloader** 让一个样本能包含**多个 phase 的 cascade 串联**（至少 2 个）。当前数据格式不支持。

#### 2.5.4 ★ stage_completion_reasoning（用户提议）

**用户原话**：
> "在每次新的 stage 时要求 CE: `<stage>xxx<stage>Since the blue block has been stacked onto xxx, xxx` 即多了一个 `<stage 完成时_reasoning>` 部分是不是合理的，虽然增加了 infer 负担"

**判断：非常合理**。理由：

1. **训练数据已有"现在时" reasoning**（用户引用的 `subgoal_reasoning`），但**没有"完成时"的总结性 reasoning**
2. 加完成时 reasoning 强制模型显式表达"为什么 stage k 结束 + 为什么 stage k+1 开始" → 让 VLM 学会把视觉变化 → 文本表达 → 触发 stage 切
3. 类比 CoT 训练：先 reasoning 再决策 vs 直接决策 — reasoning 通常更鲁棒
4. cascade 文本结构变成：

```
[stage] <stage_k 进行中 reasoning>
       <stage_k 完成时 reasoning>  ← NEW
[stage] <stage_k+1 进行中 reasoning>
       <stage_k+1 完成时 reasoning>
[stage] ...
```

##### 2.5.4.1 数据准备

`subgoal_reasoning`（现在时）已经在每个 phase 的 metadata 里有 4 个变体。需要为每个 phase 也产出 1-4 个"完成时" reasoning。

获取途径选项：
- (a) **现在时 reasoning 反过来改写**：用 LLM 把 "we picked the purple cube because..." → "the purple cube has been picked because..."。批处理，离线一次性
- (b) **手工模板**：从 `phase.kind + task_spec` 模板生成（类似已有的 `_PICKPLACE_REASONING_TEMPLATES`）— 增量加 `_COMPLETED_TEMPLATES`
- (c) **混合**：arrange/stack 用模板，pick_place 用现成的

**推荐 (b)**：模板可控、零外部 LLM 依赖、好维护。如：
```python
_COMPLETED_TEMPLATES = {
    "lift_down": [
        "The {color} cube has been placed at {dest}.",
        "Now that the {color} cube is set down at {dest}, the {arm} gripper can release and retract.",
        "{color} is in position at {dest}; that subtask is complete.",
    ],
    ...
}
```

##### 2.5.4.2 训练 supervision

完成时 reasoning 也算进 `[stage]` 段的 AR target — 用 `tokenized_stage_mask=True` 覆盖。模型对它做 CE。`tokenized_plan_mask` / `tokenized_ar_target_mask` 一并扩展覆盖该段。

##### 2.5.4.3 推理收益

- 输出含"<completed_reasoning> + 下一个 stage"显式信号 → 客户端容易解析"我现在在 stage k 还是 k+1"
- last_arm 解析也变可靠（completed reasoning 通常会显式说"left gripper has released..."）
- VLM 学到了"看到这种视觉状态 → 描述完成 → 触发新 stage"，鲁棒性比纯依赖 image cross-attn 更高

##### 2.5.4.4 代价

- 数据预处理：写 `_COMPLETED_TEMPLATES`（数据集已有 `_PICKPLACE_REASONING_TEMPLATES` 蓝本，工作量低）
- 训练 token 长度：每 phase 多 ~20-40 tokens（completed reasoning），单次推理 cascade 长度 +20-40 tokens
- 推理延迟：AR 多生成 20-40 tokens ≈ +200ms per phase boundary

#### 2.5.5 Plan-cache 短期方案（**采纳为短期路径**）

不重训。客户端做：
1. Episode 首帧 infer → server AR 生成完整 cascade
2. 客户端从 `result["reasoning_text"]` 用 `[plan]/[stage]` 分隔提取 plan 字段，缓存在 `self.cached_plan`
3. 后续 infer：把 plan 作为 obs 字段送回 server，让 tokenizer 走 `plan_position="prompt"` 路径（训练里 85% full-cascade 帧就是这种格式）
4. server AR 只重新生成 stage + action（仍然每次重生）
5. `reset_obsrvationwindows` 清掉 cached_plan

实施位置：
- 客户端：`lap_model.py` 加 `self.cached_plan` + 解析
- 客户端：`update_observation_window` 把 cached_plan + plan_position 加到 obs
- server：不改（tokenizer 路径已支持）

### 2.6 待办

- [x] 短期：Plan-cache 设计已敲定（待实现）
- [ ] 短期：实现 client-side Plan-cache
- [ ] 长期：dataloader 改造支持 2-phase cascade 串联（C2.1）
- [ ] 长期：写 `_COMPLETED_TEMPLATES` 数据预处理（2.5.4.1）
- [ ] 长期：tokenizer 扩展支持 stage_completion_reasoning（2.5.4.2）
- [ ] 长期：Stage 1 + Stage 2 用 C2 重训
- [ ] 长期：监控 token budget，超出时上 sliding window

---

## 3. Teacher-forcing 训练 vs self-generated 推理 的分布 gap

### 3.1 训练里"喂给 action expert 的 prefix"长什么样

```
[BOS] <task_prompt> [plan] <GT_plan_text> [stage] <GT_reasoning> [action] <GT_langact> [EOS] [PAD]...
```

其中 GT 来自 dataloader 的 `meta.json`。`TokenizePromptAndReasoning` 把这些字段拼成一条 `tokenized_prompt`，`ar_target_mask=True` 覆盖 plan+stage+action 段（这些段一边给 action expert 看一边对 VLM 做 CE supervision）。

**所以训练时**：action expert 看到的 cascade 始终是**专家级 GT 文本**，且 90% 都"语义干净"（来自人工模板）。

### 3.2 推理里"喂给 action expert 的 prefix"长什么样

```
[BOS] <task_prompt> <AR_generated_plan_stage_action> [EOS]
```

AR 生成的可能是：
- 正确的 plan 格式（"[plan] place the red block..."）
- 也可能格式错乱（少 marker、错位、内容偏离训练分布）
- 因为推理时 observation 随物体被搬动而变化，**生成出来的 cascade 文本不再是任何 GT 模板的精确复述**

**train/test gap**：action expert 训练时看的是干净 GT，推理时看的是"模型自己生成、可能错乱、随时间变化"的文本。

### 3.3 缓解方案

#### 方案 a：**Stage 3 联合训练**（用户提到的）

在 Stage 2 后加一个 Stage 3：
- 训练步骤包含 "rollout cascade → 拼回 prefix → 算 action loss"
- 让 action expert 学会处理"自生成"的 cascade（即使有错也能 robust）

这是 Schedule sampling 在 cascade 上的扩展。代价是**训练管线巨复杂**：每步训练要先做一次 AR 采样再做 flow 训练，~2-3x 单步成本。

#### 方案 b：**Stage 2 训练加 cascade dropout / corruption**

更轻量：训练时**随机扰乱 cascade text**（删 marker / 替换近义词 / 截断），让 model 学会容忍噪声 prefix。

#### 方案 c：**先看 inference 质量再说**

用户原话："如果看 inference 的质量觉得这个问题不是很严重，可以暂时不管"。

我们当前观察到的问题：
- 之前 sim eval 0/N 全失败
- Dry-run 在 GT 帧上 MSE=0.001 → 模型在 GT-prefix 条件下完美
- 推理时 cascade 是 AR 自生成 → prefix 不再是 GT
- **A 重训（state-noise）能不能闭合这条 gap 还要看**

### 3.4 决策

**暂缓**。等 state-noise 重训 + cascade pipeline + plan cache 等改完跑 eval 出结果。如果还是 0/N，那 teacher-forcing gap 就是下一个嫌疑。优先级低于其它项。

### 3.5 待办

- [ ] state-noise 训练完后跑 eval，记录 cascade prefix 的实际效果
- [ ] 如果效果差，对比"喂 GT cascade vs 自生成 cascade"两种 eval 配置看 gap 是不是这里

---

## 4. 视频 overlay 排版 + 调试可读性

### 4.1 当前问题

视频左上角 overlay 显示：
```
infer@step=NN  step=MM/1000
chunk=N/7
wrist=left
[plan] Place the purple block at the corner of the L. Place the yellow block ...
```

**问题**：
1. `[plan]` 整段文本动辄 400+ chars，一行显示在 320x240 视频上只能看到开头几个词
2. `[stage]` 和 `[action]` 完全显示不出来（被 [plan] 长度挤掉）
3. 字号太大（0.45 scale），更难显示长文本

### 4.2 修复

代码：`envs/_base_task.py::_write_eval_video_frame`

需要做：
- 把 `reasoning_text` 按 `[plan]/[stage]/[action]` marker 切段
- 每段单独换行
- 每段内部按 ~36 chars wrap
- 字号缩小（0.45 → 0.35）

### 4.3 待办

- [ ] 改 `_write_eval_video_frame` 排版

---

## 5. 推理耗时 profiling

### 5.1 当前观察

- 单 episode 1000 step 约 7-8 min 墙时
- `exec_horizon=8` 时模型每 8 step 推理一次 → 125 次 infer / episode
- cascade pipeline = AR + flow，两次 forward
- 没有任何 timing 打印

### 5.2 想知道

1. 每次 infer 的总耗时（client 侧）
2. server-side `infer_ms`（policy_timing 已经在返回里）
3. AR sample_tokens 耗时 vs flow sample_actions 耗时分别多少
4. AR 生成的 token 数（cascade 长度）
5. 网络 / kubectl pf 序列化耗时

### 5.3 修复

`policy/lap/lap_model.py::LAP.get_action`：

```python
result = self.policy.infer(self.observation_window)
elapsed = time.time() - t0
server_ms = result.get("server_timing", {}).get("infer_ms")
policy_ms = result.get("policy_timing", {}).get("infer_ms")
n_tokens = (len(result.get("reasoning_text", "").split()) if result.get("reasoning_text") else None)
print(f"[LAP-perf] infer={elapsed*1000:.0f}ms  server={server_ms}  policy={policy_ms}  reasoning_words={n_tokens}")
```

`CascadePipelinePolicy.infer` 已经记了 `policy_timing.infer_ms`（flow 部分），可以再分别记 AR 阶段时长：

```python
ar_start = time.monotonic()
ar_tokens = self._sample_tokens(...)
ar_ms = (time.monotonic() - ar_start) * 1000
# ...
flow_start = time.monotonic()
actions = self._sample_actions(...)
flow_ms = (time.monotonic() - flow_start) * 1000
outputs["policy_timing"] = {"ar_ms": ar_ms, "flow_ms": flow_ms, "total_ms": ar_ms + flow_ms}
```

### 5.4 待办

- [ ] server 侧：拆分记录 AR 和 flow 各自耗时
- [ ] client 侧：每次 infer 打 timing
- [ ] 跑一次 eval 看 LLM (AR) 占整个 infer 的百分比
- [ ] 也观察 exec_horizon = 1 vs 8 对总墙时的影响

---

## 6. Wandb visibility 问题

**为什么这次训练 wandb 看不到？**

第一次启动 `lap_robotwin_run_qpos_noise1` 卡在 `wandb: Network error (ConnectTimeout), entering retry loop` 死循环（20 min 没进 step 0）。
pod cgroup 内出不去 `api.wandb.ai`。第二次启动加了 `WANDB_MODE=offline` 让 wandb 写本地。

**当前 offline run** 在 pod：
```
/data/zhaoqc/RoboTwin/policy/lap/wandb/offline-run-20260512_191808-1zcjankp/
└── run-1zcjankp.wandb  (7 MB+, 持续增长)
```

**上传方法**：
```bash
# 需要 wandb.ai 可达：
kubectl -n zhaoqc exec ... -- bash -lc '
  cd /data/zhaoqc/RoboTwin/policy/lap
  source .venv/bin/activate
  wandb sync wandb/latest-run
'
```

2026-05-12 尝试一次 sync 仍然 ConnectTimeout — 当前 pod 网络无法访问 wandb.ai。
**等训练完后再试**，或者在网络复活时跑。

**实时看 metrics**（不依赖 wandb）：
```bash
kubectl -n zhaoqc exec ... -- bash -lc \
  'grep -E "Step .* \\(train\\):" /data/zhaoqc/RoboTwin/policy/lap/logs/robotwin_run_qpos_noise1.log | tail'
```

---

## 7. 概念性讨论（covariate shift / DAgger / state-image 依赖）

整理 2026-05-13 讨论。这些是**长期需要保持注意的概念**，每次涉及训练 / 推理 / 数据集设计时回来翻一翻。

### 7.1 Covariate shift 的严格含义

**经典定义**：训练数据 `p_train(x)`、测试数据 `p_test(x)` 不一致；但任务的真实条件分布 `p(y|x)` 没变。

**在模仿学习里的特殊版本** = **compounding error**：
- 训练时 state 来自专家 trajectory：`s_t ∼ p_expert(s)`
- 测试时 robot 自己产生 state：`s_t ∼ p_policy(s | π)`
- 每一步微小预测误差 → 下一步 state 偏移 → 在更偏的 state 上做预测 → 进一步偏移 → 雪崩

**我们具体情况**：
- 训练里 `state[t] == joint_action/vector[t]` 都是 motion-planning 算法的精确解
- 推理里 state 由 TOPP-interpolate(prev_action) 产生，有不可避免的 servo / numerical 误差
- 误差累积 → state 偏离脆弱状态机 → 模型卡死

### 7.2 DAgger 本质 + 我们这能不能用

**算法**（Ross et al. 2011）：
```
循环 k 轮：
  1. 当前 policy π_k 在 env rollout → 收集 on-policy state {s_t}
  2. 对每个 s_t 问 expert：a* = expert(s_t)
  3. 把 (s_t, a*) 加入训练集
  4. 重训 π_{k+1}
```

**对我们可用性**：
- ✅ 有 expert（mplib motion planner 接受任意 state 都能算）
- ✅ 有 sim 可 rollout
- 工程量：~1 day 写脚本 + 多轮 (~6h 每轮) 重训

**和 BC + state noise 的本质区别**：
- state noise：ε-球扩散，shape 是高斯
- DAgger：on-policy 自然产生 state，shape 是 policy 实际行为对应的真实分布
- 后者效率高得多（覆盖的就是 policy 实际会去的地方）

### 7.3 State 还是 Image：模型主要靠哪个？

**当前架构事实**：
- Image 输入：head_camera + wrist_camera（wrist 槽可见手腕到夹爪）
- State 输入：14 维 qpos 连续注入到 action expert（`discrete_state_input=False`）

**模型实际依赖**：
- 训练数据不变量：`actions[t][0] == state[t]` → 模型只要把 state copy 到 action[0] 就 perfect 那部分 loss
- 后 7 帧通过 state 锚定坐标系 + 看图象 / cascade 决定方向
- → **数据结构强制 state 成为主要依赖**

**State OOD 时的退化**：
- Image 还在但只给"语义/粗位置"
- State 偏离 → 模型按训练里相似 state 找最近邻 → 找不到 → 预测退化到几个 mode 的平均 → 动作幅度变小，"定在原地"

**减弱 state 依赖的可能改造**：
| 方案 | 改动 | 风险 |
|---|---|---|
| `state_dropout: p` | 训练时按概率 p 把 state 置零 | 影响精度 |
| `discrete_state_input=True` | state 离散化成 bin token | π0.5 这么做，可用 |
| 完全去掉 state | 改架构成 vision-only | 工程量大，可能精度掉太多 |
| **Atomic skill data**（用户提议，见 §0.A.5）| 训练 primitive level → state-coverage 天然高 | 数据生成 + cascade 模板 |

### 7.4 Zero-shot 恢复策略（不重训）

用户想问：模型遇到 OOD 时，**能不能比"原地抖"更聪明地"晃回 in-distribution"**，不用 expert 标签？

**纯 zero-shot 几乎不行**，原因：模型不知道"哪里是 in-distribution"。但有几个 hack：

#### 7.4.1 客户端 heuristic（最快、不重训）

- **stuck-detection + jiggle**：检测 N 步 state 几乎不变 → 注入小幅度随机扰动 → 推到附近可能 in-distribution 位置。**风险**：撞坏物体
- **gripper override on cascade keyword**：cascade [action] 含 "close / grasp / squeeze" + arm 悬停 → 强制 gripper=0。**针对当前 0/10 的 grasp 失败模式**
- **uncertainty-aware action gating**：flow matching 多次采样观察 variance；高 variance → 切保守模式（动作幅度 ×0.5）

#### 7.4.2 训练时学一个"密度估计 / 不确定性 head"（需重训，中等工程量）

- 训练辅助 head 预测 `log p(s_t)` 或 action variance
- 运行时实时估计 "我现在 in-distribution 吗"
- 高不确定 → 切到保守 / 探索模式

#### 7.4.3 VLM 作 fallback：高层规划 → 运动规划

- Action expert uncertain 时，cascade 直接 emit "move +x 5cm" 这种粗粒度 motion
- Motion-planner 解算到具体 joint trajectory
- 这是 §0.A.5 atomic skill 的另一种用法 — 把 VLM 当 zero-shot 高层 (来自互联网视觉常识) + 运动规划当低层 → **hierarchical-VLA**

### 7.5 其它常规方案综合表

| 方法 | 关键思想 | 重训 | 收益 | 我们这适用度 |
|---|---|---|---|---|
| **DAgger** | on-policy state + expert label | ✓ | 高 (★★★) | 可用，需工程 |
| **遥操作数据混合** | 真人多样性 | ✓ | 高 (★★★) | 需 hardware |
| **BCN (state-noise)** | 训练 state 加噪声 | ✓ | 低 | σ=0.02 失败，σ=0.1 没意思 |
| **State dropout** | 训练随机置零 state | ✓ | 中 | 改 dataloader |
| **Discrete state** | state token 化 | ✓ | 中 | 改 1 个 config flag |
| **Atomic skill data** | task-agnostic motion primitive | ✓ | 中-高 (★★) | **本文提议，新颖** |
| **Error-correction reasoning** | cascade 含 "oops" 纠错 | ✓ | 中 | 需失败数据 |
| **Hierarchical (VLM keyframe + motion planner)** | VLM 出 waypoint，planner 解算 | 架构改 | 高 | 已有 mplib |
| **Online RL fine-tuning** | IL init + RL refine | ✓ | 高，慢 | 需 reward 设计 |
| **Diffusion at sequence level** | 长序列 action 分布 | ✓✓ | 高 | 架构大改 |
| **Stuck-detection + jiggle** | 客户端启发 | ✗ | 低 | 立即可做 |
| **Gripper override** | 客户端启发 | ✗ | 低-中 | 立即可做 |

---

## 总览 todo（按优先级）

### 立即（不重训）

1. ✅ A: state-noise σ=0.02 重训（运行中 — `lap_robotwin_run_qpos_noise1` PID 180900, WANDB_MODE=offline）
2. ✅ A1.b last_arm 客户端解析 + arm-switch 1 步 exec
3. ❌ Plan-cache 客户端实现（短期方案，§2.5.5）
4. ✅ 视频 overlay 多行排版
5. ✅ 推理 timing profiling

### A 训练完后

6. 跑 sim eval（带 last_arm + plan_cache）看 success rate
7. 视频 + profiling 数据 → 决定是否继续推进 C2

### 长期（要重训 Stage 1）

8. 🎯 C2 + stage_completion_reasoning 全面改造（§2.5.4）
9. Token budget 超出时上 sliding window

### 暂缓

10. 🔬 Stage 3 联合训练 / cascade corruption（teacher-forcing gap，§3）

---

## 工作 positioning（2026-05-13 决策）

两条岔路（用户视角）：

**A. 论文卖点 = "用 cascade reasoning 解决长程任务"**
- 简单 pick-place 任务（带 grasp 成功）就够 → VLM reasoning 是 selling point，action expert 不需要超精细
- 路径：用 DAgger 或遥操作数据补齐 pick-place 成功 → 用 cascade 做 sequencing → 论文卖点是 long-horizon reasoning
- 投入：1-2 周

**B. 通用精细操作的 cascade-VLA**
- 必须解决 action expert 的 state covariate shift
- 路径：Atomic skill data + cascade-decoupling（§0.A.5） — **新颖**且**直击瓶颈**
- 投入：2-3 月级别

**当前默认走 A** —— 路径稳，时间合理。B 留作后续 follow-up 工作。
