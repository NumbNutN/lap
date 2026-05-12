# Stage 2 关键设计讨论（中文）

> 这个文档记录 Stage 2 推理 / 训练管线里几个**长期需要保持注意力**的设计权衡。每次涉及到这些点的修改，回来读一下相应小节，确认不要破坏既定决策或忘掉未决项。
>
> 配套文档：
> - `stage2_sim_eval_diagnosis.md` ：6 个 mitigation 方向 (A/B/C/D/E/F) 的细节
> - `cascade-bridge-pretraining-discussion.md` ：Stage 1 cascade 预训练讨论
>
> 状态图例：✅ 已实现 / ⚠️ 部分实现 / ❌ 未实现 / 🔬 待实验

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

### 1.6 待办

- [ ] 客户端实现 cascade 解析 + last_arm 切换
- [ ] 视频 overlay 显示当前 `wrist=left/right`（已有，但解析后会反映真实值）
- [ ] 用 arrange_blocks 跑 eval 对比

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

### 2.5 待办

- [ ] 短期方案 1：实现 Plan-cache（episode 内复用）
- [ ] 短期方案 2：实现 K=16 stage 重生策略
- [ ] 长期方案：dataloader 加 `[stage_done]` 标注 + 推理触发器

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

## 总览 todo（按优先级）

1. ✅ A: state-noise σ=0.02 重训（已启动 — `lap_robotwin_run_qpos_noise1` PID 175058）
2. ❌ A1.b last_arm 客户端解析（问题 1） — 短期高优先级修复
3. ❌ Plan-cache + K-step stage 重生（问题 2 短期方案） — 等 A 出 ckpt 一起测
4. ❌ 视频 overlay 多行排版（问题 4） — 立即做
5. ❌ 推理 timing profiling（问题 5） — 立即做
6. 🔬 Stage 3 联合训练 / cascade corruption（问题 3） — 暂缓，等其他 fix 结果
