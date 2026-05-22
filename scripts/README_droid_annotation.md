# DROID Embodied-CoT 标注 —— 实施文档

本文档配合 [`policy/Your_Policy/data_processing/README_cot_annotation_strategy.md`](../../Your_Policy/data_processing/README_cot_annotation_strategy.md)（策略层），覆盖 **DROID 数据集思维链标注的实施细节**：keyframe 规则、prompt 风格、脚本运行、QC 流程。

代码位置：[`policy/lap/scripts/annotate_droid/`](annotate_droid/)
顶层 CLI：[`annotate_droid_qwen.py`](annotate_droid_qwen.py) / [`annotate_droid_gemini.py`](annotate_droid_gemini.py)

## 1. 整体架构

```
DROID RLDS / JSONL
       │
       ▼  iter_droid_rlds / iter_jsonl
EpisodeBundle(gripper_width, ee_pos, frame_loader, ...)
       │
       ▼  detect_keyframes        (rule-based, see §2)
list[Keyframe(t, type, gripper_state)]
       │
       ▼  bundle.frame_loader(kf.t)
list[np.ndarray]  ← lazy JPEG decode, only for keyframes
       │
       ▼  build_openai_messages / build_gemini_contents  (see §3)
VLM request (system + fewshot + task + image-N)
       │
       ▼  QwenVLClient / GeminiClient
VlmReply(text, latency, tokens)
       │
       ▼  parse_vlm_output       (§4, robust JSON extractor)
plan: str, keyframes: list[KeyframeAnnotation]
       │
       ▼  audit_episode          (§5, A1-A8 checks)
AuditReport(passed, errors, warnings)
       │
       ▼  EpisodeAnnotation.to_jsonl_line
output.jsonl (resume-safe append)
```

每个箭头都是独立的、可单测的模块；规则改动只动单个文件。

## 2. Keyframe 检测规则

代码：[`annotate_droid/keyframe.py`](annotate_droid/keyframe.py)。运行 `python policy/lap/scripts/annotate_droid/keyframe.py` 看 synthetic demo。

### 2.1 输入与阈值

| 信号 | 来源 | 阈值常量 |
|------|------|----------|
| Franka gripper 宽度 | `obs/gripper_position` | `GRIP_CLOSED_MAX=0.005`, `GRIP_OPEN_MIN=0.060` |
| EE 位置 | `obs/cartesian_position[:3]` | (用于 R4) |
| FPS | 默认 `15.0` | DROID 标准 |

### 2.2 规则汇总

| ID | 规则 | 触发条件 | 输出类型 |
|----|------|---------|---------|
| **R1** | gripper 离散化 | `width < 0.005` → closed；`> 0.060` → open；中间 → partial | — |
| **R2** | gripper 转移 | 状态变化 + 持续 ≥ 3 frames（hysteresis 防抖） | `grasp` / `release` |
| **R3** | 失败重试 | close→open→close 落在 1.5s 窗口内 | `retry` |
| **R4** | EE 运动相变 | `cos(v_before, v_after) < 0.30`（两侧速度均 > 0.02 m/s）或速度差 > 0.05 m/s | `motion` |
| **R5** | 端点 | 首帧 / 末帧 | `begin` / `end` |
| **R6** | 长 stage 填充 | 相邻 keyframe 间隔 > 60 frames → 等分插填 | `filler` |

**为什么用规则而不让 VLM 自己挑**：

1. 可复现 —— 同 episode 同 keyframe 集，prompt 迭代不会扰动选择
2. 便宜 —— 全部本地 numpy 运算
3. VLM 在 fixed inflection point 上 reason 比"先挑点再 reason"两步走的一致性高

**阈值调参建议**：先用合成 demo（`python -m policy.lap.scripts.annotate_droid.keyframe`）验证，再用 5-10 个真实 DROID episode 调（看每集 keyframe 数是否落在 5-15）。

### 2.3 输出格式

```python
@dataclass
class Keyframe:
    t: int                         # frame_idx
    type: KeyframeType             # begin / grasp / release / retry / motion / filler / end
    gripper_state: GripperState    # open / partial / closed
    extra: dict                    # retry 时存 previous_attempt_frame
```

每集典型输出 5-15 个 Keyframe。

## 3. Prompt 设计

代码：[`annotate_droid/prompts.py`](annotate_droid/prompts.py)。**这是迭代的核心文件**。

### 3.1 锁定的风格决策

- 4 个 marker：`[plan]` `[stage]` `[think]`（可选）`[action]`
- **`[think]` 是普通文本**，不加 special token（user 锁定）
- **`[action]` 只在 keyframe 出现**（user 锁定）—— 非 keyframe 帧由 cascade 缓存 + flow 续推
- **不写 negative reasoning**（user 锁定）
- VLM 输出严格 JSON 而非自由文本 —— 解析鲁棒性 > 风格自由度

### 3.2 输出 JSON schema

```jsonc
{
  "plan": "2-5 句，描述总目标 + 数字序列 sub-goals",
  "keyframes": [
    {
      "frame_idx": 38,
      "stage": "1-3 句，自然语言；不要重述 plan",
      "think": null,                  // 或 1-2 句失败/替代方案推理
      "action": "≤ 12 词，祈使句"
    }
  ]
}
```

### 3.3 ECoT 风格总结（参考但不照搬）

UC Berkeley 的 Embodied-Chain-of-Thought (Zhao et al., 2024) 是我们的风格参照源。它在每帧都生成 7 层 reasoning：

| ECoT 层级 | 内容示例 | 我们的对应 |
|----------|---------|----------|
| **TASK** | "Put the watermelon on the towel" | `task_instruction` (prompt) |
| **PLAN** | 数字序列："1. Move to watermelon 2. Firmly grasp it 3. ..." | `[plan]` |
| **SUBTASK_REASONING** | "The watermelon is the first object the robot needs to interact with. The robot is not yet close to the watermelon, so the robot needs to move closer." | `[stage]` 段的前半 |
| **SUBTASK** | "Move to the watermelon" | `[stage]` 段的后半（融合） |
| **MOVE_REASONING** | "The watermelon is behind the robot, so it needs to move backward." | `[think]`（可选） |
| **MOVE** | "Move backward" | `[action]`（我们偏长，但不绝对） |
| **GRIPPER POS / VISIBLE OBJECTS** | 像素坐标 / bbox | **暂不采用**（v1 后视情况加） |

**我们与 ECoT 的关键区别**：

| 维度 | ECoT | 我们 |
|------|------|------|
| 触发节奏 | 每帧全 7 层 | 仅 keyframe (~5-15 / ep) |
| MOVE 粒度 | 离散方向 ("Move backward") | 较细 ("Lower the gripper onto the cup") |
| 失败语义 | 无显式标注 | `[think]` 必填于 `retry` keyframe |
| 输出格式 | 自由文本 | 严格 JSON |

> ⚠️ **风格校验**: ECoT 论文原文如果你能附上，我会用具体示例对照检查我的总结。当前是基于我对论文 + 他们公开的 `embodied_features_bridge` 数据的理解推断。

### 3.4 Fewshot

代码里有 **1 个 fewshot 例子**（`FEWSHOT_USER` / `FEWSHOT_ASSISTANT`）—— text-only，不含图（图片成本高，pilot 验证后再决定是否升级为带图 fewshot）。

## 4. JSON 输出 / 解析

代码：[`annotate_droid/schema.py`](annotate_droid/schema.py)。

`parse_vlm_output(text)` 容错处理：
- 剥离 markdown 围栏 ```` ```json ... ``` ````
- 用括号配对找最外层 `{...}`（丢弃 JSON 前后的杂散 prose）
- `think` 字段允许 `null` / `""` / `"null"` 都视作 None

解析失败时 `EpisodeAnnotation.raw_output` 保留原始字符串，方便事后看 VLM 输错了什么。

## 5. 自动 QC

代码：[`annotate_droid/audit.py`](annotate_droid/audit.py)。

| ID | 检查 | 严重度 |
|----|------|--------|
| **A1** | `len(keyframes) == 输入 keyframe 数` | error |
| **A2** | `frame_idx` 与输入对齐 | error |
| **A3** | `type==grasp` → action 含 grasp 动词；`type==release` → action 含 release 动词 | error |
| **A4** | `type==retry` 必须 `think != null` | error |
| **A5** | plan ≤ 600 chars, stage ≤ 350, action ≤ 18 词 | warning |
| **A6** | `type ∈ {begin, end, filler}` 不应有 think（仅 noise 警告） | warning |
| **A7** | plan / stage / action 不可为空字符串 | error |
| **A8** | gripper open 时 action 不可是 grasp 动词；closed 时不可是 release | error |

任一 error → `audit.passed = False`，runner 仍把这条写入 JSONL（带 `audit.errors`），方便 retry 流程区分"已成功"和"需重跑"。

**词边界检查**：`grip` 是 GRASP 动词，但 `gripper` 不应触发误判 → 用 `\bgrip\b` 正则。已踩过这个坑。

## 6. 运行 pilot

### 6.1 准备 100 episodes（offline JSONL 路径，推荐）

为避免给标注 worker 装 1.7 TB DROID RLDS，提前 dump 100 集：

```python
# pilot_export.py（一次性脚本，待写）
# - 从 DROID RLDS 流式取 100 个 success episode
# - 写出 manifest.jsonl 和 images/<ep_id>/primary_NNNN.jpg
```

manifest 格式见 `droid_reader.iter_jsonl` 的 docstring。

### 6.2 跑 Qwen-VL-72B

```bash
# H200 host
uv run vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=20 \
    --port 8100

# annotation worker
uv run python policy/lap/scripts/annotate_droid_qwen.py \
    --jsonl  /data/droid_pilot/manifest.jsonl \
    --images /data/droid_pilot/images \
    --output /data/droid_cot/qwen_v0.1_pilot.jsonl \
    --max-episodes 100 \
    --base-url http://<h200_host>:8100/v1
```

### 6.3 跑 Gemini 2.5 Pro

```bash
export GOOGLE_API_KEY=...
uv run python policy/lap/scripts/annotate_droid_gemini.py \
    --jsonl  /data/droid_pilot/manifest.jsonl \
    --images /data/droid_pilot/images \
    --output /data/droid_cot/gemini_v0.1_pilot.jsonl \
    --max-episodes 100
```

### 6.4 评估

写一个简单评估脚本（待补，类似 `view_robotwin_dataset.py`）：

1. 自动：聚合两 JSONL 的 `audit.passed` 率、warning 分布、平均 token 用量
2. 人工：随机抽 20 条，对照视频 + 标注，1-5 分打分 plan / stage / think 质量
3. 输出：哪个 VLM 更好？prompt 哪些地方需要调？

通过率目标：**> 90%**。低于此值 → 改 system prompt 再跑一轮。

## 7. 进入大规模标注

| 阶段 | 触发条件 | 操作 |
|------|---------|------|
| pilot 100 | 已完成 | 评估 + prompt 迭代 |
| 标 7600 (1/10 DROID) | acceptance ≥ 90% on pilot | 选定 VLM；跑 ~3-4 hr (Gemini) / ~7 day (Qwen 1 卡对) |
| 小规模 pretrain | 1/10 已标 | 写 `DROIDCoTDataset`（仿 `BridgeECoTDataset`），训 LAP cascade-VLA |
| 标全集 76k | pretrain 表明数据有用 | 全量跑 |

## 8. 资源 / 成本估算（重申）

| VLM | 单 episode | 全 76k 集 |
|-----|------------|----------|
| Gemini 2.5 Pro | ~$0.02 + ~2s API | ~$1,500 + ~21 hr wall |
| Qwen2.5-VL-72B (2× H200, fp16) | ~0.04 H200-hr + 电费 | ~80 H200-day |

## 9. 待办

- [ ] 写 `pilot_export.py`（从 DROID RLDS dump 100 集到 JSONL+images）
- [ ] 跑 100-ep pilot（Qwen + Gemini 两条）
- [ ] 用户用 ECoT 论文原文校对 §3.3 风格总结的准确性
- [ ] 写 `evaluate_pilot.py`（自动 stats + 人工评分模板 CSV）
- [ ] 写 `DROIDCoTDataset`（仿 `BridgeECoTDataset` 接 LAP 训练）
- [ ] 决策：基于 pilot acceptance 选 Qwen 还是 Gemini 跑全集

## 10. 文件索引

```
policy/lap/scripts/
├── annotate_droid_qwen.py             # CLI (Qwen-VL)
├── annotate_droid_gemini.py           # CLI (Gemini)
└── annotate_droid/
    ├── __init__.py
    ├── keyframe.py                    # §2: rule-based detection
    ├── prompts.py                     # §3: system prompt + builder
    ├── schema.py                      # §4: dataclasses + parser
    ├── audit.py                       # §5: A1-A8 checks
    ├── droid_reader.py                # RLDS + JSONL iterators
    ├── client_base.py                 # VlmClient protocol
    ├── client_qwen.py                 # OpenAI / vLLM wrapper
    ├── client_gemini.py               # google-genai wrapper
    ├── runner.py                      # annotate_episode + run_batch
    └── _dryrun.py                     # mock-VLM E2E test (run me!)
```

E2E sanity check（不需要 VLM）：

```bash
python -m policy.lap.scripts.annotate_droid._dryrun /tmp/dryrun.jsonl
```

输出应见到 `emitted=3 skipped=0 failed=0` 和 3 条 `pass=True` 的 JSONL。
