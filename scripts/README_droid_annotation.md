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

### 3.3 ECoT 风格总结（已对照论文原文校对）

参照：Zawalski et al. _"Robotic Control via Embodied Chain-of-Thought Reasoning"_ (CoRL 2024, arXiv:2407.08693v3)。

**ECoT 的 8 段链式输出**（论文 §4.1, Fig. 3）：

| # | ECoT 字段 | 内容 / 示例 | 我们的对应 |
|---|----------|----------|----------|
| 1 | **TASK** | 重述用户指令为 1 句陈述："Place the watermelon on the towel" | `task_instruction` (prompt) |
| 2 | **PLAN** | 高层 sub-task 数字列表："1. Move to watermelon 2. Firmly grasp it 3. Move to towel 4. Place watermelon on towel" | `[plan]` |
| 3 | **SUBTASK_REASONING** | 当前 sub-task 的理由 1-2 句："The watermelon is the first object the robot needs to interact with. The robot is not yet close to the watermelon, so the robot needs to move closer." | `[stage]` 前半 |
| 4 | **SUBTASK** | 当前 sub-task 1 句："Move to the watermelon" | `[stage]` 后半 |
| 5 | **MOVE_REASONING** | 当前 move 的理由 1 句："The watermelon is behind the robot, so it needs to move backward." | `[think]`（可选） |
| 6 | **MOVE** | 离散方向 primitive："Move backward" | `[action]`（我们偏长） |
| 7 | **GRIPPER POS** | 末端执行器像素坐标：`[156, 55]` | **暂不采用** |
| 8 | **VISIBLE OBJECTS** | object bbox 列表：`Watermelon [126, 146, 141, 125], Towel [20, 59, 218, 198], ...` | **暂不采用** |

**关键发现 1：MOVE 是高度模板化的 primitive**（论文 §4.2 + Appendix B）：

```
move [forward/backward] [left/right] [up/down], tilt [up/down], rotate [clockwise/counterclockwise], [close/open] gripper
```

理论上 3^6 = 729 种组合，**实际只有 54 种在 >0.1% 样本里出现**。出现最多的：
- `stop` (26.9%)
- `close gripper` (10.8%) / `open gripper` (7.2%)
- `move {down,left,right,up,forward,backward}` 单方向 (各 2-7%)

MOVE 是从 4-step proprioception delta（阈值 0.03m）派生的，**不是 VLM 直接生成的**。

**关键发现 2：ECoT 的数据生成是多阶段流水线**（论文 §4.2 Fig. 4），而非单 VLM 调用：

```
1. Prismatic-VLM   → 场景描述 ("Briefly describe the things in this scene...")
2. Grounding DINO  → 物体 bbox (text conf > 0.2, box conf > 0.3)
3. proprio → primitive  → MOVE 标签（规则，非 VLM）
4. OWLv2 + SAM     → 2D 末端执行器像素位置 (GRIPPER POS)
5. Gemini 1.0      → PLAN + SUBTASK + 各 reasoning 段
```

**关键发现 3：Gemini 输出用 XML-like 标签包裹**（论文 Fig. 11 prompt 原文）：

```
<task>...</task>  <plan>...</plan>  <subtask>...</subtask>
<subtask_reason>...</subtask_reason>
<move>...</move>  <move_reason>...</move_reason>
```

每个 step 一个 reasoning 字符串，所有 step 写成 Python dict `{step_id: reasoning_str}`。

### 3.4 我们与 ECoT 的关键区别

| 维度 | ECoT | 我们 | 影响 |
|------|------|------|------|
| **触发节奏** | 每帧全 8 段 (per-step AR) | 仅 keyframe (~5-15 / ep) | 我们 token 量降一个数量级，cascade-friendly |
| **MOVE 来源** | 规则派生 + 729 模板 | VLM 直接生成（自然语言） | 我们更灵活但失去 primitive 一致性；可后续考虑加 primitive 校验 |
| **MOVE 粒度** | 离散方向 (`"Move backward"`) | 较细 (`"Lower the gripper onto the cup"`) | 我们偏 ECoT 的 SUBTASK 风格 |
| **数据生成** | 多阶段（Prismatic + GDINO + OWL+SAM + Gemini） | 单 VLM 调用（Gemini 一次性出全部） | 我们简化，靠 Gemini 视觉 grounding；牺牲 bbox 精确度 |
| **失败语义** | 无显式标注 | `[think]` 必填于 `retry` keyframe | DROID retry 案例是我们的差异化价值 |
| **输出格式** | XML-like tags in Python dict | 严格 JSON | 鲁棒性 ≈ 等价 |
| **GRIPPER POS / OBJECTS** | 必有（核心 grounding 监督） | 暂不采用 | v1 后视 grounding 弱不弱再加 |

**对我们 prompt 设计的指导**：
1. **保留 ECoT 的层级**（TASK / PLAN / [SUBTASK_REASONING + SUBTASK] / [MOVE_REASONING + MOVE]）作为风格参照
2. **去掉每帧节奏** → 我们只在 keyframe 出现
3. **MOVE 自由化** → 但 fewshot 里可以混入"Move backward / Move left / Close gripper"类近-primitive 句式让模型保有该 prior
4. **GRIPPER POS / OBJECTS 推迟** → 等基线训完看真实 grounding 表现

> ⚠️ 当前 fewshot 里的 [`FEWSHOT_ASSISTANT`](annotate_droid/prompts.py) 描述偏 verbose。pilot 后可考虑调成 ECoT 风格的简短句式，对比 acceptance rate。

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

## 6. 集群 pilot 部署

数据流：

```
本地电脑 (numbnut)                      集群登录节点                          K8s Pod
 ┌──────────────────┐                   ┌──────────────────┐               ┌─────────────────────┐
 │ pod-tunnel proxy │ ◄── ssh tunnel ── │ kubectl exec ──→ │ ──── exec ──→ │ gsutil / python ann. │
 │ (clash:10808)    │                   │                  │               │ (HTTP via jump:8906) │
 └──────────────────┘                   └──────────────────┘               └─────────────────────┘
```

### 6.1 启动代理（本地电脑！）

**必须在本地电脑跑**，不是在 pod 里。跑一次后保留终端，整段 pilot 期间都需要它存活：

```bash
# 本地电脑 (numbnut)
~/.local/bin/pod-tunnel proxy &
# 校验隧道：应看到 "OK: cluster-jump is listening on 8906"
```

### 6.2 数据 + 模型已就位（user-confirmed paths）

```
DROID_100 数据集 (2 GB):  /data/datasets/droid_data_template/
Qwen weights:             policy/lap/Qwen2.5-VL-72B-Instruct/   (在 pod 内 /data/zhaoqc/RoboTwin/...)
```

如果以后要从 0 重新下载：

```bash
# 集群登录节点
kubectl exec -it deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 -- bash

# Inside pod:
export https_proxy=http://192.168.3.225:8906
export http_proxy=$https_proxy

# 验证代理是否工作
curl -sI -x $https_proxy https://www.google.com | head -2

# 下载 DROID_100 (~2 GB) —— 已经做过这步则跳过
mkdir -p /data/datasets
gsutil -m cp -n -r gs://gresearch/robotics/droid_100 /data/datasets/droid_data_template/
```

**关于 `gsutil` 断点续传**：
- `gsutil cp` 对**单个大文件 > 8 MiB** 内置 resumable upload/download（用 tracker file）
- 对**多文件**，加 `-n`（no-clobber）：已存在的 destination 跳过，重新跑命令即可"续传"
- `-m`（multi-thread）+ `-r`（recursive）是标准批量传输组合
- 失败重启就再跑同一条命令；不需要额外 flag

下载全集（1.7 TB）：

```bash
# 仅在 pilot 通过后跑
gsutil -m cp -n -r gs://gresearch/robotics/droid /data/zhaoqc/droid_data/
```

### 6.3 同步代码 + 安装依赖到 pod

```bash
# 本地电脑
cd /home/numbnut/worksapce/RoboTwin
rsync -avz policy/lap/scripts/annotate_droid/ \
    --exclude __pycache__ \
    k98s:/data/zhaoqc/RoboTwin/policy/lap/scripts/annotate_droid/
rsync -avz policy/lap/scripts/annotate_droid_gemini.py \
          policy/lap/scripts/annotate_droid_qwen.py \
          k98s:/data/zhaoqc/RoboTwin/policy/lap/scripts/
```

Pod 内安装依赖 —— **挑你要走的那条路径装**：

```bash
cd /data/zhaoqc/RoboTwin/policy/lap
# 通用依赖（所有路径都要）
uv pip install --python .venv/bin/python pillow numpy

# === Gemini 2.5 Pro path ===
uv pip install --python .venv/bin/python google-genai

# === Qwen-VL local HF path (推荐：因为 user 已经下了权重) ===
uv pip install --python .venv/bin/python transformers accelerate qwen-vl-utils
# torch / flash-attn 应该已经在 .venv 里（pi05/lap 训练环境复用）

# === Qwen-VL vLLM HTTP path (只在另起 vLLM 服务端时需要) ===
uv pip install --python .venv/bin/python openai
```

### 6.4 端到端 mock 测试（确认环境无误）

不调真实 VLM，跑 mock dry-run：

```bash
# Inside pod
cd /data/zhaoqc/RoboTwin
python -m policy.lap.scripts.annotate_droid._dryrun /tmp/dryrun.jsonl
# 期望: emitted=3 skipped=0 failed=0
```

### 6.5 跑 Qwen-VL-72B local HF pilot 100 集（**推荐主路径**）

权重已在 pod 内 `/data/zhaoqc/RoboTwin/policy/lap/Qwen2.5-VL-72B-Instruct/`，数据集
已在 `/data/datasets/droid_data_template/`。一条命令搞定：

```bash
# Inside pod — 不需要代理（local inference 全离线）
cd /data/zhaoqc/RoboTwin
.venv/bin/python policy/lap/scripts/annotate_droid_qwen.py \
    --mode local \
    --model-path  /data/zhaoqc/RoboTwin/policy/lap/Qwen2.5-VL-72B-Instruct \
    --rlds-dir    /data/datasets/droid_data_template \
    --rlds-name   droid_100 \
    --output      /data/zhaoqc/droid_cot/qwen_v0.1_pilot.jsonl \
    --max-episodes 100 \
    --attn flash_attention_2 \
    -v
```

注意点：

- `--rlds-name droid_100` —— DROID_100 子集的 TFDS builder name 不同于全集 `droid`。
- `--attn flash_attention_2` —— **可选**。在 H100/H200 上快 20-30%。**venv Python
  版本 (3.11) 与系统 Python (3.10) 不同时，`pip install flash-attn` 装到系统不进 venv**
  —— 必须 `uv pip install --python .venv/bin/python flash-attn --no-build-isolation`。
  装不上就别加这个 flag，默认走 SDPA 也很快。client 已经做了 fallback：要 flash 但
  venv 装不上时会自动降到 SDPA 并 warning，不会 crash。
- Qwen2.5-VL-72B fp16 ~145 GB —— 2× H200 (282 GB) 或 1× H200+CPU offload。`device_map="auto"`
  会自动 tensor-split。如果 OOM 可改 `--max-pixels 200704` (256×28×28) 减视觉 token。

吞吐预估（H200，flash_attention_2，bf16）：
- 单 episode ~10 keyframes ≈ 2-5 秒推理
- 100 集 ≈ 5-10 分钟 wall-clock

### 6.6 跑 Gemini 2.5 Pro pilot 100 集（备选 / 对照组）

```bash
# Inside pod — ensure proxy env is set first
export https_proxy=http://192.168.3.225:8906
export http_proxy=$https_proxy
export GOOGLE_API_KEY="...your key..."

cd /data/zhaoqc/RoboTwin
.venv/bin/python policy/lap/scripts/annotate_droid_gemini.py \
    --rlds-dir  /data/datasets/droid_data_template \
    --rlds-name droid_100 \
    --output    /data/zhaoqc/droid_cot/gemini_v0.1_pilot.jsonl \
    --max-episodes 100 \
    -v
```

预计：~$2 / 100 集，~5 min wall-clock（含 Gemini API 排队）。

### 6.7 跑 Qwen-VL-72B vLLM HTTP pilot（更高吞吐场景）

如果 pilot 通过后要做全集（76k）标注，单进程 HF transformers 跑会比较慢。
那时切到 vLLM 路径：

```bash
# 在 H200 deployment 启 vLLM server
uv run vllm serve /data/zhaoqc/RoboTwin/policy/lap/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=20 \
    --port 8100

# 标注 worker
.venv/bin/python policy/lap/scripts/annotate_droid_qwen.py \
    --mode vllm \
    --base-url http://<h200_pod_ip>:8100/v1 \
    --rlds-dir  /data/datasets/droid_data_template \
    --rlds-name droid_100 \
    --output    /data/zhaoqc/droid_cot/qwen_v0.1_pilot.jsonl \
    --max-episodes 100 \
    -v
```

### 6.8 评估

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
