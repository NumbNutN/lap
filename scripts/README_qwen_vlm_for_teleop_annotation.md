# Qwen2.5-VL-72B vLLM 部署 + prompt 设计（给 teleop 手动标注做 auto-fill）

## 1. 上下文

本项目的标注流程是：
- **人**：用 `scripts/annotate_teleop_episodes.py` 的 GUI 标 keyframe 段 `[frame_start, frame_end]` 和高层 `task_instruction` + `plan`
- **VLM**：拿到 `[frame_start, frame_end]` 段后，**自动补全** 该段的 `type` / `gripper_state` / `stage` / `think` / `action` 五个字段
- **人**：在 GUI 里复核 + 微调（特别是 `retry` 类的 `think` 字段、`stage` 跨语言表达）后 SAVE

VLM 的角色是**省力，不是替代**：人选段、人定义任务、人决定语义边界；VLM 填模板化文本 + 推断 type/gripper_state。

这个 doc 给另一个 agent 用，负责在 K8s 集群拉起 vLLM 服务 + 实现请求 client，让本机的 GUI 能 HTTP 调它。

---

## 2. 整体架构

```
┌──── 本机 (numbnut, has Quest + scenes) ────┐         ┌──── 集群 H200 节点 ────┐
│ scripts/annotate_teleop_episodes.py        │         │ vllm serve Qwen2.5-VL │
│ ┌─ episode hdf5 ─┐                         │  HTTPS  │ -72B-Instruct          │
│ │ 1. 用户选段     │ ──────────────────────▶│ ▲       │ tensor-parallel=2     │
│ │ [k_start, k_end]│   POST /v1/chat/...    │ │       │ max-model-len=32768   │
│ │ + frames + ctx  │                        │ │       └─ OpenAI-compatible ──┘
│ │ 2. 用户复核+SAVE│ ◀──────────────────────│ ▼
│ └─────────────────┘    JSON: type/stage/...│
└────────────────────────────────────────────┘
```

入口侧（本机）已有：annotator GUI。本 doc 要交付的：
1. K8s deployment + vLLM 启动脚本
2. Python client（gradio 进程 import）
3. Prompt 模板 + fewshot 例子
4. 输出解析 + 校验
5. 失败 fallback 策略

---

## 3. 硬件 / 部署

### 3.1 资源需求

| 模型 | 精度 | 显存 | 推荐配置 |
|---|---|---|---|
| Qwen2.5-VL-72B-Instruct | fp16 | ~140 GB | **2× H200 (141GB each)** 或 4× A100-80GB |
| Qwen2.5-VL-72B-Instruct | AWQ-int4 | ~40 GB | 1× H200 / 2× A100-80GB（推荐降级方案）|

如果只有 1 卡可用，**强烈建议先跑 AWQ-int4 版本**（Qwen 官方提供 `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`）。72B fp16 单卡装不下。

### 3.2 vLLM 启动命令

```bash
# 在 H200 节点
uv run vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --limit-mm-per-prompt image=8 \
    --port 8100 \
    --trust-remote-code \
    --served-model-name qwen2.5-vl-72b
```

关键参数：
- `--tensor-parallel-size 2`：跨 2 卡分片
- `--max-model-len 32768`：上下文窗口；6 张 320×240 图 + 系统/用户文本约 4-6k token，留足空间
- `--limit-mm-per-prompt image=8`：单次请求最多 8 张图（我们一段最多发 6 张，留 buffer）
- `--trust-remote-code`：Qwen-VL 的 vision tower 需要

### 3.3 K8s service（参考片段）

```yaml
apiVersion: v1
kind: Service
metadata:
  name: qwen-vl-72b
spec:
  selector:
    app: qwen-vl-72b
  ports:
    - port: 8100
      targetPort: 8100
  type: ClusterIP
```

本机 GUI 通过现有 `pod-tunnel` 隧道（同 `README_droid_annotation.md` §6.1）访问 `http://<jump>:<port>/v1/...`。

### 3.4 健康检查

```bash
# 在 pod 里
curl -X POST http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-vl-72b",
       "messages": [{"role":"user","content":"ping"}],
       "max_tokens": 16}'
# 应返回 OpenAI 格式的 chat completion
```

---

## 4. API 契约

vLLM serve 暴露 OpenAI-compatible `/v1/chat/completions`。请求格式（多模态）：

```python
{
  "model": "qwen2.5-vl-72b",
  "messages": [
    {"role": "system", "content": "<SYSTEM_PROMPT>"},
    # Fewshot (text-only) — 见 §5.3
    {"role": "user",      "content": "<FEWSHOT_USER>"},
    {"role": "assistant", "content": "<FEWSHOT_ASSISTANT_JSON>"},
    # 真实请求 — 文本 + 多张图（base64 data URLs）
    {"role": "user", "content": [
      {"type": "text", "text": "<USER_TEXT>"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      ...
    ]}
  ],
  "temperature": 0.2,
  "max_tokens": 800,
  "response_format": {"type": "json_object"}
}
```

`response_format: json_object` 强制模型只输出 JSON（vLLM 0.6+ 支持）。

---

## 5. Prompt 设计

### 5.1 设计约束（必须满足）

输出 JSON schema **必须**跟手动 GUI 写的 schema 完全一致：

```json
{
  "type": "begin | grasp | release | retry | motion | filler | end",
  "gripper_state": "open | partial | closed",
  "stage": "1-3 句陈述句，描述这段帧里机械臂在做什么子任务",
  "think": "1-2 句因果推理；仅 retry/failure 段才填，其它段填 null",
  "action": "≤12 词祈使句，单一动作或紧密相关的动作组合"
}
```

注意我们**没有 `plan` 和 `task_instruction`** 在 VLM 输出里——那两个是**整 episode 一次**，已经由人填好作为 context 传给 VLM。

### 5.2 System prompt（写死的模板）

```text
You are an embodied-AI annotator. Given a sequence of video frames showing
a robot arm performing a manipulation task, plus the episode-level task
description and plan, you produce a structured ECoT (Embodied Chain-of-
Thought) annotation for the specific frame range shown.

Output STRICT JSON with exactly these five fields:
  - type: one of "begin", "grasp", "release", "retry", "motion", "filler", "end"
  - gripper_state: one of "open", "partial", "closed" (state at the END of this segment)
  - stage: 1-3 declarative sentences describing the sub-task happening in this segment. Use "The robot..." as subject. Do NOT repeat the high-level plan.
  - think: 1-2 sentences of causal reasoning. ONLY fill this when the segment is a failure / retry / non-obvious decision; otherwise set to null.
  - action: <= 12 words, imperative sentence ("Move...", "Close...", "Lift..."). Use object descriptions like "blue cube", "leftmost red cube", "middle cube" for consistency. Use axis names ("yaw counterclockwise", "tilt up") rather than bare directions.

Type semantics:
  - begin: first segment of the episode, robot at ready pose
  - grasp: gripper closes around an object; gripper_state ends "closed"
  - release: gripper opens and releases an object; gripper_state ends "open"
  - retry: previous action failed (missed grasp / slipped cube / collision); MUST fill `think` with the failure cause + correction strategy
  - motion: transport / approach / re-orient; no grip-state change
  - filler: long gap between meaningful events
  - end: last segment; robot returns to a ready pose

If you are uncertain about gripper_state from images alone, use the
prior segment's state and any visible gripper width as a hint.

Do NOT include markdown fences. Do NOT add explanatory prose. Output ONLY the JSON object.
```

### 5.3 Fewshot（1 例，text-only 节省 token）

**FEWSHOT_USER**:
```text
Task: Stack the blue cube on the leftmost red cube, avoiding the middle obstacle cube.
Plan: 1. Approach and grasp the blue cube. 2. Lift and arc over the middle cube. 3. Lower onto the leftmost red cube. 4. Open the gripper to release.

Previously-annotated keyframes in this episode (for context, do not re-annotate):
- frame [0..8] type=begin grip=open: "Robot starts at the ready pose."
- frame [9..27] type=motion grip=open: "The robot approaches the blue cube from above."

Current segment: frames [28..36]. Annotate this segment only. Images below are sampled uniformly from this range.
[images]
```

**FEWSHOT_ASSISTANT** (JSON, no fence):
```json
{
  "type": "grasp",
  "gripper_state": "closed",
  "stage": "The gripper closes firmly around the blue cube.",
  "think": null,
  "action": "Close the gripper to grasp the blue cube"
}
```

### 5.4 User message 模板（每次请求填空）

```python
def build_user_message(*, task_instruction, plan,
                        prior_keyframes_summary, frame_start, frame_end,
                        images_b64):
    text = f"""Task: {task_instruction}
Plan: {plan}

Previously-annotated keyframes in this episode (for context, do not re-annotate):
{prior_keyframes_summary}

Current segment: frames [{frame_start}..{frame_end}]. Annotate this segment only. Images below are sampled uniformly from this range."""
    content = [{"type": "text", "text": text}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
    return {"role": "user", "content": content}

def fmt_prior_summary(keyframes):
    if not keyframes:
        return "(none — this is the first segment to annotate)"
    lines = []
    for kf in keyframes:
        lines.append(f"- frame [{kf['frame_start']}..{kf['frame_end']}] "
                     f"type={kf['type']} grip={kf['gripper_state']}: "
                     f"\"{kf['stage'][:120]}\"")
    return "\n".join(lines)
```

---

## 6. 帧采样策略

对每个段 `[frame_start, frame_end]`：

| 段长 | 采样帧数 | 策略 |
|---|---|---|
| ≤ 4 | 全部 | 取所有帧 |
| 5-12 | 4 | 均匀采 4 张：start, 1/3, 2/3, end |
| 13-30 | 5 | 均匀采 5 张 |
| 30+ | 6 | 均匀采 6 张 |

**为什么不全采**：
- 帧间冗余高（30Hz × 段几秒 = 几十 frames，相邻帧几乎一样）
- vLLM token cost：每张 320×240 ≈ 256 vision tokens；多张图触发 OOM 或慢
- 6 张已经足够覆盖一个完整 sub-task 的视觉演化

帧来源：直接从 hdf5 解 JPEG（手动标注 GUI 用的就是这个路径）。

```python
def sample_frames_from_hdf5(hdf5_path, frame_start, frame_end, max_n=6):
    import h5py, cv2, numpy as np, base64
    with h5py.File(hdf5_path, "r") as f:
        rgb_raw = f["observation/head_camera/rgb"]
        total_frames = (frame_end - frame_start + 1)
        n = min(max_n, total_frames)
        if n <= 1:
            indices = [frame_start]
        else:
            step = (total_frames - 1) / (n - 1)
            indices = [int(round(frame_start + i * step)) for i in range(n)]
        out = []
        for idx in indices:
            buf = rgb_raw[idx]
            # JPEG bytes → re-encode as base64 (no decode needed)
            out.append(base64.b64encode(buf).decode("ascii"))
    return out, indices
```

注意：RoboTwin 在 hdf5 里存的 RGB **是 cv2.imencode(RGB) 后的 JPEG bytes**（详见 quest_teleop README §10.6），但因为我们直接把 JPEG 字节原样 base64 传给 Qwen-VL（不解码），Qwen-VL 内部用 PIL 解 JPEG —— PIL 默认按 RGB 解，所以**通道顺序正确**（cv2 那套坑只在 imdecode 路径有问题）。

---

## 7. 输出解析 + 校验

```python
import json, re

def parse_vlm_output(text: str) -> dict:
    # Strip optional ```json ... ``` fence
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    # Find outermost { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found: {text[:200]}")
    obj = json.loads(m.group())
    # Normalise: think can be "null" / "" / "None"
    th = obj.get("think")
    if th in (None, "", "null", "None"):
        obj["think"] = None
    return obj

def audit_vlm_output(obj: dict, frame_start: int, frame_end: int) -> list[str]:
    """Returns list of error messages. Empty = pass."""
    errors = []
    if obj.get("type") not in {"begin","grasp","release","retry","motion","filler","end"}:
        errors.append(f"bad type: {obj.get('type')}")
    if obj.get("gripper_state") not in {"open","partial","closed"}:
        errors.append(f"bad gripper_state: {obj.get('gripper_state')}")
    if not obj.get("stage") or len(obj["stage"]) > 350:
        errors.append("stage missing or too long (>350 chars)")
    if not obj.get("action"):
        errors.append("action missing")
    elif len(obj["action"].split()) > 14:
        errors.append(f"action too long ({len(obj['action'].split())} words, max 12)")
    if obj.get("type") == "retry" and not obj.get("think"):
        errors.append("type=retry requires think (audit A4)")
    # Action verb / gripper state consistency
    if obj.get("type") == "grasp":
        if "close" not in obj.get("action","").lower() and "grasp" not in obj.get("action","").lower():
            errors.append("type=grasp but action lacks close/grasp verb")
    if obj.get("type") == "release":
        if "open" not in obj.get("action","").lower() and "release" not in obj.get("action","").lower():
            errors.append("type=release but action lacks open/release verb")
    return errors
```

如果 audit 失败，GUI 应该 **仍然填入 VLM 输出**（让用户看到生成的内容）但在状态栏标红显示 audit error 列表，提示用户修正后再 SAVE。

---

## 8. Python client（GUI 进程 import）

```python
# scripts/vlm_client.py
import base64, json, requests, h5py, numpy as np
from typing import Optional

# Hardcoded for now; pull from env var or config later
VLM_ENDPOINT = "http://<jump>:8100/v1/chat/completions"
VLM_MODEL = "qwen2.5-vl-72b"

SYSTEM_PROMPT = """<paste §5.2 system prompt here>"""
FEWSHOT_USER = """<paste §5.3 fewshot user here>"""
FEWSHOT_ASSISTANT = """<paste §5.3 fewshot assistant JSON here>"""

def annotate_segment(*, hdf5_path: str, frame_start: int, frame_end: int,
                      task_instruction: str, plan: str,
                      prior_keyframes: list[dict],
                      timeout: float = 60.0) -> dict:
    """Call Qwen-VL to auto-fill annotation fields for one segment.

    Returns dict with keys: type, gripper_state, stage, think, action.
    On failure raises with diagnostic message.
    """
    images_b64, sampled_indices = sample_frames_from_hdf5(
        hdf5_path, frame_start, frame_end, max_n=6)
    prior_summary = fmt_prior_summary(prior_keyframes)
    user_msg = build_user_message(
        task_instruction=task_instruction, plan=plan,
        prior_keyframes_summary=prior_summary,
        frame_start=frame_start, frame_end=frame_end,
        images_b64=images_b64,
    )
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": FEWSHOT_USER},
            {"role": "assistant", "content": FEWSHOT_ASSISTANT},
            user_msg,
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(VLM_ENDPOINT, json=payload, timeout=timeout)
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    obj = parse_vlm_output(reply)
    errors = audit_vlm_output(obj, frame_start, frame_end)
    obj["_audit_errors"] = errors    # propagate to GUI for user review
    obj["_sampled_frames"] = sampled_indices
    return obj
```

---

## 9. 标注 GUI 集成（待 numbnut 这边实现）

在 `scripts/annotate_teleop_episodes.py` 加一个按钮：

```
┌─ Annotation form ─┐
│ frame_start, end  │
│ type ▾  grip ▾    │
│ stage             │
│ think             │
│ action            │
├───────────────────┤
│ [Add keyframe]    │
│ [✨ Auto-fill VLM] │  ← 新增
└───────────────────┘
```

点 `Auto-fill VLM`：
1. 读当前 `frame_start` / `frame_end`、UI 上的 `task_instruction` + `plan`、表格里已存在的 keyframe 作为 prior context
2. 调 `annotate_segment(...)`
3. 把返回的 type/grip/stage/think/action 填进 form 字段
4. 在 add_status 显示 `VLM filled; audit: <errors> -- review and edit before Add`
5. 用户复核 + 改 → 点 `Add keyframe` 正常入表

**关键**：VLM 输出**永远不直接进 keyframes 表**——必须经人复核。这保证 VLM 出锅时不污染数据集。

---

## 10. 失败 fallback / 成本控制

| 失败类型 | 处理 |
|---|---|
| 网络超时 / 5xx | 状态栏报错；用户继续手填 |
| 输出非合法 JSON | parse_vlm_output 抛错 → 状态栏 `VLM produced invalid JSON: <snippet>`，用户手填 |
| Audit error | 仍把字段填进 form，状态栏列警告，用户修后入表 |
| Pod 还没起 | client 配置 `VLM_ENDPOINT=None` → button 显示灰；用户回退手填 |

成本估算（Qwen2.5-VL-72B AWQ）：
- 单次请求：6 张 320×240 + ~2k text token ≈ 4k input + ~200 output
- 单卡 H200 throughput：~5-10 req/s（视 batch）
- 一个 episode 8 段 × 1 次/段 = 8 次调用 ≈ 1-2s wall（异步 batch 后）

每集 8 次调用 × 1000 集 = 8000 次 → 1 张 H200 跑约 15-25 分钟（如果队列起来）。

---

## 11. 后续 iteration 方向

1. **Fewshot 升级**：用 3-5 个真实标注过的 episode 段做 fewshot（带图）。等本地标 30 集后再迭代 prompt
2. **Multi-camera input**：除 head_camera 外加 left/right wrist cam → 给 VLM 更多视觉 grounding（特别是 grasp 是否成功要看 wrist 视角）
3. **Object detection grounding**：跑 GroundingDINO 先标 bbox，把 bbox 注入 prompt（"the blue cube at [126, 146, 141, 125]"）—— 对应 ECoT 论文 §4.2 step 2，提高朝向 / 位置描述准确度
4. **Audit 升级**：A3-A8 全部上线，自动 reject 不合规输出 + retry 一次
5. **本地小模型 cascade**：用 Qwen-VL 72B 做 master，再蒸馏到 Qwen-VL 7B 做 fast inference path

---

## 12. 文件索引（本 doc 涉及）

```
policy/lap/scripts/
├── README_droid_annotation.md            # 自动标 DROID 的姊妹流程（与本 doc 互参）
├── README_qwen_vlm_for_teleop_annotation.md   # 你正在读这个
├── TERMS_pick_and_place.md               # 标注英文术语对照（人 + VLM 都参考）
└── annotate_droid/                       # DROID 自动标注代码 — 本流程可复用很多组件
    ├── prompts.py                        # ← prompt 设计可借鉴
    ├── schema.py                         # ← 输出 schema + parser 可复用
    ├── audit.py                          # ← audit 规则可复用
    └── client_qwen.py                    # ← vLLM 调用 wrapper 可复用

scripts/
├── annotate_teleop_episodes.py            # 本机 GUI；待加 Auto-fill VLM 按钮
└── vlm_client.py                          # (待写) GUI 进程 import 的 client
```

---

## 13. 给本 doc 读者（部署 agent）的 acceptance criteria

完成下列才算交付：

- [ ] vLLM serve Qwen2.5-VL-72B-Instruct 在 H200 节点跑起来，`curl /v1/chat/completions` 能响应
- [ ] K8s service 暴露 + 本机通过 pod-tunnel 能 hit endpoint
- [ ] 写 `scripts/vlm_client.py`，含 §8 的 `annotate_segment(...)` 接口
- [ ] 把本 doc §5.2 / §5.3 的 system prompt + fewshot 复制进 `vlm_client.py` 顶部
- [ ] 提供一个 mock test：用任意 episode hdf5 + 任意 [start, end] 跑一次 `annotate_segment`，把返回 JSON 打到 stdout
- [ ] 写一个 README 说明 endpoint URL / model name / auth (如有) 怎么配
