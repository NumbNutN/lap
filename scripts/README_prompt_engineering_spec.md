# Prompt Engineering Spec：大规模 VLM 标注接口契约

> **owner**: 部署 VLM 服务的 agent（可修改）
> **依据**: [`README_annotation_design.md`](README_annotation_design.md) §7 + §11
> **基础设施 doc**: [`README_qwen_vlm_for_teleop_annotation.md`](README_qwen_vlm_for_teleop_annotation.md)

这个 doc 定义 **VLM 自动标注的 prompt 接口**：
- 输入格式 (frames + context)
- 输出 schema (JSON 字段 + marker token)
- system prompt 模板
- fewshot 设计原则
- audit 规则
- 迭代流程

下游接口（teleop GUI 调 `vlm_client.annotate_segment(...)`）契约写在这里。**改这个文档 = 改 prompt = 影响所有自动标注的 output**。

---

## 1. 输出 JSON schema（**严格**）

VLM 必须返回 STRICT JSON，字段与手工标注完全一致 + 1 个新字段 `mode_marker`：

```json
{
  "mode_marker": "[stage] | [act] | [think_act]",   // 见 §2
  "type": "begin|grasp|release|retry|motion|filler|end",
  "gripper_state": "open|partial|closed",
  "stage": "<15-40 words; describes current state + IMAGE-INVISIBLE history (plan progress / past failures / counters)>",
  "think": "<null OR 1-2 sentences for retry/non-obvious decision>",
  "action": "<5-12 words; next-step intent; imperative>"
}
```

### 1.1 字段语义（不可改）

| 字段 | 内容 | 约束 |
|---|---|---|
| `type` | 关键帧类别 | 7-class 枚举 |
| `gripper_state` | 段末夹爪状态 | open/partial/closed |
| `stage` | **当前段状态 + 图像看不出的历史摘要** | 15-40 词，陈述句 |
| `think` | 失败反思 / 非显然决策推理 | null 或 1-2 句；retry 必填 |
| `action` | **下一步意图** | 5-12 词，祈使句 |
| `mode_marker` | VLM 决定生成哪个模式的输出 | 见 §2 |

### 1.2 `stage` 写法关键 (memory-augmented)

| ✅ DO | ❌ DON'T |
|---|---|
| "Having released the first cyan cube on the red target, the gripper hovers above the second cyan cube ready for the next pick." | "The gripper hovers above the cyan cube." (没历史) |
| "This is the second pair in plan step 2; the first attempt failed." | "Stack the cyan cube on the orange cube" (plan 复读) |
| "The robot is now grasping; previous attempt closed empty." | "As mentioned in the previous keyframe..." (无效引用) |

**核心**: stage 内容必须包含**至少一个图像无法唯一推断的事实**（plan 进度 / 重试计数 / 子任务编号 / 上一次结果）。

### 1.3 `think` 触发规则

| 场景 | think 是否填 |
|---|---|
| retry / 失败恢复 | **必填**（A4 audit） |
| 多步规划决策（"先抓哪个"） | **应填** |
| 避障 / 朝向选择 | **应填** |
| 不可见信息推理（"为什么 target 是 X"） | **应填** |
| 常规 approach / transport / release | **null** |

目标 think 覆盖率: **30-40% keyframe**（详见 design doc §11.4 / §11.5）。

---

## 2. Mode marker（用于 multi-objective 训练）

> 这是给 **训练消费者** 用的字段，VLM 在标注阶段不必区分模式，统一按 `[think_act]` 模式输出（最丰富的字段集），下游训练时根据 marker 切分。

| Marker | 训练时输出 | 推理时触发 |
|---|---|---|
| `[stage]` | 仅输出 stage 描述（VQA 任务） | 用户 query 描述当前状态 |
| `[act]` | 仅输出 action | 实时部署（节省延迟）|
| `[think_act]` | 输出 think + action | 复杂决策（retry / 避障）|

**标注阶段简化**：每条 VLM 标注**默认 `mode_marker = "[think_act]"`**，下游训练 sampler 根据需要把同一条 sample 用作三种 mode 的训练样本（mask 不同字段）。

### 训练采样矩阵（参考）

| Marker | 含义 | 采样比例 | 输入 | 输出（loss target） |
|---|---|---|---|---|
| `[stage]` | VQA | 25% | image + history + `[stage]` | stage 文本 |
| `[act]` | Policy | 40% | image + history + `[act]` | action 文本 |
| `[think_act]` | Reason+act | 25% | image + history + `[think_act]` | think + action |
| `[plan]` | 全集规划 | 10% | episode 首帧 + task_instruction + `[plan]` | plan 文本 |

---

## 3. System prompt（v1，可迭代）

```text
You are an embodied-AI annotator. Given a sequence of video frames showing
a robot arm performing a manipulation task, plus the episode-level task
description, plan, and PRIOR keyframe annotations as memory context, you
produce a structured ECoT annotation for the specific frame range shown.

Output STRICT JSON with exactly these six fields:
  - mode_marker: always "[think_act]" (used by downstream training sampler)
  - type: one of "begin", "grasp", "release", "retry", "motion", "filler", "end"
  - gripper_state: one of "open", "partial", "closed" (state at the END of this segment)
  - stage: 15-40 words. Describe the CURRENT state of the robot AND any
    image-invisible context the model needs to know -- specifically:
      * Plan-step progress ("this is the second pair in step 2")
      * Past failures ("after a failed grasp attempt")
      * Counters ("the third cube has been placed")
      * Cross-keyframe causality ("having released the previous cube")
    Use "The robot..." as subject. Do NOT just repeat the plan field.
    Do NOT say "as mentioned in the previous keyframe" -- be self-contained.
  - think: null UNLESS the segment requires reasoning that cannot be derived
    from the image alone. Fill (1-2 sentences) when:
      * type == "retry" (REQUIRED)
      * Multi-step planning decision ("picking leftmost first to free space")
      * Avoidance / orientation choice ("lifting higher to clear obstacle")
      * Reasoning about invisible info ("target is orange because we already placed on red")
  - action: 5-12 words, imperative sentence ("Move...", "Close...", "Lift...").
    Describes the NEXT-step intent. Use object names like "blue cube",
    "leftmost red cube" for consistency. Use axis names ("yaw counterclockwise",
    "tilt up") rather than bare directions.

Type semantics:
  - begin: first segment; robot at ready pose
  - grasp: gripper closes around an object; gripper_state ends "closed"
  - release: gripper opens and releases; gripper_state ends "open"
  - retry: previous action failed; MUST fill think
  - motion: transport / approach / re-orient; no grip-state change
  - filler: long gap between meaningful events
  - end: last segment; robot returns to ready pose

If uncertain about gripper_state from images alone, use the prior segment's
state and any visible gripper width as a hint.

Do NOT include markdown fences. Do NOT add explanatory prose. Output ONLY
the JSON object.
```

---

## 4. User message 模板

```python
def build_user_message(*, task_instruction, plan,
                        prior_keyframes_summary, frame_start, frame_end,
                        images_b64):
    text = f"""Task: {task_instruction}
Plan: {plan}

PRIOR KEYFRAMES IN THIS EPISODE (memory context — these have already happened):
{prior_keyframes_summary}

CURRENT SEGMENT to annotate: frames [{frame_start}..{frame_end}]
Images below are sampled uniformly from this range.

Your stage should reference relevant prior keyframes (progress, failures, counters)
that the image alone cannot convey."""
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
        line = (f"- frame [{kf['frame_start']}..{kf['frame_end']}] "
                f"type={kf['type']} grip={kf['gripper_state']}: "
                f"\"{kf['stage']}\"")
        if kf.get('think'):
            line += f" [think: {kf['think']}]"
        lines.append(line)
    return "\n".join(lines)
```

**关键设计**：prior_summary 包含**完整 stage 文本**（不截断），让 VLM 看到 memory chain 全貌再决定本段如何呼应。

---

## 5. Fewshot 例子（v1，1 example）

### FEWSHOT_USER (text-only, no images)

```text
Task: Stack the cyan cube on top of the leftmost red cube, avoiding the middle cube on the path.
Plan: 1. Approach and grasp the cyan cube. 2. Lift and arc over the middle cube. 3. Lower onto the leftmost red cube. 4. Open the gripper to release.

PRIOR KEYFRAMES IN THIS EPISODE (memory context — these have already happened):
- frame [0..8] type=begin grip=open: "Robot starts at the ready pose above the table; this is the start of plan step 1."
- frame [9..27] type=motion grip=open: "The robot approaches the cyan cube from above, beginning plan step 1's pick phase."
- frame [28..36] type=grasp grip=closed: "The gripper has closed firmly around the cyan cube. Plan step 1 grasp completed successfully on the first attempt." [think: null]

CURRENT SEGMENT to annotate: frames [37..62]
Images below are sampled uniformly from this range.

Your stage should reference relevant prior keyframes (progress, failures, counters)
that the image alone cannot convey.

[images]
```

### FEWSHOT_ASSISTANT (strict JSON, no fence)

```json
{
  "mode_marker": "[think_act]",
  "type": "motion",
  "gripper_state": "closed",
  "stage": "Having grasped the cyan cube, the robot now lifts it high and arcs the cube over the middle cube. This is the obstacle-avoidance phase of plan step 2.",
  "think": "The middle cube is in the direct path to the leftmost red target. Lifting above its top before translating sideways avoids collision.",
  "action": "Lift and arc over the middle cube"
}
```

### 迭代建议

- v1: 1 个 text-only fewshot（最便宜）
- v2 (after pilot 100): 3-5 个 fewshot **带图**，覆盖 grasp / release / retry / obstacle 各场景
- v3: instance-aware few-shot retrieval (从已标 episode 里 retrieve top-3 相似段)

---

## 6. 帧采样策略（输入侧）

| 段长 (frame_end - frame_start + 1) | 采样帧数 | 策略 |
|---|---|---|
| ≤ 4 | 全部 | 取所有 |
| 5-12 | 4 | 均匀采: start, 1/3, 2/3, end |
| 13-30 | 5 | 均匀采 5 张 |
| 30+ | 6 | 均匀采 6 张 |

来源：hdf5 `observation/head_camera/rgb`（已 JPEG-encoded bytes，可直接 base64 后给 Qwen-VL 不解码）。

---

## 7. Audit 规则（输出侧 reject）

```python
def audit_vlm_output(obj, frame_start, frame_end):
    errors = []
    # A0: schema completeness
    for field in ("mode_marker","type","gripper_state","stage","action"):
        if not obj.get(field) and field != "think":
            errors.append(f"missing field: {field}")

    # A1: type in canon
    if obj.get("type") not in {"begin","grasp","release","retry","motion","filler","end"}:
        errors.append(f"bad type: {obj.get('type')}")

    # A2: gripper_state in canon
    if obj.get("gripper_state") not in {"open","partial","closed"}:
        errors.append(f"bad gripper_state: {obj.get('gripper_state')}")

    # A3: action verb / type consistency
    act = obj.get("action","").lower()
    if obj.get("type") == "grasp":
        if not any(v in act for v in ("close","grasp")):
            errors.append("type=grasp but action lacks close/grasp verb")
    if obj.get("type") == "release":
        if not any(v in act for v in ("open","release")):
            errors.append("type=release but action lacks open/release verb")

    # A4: retry requires think
    if obj.get("type") == "retry":
        th = obj.get("think")
        if not th or not th.strip():
            errors.append("type=retry requires think (audit A4)")

    # A5: length limits
    if len(obj.get("stage","")) > 350:
        errors.append("stage too long (>350 chars)")
    if len(obj.get("action","").split()) > 14:
        errors.append(f"action too long ({len(obj['action'].split())} words)")

    # A6: stage should NOT be empty / pure plan repeat
    if not obj.get("stage","").strip():
        errors.append("stage missing")
    # (light heuristic — actual plan-repeat detection is hard; rely on
    # human spot-check or sentence similarity later)

    # A7: gripper / action consistency
    grip = obj.get("gripper_state")
    if grip == "open" and "grasp" in act:
        errors.append("gripper_state=open but action says grasp")
    if grip == "closed" and "release" in act:
        errors.append("gripper_state=closed but action says release")

    # A8: mode_marker fixed at annotation time
    if obj.get("mode_marker") != "[think_act]":
        errors.append(f"mode_marker must be [think_act] for annotation, "
                      f"got {obj.get('mode_marker')}")

    return errors
```

### Audit fail handling

- **errors empty** → mark as `audit_pass=True`，写入 dataset
- **errors non-empty** → mark `audit_pass=False`，**仍然写入** JSONL 行带 `audit_errors` 列表，便于 retry / 人 review

---

## 8. 集成入口（vlm_client.py 期望签名）

```python
def annotate_segment(*,
    hdf5_path: str,
    frame_start: int,
    frame_end: int,
    task_instruction: str,
    plan: str,
    prior_keyframes: list[dict],   # 同 episode 已标注的全部 keyframes
    timeout: float = 60.0,
) -> dict:
    """
    Return:
      {
        "mode_marker": "[think_act]",
        "type": ..., "gripper_state": ..., "stage": ..., "think": ..., "action": ...,
        "_audit_errors": [...],
        "_sampled_frames": [...],   # indices into hdf5
        "_latency_ms": float,
        "_vlm_raw": str,            # raw VLM text response (for debug)
      }
    """
```

字段以 `_` 开头是元数据，不写入最终 cot_annotations JSON。

---

## 9. 性能 / 成本

| 项 | 数字 (Qwen2.5-VL-72B AWQ on 1× H200) |
|---|---|
| 单 segment 输入 | 6 frames × 320×240 ≈ 1500 vision tokens + 2k text tokens |
| 单 segment 输出 | ~150-250 tokens (full JSON) |
| 单 segment latency | 1.5-3s (depends on queue) |
| 单 episode (8 segments) | ~15-25s sequential / ~5s batched |
| 1000 episodes | ~30-90 min wall clock |
| 成本 | 1× H200 self-hosted ≈ free if pod 已起 |

---

## 10. 迭代流程

```
v1 (now)
  └── 1 text-only fewshot + system prompt §3 + audit §7
  └── annotate 100 ep pilot
  └── 人审 random 20 ep，打分 (1-5)：accuracy / completeness / style

v2 (after pilot)
  └── 升级 fewshot 到 3-5 带图例子（covering grasp/release/retry/obstacle）
  └── 调整 system prompt 根据 v1 failure 模式
  └── annotate 全集

v3 (mature)
  └── instance retrieval fewshot
  └── multi-camera input (head + wrist)
  └── GroundingDINO bbox prefix
```

每次迭代写一个 `v{N}_changelog.md` 说明：
- 改了哪段 prompt
- pilot acceptance 从多少 → 多少
- 触发的失败模式

---

## 11. owner 修改本 doc 的指南

这个 doc 是**接口契约**，标注的另一个 agent 可以改。改的时候**必须**：

1. **schema 改动**（§1）→ 同步通知本机 GUI 维护者（@numbnut），需要更新 GUI 的字段渲染 + 训练 dataset class
2. **mode_marker 改动**（§2）→ 同步通知训练 sampler 维护者
3. **system prompt 改动**（§3）→ 必须 changelog 记录原因 + 跑一次 pilot 100 ep 看 acceptance 变化
4. **audit 规则改动**（§7）→ 同步更新 [`README_annotation_design.md`](README_annotation_design.md) §7 的人工标注规则保持一致

修改不影响**已存在数据**：所有 cot_annotations JSON 都 forward-compatible（新字段可选；旧字段不删除）。

如果要做 breaking change（删字段 / 改语义）：开 issue / 直接找 @numbnut 讨论先。

---

## 12. 引用关系

- 设计哲学：[`README_annotation_design.md`](README_annotation_design.md)
- 部署 / 集群：[`README_qwen_vlm_for_teleop_annotation.md`](README_qwen_vlm_for_teleop_annotation.md)
- 标注术语表：[`TERMS_pick_and_place.md`](TERMS_pick_and_place.md)
- 手标 GUI：[`scripts/annotate_teleop_episodes.py`](../../../scripts/annotate_teleop_episodes.py)
- 标注 schema 原型：[`annotate_droid/schema.py`](annotate_droid/schema.py)
