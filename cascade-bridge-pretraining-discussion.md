# Cascade-VLA Bridge Pretraining — 设计讨论

本文档讨论使用 `Embodied-CoT/embodied_features_bridge` 进行 expert-0 预训练的设计决策。
对应于在 [cascade-gradient-flow-discussion.md](cascade-gradient-flow-discussion.md) 已敲定的格式 / 训练变体之上的具体执行计划。

---

## 1. Bridge ECoT 数据结构概览

数据位于：`~/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/snapshots/.../embodied_features_bridge.json`

**注意**：该数据集**仅包含 ECoT 标注**（1.4 GB JSON），不包含图像。图像通过 `file_path` 字段引用原始 Bridge V2 .npy 文件（位于 Berkeley NFS，本地不可用）。

### 顶层布局
```
{
  "<original_npy_file_path>": {        # 例: ".../stack_blocks/19/train/out.npy"
    "<episode_id>": {
      "metadata": {
        "episode_id": str,
        "file_path": str,
        "n_steps": int,
        "language_instruction": str    # 用户级原始指令
      },
      "features": {
        "move_primitive": [str] * n_steps,    # "stop" / "move up" / "close gripper" / ...
        "gripper_position": [[u, v]] * n_steps,
        "bboxes": [[[conf, label, [x1,y1,x2,y2]], ...]] * n_steps
      },
      "reasoning": {
        "0": {task, plan, subtask, subtask_reason, move, move_reason},
        "1": {...},
        ...
      }
    }
  }
}
```

### Reasoning 字段语义（关键）

| 字段 | 粒度 | 例 | 是否使用 |
|------|------|-----|---------|
| `task` | episode-level | "Move the wooden arch onto the table." | ✅ → `<prompt>` |
| `plan` | episode-level（恒定） | "Reach for the wooden arch. Grasp ... Move ... Drop ..." | ✅ → `<plan>` 或 `<prompt>` |
| `subtask` | phase-level（典型 10–15 步切换一次） | "Reach for the wooden arch." | ✅ → `<langact>` |
| `subtask_reason` | phase-level | "The wooden arch is the object that needs to be moved..." | ✅ → `<reasoning>` |
| `move` | per-step（每帧） | "move up" / "stop" / ... | ❌ 不用（用户明确：自回归 VLA 不取） |
| `move_reason` | per-step | "The arm needs to move up..." | ❌ 不用 |

### 实际样本验证（episode 43）

Phase 切换点：
- step 0–13：`subtask = "Reach for the wooden arch."`
- step 14–22：`subtask = "Grasp the wooden arch."`
- step 23–34：`subtask = "Move the wooden arch to the table."`
- step 35–39：`subtask = "Drop the wooden arch onto the table."`

`plan` 在所有 40 步保持完全相同。这印证了 plan 是 episode-level、phase 切换不变。

---

## 2. 格式映射决策

### 2.1 我们的两段格式 → Bridge 字段映射

```
[BOS] <task> [plan] <plan> [think] <subtask_reason> [action] <subtask> [EOS]
        ↑       ↑              ↑                          ↑
       task    plan       subtask_reason              subtask
   (episode)(episode)      (phase)                    (phase)
```

### 2.2 关于 `plan` 应该放哪里 —— **采用独立 [plan] 段**

**用户提出的两种候选**：
- (A) `<image><task>[think]<plan + subtask_reason>[action]<subtask>` — plan 并入 reasoning
- (B) `<image><task>[plan]<plan>[think]<subtask_reason>[action]<subtask>` — plan 独立段

**结论：采用方案 B（独立 [plan] 段）**。

**理由**：

1. **粒度本质不同**。task 是"做什么"（episode-level WHAT），plan 是"怎么做的整体规划"（episode-level HOW），subtask_reason 是"当前 phase 为什么这么做"（phase-level）。混在一起后，方案 A 的 `[think]` 段每个 phase 都要重复 plan 内容（plan 不变），训练时 next-token CE loss 会被冗长固定文本主导。

2. **与 §10.2 的"phase 边界 AR 不依赖前 phase reasoning"决策一致**。方案 B 把 plan 放在 phase 边界**之上**的持久 prefix 里：

   ```
   持久 prefix（每个新 phase AR 都看到）:
       <image_t><task>[plan]<plan>
   per-phase 部分（每个新 phase AR 重新生成，旧的丢弃）:
       [think]<subtask_reason_i>[action]<subtask_i>
   ```

   方案 A 把 plan 和 subtask_reason 混在 `[think]` 段，phase 切换时 plan 也会被丢掉，用户必须每次 AR 都重新生成 plan（代价：长输出 + 每 phase 都要再现 plan 一字不差，模型容易学糊）。

3. **可消融**。方案 B 给 `[plan]` 段独立的 mask，未来可做"是否屏蔽 plan / 是否在新 phase mask plan / 等"消融。方案 A 没法做这种消融。

### 2.3 token 标记符的选择

考虑到 PaliGemma tokenizer 是 SentencePiece + 257k 词表，方括号包裹的标记会被切成多个 subword，不是问题。建议：

- `[plan]` — 持久段开始
- `[think]` — phase-level reasoning 开始
- `[action]` — phase-level langact 开始
- 不需要 `[/plan]` / `[/think]` 闭合，因为下一个标记的出现就隐式闭合了上一个

### 2.4 两段 mask 在 Bridge 数据下的具体含义

回顾 [tokenizer.py 改造](policy/lap/src/lap/models/tokenizer.py)：

| Mask | Bridge 数据下覆盖范围 | 用于 |
|------|--------------------|------|
| `tokenized_langact_mask` | `[think]<subtask_reason>[action]<subtask>` 全部 | next-token CE loss + ar_mask |
| `tokenized_reasoning_mask` | 只覆盖 `[think]<subtask_reason>` 段 | action attention 屏蔽（本次预训练 disable action 不会用到，但保留以便后续 fine-tune） |

`<task>` 和 `[plan]<plan>` **不在任何 mask 内**，它们是给定的 prefix 条件，prompt_mask 已涵盖。

---

## 3. 用户问题逐条回答

### Q1: 数据合成

> "可以合成，不过这次训练先不用这些数据，只用 bridge"

✅ 同意。pick_place_primitive 的 subgoal 合成留到下一阶段。

### Q2: 关于 `<task>` vs `<reasoning>` 的边界

> "为了应对数据集之前的差异，我们可能要明确一下 `<task>[think]<reasoning_i>[action]<langact_i>` 一部分文本应该属于 `<reasoning_i>` 还是属于 `<task>`，由于 10.2，在新的 phase 的 AR 不会 condition 之前的 `[think]<reasoning_i>`"

**进一步澄清**：用户的核心担心是——如果某条信息**phase 间应该共享**，那它必须在 `<task>` 或 `[plan]` 里，**不能**在 `[think]<reasoning_i>` 里（否则 phase 切换后这条信息就丢了）。

**判断准则**：
| 信息层级 | 应该放哪 | 例 |
|---------|---------|-----|
| Episode 级常量（任务什么时候都不变） | `<task>` | "Move the wooden arch onto the table." |
| Episode 级长期计划（phase 间不变） | `[plan]<plan>` | "Reach ... Grasp ... Move ... Drop ..." |
| Phase 级（每个 phase 重新生成） | `[think]<reasoning_i>` | "The wooden arch is the object that needs to be moved..." |
| Phase 级动作描述 | `[action]<langact_i>` | "Reach for the wooden arch." |

**关键 invariant**：`[think]` + `[action]` 段是"可丢弃单元"。Phase 切换时这部分整段丢弃 + 重新生成，模型不会再看到它。

### Q3: 失败 phase 处理

> "失败的 phase 先不纳入训练，除非以后在失败的基础上有 recovery 的数据"

✅ 同意。Bridge 数据的 `metadata.episode_success` 不存在，但 episode 通常都是成功示范（Bridge V2 整体是 success demos）。本次训练全收。

### Q4: action chunk 跨 phase 时 langact 取哪个

由于本次预训练 **disable action expert**（仅 expert 0 / 仅语言 CE loss），这个问题不影响本次。
留作 fine-tune 阶段决策。**本次直接用 chunk 起始帧的 langact**（最自然、与训练数据对齐）。

### Q10.5 重申: phase 切换时是否要"推掉" `[think]...[action]...`

> "推理时，如果在 phase 中再次中间观察，emit 同一个 langact；或者跨入下一个 phase，emit 了新的 langact，问题：是不是要把序列中的 `[think]...[action]...` 推掉，只 condition `<task>` 或者 `<task><plan>` 预测新的 langact"

**答：是的，需要"推掉"**。具体实现两种等价方法：

**方法 A — 重置 KV cache（推荐起步，简单）**
```python
# 检测到 phase 切换（langact 文本变化或固定每 chunk 重生成）
kv_cache = None  # 丢弃前 phase 的 KV
# 重新跑 prefix
prefix_tokens = [<image_new>, <task>, [plan], <plan>]
kv_cache = trunk_prefill(prefix_tokens)
# AR 生成新 phase 的 think + action
new_reasoning, new_langact = ar_decode(kv_cache, max_steps=64)
```
缺点：每 phase 切换都要重跑 image+task+plan 的前向。优化空间大但实现复杂。

**方法 B — 保留 KV cache + 屏蔽 attention（推荐熟悉后优化）**
```python
# 保留 KV cache 不变，但生成新 phase 的 token 时使用一个特殊 attention mask:
# 新 token 只能 attend [image, task, plan]，不能 attend 旧的 [think][action]
# 实现: 给 KV cache 中旧 [think][action] 段位置的 attention 设为 False
```
更高效（避免重算 image SigLIP），但需要精确管理 mask 边界。

**训练侧自然支持**：因为训练时每个样本就是 `(image_t, task, plan, subtask_reason_at_t, subtask_at_t)`，模型从来没见过"前 phase 的 [think][action] 在 prefix 里"的情况，所以推理时不放进去也是分布内的。

### 用户提到的"使用 bridge 不使用 action 数据，只训练 expert 0"

✅ 完全可行。具体配置：
- `enable_action_training=False` → 单 expert 架构（[lap.py:40-74](policy/lap/src/lap/models/lap.py#L40-L74) 走 else 分支）
- `enable_langact_training=True`
- 不需要 action_dim / action_horizon 字段（虽然 dataclass 仍要求填）

---

## 4. ⚠️ 关键阻塞：Bridge V2 图像数据

### 4.1 现状

```
本地有：
  ✓ ECoT 标注 JSON （1.4 GB，含 task / plan / subtask / subtask_reason / move / move_reason）

本地缺：
  ✗ Bridge V2 原始图像 .npy 文件（参考路径形如 /nfs/kun2/users/homer/...）
```

VLM 训练**必须有图像**。当前 `embodied_features_bridge` 单独使用没法做 VLM 预训练（只能做纯文本 LM 训练，意义不大）。

### 4.2 三个解决方向

| 方向 | 描述 | 复杂度 | 数据完整性 |
|------|------|--------|-----------|
| **(I) 下载 OXE TFDS Bridge** | `gs://gresearch/robotics/bridge/...` 或 `tfds.builder("bridge")` | 中（~400 GB 数据） | ✅ 有图像有 action |
| **(II) 下载 OpenX HF Bridge** | `IPEC-COMMUNITY/bridge_orig_lerobot` （HuggingFace LeRobot 格式） | 低 | ✅ 有图像 |
| **(III) 下载 Bridge V2 raw** | 从 Berkeley RAIL 直接下载（需要确认链接） | 中 | ✅ 有图像有 metadata |

**关键挑战**：上述任意一种获得的图像数据，要和 ECoT JSON 的 `file_path` + `episode_id` + step index **精确对齐**。理论上 file_path 是匹配键，但需要验证：
- ECoT JSON 的 file_path 是绝对 NFS 路径（`/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/...`）
- 下载的 Bridge V2 是某种 TFDS 或 LeRobot 格式，**没有这个路径作 key**
- 需要建立映射：file_path 路径中的子目录结构（`bridge_data_v2/<workspace>/<task_subdir>/<run_id>/<split>/out.npy`）→ TFDS / LeRobot 中的 trajectory id

### 4.3 推荐路径

**方向 (II) - HuggingFace LeRobot Bridge** 最务实：
- 已经在 HF 缓存基础设施上，下载方便（`huggingface-cli download IPEC-COMMUNITY/bridge_orig_lerobot --repo-type dataset`）
- LeRobot 格式有标准 episode_index → frames 索引
- 与 ECoT 对齐需要写一个映射脚本，但可一次性预处理

**Alternative**：直接放弃文件路径精准对齐，用 ECoT 的 `language_instruction` 字段在 LeRobot 数据里做近似匹配（每个 episode 用 instruction + episode_index 模糊配对）。简单但可能错配少量样本。

### 4.4 本次实施先做什么

由于图像数据未确认，**本次先完成**：

1. 数据 schema 定义和 dataloader 接口
2. Bridge ECoT JSON 解析器
3. 训练 config 草案（Bridge-pretrain 配置）
4. 占位的图像 loader（可插拔，待图像数据到位后填充）

**不做**：
- 实际启动训练（无图像数据）

待用户决定图像数据来源后，再补完 dataloader 的图像 IO 部分并启动训练。

---

## 5. 训练 config 提案

```python
TrainConfig(
    name="lap_bridge_pretrain",
    model=lap_config.LAPConfig(
        action_dim=7,                          # 占位（不用）
        action_horizon=1,                      # 占位（不用）
        max_token_len=256,                     # task + plan 偏长，留余量
        pi05=True,
        discrete_state_input=False,            # bridge 没有 state 向量
        # === 仅训练 expert 0 ===
        enable_action_training=False,          # 关键：单 expert 架构
        enable_langact_training=True,
        enable_prediction_training=False,
        enable_vqa_training=False,
        # === Cascade-VLA 配置（预训练阶段不重要，但保持一致） ===
        action_attention_mode="lap_original",  # 单 expert 时不生效
        stop_grad_mode="off",                  # 单 expert 时不生效
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",    # 占位（不用）
        prompt_format="lap",
        language_loss_weight=1.0,
        enable_image_augmentation=True,        # 标准图像增广
    ),
    data=BridgeECoTDataConfig(...),            # 新增 — 见下
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=2000,
        peak_lr=5e-5,
        decay_steps=100_000,
        decay_lr=5e-6,
    ),
    save_interval=5000,
    keep_period=5000,
    num_train_steps=100_001,
    batch_size=128,
    weight_loader=weight_loaders.WeightLoaderChoice(
        kind="paligemma",
        params_path="checkpoints/paligemma-2b-mix-224",
    ),
)
```

### 新 DataConfig：`BridgeECoTDataConfig`

需要新建一个 `RLDSDataConfig` 的姐妹类，或者直接复用并扩展。要点：

```python
@dataclasses.dataclass(frozen=True)
class BridgeECoTDataConfig(BaseDataConfigFactory):
    # ECoT JSON 路径
    ecot_json_path: str = "~/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/snapshots/.../embodied_features_bridge.json"
    # 图像数据源（待定）
    bridge_images_root: str | None = None
    # 是否使用 plan 段
    include_plan: bool = True
    # task / plan / subtask / subtask_reason 字段名（与 JSON 一致）
    use_fields: tuple[str, ...] = ("task", "plan", "subtask_reason", "subtask")
    # phase 切换时是否做 prefix 重置（仅推理用，不影响训练）
    reset_kv_at_phase_boundary: bool = True

    # 关闭 OXE 特有字段
    repo_id: str = "bridge_ecot"
    asset_id: str = "bridge_ecot"
```

---

## 6. DataLoader 实现思路

由于 Bridge 不是 RLDS 格式，不能直接用 `BaseRobotDataset`。需要写一个新的 loader：

```python
class BridgeECoTDataset:
    """Loads (image, ECoT-text) pairs from Bridge V2 + ECoT annotations."""

    def __init__(self, ecot_json_path, bridge_images_root, ...):
        # Stream-parse ECoT JSON with ijson (1.4 GB 不要全加载)
        self.episodes = list(self._stream_episodes(ecot_json_path))
        self.image_loader = BridgeV2ImageLoader(bridge_images_root)

    def __iter__(self):
        for ep in self.episodes:
            for step_idx in range(ep["n_steps"]):
                image = self.image_loader.get(ep["file_path"], ep["episode_id"], step_idx)
                r = ep["reasoning"][str(step_idx)]
                yield {
                    "image": image,
                    "prompt": self._build_prompt(r["task"], r["plan"]),    # see below
                    "language_actions": r["subtask_reason"],   # → reasoning
                    "langact": r["subtask"],                   # → langact
                    # 标记位
                    "is_vqa_sample": False,
                    "is_prediction_sample": False,
                    "sample_mask": True,
                }

    @staticmethod
    def _build_prompt(task: str, plan: str | None) -> str:
        if plan is not None:
            return f"{task} [plan] {plan}"
        return task
```

**关键设计**：
- `prompt = task + " [plan] " + plan` —— 把 plan 拼到 prompt 字符串末尾，由 prompt module 整体作为持久 prefix
- `language_actions = subtask_reason` —— 走现有 `[think]` 段路径（tokenizer 现有的 reasoning 字段）
- `langact = subtask` —— 走我们新加的 `[action]` 段路径
- 不需要 action / state / wrist_image —— 单 expert 架构会忽略

### Bridge V2 图像加载器（占位）

```python
class BridgeV2ImageLoader:
    """Maps (file_path, episode_id, step_idx) -> RGB image array.

    TODO: 待 Bridge 图像数据下载完成后实现具体逻辑：
      - 方案 II: HuggingFace LeRobot bridge_orig
      - 方案 I: OXE TFDS bridge
      - 方案 III: Bridge V2 raw .npy
    """
    def get(self, file_path: str, episode_id: str, step_idx: int) -> np.ndarray:
        raise NotImplementedError("Bridge V2 image source not yet configured")
```

---

## 7. 实施步骤清单

| # | 任务 | 状态 | 备注 |
|---|------|------|------|
| 1 | ~~新增 `[plan]` segment 支持到 tokenizer~~ | ✅ N/A | 改用更简方案：plan 拼到 prompt 字符串末尾 (`task + " [plan] " + plan_text`) |
| 2 | 写 `BridgeECoTDataset` 数据类 | ✅ 完成 | [bridge_ecot_dataset.py](policy/lap/src/lap/datasets/bridge_ecot_dataset.py) 流式 JSON 解析 + 按 step 产样本 |
| 3 | 写 `BridgeV2ImageLoader` 接口 | ✅ 占位完成 | 同上文件，含 `NullImageLoader` 占位 |
| 4 | 在 `training/config.py` 加 `lap_bridge_pretrain` config | ✅ 完成 | 加载验证通过 |
| 5 | 加 `BridgeECoTDataConfig` 到 config 模块 | ✅ 完成 | 同上 |
| 6 | 端到端单样本验证脚本 | ✅ 完成 | [test_bridge_ecot_pipeline.py](policy/lap/scripts/test_bridge_ecot_pipeline.py) 跑通 3 样本 |
| 7 | 修改 train.py 入口让 dataloader 工厂能识别 Bridge config | **待** | RLDS path 与 Bridge ECoT 不同，需新分支 |
| 8 | 解决图像数据来源 | **🚧 阻塞** | 需用户决策 §4.3 三个方向 |
| 9 | 端到端跑通 1 个 batch（含真图） | 待 | 验证 token 化、masks、loss 计算无错 |
| 10 | 启动 100K step 预训练 | 待 | 评估 langact_acc / reasoning perplexity 等 |

### 已实现部分的验证

```bash
cd policy/lap
.venv/bin/python -m pip install ijson  # 一次性
.venv/bin/python scripts/test_bridge_ecot_pipeline.py --num-samples 3 --skip-repeat
```

输出确认：
- ✅ ECoT JSON 流式加载（无 OOM）
- ✅ 三个 phase 切换样本 (Reach → Grasp → Move) 都被正确解析
- ✅ Prompt 包含 task + `[plan]` + plan 完整文本
- ✅ Tokenizer 产出 `tokenized_langact_mask`（reasoning + langact 联合）和 `tokenized_reasoning_mask`（仅 reasoning）
- ✅ Decode 验证：解码出的 [think] 段和 [action] 段文本与输入完全一致
- ✅ Mask 互斥性 assert 通过

### 未实现部分（按优先级）

**P0 - 用户必决**：图像数据来源（§4.3 三选一）

**P1 - 可与 P0 并行**：把 `BridgeECoTDataset` 接入 `data_loader.create_data_loader`。当前的 RLDS-only 路径需要新增分支判断 `isinstance(data_cfg, BridgeECoTDataConfig)` 走 Bridge 专用 dataloader。

**P2 - 等 P0 完成后**：在 `BridgeECoTDataConfig` 里把 `bridge_images_root` 字段实际接到 image loader（`LeRobotBridgeImageLoader` 或 `RawNpyBridgeImageLoader`），然后启动训练。

---

## 8. 留给用户的决策点

1. **图像数据源**：请确认走 §4.3 的哪一条（推荐 II = HF LeRobot bridge_orig）。需要约 ~50 GB 磁盘空间和约 1 小时下载时间。
2. **是否启用 plan 段**：默认 `include_plan=True`。可考虑做"with vs without plan"消融，看 plan 是否真的提升 phase-level reasoning 质量。
3. **训练目标 token_len**：当前提案 256。task + plan + subtask_reason + subtask 累计经验上 ~150-200 tokens，留余量。先小批 dry-run 看实际长度分布。
4. **是否保留 `move_primitive` 字段做辅助**：每帧 `move` ("move up" / "stop" / ...) 是 atomic 动作语义，可以作为额外的 per-step VQA 任务（训 expert 0 输出 move primitive）。**本次先不用**，留着以后做。
5. **预训练后下游评估**：跑 RoboTwin 仿真还是 LIBERO？建议两边都跑，因为预训练目的就是让模型学会"看图说 reasoning + langact"，下游迁移到 RoboTwin/LIBERO 时再加 action expert fine-tune。
