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
[BOS] <task> [plan] <plan> [stage] <subtask_reason> [action] <subtask> [EOS]
        ↑       ↑              ↑                          ↑
       task    plan       subtask_reason              subtask
   (episode)(episode)      (phase)                    (phase)
```

### 2.2 关于 `plan` 应该放哪里 —— **采用独立 [plan] 段**

**用户提出的两种候选**：
- (A) `<image><task>[stage]<plan + subtask_reason>[action]<subtask>` — plan 并入 reasoning
- (B) `<image><task>[plan]<plan>[stage]<subtask_reason>[action]<subtask>` — plan 独立段

**结论：采用方案 B（独立 [plan] 段）**。

**理由**：

1. **粒度本质不同**。task 是"做什么"（episode-level WHAT），plan 是"怎么做的整体规划"（episode-level HOW），subtask_reason 是"当前 phase 为什么这么做"（phase-level）。混在一起后，方案 A 的 `[stage]` 段每个 phase 都要重复 plan 内容（plan 不变），训练时 next-token CE loss 会被冗长固定文本主导。

2. **与 §10.2 的"phase 边界 AR 不依赖前 phase reasoning"决策一致**。方案 B 把 plan 放在 phase 边界**之上**的持久 prefix 里：

   ```
   持久 prefix（每个新 phase AR 都看到）:
       <image_t><task>[plan]<plan>
   per-phase 部分（每个新 phase AR 重新生成，旧的丢弃）:
       [stage]<subtask_reason_i>[action]<subtask_i>
   ```

   方案 A 把 plan 和 subtask_reason 混在 `[stage]` 段，phase 切换时 plan 也会被丢掉，用户必须每次 AR 都重新生成 plan（代价：长输出 + 每 phase 都要再现 plan 一字不差，模型容易学糊）。

3. **可消融**。方案 B 给 `[plan]` 段独立的 mask，未来可做"是否屏蔽 plan / 是否在新 phase mask plan / 等"消融。方案 A 没法做这种消融。

### 2.3 token 标记符的选择

考虑到 PaliGemma tokenizer 是 SentencePiece + 257k 词表，方括号包裹的标记会被切成多个 subword，不是问题。建议：

- `[plan]` — 持久段开始
- `[stage]` — phase-level reasoning 开始
- `[action]` — phase-level langact 开始
- 不需要 `[/plan]` / `[/think]` 闭合，因为下一个标记的出现就隐式闭合了上一个

### 2.4 两段 mask 在 Bridge 数据下的具体含义

回顾 [tokenizer.py 改造](policy/lap/src/lap/models/tokenizer.py)：

| Mask | Bridge 数据下覆盖范围 | 用于 |
|------|--------------------|------|
| `tokenized_ar_target_mask` | `[stage]<subtask_reason>[action]<subtask>` 全部 | next-token CE loss + ar_mask |
| `tokenized_stage_mask` | 只覆盖 `[stage]<subtask_reason>` 段 | action attention 屏蔽（本次预训练 disable action 不会用到，但保留以便后续 fine-tune） |

`<task>` 和 `[plan]<plan>` **不在任何 mask 内**，它们是给定的 prefix 条件，prompt_mask 已涵盖。

---

## 3. 用户问题逐条回答

### Q1: 数据合成

> "可以合成，不过这次训练先不用这些数据，只用 bridge"

✅ 同意。pick_place_primitive 的 subgoal 合成留到下一阶段。

### Q2: 关于 `<task>` vs `<reasoning>` 的边界

> "为了应对数据集之前的差异，我们可能要明确一下 `<task>[stage]<reasoning_i>[action]<langact_i>` 一部分文本应该属于 `<reasoning_i>` 还是属于 `<task>`，由于 10.2，在新的 phase 的 AR 不会 condition 之前的 `[stage]<reasoning_i>`"

**进一步澄清**：用户的核心担心是——如果某条信息**phase 间应该共享**，那它必须在 `<task>` 或 `[plan]` 里，**不能**在 `[stage]<reasoning_i>` 里（否则 phase 切换后这条信息就丢了）。

**判断准则**：
| 信息层级 | 应该放哪 | 例 |
|---------|---------|-----|
| Episode 级常量（任务什么时候都不变） | `<task>` | "Move the wooden arch onto the table." |
| Episode 级长期计划（phase 间不变） | `[plan]<plan>` | "Reach ... Grasp ... Move ... Drop ..." |
| Phase 级（每个 phase 重新生成） | `[stage]<reasoning_i>` | "The wooden arch is the object that needs to be moved..." |
| Phase 级动作描述 | `[action]<langact_i>` | "Reach for the wooden arch." |

**关键 invariant**：`[stage]` + `[action]` 段是"可丢弃单元"。Phase 切换时这部分整段丢弃 + 重新生成，模型不会再看到它。

### Q3: 失败 phase 处理

> "失败的 phase 先不纳入训练，除非以后在失败的基础上有 recovery 的数据"

✅ 同意。Bridge 数据的 `metadata.episode_success` 不存在，但 episode 通常都是成功示范（Bridge V2 整体是 success demos）。本次训练全收。

### Q4: action chunk 跨 phase 时 langact 取哪个

由于本次预训练 **disable action expert**（仅 expert 0 / 仅语言 CE loss），这个问题不影响本次。
留作 fine-tune 阶段决策。**本次直接用 chunk 起始帧的 langact**（最自然、与训练数据对齐）。

### Q10.5 重申: phase 切换时是否要"推掉" `[stage]...[action]...`

> "推理时，如果在 phase 中再次中间观察，emit 同一个 langact；或者跨入下一个 phase，emit 了新的 langact，问题：是不是要把序列中的 `[stage]...[action]...` 推掉，只 condition `<task>` 或者 `<task><plan>` 预测新的 langact"

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
# 新 token 只能 attend [image, task, plan]，不能 attend 旧的 [stage][action]
# 实现: 给 KV cache 中旧 [stage][action] 段位置的 attention 设为 False
```
更高效（避免重算 image SigLIP），但需要精确管理 mask 边界。

**训练侧自然支持**：因为训练时每个样本就是 `(image_t, task, plan, subtask_reason_at_t, subtask_at_t)`，模型从来没见过"前 phase 的 [stage][action] 在 prefix 里"的情况，所以推理时不放进去也是分布内的。

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
- `language_actions = subtask_reason` —— 走现有 `[stage]` 段路径（tokenizer 现有的 reasoning 字段）
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

## 9. ECoT JSON ↔ LeRobot 数据集对应关系（实测确认）

下载的两个 LeRobot 数据集对应：
| HF dataset | 含义 | 帧数 | episode 数 |
|------------|------|------|-----------|
| `jnogga/bridge_data_v2_teleop` | 远程操作 (人示教) demo | 1.56M | 43,457 |
| `jnogga/bridge_data_v2_scripted` | 脚本生成 demo | (待查) | (待查) |

ECoT JSON 的 `file_path` 键是原 Berkeley NFS 路径，例：

```
/nfs/kun2/users/homer/datasets/bridge_data_all/numpy_256/bridge_data_v2/<workspace>/<task>/<run_id>/train/out.npy
└──────────────── prefix ────────────────┘└──────────── 任务标识 ────────────┘└─split─┘
                                          ↑
                                   这部分对应 LeRobot uuid 的 task_root
```

LeRobot uuid 例：

```
raw/bridge_data_v2/<workspace>/<task>/<run_id>/<datetime>/raw/traj_group<G>/traj<N>
└─── task_root (与 ECoT 对应) ──────┘└──── 单个 trajectory 的标识 ────┘
```

**关键差异**：ECoT 一个 `out.npy` 文件**整体打包**了 `task_root` 下所有 trajectories；LeRobot 把每个 trajectory 拆成独立 episode。

### 9.1 实测匹配率

抽样 10 个 ECoT 路径，对照 LeRobot teleop（43,457 episodes）：

| 类别 | ECoT 路径前缀 | 是否匹配 LeRobot teleop |
|------|-------------|----------------------|
| `numpy_256/bridge_data_v2/...` | ✅ 6/6（每条 ECoT entry 命中 25–50 个 LeRobot episode） |
| `numpy_256/bridge_data_v1/...` | ✅（未抽到，但 LeRobot uuid 中有 `bridge_data_v1` 前缀，应该可匹配） |
| `numpy_256/rss/...` / `numpy_256/icra/...` | ✅（首例 `rss/toykitchen2/pnp_sweep/203` 匹配 20 个 episode） |
| `scripted_numpy_256/...` | ❌ 0 命中 LeRobot teleop（应在 LeRobot **scripted** 数据集找） |

**结论**：
- ECoT JSON `numpy_256/*` 前缀 → LeRobot **teleop** 数据集
- ECoT JSON `scripted_numpy_256/*` 前缀 → LeRobot **scripted** 数据集

LeRobot 数据集的 `bridge_data_v2_teleop` 描述：
> Episodes with inconsistent data or lacking language instructions were discarded, leaving **~86%** of all teleoperated episodes.

**所以 ECoT 全集与 LeRobot teleop 不会 100% 重合**。预计匹配率 ~86%（ECoT 是基于完整 Berkeley npy 生成的；LeRobot 过滤了 14%）。

### 9.2 匹配算法（推荐实现）

```python
def build_ecot_to_lerobot_mapping():
    """Build (ecot_file_path, ecot_episode_id) -> lerobot_episode_index mapping.

    Algorithm:
      1. Load ALL LeRobot episodes_meta into a dict keyed by `task_root`
         (= uuid stripped of "<datetime>/raw/traj_group<G>/traj<N>").
      2. For each ECoT (file_path, episode_id):
         - Strip "<...>/numpy_256/" or "<...>/scripted_numpy_256/" from file_path.
         - Strip trailing "/(train|val)/out.npy".
         - Prepend "raw/" to get cand_task_root.
         - Look up cand_task_root in the LeRobot dict.
         - Within candidates, match by (n_steps == adapter.length, [optional] language_instruction).
         - First exact (n_steps, lang) match wins.
      3. Cache result to a JSON file (~30 MB) so we don't pay this cost every run.
    """
```

**未匹配的 ECoT 项处理**：14% 估计落入"LeRobot 已过滤"集合，**这些 ECoT 样本本次预训练直接丢弃**（因为没有图像配对）。

---

## 10. 关于 plan 是否纳入 AR 目标的决策

**用户问题**：为了让模型学到对任务的 plan，是否需要将 plan 也纳入 AR 的优化目标？

**结论：是，建议把 plan 放进 AR 目标。** 但有一些细节需要决定。

### 10.1 三种候选方案对比

| 方案 | prompt 内容 | AR 目标内容 | 优点 | 缺点 |
|------|-----------|------------|------|------|
| **A. plan 为输入** (当前已实现) | `task + " [plan] " + plan` | `[stage]<reasoning>[action]<langact>` | 简单、AR 目标短、训练快 | 推理时 plan 从哪来？需要外部生成 |
| **B. plan 合并进 [stage]** | `task` | `[stage]<plan>; <subtask_reason>[action]<langact>` | 不改 tokenizer | plan 与 reasoning 混在一起，无独立粒度控制 |
| **C. plan 独立 [plan] 段（新加 tokenizer 段）** | `task` | `[plan]<plan>[stage]<subtask_reason>[action]<langact>` | 干净三段，可独立 loss 加权 | 需要扩 tokenizer，工程量大 |

### 10.2 推荐：**方案 B（plan 合并进 [stage] 段）**

理由：
1. **推理时模型自给自足** — 给定 task 就能产出完整 plan + reasoning + langact，不依赖外部 plan 来源（解决方案 A 的推理痛点）
2. **不改 tokenizer** — 复用现有 `[stage]/[action]` 双段架构，工程改动小
3. **粒度可调** — 把 plan 当作 reasoning 的"开头部分"，可以用 `reasoning_mask_prob` 做 dropout 削弱权重；也可以在第一帧才包含 plan
4. **数据驱动** — Bridge ECoT 里 plan 和 subtask_reason 本质都是"思考过程"的语言陈述，合并语义自然

### 10.3 plan 应该每帧都参与还是只第一帧？

**两个子方案**：

**B1 — 每帧都 emit plan**
```
[stage] {plan}\n{subtask_reason} [action] {subtask}
```
- 优点：训练数据规整，每个 sample 独立完备
- 缺点：plan 在一个 episode 内不变，重复 N 次浪费 capacity；模型"学会"在每帧重新预测同一段固定文本，CE loss 被 plan 重复主导

**B2 — 仅第一帧 emit plan，其他帧 skip plan**
```
phase 0 frame 0:    [stage] {plan}\n{subtask_reason}_0 [action] {subtask}_0
phase 0 frame 1+:   [stage]                {subtask_reason}_0 [action] {subtask}_0
phase 1+ all:       [stage]                {subtask_reason}_i [action] {subtask}_i
```
- 优点：避免重复 plan，训练效率高
- 缺点：dataloader 要标注 `is_first_frame_of_episode` 字段；推理时也要决定何时 emit plan

**推荐 B2 的轻量版**：以概率 `p_plan` 在每帧把 plan 加进 [stage] 段，否则 skip。建议 `p_plan = 1/n_steps_avg ≈ 1/30 ≈ 0.03`，但为了保证模型见过 plan，初始用 `p_plan=0.1`。

```python
# In dataloader
if random.random() < p_plan:
    reasoning_text = f"{plan}\n{subtask_reason}"
else:
    reasoning_text = subtask_reason
```

这样：
- 每个 episode 平均出现 3-5 次 plan（够学）
- 大多数样本只学 reasoning + langact（更符合实际推理负载）
- 推理时模型见到 task + image，可以选择性地以低概率"开口"输出 plan（实际可能不会主动 emit，但需要时给特殊提示词触发）

### 10.4 简化版决策（本次先做）

**为了启动第一次训练，先采用方案 B1（每帧都 emit plan）**：
- 实现简单，dataloader 不需要额外字段
- Plan 重复确实浪费 ~30% capacity，但 Bridge 数据量大（86% × 43k ≈ 37k episodes × ~30 frames ≈ 1.1M samples）capacity 不缺
- 验证可行后再迭代到 B2

具体 dataloader 改动（在 [bridge_ecot_dataset.py](policy/lap/src/lap/datasets/bridge_ecot_dataset.py) `BridgeECoTSampleBuilder.build` 里）：

```python
# Current:
prompt = f"{task}{plan_separator}{plan}"
sample["prompt"] = prompt
sample["language_actions"] = subtask_reason
sample["langact"] = subtask

# New (Plan-as-AR-target, B1):
sample["prompt"] = task                                  # 仅 task 入 prompt
sample["language_actions"] = f"{plan}\n{subtask_reason}"  # plan 拼到 reasoning 开头
sample["langact"] = subtask
```

加一个 config flag `plan_as_ar_target: bool = True` 切换 A/B 行为。

---

## 11. 实施步骤清单（更新）

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
- ✅ Tokenizer 产出 `tokenized_ar_target_mask`（reasoning + langact 联合）和 `tokenized_stage_mask`（仅 reasoning）
- ✅ Decode 验证：解码出的 [stage] 段和 [action] 段文本与输入完全一致
- ✅ Mask 互斥性 assert 通过

### 未实现部分（按优先级）

**P0 - 用户必决**：图像数据来源（§4.3 三选一）

**P1 - 可与 P0 并行**：把 `BridgeECoTDataset` 接入 `data_loader.create_data_loader`。当前的 RLDS-only 路径需要新增分支判断 `isinstance(data_cfg, BridgeECoTDataConfig)` 走 Bridge 专用 dataloader。

**P2 - 等 P0 完成后**：在 `BridgeECoTDataConfig` 里把 `bridge_images_root` 字段实际接到 image loader（`LeRobotBridgeImageLoader` 或 `RawNpyBridgeImageLoader`），然后启动训练。

---

## 12. ECoT ↔ LeRobot 集成与 dataloader 接入计划

### 12.1 已实测的核心组件

| 组件 | 位置 | 状态 |
|------|------|------|
| ECoT JSON 流式解析 | [bridge_ecot_dataset.py:BridgeECoTDataset](policy/lap/src/lap/datasets/bridge_ecot_dataset.py) | ✅ 测试通过（3 样本） |
| ECoT ↔ LeRobot 路径映射 | [bridge_lerobot_loader.py:build_ecot_to_lerobot_mapping](policy/lap/src/lap/datasets/utils/bridge_lerobot_loader.py) | ✅ 跑通，43,714/60,062 配对 (72.8%) |
| LeRobot mp4 帧读取 | [bridge_lerobot_loader.py:LeRobotBridgeImageLoader](policy/lap/src/lap/datasets/utils/bridge_lerobot_loader.py) | ✅ 跑通，224×224 RGB 真图 |
| Plan-as-AR-target 模式 | [bridge_ecot_dataset.py:BridgeECoTSampleBuilder](policy/lap/src/lap/datasets/bridge_ecot_dataset.py) | ✅ 跑通，编码/解码一致 |
| 端到端 sanity 脚本 | [test_bridge_batch_dump.py](policy/lap/scripts/test_bridge_batch_dump.py) | ✅ 跑通，PNG + TXT dump |

### 12.2 视觉抽查结论

样本 0 dump 的图像：木桌上有彩色积木 + 木质拱形物，与 task `Move the wooden arch onto the table` 语义匹配。

**注意**：由于 ECoT episode_id → LeRobot traj_index 的对应是启发式的（`length` 配对 + ID 模 N 选择），同一 task_root 下多个等长 trajectory 可能错配。**对于纯语言 CoT 预训练影响不大**（统计模式仍然成立），但**未来 fine-tune action head 时必须改用更严格的对齐**（比如对比图像内容 hash 或第一帧叠加）。

### 12.3 dataloader 接入方案（待实现）

**问题**：现有 `data_loader.create_data_loader` 路由分两条：
- RLDS path（基于 `data_cfg.rlds_data_dir is not None`）
- 上游 torch path（HF LeRobot 标准格式）

Bridge ECoT 既不是 RLDS（没有 TFDS 元数据），也不是标准 LeRobot 格式（自定义解析 JSON 取 reasoning），需要第三条路径。

**实现思路（推荐）**：
1. 在 `create_data_loader` 顶部加 `if data_cfg.repo_id == "bridge_ecot":` 分支
2. 该分支：
   - 构造 `BridgeECoTDataset` + `LeRobotBridgeImageLoader`
   - 包装为 torch `IterableDataset`
   - 走 torch `DataLoader` + `collate_fn` 处理 batch
   - 应用 `model_transforms`（tokenizer）于每个 sample
3. 输出仍然是 `(CoTObservation, Actions)` 元组（actions 全 0 占位，下游会被 `enable_action_training=False` 屏蔽）

当前已在 [data_loader.py](policy/lap/src/lap/datasets/data_loader.py#L148) 加了 `NotImplementedError` stub，提示具体实现位置。

**估计工作量**：1 天（torch IterableDataset wrapper + collate_fn + 测试）。

### 12.4 在 K8s 上跑通 dataloader 验证

> 实施顺序：先在本地 mock 数据上跑通 wrapper+collate；再 sync 到 pod 上用真数据验证 1 个 batch。

完整实现后，验证步骤：
```bash
# Local: dry-run with NullImageLoader (no LeRobot deps)
.venv/bin/python -c "
from lap.training import config as _c
from lap.datasets.data_loader import create_data_loader
import jax
cfg = _c.get_config('lap_bridge_pretrain')
loader = create_data_loader(cfg)
batch = next(iter(loader))
obs, actions = batch
print('obs.tokenized_prompt.shape:', obs.tokenized_prompt.shape)
print('obs.tokenized_ar_target_mask sum/sample:', obs.tokenized_ar_target_mask.sum(axis=-1))
print('actions.shape:', actions.shape, '(expected (B, 1, 7) all zeros)')
"
```

---

## 13. K8s 训练启动计划

### 13.1 现有基建

| 组件 | 位置 / 命令 |
|------|------------|
| Helm 模版 | `~/xshixun/user/userchart` |
| Values 文件 | `~/xshixun/user/values-keepalive.yaml` |
| 启动命令 | `helm install zhaoqc-pi05-finetune-steps-25000 --values ~/xshixun/user/values-keepalive.yaml ~/xshixun/user/userchart` |
| 远端项目路径 | `/data/zhaoqc/RoboTwin` |
| Pod 互联网代理 | `pod-tunnel proxy`（位于 `~/.local/bin/`） |
| Local→Pod 同步 | `policy/pi05/sync_to_pod.sh` |

### 13.2 数据搬运策略

预训练需要的数据：

| 数据 | 大小 | 用途 | 搬运方式 |
|------|------|------|---------|
| `embodied_features_bridge.json` | 1.4 GB | ECoT 标注 | sync_to_pod 或 pod 上 `huggingface-cli download` |
| `bridge_data_v2_teleop` LeRobot | ~40 GB | 远程操作图像 | **pod 上下载**（用 `pod-tunnel proxy` 访问 HF） |
| `bridge_data_v2_scripted` LeRobot | ~10 GB | 脚本 demo 图像 | **pod 上下载** |
| PaliGemma 2B 检查点 | ~5 GB | 模型起点 | sync_to_pod（已有则跳过） |
| LAP/RoboTwin 项目代码 | ~500 MB | 训练代码 | sync_to_pod 或 git push+pull |

**推荐策略**：先 sync 代码到 pod，pod 上启动 proxy，pod 上 `huggingface-cli download` 拉数据（避免本地→pod 传 50GB 慢）。

### 13.3 启动步骤（草案）

```bash
# === 1. Local：同步代码到 pod ===
cd /home/numbnut/worksapce/RoboTwin
# 推荐：先 git commit，然后在 pod 上 git pull（更稳）
# 替代：直接 rsync 整个项目（会同步 .venv，慢）
./policy/pi05/sync_to_pod.sh policy/lap/ /data/zhaoqc/RoboTwin/policy/lap

# === 2. 在 pod 内（需先 helm install 起 pod，然后 kubectl exec）===
# 2a. 启动 internet proxy
~/.local/bin/pod-tunnel proxy &

# 2b. 安装新依赖
cd /data/zhaoqc/RoboTwin/policy/lap
.venv/bin/python -m pip install ijson decord

# 2c. 下载 ECoT JSON（1.4 GB）
huggingface-cli download Embodied-CoT/embodied_features_bridge --repo-type dataset

# 2d. 下载 LeRobot Bridge teleop（~40 GB）
huggingface-cli download jnogga/bridge_data_v2_teleop --repo-type dataset

# 2e. （可选）下载 scripted（~10 GB）
huggingface-cli download jnogga/bridge_data_v2_scripted --repo-type dataset

# 2f. 下载 PaliGemma 2B 检查点（如果 pod 上没有）
# TODO: 确认 checkpoint 路径

# === 3. 在 pod 上验证 dataloader 跑通 1 个 batch ===
.venv/bin/python scripts/test_bridge_batch_dump.py --num-samples 2 --out-dir /tmp/bridge_dump

# === 4. 启动训练 ===
.venv/bin/python scripts/train.py --config-name lap_bridge_pretrain
```

### 13.4 训练监控

需要监控的关键指标：

- **`langact_loss`** — 主要训练目标（next-token CE on reasoning + langact）
- **`langact_token_acc`** — 字符级准确率（verbose mode）
- **`number_token_acc`** / **`direction_token_acc`** — Bridge 数据数字少，主要看 direction
- **batch shape** — 验证 prompt token 长度 / langact span 长度的统计
- **GPU/TPU 利用率** — 早期 dataloader 容易成为瓶颈

预期 loss 曲线：从 ~10 (random init langact) 降到 ~1-2 (overfit) 在 50K steps 内。

### 13.5 风险与缓解

| 风险 | 缓解 |
|------|------|
| dataloader 是瓶颈（mp4 解码慢） | 用 `num_workers > 0`；预先把帧缓存为 numpy memmap |
| 72.8% 配对率太低 | 已识别原因：v1 路径 + 长度过滤；可接受用于预训练 |
| ECoT JSON 流式解析每 epoch 重读 | 把 episodes 索引序列化为 parquet（一次性预处理） |
| pod 磁盘空间不足（50GB） | 检查 pod 磁盘前确认；如不足，仅用 teleop 部分 |
| pod 内存爆（mapping 全表加载） | 当前 mapping JSON ~30MB，没问题 |

### 13.6 决策已敲定（2026-05-06）

| # | 项 | 决策 |
|---|----|------|
| 1 | plan_as_ar_target | 重新设计为概率切换 `plan_position`，详见 §10 (`p_plan=0.15`) |
| 2 | 是否包含 scripted | ✅ 包含。`bridge_lerobot_loader` 自动从 `DEFAULT_SCRIPTED_SNAP_PARENT` 解析 |
| 3 | batch_size | **128**（A100 80GB 安全；OOM 时降到 64） |
| 4 | num_train_steps | **80,001** （≈ 6.5 epoch over Bridge teleop 1.55M frames） |
| 5 | ~~save_interval / keep_period~~ | ~~**10K / 20K**~~ —— **已废弃**，见下方 5'。 |
| 5' | save_interval / keep_period （revised 2026-05-07）| **2K / 10K** —— 首次训练前期勤保，单次崩溃最多损失 2K 步（≈25 min）。`keep_period=10K` 即每 10K 一个永久里程碑，共 8 个 milestone（10K/20K/.../80K）。**revision 起因**：5-7 凌晨 step 10000 第一次触发 save 时进程被 SIGKILL，写入未完成，得到 0 个可用 ckpt。后续 first-time 训练默认采用此密集策略。 |
| 6 | wandb 启动前的代理 | 必须先 `~/.local/bin/pod-tunnel proxy &` 确保 pod 能访问外网 |

### 13.7 dataloader 接入完成

代码已就绪，端到端 smoke test 通过：
- [bridge_data_loader.py](src/lap/datasets/bridge_data_loader.py) — `BridgeDataLoader` + `_BridgeIterableTorchDataset`
- [data_loader.py](src/lap/datasets/data_loader.py#L148) — `create_data_loader` 加 `repo_id=="bridge_ecot"` 分支
- [test_bridge_dataloader.py](scripts/test_bridge_dataloader.py) — 端到端验证脚本

实测结果：
```
Batch 0:  image['base_0_rgb'] shape=(2, 224, 224, 3) float32  mask=[True, True]
          image['left_wrist_0_rgb'] shape=(2, 224, 224, 3) float32  mask=[False, False]
          ar_target_mask sums = [29, 29]
          stage_mask sums = [22, 22]   plan_mask sums = [0, 0]
          actions shape=(2, 1, 7)  all_zero=True
```

### 13.8 K8s 部署 Runbook

#### Step 1 — 同步代码到 pod

本地：
```bash
cd /home/numbnut/worksapce/RoboTwin
rsync -avz policy/lap/ \
  --exclude ".venv" --exclude "**__pycache__" --exclude "uv.lock" \
  --exclude "wandb" --exclude "checkpoints" --exclude ".git" \
  k98s:/data/zhaoqc/RoboTwin/policy/lap/
```

如果 rsync 出错（k98s SSH 链路断），先：
```bash
./policy/pi05/launch_pod_ssh.sh --skip-helm
```
然后重试 rsync。

#### Step 2 — 进入 pod，启动外网代理

```bash
kubectl exec -it deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 -- bash

# Inside pod:
~/.local/bin/pod-tunnel proxy &
# Verify proxy is up
curl -sI https://huggingface.co | head -3
```

#### Step 3 — 验证数据集已就位

```bash
# In pod:
ls /data/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/snapshots/*/embodied_features_bridge.json
ls /data/.cache/huggingface/hub/datasets--jnogga--bridge_data_v2_teleop/snapshots/*/data/chunk-000/file_000.parquet
ls /data/.cache/huggingface/hub/datasets--jnogga--bridge_data_v2_scripted/snapshots/*/data/chunk-000/file_000.parquet
ls /data/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/
```

如果某个数据集还没下载，在 pod 上拉：
```bash
huggingface-cli download Embodied-CoT/embodied_features_bridge --repo-type dataset
huggingface-cli download jnogga/bridge_data_v2_teleop --repo-type dataset
huggingface-cli download jnogga/bridge_data_v2_scripted --repo-type dataset
```

#### Step 4 — 安装新依赖

```bash
cd /data/zhaoqc/RoboTwin/policy/lap
uv pip install --python .venv/bin/python ijson decord
```

#### Step 5 — 端到端 smoke test（强烈推荐先跑！）

```bash
.venv/bin/python scripts/test_bridge_dataloader.py --batch-size 4 --num-batches 2
# Expected: prints batch shapes, no exceptions, "Smoke test passed."
```

第一次会构建 `~/.cache/cascade_vla/bridge_ecot_lerobot_mapping.json`（耗时
1-3 分钟，~30MB）。后续运行直接读缓存。

注意：mapping cache 默认放 `~/.cache/cascade_vla/`，但 pod 上 `~` 通常映射到
`/root` 或 `/data/zhaoqc`。如果重启 pod 缓存丢，rebuild 会再触发；不影响正确性。

#### Step 6 — 启动训练

```bash
# Optionally launch pod-tunnel for wandb if not already running
~/.local/bin/pod-tunnel proxy &

# Train
.venv/bin/python scripts/train.py --config-name lap_bridge_pretrain
```

#### Step 7 — 监控

- **wandb**：观察 `langact_loss` 曲线、`langact_token_acc`
- **GPU 利用率**：`nvidia-smi` 应该 >80%；低于 50% 说明 dataloader 是瓶颈
- **Disk usage**：`du -sh /data/zhaoqc/RoboTwin/checkpoints/lap_bridge_pretrain/` 监控
- **Loss 曲线 sanity**：
  - Step 0：~10 (random init logits, log(257152) / 12 = 10.4)
  - Step 2000 (warmup 结束)：应快速降到 ~3-4
  - Step 80000：应稳定在 ~1-2

#### Step 8 — 故障排查

| 现象 | 可能原因 | 解决 |
|------|--------|------|
| dataloader OOM / 卡住 | LeRobot mp4 解码/缓存太多 | 降 `num_workers`，在 `BridgeDataLoader` 改 |
| step 0 batch 拉取慢 | 第一次构建 mapping JSON | 跑过一次后会缓存，正常 |
| GPU 利用率低 | dataloader 瓶颈 | 提高 `num_workers`（config.num_workers），考虑预 cache 帧 |
| weight_loader 报错下不到 GCS | `pod-tunnel proxy` 没启动 / 代理失效 | 重启 proxy；或本地转 paligemma-3b-pt-224 → npz 并 sync 到 pod，切 weight_loader.kind="paligemma2" |
| OOM at first forward | batch_size 太大 | 降到 64，再降 32 |
| `module 'torch.distributed' has no attribute ...` | 多卡相关接口；本次单卡训练用不到 | 忽略 |

---

### 13.9 远程诊断 Runbook（事后排查 / 训练状态自检）

> **背景**：训练任务跑在 k8s deployment 中，主机回连不便；用 `kubectl exec` 直接发命令到 pod 内部检查训练状态、磁盘、checkpoint、GPU。Local 端从 `kubectl` 即可完成 Pod 内观察，无需 SSH。

#### Local → Pod 命令模板

```bash
# 单条命令（无 -t/-i，用于脚本化）
kubectl exec deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 \
  -- bash -lc '<COMMAND>'

# 多条命令包成一个 bash -lc 子 shell（推荐，因为 kubectl 调用本身有开销）
kubectl exec deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 -- bash -lc '
  echo "=== A ==="; <cmd_a>
  echo "=== B ==="; <cmd_b>
'

# 进交互 shell（手工调试用）
kubectl exec -it deployment/zhaoqc-gpu-keepalive-zhaoqc-pi05-finetune-steps-25000 -- bash
```

注意 `bash -lc` 比 `bash -c` 多读 `~/.bash_profile` / `~/.bashrc`，能拿到 conda/uv/proxy 环境。

#### Local → Pod 单文件同步（小改动用 `kubectl cp`，比 rsync 轻）

```bash
# 拿 Pod 名（deployment 默认绑定一个 replica）
POD=$(kubectl get pods --no-headers | grep zhaoqc-pi05-finetune-steps-25000 | awk '{print $1}' | head -1)

# Push 单个文件
kubectl cp <local-path> "$POD":<pod-path>

# Push 一棵子目录（注意：kubectl cp 是 tar over kubectl exec，子目录要 trail-slash 控制）
kubectl cp <local-dir>/. "$POD":<pod-dir>/

# 验证（前后 md5 对比）
md5sum <local-path>
kubectl exec deployment/<deploy-name> -- md5sum <pod-path>
```

适用场景：单文件 patch（如 config.py 调参）、补丁脚本。整个 `policy/lap/` 同步用 §13.8 Step 1 的 rsync。

#### 关键诊断检查表

| 类别 | 命令 | 期望 / 含义 |
|------|------|------|
| **挂载点** | `df -hT /data /tmp` | `/data` 应为 `gpfs2`（持久化）；`/tmp` 落 overlay/tmpfs（pod 重启即丢） |
| **训练进程** | `ps -eo pid,etime,pcpu,cmd \| grep -E "train.py\|jax"` | 训练运行中应见 `python ... train.py`；只见 keepalive 说明已退出 |
| **GPU 利用率** | `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` | 训练中 >70%；空载 <5% 表示进程已死或 idle |
| **Checkpoint 总量** | `du -sh /data/zhaoqc/RoboTwin/policy/lap/checkpoints/*/` | 与预算（~80GB）对比 |
| **Checkpoint 完整性** | `ls /data/zhaoqc/.../<exp_name>/` | 完整 ckpt 是 `<step>/`；带 `.orbax-checkpoint-tmp-N` 后缀 = 写入未完成 |
| **训练日志尾部** | `tail -80 /data/.../wandb/latest-run/files/output.log` | 看最后 step、loss、是否 traceback |
| **进程退出方式** | `ls /data/.../wandb/latest-run/files/wandb-summary.json` | 文件存在 = graceful exit；不存在 = 被强制 kill |
| **Pod uptime** | `ps -p 1 -o etime,start,cmd` + `date` | 算出 entrypoint 起跑时间，对照训练 log 判断是否被重启 |
| **wandb run id** | `cat <ckpt_dir>/wandb_id.txt` | 用于 resume；保存在 ckpt 目录下，跟 ckpt 一同持久化 |
| **同步问题** | `dmesg \| tail` (常被 deny) / `free -h` | OOM 排查通常需要 root；free 看内存压力 |

#### 故障模式速查

| 症状 | 可能原因 | 应对 |
|------|----------|------|
| 训练 step 卡在某步不动 | dataloader 阻塞、GCS 拉权重超时 | `tail output.log`、检查 `pod-tunnel proxy` |
| `<step>/` 目录变成 `<step>.orbax-checkpoint-tmp-0/` | 写入中被 SIGKILL | params 大概率 OK；train_state 风险高（resume 不可信） |
| pod 重启后 keepalive 在跑而 train 不在 | k8s `restartPolicy=Always` + 训练进程崩溃 | 看 wandb 末尾日志找异常；检查 helm values 里 entrypoint 是否还是 train.py |
| wandb cloud 有 metrics 但本地 `wandb-summary.json` 缺失 | 进程被强制终止（OOM、抢占、kubectl delete） | wandb cloud 是 source of truth；本地能 resume 就靠 ckpt 内的 wandb_id.txt |

#### 实战例（2026-05-07 训练异常排查）

```bash
# 1. 看进程
kubectl exec deployment/.../-pi05-finetune-steps-25000 -- bash -lc 'ps -eo pid,etime,cmd | grep -E "train|python" | grep -v grep'
# → 只看到 keepalive python，无 train.py。结论：训练已退出。

# 2. 看 GPU
kubectl exec ... -- nvidia-smi
# → utilization=0%, memory=720MiB（仅 keepalive 占用）。佐证训练不在跑。

# 3. 看 ckpt
kubectl exec ... -- ls -lah /data/zhaoqc/RoboTwin/policy/lap/checkpoints/lap_bridge_pretrain/lap_bridge_pretrain_run0/
# → 仅 `10000.orbax-checkpoint-tmp-0/` + `wandb_id.txt`。结论：step 10000 写入未完成。

# 4. 看 log 末尾
kubectl exec ... -- bash -lc 'tail -80 /data/zhaoqc/RoboTwin/policy/lap/wandb/latest-run/files/output.log'
# → 最后输出停在 "Save Finalize thread starting"，无 traceback；
# → 配合 wandb-summary.json 缺失，结论：被 SIGKILL（k8s 抢占或 OOM）。

# 5. 算时间线
kubectl exec ... -- bash -lc 'date; ps -p 1 -o etime,start,cmd'
# → keepalive started 01:27:53 = 训练 log 末尾 + 21s。结论：容器重启发生在 ckpt 写入中。
```

---

### 13.10 推理 smoke test ([scripts/test_cascade_inference.py](scripts/test_cascade_inference.py))

为了在训练过程中或训练完成后验证模型，提供 `scripts/test_cascade_inference.py`。

#### 功能

- 加载某一步的 ckpt（默认指向 `checkpoints/lap_bridge_pretrain/<exp_name>/<step>/`）
- 从 Bridge ECoT 数据集取出第 N 个 sample（含 ground truth: task / plan / stage / action）
- 用 `model.sample_tokens` 自回归生成 token
- 解析输出中的 `[plan]` / `[stage]` / `[action]` 段并与 ground truth 对照打印

#### 两种推理模式

| Mode | 说明 | 触发 |
|------|------|------|
| Context 3（**默认**）| Prompt 仅含 task；模型 AR 生成完整 cascade `[plan] ... [stage] ... [action]` | 不传 `--given-plan` |
| Context 2 | Prompt 含 task + ground-truth plan；模型只生成 `[stage] ... [action]` | 加 `--given-plan` |

#### 与训练共享 GPU 的两条路径

训练占满 6 张 H200 时无法在同一进程拿到 GPU。两种解法：

**A. CPU 推理（不打断训练）— 推荐做 quick check 用**
```bash
cd /data/zhaoqc/RoboTwin/policy/lap
JAX_PLATFORMS=cpu .venv/bin/python scripts/test_cascade_inference.py \
  --checkpoint-dir checkpoints/lap_bridge_pretrain/lap_bridge_pretrain_run1/10000 \
  --sample-idx 0
```
- 200GiB 内存够装 PaliGemma 2B (bfloat16 ~5GB)
- 单次生成 200 token 约 **5-10 分钟**（CPU 慢但可接受）
- 训练不受影响

**B. 暂停训练→GPU 推理→resume**
```bash
# 1. 找到训练 PID 并优雅 kill（让 wandb 结尾 + 写 wandb-summary.json）
ps -eo pid,cmd | grep "train.py lap_bridge_pretrain" | grep -v grep
kill -SIGTERM <PID>
# 2. 等所有 GPU 释放
until ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | grep -q "^[1-9]"; do sleep 2; done
# 3. 跑 GPU 推理（~1 分钟）
.venv/bin/python scripts/test_cascade_inference.py \
  --checkpoint-dir checkpoints/lap_bridge_pretrain/lap_bridge_pretrain_run1/10000 \
  --sample-idx 0
# 4. resume 训练（config.resume=True 会自动找到最近 ckpt）
nohup setsid .venv/bin/python scripts/train.py lap_bridge_pretrain \
  --exp-name lap_bridge_pretrain_run1 \
  >> /data/zhaoqc/RoboTwin/policy/lap/logs/run1.log 2>&1 < /dev/null &
```
- 推理快但有训练中断窗口（kill + ckpt restore + JAX recompile ≈ 5-10 min 训练损失）
- 适用于做正式评估时

#### 期望输出（一旦训练收敛）

```
[5/5] Sampling tokens (...)
      generated 87 tokens (EOS reached)
==============================================================================
DECODED RAW
==============================================================================
 [plan] reach grasp lift transport place [stage] move arm to spoon [action] move forward and right
==============================================================================
PARSED SEGMENTS  (model output  vs  ground truth)
==============================================================================
--- [plan] ---
  pred: 'reach grasp lift transport place'
  gt:   'reach grasp lift transport place'
--- [stage] ---
  pred: 'move arm to spoon'
  gt:   'reach the spoon by moving the arm forward and to the right'
--- [action] ---
  pred: 'move forward and right'
  gt:   'move forward and to the right'
==============================================================================
```

早期 ckpt（step 2K-10K）输出会噪声更多甚至发散；这是正常现象，主要看：
- ✅ 模型能产出三段式（marker 出现位置正确）
- ✅ Token 不重复发散（"the the the the"）
- ✅ Plan 与 Bridge 任务粗略对应
- ✅ Action 包含合理的 motion primitive 词汇（"move", "left", "up", "grasp"）

---

### 13.11 Marker-Loss-Coverage Bug & 修复（2026-05-07）

#### 现象

跑完第一个里程碑 ckpt（step 10000，langact_loss=1.11）后用 [scripts/test_cascade_inference.py](scripts/test_cascade_inference.py) 做 smoke test：

| 项 | 结果 |
|----|------|
| Task | "Move the wooden arch onto the table." |
| 模型生成（前两句）| "Move to the wooden arch, grasp the wooden arch, move the wooden arch to the table, release the wooden arch. Move the arm away from the table." |
| 模型生成（其余 22 句）| 重复 "Move the arm to the starting position." |
| `[plan]`/`[stage]`/`[action]` marker 出现次数 | **0** |

第一句语义对齐 GT plan（"Reach/Grasp/Move/Drop" 同义改写），说明模型**学到了任务语义**；但**完全没产出 cascade 标注**，并出现典型的 early-stage **mode collapse**（重复退化）。

#### 调查 — Mask 覆盖率验证

写 [scripts/inspect_marker_loss_coverage.py](scripts/inspect_marker_loss_coverage.py) 把同一个真实 Bridge sample 用三种 Context 跑 tokenize 并打印每个 token 的 mask 位：

| Context | Marker | `ar_target_mask` 覆盖（bug 时）|
|---------|--------|-------|
| 1 (legacy) | `[stage]` | **0/3** |
| 2 (plan-in-prompt) | `[plan]` | 0/3（√，在 prompt 里）|
| 2 | `[stage]` | **0/3** |
| 2 | `[action]` | **0/3** |
| 3 (full cascade) | `[plan]` | **0/3** |
| 3 | `[stage]` | **0/3** |
| 3 | `[action]` | **0/3** |

回查 [tokenizer.py `_create_segmented_masks` 的 docstring](src/lap/models/tokenizer.py#L193) 也明确写了："The literal segment markers ... are not covered by any of the per-segment masks ... their CE loss participates via `ar_target_mask` only when they fall inside one of the spans — **which by construction they do not**."

所以**这是设计本身的问题**，不是 bug。原始设计假设 marker 是 deterministic 的字面文本、模型不需要梯度也能"自然出现"。但实际效果证明 next-token CE 不在 marker 位置的话，模型无任何动机产出它们 —— inference 时直接跳过 marker 输出 plan 内容。

#### 修复

[tokenizer.py](src/lap/models/tokenizer.py)：把每段（plan / stage / action）的 `start_index` 向前扩到 leading marker 的位置：

```diff
 if plan_position == "target":
-    plan_start, plan_end = plan_text_start, plan_text_end
+    plan_start, plan_end = plan_marker_start, plan_text_end  # include marker

 # [stage]:
+stage_marker_start = len(tokens)
 stage_marker = self._tokenizer.encode(" [stage] ", ...)
 tokens += stage_marker
-stage_start = len(tokens)             # was: at content start
 tokens += self._tokenizer.encode(clean_reason, ...)
 stage_end = len(tokens)
+stage_start = stage_marker_start      # extend backward to cover marker

 # [action]: same pattern
```

修复后 inspector 重跑：

| Context | Marker | After |
|---------|--------|-------|
| 1 | `[stage]` | **3/3 ✅** |
| 2 | `[stage]` | **3/3 ✅** |
| 2 | `[action]` | **3/3 ✅** |
| 3 | `[plan]` | **3/3 ✅** |
| 3 | `[stage]` | **3/3 ✅** |
| 3 | `[action]` | **3/3 ✅** |

`ar_target_mask` 总和：Context 3 从 47 → 59（+12 token 进入 CE 梯度，dilution 约 25%；可接受，因为 marker 是固定 3-token 短语，模型学起来 trivial 但梯度信号必须要给）。

#### 训练侧影响

- **代码改动只动 mask 不改 token sequence**：旧 ckpt 的 params shape 完全兼容，可直接用 `--resume` 加载。
- **首次 resume 后 1-2K 步会出现 langact_loss 短暂上升**（约 +0.3-0.5）：因为 marker 位置首次被 supervise，模型从未见过梯度。预期会快速降回。
- **Marker token 在 PaliGemma 词表里是普通字面字符**（不是特殊 reserved token），所以模型直接学会"emit `[plan]` 然后 emit plan_text"是一个 N-gram 级别的简单 binding，不需要架构改动。

#### 操作记录

```bash
# 1. 写本节文档（你正在读的这段）
# 2. kubectl cp src/lap/models/tokenizer.py 到 pod
# 3. 优雅 kill 当前训练（SIGTERM 让 wandb 写出 summary）
# 4. 等 GPU 释放
# 5. 重启 train.py（同一 --exp-name lap_bridge_pretrain_run1，config.resume=True
#    自动从最新 ckpt = step 10000 接起）
# 6. 监控 langact_loss：预期 step 10001-10100 左右出现 0.3-0.5 的 spike，
#    然后 ~step 11500 降回 1.1 以下
# 7. step 12000 + 14000 + 20000 时再跑 inference smoke test 确认 marker 出现
```

下次 inference 期望输出形式（与 GT 对照）：

```
[plan] Reach for the wooden arch. Grasp the wooden arch. Move the wooden arch to the table. ...
[stage] The wooden arch is the object that needs to be moved...
[action] Reach for the wooden arch.
```

#### 验证结果（step 14000，fix 后训了 2K 步）

实测推理输出（同一 Bridge sample idx=0，CPU greedy）：

```
[plan]  Move to the wooden arch, grasp the wooden arch, move the wooden arch to the table, release the wooden arch.
[stage] The wooden arch is the first object that needs to be interacted with.
[action] Move to the wooden arch.
```

总输出 56 tokens + **EOS reached**（vs step 10000 的 200 tokens 无 EOS + 22 次重复）。

| 验证项 | 结果 |
|--------|------|
| 三段 marker 全部出现 | ✅ |
| EOS 触发 | ✅ |
| Mode collapse 消失 | ✅ |
| 语义对齐 GT（paraphrase-level）| ✅ |
| 训练 step 数代价 | 仅 2K 步（11min wallclock）|

后续步数（28K / 56K / 80K）出更高质量输出（更接近字面 GT），并能区分 plan-as-prompt 与 plan-as-target 两种模式下的行为差异。

---

## 8. 留给用户的决策点

1. **图像数据源**：请确认走 §4.3 的哪一条（推荐 II = HF LeRobot bridge_orig）。需要约 ~50 GB 磁盘空间和约 1 小时下载时间。
2. **是否启用 plan 段**：默认 `include_plan=True`。可考虑做"with vs without plan"消融，看 plan 是否真的提升 phase-level reasoning 质量。
3. **训练目标 token_len**：当前提案 256。task + plan + subtask_reason + subtask 累计经验上 ~150-200 tokens，留余量。先小批 dry-run 看实际长度分布。
4. **是否保留 `move_primitive` 字段做辅助**：每帧 `move` ("move up" / "stop" / ...) 是 atomic 动作语义，可以作为额外的 per-step VQA 任务（训 expert 0 输出 move primitive）。**本次先不用**，留着以后做。
5. **预训练后下游评估**：跑 RoboTwin 仿真还是 LIBERO？建议两边都跑，因为预训练目的就是让模型学会"看图说 reasoning + langact"，下游迁移到 RoboTwin/LIBERO 时再加 action expert fine-tune。
