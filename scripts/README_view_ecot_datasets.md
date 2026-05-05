# view_ecot_datasets.py — ECoT 数据集查看器

查看 Embodied-CoT 系列两个数据集的内部结构与样本内容,无需写额外解码代码。

- **bridge** (`embodied_features_bridge`): 单个 1.4 GB JSON,只含每帧的 ECoT 标签(原始 demo 在 Bridge V2 数据集中,通过 `file_path` 字段引用)
- **libero** (`embodied_features_and_demos_libero`): 128 个 TFRecord 分片,demo + ECoT 标签打包在一起

脚本路径: [policy/lap/scripts/view_ecot_datasets.py](view_ecot_datasets.py)

---

## 1. 安装依赖

`source .venv/bin/activate` 进入项目 venv,然后:

```bash
uv pip install ijson tfrecord
```

| 包 | 作用 | 体积 |
|---|---|---|
| `ijson` | bridge 大 JSON 流式解析(否则 fallback 到 `json.load`,需要 ~10 GB RAM) | 几十 KB |
| `tfrecord` | libero TFRecord 解码(纯 Python + protobuf,**不需要 tensorflow**) | ~30 KB |

> 注意: 不要装 `tensorflow`。当前 venv 的 numpy 被 SAPIEN/mujoco/mplib 等仿真依赖钉在 1.24,而新版 TF 要求 numpy ≥ 1.26,会直接 ImportError。脚本已改用 `tfrecord` 包,完全绕开 TF。

---

## 2. 数据位置

脚本默认从 HuggingFace 缓存读:

```
~/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_bridge/...
~/.cache/huggingface/hub/datasets--Embodied-CoT--embodied_features_and_demos_libero/...
```

如果还没下载,先 `huggingface-cli download Embodied-CoT/embodied_features_bridge` / `Embodied-CoT/embodied_features_and_demos_libero --repo-type dataset`。

---

## 3. 命令行用法

```bash
python policy/lap/scripts/view_ecot_datasets.py {bridge|libero} [options]
```

### 3.1 bridge

```bash
python policy/lap/scripts/view_ecot_datasets.py bridge \
    --num-files 1 --num-episodes 2 --num-steps 4
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--num-files` | 2 | 看前几个原始 npy `file_path` |
| `--num-episodes` | 2 | 每个 file 内看前几集 |
| `--num-steps` | 5 | 每集打印前几步的 features + reasoning |

输出片段(节选):

```
[file 0] /nfs/.../stack_blocks/19/train/out.npy
  num_episodes_in_file: 45    sample_ids: ['43', '11', '27', ...]

[file 0 / episode 43]
  >>> metadata
      episode_id: 43
      n_steps: 40
      language_instruction: move the wooden arch to the table

  >>> features  (n_steps=40)
      keys: move_primitive, gripper_position, bboxes
      step 0:  move='stop'  gripper=[97,45]  bboxes=[[0.34, "wooden blocks", [150,4,188,100]]]
      step 2:  move='move up'  gripper=[89,52]  bboxes=...

  >>> reasoning  (n_steps=40)
      step 0:
          task: Move the wooden arch onto the table.
          plan: Reach for the wooden arch. Grasp ... Drop the wooden arch onto the table.
          subtask: Reach for the wooden arch.
          subtask_reason: The wooden arch is the object that needs to be moved, ...
          move: stop
          move_reason: The arm is already in a good position to reach for the wooden arch.
```

### 3.2 libero

```bash
python policy/lap/scripts/view_ecot_datasets.py libero \
    --shard 0 --num-episodes 2 --num-steps 5
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--shard` | 0 | 解码哪个 TFRecord 分片(0..127) |
| `--num-episodes` | 2 | 该分片内看前几集 |
| `--num-steps` | 5 | 每集打印前几步 |

输出片段(节选):

```
total_episodes: 3917,  shard 0 length: 31

>>> per-step features (from features.json)
    language_instruction         text
    language_motions             text  ('|' 分隔的过去 moves)
    language_motions_future      text
    action                       float32[7]
    observation.image            uint8[224,224,3]
    observation.wrist_image      uint8[224,224,3]
    observation.joint_state      float32[7]
    observation.state            float32[8]   # 6D EEF + 2D gripper
    is_first/is_last/is_terminal/reward/discount   scalar

[shard 0 / episode 0]
  episode_metadata/demo_id : [34]
  episode_metadata/file_path: STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5

  >>> per-step preview  (episode_n_steps=167)
      step 0:
          language_instruction: pick up the book and place it in the left compartment of the caddy
          language_motions: move back
          language_motions_future: move back|move back and left|...
          is_first: 1   is_last: 0   reward: 0.0

      steps/action: total_floats=1169 -> ~167 steps × dim=7
        step 0 vector: [0.230, -0.056, 0.0, 0.0, 0.001, -0.025, -1.0]
      steps/observation/joint_state: total_floats=1169 -> ~167 steps × dim=7
        step 0 vector: [0.014, -0.129, 0.001, -2.418, 0.002, 2.230, 0.800]
      steps/observation/image: n_steps=167  first_jpeg_bytes=17009
      steps/observation/wrist_image: n_steps=167  first_jpeg_bytes=13021
```

---

## 4. 输出含义速查

### bridge 每集字段
- `metadata`: `episode_id` / `file_path` / `n_steps` / `language_instruction`
- `features`: 三个数组,**长度均为 n_steps**
  - `move_primitive[t]`: 文本动作("stop", "move up", "close gripper", ...)
  - `gripper_position[t]`: `[u, v]` 像素坐标
  - `bboxes[t]`: 列表,元素为 `[confidence, label, [x1,y1,x2,y2]]`
- `reasoning[t]`: 6 个字段
  - `task` / `plan` / `subtask` / `subtask_reason` / `move` / `move_reason`

### libero 每集字段
- 顶层 `episode_metadata/*`: `demo_id`、`file_path`(原始 LIBERO HDF5 文件名)
- `steps/<key>` 是逐帧序列,长度都是 `n_steps`
  - 文本: `language_instruction` / `language_motions` / `language_motions_future`(motion 串`|`分隔)
  - 数值向量: `action[7]` / `observation.joint_state[7]` / `observation.state[8]`(逐帧拼接成一维)
  - 图像: `observation.image` / `observation.wrist_image` 各 `224×224×3` 的 JPEG 字节
  - 标量: `is_first` / `is_last` / `is_terminal` / `reward` / `discount`

---

## 5. 常见问题

**Q1: bridge 跑起来很慢/吃内存爆掉**
确认装了 `ijson`。没装时 fallback 到 `json.load(open(path))`,会把整个 1.4 GB 文件全塞进 Python 字典,内存峰值 8–12 GB。

**Q2: `ModuleNotFoundError: No module named 'tfrecord'`**
`uv pip install tfrecord` 即可。**不要**改装 `tensorflow`。

**Q3: 为什么数值字段长度是 `n_steps × dim` 拼起来的一维数组?**
这是 TFDS 的 SequenceFeature 编码方式: `Tensor[7]` 的逐帧序列被存成 `float_list` 的一维平铺。脚本内置维度表(`action=7`, `joint_state=7`, `state=8`)做了 reshape 提示。

**Q4: 想看图像怎么办?**
脚本只打印 JPEG 字节长度。要可视化的话取出 `ex["steps/observation/image"][t]` 这块 bytes,直接喂给 `cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)` 或 `PIL.Image.open(io.BytesIO(b))`。

**Q5: 想批量统计/导出怎么办?**
viewer 的目的是肉眼看 schema 和样本。如要批量处理:
- bridge: 拿 [view_ecot_datasets.py:114](view_ecot_datasets.py#L114) 的 `_stream_bridge_with_ijson` 直接复用,改写循环体即可。
- libero: 直接调 `tfrecord.reader.tfrecord_loader(shard_path, None, description=None)`,迭代得到 `dict[str, ndarray]`,每集对应一个 dict。
