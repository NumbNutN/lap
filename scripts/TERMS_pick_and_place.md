# Pick-and-place 标注术语表（中→英）

> 用于手动 CoT 标注（`scripts/annotate_teleop_episodes.py`）的 `stage` / `action` / `think` 字段填写参考。风格对齐 ECoT 论文 + DROID 标注流水线，保证手动标注跟 VLM 自动标注的语料分布一致。

## 1. 通用规则

- `stage`：**陈述句**（"The robot ..."），1-3 句，描述这段帧里发生的事
- `action`：**祈使句**（"Move ..."），≤12 词，单一动作
- `think`：可选，**因果推理**（失败原因 + 修正方向 / 替代方案）
- 物体命名：`<color> <type>` —— `blue cube`、`leftmost red cube`、`middle cube`
- 朝向：**先轴名再方向** —— `yaw counterclockwise` 而非纯 `turn left`
- 避障：用 `clear the [obstacle]` / `arc around the [obstacle]`，别用 `dodge` / `bypass`

## 2. 抓前阶段（pre-grasp）

### 移动夹爪到抓取位

| 中文 | 推荐英文（stage） | 等价 action |
|---|---|---|
| 移动夹爪到 X 的抓取位 | _The robot is approaching the [blue cube] from above._ | `Move the gripper above the blue cube` |
| 强调下降 | _The robot lowers the gripper to the pre-grasp pose._ | `Lower the gripper to the pre-grasp pose` |
| 微调对齐 | _The gripper is being centered over the [blue cube]._ | `Align the gripper with the blue cube` |
| 接近物体 | _Approaching the target object._ | `Approach the blue cube` |

**关键词**：`approach` / `move above` / `lower to pre-grasp` / `align with` / `center over`

### 朝向调整

把"逆时针旋转 / 向左摆 / 向上仰"这些**先轴名后方向**，模型 + 人 reader 都不歧义：

| 中文 | 物理含义 | 英文 |
|---|---|---|
| 逆时针旋转夹爪 | yaw 旋转 | `Rotate the gripper counterclockwise (yaw)` |
| 顺时针旋转夹爪 | yaw 旋转 | `Rotate the gripper clockwise (yaw)` |
| 向左摆（yaw 小角） | yaw 旋转 | `Yaw the gripper to the left` 或 `Turn the gripper leftward` |
| 向上仰（pitch） | pitch 旋转 | `Tilt the gripper upward (pitch)` |
| 向下俯（pitch） | pitch 旋转 | `Tilt the gripper downward (pitch)` 或 `Pitch the gripper down` |
| 翻滚（roll） | roll 旋转 | `Roll the gripper clockwise/counterclockwise` |
| 使开口对齐方块 | 朝向匹配 | `Align the gripper jaws with the cube's edge` |
| 使开口跟方块长边对齐 | 朝向匹配 | `Align the gripper's opening direction with the cube's long axis` |

**关键词**：`rotate` / `tilt` / `pitch` / `yaw` / `roll` / `align ... with ...`

**ECoT 简化版**（论文里几乎都用方向词，不用 yaw/pitch/roll）：

- 笼统"调整朝向"：`Adjust the gripper orientation` / `Reorient the gripper`
- primitive 风格：`tilt up` / `tilt down` / `rotate clockwise` / `rotate counterclockwise`

## 3. 抓取本身（grasp）

| 中文 | 英文 |
|---|---|
| 闭合夹爪夹住 | `Close the gripper to grasp the [blue cube]` |
| 牢牢夹住 | `Firmly grasp the [blue cube]` |
| 抓起 | `Pick up the [blue cube]` |
| 抬起脱离桌面 | `Lift the [blue cube] off the table` |

**关键词**：`close gripper` / `grasp` / `pick up` / `lift`

## 4. 运输（transport / 避障）

| 中文 | 英文 |
|---|---|
| 搬运到 Y 上方 | `Carry the [blue cube] above the [leftmost red cube]` |
| 移动到 Y 上方 | `Move the [blue cube] over the target cell` |
| 为避免撞到中间方块，抬高夹爪 | `Raise the gripper to clear the middle cube before crossing over it` |
| 沿弧形路径移动避开障碍 | `Arc the gripper around the middle obstacle` |
| 抬升再水平移动 | `Lift first, then translate horizontally to the target` |

**关键词**：`carry` / `transport` / `raise to clear` / `clear the obstacle` / `arc around` / `lift before translating`

## 5. 放置（place）

| 中文 | 英文 |
|---|---|
| 下降到放置位 | `Lower the [blue cube] onto the [red cube]` |
| 轻放 | `Gently place the [blue cube] on top of the [red cube]` |
| 张开夹爪释放 | `Open the gripper to release the [blue cube]` |
| 释放后退离 | `Retract the gripper away from the placed cube` |
| 复位 | `Return the gripper to a ready pose above the workspace` |

**关键词**：`lower onto` / `place on top of` / `release` / `open gripper` / `retract` / `return to ready pose`

## 6. 失败 / 重试（type=retry 的 `think` 字段）

| 中文情境 | 推荐 think 句式 |
|---|---|
| 抓偏了 | _The previous attempt grasped the cube off-center, causing it to slip. The robot will reposition the gripper and retry._ |
| 抓空了 | _The gripper closed before reaching the cube; it grasped empty air. Retry with a deeper approach._ |
| 抓住但滑落 | _The cube slipped out of the gripper during lifting. Lower the gripper and re-grasp with better alignment._ |
| 方块掉落 | _The placed cube was unstable and toppled off the target. Pick it up again and place it more carefully._ |
| 撞到障碍 | _The previous trajectory collided with the [middle cube]. Replan with extra clearance above obstacles._ |
| 朝向错了导致夹不住 | _The gripper jaws were misaligned with the cube's edges and could not get a grip. Reorient and retry._ |

对应 `action`：
- `Re-grasp the [blue cube]`
- `Retry the grasp with adjusted approach`
- `Re-attempt to pick up the slipped cube`
- `Reposition the gripper and grasp again`

## 7. 方位 / 空间介词词典

| 关系 | 介词 |
|---|---|
| X 在 Y 正上方 | `directly above` / `right above` |
| X 在 Y 上方（较高的空中） | `above` / `over` |
| X 接触在 Y 上面 | `on top of` / `on` |
| X 在 Y 旁边（贴近） | `next to` / `beside` / `adjacent to` |
| X 在 Y 前方（远端） | `in front of` |
| X 在 Y 后方（近端） | `behind` |
| 桌子最左 | `leftmost` / `on the far left of the table` |
| 桌子最右 | `rightmost` / `on the far right of the table` |
| 桌子中间 | `middle` / `in the center of the table` |
| 桌子前缘（远离机器人） | `at the far edge of the table` |
| 桌子后缘（靠近机器人） | `at the near edge of the table` |

## 8. 用户句子的英文译版（直接抄）

```
移动夹爪到蓝色方块的抓取位
→ Move the gripper to a pre-grasp pose directly above the blue cube.

（逆时针旋转）夹爪使开口和下方方块对齐
→ Rotate the gripper counterclockwise (yaw) so its opening aligns with the cube below.

（向左摆）夹爪使开口和下方方块对齐
→ Yaw the gripper to the left so its jaws align with the cube's edges.

（向上仰）夹爪使开口和下方方块对齐
→ Tilt the gripper upward (pitch) so its opening faces the cube squarely.

抓偏了，重新移动夹爪到蓝色方块的抓取位
stage:  The previous grasp was off-center; reposition the gripper for a second attempt.
think:  The cube slipped from an off-center grip last time; retry with better centering.
action: Re-approach the blue cube for another grasp.

方块没放稳，掉落了下来，重新移动夹爪到..
stage:  The placed cube toppled off the target; the robot will pick it up again.
think:  The placement was unstable and the cube fell; re-grasp and place more carefully.
action: Re-grasp the fallen blue cube.

将蓝色方块搬运到最左边的红色方块上方，为了避免撞到中间的方块，运输时抬高夹爪
→ Carry the blue cube toward the leftmost red cube, raising the gripper to clear the middle cube along the way.
(简洁版 action) Lift and arc over the middle cube to reach the leftmost red cube.
```

## 9. 完整范例：一集 pick-and-place 的全套标注

任务：拿起蓝方块放到最左边红方块上，中间有一个挡路的方块。

```json
{
  "task_instruction": "Pick up the blue cube and stack it on top of the leftmost red cube, avoiding the middle cube on the path.",
  "plan": "1. Approach and grasp the blue cube. 2. Lift it and carry it over the middle cube. 3. Place it on top of the leftmost red cube. 4. Retract the gripper.",
  "keyframes": [
    {
      "frame_start": 0, "frame_end": 8,
      "type": "begin", "gripper_state": "open",
      "stage": "Robot starts at the ready pose above the table.",
      "think": null,
      "action": "Wait at the ready pose"
    },
    {
      "frame_start": 9, "frame_end": 28,
      "type": "motion", "gripper_state": "open",
      "stage": "The robot moves the gripper above the blue cube and lowers to a pre-grasp pose.",
      "think": null,
      "action": "Approach and lower to the blue cube"
    },
    {
      "frame_start": 29, "frame_end": 36,
      "type": "motion", "gripper_state": "open",
      "stage": "The gripper is yawed counterclockwise so its opening aligns with the cube's edges.",
      "think": null,
      "action": "Yaw the gripper counterclockwise to align with the cube"
    },
    {
      "frame_start": 37, "frame_end": 44,
      "type": "grasp", "gripper_state": "closed",
      "stage": "The gripper closes firmly around the blue cube.",
      "think": null,
      "action": "Close the gripper to grasp the blue cube"
    },
    {
      "frame_start": 45, "frame_end": 60,
      "type": "motion", "gripper_state": "closed",
      "stage": "The robot lifts the cube and arcs it over the middle cube to clear the obstacle.",
      "think": null,
      "action": "Lift and arc over the middle cube"
    },
    {
      "frame_start": 61, "frame_end": 74,
      "type": "motion", "gripper_state": "closed",
      "stage": "The robot lowers the blue cube onto the leftmost red cube.",
      "think": null,
      "action": "Lower the cube onto the red target"
    },
    {
      "frame_start": 75, "frame_end": 82,
      "type": "release", "gripper_state": "open",
      "stage": "The gripper opens to release the blue cube.",
      "think": null,
      "action": "Open the gripper to release"
    },
    {
      "frame_start": 83, "frame_end": 90,
      "type": "end", "gripper_state": "open",
      "stage": "The robot retracts the gripper to a ready pose above the workspace.",
      "think": null,
      "action": "Retract to the ready pose"
    }
  ]
}
```

## 10. 简洁原则（让标注更一致）

- `stage` 用陈述句（"The robot ..."），1-3 句
- `action` 用祈使句（"Move ..." "Close ..." "Lift ..."），≤12 词
- 颜色 + 物体类型命名：`blue cube` / `leftmost red cube` / `middle cube`
- 朝向：先轴名再方向 — `yaw counterclockwise` 不歧义
- 避障：`clear the [obstacle]` / `arc around the [obstacle]`
- retry 类 `think`：失败原因 + 修正方向两点

## 11. Axis-aware action 词典（v3 memory-augmented 模式）

> 与 [`README_prompt_engineering_spec.md`](README_prompt_engineering_spec.md) §13 v3 实现配套。
> Auto-annotator 现在接收 per-keyframe pose delta：`Δxyz=(+1.2cm,-0.4cm,-3.1cm) Δrot=8° around -pitch`，VLM 应输出引用具体轴和量级的动作。

### 11.1 输入 pose delta → 推荐 action 句式

| pose delta 主导分量 | 推荐 action 句式 |
|---|---|
| `Δxyz` 中 +x 大（forward） | `Translate forward N cm to approach the [cube]` |
| `Δxyz` 中 -x 大（backward） | `Retract N cm away from the [cube]` |
| `Δxyz` 中 +y 大（leftward） | `Translate leftward N cm towards the [leftmost red cube]` |
| `Δxyz` 中 -y 大（rightward） | `Translate rightward N cm towards the [rightmost red cube]` |
| `Δxyz` 中 +z 大（up） | `Lift N cm above the workspace` |
| `Δxyz` 中 -z 大（down） | `Lower N cm to the pre-grasp pose` |
| `Δrot` axis=`yaw` 正 | `Yaw counterclockwise N degrees` |
| `Δrot` axis=`-yaw` | `Yaw clockwise N degrees` |
| `Δrot` axis=`pitch` 正 | `Pitch upward N degrees (tilt face up)` |
| `Δrot` axis=`-pitch` | `Pitch downward N degrees (tilt face down)` |
| `Δrot` axis=`roll` ± | `Roll {clockwise / counterclockwise} N degrees` |
| `Δrot` axis=`compound` | `Reorient toward the [target object]` (不强求拆轴) |

### 11.2 复合动作（多维度同时变）

当 pose delta 同时有 translation + rotation：

| 情境 | 推荐句式 |
|---|---|
| 下降 + yaw | `Lower N cm while yawing counterclockwise to align with the cube` |
| 前进 + pitch | `Translate forward N cm while pitching downward toward the table` |
| 抬升 + reorient | `Lift while reorienting toward the target row` |

### 11.3 何时**忽略** pose delta（grip 动词优先）

在以下 keyframe 类型，**抓握/释放语义比 pose delta 重要**，action 必须含 grip 动词：

| keyframe `type` | action 必含 | 不要写 |
|---|---|---|
| `grasp` | `close` / `grasp` / `pick up` | "Pitch downward and move closer to the cube"（缺 grip 动词） |
| `release` | `open` / `release` / `place` | "Yaw slightly while moving closer to the sink"（缺 release 动词） |
| `retry` | `re-grasp` / `re-approach` / `retry` | 任何只描述 pose 调整的句子 |

如果 grip 动作之外还有显著 pose 变化（比如 grasp 时还在 yaw），优先写 grip 动词，pose 用副句修饰：

- ✅ `Close the gripper to grasp the marker while yawing counterclockwise`
- ❌ `Yaw counterclockwise to align with the marker` （对 grasp keyframe 缺动词）

### 11.4 量级修饰词（when N too small for cm/deg precision）

| 数值范围 | 修饰词 |
|---|---|
| < 0.5 cm 或 < 3° | `slightly` / `a bit` / `marginal` |
| 0.5-2 cm 或 3-10° | `a little` / `minor` |
| 2-5 cm 或 10-25° | `moderately` / `clearly` |
| > 5 cm 或 > 25° | `significantly` / 直接报数字 (`5 cm`, `30°`) |

### 11.5 v3 实测 action 例子（从 pilot 抓的）

来自 [`REPORT_v3_memory_pose_pilot.md`](REPORT_v3_memory_pose_pilot.md) 实际 VLM 输出：

```
Adjust position by tilting downward and moving right.
Yaw counterclockwise and move closer to the sachet.
Pitch upward and move closer to the sachet.
Pitch downward and lift the sachet higher.
Move towards the bowl while pitching upward.
Move closer to the bowl while adjusting pitch.
```

这些 action **从 v2 升级到 v3** 后才出现 —— 没有 pose delta 输入时模型只会说 "Adjust the arm's position" / "Move closer"。

### 11.6 Memory-augmented stage 例子（v3 实测）

stage 现在 reference 之前发生的事件（image 看不出的历史）：

```
Following the initial search, the robot adjusts its position to better locate the sachet.
After locating the sachet, the robot adjusts its orientation to prepare for grasping.
With the sachet partially grasped, the robot prepares to lift it off the table.
Post-release, the robot makes minor adjustments to confirm the sachet is properly placed in the bowl.
```

特征短语：`Following ...` / `After ...ing` / `With X having ...` / `Post-release` / `Now that ...`
