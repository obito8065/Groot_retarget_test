# Policy FK 与数据集 FK 一致性深度对比报告

## 1. 概述

本报告对比 **推理时 Policy 的 FK 流程** 与 **数据集处理的 body_retarget_robocasa_eepose_keypoints_v5.py** 的 FK 流程，聚焦运动学链、URDF、关节映射的完全一致性。

---

## 2. GR1T2 全身 URDF 差异

| 项目 | Policy (gr1_pos_transform) | 数据集 (body_retarget_v5) |
|------|----------------------------|---------------------------|
| **文件路径** | `groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf` | `robot_retarget/retarget/body_retarget/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf` |
| **差异** | 两份文件存在差异 | 同上 |

**diff 结果**：两文件在 **关节 limit (lower/upper)** 上不同：
- groot_retarget 版：部分关节使用宽松限位（如 -3.0 ~ 3.0）
- robot_retarget 版：使用更紧的限位（如 -2.79 ~ 1.92）

**FK 影响**：关节 limit 在 FK 时**不参与计算**，仅影响 IK 或约束求解。若 joint 值在限位内，**FK 结果一致**。但为统一维护，建议 **Policy 和数据集使用同一份 URDF**。

---

## 3. 手腕运动学链（arm chain）

### 3.1 链结构

| | Policy | 数据集 |
|---|--------|--------|
| **实现** | KDL，`getChain(torso_link, L_hand_base_link)` | Pinocchio 全机 FK，取 `L_hand_base_link` |
| **末端 link** | `L_hand_base_link` | `L_hand_base_link` |
| **基座** | `torso_link` | world → torso 相对变换 |
| **手腕 link** | ✓ 一致 | ✓ 一致 |

### 3.2 关节映射（arm：action[0:7] → q）

| 索引 | 关节名 | body_retarget | gr1_pos_transform |
|------|--------|---------------|-------------------|
| 0 | left_shoulder_pitch | action[0] | action_slices["left_arm"][0] |
| 1 | left_shoulder_roll | action[1] | action_slices["left_arm"][1] |
| 2 | left_shoulder_yaw | action[2] | action_slices["left_arm"][2] |
| 3 | left_elbow_pitch | action[3] | action_slices["left_arm"][3] |
| 4 | left_wrist_yaw | action[4] | action_slices["left_arm"][4] |
| 5 | left_wrist_roll | action[5] | action_slices["left_arm"][5] |
| 6 | left_wrist_pitch | action[6] | action_slices["left_arm"][6] |

**结论**：arm 关节 **无符号翻转**，映射 **一致**。

---

## 4. 手指运动学链（hand chain）

### 4.1 Link 与关节顺序

| | 数据集 (GR1T2 内嵌手) | Policy (fourier_left_hand.urdf) | 一致性 |
|---|----------------------|----------------------------------|--------|
| **手腕** | L_hand_base_link | hand_base 作为根 | ✓ |
| **5 指尖** | thumb, index, middle, ring, pinky | 同左 | ✓ |
| **6 主动关节** | L_index/middle/ring/pinky_proximal, L_thumb_proximal_yaw/pitch | 同左 | ✓ |
| **mimic** | 0.974, 1.128, 1.131, 1.143, 1.129 | 已对齐 | ✓ |

### 4.2 手指关节映射与符号（关键差异）

**数据集 (build_full_joint_array)**：
```python
# action[7:13] = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
L_pinky_proximal_joint    = clip(-action_vector[7])
L_ring_proximal_joint     = clip(-action_vector[8])
L_middle_proximal_joint   = clip(-action_vector[9])
L_index_proximal_joint    = clip(-action_vector[10])
L_thumb_proximal_pitch    = clip(+action_vector[11])   # 无负号
L_thumb_proximal_yaw      = clip(-action_vector[12])
```

**Policy (gr1_hand_fk)**：
```python
# left_hand_orig 格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# 仅做重排: left_hand_fk = left_hand_orig[..., [3,2,1,0,5,4]]
# 即 [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
# 直接写入 URDF 关节，无 sign flip，无 clip
```

| 关节 | 数据集 q 值 | Policy q 值 | 一致性 |
|------|-------------|-------------|--------|
| pinky | -action[7] (clip) | action[7] (重排后) | ❌ 符号相反 |
| ring | -action[8] (clip) | action[8] | ❌ 符号相反 |
| middle | -action[9] (clip) | action[9] | ❌ 符号相反 |
| index | -action[10] (clip) | action[10] | ❌ 符号相反 |
| thumb_pitch | +action[11] (clip) | action[11] | ✓ |
| thumb_yaw | -action[12] (clip) | action[12] | ❌ 符号相反 |

**结论**：Policy 端对 5 个手指关节（pinky, ring, middle, index, thumb_yaw）**未做符号翻转**，与数据集不一致。thumb_pitch 一致。这会导致在相同 state 下，**指尖 FK 结果不同**。

### 4.3 是否需要 clip

`body_retarget` 对 finger 做了 `np.clip` 到 URDF 关节 limit。若仿真 state 已在此范围内，clip 影响很小；若会越界，则会产生差异。建议在 Policy 侧也做同样的 clip，以完全对齐。

---

## 5. 相机外参

| | Policy (gr1_pos_transform) | 数据集 (body_retarget) |
|---|----------------------------|------------------------|
| t_cam_in_head | [2.650-2.65017178+0.23, -1.944+2.174-0.23, 1.538-1.4475] | 相同 |
| q_cam_in_head | [-0.205, 0.676, -0.676, 0.205] | 相同 |

**结论**：相机外参 **一致**。

---

## 6. 问题汇总与修复建议

### 6.1 必须修复

| 问题 | 位置 | 建议 |
|------|------|------|
| **手指关节符号** | policy.py 第 1 步输入处理 | 对 `left_hand_fk` / `right_hand_fk` 按 body_retarget 规则做 sign flip：pinky, ring, middle, index, thumb_yaw 取负，thumb_pitch 保持原样 |
| **手指 clip** | 同上 | 对 6 维 finger 做 `np.clip` 到 `hand_action_ranges`，与 body_retarget 一致 |

### 6.2 建议统一

| 问题 | 建议 |
|------|------|
| **GR1T2 URDF 来源** | Policy 与数据集使用同一份 URDF（建议采用 robot_retarget 版本） |
| **GR1T2 URDF 副本** | 将 `robot_retarget/.../GR1T2_fourier_hand_6dof.urdf` 复制或软链接到 `groot_retarget/gr00t/eval/robot_assets/GR1T2/`，并统一引用 |

### 6.3 已确认一致

- 手腕 link：L_hand_base_link, R_hand_base_link
- 指尖 link 及顺序：thumb, index, middle, ring, pinky
- 手部 mimic 参数：fourier_left/right_hand.urdf 已与 GR1T2 对齐
- Arm 关节映射：无符号翻转
- 相机外参：一致

---

## 7. Policy 输入处理建议代码变更（伪代码）

在 `policy.py` 第 1 步中，将：

```python
left_hand_fk = left_hand_orig[..., [3, 2, 1, 0, 5, 4]]
right_hand_fk = right_hand_orig[..., [3, 2, 1, 0, 5, 4]]
```

改为（与 body_retarget 一致）：

```python
# 数据集格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# 符号: pinky,ring,middle,index,thumb_yaw 取负; thumb_pitch 保持
hand_action_ranges = {
    "pinky": (-1.57, 0), "ring": (-1.57, 0), "middle": (-1.57, 0),
    "index": (-1.57, 0), "thumb_pitch": (0, 1.22), "thumb_yaw": (-1.74, 0),
}
# 左手
lh = left_hand_orig  # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
lh_clipped = np.stack([
    np.clip(-lh[..., 0], hand_action_ranges["pinky"][0], hand_action_ranges["pinky"][1]),
    np.clip(-lh[..., 1], hand_action_ranges["ring"][0], hand_action_ranges["ring"][1]),
    np.clip(-lh[..., 2], hand_action_ranges["middle"][0], hand_action_ranges["middle"][1]),
    np.clip(-lh[..., 3], hand_action_ranges["index"][0], hand_action_ranges["index"][1]),
    np.clip( lh[..., 4], hand_action_ranges["thumb_pitch"][0], hand_action_ranges["thumb_pitch"][1]),
    np.clip(-lh[..., 5], hand_action_ranges["thumb_yaw"][0], hand_action_ranges["thumb_yaw"][1]),
], axis=-1)
# FK 期望顺序: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
left_hand_fk = lh_clipped[..., [3, 2, 1, 0, 5, 4]]
# 右手同理
```

---

## 8. 验证建议

在同一帧 44 维 state 下：

1. 用 `body_retarget` 的 `compute_keypoints` 得到 `left_keypoints`、`right_keypoints`
2. 用 Policy 的 `_build_full_44dof_vector` + `process_frame_kinematics_axisangle` + `policy_fourier_hand_keypoints.compute_state_45d` 得到 `state_45d` 中的关键点
3. 逐项对比 wrist_xyz、5 个指尖 xyz、wrist_rotvec，误差应在 1e-5 量级

若误差较大，重点检查手指关节的 sign 与 clip 是否与 body_retarget 完全一致。
