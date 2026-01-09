# Fourier Hand FK & Retarget Integration Guide

## 概述

本文档详细说明了Fourier Hand FK和Retarget在Policy中的集成方案，**重点关注各个阶段的关节/关键点顺序对齐**。

## 整体架构

```
数据集 (Joint Angles)
    ↓ FK
Policy输入 (6 Keypoints)
    ↓ 模型推理
Policy输出 (6 Keypoints)
    ↓ Retarget
最终输出 (Joint Angles)
```

## 关键点顺序对齐

### 1. 数据集格式

**输入到Policy的数据集格式**：

```python
obs = {
    "state.left_arm": (B, T, 7),    # wrist_pose [pos3, rotvec3, optional1]
    "state.left_hand": (B, T, 6),   # finger joints
}
```

**数据集中的Hand Joint顺序**（参考`policy.py`注释）：
```python
# [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# 即索引: [0, 1, 2, 3, 4, 5]
```

### 2. FK阶段（输入处理）

**FK输入期望的Joint顺序**（参考`gr1_hand_fk.py`）：
```python
# L_finger_joint_names_6: 
#   L_index_proximal_joint 
#   L_middle_proximal_joint 
#   L_ring_proximal_joint 
#   L_pinky_proximal_joint 
#   L_thumb_proximal_yaw_joint 
#   L_thumb_proximal_pitch_joint

# 即: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
```

**重排序映射**：
```python
# 数据集 -> FK输入
dataset_to_fk_mapping = [3, 2, 1, 0, 5, 4]

# 具体对应:
# 数据集[0] pinky       -> FK[3] pinky
# 数据集[1] ring        -> FK[2] ring
# 数据集[2] middle      -> FK[1] middle
# 数据集[3] index       -> FK[0] index
# 数据集[4] thumb_pitch -> FK[5] thumb_pitch
# 数据集[5] thumb_yaw   -> FK[4] thumb_yaw
```

**FK输出格式**：
```python
# (B, T, 6, 3) - 6个关键点坐标
# 顺序: [wrist, thumb, index, middle, ring, pinky]
# 每个关键点: [x, y, z] in camera frame
```

**模型输入格式**（FK输出flatten）：
```python
obs_copy["state.left_hand"] = left_keypoints.reshape(B, T, 18)
# 即: [wrist_x, wrist_y, wrist_z, 
#      thumb_x, thumb_y, thumb_z,
#      index_x, index_y, index_z,
#      middle_x, middle_y, middle_z,
#      ring_x, ring_y, ring_z,
#      pinky_x, pinky_y, pinky_z]
```

### 3. 模型推理

模型在18维的关键点空间中学习。

### 4. Retarget阶段（输出处理）

**模型输出格式**：
```python
unnormalized_action["action.left_hand"]: (B, horizon, 18)
# 需要reshape成 (B, horizon, 6, 3)
```

**Retarget输入格式**（与FK输出一致）：
```python
# (6, 3) - 6个关键点
# 顺序: [wrist, thumb, index, middle, ring, pinky]
```

**Retarget输出格式**（参考`fourier_hand_retarget_api.py`）：
```python
result = {
    'left': {
        'wrist_pose': (6,),      # [pos_xyz(3), rotvec_xyz(3)]
        'finger_joints': (6,),   # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    },
    'right': {...}
}
```

**重要**：Retarget输出的`finger_joints`顺序**正好与数据集一致**！
```python
# Retarget输出: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# 数据集格式:   [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# ✓ 完全一致，无需重排序！
```

## 代码实现

### 1. Policy初始化

```python
# policy.py __init__方法中
if self.use_fourier_hand_retarget:
    if "robocasa" in self.embodiment_tag.value:
        # 初始化FK
        from gr00t.eval.gr1_hand_fk import PolicyFourierHandKeypoints
        self.policy_fourier_hand_keypoints = PolicyFourierHandKeypoints(...)
        
        # 初始化Retarget
        from gr00t.eval.fourier_hand_retarget_api import FourierHandRetargetAPI
        self.fourier_hand_retargeter = FourierHandRetargetAPI()
```

### 2. 输入处理（FK）

```python
# policy.py get_action方法中，模型推理前
if self.use_fourier_hand_retarget:
    # 1. 读取原始joint angles
    left_hand_orig = obs_copy["state.left_hand"]  # (B, T, 6)
    
    # 2. 重排序: 数据集格式 -> FK期望格式
    dataset_to_fk_mapping = [3, 2, 1, 0, 5, 4]
    left_hand_fk = left_hand_orig[..., dataset_to_fk_mapping]
    
    # 3. 执行FK
    left_keypoints, right_keypoints = self.policy_fourier_hand_keypoints.compute_keypoints(
        left_arm=left_arm_orig,
        left_hand=left_hand_fk,
        right_arm=right_arm_orig,
        right_hand=right_hand_fk,
    )
    
    # 4. Flatten并替换obs
    obs_copy["state.left_hand"] = left_keypoints.reshape(B, T, 18)
```

### 3. 输出处理（Retarget）

```python
# policy.py get_action方法中，模型推理后
if self.use_fourier_hand_retarget:
    # 1. 获取模型输出
    pred_left_hand_seq = unnormalized_action["action.left_hand"]  # (B, horizon, 18)
    
    # 2. 逐帧retarget
    for b in range(batch_size):
        for t in range(horizon):
            # Reshape: (18,) -> (6, 3)
            left_kp_t = pred_left_hand_seq[b, t].reshape(6, 3)
            
            # Retarget
            result = self.fourier_hand_retargeter.retarget(
                left_keypoints=left_kp_t,
                right_keypoints=right_kp_t
            )
            
            # 提取finger joints (顺序已经正确)
            retarget_left_hand_seq[b, t] = result['left']['finger_joints']
    
    # 3. 更新action
    unnormalized_action["action.left_hand"] = retarget_left_hand_seq
```

### 4. Reset机制

```python
# policy.py reset_ik_cache方法中
def reset_ik_cache(self, env_idx: Optional[int] = None):
    # 清空Hand Retarget缓存
    if hasattr(self, "fourier_hand_retargeter"):
        self.fourier_hand_retargeter.reset(env_idx)
```

在每个episode开始时调用：
```python
if reset_mask is not None:
    for env_idx, flag in enumerate(reset_mask):
        if bool(flag):
            self.reset_ik_cache(env_idx=env_idx)
```

## 验证要点

### 1. 关节顺序验证

**输入验证**：
```python
# 数据集: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
dataset_hand = obs["state.left_hand"][0, -1]  # 最后一帧

# FK输入重排序后
fk_input_hand = dataset_hand[[3, 2, 1, 0, 5, 4]]
# FK输入: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]

# 验证：pinky在数据集中是索引0，在FK输入中应该是索引3
assert dataset_hand[0] == fk_input_hand[3]  # pinky
```

**输出验证**：
```python
# Retarget输出
retarget_output = result['left']['finger_joints']  # (6,)
# 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]

# 与数据集格式一致，可以直接使用
action["action.left_hand"] = retarget_output
```

### 2. 数值验证

使用`ROTATION_DIFFERENCE_EXPLAINED.md`中的结论：
- Wrist位置误差应 < 0.1mm （非常准确）
- Wrist旋转会有差异（这是正常的，因为position retargeting只约束位置）
- Fingertip位置应非常准确（这是优化目标）

### 3. 端到端验证

```python
# 1. 输入joint angles -> FK -> keypoints
# 2. Keypoints -> 模型 -> predicted keypoints
# 3. Predicted keypoints -> Retarget -> output joint angles

# 如果模型预测完美，则：
# output joint angles ≈ input joint angles (手指部分)
# 但wrist rotation可能不同（参考ROTATION_DIFFERENCE_EXPLAINED.md）
```

## 性能考虑

1. **FK计算**：每次`get_action`调用一次，批量处理所有时间步
2. **Retarget计算**：循环处理每个batch和时间步
   - 对于batch=5, horizon=16: 需要80次retarget调用
   - 考虑优化：批量retarget API（未来改进）

## 调试技巧

### 1. 打印关键信息

```python
# 在FK阶段
print(f"Dataset hand shape: {left_hand_orig.shape}")
print(f"FK input hand (reordered): {left_hand_fk[0, -1]}")
print(f"FK output keypoints shape: {left_keypoints.shape}")

# 在Retarget阶段
print(f"Model output hand shape: {pred_left_hand_seq.shape}")
print(f"Retarget input keypoints: {left_kp_t}")
print(f"Retarget output joints: {result['left']['finger_joints']}")
```

### 2. 可视化验证

参考`hand_robot_viewer_robocasa_txt.py`：
- 渲染原始6个关键点（红色小球）
- 渲染retarget后的机器人手
- 直观检查对齐情况

### 3. 数值对比

创建测试脚本：
```python
# 对比FK输出的keypoints和直接从数据集读取的keypoints
fk_keypoints = compute_fk(arm, hand)
dataset_keypoints = load_from_txt(episode_*_6keypoints_xyz.txt)
error = np.abs(fk_keypoints - dataset_keypoints).max()
print(f"FK vs Dataset max error: {error * 1000:.3f} mm")
```

## 常见问题

### Q1: 为什么需要重排序？

A: 因为数据集中的joint顺序与FK/URDF中定义的顺序不同。必须保证输入到FK的joint angles顺序与URDF中定义的一致。

### Q2: Retarget输出为什么不需要重排序？

A: `FourierHandRetargetAPI`内部已经处理了顺序转换，输出的`finger_joints`顺序固定为`[pinky, ring, middle, index, thumb_pitch, thumb_yaw]`，与数据集格式一致。

### Q3: Position Retargeting的旋转误差是否影响使用？

A: 对于抓取/操作任务，fingertip位置准确就足够了，wrist旋转的差异可以接受。详见`ROTATION_DIFFERENCE_EXPLAINED.md`。

### Q4: 如何验证集成正确？

A: 
1. 检查所有shape是否正确
2. 验证关节顺序映射
3. 端到端测试：输入joint angles -> FK -> 模型 -> Retarget -> 输出joint angles
4. 可视化检查（如有条件）

## 参考文档

1. `ROTATION_DIFFERENCE_EXPLAINED.md` - Position Retargeting的旋转差异解释
2. `gr1_hand_fk.py` - FK实现和输入输出格式
3. `fourier_hand_retarget_api.py` - Retarget API和输出格式
4. `hand_robot_viewer_robocasa_txt.py` - 完整的可视化验证脚本

## 总结

关键要记住的顺序对应：

```
数据集: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
         ↓ 重排序 [3,2,1,0,5,4]
FK输入: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
         ↓ FK计算
关键点: [wrist, thumb, index, middle, ring, pinky]
         ↓ 模型推理
关键点: [wrist, thumb, index, middle, ring, pinky]
         ↓ Retarget
输出:    [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
         ✓ 与数据集格式一致！
```

