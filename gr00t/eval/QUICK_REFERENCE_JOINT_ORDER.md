# Fourier Hand 关节顺序快速参考

## 关键信息一览

### 数据集格式
```python
state.left_hand: (B, T, 6)
# 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# 索引:  [  0  ,  1  ,   2   ,   3  ,      4      ,     5     ]
```

### FK输入格式
```python
# gr1_hand_fk.py期望的顺序
finger_joint_names_6 = [
    "index_proximal_joint",     # 0
    "middle_proximal_joint",    # 1
    "ring_proximal_joint",      # 2
    "pinky_proximal_joint",     # 3
    "thumb_proximal_yaw_joint", # 4
    "thumb_proximal_pitch_joint" # 5
]
# 顺序: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
```

### 重排序映射
```python
dataset_to_fk_mapping = [3, 2, 1, 0, 5, 4]

# 含义:
# FK[0] = 数据集[3]  (index)
# FK[1] = 数据集[2]  (middle)
# FK[2] = 数据集[1]  (ring)
# FK[3] = 数据集[0]  (pinky)
# FK[4] = 数据集[5]  (thumb_yaw)
# FK[5] = 数据集[4]  (thumb_pitch)
```

### FK输出格式
```python
keypoints: (B, T, 6, 3)
# 顺序: [wrist, thumb, index, middle, ring, pinky]
# 坐标系: camera frame
```

### Retarget输入格式
```python
input: (6, 3)
# 顺序: [wrist, thumb, index, middle, ring, pinky]
# 与FK输出一致
```

### Retarget输出格式
```python
result = {
    'left': {
        'wrist_pose': (6,),     # [x, y, z, rx, ry, rz]
        'finger_joints': (6,),  # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    }
}

# finger_joints顺序与数据集格式完全一致！✓
```

## 代码片段

### Policy中的FK调用
```python
# 重排序
dataset_to_fk_mapping = [3, 2, 1, 0, 5, 4]
left_hand_fk = left_hand_orig[..., dataset_to_fk_mapping]

# FK计算
left_keypoints, right_keypoints = self.policy_fourier_hand_keypoints.compute_keypoints(
    left_arm=left_arm_orig,
    left_hand=left_hand_fk,
    right_arm=right_arm_orig,
    right_hand=right_hand_fk,
)

# Flatten for model input
obs_copy["state.left_hand"] = left_keypoints.reshape(B, T, 18)
```

### Policy中的Retarget调用
```python
# Reshape model output
left_kp_t = pred_left_hand_seq[b, t].reshape(6, 3)

# Retarget
result = self.fourier_hand_retargeter.retarget(
    left_keypoints=left_kp_t,
    right_keypoints=right_kp_t
)

# Extract (顺序已正确，直接使用)
retarget_left_hand_seq[b, t] = result['left']['finger_joints']
```

## 验证清单

- [x] 数据集格式: `[pinky, ring, middle, index, thumb_pitch, thumb_yaw]`
- [x] FK输入格式: `[index, middle, ring, pinky, thumb_yaw, thumb_pitch]`
- [x] 重排序映射: `[3, 2, 1, 0, 5, 4]`
- [x] FK输出格式: `[wrist, thumb, index, middle, ring, pinky]`
- [x] Retarget输出: `[pinky, ring, middle, index, thumb_pitch, thumb_yaw]` ← 与数据集一致

## 重要提示

⚠️ **只有一个地方需要重排序**: 从数据集格式到FK输入

✓ **Retarget输出不需要重排序**: 已经是数据集格式

✓ **关键点顺序始终一致**: `[wrist, thumb, index, middle, ring, pinky]`

