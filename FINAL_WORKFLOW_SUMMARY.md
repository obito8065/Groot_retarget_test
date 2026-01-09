# 最终工作流程总结

## 前提条件
- `--use_eepose=True`
- `--use_fourier_hand_retarget=True`
- `--data_config=robocasa_retarget`
- `--embodiment_tag=robocasa`

---

## 完整的4步流程

### Step 0: use_eepose的FK - 原始joint angles → wrist pose
**代码位置**: `policy.py` L358-394

**输入格式**（从client/仿真环境获取）:
```python
obs["state.left_arm"]:  (B, T, 7) = [7个arm关节角度]
obs["state.left_hand"]: (B, T, 6) = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
obs["state.right_arm"]: (B, T, 7) = [7个arm关节角度]
obs["state.right_hand"]:(B, T, 6) = [6个finger关节角度]
obs["state.waist"]:     (B, T, 3) = [3个waist关节角度]
```

**处理过程**:
```python
# 使用body_retargeter进行正向运动学
full_44dof_vector = build_full_44dof_vector(obs)
(left_pos, left_axis), (right_pos, right_axis), (left_qpos, right_qpos) = \
    body_retargeter.process_frame_kinematics_axisangle(full_44dof_vector)

# 拼接wrist pose
left_eepose = np.concatenate((left_pos, left_axis), axis=-1)   # (B, 6)
right_eepose = np.concatenate((right_pos, right_axis), axis=-1) # (B, 6)
```

**输出格式**:
```python
obs["state.left_arm"]:  (B, T, 6) = [L_wrist_xyz(3), L_rotvec(3)]
obs["state.right_arm"]: (B, T, 6) = [R_wrist_xyz(3), R_rotvec(3)]
obs["state.left_hand"]: (B, T, 6) = [保持不变，finger joints]
obs["state.right_hand"]:(B, T, 6) = [保持不变，finger joints]
```

**日志输出**:
```
(无特殊日志，use_eepose的标准处理)
```

---

### Step 1: FK转换 - wrist pose + finger joints → hand keypoints
**代码位置**: `policy.py` L413-488

**触发条件**:
```python
if (self.use_eepose and self.use_fourier_hand_retarget and "robocasa" in self.embodiment_tag.value):
```

**输入格式**（从Step 0输出）:
```python
obs["state.left_arm"]:  (B, T, 6) = [L_wrist_xyz(3), L_rotvec(3)]
obs["state.left_hand"]: (B, T, 6) = [L_finger_q1~6]
                                    # 数据集格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
obs["state.right_arm"]: (B, T, 6) = [R_wrist_xyz(3), R_rotvec(3)]
obs["state.right_hand"]:(B, T, 6) = [R_finger_q1~6]
```

**处理过程**:
```python
# 1. 重排序finger joints: 数据集格式 → FK期望格式
#    数据集: [pinky(0), ring(1), middle(2), index(3), thumb_pitch(4), thumb_yaw(5)]
#    FK期望: [index(0), middle(1), ring(2), pinky(3), thumb_yaw(4), thumb_pitch(5)]
dataset_to_fk_mapping = [3, 2, 1, 0, 5, 4]
left_hand_fk = left_hand_orig[..., dataset_to_fk_mapping]
right_hand_fk = right_hand_orig[..., dataset_to_fk_mapping]

# 2. 调用policy_fourier_hand_keypoints.compute_keypoints()执行FK
#    输入：wrist pose(6维) + finger joints(6维，FK格式)
#    输出：6个关键点的3D坐标
left_keypoints, right_keypoints = policy_fourier_hand_keypoints.compute_keypoints(
    left_arm=left_arm_orig,      # (B, T, 6): [wrist_xyz, rotvec]
    left_hand=left_hand_fk,       # (B, T, 6): FK格式的finger joints
    right_arm=right_arm_orig,     # (B, T, 6)
    right_hand=right_hand_fk,     # (B, T, 6)
    time_major=False,
    return_time_major=False,
)
# 输出：(B, T, 6, 3) = 6个关键点 [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]

# 3. Flatten: (B, T, 6, 3) → (B, T, 18)
obs["state.left_key_points"] = left_keypoints.reshape(B, T, 18)
obs["state.right_key_points"] = right_keypoints.reshape(B, T, 18)
```

**输出格式**:
```python
obs["state.left_key_points"]:  (B, T, 18) = [wrist_xyz(3), thumb_tip_xyz(3), index_tip_xyz(3),
                                               middle_tip_xyz(3), ring_tip_xyz(3), pinky_tip_xyz(3)]
obs["state.right_key_points"]: (B, T, 18) = [同上]
obs["state.waist"]:            (B, T, 3)  = [保持不变]

# 移除原始数据
# obs中不再有 state.left_arm, state.left_hand, state.right_arm, state.right_hand
```

**日志输出**:
```
[Policy Step1] Input shapes: left_arm=(B, T, 6), left_hand=(B, T, 6)
[Policy Step1] Converted to keypoints:
  left_key_points: (B, T, 18)
  right_key_points: (B, T, 18)
```

---

### Step 2: 模型推理 - keypoints → 预测keypoints
**代码位置**: `policy.py` L535-558

**输入格式**（从Step 1输出）:
```python
obs["state.left_key_points"]:  (B, T, 18)
obs["state.right_key_points"]: (B, T, 18)
obs["state.waist"]:            (B, T, 3)
obs["video.xxx"]:              (B, T, H, W, C)
obs["annotation.xxx"]:         ...
```

**处理过程**:
```python
# 1. apply_transforms(): 数据归一化等预处理
normalized_input = self.apply_transforms(obs_copy)

# 2. 模型推理
normalized_action = self._get_action_from_normalized_input(normalized_input)

# 3. unapply_transforms(): 反归一化
unnormalized_action = self._get_unnormalized_action(normalized_action)
```

**输出格式**:
```python
unnormalized_action["action.left_key_points"]:  (B, horizon, 18) = [预测的左手6个关键点]
unnormalized_action["action.right_key_points"]: (B, horizon, 18) = [预测的右手6个关键点]
unnormalized_action["action.waist"]:            (B, horizon, 3)  = [预测的waist]
```

**日志输出**:
```
[Noise Debug] Using FIXED generator with seed: 0
```

---

### Step 3: Retarget转换 - hand keypoints → wrist pose + finger joints
**代码位置**: `policy.py` L594-690

**触发条件**:
```python
if (self.use_eepose and self.use_fourier_hand_retarget and 
    "robocasa" in self.embodiment_tag.value and
    "action.left_key_points" in unnormalized_action):
```

**输入格式**（从Step 2输出）:
```python
unnormalized_action["action.left_key_points"]:  (B, horizon, 18) = [wrist_xyz(3), thumb_tip_xyz(3), ...]
unnormalized_action["action.right_key_points"]: (B, horizon, 18) = [同上]
```

**处理过程**:
```python
# 初始化存储数组
retarget_left_arm_seq = np.zeros((B, horizon, 6), dtype=np.float32)
retarget_right_arm_seq = np.zeros((B, horizon, 6), dtype=np.float32)
retarget_left_hand_seq = np.zeros((B, horizon, 6), dtype=np.float32)
retarget_right_hand_seq = np.zeros((B, horizon, 6), dtype=np.float32)

# 逐帧进行retarget
for t in range(horizon):
    for b in range(batch_size):
        # 1. Reshape: (18,) → (6, 3)
        left_kp_t = pred_left_keypoints_seq[b, t, :].reshape(6, 3)
        right_kp_t = pred_right_keypoints_seq[b, t, :].reshape(6, 3)
        
        # 2. 调用fourier_hand_retargeter.retarget()
        #    注意：FourierHandRetargetAPI内部已经为左右手分别初始化了retargeting对象
        #    在self.retargetings字典中存储了'left'和'right'两个独立的retargeter
        result = fourier_hand_retargeter.retarget(
            left_keypoints=left_kp_t,   # (6, 3)
            right_keypoints=right_kp_t  # (6, 3)
        )
        
        # 3. 提取结果
        #    result['left']['wrist_pose']:     (6,) = [wrist_xyz(3), rotvec(3)]
        #    result['left']['finger_joints']:  (6,) = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        #                                              ↑ 数据集格式，无需重排序！
        retarget_left_arm_seq[b, t] = result['left']['wrist_pose']
        retarget_left_hand_seq[b, t] = result['left']['finger_joints']
        retarget_right_arm_seq[b, t] = result['right']['wrist_pose']
        retarget_right_hand_seq[b, t] = result['right']['finger_joints']
```

**输出格式**:
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 6) = [L_wrist_xyz(3), L_rotvec(3)]
unnormalized_action["action.left_hand"]: (B, horizon, 6) = [L_finger_q1~6]
                                                            # 数据集格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
unnormalized_action["action.right_arm"]: (B, horizon, 6) = [R_wrist_xyz(3), R_rotvec(3)]
unnormalized_action["action.right_hand"]:(B, horizon, 6) = [R_finger_q1~6]

# 移除keypoints数据
# unnormalized_action中不再有 action.left_key_points, action.right_key_points
```

**日志输出**:
```
[Policy Step3] Input keypoints shape: left=(B, H, 18), right=(B, H, 18)
[Policy Step3] Converted keypoints to joint angles:
  left_arm (wrist pose): (B=B, H=H, 6)
  left_hand (finger joints): (B=B, H=H, 6)
  right_arm (wrist pose): (B=B, H=H, 6)
  right_hand (finger joints): (B=B, H=H, 6)
```

---

### Step 4: use_eepose的IK - wrist pose → arm joint angles
**代码位置**: `policy.py` L747-825

**输入格式**（从Step 3输出）:
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 6) = [L_wrist_xyz(3), L_rotvec(3)]
unnormalized_action["action.right_arm"]: (B, horizon, 6) = [R_wrist_xyz(3), R_rotvec(3)]
unnormalized_action["action.left_hand"]: (B, horizon, 6) = [保持不变]
unnormalized_action["action.right_hand"]:(B, horizon, 6) = [保持不变]
```

**处理过程**:
```python
# 初始化存储数组
q_left_arm_seq = np.zeros((B, horizon, 7))  # 7-DoF arm
q_right_arm_seq = np.zeros((B, horizon, 7))

# 使用当前arm state作为IK初始猜测
q_init_left = left_arm_state[:, -1, :]  # (B, 7)
q_init_right = right_arm_state[:, -1, :] # (B, 7)

# 逐帧进行IK
for t in range(horizon):
    # 提取wrist pose
    left_eepose_t = pred_left_eepose_seq[:, t, :]   # (B, 6)
    right_eepose_t = pred_right_eepose_seq[:, t, :] # (B, 6)
    
    # 分解为位置和轴角
    left_hand_pos = left_eepose_t[:, :3]
    left_hand_axisangle = left_eepose_t[:, 3:6]
    right_hand_pos = right_eepose_t[:, :3]
    right_hand_axisangle = right_eepose_t[:, 3:6]
    
    # 执行IK
    q_left_arm_t, q_right_arm_t = body_retargeter.inverse_kinematics_from_camera_axisangle(
        left_hand_pos=left_hand_pos,
        left_hand_axisangle=left_hand_axisangle,
        right_hand_pos=right_hand_pos,
        right_hand_axisangle=right_hand_axisangle,
        current_action_vector=full_action_vector,
        q_init_left=q_init_left,
        q_init_right=q_init_right
    )
    
    # 存储结果
    q_left_arm_seq[:, t, :] = q_left_arm_t
    q_right_arm_seq[:, t, :] = q_right_arm_t
    
    # 使用当前结果作为下一步的初始猜测（保证连续性）
    q_init_left = q_left_arm_t
    q_init_right = q_right_arm_t
```

**输出格式**（最终输出到client/仿真环境）:
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 7) = [7个arm关节角度]
unnormalized_action["action.right_arm"]: (B, horizon, 7) = [7个arm关节角度]
unnormalized_action["action.left_hand"]: (B, horizon, 6) = [6个finger关节角度]
unnormalized_action["action.right_hand"]:(B, horizon, 6) = [6个finger关节角度]
unnormalized_action["action.waist"]:     (B, horizon, 3) = [3个waist关节角度]
```

**日志输出**:
```
(use_eepose的标准IK处理，无特殊日志)
```

---

## 关键点总结

### 1. 左右手Retarget实例化
`FourierHandRetargetAPI`在初始化时（`__init__`方法）已经为左右手分别创建了独立的retargeting对象：
```python
# 在fourier_hand_retarget_api.py的__init__中
for side in ["left", "right"]:
    # 为每个side创建独立的SeqRetargeting对象
    self.retargetings[side] = SeqRetargeting(config, ...)
```

所以调用`retarget()`方法时，左右手是独立处理的，不会互相干扰。

### 2. Finger joints顺序
在整个流程中，finger joints的顺序保持数据集格式不变：
```
[pinky, ring, middle, index, thumb_pitch, thumb_yaw]
```

唯一的转换发生在Step 1的FK输入准备阶段（重排序为FK期望格式），但Step 3的Retarget输出直接就是数据集格式，无需重排序。

### 3. 数据维度变化
```
原始: arm(7) + hand(6) → Step 0 → wrist_pose(6) + hand(6) 
                       → Step 1 → keypoints(18)
                       → Step 2 → keypoints(18) [预测]
                       → Step 3 → wrist_pose(6) + hand(6)
                       → Step 4 → arm(7) + hand(6)
```

### 4. 警告信息
如果看到`Warning: action.left_hand dim=6, expected 18. Skipping retarget.`，这是旧日志残留，因为旧的retarget逻辑已经被条件判断`not self.use_eepose`屏蔽了。重新启动server后不会再出现。

---

## 测试验证

### 运行命令
```bash
# Server端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_server.sh

# Client端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_client.sh
```

### 预期日志
```
✓ Initialized Fourier Hand FK for Step 1 (input processing).
✓ Initialized Fourier Hand Retargeter for Step 3 (output processing).
...
[Policy Step1] Input shapes: left_arm=(1, 1, 6), left_hand=(1, 1, 6)
[Policy Step1] Converted to keypoints:
  left_key_points: (1, 1, 18)
  right_key_points: (1, 1, 18)
[Noise Debug] Using FIXED generator with seed: 0
[Policy Step3] Input keypoints shape: left=(1, 16, 18), right=(1, 16, 18)
[Policy Step3] Converted keypoints to joint angles:
  left_arm (wrist pose): (B=1, H=16, 6)
  left_hand (finger joints): (B=1, H=16, 6)
  right_arm (wrist pose): (B=1, H=16, 6)
  right_hand (finger joints): (B=1, H=16, 6)
```

**无警告信息！**

---

最后更新: 2026-01-09
