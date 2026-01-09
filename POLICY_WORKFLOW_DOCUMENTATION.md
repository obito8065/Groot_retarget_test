# Policy工作流程详细文档

## 概述
当同时使用`--use_eepose`和`--use_fourier_hand_retarget`时，policy会执行以下4步流程：

```
原始joint angles → [Step 0] → EEPose → [Step 1] → Keypoints → [Step 2] → 预测Keypoints → [Step 3] → EEPose → [Step 4] → 最终joint angles
```

---

## 详细流程

### Step 0: use_eepose的FK处理（代码位置: policy.py L358-394）

**输入格式**（从仿真环境获取）：
```python
obs_copy["state.left_arm"]:  (B, T, 7)  # 左臂7个关节角度
obs_copy["state.left_hand"]: (B, T, 6)  # 左手6个手指关节角度
                                        # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
obs_copy["state.right_arm"]:  (B, T, 7)  # 右臂7个关节角度
obs_copy["state.right_hand"]: (B, T, 6)  # 右手6个手指关节角度
obs_copy["state.waist"]:      (B, T, 3)  # 腰部3个关节角度
```

**处理过程**：
- 使用`body_retargeter.process_frame_kinematics_axisangle()`进行正向运动学
- 将arm的7个关节角度转换为wrist的6维pose（xyz位置 + 轴角姿态）

**输出格式**：
```python
obs_copy["state.left_arm"]:  (B, T, 6)  # [wrist_xyz(3), rotvec(3)]
obs_copy["state.right_arm"]: (B, T, 6)  # [wrist_xyz(3), rotvec(3)]
obs_copy["state.left_hand"]: (B, T, 6)  # 保持不变
obs_copy["state.right_hand"]:(B, T, 6)  # 保持不变
```

---

### Step 1: FK转换 - wrist pose + finger joints → hand keypoints（代码位置: policy.py L396-478）

**输入格式**（从Step 0输出）：
```python
obs_copy["state.left_arm"]:  (B, T, 6)  # L_wrist_xyz(3) + L_rotvec(3)
obs_copy["state.left_hand"]: (B, T, 6)  # L_finger_q1~6
                                        # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
obs_copy["state.right_arm"]: (B, T, 6)  # R_wrist_xyz(3) + R_rotvec(3)
obs_copy["state.right_hand"]:(B, T, 6)  # R_finger_q1~6
```

**处理过程**：
1. 重排序finger joints: 数据集格式 → FK期望格式
   - 数据集: `[pinky(0), ring(1), middle(2), index(3), thumb_pitch(4), thumb_yaw(5)]`
   - FK期望: `[index(0), middle(1), ring(2), pinky(3), thumb_yaw(4), thumb_pitch(5)]`
   - 映射: `[3, 2, 1, 0, 5, 4]`

2. 调用`policy_fourier_hand_keypoints.compute_keypoints()`执行FK
   - 基于wrist pose和finger joints计算6个关键点的3D位置

**输出格式**：
```python
obs_copy["state.left_key_points"]:  (B, T, 18)  # [wrist_xyz, thumb_tip_xyz, index_tip_xyz, 
                                                 #  middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz]
obs_copy["state.right_key_points"]: (B, T, 18)  # 同上
obs_copy["state.waist"]:            (B, T, 3)   # 保持不变

# 移除原始的arm和hand数据
# obs_copy中不再有 state.left_arm, state.left_hand, state.right_arm, state.right_hand
```

**关键点顺序说明**：
- 第0个点 (0:3):   wrist_xyz
- 第1个点 (3:6):   thumb_tip_xyz
- 第2个点 (6:9):   index_tip_xyz
- 第3个点 (9:12):  middle_tip_xyz
- 第4个点 (12:15): ring_tip_xyz
- 第5个点 (15:18): pinky_tip_xyz

---

### Step 2: 模型推理（代码位置: policy.py L535-558）

**输入格式**（从Step 1输出）：
```python
obs_copy["state.left_key_points"]:  (B, T, 18)  # 左手6个关键点
obs_copy["state.right_key_points"]: (B, T, 18)  # 右手6个关键点
obs_copy["state.waist"]:            (B, T, 3)   # 腰部关节
obs_copy["video.xxx"]:              (B, T, H, W, C)  # 视频数据
obs_copy["annotation.xxx"]:         ...         # 任务描述等
```

**处理过程**：
1. `apply_transforms()`: 数据归一化等预处理
2. `_get_action_from_normalized_input()`: 模型推理
3. `_get_unnormalized_action()`: 反归一化

**输出格式**：
```python
unnormalized_action["action.left_key_points"]:  (B, horizon, 18)  # 预测的左手6个关键点
unnormalized_action["action.right_key_points"]: (B, horizon, 18)  # 预测的右手6个关键点
unnormalized_action["action.waist"]:            (B, horizon, 3)   # 预测的腰部关节
```

**注意**：
- B: batch size（通常为环境数量）
- T: 历史时间步（输入）
- horizon: 预测时间步（输出，通常为16）

---

### Step 3: Retarget转换 - hand keypoints → wrist pose + finger joints（代码位置: policy.py L586-687）

**输入格式**（从Step 2输出）：
```python
unnormalized_action["action.left_key_points"]:  (B, horizon, 18)  # 预测的左手关键点
unnormalized_action["action.right_key_points"]: (B, horizon, 18)  # 预测的右手关键点
```

**处理过程**：
1. 逐帧处理每个时间步
2. 对每一帧调用`fourier_hand_retargeter.retarget()`
   - 输入: (6, 3)的关键点数组
   - 输出: 
     - `wrist_pose`: (6,) = [wrist_xyz(3), rotvec(3)]
     - `finger_joints`: (6,) = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]

**输出格式**：
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 6)  # L_wrist_xyz(3) + L_rotvec(3)
unnormalized_action["action.left_hand"]: (B, horizon, 6)  # L_finger_q1~6
                                                          # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
unnormalized_action["action.right_arm"]: (B, horizon, 6)  # R_wrist_xyz(3) + R_rotvec(3)
unnormalized_action["action.right_hand"]:(B, horizon, 6)  # R_finger_q1~6

# 移除keypoints数据
# unnormalized_action中不再有 action.left_key_points, action.right_key_points
```

**finger joints顺序说明**：
```
数据集格式 = Retarget输出格式 = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
```
- 索引0: pinky_proximal_joint
- 索引1: ring_proximal_joint
- 索引2: middle_proximal_joint
- 索引3: index_proximal_joint
- 索引4: thumb_proximal_pitch_joint
- 索引5: thumb_proximal_yaw_joint

---

### Step 4: use_eepose的IK处理 - wrist pose → arm joint angles（代码位置: policy.py L744-825）

**输入格式**（从Step 3输出）：
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 6)  # L_wrist_xyz(3) + L_rotvec(3)
unnormalized_action["action.right_arm"]: (B, horizon, 6)  # R_wrist_xyz(3) + R_rotvec(3)
unnormalized_action["action.left_hand"]: (B, horizon, 6)  # 保持不变
unnormalized_action["action.right_hand"]:(B, horizon, 6)  # 保持不变
```

**处理过程**：
1. 逐帧处理每个时间步
2. 提取wrist的位置和轴角姿态
3. 调用`body_retargeter.inverse_kinematics_from_camera_axisangle()`执行IK
   - 使用前一帧的结果作为初始猜测，保证动作连续性

**输出格式**（最终输出到客户端）：
```python
unnormalized_action["action.left_arm"]:  (B, horizon, 7)  # 左臂7个关节角度
unnormalized_action["action.right_arm"]: (B, horizon, 7)  # 右臂7个关节角度
unnormalized_action["action.left_hand"]: (B, horizon, 6)  # 左手6个手指关节角度
unnormalized_action["action.right_hand"]:(B, horizon, 6)  # 右手6个手指关节角度
unnormalized_action["action.waist"]:     (B, horizon, 3)  # 腰部3个关节角度
```

---

## 完整数据流示意图

```
仿真环境输入:
├── state.left_arm:  (B,T,7) [joint angles]
├── state.left_hand: (B,T,6) [finger joints: pinky,ring,middle,index,thumb_pitch,thumb_yaw]
├── state.right_arm:  (B,T,7)
├── state.right_hand: (B,T,6)
└── state.waist: (B,T,3)

↓ [Step 0: body_retargeter FK]

├── state.left_arm:  (B,T,6) [wrist_xyz, rotvec]
├── state.left_hand: (B,T,6) [finger joints]
├── state.right_arm:  (B,T,6)
├── state.right_hand: (B,T,6)
└── state.waist: (B,T,3)

↓ [Step 1: fourier_hand FK]

├── state.left_key_points:  (B,T,18) [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
├── state.right_key_points: (B,T,18)
└── state.waist: (B,T,3)

↓ [Step 2: Model Inference]

├── action.left_key_points:  (B,H,18) [预测的关键点]
├── action.right_key_points: (B,H,18)
└── action.waist: (B,H,3)

↓ [Step 3: fourier_hand Retarget]

├── action.left_arm:  (B,H,6) [wrist_xyz, rotvec]
├── action.left_hand: (B,H,6) [finger joints]
├── action.right_arm:  (B,H,6)
├── action.right_hand: (B,H,6)
└── action.waist: (B,H,3)

↓ [Step 4: body_retargeter IK]

最终输出到仿真环境:
├── action.left_arm:  (B,H,7) [joint angles]
├── action.left_hand: (B,H,6) [finger joints]
├── action.right_arm:  (B,H,7)
├── action.right_hand: (B,H,6)
└── action.waist: (B,H,3)
```

---

## 关键点说明

### 数据格式转换
- **Step 0 & 4**: 7维arm joint angles ↔ 6维wrist pose (xyz + rotvec)
- **Step 1 & 3**: 6维wrist pose + 6维finger joints ↔ 18维hand keypoints

### Finger joints顺序
在整个流程中保持数据集格式不变：
```
[pinky, ring, middle, index, thumb_pitch, thumb_yaw]
```

### 模型训练数据格式
训练数据使用的就是keypoints格式：
- state: left_key_points (18维) + right_key_points (18维) + waist (3维)
- action: left_key_points (18维) + right_key_points (18维) + waist (3维)

---

## 测试与验证

### 运行命令
```bash
# Server端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_server.sh

# Client端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_client.sh
```

### 关键参数
```bash
--use_eepose                    # 启用arm的FK/IK转换
--use_fourier_hand_retarget     # 启用hand的keypoints转换
--data_config robocasa_retarget # 使用keypoints格式的数据配置
```

### 调试日志
代码中添加了详细的打印信息，用于验证各步骤的数据形状：
- `[Policy Step1]`: 输入处理阶段
- `[Policy Step3]`: 输出处理阶段
- 各步骤都会打印输入/输出的shape信息

---

## 常见问题

### Q1: 为什么需要Step 0和Step 4？
A: Step 0将原始的7维arm joint angles转换为6维wrist pose，便于后续的keypoints计算。Step 4将6维wrist pose转换回7维joint angles，供仿真环境执行。

### Q2: keypoints和eepose的区别？
A: 
- **EEpose**: 6维 (wrist的xyz位置 + 轴角姿态)
- **Keypoints**: 18维 (wrist + 5个finger tips的xyz位置)

### Q3: 为什么模型使用keypoints而不是joint angles？
A: Keypoints是空间位置表示，更接近视觉观测，便于模型学习手部的空间关系。

### Q4: finger joints的顺序为什么要转换？
A: 数据集格式与FK模块期望的格式不同，需要在Step 1中转换（但Step 3的输出保持数据集格式）。

---

## 相关文件

- `gr00t/model/policy.py`: 主要实现代码
- `gr00t/eval/gr1_hand_fk.py`: Fourier Hand FK模块
- `gr00t/eval/fourier_hand_retarget_api.py`: Fourier Hand Retarget模块
- `gr00t/eval/gr1_pos_transform.py`: Body Retargeter (arm FK/IK)
- `gr00t/eval/QUICK_REFERENCE_JOINT_ORDER.md`: 关节顺序参考
- `gr00t/eval/FOURIER_HAND_INTEGRATION_GUIDE.md`: Fourier Hand集成指南

---

最后更新: 2026-01-09
