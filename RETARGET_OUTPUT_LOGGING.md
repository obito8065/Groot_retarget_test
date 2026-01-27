# Retarget输出日志记录功能说明

## 功能描述

在Step 3 retarget转换之后，自动保存每个chunk中每个时间步的输出数据（wrist pose + finger joints），用于调试和验证retarget效果。

## 保存位置

```
/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_YYYYMMDD_HHMMSS.txt
```

文件名包含时间戳，每次启动server会创建新文件。

## 保存时机

- **Step 3结束后**：在retarget转换完成，将keypoints转换为wrist pose + finger joints之后
- **Step 4开始前**：在IK将6维wrist pose转换为7维arm joint angles之前

这样保存的是**6维wrist pose + 6维finger joints**的纯净retarget输出。

## 文件格式

### 文件头
```
# Retarget后的动作输出（每个chunk的每个时间步）
# 格式：chunk_id t L_wrist_x L_wrist_y L_wrist_z L_rotvec_x L_rotvec_y L_rotvec_z L_finger_q1 L_finger_q2 L_finger_q3 L_finger_q4 L_finger_q5 L_finger_q6 R_wrist_x R_wrist_y R_wrist_z R_rotvec_x R_rotvec_y R_rotvec_z R_finger_q1 R_finger_q2 R_finger_q3 R_finger_q4 R_finger_q5 R_finger_q6
# L_finger_joint_names_6: L_index_proximal_joint L_middle_proximal_joint L_ring_proximal_joint L_pinky_proximal_joint L_thumb_proximal_yaw_joint L_thumb_proximal_pitch_joint
# R_finger_joint_names_6: R_index_proximal_joint R_middle_proximal_joint R_ring_proximal_joint R_pinky_proximal_joint R_thumb_proximal_yaw_joint R_thumb_proximal_pitch_joint
# 注意：finger joints已从retarget输出格式 [pinky, ring, middle, index, thumb_pitch, thumb_yaw] 重排序为目标格式 [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
# batch=0的数据，每个chunk的16个时间步(t=0~15)各记录一行
#
```

### 数据格式
每行格式：
```
chunk_id t coord_1 coord_2 ... coord_24
```

- `chunk_id`: Chunk编号（从0开始递增，每次`get_action()`调用产生一个chunk）
- `t`: 当前chunk内的时间步索引（0~15，共16步action horizon）
- `coord_1~coord_24`: 24个数值（左手12维 + 右手12维）

### 坐标顺序（共24维）

#### 左手12维 (1-12)
1. **L_wrist_x** (1): 手腕X坐标（相机坐标系）
2. **L_wrist_y** (2): 手腕Y坐标
3. **L_wrist_z** (3): 手腕Z坐标
4. **L_rotvec_x** (4): 手腕旋转轴角X分量
5. **L_rotvec_y** (5): 手腕旋转轴角Y分量
6. **L_rotvec_z** (6): 手腕旋转轴角Z分量
7. **L_finger_q1** (7): L_index_proximal_joint（食指关节）
8. **L_finger_q2** (8): L_middle_proximal_joint（中指关节）
9. **L_finger_q3** (9): L_ring_proximal_joint（无名指关节）
10. **L_finger_q4** (10): L_pinky_proximal_joint（小指关节）
11. **L_finger_q5** (11): L_thumb_proximal_yaw_joint（拇指yaw关节）
12. **L_finger_q6** (12): L_thumb_proximal_pitch_joint（拇指pitch关节）

#### 右手12维 (13-24)
13-24: 与左手相同结构

### 示例数据
```
0 0 -0.288640 0.172033 0.330034 0.012345 -0.034567 0.089012 0.234567 0.456789 0.678901 0.890123 1.234567 0.876543 -0.288640 0.172033 0.330034 0.012345 -0.034567 0.089012 0.234567 0.456789 0.678901 0.890123 1.234567 0.876543
0 1 -0.289123 0.173456 0.331245 0.013456 -0.035678 0.090123 0.235678 0.457890 0.679012 0.891234 1.235678 0.877654 -0.289123 0.173456 0.331245 0.013456 -0.035678 0.090123 0.235678 0.457890 0.679012 0.891234 1.235678 0.877654
...
0 15 -0.290234 0.174567 0.332356 0.014567 -0.036789 0.091234 0.236789 0.458901 0.680123 0.892345 1.236789 0.878765 -0.290234 0.174567 0.332356 0.014567 -0.036789 0.091234 0.236789 0.458901 0.680123 0.892345 1.236789 0.878765
1 0 -0.291345 0.175678 0.333467 0.015678 -0.037890 0.092345 0.237890 0.459012 0.681234 0.893456 1.237890 0.879876 -0.291345 0.175678 0.333467 0.015678 -0.037890 0.092345 0.237890 0.459012 0.681234 0.893456 1.237890 0.879876
...
```

说明：
- chunk_id=0的chunk包含16行（t=0~15）
- chunk_id=1的chunk包含16行（t=0~15）
- 依此类推

## 重要说明

### 1. Finger Joints顺序转换
**Retarget模块输出顺序**（内部格式）:
```
[pinky, ring, middle, index, thumb_pitch, thumb_yaw]
索引: [0,    1,    2,      3,     4,            5]
```

**保存时的目标顺序**（你要求的格式）:
```
[index, middle, ring, pinky, thumb_yaw, thumb_pitch]
索引: [0,     1,      2,    3,     4,         5]
```

**转换映射**: `[3, 2, 1, 0, 5, 4]`

代码中已自动完成此转换，保存的数据是目标格式。

### 2. 坐标系
- **wrist pose**: 相机坐标系中的6-DoF pose
  - xyz位置单位：米（m）
  - rotvec：轴角表示（axis-angle），单位：弧度（rad）
- **finger joints**: 关节角度，单位：弧度（rad）

### 3. 数据来源
保存的数据来自：
```python
retarget_left_arm_seq[0, t, :]   # batch=0, 时间步t, 6维wrist pose
retarget_left_hand_seq[0, t, :]  # batch=0, 时间步t, 6维finger joints
```

这是**纯净的retarget输出**，尚未经过IK转换为arm joint angles。

## 使用方法

### 1. 启动server并运行评测
```bash
# Server端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_server.sh

# Client端
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_client.sh
```

### 2. 查看生成的文件
```bash
# 列出日志文件
ls -lht /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_*.txt

# 查看内容（前20行）
head -20 /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260109_XXXXXX.txt

# 统计数据行数（不包括注释行）
grep -v "^#" /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260109_XXXXXX.txt | wc -l
```

### 3. Python分析示例

#### 读取和解析数据
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = []
with open('retargeted_actions_20260109_XXXXXX.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        values = [float(x) for x in line.strip().split()]
        data.append(values)

data = np.array(data)  # shape: (N, 26) = (N, chunk_id + t + 24values)

# 提取数据
chunk_ids = data[:, 0].astype(int)
time_steps = data[:, 1].astype(int)
actions = data[:, 2:]  # (N, 24)

# 分离左右手数据
left_wrist_pose = actions[:, 0:6]      # (N, 6): [xyz(3), rotvec(3)]
left_finger_joints = actions[:, 6:12]  # (N, 6): [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
right_wrist_pose = actions[:, 12:18]   # (N, 6)
right_finger_joints = actions[:, 18:24] # (N, 6)

print(f"总数据行数: {len(data)}")
print(f"总chunk数: {chunk_ids.max() + 1}")
print(f"每个chunk的时间步数: {len(data) // (chunk_ids.max() + 1)}")
```

#### 检查finger joints范围
```python
finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb_yaw', 'thumb_pitch']

print("左手finger joints统计:")
for i, name in enumerate(finger_names):
    values = left_finger_joints[:, i]
    print(f"  {name:12s}: min={values.min():7.3f}, max={values.max():7.3f}, mean={values.mean():7.3f}, std={values.std():7.3f}")

print("\n右手finger joints统计:")
for i, name in enumerate(finger_names):
    values = right_finger_joints[:, i]
    print(f"  {name:12s}: min={values.min():7.3f}, max={values.max():7.3f}, mean={values.mean():7.3f}, std={values.std():7.3f}")
```

#### 可视化单个chunk的轨迹
```python
# 提取第0个chunk的数据
chunk_0_mask = chunk_ids == 0
chunk_0_data = actions[chunk_0_mask]
chunk_0_t = time_steps[chunk_0_mask]

# 绘制左手wrist位置轨迹
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(chunk_0_t, chunk_0_data[:, 0], 'o-', label='X')
plt.plot(chunk_0_t, chunk_0_data[:, 1], 's-', label='Y')
plt.plot(chunk_0_t, chunk_0_data[:, 2], '^-', label='Z')
plt.xlabel('Time Step')
plt.ylabel('Position (m)')
plt.title('Left Wrist Position (Chunk 0)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(chunk_0_t, chunk_0_data[:, 3], 'o-', label='rotvec_x')
plt.plot(chunk_0_t, chunk_0_data[:, 4], 's-', label='rotvec_y')
plt.plot(chunk_0_t, chunk_0_data[:, 5], '^-', label='rotvec_z')
plt.xlabel('Time Step')
plt.ylabel('Rotation (rad)')
plt.title('Left Wrist Orientation (Chunk 0)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
for i, name in enumerate(finger_names):
    plt.plot(chunk_0_t, chunk_0_data[:, 6+i], 'o-', label=name)
plt.xlabel('Time Step')
plt.ylabel('Joint Angle (rad)')
plt.title('Left Finger Joints (Chunk 0)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### 检查连续性（相邻时间步的差异）
```python
# 计算相邻时间步的差异
diff = np.diff(left_finger_joints, axis=0)

print("左手finger joints相邻时间步的最大变化量:")
for i, name in enumerate(finger_names):
    max_diff = np.abs(diff[:, i]).max()
    print(f"  {name:12s}: {max_diff:.6f} rad ({np.degrees(max_diff):.2f} deg)")

# 如果最大变化量过大，可能存在问题
threshold_deg = 30  # 度
threshold_rad = np.radians(threshold_deg)
for i, name in enumerate(finger_names):
    max_diff = np.abs(diff[:, i]).max()
    if max_diff > threshold_rad:
        print(f"⚠️  警告: {name} 的最大变化量超过{threshold_deg}度，可能存在跳变！")
```

#### 对比预测keypoints和retarget输出
```python
# 读取预测的keypoints
kp_data = np.loadtxt('predicted_keypoints_20260109_XXXXXX.txt', comments='#')
kp_left_wrist = kp_data[:, 2:5]  # (N, 3): 预测的左手腕位置

# 读取retarget的wrist pose
ret_data = np.loadtxt('retargeted_actions_20260109_XXXXXX.txt', comments='#')
ret_left_wrist = ret_data[:, 2:5]  # (N, 3): retarget的左手腕位置

# 确保两个文件的数据行数一致（每个chunk 16行）
min_len = min(len(kp_left_wrist), len(ret_left_wrist))
kp_left_wrist = kp_left_wrist[:min_len]
ret_left_wrist = ret_left_wrist[:min_len]

# 计算差异
diff = np.abs(kp_left_wrist - ret_left_wrist)
print(f"左手腕位置差异统计 (Keypoint vs Retarget):")
print(f"  平均差异: {diff.mean():.6f} m")
print(f"  最大差异: {diff.max():.6f} m")
print(f"  标准差:   {diff.std():.6f} m")

# 如果差异过大，说明retarget可能有问题
if diff.max() > 0.01:  # 1cm
    print("⚠️  警告: 手腕位置差异超过1cm，retarget可能存在问题！")
```

## 日志输出

### 启动时
```
[Retarget Logger] 创建日志文件: /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260109_175315.txt
```

### 运行时（每10个chunk）
```
[Retarget Logger] 已保存 10 个chunk数据 (共 160 个时间步)
[Retarget Logger] 已保存 20 个chunk数据 (共 320 个时间步)
...
```

## 检查清单

使用此日志文件可以检查以下问题：

### 1. ✅ Retarget输出合理性
- [ ] wrist pose的xyz位置在合理范围内（相机坐标系）
- [ ] rotvec的模在合理范围内（通常 < π ≈ 3.14）
- [ ] finger joints在机械限位内（查看URDF的joint limits）

### 2. ✅ 动作连续性
- [ ] 相邻时间步的finger joints变化量合理（通常 < 30度）
- [ ] 没有突然的跳变
- [ ] 轨迹平滑

### 3. ✅ 与预测keypoints一致性
- [ ] retarget后的wrist位置与输入keypoints的wrist位置接近
- [ ] 差异在合理范围内（通常 < 5mm）

### 4. ✅ 左右手对称性
- [ ] 左右手的数据范围相似
- [ ] 左右手的行为模式合理

## 故障排查

### 文件未创建
```bash
# 检查目录权限
ls -ld /vla/users/lijiayi/code/groot_retarget/output_video_record

# 检查是否触发了正确的条件
# - use_eepose=True
# - use_fourier_hand_retarget=True
# - data_config=robocasa_retarget
```

### 数据异常
1. **wrist pose异常**
   - 检查预测的keypoints是否正常
   - 检查retarget配置是否正确

2. **finger joints异常**
   - 检查关节角度是否超出限位
   - 检查是否有NaN或Inf值
   - 验证顺序转换是否正确

3. **连续性问题**
   - 检查模型预测是否平滑
   - 检查retarget优化是否收敛

## 关闭日志

如需关闭此功能，注释掉`policy.py`中L738-794的代码段。

---

最后更新: 2026-01-09
