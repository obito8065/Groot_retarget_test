# 关键点日志记录功能说明

## 功能描述

在policy推理时自动保存模型预测的手部关键点坐标，用于调试和分析预测质量。

## 保存位置

```
/vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_YYYYMMDD_HHMMSS.txt
```

文件名包含时间戳，每次启动server会创建新文件。

## 文件格式

### 文件头
```
# 模型预测的手部关键点坐标
# 格式：frame_id t L_wrist_x L_wrist_y L_wrist_z L_thumb_tip_x L_thumb_tip_y L_thumb_tip_z ...
# 左手6个关键点: 手腕(L_hand_base_link) + 5个指尖(thumb/index/middle/ring/pinky)
# 右手6个关键点: 手腕(R_hand_base_link) + 5个指尖(thumb/index/middle/ring/pinky)
# batch=0的数据，每个时间步(t=0~15)记录一行
#
```

### 数据格式
每行格式：
```
frame_id t coord_1 coord_2 ... coord_36
```

- `frame_id`: 推理帧编号（从0开始递增，每次get_action调用后+1）
- `t`: 时间步索引（0~15，共16步action horizon）
- `coord_1~coord_36`: 36个坐标值（左手18维 + 右手18维）

### 坐标顺序（共36维）

#### 左手18维 (coord_1 ~ coord_18)
1. **L_wrist** (0:3): 手腕坐标 [x, y, z]
2. **L_thumb_tip** (3:6): 拇指指尖 [x, y, z]
3. **L_index_tip** (6:9): 食指指尖 [x, y, z]
4. **L_middle_tip** (9:12): 中指指尖 [x, y, z]
5. **L_ring_tip** (12:15): 无名指指尖 [x, y, z]
6. **L_pinky_tip** (15:18): 小指指尖 [x, y, z]

#### 右手18维 (coord_19 ~ coord_36)
7. **R_wrist** (18:21): 手腕坐标 [x, y, z]
8. **R_thumb_tip** (21:24): 拇指指尖 [x, y, z]
9. **R_index_tip** (24:27): 食指指尖 [x, y, z]
10. **R_middle_tip** (27:30): 中指指尖 [x, y, z]
11. **R_ring_tip** (30:33): 无名指指尖 [x, y, z]
12. **R_pinky_tip** (33:36): 小指指尖 [x, y, z]

### 示例数据
```
0 0 -0.288640 0.172033 0.330034 -0.245769 0.068082 0.255608 -0.273520 0.034938 0.415084 -0.278873 0.041133 0.437147 -0.279759 0.060547 0.448819 -0.280690 0.080731 0.458463 -0.288640 0.172033 0.330034 -0.245769 0.068082 0.255608 -0.273520 0.034938 0.415084 -0.278873 0.041133 0.437147 -0.279759 0.060547 0.448819 -0.280690 0.080731 0.458463
0 1 -0.289123 0.173456 0.331245 -0.246891 0.069234 0.256789 -0.274632 0.036012 0.416234 -0.279987 0.042267 0.438301 -0.280873 0.061689 0.449973 -0.281804 0.081865 0.459617 -0.289123 0.173456 0.331245 -0.246891 0.069234 0.256789 -0.274632 0.036012 0.416234 -0.279987 0.042267 0.438301 -0.280873 0.061689 0.449973 -0.281804 0.081865 0.459617
...
```

## 数据说明

### 关键点来源
- 这些关键点是**模型直接预测**的输出（Step 2的结果）
- 数据位于**相机坐标系**中
- 尚未经过retarget转换（Step 3之前保存）

### 批次处理
- 只保存`batch=0`的数据（通常是第一个环境）
- 如果有多个并行环境，其他环境的数据不保存

### 时间步
- 每次`get_action()`调用会产生16行数据（action_horizon=16）
- `frame_id`记录是第几次调用
- `t`记录在当前调用中的时间步索引

## 使用方法

### 1. 启动server
```bash
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_server.sh
```

### 2. 运行client评测
```bash
bash sft_notebook_robocasa_1task_1000ep_train/eval_norm/nopretrain/1_eval_retarget_robocasa_client.sh
```

### 3. 查看日志文件
```bash
# 查看最新的日志文件
ls -lht /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_*.txt | head -1

# 查看文件内容
cat /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260109_XXXXXX.txt
```

## 日志信息

启动时会显示：
```
[Keypoints Logger] 创建日志文件: /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260109_XXXXXX.txt
```

每10帧显示进度：
```
[Keypoints Logger] 已保存 10 帧数据
[Keypoints Logger] 已保存 20 帧数据
...
```

## 数据分析示例

### Python读取和分析
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = []
with open('predicted_keypoints_20260109_XXXXXX.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        values = [float(x) for x in line.strip().split()]
        data.append(values)

data = np.array(data)  # shape: (N, 38) = (N, frame_id + t + 36coords)

# 提取坐标
frame_ids = data[:, 0]
time_steps = data[:, 1]
keypoints = data[:, 2:]  # (N, 36)

# 分离左右手
left_kp = keypoints[:, :18]   # (N, 18)
right_kp = keypoints[:, 18:]  # (N, 18)

# 提取特定关键点（如左手手腕）
left_wrist = left_kp[:, 0:3]  # (N, 3)

# 可视化轨迹
plt.figure(figsize=(12, 4))
plt.plot(left_wrist[:, 0], label='X')
plt.plot(left_wrist[:, 1], label='Y')
plt.plot(left_wrist[:, 2], label='Z')
plt.xlabel('Frame')
plt.ylabel('Position (m)')
plt.title('Left Wrist Trajectory')
plt.legend()
plt.grid(True)
plt.show()
```

### 检查异常值
```python
# 检查是否有NaN或Inf
if np.any(np.isnan(keypoints)):
    print("警告：存在NaN值！")
    
if np.any(np.isinf(keypoints)):
    print("警告：存在Inf值！")

# 检查坐标范围（假设相机坐标系合理范围）
print(f"X范围: [{keypoints[:, 0::3].min():.3f}, {keypoints[:, 0::3].max():.3f}]")
print(f"Y范围: [{keypoints[:, 1::3].min():.3f}, {keypoints[:, 1::3].max():.3f}]")
print(f"Z范围: [{keypoints[:, 2::3].min():.3f}, {keypoints[:, 2::3].max():.3f}]")
```

## 注意事项

1. **文件大小**
   - 每帧16行，每行约200字节
   - 运行100次get_action约产生320KB数据
   - 长时间运行请注意磁盘空间

2. **性能影响**
   - 文件写入操作可能略微影响推理速度
   - 建议调试完成后关闭此功能

3. **坐标系**
   - 关键点在**相机坐标系**中
   - 需要与你的数据集坐标系一致
   - 检查时注意坐标系定义

4. **关闭日志**
   如需关闭，注释掉policy.py中的相关代码段（L619-657）

## 故障排查

### 文件未创建
- 检查目录权限：`ls -ld /vla/users/lijiayi/code/groot_retarget/output_video_record`
- 查看错误日志：`[Keypoints Logger] 保存失败: ...`

### 数据异常
1. 检查模型输出是否正常
2. 验证数据范围是否合理
3. 对比训练数据的统计信息

### 无进度输出
- 确认`use_eepose=True`且`use_fourier_hand_retarget=True`
- 确认`data_config=robocasa_retarget`

---

最后更新: 2026-01-09
