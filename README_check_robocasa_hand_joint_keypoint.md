# RoboCasa手部关节关键点检查脚本

## 功能说明

`check_robocasa_hand_joint_keypoint.py` 脚本用于：
1. 读取txt文件中的关节角度（L_arm_q1-7, L_finger_q1-6, R_arm_q1-7, R_finger_q1-6, waist_q1-3）
2. 使用完整机器人FK计算wrist pose（相机坐标系）
3. 使用手部FK计算6个关键点（wrist + 5 tips）
4. 重投影到视频上

## 使用方法

### 基本使用

```bash
cd /vla/users/lijiayi/code/groot_retarget

python check_robocasa_hand_joint_keypoint.py \
    --action-txt /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify1/robocasa_action_20260203_095725.txt \
    --video-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify1/task2.mp4 \
    --robot-urdf /vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf \
    --left-hand-urdf /vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/fourier_hand/fourier_left_hand.urdf \
    --right-hand-urdf /vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/fourier_hand/fourier_right_hand.urdf \
    --output-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify1/task2_keypoints_reprojected.mp4 \
    --fps 5.0
```

### 参数说明

- `--action-txt`: 输入的action txt文件路径（包含关节角度数据）
- `--video-path`: 输入视频文件路径
- `--robot-urdf`: 完整机器人URDF文件路径（用于arm FK计算）
- `--left-hand-urdf`: 左手URDF文件路径（用于手部FK计算）
- `--right-hand-urdf`: 右手URDF文件路径（用于手部FK计算）
- `--output-path`: 输出视频文件路径
- `--fps`: 输出视频FPS（默认: 5.0）

## 输入文件格式

### Action TXT文件格式

每行格式：
```
chunk_id t L_arm_q1 L_arm_q2 ... L_arm_q7 L_finger_q1 ... L_finger_q6 R_arm_q1 ... R_arm_q7 R_finger_q1 ... R_finger_q6 waist_q1 waist_q2 waist_q3
```

- `chunk_id`: chunk序号（0, 1, 2, ...）
- `t`: chunk内的时间步序号（0-15）
- `L_arm_q1-7`: 左手臂7个关节角度
- `L_finger_q1-6`: 左手指6个关节角度（顺序：pinky, ring, middle, index, thumb_pitch, thumb_yaw）
- `R_arm_q1-7`: 右手臂7个关节角度
- `R_finger_q1-6`: 右手指6个关节角度（顺序：pinky, ring, middle, index, thumb_pitch, thumb_yaw）
- `waist_q1-3`: 腰部3个关节角度

## 工作流程

1. **解析action txt文件** - 读取所有时间步的关节角度
2. **初始化FK计算器** - 初始化arm FK和hand FK计算器
3. **计算关键点**：
   - 使用arm FK从关节角度计算wrist pose（相机坐标系）
   - 转换finger关节角度格式（从数据集格式转换为FK期望格式）
   - 使用hand FK计算6个关键点（wrist + 5 tips）
4. **处理视频并重投影**：
   - 读取视频帧
   - 将3D关键点投影到2D图像平面
   - 绘制关键点、连线和标签
   - 写入输出视频

## 关键点说明

每只手有6个关键点：
1. **wrist** - 手腕（基准点）
2. **thumb** - 拇指尖
3. **index** - 食指尖
4. **middle** - 中指尖
5. **ring** - 无名指尖
6. **pinky** - 小指尖

## 可视化说明

- **绿色**：左手关键点和连线
- **红色**：右手关键点和连线
- **青色**：手腕位置（特殊标记）
- **白色文字**：关键点标签和帧信息

## 注意事项

1. **坐标系**：关键点坐标在相机坐标系中（z轴指向前方）
2. **深度阈值**：只有z>0.01的点（在相机前方）才会被投影和显示
3. **chunk对应关系**：每个chunk有16个时间步（t=0~15）
4. **FPS采样**：脚本会根据输入视频FPS和输出FPS自动进行帧采样

## 输出示例

```
================================================================================
RoboCasa手部关节关键点检查脚本
================================================================================

输入action文件: .../robocasa_action_20260203_095725.txt
输入视频: .../task2.mp4
输出视频: .../task2_keypoints_reprojected.mp4
输出FPS: 5.0

[1/4] 解析action txt文件...
✓ 加载了 756 个时间步的数据

[2/4] 初始化FK计算器...
正在初始化 ArmFKComputer...
ArmFKComputer 初始化完成。
✓ FK计算器初始化完成

[3/4] 计算关键点...
计算关键点: 100%|██████████| 756/756 [00:05<00:00, 150.23it/s]
✓ 计算了 756 个时间步的关键点

[4/4] 处理视频并重投影关键点...
✓ 原始视频信息: 1280x800, 10.00 fps, 32 帧
✓ 帧采样: 每隔 2 帧取1帧 (从 10.00fps 降至 5fps)
处理进度: 100%|██████████| 32/32 [00:00<00:00, 143.88it/s]

✓ 处理完成!
  输入帧数: 32 (10.00 fps)
  输出帧数: 16 (5.00 fps)
  输出文件: .../task2_keypoints_reprojected.mp4
```
