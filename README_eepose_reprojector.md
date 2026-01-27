# EE Pose 重投影可视化工具

## 功能说明

`eval_after_retarget_reprojector_cli.py` 脚本用于将retarget后的手腕eepose（位置+方向）重投影到原始视频上，验证retarget输出是否正确。

## 主要特点

- **读取retargeted_actions文件**：解析每个时间步的左右手wrist pose（xyz + rotvec）
- **3D到2D投影**：将相机坐标系中的3D位置投影到图像平面
- **方向可视化**：用RGB坐标轴显示手腕的旋转方向
  - **红色箭头**：X轴方向
  - **绿色箭头**：Y轴方向  
  - **蓝色箭头**：Z轴方向
- **左右手区分**：
  - **绿色点**：左手手腕
  - **红色点**：右手手腕

## 使用方法

### 基本用法

```bash
python eval_after_retarget_reprojector_cli.py \
    --video 视频文件路径.mp4 \
    --retarget retargeted_actions文件.txt \
    --fps 5 \
    --axis-length 0.05
```

### 完整示例

```bash
python eval_after_retarget_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget/74ksteps/042eb3aa-5311-4e27-98c0-ecf0370f4ede_success0.mp4 \
    --retarget /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260112_113746.txt \
    --fps 5 \
    --axis-length 0.05 \
    --thickness 3
```

## 命令行参数

### 必需参数

- `--video`: 输入视频文件路径
- `--retarget`: retargeted_actions txt文件路径

### 可选参数

- `--output`: 输出视频路径（默认：`{输入视频}_eepose_reprojected.mp4`）
- `--fps`: 输出视频帧率（默认：5）
- `--axis-length`: 坐标轴长度，单位米（默认：0.05）
- `--thickness`: 线条粗细（默认：3）

### 相机内参（高级选项）

- `--fx`: 相机焦距fx（默认：502.8689）
- `--fy`: 相机焦距fy（默认：502.8689）
- `--cx`: 相机主点cx（默认：640.0）
- `--cy`: 相机主点cy（默认：400.0）

## 输入文件格式

### retargeted_actions文件格式

每行包含26个数值：

```
chunk_id t L_wrist_xyz(3) L_rotvec(3) L_finger_q(6) R_wrist_xyz(3) R_rotvec(3) R_finger_q(6)
```

具体说明：
- 索引0: `chunk_id` - chunk序号
- 索引1: `t` - 时间步序号（0-15）
- 索引2-4: `L_wrist_xyz` - 左手手腕位置（相机坐标系）
- 索引5-7: `L_rotvec` - 左手手腕旋转向量（axis-angle表示）
- 索引8-13: `L_finger_q` - 左手手指关节角度（本脚本不使用）
- 索引14-16: `R_wrist_xyz` - 右手手腕位置（相机坐标系）
- 索引17-19: `R_rotvec` - 右手手腕旋转向量（axis-angle表示）
- 索引20-25: `R_finger_q` - 右手手指关节角度（本脚本不使用）

## 可视化说明

### 手腕位置
- 每个手腕用一个大圆点表示
- 左手：绿色圆点
- 右手：红色圆点
- 带有标签："L_wrist" 或 "R_wrist"

### 坐标轴方向
从每个手腕位置延伸出三条彩色线段，表示手腕的局部坐标系：
- **红色线段（X轴）**：指向手腕坐标系的X轴正方向
- **绿色线段（Y轴）**：指向手腕坐标系的Y轴正方向
- **蓝色线段（Z轴）**：指向手腕坐标系的Z轴正方向

线段长度由 `--axis-length` 参数控制（单位：米）。

### 帧信息
视频左上角显示当前帧信息：
```
Chunk 0 | Step 5 | Frame 10
```

## 示例输出

```
================================================================================
EE Pose Reprojection Visualizer
================================================================================

输入视频: .../042eb3aa-5311-4e27-98c0-ecf0370f4ede_success0.mp4
Retarget文件: .../retargeted_actions_20260112_113746.txt
输出视频: .../042eb3aa-5311-4e27-98c0-ecf0370f4ede_success0_eepose_reprojected.mp4
输出FPS: 5
可视化参数: axis_length=0.05m, thickness=3

[1/3] 解析retarget数据...
✓ 加载了 368 个时间步的eepose数据
✓ Chunk范围: 0 ~ 22 (共23个chunk)

[2/3] 打开输入视频...
✓ 视频信息: 1280x800, 10.00 fps, 180 帧
✓ 帧采样: 每隔 2 帧取1帧 (从 10.00fps 降至 5fps)

[3/3] 处理视频并重投影eepose...
处理进度: 100%|██████████| 180/180 [00:01<00:00, 159.00it/s]

✓ 处理完成!
  输入帧数: 180 (10.00 fps)
  输出帧数: 90 (5.00 fps)
  输出文件: .../042eb3aa-5311-4e27-98c0-ecf0370f4ede_success0_eepose_reprojected.mp4
```

## 技术细节

### 坐标系统
- **相机坐标系**：右手坐标系，Z轴指向相机前方
- **手腕坐标系**：与机器人手腕link固连的局部坐标系
- **旋转表示**：使用axis-angle（旋转向量）表示，通过scipy转换为旋转矩阵

### 3D到2D投影
使用针孔相机模型：

```
u = fx * (x/z) + cx
v = fy * (y/z) + cy
```

其中：
- (x, y, z) 是相机坐标系中的3D点
- (u, v) 是图像坐标系中的2D像素坐标

### 坐标轴绘制
1. 在手腕坐标系中定义三个单位向量：[1,0,0], [0,1,0], [0,0,1]
2. 乘以坐标轴长度（如0.05米）
3. 通过旋转矩阵转换到相机坐标系
4. 加上手腕位置，得到坐标轴端点的3D坐标
5. 投影到2D图像平面并绘制

## 参数调优建议

### 坐标轴长度 (--axis-length)
- **太小**（<0.03）：坐标轴不明显，难以看清方向
- **合适**（0.03-0.08）：清晰显示方向，不遮挡其他元素
- **太大**（>0.1）：可能遮挡画面或延伸到画面外

建议根据场景大小调整：
- 近距离操作：0.03-0.05米
- 中等距离：0.05-0.08米
- 远距离：0.08-0.15米

### 线条粗细 (--thickness)
- **较细**（1-2）：适合高分辨率视频
- **中等**（3-4）：适合1280x800等常规分辨率
- **较粗**（5+）：适合低分辨率或需要强调的场合

## 故障排除

### 问题1：手腕位置不显示
- 检查坐标是否在相机坐标系中（不是世界坐标系）
- 确认Z坐标为正值（在相机前方）
- 检查相机内参是否正确

### 问题2：坐标轴方向不正确
- 检查rotvec是否使用正确的旋转约定（axis-angle）
- 确认旋转矩阵的坐标系定义
- 检查左右手坐标系的chirality

### 问题3：投影位置偏移
- 验证相机内参（fx, fy, cx, cy）
- 检查retargeted_actions中的坐标单位（应该是米）
- 确认相机外参（如果有）是否正确

### 问题4：帧数不匹配
- 检查chunk数量和每个chunk的timestep数量
- 调整OUTPUT_FPS以改变采样率
- 确认retargeted_actions文件的完整性

## 与关键点重投影工具的区别

| 特性 | eval_retarget_reprojector_cli.py | eval_after_retarget_reprojector_cli.py |
|------|----------------------------------|----------------------------------------|
| 输入数据 | predicted_keypoints (6个关键点) | retargeted_actions (wrist pose) |
| 可视化内容 | 手指关键点+连线 | 手腕位置+方向坐标轴 |
| 处理阶段 | Policy输出（Step 2） | Retarget输出（Step 3） |
| 方向显示 | ❌ | ✅ RGB坐标轴 |
| 用途 | 验证关键点预测准确性 | 验证retarget后的pose准确性 |
