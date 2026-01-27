# 关键点重投影验证工具

## 功能说明

`eval_retarget_reprojector.py` 脚本用于将模型推理时保存的手部关键点坐标重新投影到原始视频上，帮助验证模型输出是否正确。

## 使用方法

### 1. 基本使用

直接运行脚本（使用默认配置）：

```bash
cd /vla/users/lijiayi/code/groot_retarget
python eval_retarget_reprojector.py
```

### 2. 自定义配置

如果需要处理不同的视频和关键点文件，可以修改脚本开头的配置参数：

```python
# 输入文件路径
VIDEO_PATH = Path("你的视频路径.mp4")
KEYPOINTS_PATH = Path("你的关键点文件.txt")

# 输出视频fps（可调，建议5-30之间）
OUTPUT_FPS = 5  # 输出视频的帧率
```

### 3. 输出结果

脚本会在原视频所在目录生成 `*_reprojected.mp4` 文件，其中：
- **绿色**：左手关键点和连线
- **红色**：右手关键点和连线
- **青色**：手腕位置（特殊标记）
- **白色文字**：关键点标签和帧信息

## 关键点说明

每只手有6个关键点：
1. **wrist** - 手腕（基准点）
2. **thumb** - 拇指尖
3. **index** - 食指尖
4. **middle** - 中指尖
5. **ring** - 无名指尖
6. **pinky** - 小指尖

## 配置参数说明

### 相机内参（来自GR1RetargetConfig）
```python
CAMERA_INTRINSICS = {
    'fx': 502.8689,  # x方向焦距
    'fy': 502.8689,  # y方向焦距
    'cx': 640.0,     # 主点x坐标
    'cy': 400.0      # 主点y坐标
}
```

### 可视化参数
```python
KEYPOINT_RADIUS = 5      # 关键点圆圈半径（像素）
LINE_THICKNESS = 2       # 连线粗细（像素）
OUTPUT_FPS = 5          # 输出视频帧率（可动态调整）
```

## 数据格式

### 关键点文件格式 (predicted_keypoints_*.txt)

每行格式：
```
frame_id t L_wrist_xyz L_thumb_xyz L_index_xyz L_middle_xyz L_ring_xyz L_pinky_xyz R_wrist_xyz R_thumb_xyz R_index_xyz R_middle_xyz R_ring_xyz R_pinky_xyz
```

- `frame_id`: chunk序号（0, 1, 2, ...）
- `t`: chunk内的时间步序号（0-15）
- `L_*_xyz`: 左手关键点的3D坐标（相机坐标系）
- `R_*_xyz`: 右手关键点的3D坐标（相机坐标系）

总共：2 + 36 = 38个数值（2个索引 + 12个关键点×3）

## 工作流程

1. **解析关键点数据** - 从txt文件读取所有关键点坐标
2. **打开输入视频** - 自动检测视频的fps和分辨率
3. **处理每一帧**：
   - 根据设定的OUTPUT_FPS进行帧采样
   - 将3D关键点投影到2D图像平面
   - 绘制关键点、连线和标签
   - 写入输出视频

## 示例输出

```
================================================================================
Keypoints Reprojection Visualizer
================================================================================

输入视频: .../d77fd305-7431-41e5-bcca-bb1baa24c22b_success0.mp4
关键点文件: .../predicted_keypoints_20260112_110653.txt
输出视频: .../d77fd305-7431-41e5-bcca-bb1baa24c22b_success0_reprojected.mp4
输出FPS: 5

[1/3] 解析关键点数据...
✓ 加载了 64 个时间步的关键点数据
✓ Chunk范围: 0 ~ 3 (共4个chunk)

[2/3] 打开输入视频...
✓ 视频信息: 1280x800, 10.00 fps, 32 帧
✓ 帧采样: 每隔 2 帧取1帧 (从 10.00fps 降至 5fps)

[3/3] 处理视频并重投影关键点...
处理进度: 100%|██████████| 32/32 [00:00<00:00, 143.88it/s]

✓ 处理完成!
  输入帧数: 32 (10.00 fps)
  输出帧数: 16 (5.00 fps)
  输出文件: .../d77fd305-7431-41e5-bcca-bb1baa24c22b_success0_reprojected.mp4
```

## 注意事项

1. **fps自动适配**：脚本会自动检测输入视频的实际fps，并根据OUTPUT_FPS设置进行采样
2. **坐标系统**：关键点坐标应该在相机坐标系中（z轴指向前方）
3. **深度阈值**：只有z>0.01的点（在相机前方）才会被投影和显示
4. **chunk对应关系**：
   - 每个chunk有16个时间步（t=0~15）
   - 视频帧与chunk/timestep的对应关系按顺序匹配
   - chunk_0的16步对应前16个采样帧，以此类推

## 故障排除

### 问题1：关键点没有显示
- 检查关键点坐标是否在相机坐标系中（不是世界坐标系）
- 检查z坐标是否为正值
- 调整相机内参是否正确

### 问题2：关键点位置偏移
- 确认相机内参（fx, fy, cx, cy）是否与实际相机匹配
- 检查关键点数据的坐标系定义

### 问题3：帧数不匹配
- 检查chunk数量和timestep数量是否与视频帧数对应
- 调整OUTPUT_FPS以改变采样率

## 技术细节

### 3D到2D投影公式

```python
u = fx * (x/z) + cx
v = fy * (y/z) + cy
```

其中：
- (x, y, z) 是相机坐标系中的3D点
- (u, v) 是图像坐标系中的2D像素坐标
- (fx, fy) 是焦距
- (cx, cy) 是主点坐标
