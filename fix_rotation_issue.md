# Retarget手腕方向偏移问题分析

## 问题描述
retarget后的手腕位置正确，但方向完全偏移

## 对比分析

### 原始脚本 (hand_robot_viewer_fourier.py)

**Line 564 - Warmup输入**:
```python
warmup_qpos6d = self.multi_robot_warmup(
    joint=joint[0],  # 手腕位置 (3D点)
    wrist_quat=rotations.quaternion_from_matrix(rot),  # 从旋转矩阵转换
    hand_side=side,
)
```

**关键点**:
1. `rot` 是一个 **3x3旋转矩阵**
2. 使用 `pytransform3d.rotations.quaternion_from_matrix(rot)` 转换
3. 返回格式: `[w, x, y, z]`
4. 这个旋转矩阵已经在正确的坐标系中（数据集处理时已转换）

### 新的API (fourier_hand_retarget_api.py)

**Line 269 - Warmup输入**:
```python
wrist_quat = R.from_rotvec(wrist_rotvec).as_quat()  # [x, y, z, w]

# 转换为 [w, x, y, z]
wrist_quat_wxyz = np.array([
    wrist_quat[3],  # w
    wrist_quat[0],  # x
    wrist_quat[1],  # y
    wrist_quat[2],  # z
])
```

**关键点**:
1. `wrist_rotvec` 是 **轴角表示** (rotation vector)
2. 使用 `scipy.spatial.transform.Rotation.from_rotvec()` 转换
3. 返回格式先是 `[x, y, z, w]`，然后手动转换为 `[w, x, y, z]`

## 核心问题

### 问题1: 数据来源不同

**原始脚本的数据流**:
```
数据集MANO参数 -> hand_rot (旋转矩阵)
                    ↓
              warmup使用旋转矩阵
```

**新API的数据流**:
```
模型预测 -> wrist_rotvec (轴角)
              ↓
          转换为四元数
              ↓
          warmup使用四元数
```

### 问题2: 坐标系问题 ⚠️ **最关键的问题**

**原始脚本中**:
- `hand_rot` 来自数据集的MANO模型输出
- 这个旋转已经在 **世界坐标系/相机坐标系** 中正确定义
- 通过数据集的处理流程（包括相机位姿等），旋转已经在正确的参考系中

**新API中**:
- `wrist_rotvec` 来自模型预测
- 这个`wrist_rotvec`是在**相机坐标系**下的
- ⚠️ **但没有进行坐标系转换就直接用于warmup**
- warmup需要的应该是机器人基座坐标系下的旋转

### 问题3: 旋转矩阵 vs 轴角表示

虽然格式不同，但只要坐标系正确，两种表示可以相互转换：
- 旋转矩阵 `rot` (3x3)
- 轴角 `rotvec` (3,) - 方向是轴，长度是角度

转换关系：
```python
# 旋转矩阵 -> 轴角
rotvec = R.from_matrix(rot).as_rotvec()

# 轴角 -> 旋转矩阵
rot = R.from_rotvec(rotvec).as_matrix()

# 轴角 -> 四元数
quat = R.from_rotvec(rotvec).as_quat()  # [x,y,z,w] scipy格式
```

## 解决方案

### 方案1: 提供正确坐标系的rotvec

如果模型预测的`wrist_rotvec`已经在正确的坐标系（机器人基座坐标系）:
- ✅ 当前API应该能正常工作
- ❌ 但目前明显不是这样

### 方案2: 添加坐标系转换 (推荐)

在使用`wrist_rotvec`之前，需要从相机坐标系转换到机器人基座坐标系：

```python
# 在 fourier_hand_retarget_api.py 的 retarget_from_45d 函数中修改

# 添加相机到基座的旋转矩阵（需要从配置或外参中获取）
# 假设已有 R_base_camera (3x3 旋转矩阵)

# 将wrist_rotvec从相机系转换到基座系
R_camera_wrist = R.from_rotvec(wrist_rotvec)  # 相机系的手腕旋转
R_base_wrist = R_base_camera @ R_camera_wrist.as_matrix()  # 转到基座系
wrist_rotvec_base = R.from_matrix(R_base_wrist).as_rotvec()  # 转回rotvec

# 然后使用转换后的rotvec
wrist_quat = R.from_rotvec(wrist_rotvec_base).as_quat()
```

### 方案3: 使用旋转矩阵而非rotvec

如果可以修改模型输出格式，直接输出旋转矩阵而不是轴角：
- 更接近原始脚本的处理方式
- 可以更直接地进行坐标系转换

## 验证方法

1. **检查坐标系对齐**:
   ```python
   # 对比第一帧
   original_hand_rot  # 原始脚本的旋转矩阵
   predicted_wrist_rotvec  # 新API的轴角

   # 如果在同一坐标系，应该满足：
   R_original = original_hand_rot
   R_predicted = R.from_rotvec(predicted_wrist_rotvec).as_matrix()
   
   # 差异应该很小
   angle_diff = np.linalg.norm(R.from_matrix(R_original.T @ R_predicted).as_rotvec())
   print(f"旋转差异: {np.degrees(angle_diff):.2f}°")  # 应该接近0
   ```

2. **可视化对比**:
   - 在同一图像上绘制原始方向和retarget后的方向
   - 如果偏移很大，说明坐标系不一致

## 下一步行动

1. ✅ 确认`wrist_rotvec`是在相机坐标系下
2. ✅ 获取相机到机器人基座的转换矩阵
3. ⚠️ 在API中添加坐标系转换代码
4. ⚠️ 测试验证修复效果
