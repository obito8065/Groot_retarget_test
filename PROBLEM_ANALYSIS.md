# Fourier Hand Retarget API 问题分析

## 问题现象
使用相同的 `predicted_keypoints.txt` 输入：
- ✅ **正常**: `hand_robot_viewer_fourier_for_eval.py` 加载 HDF5 数据进行 retarget 是正确的
- ❌ **异常**: `fourier_hand_retarget_api.py` 从 predicted_keypoints 进行 retarget 输出错误

## 根本原因

### 问题1: **坐标系转换缺失**

#### 正常脚本 (`hand_robot_viewer_fourier.py`)
```python
# Line 1194: 从 HDF5 读取
hand_rot = raw_task_data['hand_dict'][side]['pose_seqs']['cam']['rot']  # (T, 3, 3)

# Line 625: warmup 使用旋转矩阵
wrist_quat = rotations.quaternion_from_matrix(rot)  # rot 是 3x3 矩阵
```

**关键**: HDF5 中的 `hand_rot` 虽然标注为 'cam'，但这是 **MANO 模型输出的手部姿态**，已经是相对于手腕局部坐标系的旋转，可以直接用于 warmup。

#### API 脚本 (`fourier_hand_retarget_api.py`)  
```python
# Line 254: 从 predicted_keypoints 读取
wrist_rotvec = key_points_21[18:21]  # 轴角表示（相机坐标系的世界旋转）

# Line 269: 直接转换为四元数
wrist_quat = R.from_rotvec(wrist_rotvec).as_quat()  # ❌ 缺少坐标系转换！
```

**问题**: `predicted_keypoints` 中的 `wrist_rotvec` 是 **相机坐标系下的手腕世界姿态**，而不是 MANO 的手腕局部姿态。直接用于 warmup 会导致姿态错误。

---

## 详细解释

### 1. 数据源的本质区别

**HDF5 数据 (正常脚本)**:
- `hand_joint`: (T, 21, 3) - MANO 模型输出的 21 个关键点
- `hand_rot`: (T, 3, 3) - MANO 模型输出的**手腕局部姿态**
  - 这是相对于手部自身坐标系的旋转
  - 已经考虑了手部的内在结构
  - 可以直接用于 retargeting

**Predicted Keypoints (API)**:
- `wrist_xyz`: (3,) - 相机坐标系下的手腕位置
- `tips_xyz`: (15,) - 相机坐标系下的指尖位置  
- `wrist_rotvec`: (3,) - 相机坐标系下的**手腕全局姿态**
  - 这是相对于相机坐标系的旋转
  - 没有考虑手部的内在结构
  - **不能直接用于 retargeting**

### 2. Warmup 的作用

Warmup 的目的是给优化器一个好的初始姿态，使用的是 **MANO 约定的手腕局部姿态**：
```python
retargeting.warm_start(
    wrist_pos=wrist_xyz,
    wrist_quat=wrist_quat,  # 应该是手腕的局部姿态
    is_mano_convention=True  # 使用 MANO 约定
)
```

### 3. 为什么位置可以直接用但旋转不行？

- **位置信息**: IK 优化器会根据目标关键点的 3D 位置自动求解关节角度，坐标系会在优化过程中隐式处理
- **旋转信息**: Warmup 使用的旋转是作为**初始猜测**，如果这个猜测的坐标系不对，会导致优化陷入错误的局部最优解

---

## 解决方案

需要将 `wrist_rotvec` 从相机坐标系转换到手腕局部坐标系，或者**不使用 wrist_rotvec 进行 warmup**。

### 方案1: 去除 warmup 中的旋转信息（推荐）
```python
# 只使用位置进行 warmup，让优化器自己找旋转
warmup_qpos6d = retargeting.warm_start(
    wrist_pos=wrist_xyz,
    wrist_quat=np.array([1, 0, 0, 0]),  # 单位四元数（无旋转）
    hand_type=hand_type,
    is_mano_convention=True,
)
```

### 方案2: 从关键点几何关系推导手部姿态
利用手腕到指尖的向量来估计手部的局部坐标系：
```python
# 从手腕指向中指的向量作为手部前向
forward = normalize(middle_tip - wrist_xyz)
# 从手腕指向拇指和小指的向量计算侧向
side_vec = normalize(thumb_tip - pinky_tip)
# 构造旋转矩阵
...
```

### 方案3: 提供相机外参进行坐标转换（不推荐）
需要相机到机器人 base 的外参，计算复杂且容易出错。

---

## 具体修改位置

**文件**: `fourier_hand_retarget_api.py`  
**行数**: Line 266-284 (warmup 部分)

当前错误代码：
```python
if self._episode_frame_count[side] < self.warm_up_steps:
    wrist_quat = R.from_rotvec(wrist_rotvec).as_quat()  # ❌ 错误！
    wrist_quat_wxyz = np.array([wrist_quat[3], wrist_quat[0], wrist_quat[1], wrist_quat[2]])
    
    self._warmup(
        wrist_pos=wrist_xyz,
        wrist_quat=wrist_quat_wxyz,  # ❌ 错误的坐标系
        side=side
    )
```

建议修改：
```python
if self._episode_frame_count[side] < self.warm_up_steps:
    # 使用单位四元数，只用位置信息进行 warmup
    wrist_quat_identity = np.array([1, 0, 0, 0])  # [w, x, y, z]
    
    self._warmup(
        wrist_pos=wrist_xyz,
        wrist_quat=wrist_quat_identity,  # ✓ 让优化器自己找旋转
        side=side
    )
```

---

## 验证方法

1. 修改后重新运行 API，生成新的 `retargeted_actions.txt`
2. 使用 `eval_after_retarget_reprojector_cli.py` 可视化，检查手腕方向是否正确
3. 使用 `hand_robot_viewer_fourier_for_eval.py` 进行 Sapien 可视化，检查整体姿态

---

## 补充说明

这个问题很隐蔽，因为：
1. 位置信息是正确的（IK 自动处理）
2. 手指关节可能看起来也基本正确（因为主要由指尖位置决定）
3. 但手腕的朝向会明显错误（因为 warmup 使用了错误坐标系的旋转）

这也解释了为什么用户看到"手腕和手指数据有问题" - 手腕的方向错了，会连带影响整个手部的姿态。
