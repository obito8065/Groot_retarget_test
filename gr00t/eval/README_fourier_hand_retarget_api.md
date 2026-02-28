# Fourier Hand Retarget API

Fourier 灵巧手重定向 API，将人体关键点数据转换为机器人手部控制指令。

## 功能概述

- **输入**: 45维关键点数据（左右手关键点 + 腰部）
- **输出**: 手腕位姿（6D） + 6个手指关节角度
- **特性**: 
  - 自动 warmup 处理（episode 前 N 帧）
  - 轴角连续性保证
  - 符号修正（直接可用于 MuJoCo 控制）

## 输入输出格式

### 输入格式 (45维)

```python
state_45d: np.ndarray, shape=(45,)
```

**数据布局**:
- `[0:21]` - **left_key_points**: 
  - `[0:3]` - wrist_xyz (手腕位置)
  - `[3:6]` - thumb_tip_xyz (拇指尖)
  - `[6:9]` - index_tip_xyz (食指尖)
  - `[9:12]` - middle_tip_xyz (中指尖)
  - `[12:15]` - ring_tip_xyz (无名指尖)
  - `[15:18]` - pinky_tip_xyz (小指尖)
  - `[18:21]` - wrist_rotvec (手腕轴角旋转)
- `[21:42]` - **right_key_points**: 同上格式
- `[42:45]` - **waist**: 腰部位置 (3维)

### 输出格式

```python
result: Dict[str, Dict[str, np.ndarray]]
```

**结构**:
```python
{
    'left': {
        'wrist_pose': np.ndarray,      # (6,) [pos_xyz(3), rotvec_xyz(3)]
        'finger_joints': np.ndarray,   # (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    },
    'right': {
        'wrist_pose': np.ndarray,      # (6,) [pos_xyz(3), rotvec_xyz(3)]
        'finger_joints': np.ndarray,   # (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    }
}
```

**说明**:
- `wrist_pose`: 手腕位姿，前3维为位置，后3维为轴角旋转
- `finger_joints`: 6个主动关节角度，**已进行符号修正**，可直接用于 MuJoCo 控制
  - 顺序: `[pinky, ring, middle, index, thumb_pitch, thumb_yaw]`
  - 符号修正: `pinky, ring, middle, index, thumb_yaw` 取负号，`thumb_pitch` 保持不变

## 使用方法

### 1. 初始化

```python
from gr00t.eval.fourier_hand_retarget_api import FourierHandRetargetAPI
import numpy as np

# 创建 API 实例（只需初始化一次）
retargeter = FourierHandRetargetAPI(
    robot_name="fourier",           # 机器人名称
    hand_sides=["left", "right"],    # 手部列表
    wrist_enhance_weight=2.0,        # 手腕优化权重
    warm_up_steps=5,                 # warmup 帧数
)
```

### 2. Episode 开始时重置

```python
# 每个新 episode 开始前必须调用 reset()
retargeter.reset()
```

### 3. 处理每一帧

```python
# 输入 45 维数据
state_45d = np.array([...], dtype=np.float32)  # shape=(45,)

# 执行 retarget
result = retargeter.retarget_from_45d(state_45d)

# 提取结果
left_wrist_pose = result['left']['wrist_pose']        # (6,)
left_finger_joints = result['left']['finger_joints']  # (6,)
right_wrist_pose = result['right']['wrist_pose']      # (6,)
right_finger_joints = result['right']['finger_joints']  # (6,)
```

## 完整示例

```python
import numpy as np
from gr00t.eval.fourier_hand_retarget_api import FourierHandRetargetAPI

def test_fourier_hand_retarget():
    """测试 Fourier Hand Retarget API"""
    
    # 1. 初始化
    print("初始化 API...")
    retargeter = FourierHandRetargetAPI(warm_up_steps=5)
    
    # 2. 开始新 episode
    print("开始新 episode...")
    retargeter.reset()
    
    # 3. 模拟多帧数据
    for frame_idx in range(10):
        # 构造 45 维输入数据（示例）
        state_45d = np.random.randn(45).astype(np.float32)
        
        # 执行 retarget
        result = retargeter.retarget_from_45d(state_45d)
        
        # 打印结果
        for side in ['left', 'right']:
            wrist_pose = result[side]['wrist_pose']
            finger_joints = result[side]['finger_joints']
            
            print(f"\nFrame {frame_idx} - {side} hand:")
            print(f"  手腕位置: {wrist_pose[0:3]}")
            print(f"  手腕旋转: {wrist_pose[3:6]}")
            print(f"  手指关节: {finger_joints}")

if __name__ == "__main__":
    test_fourier_hand_retarget()
```

## 处理流程

1. **数据解析**: 从 45 维输入中提取左右手关键点
2. **Warmup**: 前 N 帧（`warm_up_steps`）使用手腕位姿进行 warmup 初始化
3. **Retarget**: 基于关键点位置进行优化求解，得到完整关节角度
4. **FK 计算**: 通过 SAPIEN 前向运动学计算 `hand_base_link` 的实际位姿
5. **轴角连续性**: 确保旋转表示的连续性，避免跳变
6. **符号修正**: 对关节角度进行符号修正，适配 MuJoCo 控制

## 注意事项

1. **必须调用 reset()**: 每个新 episode 开始前必须调用 `reset()`，否则 warmup 状态不会重置
2. **输入维度**: 输入必须是 45 维，否则会抛出 `ValueError`
3. **数据类型**: 建议使用 `np.float32` 类型
4. **坐标系**: 默认使用相机坐标系（`is_mano_convention=False`）

## API 参数说明

### `__init__()` 参数

- `robot_name` (str): 机器人名称，默认 `"fourier"`
- `hand_sides` (List[str]): 手部列表，默认 `["left", "right"]`
- `wrist_enhance_weight` (float): 手腕优化权重，默认 `2.0`
- `warm_up_steps` (int): warmup 帧数，默认 `5`

### `reset()` 参数

- `env_idx` (Optional[int]): 环境索引，用于并行环境场景，默认 `None`（重置所有）

### `retarget_from_45d()` 参数

- `state_45d` (np.ndarray): 45 维输入数据，shape 必须为 `(45,)`

## 依赖

- `numpy`
- `scipy`
- `sapien`
- `pytransform3d`
- `dex_retargeting` (来自 `robot_retarget_for_anything`)
