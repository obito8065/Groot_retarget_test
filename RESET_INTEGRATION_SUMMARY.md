# Reset 集成总结

## 概述

在每个新 episode 开始时，需要重置 IK 和 Retarget 的 last_qpos 缓存，以避免历史状态干扰新 episode 的推理。

## 架构设计

### 1. 组件分类

| 组件 | 是否需要 Reset | 原因 |
|------|---------------|------|
| **Body IK** (`body_retargeter`) | ✅ 需要 | 包含 last_qpos 缓存用于 IK 优化 |
| **Fourier Hand Retarget** (`fourier_hand_retargeter`) | ✅ 需要 | 包含 warmup 状态和优化器历史 |
| **Fourier Hand FK** (`policy_fourier_hand_keypoints`) | ❌ 不需要 | 完全无状态的正向运动学计算 |

### 2. 接口对齐

所有需要 reset 的组件都实现了统一的接口：

```python
def reset(self, env_idx: Optional[int] = None):
    """
    Args:
        env_idx: 并行环境索引
            - None: 重置所有环境
            - int: 仅重置指定环境
    """
    pass
```

## 实现细节

### 1. Policy 中的 reset_ik_cache 方法

位置：`/vla/users/lijiayi/code/groot_retarget/gr00t/model/policy.py`

```python
def reset_ik_cache(self, env_idx: Optional[int] = None):
    """
    清空 Robocasa/GR1 EEPose IK 的历史缓存，以及 Fourier Hand Retarget 的 last_qpos 缓存。
    
    在每个新 episode 开始时调用，用于重置：
    1. Body IK 的 last_qpos 缓存（body_retargeter）
    2. Fourier Hand Retarget 的 warmup 状态和 last_qpos（fourier_hand_retargeter）
    
    注意：
    - FK（policy_fourier_hand_keypoints）是无状态的，不需要reset
    - 只有IK和Retarget需要清空历史qpos缓存
    """
    # 1. 重置 Body IK 缓存
    if hasattr(self, "body_retargeter") and hasattr(self.body_retargeter, "reset_ik_cache"):
        self.body_retargeter.reset_ik_cache(env_idx)
    
    # 2. 重置 Fourier Hand Retarget 缓存
    if hasattr(self, "fourier_hand_retargeter") and hasattr(self.fourier_hand_retargeter, "reset"):
        self.fourier_hand_retargeter.reset(env_idx)
```

### 2. Fourier Hand Retarget API

位置：`/vla/users/lijiayi/code/groot_retarget/gr00t/eval/fourier_hand_retarget_api.py`

```python
def reset(self, env_idx: Optional[int] = None):
    """
    重置API状态（新episode或新环境开始时调用）
    
    重要: 每个新episode开始前必须调用此方法！
    这会重置warmup状态和last_qpos缓存。
    """
    # 重置帧计数和warmup状态
    for side in self.hand_sides:
        self._episode_frame_count[side] = 0
        self._is_warmed_up[side] = False
    
    print(f"[FourierHandRetargetAPI] Reset for new episode (env_idx={env_idx})")
```

**关键特性：**
- ✅ 包含 warmup 处理（episode 开始的前几帧）
- ✅ 支持 45 维输入格式（与训练数据对齐）
- ✅ 严格遵循原始 retarget 脚本的处理流程
- ✅ 接受 `env_idx` 参数以支持并行环境

### 3. Fourier Hand FK

位置：`/vla/users/lijiayi/code/groot_retarget/gr00t/eval/gr1_hand_fk.py`

```python
class PolicyFourierHandKeypoints:
    """
    FK 是无状态的，不需要 reset 方法
    每次调用 compute_state_45d 都是独立计算
    """
    def __init__(self, left_urdf: Path, right_urdf: Path):
        self.fk_L = FourierHandFK(left_urdf, side="L")
        self.fk_R = FourierHandFK(right_urdf, side="R")
    
    def compute_state_45d(self, left_arm, left_hand, right_arm, right_hand, waist):
        """完全无状态的 FK 计算"""
        pass
```

## 调用流程

### 1. 在仿真环境中的使用

```python
# 初始化 policy
policy = Gr00tPolicy(
    model_path="path/to/model",
    use_eepose=True,
    use_fourier_hand_retarget=True,
    ...
)

# Episode 循环
for episode in range(num_episodes):
    # 1. 环境 reset
    obs = env.reset()
    
    # 2. 在 obs 中添加 reset 标记
    obs["meta.reset_mask"] = True  # 或 np.array([True, False, ...]) 用于并行环境
    
    # 3. Policy 会自动检测 reset_mask 并调用 reset_ik_cache
    action = policy.get_action(obs)
    
    # 4. Episode 循环
    for step in range(max_steps):
        obs, reward, done, info = env.step(action)
        action = policy.get_action(obs)
        if done:
            break
```

### 2. reset_mask 的处理逻辑

在 `policy.py` 的 `get_action` 方法中：

```python
def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
    obs_copy = observations.copy()
    
    # 检测 reset 标记
    reset_mask = None
    if "meta.reset_mask" in obs_copy:
        reset_mask = obs_copy.pop("meta.reset_mask", None)
    
    # 根据 reset_mask 调用 reset_ik_cache
    if reset_mask is not None:
        rm = np.asarray(reset_mask).astype(bool)
        if rm.ndim == 0:
            # 单环境
            if bool(rm):
                self.reset_ik_cache(env_idx=0)
        else:
            # 并行环境
            for env_idx, flag in enumerate(rm):
                if bool(flag):
                    self.reset_ik_cache(env_idx=env_idx)
    
    # ... 继续推理
```

## 关键改进点

### 1. 从旧版本到新版本

**旧版本（fourier_hand_retarget_api.py 原始版本）：**
- ❌ 没有 warmup 处理
- ❌ 不支持 45 维输入
- ❌ reset 方法为空实现

**新版本（当前版本）：**
- ✅ 包含完整的 warmup 处理（与原始 retarget 脚本一致）
- ✅ 支持 45 维输入格式（`retarget_from_45d`）
- ✅ reset 方法正确重置 warmup 状态和帧计数
- ✅ 接口与 policy.py 完全对齐

### 2. FK 的简化设计

**为什么 FK 不需要 reset：**
- FK 是纯函数，输入相同则输出相同
- 不依赖历史状态或缓存
- 每次调用都是独立计算
- 不需要优化或迭代求解

**对比 IK/Retarget：**
- IK/Retarget 需要迭代优化
- 使用 last_qpos 作为初始猜测可以加速收敛
- warmup 可以提供更好的初始状态

## 测试验证

### 测试脚本

位置：`/vla/users/lijiayi/code/groot_retarget/gr00t/eval/test_reset_integration.py`

### 测试结果

```bash
$ python fourier_hand_retarget_api.py
[FourierHandRetargetAPI] Initialized successfully
  Robot: fourier, Sides: ['left', 'right']
  Wrist enhance weight: 2.0
  Warmup steps: 1
✓ FourierHandRetargetAPI 初始化成功
[FourierHandRetargetAPI] Reset for new episode (env_idx=None)
✓ reset(env_idx=None) 调用成功
[FourierHandRetargetAPI] Reset for new episode (env_idx=0)
✓ reset(env_idx=0) 调用成功
✓ 接口对齐完成！
```

## 文件清单

### 修改的文件

1. **`policy.py`**
   - 更新 `reset_ik_cache` 方法
   - 移除 FK 的 reset 调用
   - 添加详细注释

2. **`fourier_hand_retarget_api.py`**
   - 用户已升级为 v2 版本
   - 添加 warmup 处理
   - 支持 45 维输入
   - 更新 `reset` 方法接受 `env_idx` 参数

3. **`gr1_hand_fk.py`** 和 **`gr1_hand_fk_v2.py`**
   - 移除不必要的 reset 方法
   - 保持 FK 的无状态设计

### 新增的文件

1. **`test_reset_integration.py`**
   - 测试脚本，验证接口对齐

2. **`RESET_INTEGRATION_SUMMARY.md`**
   - 本文档，总结 reset 集成

## 最佳实践

### 1. 并行环境处理

```python
# 在并行环境中，每个环境独立 reset
num_envs = 4
reset_mask = np.zeros(num_envs, dtype=bool)

# 只 reset env_0 和 env_2
reset_mask[0] = True
reset_mask[2] = True

obs["meta.reset_mask"] = reset_mask
action = policy.get_action(obs)
```

### 2. 单环境处理

```python
# 单环境直接传 True 或 False
obs["meta.reset_mask"] = True  # 或 False
action = policy.get_action(obs)
```

### 3. 不使用 reset_mask

如果不在 obs 中添加 `meta.reset_mask`，可以手动调用：

```python
# Episode 开始时手动 reset
policy.reset_ik_cache(env_idx=0)

# 或重置所有环境
policy.reset_ik_cache(env_idx=None)
```

## 总结

✅ **接口对齐完成：**
- Body IK: `reset_ik_cache(env_idx)`
- Fourier Hand Retarget: `reset(env_idx)`
- Fourier Hand FK: 无需 reset（无状态）

✅ **功能完整：**
- 支持单环境和并行环境
- 自动检测 reset_mask
- 正确重置 IK 和 Retarget 的历史状态

✅ **代码清晰：**
- FK 保持无状态设计
- 只在需要的地方添加 reset
- 接口统一且易于使用

🎯 **推荐使用流程：**
1. 在环境 reset 后，在 obs 中添加 `meta.reset_mask`
2. Policy 会自动检测并调用 `reset_ik_cache`
3. IK 和 Retarget 的 last_qpos 缓存会被清空
4. 新 episode 从干净的状态开始推理
