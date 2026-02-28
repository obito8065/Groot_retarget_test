import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 文件路径
file_path = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260126_104537.txt")

# 手指关节名称（按数据文件顺序）
finger_names = ['index', 'middle', 'ring', 'pinky', 'thumb_yaw', 'thumb_pitch']

# URDF中的关节限位（按finger_names顺序）
# 注意：数据文件中的顺序是 [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
joint_limits_left = {
    'index': (-1.57, 0),
    'middle': (-1.57, 0),
    'ring': (-1.57, 0),
    'pinky': (-1.57, 0),
    'thumb_yaw': (-1.74, 0),
    'thumb_pitch': (0, 1.22)
}

joint_limits_right = {
    'index': (-1.57, 0),
    'middle': (-1.57, 0),
    'ring': (-1.57, 0),
    'pinky': (-1.57, 0),
    'thumb_yaw': (-1.74, 0),
    'thumb_pitch': (0, 1.22)
}

# 读取数据
data = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        if len(parts) >= 26:
            chunk_id = int(parts[0])
            t = int(parts[1])
            left_fingers = [float(parts[i]) for i in range(8, 14)]
            right_fingers = [float(parts[i]) for i in range(20, 26)]
            data.append({
                'chunk_id': chunk_id,
                't': t,
                'left_fingers': left_fingers,
                'right_fingers': right_fingers
            })

# 按chunk_id和t排序
data = sorted(data, key=lambda x: (x['chunk_id'], x['t']))

# 创建全局时间步
global_timesteps = []
left_finger_data = [[] for _ in range(6)]
right_finger_data = [[] for _ in range(6)]

current_global_t = 0
for d in data:
    global_timesteps.append(current_global_t)
    for finger_idx in range(6):
        left_finger_data[finger_idx].append(d['left_fingers'][finger_idx])
        right_finger_data[finger_idx].append(d['right_fingers'][finger_idx])
    current_global_t += 1

# 转换为numpy数组
global_timesteps = np.array(global_timesteps)
left_finger_data = [np.array(values) for values in left_finger_data]
right_finger_data = [np.array(values) for values in right_finger_data]

# 检查超出限位的情况
print("=" * 80)
print("手指关节限位检查报告")
print("=" * 80)

violations_found = False

for finger_idx, finger_name in enumerate(finger_names):
    # 左手检查
    left_values = left_finger_data[finger_idx]
    left_lower, left_upper = joint_limits_left[finger_name]
    
    left_below = left_values < left_lower
    left_above = left_values > left_upper
    left_violations = np.sum(left_below) + np.sum(left_above)
    
    if left_violations > 0:
        violations_found = True
        print(f"\n【左手 {finger_name}】超出限位！")
        print(f"  限位范围: [{left_lower}, {left_upper}]")
        print(f"  最小值: {np.min(left_values):.6f} (低于下限: {np.sum(left_below)} 次)")
        print(f"  最大值: {np.max(left_values):.6f} (超过上限: {np.sum(left_above)} 次)")
        if np.sum(left_below) > 0:
            below_indices = np.where(left_below)[0]
            print(f"  低于下限的时间步: {below_indices[:10].tolist()}{'...' if len(below_indices) > 10 else ''}")
        if np.sum(left_above) > 0:
            above_indices = np.where(left_above)[0]
            print(f"  超过上限的时间步: {above_indices[:10].tolist()}{'...' if len(above_indices) > 10 else ''}")
    
    # 右手检查
    right_values = right_finger_data[finger_idx]
    right_lower, right_upper = joint_limits_right[finger_name]
    
    right_below = right_values < right_lower
    right_above = right_values > right_upper
    right_violations = np.sum(right_below) + np.sum(right_above)
    
    if right_violations > 0:
        violations_found = True
        print(f"\n【右手 {finger_name}】超出限位！")
        print(f"  限位范围: [{right_lower}, {right_upper}]")
        print(f"  最小值: {np.min(right_values):.6f} (低于下限: {np.sum(right_below)} 次)")
        print(f"  最大值: {np.max(right_values):.6f} (超过上限: {np.sum(right_above)} 次)")
        if np.sum(right_below) > 0:
            below_indices = np.where(right_below)[0]
            print(f"  低于下限的时间步: {below_indices[:10].tolist()}{'...' if len(below_indices) > 10 else ''}")
        if np.sum(right_above) > 0:
            above_indices = np.where(right_above)[0]
            print(f"  超过上限的时间步: {above_indices[:10].tolist()}{'...' if len(above_indices) > 10 else ''}")

if not violations_found:
    print("\n✓ 所有手指关节数据都在限位范围内！")
else:
    print("\n" + "=" * 80)
    print("⚠️  发现超出限位的关节数据！这可能导致机器人碰撞或异常行为。")
    print("=" * 80)

# 创建可视化图表，标注限位线
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('Left and Right Hand Finger Joint Trajectories (with Joint Limits)', 
             fontsize=16, fontweight='bold')

for finger_idx in range(6):
    finger_name = finger_names[finger_idx]
    ax_left = axes[0, finger_idx]
    ax_right = axes[1, finger_idx]
    
    # 左手数据
    left_values = left_finger_data[finger_idx]
    left_lower, left_upper = joint_limits_left[finger_name]
    
    ax_left.plot(global_timesteps, left_values, marker='o', markersize=2, 
                linewidth=1.5, alpha=0.8, label='Joint Value')
    ax_left.axhline(y=left_lower, color='r', linestyle='--', linewidth=2, 
                    label=f'Lower Limit ({left_lower})', alpha=0.7)
    ax_left.axhline(y=left_upper, color='r', linestyle='--', linewidth=2, 
                    label=f'Upper Limit ({left_upper})', alpha=0.7)
    
    # 标记超出限位的点
    left_below = left_values < left_lower
    left_above = left_values > left_upper
    if np.any(left_below):
        ax_left.scatter(global_timesteps[left_below], left_values[left_below], 
                       color='red', s=50, marker='x', zorder=5, label='Below Limit')
    if np.any(left_above):
        ax_left.scatter(global_timesteps[left_above], left_values[left_above], 
                       color='red', s=50, marker='x', zorder=5, label='Above Limit')
    
    ax_left.set_title(f'Left {finger_name}', fontsize=10, fontweight='bold')
    ax_left.set_xlabel('Global Time Step', fontsize=9)
    ax_left.set_ylabel('Joint Value (rad)', fontsize=9)
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(fontsize=6, loc='best')
    
    # 右手数据
    right_values = right_finger_data[finger_idx]
    right_lower, right_upper = joint_limits_right[finger_name]
    
    ax_right.plot(global_timesteps, right_values, marker='o', markersize=2, 
                 linewidth=1.5, alpha=0.8, label='Joint Value')
    ax_right.axhline(y=right_lower, color='r', linestyle='--', linewidth=2, 
                     label=f'Lower Limit ({right_lower})', alpha=0.7)
    ax_right.axhline(y=right_upper, color='r', linestyle='--', linewidth=2, 
                     label=f'Upper Limit ({right_upper})', alpha=0.7)
    
    # 标记超出限位的点
    right_below = right_values < right_lower
    right_above = right_values > right_upper
    if np.any(right_below):
        ax_right.scatter(global_timesteps[right_below], right_values[right_below], 
                        color='red', s=50, marker='x', zorder=5, label='Below Limit')
    if np.any(right_above):
        ax_right.scatter(global_timesteps[right_above], right_values[right_above], 
                        color='red', s=50, marker='x', zorder=5, label='Above Limit')
    
    ax_right.set_title(f'Right {finger_name}', fontsize=10, fontweight='bold')
    ax_right.set_xlabel('Global Time Step', fontsize=9)
    ax_right.set_ylabel('Joint Value (rad)', fontsize=9)
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=6, loc='best')

plt.tight_layout()

# 保存图片
output_path = file_path.parent / 'finger_joint_trajectories_with_limits.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n图片已保存至: {output_path}")

plt.show()