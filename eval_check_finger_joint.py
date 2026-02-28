import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 文件路径
file_path = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260126_104537.txt")

# 手指关节名称
finger_names = ['index', 'middle', 'ring', 'pinky',  'thumb_pitch','thumb_yaw']

# 读取数据
data = []
with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()
        # 跳过注释行和空行
        if line.startswith('#') or not line:
            continue
        # 解析数据
        parts = line.split()
        if len(parts) >= 26:  # 确保有足够的数据列
            chunk_id = int(parts[0])
            t = int(parts[1])
            # 左手手指关节 (索引 8-13)
            left_fingers = [float(parts[i]) for i in range(8, 14)]
            # 右手手指关节 (索引 20-25)
            right_fingers = [float(parts[i]) for i in range(20, 26)]
            data.append({
                'chunk_id': chunk_id,
                't': t,
                'left_fingers': left_fingers,
                'right_fingers': right_fingers
            })

# 按chunk_id和t排序
data = sorted(data, key=lambda x: (x['chunk_id'], x['t']))

# 创建全局时间步（将所有chunk连续连接）
global_timesteps = []
left_finger_data = [[] for _ in range(6)]  # 6个手指关节
right_finger_data = [[] for _ in range(6)]  # 6个手指关节

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

# 创建图形，12个子图：2行6列（左手一行，右手一行）
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.suptitle('Left and Right Hand Finger Joint Trajectories', fontsize=16, fontweight='bold')

# 为每个手指关节绘制轨迹
for finger_idx in range(6):
    # 左手子图（第一行）
    ax_left = axes[0, finger_idx]
    
    # 右手子图（第二行）
    ax_right = axes[1, finger_idx]
    
    # 绘制左手连续轨迹
    ax_left.plot(global_timesteps, left_finger_data[finger_idx], 
                marker='o', markersize=2, linewidth=1.5, alpha=0.8)
    
    # 绘制右手连续轨迹
    ax_right.plot(global_timesteps, right_finger_data[finger_idx], 
                 marker='o', markersize=2, linewidth=1.5, alpha=0.8)
    
    # 设置左手子图标题和标签
    ax_left.set_title(f'Left {finger_names[finger_idx]}', fontsize=10, fontweight='bold')
    ax_left.set_xlabel('Global Time Step', fontsize=9)
    ax_left.set_ylabel('Joint Value (rad)', fontsize=9)
    ax_left.grid(True, alpha=0.3)
    
    # 设置右手子图标题和标签
    ax_right.set_title(f'Right {finger_names[finger_idx]}', fontsize=10, fontweight='bold')
    ax_right.set_xlabel('Global Time Step', fontsize=9)
    ax_right.set_ylabel('Joint Value (rad)', fontsize=9)
    ax_right.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
output_path = file_path.parent / 'finger_joint_trajectories.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图片已保存至: {output_path}")

# 显示图片
plt.show()