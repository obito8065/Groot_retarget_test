#!/usr/bin/env python3
"""
可视化左右臂各7个关节的角度变化趋势，读取robocasa_action,可视化检查IK后每个关节是否存在跳变

输入：
- robocasa_action_*.txt: 包含关节角度的数据文件

输出：
- 7*2=14 个子图，显示左右臂各7个关节的角度轨迹（散点图）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def parse_robocasa_action_file(file_path):
    """
    解析robocasa_action文件，提取关节角度数据
    将所有chunk的数据合并成连续的轨迹
    
    返回：
        left_arm: (T_total, 7) - 左手7个关节角度（所有chunk合并）
        right_arm: (T_total, 7) - 右手7个关节角度（所有chunk合并）
        global_time_steps: (T_total,) - 全局时间步（0, 1, 2, ..., T_total-1）
    """
    left_arm = []
    right_arm = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释和空行
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # 解析数据行
            parts = line.strip().split()
            if len(parts) < 31:  # 至少需要31个字段（chunk_id, t, 7+6+7+6+3）
                continue
            
            # 提取左手7个关节角度 (索引2-8)
            left_arm.append([
                float(parts[2]),   # L_arm_q1
                float(parts[3]),   # L_arm_q2
                float(parts[4]),   # L_arm_q3
                float(parts[5]),   # L_arm_q4
                float(parts[6]),   # L_arm_q5
                float(parts[7]),   # L_arm_q6
                float(parts[8]),   # L_arm_q7
            ])
            
            # 提取右手7个关节角度 (索引15-21)
            right_arm.append([
                float(parts[15]),  # R_arm_q1
                float(parts[16]),  # R_arm_q2
                float(parts[17]),  # R_arm_q3
                float(parts[18]),  # R_arm_q4
                float(parts[19]),  # R_arm_q5
                float(parts[20]),  # R_arm_q6
                float(parts[21]),  # R_arm_q7
            ])
    
    # 转换为numpy数组
    left_arm = np.array(left_arm)
    right_arm = np.array(right_arm)
    
    # 创建全局时间步（从0开始，连续递增）
    total_frames = len(left_arm)
    global_time_steps = np.arange(total_frames)
    
    return left_arm, right_arm, global_time_steps

def visualize_arm_joints(action_file, output_path=None):
    """
    可视化左右臂各7个关节的角度变化
    
    Args:
        action_file: robocasa_action文件的路径
        output_path: 输出图片路径（可选）
    """
    # 读取数据
    print(f"读取文件: {action_file}")
    left_arm, right_arm, time_steps = parse_robocasa_action_file(action_file)
    print(f"  左手: {left_arm.shape}, 右手: {right_arm.shape}")
    print(f"  总帧数: {len(time_steps)}, 时间步范围: [{time_steps[0]}, {time_steps[-1]}]")
    
    # 关节名称（GR1的7个关节）
    joint_names = [
        'Shoulder Pitch',
        'Shoulder Roll', 
        'Shoulder Yaw',
        'Elbow Pitch',
        'Wrist Yaw',
        'Wrist Roll',
        'Wrist Pitch'
    ]
    
    # 创建图形：7行2列，每行对应一个关节，每列对应一只手
    fig, axes = plt.subplots(7, 2, figsize=(16, 24))
    fig.suptitle('Arm Joint Angles Trajectory', fontsize=16, fontweight='bold')
    
    # 绘制每个关节的对比
    for joint_idx in range(7):
        # 左手
        ax_left = axes[joint_idx, 0]
        # 先绘制连线（轨迹）
        ax_left.plot(time_steps, left_arm[:, joint_idx], 'b-', linewidth=2, alpha=0.6, label='Trajectory')
        # 再绘制散点（每个时间步）
        ax_left.scatter(time_steps, left_arm[:, joint_idx], c='blue', s=60, alpha=0.8, marker='o', 
                       edgecolors='darkblue', linewidths=1, zorder=5, label='Time Steps')
        ax_left.set_ylabel(f'Left Arm\n{joint_names[joint_idx]}\n(rad)', fontsize=10)
        ax_left.grid(True, alpha=0.3)
        ax_left.set_xlim([time_steps[0] - 0.5, time_steps[-1] + 0.5])  # 确保显示所有时间步
        if joint_idx == 0:
            ax_left.set_title('Left Arm', fontsize=12, fontweight='bold')
            ax_left.legend(loc='upper right', fontsize=8)
        if joint_idx == 6:
            ax_left.set_xlabel('Time Step', fontsize=11)
        
        # 右手
        ax_right = axes[joint_idx, 1]
        # 先绘制连线（轨迹）
        ax_right.plot(time_steps, right_arm[:, joint_idx], 'r-', linewidth=2, alpha=0.6, label='Trajectory')
        # 再绘制散点（每个时间步）
        ax_right.scatter(time_steps, right_arm[:, joint_idx], c='red', s=60, alpha=0.8, marker='x', 
                        linewidths=2, zorder=5, label='Time Steps')
        ax_right.set_ylabel(f'Right Arm\n{joint_names[joint_idx]}\n(rad)', fontsize=10)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_xlim([time_steps[0] - 0.5, time_steps[-1] + 0.5])  # 确保显示所有时间步
        if joint_idx == 0:
            ax_right.set_title('Right Arm', fontsize=12, fontweight='bold')
            ax_right.legend(loc='upper right', fontsize=8)
        if joint_idx == 6:
            ax_right.set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = Path(action_file).parent / 'arm_joints_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 计算并打印统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    
    for joint_idx, joint_name in enumerate(joint_names):
        print(f"\n{joint_name}:")
        # 左手
        left_values = left_arm[:, joint_idx]
        left_mean = np.mean(left_values)
        left_std = np.std(left_values)
        left_min = np.min(left_values)
        left_max = np.max(left_values)
        left_range = left_max - left_min
        print(f"  左手 - 均值: {left_mean:.4f}, 标准差: {left_std:.4f}, 范围: [{left_min:.4f}, {left_max:.4f}], 跨度: {left_range:.4f}")
        
        # 右手
        right_values = right_arm[:, joint_idx]
        right_mean = np.mean(right_values)
        right_std = np.std(right_values)
        right_min = np.min(right_values)
        right_max = np.max(right_values)
        right_range = right_max - right_min
        print(f"  右手 - 均值: {right_mean:.4f}, 标准差: {right_std:.4f}, 范围: [{right_min:.4f}, {right_max:.4f}], 跨度: {right_range:.4f}")
    
    plt.show()

if __name__ == "__main__":
    # 文件路径
    parser = argparse.ArgumentParser(description="可视化左右臂各7个关节的角度变化趋势")
    parser.add_argument("--action_file", type=str, default="/vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260128_174917.txt",
                        help="robocasa_action文件路径")
    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.action_file).exists():
        print(f"错误: 找不到文件 {args.action_file}")
        exit(1)
    
    # 可视化
    visualize_arm_joints(args.action_file)


"""
python eval_visualize_arm_joints.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260128_174917.txt

"""
