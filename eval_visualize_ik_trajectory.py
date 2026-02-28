#!/usr/bin/env python3
"""
可视化IK输入和输出的完整轨迹对比
用于分析轴角跳变是在IK前还是IK后产生的

输入：
- robocasa_action_*.txt: 包含IK输入和输出的日志文件

输出：
- 一张大图，包含所有轨迹：
  - 横轴：全局时间戳（连续）
  - 纵轴：角度值
  - 包含：wrist pose的xyz位置、rotvec（3个轴角）、7个arm joint angles
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_ik_log_file(file_path):
    """
    解析IK日志文件，提取所有轨迹数据
    
    返回：
        timestamps_input: (T,) - IK输入时间戳
        timestamps_output: (T,) - IK输出时间戳
        left_ik_input_pos: (T, 3) - 左手IK输入位置
        left_ik_input_rotvec: (T, 3) - 左手IK输入轴角
        left_ik_output: (T, 7) - 左手IK输出关节角度
        right_ik_input_pos: (T, 3) - 右手IK输入位置
        right_ik_input_rotvec: (T, 3) - 右手IK输入轴角
        right_ik_output: (T, 7) - 右手IK输出关节角度
    """
    timestamps_input = []
    timestamps_output = []
    left_ik_input_pos = []
    left_ik_input_rotvec = []
    left_ik_output = []
    right_ik_input_pos = []
    right_ik_input_rotvec = []
    right_ik_output = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释和空行
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # 解析数据行
            parts = line.strip().split()
            if len(parts) < 45:  # chunk_id + t + timestamp_input + timestamp_output + 41个数据值
                continue
            
            # 提取时间戳（索引2和3）- IK输入和IK输出的真实时间戳
            timestamp_input = float(parts[2])
            timestamp_output = float(parts[3])
            timestamps_input.append(timestamp_input)
            timestamps_output.append(timestamp_output)
            
            # 提取左手IK输入：pos(3) + rotvec(3) = 索引4-9
            left_ik_input_pos.append([
                float(parts[4]),  # L_IK_input_pos_x
                float(parts[5]),  # L_IK_input_pos_y
                float(parts[6]),  # L_IK_input_pos_z
            ])
            left_ik_input_rotvec.append([
                float(parts[7]),  # L_IK_input_rotvec_x
                float(parts[8]),  # L_IK_input_rotvec_y
                float(parts[9]),  # L_IK_input_rotvec_z
            ])
            
            # 提取左手IK输出：arm joints(7) = 索引10-16
            left_ik_output.append([
                float(parts[10]),   # L_IK_output_q1
                float(parts[11]),  # L_IK_output_q2
                float(parts[12]),  # L_IK_output_q3
                float(parts[13]),  # L_IK_output_q4
                float(parts[14]),  # L_IK_output_q5
                float(parts[15]),  # L_IK_output_q6
                float(parts[16]),  # L_IK_output_q7
            ])
            
            # 提取右手IK输入：pos(3) + rotvec(3) = 索引17-22
            right_ik_input_pos.append([
                float(parts[17]),  # R_IK_input_pos_x
                float(parts[18]),  # R_IK_input_pos_y
                float(parts[19]),  # R_IK_input_pos_z
            ])
            right_ik_input_rotvec.append([
                float(parts[20]),  # R_IK_input_rotvec_x
                float(parts[21]),  # R_IK_input_rotvec_y
                float(parts[22]),  # R_IK_input_rotvec_z
            ])
            
            # 提取右手IK输出：arm joints(7) = 索引23-29
            right_ik_output.append([
                float(parts[23]),  # R_IK_output_q1
                float(parts[24]),  # R_IK_output_q2
                float(parts[25]),  # R_IK_output_q3
                float(parts[26]),  # R_IK_output_q4
                float(parts[27]),  # R_IK_output_q5
                float(parts[28]),  # R_IK_output_q6
                float(parts[29]),  # R_IK_output_q7
            ])
    
    return (
        np.array(timestamps_input),
        np.array(timestamps_output),
        np.array(left_ik_input_pos),
        np.array(left_ik_input_rotvec),
        np.array(left_ik_output),
        np.array(right_ik_input_pos),
        np.array(right_ik_input_rotvec),
        np.array(right_ik_output)
    )

def visualize_ik_trajectory(log_file, output_path=None):
    """
    可视化IK输入和输出的完整轨迹
    
    Args:
        log_file: IK日志文件路径
        output_path: 输出图片路径（可选）
    """
    # 读取数据
    print(f"读取文件: {log_file}")
    (timestamps_input, timestamps_output, left_ik_input_pos, left_ik_input_rotvec, left_ik_output,
     right_ik_input_pos, right_ik_input_rotvec, right_ik_output) = parse_ik_log_file(log_file)
    
    print(f"  总帧数: {len(timestamps_input)}")
    print(f"  IK输入时间戳范围: [{timestamps_input[0]:.9f}, {timestamps_input[-1]:.9f}]")
    print(f"  IK输出时间戳范围: [{timestamps_output[0]:.9f}, {timestamps_output[-1]:.9f}]")
    print(f"  IK输入时间跨度: {timestamps_input[-1] - timestamps_input[0]:.6f} 秒")
    print(f"  IK输出时间跨度: {timestamps_output[-1] - timestamps_output[0]:.6f} 秒")
    
    # 分析chunk分布（通过检测时间戳的大跳跃）
    chunk_boundaries = [0]
    for i in range(1, len(timestamps_input)):
        # 如果时间戳跳跃超过0.1秒，认为是新的chunk
        if timestamps_input[i] - timestamps_input[i-1] > 0.1:
            chunk_boundaries.append(i)
    chunk_boundaries.append(len(timestamps_input))
    
    print(f"  检测到 {len(chunk_boundaries)-1} 个chunk")
    for i in range(len(chunk_boundaries)-1):
        start_idx = chunk_boundaries[i]
        end_idx = chunk_boundaries[i+1]
        print(f"    Chunk {i}: 索引 {start_idx}-{end_idx-1}, "
              f"时间戳范围 [{timestamps_input[start_idx]:.6f}, {timestamps_input[end_idx-1]:.6f}], "
              f"跨度 {timestamps_input[end_idx-1] - timestamps_input[start_idx]:.6f} 秒")
    
    # 将时间戳转换为相对时间（从0开始，以秒为单位）
    # IK输入使用输入时间戳，IK输出使用输出时间戳
    timestamps_input_relative = timestamps_input - timestamps_input[0]  # 相对时间（秒）
    timestamps_output_relative = timestamps_output - timestamps_output[0]  # 相对时间（秒）
    
    print(f"  相对时间范围（输入）: [{timestamps_input_relative[0]:.6f}, {timestamps_input_relative[-1]:.6f}] 秒")
    print(f"  相对时间范围（输出）: [{timestamps_output_relative[0]:.6f}, {timestamps_output_relative[-1]:.6f}] 秒")
    
    # 创建图形：4行2列
    # 第1行：IK输入的rotvec（3个轴角）- 用于检查IK前的跳变
    # 第2行：IK输出的arm joints（7个关节）- 用于检查IK后的跳变
    # 第3行：IK输入的位置（3个xyz，用于参考）
    # 第4行：所有角度对比（IK输入rotvec + IK输出joints，便于对比）
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    fig.suptitle('IK Input vs Output Trajectory Comparison - All Angles on Same Scale', fontsize=16, fontweight='bold')
    
    # ========== 第1行：IK输入的rotvec（轴角）==========
    # 左手 - IK输入（使用输入时间戳）
    ax = axes[0, 0]
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 0], c='red', s=50, alpha=0.7, marker='o', label='rotvec_x', zorder=5, edgecolors='darkred', linewidths=0.5)
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 1], c='green', s=50, alpha=0.7, marker='s', label='rotvec_y', zorder=5, edgecolors='darkgreen', linewidths=0.5)
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 2], c='blue', s=50, alpha=0.7, marker='^', label='rotvec_z', zorder=5, edgecolors='darkblue', linewidths=0.5)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 0], 'r-', linewidth=1.5, alpha=0.4)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 1], 'g-', linewidth=1.5, alpha=0.4)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 2], 'b-', linewidth=1.5, alpha=0.4)
    ax.set_ylabel('Left IK Input\nRotvec (rad)', fontsize=11)
    ax.set_title('Left Arm - IK Input Rotvec (Before IK)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Relative Time (seconds) - IK Input Timestamp', fontsize=10)
    # 设置x轴刻度，确保能看到所有数据点
    ax.tick_params(axis='x', rotation=45)
    
    # 右手 - IK输入（使用输入时间戳）
    ax = axes[0, 1]
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 0], c='red', s=50, alpha=0.7, marker='o', label='rotvec_x', zorder=5, edgecolors='darkred', linewidths=0.5)
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 1], c='green', s=50, alpha=0.7, marker='s', label='rotvec_y', zorder=5, edgecolors='darkgreen', linewidths=0.5)
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 2], c='blue', s=50, alpha=0.7, marker='^', label='rotvec_z', zorder=5, edgecolors='darkblue', linewidths=0.5)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 0], 'r-', linewidth=1.5, alpha=0.4)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 1], 'g-', linewidth=1.5, alpha=0.4)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 2], 'b-', linewidth=1.5, alpha=0.4)
    ax.set_ylabel('Right IK Input\nRotvec (rad)', fontsize=11)
    ax.set_title('Right Arm - IK Input Rotvec (Before IK)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Relative Time (seconds) - IK Input Timestamp', fontsize=10)
    # 设置x轴刻度，确保能看到所有数据点
    ax.tick_params(axis='x', rotation=45)
    
    # ========== 第2行：IK输出的arm joints（7个关节）==========
    joint_names = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']
    colors = plt.cm.tab10(np.linspace(0, 1, 7))
    
    # 左手 - IK输出（使用输出时间戳）
    ax = axes[1, 0]
    for i in range(7):
        ax.scatter(timestamps_output_relative, left_ik_output[:, i], c=[colors[i]], s=50, alpha=0.7, 
                  marker='o', label=f'Joint {joint_names[i]}', zorder=5, edgecolors='black', linewidths=0.3)
        ax.plot(timestamps_output_relative, left_ik_output[:, i], '-', color=colors[i], linewidth=1.5, alpha=0.4)
    ax.set_ylabel('Left IK Output\nArm Joint Angles (rad)', fontsize=11)
    ax.set_title('Left Arm - IK Output Joint Angles (After IK)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.set_xlabel('Relative Time (seconds) - IK Output Timestamp', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # 右手 - IK输出（使用输出时间戳）
    ax = axes[1, 1]
    for i in range(7):
        ax.scatter(timestamps_output_relative, right_ik_output[:, i], c=[colors[i]], s=50, alpha=0.7, 
                  marker='o', label=f'Joint {joint_names[i]}', zorder=5, edgecolors='black', linewidths=0.3)
        ax.plot(timestamps_output_relative, right_ik_output[:, i], '-', color=colors[i], linewidth=1.5, alpha=0.4)
    ax.set_ylabel('Right IK Output\nArm Joint Angles (rad)', fontsize=11)
    ax.set_title('Right Arm - IK Output Joint Angles (After IK)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.set_xlabel('Relative Time (seconds) - IK Output Timestamp', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # ========== 第3行：IK输入的位置（xyz，用于参考）==========
    # 左手 - IK输入位置（使用输入时间戳）
    ax = axes[2, 0]
    ax.scatter(timestamps_input_relative, left_ik_input_pos[:, 0], c='red', s=30, alpha=0.6, marker='o', label='pos_x', zorder=5)
    ax.scatter(timestamps_input_relative, left_ik_input_pos[:, 1], c='green', s=30, alpha=0.6, marker='s', label='pos_y', zorder=5)
    ax.scatter(timestamps_input_relative, left_ik_input_pos[:, 2], c='blue', s=30, alpha=0.6, marker='^', label='pos_z', zorder=5)
    ax.plot(timestamps_input_relative, left_ik_input_pos[:, 0], 'r-', linewidth=1, alpha=0.3)
    ax.plot(timestamps_input_relative, left_ik_input_pos[:, 1], 'g-', linewidth=1, alpha=0.3)
    ax.plot(timestamps_input_relative, left_ik_input_pos[:, 2], 'b-', linewidth=1, alpha=0.3)
    ax.set_ylabel('Left IK Input\nPosition (m)', fontsize=11)
    ax.set_title('Left Arm - IK Input Position (Reference)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Relative Time (seconds) - IK Input Timestamp', fontsize=10)
    
    # 右手 - IK输入位置（使用输入时间戳）
    ax = axes[2, 1]
    ax.scatter(timestamps_input_relative, right_ik_input_pos[:, 0], c='red', s=30, alpha=0.6, marker='o', label='pos_x', zorder=5)
    ax.scatter(timestamps_input_relative, right_ik_input_pos[:, 1], c='green', s=30, alpha=0.6, marker='s', label='pos_y', zorder=5)
    ax.scatter(timestamps_input_relative, right_ik_input_pos[:, 2], c='blue', s=30, alpha=0.6, marker='^', label='pos_z', zorder=5)
    ax.plot(timestamps_input_relative, right_ik_input_pos[:, 0], 'r-', linewidth=1, alpha=0.3)
    ax.plot(timestamps_input_relative, right_ik_input_pos[:, 1], 'g-', linewidth=1, alpha=0.3)
    ax.plot(timestamps_input_relative, right_ik_input_pos[:, 2], 'b-', linewidth=1, alpha=0.3)
    ax.set_ylabel('Right IK Input\nPosition (m)', fontsize=11)
    ax.set_title('Right Arm - IK Input Position (Reference)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    ax.set_xlabel('Relative Time (seconds) - IK Input Timestamp', fontsize=10)
    
    # ========== 第4行：所有角度对比（IK输入rotvec + IK输出joints）==========
    # 左手：将所有角度放在同一张图上（注意：IK输入使用输入时间戳，IK输出使用输出时间戳）
    ax = axes[3, 0]
    # IK输入的rotvec（使用输入时间戳）
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 0], c='red', s=20, alpha=0.5, marker='o', label='IK_input_rotvec_x', zorder=5)
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 1], c='green', s=20, alpha=0.5, marker='s', label='IK_input_rotvec_y', zorder=5)
    ax.scatter(timestamps_input_relative, left_ik_input_rotvec[:, 2], c='blue', s=20, alpha=0.5, marker='^', label='IK_input_rotvec_z', zorder=5)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 0], 'r-', linewidth=1, alpha=0.2)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 1], 'g-', linewidth=1, alpha=0.2)
    ax.plot(timestamps_input_relative, left_ik_input_rotvec[:, 2], 'b-', linewidth=1, alpha=0.2)
    
    # IK输出的joints（使用输出时间戳）
    colors_joints = plt.cm.tab10(np.linspace(0, 1, 7))
    markers_joints = ['x', '+', '*', 'd', 'v', '<', '>']
    for i in range(7):
        ax.scatter(timestamps_output_relative, left_ik_output[:, i], c=[colors_joints[i]], s=20, alpha=0.5, 
                  marker=markers_joints[i], label=f'IK_output_q{i+1}', zorder=5)
        ax.plot(timestamps_output_relative, left_ik_output[:, i], '-', color=colors_joints[i], linewidth=1, alpha=0.2)
    
    ax.set_ylabel('Left Arm - All Angles (rad)', fontsize=11)
    ax.set_title('Left Arm - IK Input Rotvec (input ts) + IK Output Joints (output ts)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=6, ncol=2)
    ax.set_xlabel('Relative Time (seconds) - Different Timestamps for Input/Output', fontsize=10)
    
    # 右手：将所有角度放在同一张图上（注意：IK输入使用输入时间戳，IK输出使用输出时间戳）
    ax = axes[3, 1]
    # IK输入的rotvec（使用输入时间戳）
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 0], c='red', s=20, alpha=0.5, marker='o', label='IK_input_rotvec_x', zorder=5)
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 1], c='green', s=20, alpha=0.5, marker='s', label='IK_input_rotvec_y', zorder=5)
    ax.scatter(timestamps_input_relative, right_ik_input_rotvec[:, 2], c='blue', s=20, alpha=0.5, marker='^', label='IK_input_rotvec_z', zorder=5)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 0], 'r-', linewidth=1, alpha=0.2)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 1], 'g-', linewidth=1, alpha=0.2)
    ax.plot(timestamps_input_relative, right_ik_input_rotvec[:, 2], 'b-', linewidth=1, alpha=0.2)
    
    # IK输出的joints（使用输出时间戳）
    for i in range(7):
        ax.scatter(timestamps_output_relative, right_ik_output[:, i], c=[colors_joints[i]], s=20, alpha=0.5, 
                  marker=markers_joints[i], label=f'IK_output_q{i+1}', zorder=5)
        ax.plot(timestamps_output_relative, right_ik_output[:, i], '-', color=colors_joints[i], linewidth=1, alpha=0.2)
    
    ax.set_ylabel('Right Arm - All Angles (rad)', fontsize=11)
    ax.set_title('Right Arm - IK Input Rotvec (input ts) + IK Output Joints (output ts)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=6, ncol=2)
    ax.set_xlabel('Relative Time (seconds) - Different Timestamps for Input/Output', fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = Path(log_file).parent / 'ik_trajectory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 计算并打印统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    
    print("\n左手IK输入rotvec:")
    for i, name in enumerate(['rotvec_x', 'rotvec_y', 'rotvec_z']):
        values = left_ik_input_rotvec[:, i]
        print(f"  {name}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}, "
              f"范围=[{np.min(values):.4f}, {np.max(values):.4f}], 跨度={np.max(values)-np.min(values):.4f}")
    
    print("\n左手IK输出joint angles:")
    for i, name in enumerate(joint_names):
        values = left_ik_output[:, i]
        print(f"  Joint {name}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}, "
              f"范围=[{np.min(values):.4f}, {np.max(values):.4f}], 跨度={np.max(values)-np.min(values):.4f}")
    
    print("\n右手IK输入rotvec:")
    for i, name in enumerate(['rotvec_x', 'rotvec_y', 'rotvec_z']):
        values = right_ik_input_rotvec[:, i]
        print(f"  {name}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}, "
              f"范围=[{np.min(values):.4f}, {np.max(values):.4f}], 跨度={np.max(values)-np.min(values):.4f}")
    
    print("\n右手IK输出joint angles:")
    for i, name in enumerate(joint_names):
        values = right_ik_output[:, i]
        print(f"  Joint {name}: 均值={np.mean(values):.4f}, 标准差={np.std(values):.4f}, "
              f"范围=[{np.min(values):.4f}, {np.max(values):.4f}], 跨度={np.max(values)-np.min(values):.4f}")
    
    plt.show()

if __name__ == "__main__":
    # 文件路径
    log_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260127_102656.txt"
    
    # 检查文件是否存在
    if not Path(log_file).exists():
        print(f"错误: 找不到文件 {log_file}")
        exit(1)
    
    # 可视化
    visualize_ik_trajectory(log_file)
