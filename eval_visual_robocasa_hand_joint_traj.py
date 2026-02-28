#!/usr/bin/env python3
"""
可视化RoboCasa动作数据中的左右手手指关节轨迹，并检测IK大角度变化

python eval_visual_robocasa_hand_joint_traj.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260206_142705.txt \
    --check_ik_jumps --ik_threshold 3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict


def load_robocasa_action_data(file_path):
    """
    读取RoboCasa动作数据文件
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        data: numpy数组，shape为(N, 31)，包含所有时间步的数据
        chunk_ids: numpy数组，shape为(N,)，每个时间步的chunk_id
        time_steps: numpy数组，shape为(N,)，每个时间步的t
        finger_joint_names: 手指关节名称列表
    """
    data = []
    chunk_ids = []
    time_steps = []
    finger_joint_names = ['pinky', 'ring', 'middle', 'index', 'thumb_pitch', 'thumb_yaw']
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if line.startswith('#') or not line:
                continue
            
            # 解析数据行
            parts = line.split()
            if len(parts) >= 31:  # 确保有足够的列
                chunk_id = int(float(parts[0]))
                t = int(float(parts[1]))
                row_data = [float(x) for x in parts[2:]]  # 跳过chunk_id和t
                data.append(row_data)
                chunk_ids.append(chunk_id)
                time_steps.append(t)
    
    return np.array(data), np.array(chunk_ids), np.array(time_steps), finger_joint_names


def detect_large_joint_changes(joint_data, chunk_ids, threshold_rad=3.0):
    """
    检测关节角度的大幅变化
    
    Args:
        joint_data: numpy数组，shape为(N, num_joints)，关节角度数据
        chunk_ids: numpy数组，shape为(N,)，chunk_id信息
        threshold_rad: 阈值（弧度），超过此值认为是异常跳变
        
    Returns:
        anomalies: list of dict，每个异常包含 {frame_idx, joint_idx, change, is_chunk_boundary}
    """
    anomalies = []
    
    # 对每个关节，计算相邻帧之间的变化
    for joint_idx in range(joint_data.shape[1]):
        joint_angles = joint_data[:, joint_idx]
        
        # 使用unwrap处理角度周期性，确保连续性
        joint_angles_unwrapped = np.unwrap(joint_angles)
        
        # 计算相邻帧之间的变化
        for frame_idx in range(1, len(joint_angles_unwrapped)):
            change = abs(joint_angles_unwrapped[frame_idx] - joint_angles_unwrapped[frame_idx - 1])
            
            # 检查是否超过阈值
            if change > threshold_rad:
                # 检查是否是chunk边界
                is_chunk_boundary = (chunk_ids[frame_idx] != chunk_ids[frame_idx - 1])
                
                anomalies.append({
                    'frame_idx': frame_idx,
                    'joint_idx': joint_idx,
                    'change': change,
                    'is_chunk_boundary': is_chunk_boundary,
                    'prev_chunk': chunk_ids[frame_idx - 1],
                    'curr_chunk': chunk_ids[frame_idx],
                    'prev_angle': joint_angles[frame_idx - 1],
                    'curr_angle': joint_angles[frame_idx],
                })
    
    return anomalies

def visualize_hand_joint_trajectories(data, chunk_ids, time_steps, finger_joint_names, 
                                     check_ik_jumps=False, ik_threshold=3.0, output_path=None):
    """
    可视化左右手手指关节轨迹，并可选择检测IK大角度变化
    
    Args:
        data: numpy数组，shape为(N, 29)，包含关节角度（已去掉chunk_id和t）
        chunk_ids: numpy数组，shape为(N,)，chunk_id信息
        time_steps: numpy数组，shape为(N,)，时间步信息
        finger_joint_names: 手指关节名称列表
        check_ik_jumps: 是否检测IK大角度变化
        ik_threshold: IK跳变检测阈值（弧度）
        output_path: 输出图片路径（可选）
    """
    # 使用全局时间步索引作为横坐标（0到N-1）
    global_time_steps = np.arange(len(data))
    
    # 提取左右手手指关节数据
    # L_finger_q1-6: 列索引 7-12 (原索引9-14，去掉chunk_id和t后减2)
    # R_finger_q1-6: 列索引 20-25 (原索引22-27，去掉chunk_id和t后减2)
    L_finger_data = data[:, 7:13]  # 6个左手手指关节
    R_finger_data = data[:, 20:26]  # 6个右手手指关节
    
    # 提取左右手臂关节数据（用于IK检测）
    # L_arm_q1-7: 列索引 0-6
    # R_arm_q1-7: 列索引 13-19
    L_arm_data = data[:, 0:7]  # 7个左手臂关节
    R_arm_data = data[:, 13:20]  # 7个右手臂关节
    
    # 检测IK大角度变化
    arm_anomalies = {'left': [], 'right': []}
    if check_ik_jumps:
        print("\n" + "="*80)
        print("检测IK大角度变化...")
        print("="*80)
        
        arm_anomalies['left'] = detect_large_joint_changes(L_arm_data, chunk_ids, ik_threshold)
        arm_anomalies['right'] = detect_large_joint_changes(R_arm_data, chunk_ids, ik_threshold)
        
        # 打印异常报告
        arm_joint_names = [
            'Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
            'Elbow Pitch', 'Wrist Yaw', 'Wrist Roll', 'Wrist Pitch'
        ]
        
        for side in ['left', 'right']:
            anomalies = arm_anomalies[side]
            if anomalies:
                print(f"\n{side.upper()} Arm 检测到 {len(anomalies)} 个异常跳变:")
                chunk_boundary_count = sum(1 for a in anomalies if a['is_chunk_boundary'])
                print(f"  其中 {chunk_boundary_count} 个发生在chunk边界")
                
                # 按chunk边界分组显示
                chunk_boundary_anomalies = [a for a in anomalies if a['is_chunk_boundary']]
                if chunk_boundary_anomalies:
                    print(f"\n  Chunk边界异常:")
                    for a in chunk_boundary_anomalies:
                        print(f"    Frame {a['frame_idx']}: Chunk {a['prev_chunk']}->{a['curr_chunk']}, "
                              f"Joint {arm_joint_names[a['joint_idx']]} 变化 {a['change']:.4f} rad "
                              f"({a['prev_angle']:.4f} -> {a['curr_angle']:.4f})")
                
                # 非chunk边界的异常
                non_boundary_anomalies = [a for a in anomalies if not a['is_chunk_boundary']]
                if non_boundary_anomalies:
                    print(f"\n  非Chunk边界异常:")
                    for a in non_boundary_anomalies[:10]:  # 只显示前10个
                        print(f"    Frame {a['frame_idx']}: Joint {arm_joint_names[a['joint_idx']]} "
                              f"变化 {a['change']:.4f} rad")
                    if len(non_boundary_anomalies) > 10:
                        print(f"    ... 还有 {len(non_boundary_anomalies) - 10} 个异常")
            else:
                print(f"\n{side.upper()} Arm: 未检测到异常跳变")
    
    # 创建6*2的子图（6行2列）
    fig, axes = plt.subplots(6, 2, figsize=(14, 16))
    title = 'RoboCasa Hand Joint Trajectories'
    if check_ik_jumps:
        title += f' (IK Jump Detection: threshold={ik_threshold:.2f} rad)'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 为每个手指关节绘制左右手数据
    for i in range(6):
        joint_name = finger_joint_names[i]
        
        # 左手（左列）
        ax_left = axes[i, 0]
        ax_left.plot(global_time_steps, L_finger_data[:, i], 'b-', linewidth=1.5, label='Left Hand', alpha=0.7)
        
        # 标记chunk边界
        chunk_boundaries = np.where(np.diff(chunk_ids) != 0)[0] + 1
        if len(chunk_boundaries) > 0:
            for boundary_idx in chunk_boundaries:
                ax_left.axvline(x=boundary_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax_left.set_ylabel(f'L_{joint_name}\nJoint Value', fontsize=10)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc='upper right', fontsize=8)
        if i == 0:
            ax_left.set_title('Left Hand', fontsize=12, fontweight='bold')
        if i == 5:
            ax_left.set_xlabel('Time Step', fontsize=10)
        
        # 右手（右列）
        ax_right = axes[i, 1]
        ax_right.plot(global_time_steps, R_finger_data[:, i], 'r-', linewidth=1.5, label='Right Hand', alpha=0.7)
        
        # 标记chunk边界
        if len(chunk_boundaries) > 0:
            for boundary_idx in chunk_boundaries:
                ax_right.axvline(x=boundary_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax_right.set_ylabel(f'R_{joint_name}\nJoint Value', fontsize=10)
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc='upper right', fontsize=8)
        if i == 0:
            ax_right.set_title('Right Hand', fontsize=12, fontweight='bold')
        if i == 5:
            ax_right.set_xlabel('Time Step', fontsize=10)
    
    # 如果检测了IK跳变，创建额外的图显示手臂关节
    if check_ik_jumps and (len(arm_anomalies['left']) > 0 or len(arm_anomalies['right']) > 0):
        fig2, axes2 = plt.subplots(7, 2, figsize=(16, 20))
        fig2.suptitle(f'Arm Joint Angles with IK Jump Detection (threshold={ik_threshold:.2f} rad)', 
                     fontsize=16, fontweight='bold')
        
        arm_joint_names = [
            'Shoulder Pitch', 'Shoulder Roll', 'Shoulder Yaw',
            'Elbow Pitch', 'Wrist Yaw', 'Wrist Roll', 'Wrist Pitch'
        ]
        
        for joint_idx in range(7):
            joint_name = arm_joint_names[joint_idx]
            
            # 左手
            ax_left = axes2[joint_idx, 0]
            ax_left.plot(global_time_steps, L_arm_data[:, joint_idx], 'b-', linewidth=1.5, alpha=0.7)
            
            # 标记异常点
            left_anomalies_at_joint = [a for a in arm_anomalies['left'] if a['joint_idx'] == joint_idx]
            if left_anomalies_at_joint:
                anomaly_frames = [a['frame_idx'] for a in left_anomalies_at_joint]
                anomaly_values = [L_arm_data[a['frame_idx'], joint_idx] for a in left_anomalies_at_joint]
                # chunk边界的异常用红色大点标记
                chunk_boundary_frames = [a['frame_idx'] for a in left_anomalies_at_joint if a['is_chunk_boundary']]
                chunk_boundary_values = [L_arm_data[a['frame_idx'], joint_idx] for a in left_anomalies_at_joint if a['is_chunk_boundary']]
                if chunk_boundary_frames:
                    ax_left.scatter(chunk_boundary_frames, chunk_boundary_values, 
                                  c='red', s=200, marker='X', zorder=10, 
                                  label='Chunk Boundary Jump', edgecolors='darkred', linewidths=2)
                # 非chunk边界的异常用橙色点标记
                non_boundary_frames = [a['frame_idx'] for a in left_anomalies_at_joint if not a['is_chunk_boundary']]
                non_boundary_values = [L_arm_data[a['frame_idx'], joint_idx] for a in left_anomalies_at_joint if not a['is_chunk_boundary']]
                if non_boundary_frames:
                    ax_left.scatter(non_boundary_frames, non_boundary_values, 
                                  c='orange', s=100, marker='o', zorder=9, 
                                  label='Large Jump', alpha=0.7)
            
            # 标记chunk边界
            chunk_boundaries = np.where(np.diff(chunk_ids) != 0)[0] + 1
            if len(chunk_boundaries) > 0:
                for boundary_idx in chunk_boundaries:
                    ax_left.axvline(x=boundary_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            ax_left.set_ylabel(f'Left Arm\n{joint_name}\n(rad)', fontsize=10)
            ax_left.grid(True, alpha=0.3)
            if joint_idx == 0:
                ax_left.set_title('Left Arm', fontsize=12, fontweight='bold')
                ax_left.legend(loc='upper right', fontsize=8)
            if joint_idx == 6:
                ax_left.set_xlabel('Time Step', fontsize=10)
            
            # 右手
            ax_right = axes2[joint_idx, 1]
            ax_right.plot(global_time_steps, R_arm_data[:, joint_idx], 'r-', linewidth=1.5, alpha=0.7)
            
            # 标记异常点
            right_anomalies_at_joint = [a for a in arm_anomalies['right'] if a['joint_idx'] == joint_idx]
            if right_anomalies_at_joint:
                anomaly_frames = [a['frame_idx'] for a in right_anomalies_at_joint]
                anomaly_values = [R_arm_data[a['frame_idx'], joint_idx] for a in right_anomalies_at_joint]
                # chunk边界的异常用红色大点标记
                chunk_boundary_frames = [a['frame_idx'] for a in right_anomalies_at_joint if a['is_chunk_boundary']]
                chunk_boundary_values = [R_arm_data[a['frame_idx'], joint_idx] for a in right_anomalies_at_joint if a['is_chunk_boundary']]
                if chunk_boundary_frames:
                    ax_right.scatter(chunk_boundary_frames, chunk_boundary_values, 
                                   c='red', s=200, marker='X', zorder=10, 
                                   label='Chunk Boundary Jump', edgecolors='darkred', linewidths=2)
                # 非chunk边界的异常用橙色点标记
                non_boundary_frames = [a['frame_idx'] for a in right_anomalies_at_joint if not a['is_chunk_boundary']]
                non_boundary_values = [R_arm_data[a['frame_idx'], joint_idx] for a in right_anomalies_at_joint if not a['is_chunk_boundary']]
                if non_boundary_frames:
                    ax_right.scatter(non_boundary_frames, non_boundary_values, 
                                   c='orange', s=100, marker='o', zorder=9, 
                                   label='Large Jump', alpha=0.7)
            
            # 标记chunk边界
            if len(chunk_boundaries) > 0:
                for boundary_idx in chunk_boundaries:
                    ax_right.axvline(x=boundary_idx, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            ax_right.set_ylabel(f'Right Arm\n{joint_name}\n(rad)', fontsize=10)
            ax_right.grid(True, alpha=0.3)
            if joint_idx == 0:
                ax_right.set_title('Right Arm', fontsize=12, fontweight='bold')
                ax_right.legend(loc='upper right', fontsize=8)
            if joint_idx == 6:
                ax_right.set_xlabel('Time Step', fontsize=10)
        
        plt.tight_layout()
        
        # 保存手臂关节图
        if output_path:
            arm_output_path = Path(str(output_path).replace('.png', '_arm_ik_jumps.png'))
            plt.savefig(arm_output_path, dpi=150, bbox_inches='tight')
            print(f"\n手臂关节IK跳变检测图已保存到: {arm_output_path}")
        
        plt.show()
    
    plt.tight_layout()
    
    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {output_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="可视化RoboCasa手部关节轨迹，并检测IK大角度变化")
    parser.add_argument("--action_file", type=str, 
                       default="/vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260206_142705.txt",
                       help="robocasa_action文件路径")
    parser.add_argument("--check_ik_jumps", action="store_true",
                       help="是否检测IK大角度变化")
    parser.add_argument("--ik_threshold", type=float, default=3.0,
                       help="IK跳变检测阈值（弧度），默认3.0 rad (约172度)")
    args = parser.parse_args()

    # 数据文件路径
    data_file = Path(args.action_file)
    
    if not data_file.exists():
        print(f"错误: 找不到数据文件: {data_file}")
        return
    
    print(f"正在读取数据文件: {data_file}")
    
    # 加载数据
    data, chunk_ids, time_steps, finger_joint_names = load_robocasa_action_data(data_file)
    
    print(f"成功加载 {len(data)} 个时间步的数据")
    print(f"Chunk范围: {chunk_ids.min()} - {chunk_ids.max()}")
    print(f"时间步范围: {time_steps.min()} - {time_steps.max()}")
    print(f"手指关节名称: {finger_joint_names}")
    
    if args.check_ik_jumps:
        print(f"\nIK跳变检测已启用，阈值: {args.ik_threshold:.2f} rad")
    
    # 生成输出路径
    output_path = data_file.parent / 'robocasa_hand_joint_trajectories.png'
    
    # 可视化
    visualize_hand_joint_trajectories(
        data, chunk_ids, time_steps, finger_joint_names,
        check_ik_jumps=args.check_ik_jumps,
        ik_threshold=args.ik_threshold,
        output_path=output_path
    )

if __name__ == '__main__':
    main()
