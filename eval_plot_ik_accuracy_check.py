#!/usr/bin/env python3
"""
IK准确性分析脚本

功能：
1. 读取 retargeted_actions.txt（IK前的wrist pose）
2. 读取 FK_check.txt（IK→FK后的wrist pose）
3. 计算并可视化位置误差和旋转误差

生成图表（4行2列）：
   【第0-2行】位置误差 (X, Y, Z)
   - 左列：左手的XYZ三个维度的误差（单位：mm）
   - 右列：右手的XYZ三个维度的误差（单位：mm）
   
   【第3行】旋转误差（真实角度差）
   - 左列：左手的旋转矩阵角度差（单位：度）
   - 右列：右手的旋转矩阵角度差（单位：度）

关键改进：
- 不再对比原始轴角值（避免等价表示问题）
- 直接计算旋转矩阵的真实角度差
- 位置误差单位为毫米（mm），更直观

用途：
- 准确评估IK的位置精度和旋转精度
- 避免轴角等价表示导致的误判
- 判断标准：位置误差<1-5mm，旋转误差<0.5-2°
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def compute_rotation_error_deg(rotvec1, rotvec2):
    """
    计算两个轴角表示的旋转之间的真实角度差（度）
    这个值不受轴角等价表示的影响
    """
    # 转换为旋转矩阵
    R1 = R.from_rotvec(rotvec1).as_matrix()
    R2 = R.from_rotvec(rotvec2).as_matrix()
    
    # 计算相对旋转: R_error = R1^T @ R2
    R_error = R1.T @ R2
    
    # 从旋转矩阵的trace计算旋转角度
    # trace(R) = 1 + 2*cos(θ)
    trace = np.trace(R_error)
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.rad2deg(angle_rad)
    
    return angle_deg

# 在你的load_wrist_poses函数中添加旋转矩阵误差计算
def analyze_ik_accuracy(retargeted_file, fk_check_file):
    """分析IK准确性，使用旋转矩阵误差"""
    
    # 加载数据
    l_xyz_before, l_rotvec_before, _, r_xyz_before, r_rotvec_before, _ = \
        load_wrist_poses(retargeted_file)
    l_xyz_after, l_rotvec_after, _, r_xyz_after, r_rotvec_after, _ = \
        load_wrist_poses(fk_check_file)
    
    n_frames = len(l_xyz_before)
    
    # 计算位置误差（欧氏距离）
    left_pos_errors = np.linalg.norm(l_xyz_before - l_xyz_after, axis=1)
    right_pos_errors = np.linalg.norm(r_xyz_before - r_xyz_after, axis=1)
    
    # 计算旋转误差（真实角度差）
    left_rot_errors = np.array([
        compute_rotation_error_deg(l_rotvec_before[i], l_rotvec_after[i])
        for i in range(n_frames)
    ])
    right_rot_errors = np.array([
        compute_rotation_error_deg(r_rotvec_before[i], r_rotvec_after[i])
        for i in range(n_frames)
    ])
    
    # 计算轴角欧氏误差（可能很大，但不代表IK不准）
    left_axisangle_euclidean = np.linalg.norm(l_rotvec_before - l_rotvec_after, axis=1)
    right_axisangle_euclidean = np.linalg.norm(r_rotvec_before - r_rotvec_after, axis=1)
    
    print("="*60)
    print("IK准确性分析报告")
    print("="*60)
    
    print("\n【左手】")
    print(f"位置误差:")
    print(f"  平均: {np.mean(left_pos_errors)*1000:.4f} mm")
    print(f"  最大: {np.max(left_pos_errors)*1000:.4f} mm")
    print(f"  标准差: {np.std(left_pos_errors)*1000:.4f} mm")
    
    print(f"\n旋转误差 (真实角度差):")
    print(f"  平均: {np.mean(left_rot_errors):.4f}°")
    print(f"  最大: {np.max(left_rot_errors):.4f}°")
    print(f"  标准差: {np.std(left_rot_errors):.4f}°")
    
    print(f"\n轴角欧氏距离 (可能因等价表示而大):")
    print(f"  平均: {np.mean(left_axisangle_euclidean):.4f}")
    print(f"  最大: {np.max(left_axisangle_euclidean):.4f}")
    
    print("\n" + "-"*60)
    
    print("\n【右手】")
    print(f"位置误差:")
    print(f"  平均: {np.mean(right_pos_errors)*1000:.4f} mm")
    print(f"  最大: {np.max(right_pos_errors)*1000:.4f} mm")
    print(f"  标准差: {np.std(right_pos_errors)*1000:.4f} mm")
    
    print(f"\n旋转误差 (真实角度差):")
    print(f"  平均: {np.mean(right_rot_errors):.4f}°")
    print(f"  最大: {np.max(right_rot_errors):.4f}°")
    print(f"  标准差: {np.std(right_rot_errors):.4f}°")
    
    print(f"\n轴角欧氏距离 (可能因等价表示而大):")
    print(f"  平均: {np.mean(right_axisangle_euclidean):.4f}")
    print(f"  最大: {np.max(right_axisangle_euclidean):.4f}")
    
    print("\n" + "="*60)
    print("结论:")
    print("="*60)
    
    max_pos_error = max(np.max(left_pos_errors), np.max(right_pos_errors))
    max_rot_error = max(np.max(left_rot_errors), np.max(right_rot_errors))
    
    if max_pos_error < 0.001 and max_rot_error < 0.5:
        print("✓ IK非常准确！")
        print(f"  位置误差 < 1mm，旋转误差 < 0.5°")
    elif max_pos_error < 0.005 and max_rot_error < 2.0:
        print("✓ IK基本准确")
        print(f"  位置误差 < 5mm，旋转误差 < 2°")
    else:
        print("✗ IK误差较大，需要检查:")
        print(f"  最大位置误差: {max_pos_error*1000:.2f} mm")
        print(f"  最大旋转误差: {max_rot_error:.2f}°")
    
    if np.mean(right_axisangle_euclidean) > 1.0:
        print("\n注意: 轴角欧氏距离很大，但这是正常的！")
        print("  这是因为轴角的等价表示问题，不代表IK不准确。")
        print("  判断IK准确性应该看旋转误差（真实角度差），而不是轴角欧氏距离。")
    
    return {
        'left_pos_errors': left_pos_errors,
        'right_pos_errors': right_pos_errors,
        'left_rot_errors': left_rot_errors,
        'right_rot_errors': right_rot_errors,
    }


def axis_angle_to_quaternion(rotvec):
    """
    将轴角表示转换为四元数
    
    Args:
        rotvec: (3,) array, axis-angle representation
        
    Returns:
        (4,) array, quaternion [qx, qy, qz, qw]
    """
    rotation = R.from_rotvec(rotvec)
    quat = rotation.as_quat()  # Returns [qx, qy, qz, qw]
    return quat


def load_wrist_poses(filepath):
    """
    从文件中加载手腕姿态数据
    
    Args:
        filepath: 文件路径
        
    Returns:
        left_xyz: (N, 3) 左手位置
        left_rotvec: (N, 3) 左手轴角
        left_quat: (N, 4) 左手四元数
        right_xyz: (N, 3) 右手位置
        right_rotvec: (N, 3) 右手轴角
        right_quat: (N, 4) 右手四元数
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            values = list(map(float, line.split()))
            
            # 格式: chunk_id t L_xyz(3) L_rotvec(3) L_finger(6) R_xyz(3) R_rotvec(3) R_finger(6)
            # 总共: 2 + 3 + 3 + 6 + 3 + 3 + 6 = 26
            if len(values) < 26:
                continue
            
            # 提取左手数据
            left_xyz = np.array(values[2:5])
            left_rotvec = np.array(values[5:8])
            left_quat = axis_angle_to_quaternion(left_rotvec)
            
            # 提取右手数据
            right_xyz = np.array(values[14:17])
            right_rotvec = np.array(values[17:20])
            right_quat = axis_angle_to_quaternion(right_rotvec)
            
            data.append({
                'left_xyz': left_xyz,
                'left_rotvec': left_rotvec,
                'left_quat': left_quat,
                'right_xyz': right_xyz,
                'right_rotvec': right_rotvec,
                'right_quat': right_quat
            })
    
    # 转换为数组
    left_xyz = np.array([d['left_xyz'] for d in data])
    left_rotvec = np.array([d['left_rotvec'] for d in data])
    left_quat = np.array([d['left_quat'] for d in data])
    right_xyz = np.array([d['right_xyz'] for d in data])
    right_rotvec = np.array([d['right_rotvec'] for d in data])
    right_quat = np.array([d['right_quat'] for d in data])
    
    return left_xyz, left_rotvec, left_quat, right_xyz, right_rotvec, right_quat


def plot_ik_comparison(retargeted_file, fk_check_file, output_path_error):
    """
    绘制IK准确性分析图：直接对比位置误差和旋转误差
    
    Args:
        retargeted_file: IK前的数据文件（retargeted_actions）
        fk_check_file: FK后的数据文件（FK_check）
        output_path_error: 输出图像路径（误差对比图）
    """
    # 加载数据
    print("加载IK前数据...")
    l_xyz_before, l_rotvec_before, l_quat_before, r_xyz_before, r_rotvec_before, r_quat_before = load_wrist_poses(retargeted_file)
    
    print("加载FK后数据...")
    l_xyz_after, l_rotvec_after, l_quat_after, r_xyz_after, r_rotvec_after, r_quat_after = load_wrist_poses(fk_check_file)
    
    # 确保数据长度一致
    n_frames = min(len(l_xyz_before), len(l_xyz_after))
    l_xyz_before = l_xyz_before[:n_frames]
    l_rotvec_before = l_rotvec_before[:n_frames]
    r_xyz_before = r_xyz_before[:n_frames]
    r_rotvec_before = r_rotvec_before[:n_frames]
    
    l_xyz_after = l_xyz_after[:n_frames]
    l_rotvec_after = l_rotvec_after[:n_frames]
    r_xyz_after = r_xyz_after[:n_frames]
    r_rotvec_after = r_rotvec_after[:n_frames]
    
    print(f"总共 {n_frames} 个时间步")
    
    time_steps = np.arange(n_frames)
    
    # 计算位置误差（每个维度的误差）
    left_xyz_error = l_xyz_before - l_xyz_after  # (N, 3)
    right_xyz_error = r_xyz_before - r_xyz_after  # (N, 3)
    
    # 计算旋转误差（真实角度差）
    print("计算旋转矩阵误差...")
    left_rot_errors = np.array([
        compute_rotation_error_deg(l_rotvec_before[i], l_rotvec_after[i])
        for i in range(n_frames)
    ])
    right_rot_errors = np.array([
        compute_rotation_error_deg(r_rotvec_before[i], r_rotvec_after[i])
        for i in range(n_frames)
    ])
    
    # ==================== 绘制误差图（4行2列）====================
    print("\n生成IK准确性误差图...")
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('IK Accuracy: Position & Rotation Errors', fontsize=16, fontweight='bold')
    
    dim_names = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    
    # ========== 左手位置误差（第0-2行，第0列）==========
    for i in range(3):
        ax = axes[i, 0]
        ax.plot(time_steps, left_xyz_error[:, i] * 1000, color=colors[i], linewidth=2, label=f'{dim_names[i]} error')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel(f'Left {dim_names[i]} Error (mm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if i == 0:
            ax.set_title('Left Hand Position Error', fontsize=12, fontweight='bold')
        if i == 2:
            ax.set_xlabel('Time Step', fontsize=10)
    
    # ========== 右手位置误差（第0-2行，第1列）==========
    for i in range(3):
        ax = axes[i, 1]
        ax.plot(time_steps, right_xyz_error[:, i] * 1000, color=colors[i], linewidth=2, label=f'{dim_names[i]} error')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel(f'Right {dim_names[i]} Error (mm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if i == 0:
            ax.set_title('Right Hand Position Error', fontsize=12, fontweight='bold')
        if i == 2:
            ax.set_xlabel('Time Step', fontsize=10)
    
    # ========== 左手旋转误差（第3行，第0列）==========
    ax = axes[3, 0]
    ax.plot(time_steps, left_rot_errors, 'purple', linewidth=2, label='Rotation error')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='0.5° threshold')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='1.0° threshold')
    ax.set_ylabel('Left Rotation Error (degrees)', fontsize=10)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_title('Left Hand Rotation Error (Angle Difference)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # ========== 右手旋转误差（第3行，第1列）==========
    ax = axes[3, 1]
    ax.plot(time_steps, right_rot_errors, 'purple', linewidth=2, label='Rotation error')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='0.5° threshold')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1, label='1.0° threshold')
    ax.set_ylabel('Right Rotation Error (degrees)', fontsize=10)
    ax.set_xlabel('Time Step', fontsize=10)
    ax.set_title('Right Hand Rotation Error (Angle Difference)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path_error, dpi=150, bbox_inches='tight')
    print(f"误差图已保存到: {output_path_error}")
    plt.close(fig)
    
    # 计算并打印误差统计
    print("\n" + "="*60)
    print("IK准确性分析报告")
    print("="*60)
    
    print("\n【左手】")
    print(f"位置误差 (XYZ):")
    print(f"  X轴 - 平均: {np.mean(np.abs(left_xyz_error[:, 0]))*1000:.4f} mm, 最大: {np.max(np.abs(left_xyz_error[:, 0]))*1000:.4f} mm")
    print(f"  Y轴 - 平均: {np.mean(np.abs(left_xyz_error[:, 1]))*1000:.4f} mm, 最大: {np.max(np.abs(left_xyz_error[:, 1]))*1000:.4f} mm")
    print(f"  Z轴 - 平均: {np.mean(np.abs(left_xyz_error[:, 2]))*1000:.4f} mm, 最大: {np.max(np.abs(left_xyz_error[:, 2]))*1000:.4f} mm")
    print(f"  欧氏距离 - 平均: {np.mean(np.linalg.norm(left_xyz_error, axis=1))*1000:.4f} mm, 最大: {np.max(np.linalg.norm(left_xyz_error, axis=1))*1000:.4f} mm")
    
    print(f"\n旋转误差 (真实角度差):")
    print(f"  平均: {np.mean(left_rot_errors):.4f}°")
    print(f"  最大: {np.max(left_rot_errors):.4f}°")
    print(f"  标准差: {np.std(left_rot_errors):.4f}°")
    
    print("\n" + "-"*60)
    
    print("\n【右手】")
    print(f"位置误差 (XYZ):")
    print(f"  X轴 - 平均: {np.mean(np.abs(right_xyz_error[:, 0]))*1000:.4f} mm, 最大: {np.max(np.abs(right_xyz_error[:, 0]))*1000:.4f} mm")
    print(f"  Y轴 - 平均: {np.mean(np.abs(right_xyz_error[:, 1]))*1000:.4f} mm, 最大: {np.max(np.abs(right_xyz_error[:, 1]))*1000:.4f} mm")
    print(f"  Z轴 - 平均: {np.mean(np.abs(right_xyz_error[:, 2]))*1000:.4f} mm, 最大: {np.max(np.abs(right_xyz_error[:, 2]))*1000:.4f} mm")
    print(f"  欧氏距离 - 平均: {np.mean(np.linalg.norm(right_xyz_error, axis=1))*1000:.4f} mm, 最大: {np.max(np.linalg.norm(right_xyz_error, axis=1))*1000:.4f} mm")
    
    print(f"\n旋转误差 (真实角度差):")
    print(f"  平均: {np.mean(right_rot_errors):.4f}°")
    print(f"  最大: {np.max(right_rot_errors):.4f}°")
    print(f"  标准差: {np.std(right_rot_errors):.4f}°")
    
    print("\n" + "="*60)
    print("结论:")
    print("="*60)
    
    max_pos_error = max(
        np.max(np.linalg.norm(left_xyz_error, axis=1)),
        np.max(np.linalg.norm(right_xyz_error, axis=1))
    )
    max_rot_error = max(np.max(left_rot_errors), np.max(right_rot_errors))
    
    if max_pos_error < 0.001 and max_rot_error < 0.5:
        print("✓ IK非常准确！")
        print(f"  位置误差 < 1mm，旋转误差 < 0.5°")
    elif max_pos_error < 0.005 and max_rot_error < 2.0:
        print("✓ IK基本准确")
        print(f"  位置误差 < 5mm，旋转误差 < 2°")
    else:
        print("✗ IK误差较大，需要检查:")
        print(f"  最大位置误差: {max_pos_error*1000:.2f} mm")
        print(f"  最大旋转误差: {max_rot_error:.2f}°")


if __name__ == "__main__":
    # 文件路径
    retargeted_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260121_203035.txt"
    fk_check_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/FK_check.txt"
    output_path_error = "/vla/users/lijiayi/code/groot_retarget/output_video_record/ik_accuracy_error.png"
    
    # 绘制误差对比图
    plot_ik_comparison(retargeted_file, fk_check_file, output_path_error)
    
    # 同时调用分析函数打印详细报告
    print("\n" + "="*60)
    print("运行完整的IK准确性分析...")
    print("="*60)
    analyze_ik_accuracy(retargeted_file, fk_check_file)
