#!/usr/bin/env python3
"""
分析轴角跳变问题：检查是retarget阶段还是IK阶段导致的跳变

通过对比：
1. retarget输出的轴角（retargeted_actions.txt）
2. IK+FK输出的轴角（FK_check.txt）

如果retarget的输出有跳变，说明问题在retarget阶段（旋转矩阵转轴角时）
如果IK+FK的输出有跳变，说明问题在IK阶段
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def parse_txt_file(file_path):
    """解析txt文件，提取wrist pose数据"""
    left_wrist = []
    right_wrist = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 25:
                continue
            
            # 提取左手wrist pose (索引2-7: x, y, z, rotvec_x, rotvec_y, rotvec_z)
            left_wrist.append([
                float(parts[2]),  # L_wrist_x
                float(parts[3]),  # L_wrist_y
                float(parts[4]),  # L_wrist_z
                float(parts[5]),  # L_rotvec_x
                float(parts[6]),  # L_rotvec_y
                float(parts[7]),  # L_rotvec_z
            ])
            
            # 提取右手wrist pose (索引14-19: x, y, z, rotvec_x, rotvec_y, rotvec_z)
            right_wrist.append([
                float(parts[14]),  # R_wrist_x
                float(parts[15]),  # R_wrist_y
                float(parts[16]),  # R_wrist_z
                float(parts[17]),  # R_rotvec_x
                float(parts[18]),  # R_rotvec_y
                float(parts[19]),  # R_rotvec_z
            ])
    
    return np.array(left_wrist), np.array(right_wrist)

def detect_axisangle_jumps(rotvec_seq, threshold=1.0):
    """
    检测轴角序列中的跳变
    
    参数:
        rotvec_seq: (T, 3) 轴角序列
        threshold: 跳变阈值（弧度）
    
    返回:
        jump_indices: 跳变位置的索引列表
        jump_magnitudes: 跳变幅度列表（直接差值，实际旋转角度，是否等价表示）
    """
    if len(rotvec_seq) < 2:
        return [], []
    
    jump_indices = []
    jump_magnitudes = []
    
    for i in range(1, len(rotvec_seq)):
        # 将轴角转换为旋转矩阵
        R_prev = R.from_rotvec(rotvec_seq[i-1]).as_matrix()
        R_curr = R.from_rotvec(rotvec_seq[i]).as_matrix()
        
        # 计算相对旋转
        R_diff = R_prev.T @ R_curr
        rotvec_diff = R.from_matrix(R_diff).as_rotvec()
        angle_diff = np.linalg.norm(rotvec_diff)
        
        # 直接计算轴角的差值
        direct_diff = np.linalg.norm(rotvec_seq[i] - rotvec_seq[i-1])
        
        # 检查是否是等价表示跳变
        # 轴角 r 和 -r + 2πk 表示同一个旋转
        # 如果直接差值很大，但旋转矩阵相同（或非常接近），说明是等价表示跳变
        is_equiv_jump = False
        if direct_diff > threshold:
            # 检查旋转矩阵是否相同
            if np.allclose(R_prev, R_curr, atol=1e-3):
                is_equiv_jump = True
            # 或者实际旋转角度远小于直接差值
            elif angle_diff < direct_diff * 0.3:  # 实际旋转小于直接差值的30%
                is_equiv_jump = True
        
        if is_equiv_jump:
            jump_indices.append(i)
            jump_magnitudes.append((direct_diff, angle_diff, True))
        elif direct_diff > threshold * 2:  # 非常大的跳变（可能是真实的大旋转）
            jump_indices.append(i)
            jump_magnitudes.append((direct_diff, angle_diff, False))
    
    return jump_indices, jump_magnitudes

def analyze_axisangle_continuity(rotvec_seq, name="Sequence"):
    """
    分析轴角序列的连续性
    
    参数:
        rotvec_seq: (T, 3) 轴角序列
        name: 序列名称
    """
    print(f"\n{'='*80}")
    print(f"分析 {name}")
    print(f"{'='*80}")
    
    if len(rotvec_seq) < 2:
        print("数据点太少，无法分析")
        return
    
    # 检测跳变
    jump_indices, jump_magnitudes = detect_axisangle_jumps(rotvec_seq, threshold=2.0)
    
    print(f"总帧数: {len(rotvec_seq)}")
    print(f"检测到跳变数量: {len(jump_indices)}")
    
    if jump_indices:
        print("\n跳变详情:")
        equiv_count = 0
        real_count = 0
        for idx, (direct_diff, angle_diff, is_equiv) in zip(jump_indices[:20], jump_magnitudes[:20]):
            jump_type = "等价表示跳变" if is_equiv else "真实大旋转"
            if is_equiv:
                equiv_count += 1
            else:
                real_count += 1
            print(f"  帧 {idx}: {jump_type}")
            print(f"    直接差值={direct_diff:.3f} rad, 实际旋转角度={angle_diff:.3f} rad")
            print(f"    前: {rotvec_seq[idx-1]}")
            print(f"    后: {rotvec_seq[idx]}")
            
            # 检查是否是等价表示
            R_prev = R.from_rotvec(rotvec_seq[idx-1]).as_matrix()
            R_curr = R.from_rotvec(rotvec_seq[idx]).as_matrix()
            if np.allclose(R_prev, R_curr, atol=1e-3):
                print(f"    ✓ 旋转矩阵相同（等价表示）")
            print()
        
        print(f"  等价表示跳变: {equiv_count} 个")
        print(f"  真实大旋转: {real_count} 个")
    else:
        print("✓ 未检测到明显的跳变")
    
    # 计算统计信息
    direct_diffs = []
    angle_diffs = []
    for i in range(1, len(rotvec_seq)):
        direct_diff = np.linalg.norm(rotvec_seq[i] - rotvec_seq[i-1])
        R_prev = R.from_rotvec(rotvec_seq[i-1]).as_matrix()
        R_curr = R.from_rotvec(rotvec_seq[i]).as_matrix()
        R_diff = R_prev.T @ R_curr
        angle_diff = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
        direct_diffs.append(direct_diff)
        angle_diffs.append(angle_diff)
    
    print(f"\n统计信息:")
    print(f"  直接差值 - 平均: {np.mean(direct_diffs):.4f} rad, 最大: {np.max(direct_diffs):.4f} rad")
    print(f"  实际旋转角度 - 平均: {np.mean(angle_diffs):.4f} rad, 最大: {np.max(angle_diffs):.4f} rad")
    
    # 识别等价表示跳变（直接差值大但实际旋转角度小）
    equiv_jumps = []
    for i in range(1, len(rotvec_seq)):
        direct_diff = np.linalg.norm(rotvec_seq[i] - rotvec_seq[i-1])
        R_prev = R.from_rotvec(rotvec_seq[i-1]).as_matrix()
        R_curr = R.from_rotvec(rotvec_seq[i]).as_matrix()
        R_diff = R_prev.T @ R_curr
        angle_diff = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
        
        # 更宽松的条件：直接差值>1.0且实际旋转<直接差值的30%
        if direct_diff > 1.0 and angle_diff < direct_diff * 0.3:
            equiv_jumps.append(i)
    
    if equiv_jumps:
        print(f"\n⚠️  检测到 {len(equiv_jumps)} 个可能的等价表示跳变（直接差值>1.0且实际旋转<直接差值的30%）")
        print(f"   跳变位置: {equiv_jumps[:10]}...")
    else:
        print(f"\n✓ 未检测到明显的等价表示跳变")

def main():
    # 文件路径
    retarget_file = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260126_164807.txt")
    fk_file = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/FK_check.txt")
    
    print("="*80)
    print("轴角跳变分析")
    print("="*80)
    print(f"\nRetarget文件: {retarget_file}")
    print(f"FK文件: {fk_file}")
    
    # 读取数据
    print("\n读取数据...")
    left_retarget, right_retarget = parse_txt_file(retarget_file)
    left_fk, right_fk = parse_txt_file(fk_file)
    
    # 确保长度一致
    min_len = min(len(left_retarget), len(left_fk), len(right_retarget), len(right_fk))
    left_retarget = left_retarget[:min_len]
    right_retarget = right_retarget[:min_len]
    left_fk = left_fk[:min_len]
    right_fk = right_fk[:min_len]
    
    print(f"使用数据长度: {min_len}")
    
    # 分析retarget输出的轴角
    print("\n" + "="*80)
    print("1. 分析 RETARGET 输出的轴角（retarget阶段：旋转矩阵转轴角）")
    print("="*80)
    analyze_axisangle_continuity(left_retarget[:, 3:6], "Retarget - 左手轴角")
    analyze_axisangle_continuity(right_retarget[:, 3:6], "Retarget - 右手轴角")
    
    # 分析IK+FK输出的轴角
    print("\n" + "="*80)
    print("2. 分析 IK+FK 输出的轴角（IK阶段：旋转矩阵转轴角）")
    print("="*80)
    analyze_axisangle_continuity(left_fk[:, 3:6], "IK+FK - 左手轴角")
    analyze_axisangle_continuity(right_fk[:, 3:6], "IK+FK - 右手轴角")
    
    # 对比分析
    print("\n" + "="*80)
    print("3. 对比分析")
    print("="*80)
    print("\n结论:")
    print("- 如果retarget输出有跳变，说明问题在retarget阶段（pytransform3d的compact_axis_angle_from_quaternion）")
    print("- 如果IK+FK输出有跳变，说明问题在IK阶段（scipy的as_rotvec）")
    print("- 如果两者都有跳变，需要检查哪个更严重")

if __name__ == "__main__":
    main()
