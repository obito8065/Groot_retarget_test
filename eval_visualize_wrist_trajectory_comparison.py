#!/usr/bin/env python3
"""
可视化对比 retarget 和 IK+FK 两种方法的 wrist pose 轨迹差异

输入：
- retargeted_actions_20260121_174909.txt: retarget 的结果
- FK_check.txt: IK+FK 的结果

输出：
- 6*2=12 个子图，显示左右手各6个维度的轨迹对比
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_txt_file(file_path):
    """
    解析txt文件，提取wrist pose数据
    
    返回：
        left_wrist: (T, 6) - [x, y, z, rotvec_x, rotvec_y, rotvec_z]
        right_wrist: (T, 6) - [x, y, z, rotvec_x, rotvec_y, rotvec_z]
    """
    left_wrist = []
    right_wrist = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释和空行
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # 解析数据行
            parts = line.strip().split()
            if len(parts) < 25:  # 至少需要25个字段（chunk_id, t, 6*2 wrist + 6*2 finger）
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

def visualize_comparison(retarget_file, fk_file, output_path=None):
    """
    可视化对比两个文件的wrist pose轨迹
    
    Args:
        retarget_file: retarget结果的txt文件路径
        fk_file: FK结果的txt文件路径
        output_path: 输出图片路径（可选）
    """
    # 读取数据
    print(f"读取 retarget 文件: {retarget_file}")
    left_retarget, right_retarget = parse_txt_file(retarget_file)
    print(f"  Retarget: 左手 {left_retarget.shape}, 右手 {right_retarget.shape}")
    
    print(f"读取 FK 文件: {fk_file}")
    left_fk, right_fk = parse_txt_file(fk_file)
    print(f"  FK: 左手 {left_fk.shape}, 右手 {right_fk.shape}")
    
    # 确保数据长度一致
    min_len = min(len(left_retarget), len(left_fk), len(right_retarget), len(right_fk))
    left_retarget = left_retarget[:min_len]
    right_retarget = right_retarget[:min_len]
    left_fk = left_fk[:min_len]
    right_fk = right_fk[:min_len]
    
    print(f"使用数据长度: {min_len}")
    
    # 创建时间轴
    time_steps = np.arange(min_len)
    
    # 维度名称
    dim_names = ['x (m)', 'y (m)', 'z (m)', 'rotvec_x (rad)', 'rotvec_y (rad)', 'rotvec_z (rad)']
    
    # 创建图形：6行2列，每行对应一个维度，每列对应一只手
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))
    fig.suptitle('Wrist Pose Trajectory Comparison: Retarget vs IK+FK', fontsize=16, fontweight='bold')
    
    # 绘制每个维度的对比（使用散点图）
    for dim_idx in range(6):
        # 左手
        ax_left = axes[dim_idx, 0]
        ax_left.scatter(time_steps, left_retarget[:, dim_idx], c='blue', label='Retarget', s=30, alpha=0.6, marker='o')
        ax_left.scatter(time_steps, left_fk[:, dim_idx], c='red', label='IK+FK', s=30, alpha=0.6, marker='x')
        # 添加连线以便更好地追踪轨迹
        ax_left.plot(time_steps, left_retarget[:, dim_idx], 'b-', linewidth=1, alpha=0.3)
        ax_left.plot(time_steps, left_fk[:, dim_idx], 'r--', linewidth=1, alpha=0.3)
        ax_left.set_ylabel(f'Left Hand {dim_names[dim_idx]}', fontsize=11)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc='best')
        if dim_idx == 0:
            ax_left.set_title('Left Hand', fontsize=12, fontweight='bold')
        if dim_idx == 5:
            ax_left.set_xlabel('Time Step', fontsize=11)
        
        # 右手
        ax_right = axes[dim_idx, 1]
        ax_right.scatter(time_steps, right_retarget[:, dim_idx], c='blue', label='Retarget', s=30, alpha=0.6, marker='o')
        ax_right.scatter(time_steps, right_fk[:, dim_idx], c='red', label='IK+FK', s=30, alpha=0.6, marker='x')
        # 添加连线以便更好地追踪轨迹
        ax_right.plot(time_steps, right_retarget[:, dim_idx], 'b-', linewidth=1, alpha=0.3)
        ax_right.plot(time_steps, right_fk[:, dim_idx], 'r--', linewidth=1, alpha=0.3)
        ax_right.set_ylabel(f'Right Hand {dim_names[dim_idx]}', fontsize=11)
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc='best')
        if dim_idx == 0:
            ax_right.set_title('Right Hand', fontsize=12, fontweight='bold')
        if dim_idx == 5:
            ax_right.set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = Path(retarget_file).parent / 'wrist_trajectory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 计算并打印统计信息
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    
    for dim_idx, dim_name in enumerate(dim_names):
        print(f"\n{dim_name}:")
        # 左手
        left_diff = np.abs(left_retarget[:, dim_idx] - left_fk[:, dim_idx])
        left_mean_diff = np.mean(left_diff)
        left_max_diff = np.max(left_diff)
        print(f"  左手 - 平均差异: {left_mean_diff:.6f}, 最大差异: {left_max_diff:.6f}")
        
        # 右手
        right_diff = np.abs(right_retarget[:, dim_idx] - right_fk[:, dim_idx])
        right_mean_diff = np.mean(right_diff)
        right_max_diff = np.max(right_diff)
        print(f"  右手 - 平均差异: {right_mean_diff:.6f}, 最大差异: {right_max_diff:.6f}")
    
    plt.show()

if __name__ == "__main__":
    # 文件路径
    base_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
    retarget_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260128_202351.txt"
    fk_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/FK_check.txt"
    
    # 检查文件是否存在
    if not retarget_file:
        print(f"错误: 找不到文件 {retarget_file}")
        exit(1)
    if not fk_file:
        print(f"错误: 找不到文件 {fk_file}")
        exit(1)
    
    # 可视化
    visualize_comparison(retarget_file, fk_file)
