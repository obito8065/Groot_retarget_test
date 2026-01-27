#!/usr/bin/env python3
"""
对比 predicted keypoints 和 retargeted actions 中的手腕轨迹

输入：
- predicted_keypoints.txt: 模型预测的关键点（包含wrist xyz和rotvec）
- retargeted_actions.txt: retarget后的动作（包含wrist xyz和rotvec）

输出：
- 6行2列的子图，显示左右手腕各6个维度的轨迹对比

命令：
python eval_check_traj.py \
    --pred-file /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260126_164806.txt \
    --retarget-file /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260126_164807.txt \
    --output /vla/users/lijiayi/code/groot_retarget/output_video_record/wrist_trajectory_pred_vs_retarget.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def parse_predicted_keypoints(file_path):
    """
    解析 predicted_keypoints.txt 文件
    
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
            if len(parts) < 43:  # 至少需要43个字段
                continue
            
            # frame_id(0), t(1)
            # 左手腕: L_wrist_xyz(2-4), L_wrist_rotvec(19-21)
            left_wrist.append([
                float(parts[2]),   # L_wrist_x
                float(parts[3]),   # L_wrist_y
                float(parts[4]),   # L_wrist_z
                float(parts[19]),  # L_wrist_rotvec_x
                float(parts[20]),  # L_wrist_rotvec_y
                float(parts[21]),  # L_wrist_rotvec_z
            ])
            
            # 右手腕: R_wrist_xyz(22-24), R_wrist_rotvec(40-42)
            right_wrist.append([
                float(parts[22]),  # R_wrist_x
                float(parts[23]),  # R_wrist_y
                float(parts[24]),  # R_wrist_z
                float(parts[40]),  # R_wrist_rotvec_x
                float(parts[41]),  # R_wrist_rotvec_y
                float(parts[42]),  # R_wrist_rotvec_z
            ])
    
    return np.array(left_wrist), np.array(right_wrist)


def parse_retargeted_actions(file_path):
    """
    解析 retargeted_actions.txt 文件
    
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
            if len(parts) < 26:  # 至少需要26个字段
                continue
            
            # chunk_id(0), t(1)
            # 左手腕: L_wrist_xyz(2-4), L_rotvec(5-7)
            left_wrist.append([
                float(parts[2]),  # L_wrist_x
                float(parts[3]),  # L_wrist_y
                float(parts[4]),  # L_wrist_z
                float(parts[5]),  # L_rotvec_x
                float(parts[6]),  # L_rotvec_y
                float(parts[7]),  # L_rotvec_z
            ])
            
            # 右手腕: R_wrist_xyz(14-16), R_rotvec(17-19)
            right_wrist.append([
                float(parts[14]),  # R_wrist_x
                float(parts[15]),  # R_wrist_y
                float(parts[16]),  # R_wrist_z
                float(parts[17]),  # R_rotvec_x
                float(parts[18]),  # R_rotvec_y
                float(parts[19]),  # R_rotvec_z
            ])
    
    return np.array(left_wrist), np.array(right_wrist)


def visualize_comparison(pred_file, retarget_file, output_path=None):
    """
    可视化对比两个文件的手腕轨迹
    
    Args:
        pred_file: predicted_keypoints.txt 文件路径
        retarget_file: retargeted_actions.txt 文件路径
        output_path: 输出图片路径（可选）
    """
    # 读取数据
    print(f"读取 predicted keypoints 文件: {pred_file}")
    left_pred, right_pred = parse_predicted_keypoints(pred_file)
    print(f"  Predicted: 左手 {left_pred.shape}, 右手 {right_pred.shape}")
    
    print(f"读取 retargeted actions 文件: {retarget_file}")
    left_retarget, right_retarget = parse_retargeted_actions(retarget_file)
    print(f"  Retargeted: 左手 {left_retarget.shape}, 右手 {right_retarget.shape}")
    
    # 确保数据长度一致
    min_len = min(len(left_pred), len(left_retarget), len(right_pred), len(right_retarget))
    left_pred = left_pred[:min_len]
    right_pred = right_pred[:min_len]
    left_retarget = left_retarget[:min_len]
    right_retarget = right_retarget[:min_len]
    
    print(f"使用数据长度: {min_len}")
    
    # 创建时间轴
    time_steps = np.arange(min_len)
    
    # 维度名称
    dim_names = ['x (m)', 'y (m)', 'z (m)', 'rotvec_x (rad)', 'rotvec_y (rad)', 'rotvec_z (rad)']
    
    # 创建图形：6行2列，每行对应一个维度，每列对应一只手
    fig, axes = plt.subplots(6, 2, figsize=(16, 20))
    fig.suptitle('Wrist Pose Trajectory: Predicted Keypoints vs Retargeted Actions', 
                 fontsize=16, fontweight='bold')
    
    # 绘制每个维度的对比
    for dim_idx in range(6):
        # 左手
        ax_left = axes[dim_idx, 0]
        ax_left.plot(time_steps, left_pred[:, dim_idx], 'b-', label='Predicted', linewidth=2, alpha=0.7)
        ax_left.plot(time_steps, left_retarget[:, dim_idx], 'r--', label='Retargeted', linewidth=2, alpha=0.7)
        ax_left.set_ylabel(f'Left Wrist {dim_names[dim_idx]}', fontsize=11)
        ax_left.grid(True, alpha=0.3)
        ax_left.legend(loc='best')
        if dim_idx == 0:
            ax_left.set_title('Left Wrist', fontsize=12, fontweight='bold')
        if dim_idx == 5:
            ax_left.set_xlabel('Time Step', fontsize=11)
        
        # 右手
        ax_right = axes[dim_idx, 1]
        ax_right.plot(time_steps, right_pred[:, dim_idx], 'b-', label='Predicted', linewidth=2, alpha=0.7)
        ax_right.plot(time_steps, right_retarget[:, dim_idx], 'r--', label='Retargeted', linewidth=2, alpha=0.7)
        ax_right.set_ylabel(f'Right Wrist {dim_names[dim_idx]}', fontsize=11)
        ax_right.grid(True, alpha=0.3)
        ax_right.legend(loc='best')
        if dim_idx == 0:
            ax_right.set_title('Right Wrist', fontsize=12, fontweight='bold')
        if dim_idx == 5:
            ax_right.set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    
    # 保存图片
    if output_path is None:
        output_path = Path(pred_file).parent / 'wrist_trajectory_pred_vs_retarget.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图片已保存到: {output_path}")
    
    # 计算并打印统计信息
    print("\n" + "="*80)
    print("统计信息（Predicted vs Retargeted）:")
    print("="*80)
    
    for dim_idx, dim_name in enumerate(dim_names):
        print(f"\n{dim_name}:")
        # 左手
        left_diff = np.abs(left_pred[:, dim_idx] - left_retarget[:, dim_idx])
        left_mean_diff = np.mean(left_diff)
        left_max_diff = np.max(left_diff)
        left_std_diff = np.std(left_diff)
        print(f"  左手 - 平均差异: {left_mean_diff:.6f}, 最大差异: {left_max_diff:.6f}, 标准差: {left_std_diff:.6f}")
        
        # 右手
        right_diff = np.abs(right_pred[:, dim_idx] - right_retarget[:, dim_idx])
        right_mean_diff = np.mean(right_diff)
        right_max_diff = np.max(right_diff)
        right_std_diff = np.std(right_diff)
        print(f"  右手 - 平均差异: {right_mean_diff:.6f}, 最大差异: {right_max_diff:.6f}, 标准差: {right_std_diff:.6f}")
    
    # 计算总体误差
    print("\n" + "="*80)
    print("总体误差:")
    print("="*80)
    
    # 位置误差 (xyz)
    left_pos_error = np.linalg.norm(left_pred[:, :3] - left_retarget[:, :3], axis=1)
    right_pos_error = np.linalg.norm(right_pred[:, :3] - right_retarget[:, :3], axis=1)
    print(f"\n位置误差 (L2 norm):")
    print(f"  左手 - 平均: {np.mean(left_pos_error):.6f} m, 最大: {np.max(left_pos_error):.6f} m")
    print(f"  右手 - 平均: {np.mean(right_pos_error):.6f} m, 最大: {np.max(right_pos_error):.6f} m")
    
    # 旋转误差 (rotvec)
    left_rot_error = np.linalg.norm(left_pred[:, 3:] - left_retarget[:, 3:], axis=1)
    right_rot_error = np.linalg.norm(right_pred[:, 3:] - right_retarget[:, 3:], axis=1)
    print(f"\n旋转误差 (L2 norm):")
    print(f"  左手 - 平均: {np.mean(left_rot_error):.6f} rad ({np.rad2deg(np.mean(left_rot_error)):.2f}°), "
          f"最大: {np.max(left_rot_error):.6f} rad ({np.rad2deg(np.max(left_rot_error)):.2f}°)")
    print(f"  右手 - 平均: {np.mean(right_rot_error):.6f} rad ({np.rad2deg(np.mean(right_rot_error)):.2f}°), "
          f"最大: {np.max(right_rot_error):.6f} rad ({np.rad2deg(np.max(right_rot_error)):.2f}°)")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比 predicted keypoints 和 retargeted actions 中的手腕轨迹")
    parser.add_argument("--pred-file", type=str, 
                        default="/vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260122_175648.txt",
                        help="predicted_keypoints.txt 文件路径")
    parser.add_argument("--retarget-file", type=str,
                        default="/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260122_175649.txt",
                        help="retargeted_actions.txt 文件路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出图片路径（默认保存在 pred-file 同目录下）")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.pred_file).exists():
        print(f"错误: 找不到文件 {args.pred_file}")
        exit(1)
    if not Path(args.retarget_file).exists():
        print(f"错误: 找不到文件 {args.retarget_file}")
        exit(1)
    
    # 可视化
    visualize_comparison(args.pred_file, args.retarget_file, args.output)
