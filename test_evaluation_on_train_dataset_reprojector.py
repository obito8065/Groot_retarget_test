#!/usr/bin/env python3
"""
将训练集评估输出的关键点数据重投影到视频上，并可选择生成轨迹对比图

输入关键点：从数据集的parquet文件中读取（observation.state中的left_key_points和right_key_points）
输出关键点：从txt文件中读取（模型预测的action.left_key_points和action.right_key_points）

功能：
1. 视频重投影：将输入和输出关键点重投影到视频上进行可视化（视频顶部显示frame编号）
2. 轨迹对比图：生成左右手所有关键点的xyz轨迹对比图（各6行3列，18个子图，分别保存为两个PNG文件）

使用方法:
    # 生成视频重投影
    python test_evaluation_on_train_dataset_reprojector.py \
        --keypoints-txt /vla/users/lijiayi/code/groot_retarget/output_video_record/train_eval_keypoints_traj0_20260127_155341.txt\
        --dataset-root /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000_keypoints_v3 \
        --traj-id 0 \
        --chunk 0 \
        --output /vla/users/lijiayi/code/groot_retarget/output_video_record/output74ksteps.mp4 \
        --draw-input --draw-output
    
    # 生成轨迹对比图
    python test_evaluation_on_train_dataset_reprojector.py \
        --keypoints-txt /vla/users/lijiayi/code/groot_retarget/output_video_record/train_eval_keypoints_traj7_20260128_103728.txt \
        --dataset-root /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_300_keypoints_v3  \
        --traj-id 7 \
        --chunk 0 \
        --output /vla/users/lijiayi/code/groot_retarget/output_video_record/output28ksteps_traj7.mp4 \
        --draw-input --draw-output --plot-trajectory
"""

import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# 默认配置
# =============================================================================
# 相机内参 (来自GR1RetargetConfig)
DEFAULT_CAMERA_INTRINSICS = {
    'fx': 502.8689,
    'fy': 502.8689,
    'cx': 640.0,
    'cy': 400.0
}

# 关键点颜色 (BGR格式)
COLOR_LEFT_WRIST_INPUT = (0, 0, 255)      # 左手腕输入 - 红色
COLOR_RIGHT_WRIST_INPUT = (255, 0, 0)    # 右手腕输入 - 蓝色
COLOR_FINGERTIPS_INPUT = (0, 255, 0)     # 指尖输入 - 绿色

COLOR_LEFT_WRIST_OUTPUT = (128, 0, 128)   # 左手腕输出 - 紫色
COLOR_RIGHT_WRIST_OUTPUT = (128, 0, 128)  # 右手腕输出 - 紫色
COLOR_FINGERTIPS_OUTPUT = (128, 0, 128)   # 指尖输出 - 紫色

# 关键点半径
KEYPOINT_RADIUS = 2

# =============================================================================
# 工具函数（与 eval_keypoint_dataset_reprojector_cli2.py 保持一致）
# =============================================================================

def process_img_cotrain(img: np.ndarray) -> np.ndarray:
    """
    将原始图像（1280x800）处理为最终输入模型的图像（256x256）。
    这个函数与 robocasa 中的 gymnasium_groot.py 里的实现完全一致。
    """
    if not (img.shape[0] == 800 and img.shape[1] == 1280):
        if img.shape[0] == 256 and img.shape[1] == 256:
            return img
        raise ValueError(f"输入图像尺寸 {img.shape} 不符合预期的 1280x800")

    oh, ow = 256, 256
    crop = (310, 770, 110, 1130)
    img = img[crop[0] : crop[1], crop[2] : crop[3]]
    img_resized = cv2.resize(img, (720, 480), cv2.INTER_AREA)
    
    width_pad = (img_resized.shape[1] - img_resized.shape[0]) // 2
    img_pad = np.pad(
        img_resized,
        ((width_pad, width_pad), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    img_resized = cv2.resize(img_pad, (oh, ow), cv2.INTER_AREA)
    return img_resized


def transform_point_cotrain(u: float, v: float) -> tuple:
    """
    对投影到 1280x800 图像上的 2D 点应用与 process_img_cotrain 相同的几何变换。
    返回变换后在 256x256 图像中的坐标。
    """
    crop_bounds = (310, 770, 110, 1130)
    u_cropped = u - crop_bounds[2]
    v_cropped = v - crop_bounds[0]

    original_cropped_size = (1020, 460)
    resized1_size = (720, 480)
    scale_u1 = resized1_size[0] / original_cropped_size[0]
    scale_v1 = resized1_size[1] / original_cropped_size[1]
    u_resized1 = u_cropped * scale_u1
    v_resized1 = v_cropped * scale_v1

    pad_amount = (resized1_size[0] - resized1_size[1]) // 2
    u_padded = u_resized1
    v_padded = v_resized1 + pad_amount

    padded_size = (720, 720)
    final_size = (256, 256)
    scale_u2 = final_size[0] / padded_size[0]
    scale_v2 = final_size[1] / padded_size[1]
    u_final = u_padded * scale_u2
    v_final = v_padded * scale_v2

    return int(u_final), int(v_final)


def draw_projection_point(frame: np.ndarray, pos_3d: np.ndarray, 
                         camera_intrinsics: dict, color: tuple):
    """
    绘制单个3D点的投影
    
    Args:
        frame: 256x256 图像帧
        pos_3d: (3,) 3D点坐标（相机坐标系）
        camera_intrinsics: 相机内参字典
        color: (B, G, R) 颜色元组
    """
    X, Y, Z = pos_3d
    
    if Z > 0:
        # 1. 投影到虚拟的 1280x800 图像上
        u_orig = camera_intrinsics['fx'] * (X / Z) + camera_intrinsics['cx']
        v_orig = camera_intrinsics['fy'] * (Y / Z) + camera_intrinsics['cy']
        
        # 2. 将 2D 坐标点变换到 256x256 空间
        u_final, v_final = transform_point_cotrain(u_orig, v_orig)
        
        h, w = frame.shape[:2]
        
        if 0 <= u_final < w and 0 <= v_final < h:
            cv2.circle(frame, (u_final, v_final), radius=KEYPOINT_RADIUS, color=color, thickness=-1)


def parse_input_keypoints_from_parquet(parquet_path: Path):
    """
    从parquet文件中解析输入关键点数据（来自训练集的state）
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        input_keypoints_dict: dict, key=frame_idx, value={
            'left': (6, 3) 左手6个关键点 [wrist, thumb, index, middle, ring, pinky]
            'right': (6, 3) 右手6个关键点
        }
    """
    print(f"正在从parquet文件读取输入关键点: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    input_keypoints_dict = {}
    total_frames = len(df)
    
    print(f"  Episode 共有 {total_frames} 帧数据")
    
    for frame_idx, row in df.iterrows():
        # 从 observation.state 中提取关键点数据
        # 格式：45维 = [left_key_points(21), right_key_points(21), waist(3)]
        # left_key_points(21) = wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        state_45d = np.array(row['observation.state'], dtype=np.float32)
        
        # 提取左手关键点 (21维)
        left_keypoints_21 = state_45d[0:21]
        # 解析：wrist_xyz(3) + thumb_tip(3) + index_tip(3) + middle_tip(3) + ring_tip(3) + pinky_tip(3) + wrist_rotvec(3)
        left_wrist_xyz = left_keypoints_21[0:3]
        left_thumb_tip = left_keypoints_21[3:6]
        left_index_tip = left_keypoints_21[6:9]
        left_middle_tip = left_keypoints_21[9:12]
        left_ring_tip = left_keypoints_21[12:15]
        left_pinky_tip = left_keypoints_21[15:18]
        
        # 组装成 (6, 3) 的关键点数组
        left_kp_3d = np.stack([
            left_wrist_xyz,
            left_thumb_tip,
            left_index_tip,
            left_middle_tip,
            left_ring_tip,
            left_pinky_tip
        ], axis=0)  # (6, 3)
        
        # 提取右手关键点 (21维)
        right_keypoints_21 = state_45d[21:42]
        right_wrist_xyz = right_keypoints_21[0:3]
        right_thumb_tip = right_keypoints_21[3:6]
        right_index_tip = right_keypoints_21[6:9]
        right_middle_tip = right_keypoints_21[9:12]
        right_ring_tip = right_keypoints_21[12:15]
        right_pinky_tip = right_keypoints_21[15:18]
        
        right_kp_3d = np.stack([
            right_wrist_xyz,
            right_thumb_tip,
            right_index_tip,
            right_middle_tip,
            right_ring_tip,
            right_pinky_tip
        ], axis=0)  # (6, 3)
        
        # 存储：key 为帧索引，value 为左右手关键点
        input_keypoints_dict[frame_idx] = {
            'left': left_kp_3d,
            'right': right_kp_3d
        }
    
    print(f"✓ 从parquet文件解析了 {len(input_keypoints_dict)} 帧的输入关键点数据")
    return input_keypoints_dict


def visualize_right_hand_keypoints_trajectory(input_keypoints_dict: dict, output_keypoints_dict: dict, 
                                             output_path: Path, traj_id: int):
    """
    可视化右手所有关键点的xyz轨迹对比图
    
    Args:
        input_keypoints_dict: 输入关键点字典（来自parquet文件），key=frame_idx, value={'left': (6,3), 'right': (6,3)}
        output_keypoints_dict: 输出关键点字典（来自txt文件），key=global_step, value={'left': (6,3), 'right': (6,3)}
        output_path: 输出视频路径（用于生成轨迹图的保存路径）
        traj_id: 轨迹ID
    """
    # 关键点名称
    keypoint_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
    coord_names = ['x', 'y', 'z']
    
    # 获取所有可用的时间步
    input_steps = sorted(input_keypoints_dict.keys())
    output_steps = sorted(output_keypoints_dict.keys())
    
    if len(input_steps) == 0 and len(output_steps) == 0:
        print("⚠️  警告: 没有关键点数据可用于可视化")
        return
    
    # 确定时间步范围
    all_steps = sorted(set(input_steps + output_steps))
    min_step = min(all_steps)
    max_step = max(all_steps)
    
    # 准备数据：右手关键点，6个关键点 × 3个坐标轴
    # input_data: (num_steps, 6, 3)
    # output_data: (num_steps, 6, 3)
    num_steps = max_step - min_step + 1
    
    input_data = np.full((num_steps, 6, 3), np.nan)  # 用NaN填充缺失值
    output_data = np.full((num_steps, 6, 3), np.nan)
    
    # 填充输入数据
    for step in input_steps:
        idx = step - min_step
        if step in input_keypoints_dict and 'right' in input_keypoints_dict[step]:
            input_data[idx] = input_keypoints_dict[step]['right']  # (6, 3)
    
    # 填充输出数据
    for step in output_steps:
        idx = step - min_step
        if step in output_keypoints_dict and 'right' in output_keypoints_dict[step]:
            output_data[idx] = output_keypoints_dict[step]['right']  # (6, 3)
    
    # 创建时间步数组
    time_steps = np.arange(min_step, max_step + 1)
    
    # 创建6行3列的子图
    fig, axes = plt.subplots(6, 3, figsize=(18, 16))
    fig.suptitle(f'Right Hand Keypoints Trajectory Comparison (Traj {traj_id:06d})\n'
                 f'Input (Dataset) vs Output (Model Prediction)', 
                 fontsize=16, fontweight='bold')
    
    # 遍历6个关键点
    for kp_idx in range(6):
        kp_name = keypoint_names[kp_idx]
        
        # 遍历3个坐标轴
        for coord_idx in range(3):
            coord_name = coord_names[coord_idx]
            ax = axes[kp_idx, coord_idx]
            
            # 提取该关键点该坐标轴的数据
            input_coord = input_data[:, kp_idx, coord_idx]  # (num_steps,)
            output_coord = output_data[:, kp_idx, coord_idx]  # (num_steps,)
            
            # 绘制输入数据（数据集）
            valid_input_mask = ~np.isnan(input_coord)
            if np.any(valid_input_mask):
                ax.plot(time_steps[valid_input_mask], input_coord[valid_input_mask], 
                       'o-', color='blue', label='Input (Dataset)', 
                       markersize=3, linewidth=1.5, alpha=0.7)
            
            # 绘制输出数据（模型预测）
            valid_output_mask = ~np.isnan(output_coord)
            if np.any(valid_output_mask):
                ax.plot(time_steps[valid_output_mask], output_coord[valid_output_mask], 
                       's-', color='purple', label='Output (Model)', 
                       markersize=3, linewidth=1.5, alpha=0.7)
            
            # 设置标题和标签
            ax.set_title(f'{kp_name.capitalize()} {coord_name.upper()}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel(f'Position ({coord_name})', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    trajectory_plot_path = output_path.parent / f'right_hand_trajectory_traj{traj_id:06d}.png'
    plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 右手关键点轨迹图已保存: {trajectory_plot_path}")
    
    plt.close()


def visualize_left_hand_keypoints_trajectory(input_keypoints_dict: dict, output_keypoints_dict: dict, 
                                             output_path: Path, traj_id: int):
    """
    可视化左手所有关键点的xyz轨迹对比图
    
    Args:
        input_keypoints_dict: 输入关键点字典（来自parquet文件），key=frame_idx, value={'left': (6,3), 'right': (6,3)}
        output_keypoints_dict: 输出关键点字典（来自txt文件），key=global_step, value={'left': (6,3), 'right': (6,3)}
        output_path: 输出视频路径（用于生成轨迹图的保存路径）
        traj_id: 轨迹ID
    """
    # 关键点名称
    keypoint_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
    coord_names = ['x', 'y', 'z']
    
    # 获取所有可用的时间步
    input_steps = sorted(input_keypoints_dict.keys())
    output_steps = sorted(output_keypoints_dict.keys())
    
    if len(input_steps) == 0 and len(output_steps) == 0:
        print("⚠️  警告: 没有关键点数据可用于可视化")
        return
    
    # 确定时间步范围
    all_steps = sorted(set(input_steps + output_steps))
    min_step = min(all_steps)
    max_step = max(all_steps)
    
    # 准备数据：左手关键点，6个关键点 × 3个坐标轴
    # input_data: (num_steps, 6, 3)
    # output_data: (num_steps, 6, 3)
    num_steps = max_step - min_step + 1
    
    input_data = np.full((num_steps, 6, 3), np.nan)  # 用NaN填充缺失值
    output_data = np.full((num_steps, 6, 3), np.nan)
    
    # 填充输入数据
    for step in input_steps:
        idx = step - min_step
        if step in input_keypoints_dict and 'left' in input_keypoints_dict[step]:
            input_data[idx] = input_keypoints_dict[step]['left']  # (6, 3)
    
    # 填充输出数据
    for step in output_steps:
        idx = step - min_step
        if step in output_keypoints_dict and 'left' in output_keypoints_dict[step]:
            output_data[idx] = output_keypoints_dict[step]['left']  # (6, 3)
    
    # 创建时间步数组
    time_steps = np.arange(min_step, max_step + 1)
    
    # 创建6行3列的子图
    fig, axes = plt.subplots(6, 3, figsize=(18, 16))
    fig.suptitle(f'Left Hand Keypoints Trajectory Comparison (Traj {traj_id:06d})\n'
                 f'Input (Dataset) vs Output (Model Prediction)', 
                 fontsize=16, fontweight='bold')
    
    # 遍历6个关键点
    for kp_idx in range(6):
        kp_name = keypoint_names[kp_idx]
        
        # 遍历3个坐标轴
        for coord_idx in range(3):
            coord_name = coord_names[coord_idx]
            ax = axes[kp_idx, coord_idx]
            
            # 提取该关键点该坐标轴的数据
            input_coord = input_data[:, kp_idx, coord_idx]  # (num_steps,)
            output_coord = output_data[:, kp_idx, coord_idx]  # (num_steps,)
            
            # 绘制输入数据（数据集）
            valid_input_mask = ~np.isnan(input_coord)
            if np.any(valid_input_mask):
                ax.plot(time_steps[valid_input_mask], input_coord[valid_input_mask], 
                       'o-', color='blue', label='Input (Dataset)', 
                       markersize=3, linewidth=1.5, alpha=0.7)
            
            # 绘制输出数据（模型预测）
            valid_output_mask = ~np.isnan(output_coord)
            if np.any(valid_output_mask):
                ax.plot(time_steps[valid_output_mask], output_coord[valid_output_mask], 
                       's-', color='purple', label='Output (Model)', 
                       markersize=3, linewidth=1.5, alpha=0.7)
            
            # 设置标题和标签
            ax.set_title(f'{kp_name.capitalize()} {coord_name.upper()}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=10)
            ax.set_ylabel(f'Position ({coord_name})', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图表
    trajectory_plot_path = output_path.parent / f'left_hand_trajectory_traj{traj_id:06d}.png'
    plt.savefig(trajectory_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 左手关键点轨迹图已保存: {trajectory_plot_path}")
    
    plt.close()


def parse_output_keypoints_from_txt(txt_path: Path):
    """
    从txt文件中解析输出关键点数据（模型预测的）
    
    Args:
        txt_path: txt文件路径
    
    Returns:
        output_keypoints_dict: dict, key=global_step, value={
            'left': (6, 3) 左手6个关键点 [wrist, thumb, index, middle, ring, pinky]
            'right': (6, 3) 右手6个关键点
        }
    """
    print(f"正在从txt文件读取输出关键点: {txt_path}")
    output_keypoints_dict = {}
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过注释行
    data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
    
    print(f"  找到 {len(data_lines)} 行数据")
    
    for line in data_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        
        # 解析：global_step chunk_id t_in_chunk + 输入关键点(36维) + 输出关键点(36维)
        # 输入关键点：左手18维 + 右手18维 = 36维
        # 输出关键点：左手18维 + 右手18维 = 36维
        global_step = int(parts[0])
        chunk_id = int(parts[1])
        t_in_chunk = int(parts[2])
        
        # 解析输出关键点（从索引39开始，共36维）
        input_start_idx = 3
        output_start_idx = input_start_idx + 36
        output_left_18d = np.array([float(x) for x in parts[output_start_idx:output_start_idx+18]])
        output_right_18d = np.array([float(x) for x in parts[output_start_idx+18:output_start_idx+36]])
        
        # 将18维数组reshape为(6, 3)：每个关键点3个坐标值
        # 顺序：wrist(3), thumb_tip(3), index_tip(3), middle_tip(3), ring_tip(3), pinky_tip(3)
        output_left_kp = output_left_18d.reshape(6, 3)  # (6, 3)
        output_right_kp = output_right_18d.reshape(6, 3)  # (6, 3)
        
        # 存储输出关键点数据
        output_keypoints_dict[global_step] = {
            'left': output_left_kp,
            'right': output_right_kp
        }
    
    print(f"✓ 从txt文件解析了 {len(output_keypoints_dict)} 个时间步的输出关键点数据")
    return output_keypoints_dict


def main():
    parser = argparse.ArgumentParser(
        description='将训练集评估输出的关键点数据重投影到视频上\n'
                   '输入关键点：从数据集的parquet文件中读取（observation.state）\n'
                   '输出关键点：从txt文件中读取（模型预测的action）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_policy_train_dataset_reprojector.py \\
      --keypoints-txt /path/to/train_eval_keypoints_traj0_xxx.txt \\
      --dataset-root /path/to/dataset \\
      --traj-id 0 \\
      --chunk 0 \\
      --output /path/to/output.mp4 \\
      --draw-input --draw-output
        """
    )
    
    # 必需参数
    parser.add_argument('--keypoints-txt', type=str, required=True,
                       help='输出关键点txt文件路径（模型预测的）')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='数据集根目录路径')
    parser.add_argument('--traj-id', type=int, required=True,
                       help='轨迹ID（episode索引）')
    
    # 可选参数
    parser.add_argument('--chunk', type=int, default=0,
                       help='chunk索引（默认0）')
    parser.add_argument('--video-key', type=str, default='observation.images.ego_view',
                       help='视频key（默认 ego_view）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认在数据集根目录下生成）')
    parser.add_argument('--fps', type=int, default=20,
                       help='输出视频fps（默认20）')
    
    # 绘制选项
    parser.add_argument('--draw-input', action='store_true',
                       help='绘制输入关键点（从parquet文件的observation.state读取）')
    parser.add_argument('--draw-output', action='store_true',
                       help='绘制输出关键点（从txt文件的模型预测action读取）')
    parser.add_argument('--plot-trajectory', action='store_true',
                       help='生成左右手关键点xyz轨迹对比图（6行3列，18个子图，分别保存为两个PNG文件）')
    
    # 相机内参
    parser.add_argument('--fx', type=float, default=DEFAULT_CAMERA_INTRINSICS['fx'],
                       help=f'相机焦距fx（默认{DEFAULT_CAMERA_INTRINSICS["fx"]}）')
    parser.add_argument('--fy', type=float, default=DEFAULT_CAMERA_INTRINSICS['fy'],
                       help=f'相机焦距fy（默认{DEFAULT_CAMERA_INTRINSICS["fy"]}）')
    parser.add_argument('--cx', type=float, default=DEFAULT_CAMERA_INTRINSICS['cx'],
                       help=f'相机主点cx（默认{DEFAULT_CAMERA_INTRINSICS["cx"]}）')
    parser.add_argument('--cy', type=float, default=DEFAULT_CAMERA_INTRINSICS['cy'],
                       help=f'相机主点cy（默认{DEFAULT_CAMERA_INTRINSICS["cy"]}）')
    
    args = parser.parse_args()
    
    # 检查绘制选项
    if not args.draw_input and not args.draw_output and not args.plot_trajectory:
        print("⚠️  警告: 必须至少指定 --draw-input、--draw-output 或 --plot-trajectory 之一")
        print("   默认同时绘制输入和输出关键点")
        args.draw_input = True
        args.draw_output = True
    
    # 构建文件路径
    keypoints_txt = Path(args.keypoints_txt)
    dataset_root = Path(args.dataset_root)
    
    # parquet文件路径: data/chunk-{chunk:03d}/episode_{traj_id:06d}.parquet
    parquet_path = dataset_root / "data" / f"chunk-{args.chunk:03d}" / f"episode_{args.traj_id:06d}.parquet"
    
    # 视频文件路径: videos/chunk-{chunk:03d}/{video_key}/episode_{traj_id:06d}.mp4
    video_path = dataset_root / "videos" / f"chunk-{args.chunk:03d}" / args.video_key / f"episode_{args.traj_id:06d}.mp4"
    
    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_root / f"episode_{args.traj_id:06d}_policy_keypoints_reprojected.mp4"
    
    # 相机内参
    camera_intrinsics = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    print("=" * 80)
    print("训练集评估关键点重投影可视化")
    print("=" * 80)
    print(f"\n输出关键点txt文件: {keypoints_txt}")
    print(f"数据集根目录: {dataset_root}")
    print(f"轨迹ID: {args.traj_id:06d} (chunk-{args.chunk:03d})")
    print(f"Parquet文件: {parquet_path}")
    print(f"视频文件: {video_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"绘制选项: 输入={'✓' if args.draw_input else '✗'}, 输出={'✓' if args.draw_output else '✗'}, 轨迹图={'✓' if args.plot_trajectory else '✗'}")
    
    # 检查输入文件
    if not keypoints_txt.exists():
        print(f"\n❌ 错误: 输出关键点txt文件不存在: {keypoints_txt}")
        return
    
    if not parquet_path.exists():
        print(f"\n❌ 错误: Parquet文件不存在: {parquet_path}")
        return
    
    # 只有在需要处理视频时才检查视频文件
    if args.draw_input or args.draw_output:
        if not video_path.exists():
            print(f"\n❌ 错误: 视频文件不存在: {video_path}")
            return
    
    # 1. 从parquet文件解析输入关键点数据（来自训练集）
    input_keypoints_dict = {}
    if args.draw_input or args.plot_trajectory:
        print("\n[1/3] 从parquet文件解析输入关键点数据...")
        input_keypoints_dict = parse_input_keypoints_from_parquet(parquet_path)
        if len(input_keypoints_dict) == 0:
            print("⚠️  警告: 没有解析到输入关键点数据")
    
    # 2. 从txt文件解析输出关键点数据（模型预测）
    output_keypoints_dict = {}
    if args.draw_output or args.plot_trajectory:
        print("\n[2/3] 从txt文件解析输出关键点数据...")
        output_keypoints_dict = parse_output_keypoints_from_txt(keypoints_txt)
        if len(output_keypoints_dict) == 0:
            print("⚠️  警告: 没有解析到输出关键点数据")
    
    # 获取时间步范围（使用输出关键点的范围，因为输入关键点应该覆盖所有帧）
    if output_keypoints_dict:
        min_step = min(output_keypoints_dict.keys())
        max_step = max(output_keypoints_dict.keys())
        print(f"  输出关键点时间步范围: {min_step} ~ {max_step}")
    elif input_keypoints_dict:
        min_step = min(input_keypoints_dict.keys())
        max_step = max(input_keypoints_dict.keys())
        print(f"  输入关键点时间步范围: {min_step} ~ {max_step}")
    else:
        print("❌ 错误: 没有解析到任何关键点数据")
        return
    
    # 2.5. 生成轨迹对比图（如果启用）
    if args.plot_trajectory:
        print("\n[2.5/4] 生成左右手关键点xyz轨迹对比图...")
        print("  生成右手关键点轨迹图...")
        visualize_right_hand_keypoints_trajectory(
            input_keypoints_dict, output_keypoints_dict, output_path, args.traj_id
        )
        print("  生成左手关键点轨迹图...")
        visualize_left_hand_keypoints_trajectory(
            input_keypoints_dict, output_keypoints_dict, output_path, args.traj_id
        )
    
    # 如果只需要生成轨迹图，不需要处理视频，可以提前返回
    if args.plot_trajectory and not args.draw_input and not args.draw_output:
        print("\n✓ 仅生成轨迹图，跳过视频处理")
        return
    
    # 3. 打开输入视频
    print("\n[3/4] 打开输入视频...")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ 原始视频信息: {width}x{height}, {input_fps:.2f} fps, {total_frames} 帧")
    
    # 检查视频分辨率
    if width != 1280 or height != 800:
        print(f"⚠️  警告: 视频分辨率({width}x{height})不是预期的 1280x800")
        print(f"   如果视频已经是 256x256，将跳过图像处理步骤")
    
    # 确定要处理的帧数：使用关键点数据的时间步范围（处理所有时间步，不仅仅是视频帧数）
    # 确保处理所有有数据的帧，即使视频帧数不够
    total_frames_to_process = max_step + 1
    print(f"✓ 将处理 {total_frames_to_process} 帧（对应时间步 0 ~ {max_step}，覆盖所有chunk）")
    if total_frames_to_process > total_frames:
        print(f"⚠️  警告: 关键点数据有 {total_frames_to_process} 帧，但视频只有 {total_frames} 帧")
        print(f"   超出部分将使用最后一帧填充")
    
    # 4. 创建输出视频（256x256 分辨率）
    print("\n[4/4] 处理视频并重投影关键点...")
    output_resolution = (256, 256)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, args.fps, output_resolution)
    
    if not out.isOpened():
        print(f"❌ 错误: 无法创建输出视频文件")
        cap.release()
        return
    
    # 预读取最后一帧，用于填充超出视频帧数的部分
    last_frame = None
    
    # 处理每一帧
    frame_idx = 0
    written_frames = 0
    
    pbar = tqdm(total=total_frames_to_process, desc="处理进度")
    
    while frame_idx < total_frames_to_process:
        # 读取视频帧
        ret, frame = cap.read()
        
        if not ret:
            # 如果视频帧数不够，使用最后一帧
            if last_frame is None:
                print(f"⚠️  警告: 视频帧数不足，无法处理时间步 {frame_idx}")
                break
            frame = last_frame.copy()
        else:
            # 保存当前帧作为最后一帧
            last_frame = frame.copy()
        
        # 应用图像变换（1280x800 -> 256x256）
        try:
            processed_frame = process_img_cotrain(frame)
        except ValueError as e:
            # 如果图像已经是 256x256，直接使用
            processed_frame = frame
        
        vis_frame = processed_frame.copy()
        
        # 在视频顶部添加frame编号（小字）
        frame_text = f"Frame: {frame_idx}"
        cv2.putText(vis_frame, frame_text, (5, 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        has_any_keypoints = False
        
        # 绘制输入关键点（来自训练集的parquet文件）
        if args.draw_input and frame_idx in input_keypoints_dict:
            has_any_keypoints = True
            input_kp_data = input_keypoints_dict[frame_idx]
            input_left_kp = input_kp_data['left']  # (6, 3)
            input_right_kp = input_kp_data['right']  # (6, 3)
            
            # 左手腕（红色）
            draw_projection_point(vis_frame, input_left_kp[0], camera_intrinsics, COLOR_LEFT_WRIST_INPUT)
            # 左手指尖（绿色）
            for i in range(1, 6):
                draw_projection_point(vis_frame, input_left_kp[i], camera_intrinsics, COLOR_FINGERTIPS_INPUT)
            
            # 右手腕（蓝色）
            draw_projection_point(vis_frame, input_right_kp[0], camera_intrinsics, COLOR_RIGHT_WRIST_INPUT)
            # 右手指尖（绿色）
            for i in range(1, 6):
                draw_projection_point(vis_frame, input_right_kp[i], camera_intrinsics, COLOR_FINGERTIPS_INPUT)
        
        # 绘制输出关键点（模型预测，来自txt文件）
        if args.draw_output and frame_idx in output_keypoints_dict:
            has_any_keypoints = True
            output_kp_data = output_keypoints_dict[frame_idx]
            output_left_kp = output_kp_data['left']  # (6, 3)
            output_right_kp = output_kp_data['right']  # (6, 3)
            
            # 左手腕（紫色）
            draw_projection_point(vis_frame, output_left_kp[0], camera_intrinsics, COLOR_LEFT_WRIST_OUTPUT)
            # 左手指尖（紫色）
            for i in range(1, 6):
                draw_projection_point(vis_frame, output_left_kp[i], camera_intrinsics, COLOR_FINGERTIPS_OUTPUT)
            
            # 右手腕（紫色）
            draw_projection_point(vis_frame, output_right_kp[0], camera_intrinsics, COLOR_RIGHT_WRIST_OUTPUT)
            # 右手指尖（紫色）
            for i in range(1, 6):
                draw_projection_point(vis_frame, output_right_kp[i], camera_intrinsics, COLOR_FINGERTIPS_OUTPUT)
        
        # 添加其他信息（在frame编号下方）
        if has_any_keypoints:
            info_text = f"Traj {args.traj_id:06d} | Step {frame_idx}/{max_step}"
            cv2.putText(vis_frame, info_text, (5, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示chunk信息
            chunk_id = frame_idx // 16  # 假设每个chunk有16个时间步
            chunk_info = f"Chunk {chunk_id}"
            cv2.putText(vis_frame, chunk_info, (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        else:
            # 如果没有关键点数据，显示警告（在frame编号下方）
            cv2.putText(vis_frame, f"No keypoints for Step {frame_idx}",
                       (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        out.write(vis_frame)
        written_frames += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ 处理完成!")
    print(f"  轨迹ID: {args.traj_id:06d}")
    print(f"  输入视频帧数: {total_frames} ({input_fps:.2f} fps)")
    print(f"  输出视频帧数: {written_frames} ({args.fps} fps)")
    if args.draw_input:
        print(f"  输入关键点数据: {len(input_keypoints_dict)} 帧（来自parquet文件）")
    if args.draw_output:
        print(f"  输出关键点数据: {len(output_keypoints_dict)} 个时间步（来自txt文件，覆盖 {min_step} ~ {max_step}）")
    if output_keypoints_dict:
        print(f"  处理的chunk数: {(max_step + 1) // 16}")
    print(f"  输出文件: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
