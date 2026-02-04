#!/usr/bin/env python3
"""
Keypoints Reprojection Visualizer for Lerobot Dataset (CLI版本)
从 lerobot 数据集的 parquet 文件中读取关键点数据，并重投影到视频上进行可视化验证

重要特性：
    1. 使用与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 相同的投影方式
    2. 颜色和尺寸与原始投影器保持一致：
       - 左手腕：红色 (0, 0, 255)
       - 右手腕：蓝色 (255, 0, 0)
       - 指尖：绿色 (0, 255, 0)
       - 半径：5
    3. 不绘制坐标轴和连线，只绘制点

Usage:
    读取 lerobot 数据集中的 observation.state 关键点数据，并重投影到对应视频的每一帧上
    
Example:
    python eval_keypoint_dataset_reprojector_cli.py \
        --dataset-root  /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v4 \
        --episode 20 \
        --fps 5
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

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

# 关键点颜色 (BGR格式) - 与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 保持一致
COLOR_LEFT_WRIST = (0, 0, 255)    # 左手腕 - 红色
COLOR_RIGHT_WRIST = (255, 0, 0)   # 右手腕 - 蓝色
COLOR_FINGERTIPS = (0, 255, 0)    # 指尖 - 绿色

# 关键点半径 - 与原始投影器保持一致
KEYPOINT_RADIUS = 5

# =============================================================================
# 工具函数
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
    与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 中的实现一致。
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
    绘制单个3D点的投影（与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 的 _draw_projection 方法一致）
    
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
            # 绘制点（不绘制坐标轴）
            cv2.circle(frame, (u_final, v_final), radius=KEYPOINT_RADIUS, color=color, thickness=-1)


def parse_keypoints_from_parquet(parquet_path):
    """
    从 lerobot 数据集的 parquet 文件中解析关键点数据
    
    Args:
        parquet_path: parquet 文件路径
    
    Returns:
        keypoints_data: dict, key=(frame_idx,), value=(left_kp_3d, right_kp_3d)
            - left_kp_3d: (6, 3) 左手6个关键点 [wrist, thumb, index, middle, ring, pinky]（相机坐标系）
            - right_kp_3d: (6, 3) 右手6个关键点（相机坐标系）
    """
    print(f"正在读取 parquet 文件: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    keypoints_data = {}
    total_frames = len(df)
    
    print(f"  Episode 共有 {total_frames} 帧数据")
    
    for frame_idx, row in df.iterrows():
        # 从 observation.state 中提取关键点数据
        # 格式：45维 = [left_key_points(21), right_key_points(21), waist(3)]
        # left_key_points(21) = wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        # state_45d = np.array(row['observation.state'], dtype=np.float32)
        state_45d = np.array(row['action'], dtype=np.float32)
        
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
        
        # 存储：key 为帧索引，value 为 (left_kp_3d, right_kp_3d)
        keypoints_data[frame_idx] = (left_kp_3d, right_kp_3d)
    
    return keypoints_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='从 lerobot 数据集读取关键点并重投影到视频上（使用与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 相同的投影方式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_keypoint_dataset_reprojector_cli.py \\
      --dataset-root /path/to/dataset \\
      --episode 0 \\
      --chunk 0 \\
      --fps 5
        """
    )
    
    # 必需参数
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='lerobot 数据集根目录路径')
    parser.add_argument('--episode', type=int, required=True,
                       help='要可视化的 episode 索引（例如：0 表示 episode_000000）')
    
    # 可选参数
    parser.add_argument('--chunk', type=int, default=0,
                       help='chunk 索引（默认0）')
    parser.add_argument('--video-key', type=str, default='observation.images.ego_view',
                       help='视频key（默认 ego_view）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认在数据集根目录下生成）')
    parser.add_argument('--fps', type=int, default=20,
                       help='输出视频fps（默认20，与数据集一致）')
    
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
    
    # 构建文件路径
    dataset_root = Path(args.dataset_root)
    
    # parquet 文件路径: data/chunk-{chunk:03d}/episode_{episode:06d}.parquet
    parquet_path = dataset_root / "data" / f"chunk-{args.chunk:03d}" / f"episode_{args.episode:06d}.parquet"
    
    # 视频文件路径: videos/chunk-{chunk:03d}/{video_key}/episode_{episode:06d}.mp4
    video_path = dataset_root / "videos" / f"chunk-{args.chunk:03d}" / args.video_key / f"episode_{args.episode:06d}.mp4"
    
    # 输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = dataset_root / f"episode_{args.episode:06d}_keypoints_reprojected.mp4"
    
    # 相机内参
    camera_intrinsics = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    print("=" * 80)
    print("Lerobot Dataset Keypoints Reprojection Visualizer")
    print("使用与 get_robocasa_video_hand_pose_pinocchio_reprojector.py 相同的投影方式")
    print("=" * 80)
    print(f"\n数据集根目录: {dataset_root}")
    print(f"Episode: {args.episode:06d} (chunk-{args.chunk:03d})")
    print(f"Parquet 文件: {parquet_path}")
    print(f"视频文件: {video_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"可视化参数: radius={KEYPOINT_RADIUS}, 颜色: 左手腕=红色, 右手腕=蓝色, 指尖=绿色")
    
    # 检查输入文件
    if not parquet_path.exists():
        print(f"\n❌ 错误: Parquet 文件不存在: {parquet_path}")
        return
    
    if not video_path.exists():
        print(f"\n❌ 错误: 视频文件不存在: {video_path}")
        return
    
    # 1. 从 parquet 文件解析关键点数据
    print("\n[1/3] 从 Parquet 文件解析关键点数据...")
    keypoints_data = parse_keypoints_from_parquet(parquet_path)
    print(f"✓ 加载了 {len(keypoints_data)} 帧的关键点数据")
    
    # 2. 打开输入视频
    print("\n[2/3] 打开输入视频...")
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
    
    # 检查视频帧数和关键点数据是否匹配
    if total_frames != len(keypoints_data):
        print(f"⚠️  警告: 视频帧数({total_frames})与关键点数据({len(keypoints_data)})不匹配")
        print(f"   将使用最小值进行处理")
        total_frames = min(total_frames, len(keypoints_data))
    
    # 使用数据集的原始 FPS
    actual_output_fps = args.fps
    print(f"✓ 输出FPS: {actual_output_fps}")
    print(f"✓ 输出分辨率: 256x256（经过 cotrain 处理）")
    
    # 3. 创建输出视频（256x256 分辨率）
    print("\n[3/3] 处理视频并重投影关键点...")
    output_resolution = (256, 256)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, actual_output_fps, output_resolution)
    
    if not out.isOpened():
        print(f"❌ 错误: 无法创建输出视频文件")
        cap.release()
        return
    
    # 处理每一帧
    frame_idx = 0
    written_frames = 0
    
    pbar = tqdm(total=total_frames, desc="处理进度")
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ====================================================================
        # 重要：应用与数据集处理相同的图像变换（crop + resize）
        # 将原始 1280x800 图像处理为 256x256
        # ====================================================================
        try:
            processed_frame = process_img_cotrain(frame)
        except ValueError as e:
            # 如果图像已经是 256x256，直接使用
            print(f"⚠️  帧 {frame_idx}: {e}")
            processed_frame = frame
        
        vis_frame = processed_frame.copy()
        
        # 使用帧索引直接访问关键点数据
        if frame_idx in keypoints_data:
            # 获取当前帧的关键点数据（相机坐标系）
            left_kp_3d, right_kp_3d = keypoints_data[frame_idx]
            
            # 绘制左手关键点
            # 左手腕（红色）
            draw_projection_point(vis_frame, left_kp_3d[0], camera_intrinsics, COLOR_LEFT_WRIST)
            # 左手指尖（绿色）
            for i in range(1, 6):  # thumb, index, middle, ring, pinky
                draw_projection_point(vis_frame, left_kp_3d[i], camera_intrinsics, COLOR_FINGERTIPS)
            
            # 绘制右手关键点
            # 右手腕（蓝色）
            draw_projection_point(vis_frame, right_kp_3d[0], camera_intrinsics, COLOR_RIGHT_WRIST)
            # 右手指尖（绿色）
            for i in range(1, 6):  # thumb, index, middle, ring, pinky
                draw_projection_point(vis_frame, right_kp_3d[i], camera_intrinsics, COLOR_FINGERTIPS)
            
            # 添加帧信息（字体大小调整为适应 256x256）
            info_text = f"Ep {args.episode:06d} | F {frame_idx}"
            cv2.putText(vis_frame, info_text, (5, 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # 如果没有关键点数据，显示警告
            cv2.putText(vis_frame, f"No keypoints for Frame {frame_idx}",
                       (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        out.write(vis_frame)
        written_frames += 1
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ 处理完成!")
    print(f"  Episode: {args.episode:06d}")
    print(f"  输入帧数: {total_frames} ({input_fps:.2f} fps)")
    print(f"  输出帧数: {written_frames} ({actual_output_fps} fps)")
    print(f"  关键点数据: {len(keypoints_data)} 帧")
    print(f"  输出文件: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()