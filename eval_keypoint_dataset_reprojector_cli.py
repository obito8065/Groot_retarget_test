#!/usr/bin/env python3
"""
Keypoints Reprojection Visualizer for Lerobot Dataset (CLI版本)
从 lerobot 数据集的 parquet 文件中读取关键点数据，并重投影到视频上进行可视化验证

重要特性：
    1. 自动应用与数据集相同的图像处理变换（cotrain）：
       - 原始视频: 1280x800
       - Crop: (310:770, 110:1130)
       - Resize: 720x480
       - Pad: 添加宽度填充
       - Final resize: 256x256
    
    2. 3D关键点投影到2D时自动考虑图像变换：
       - 先投影到原始 1280x800 空间
       - 再应用相同的 cotrain 变换到 256x256 空间
    
    3. 输出视频为 256x256 分辨率（与数据集处理后的图像一致）

Usage:
    读取 lerobot 数据集中的 observation.state 关键点数据，并重投影到对应视频的每一帧上
    
Example:
    python eval_keypoint_dataset_reprojector_cli.py \
        --dataset-root /path/to/dataset \
        --episode 0 \
        --fps 5 \
        --radius 8

    python eval_keypoint_dataset_reprojector_cli.py \
        --dataset-root    /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000_keypoints_v3/ \
        --episode 10 \
        --chunk 0 \
        --fps 5 \
        --radius 8
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

# 关键点颜色 (BGR格式)
COLOR_LEFT_HAND = (0, 255, 0)   # 左手 - 绿色（已废弃，改用手指独立颜色）
COLOR_RIGHT_HAND = (0, 0, 255)  # 右手 - 红色（已废弃，改用手指独立颜色）
COLOR_WRIST = (255, 255, 0)     # 手腕 - 青色
COLOR_Z_AXIS = (0, 0, 255)      # Z轴方向 - 红色

# 手指独立颜色（BGR格式）- 用于区分不同手指
FINGER_COLORS = {
    'wrist': (255, 255, 0),     # 手腕 - 青色
    'thumb': (0, 0, 255),       # 拇指 - 红色
    'index': (0, 165, 255),     # 食指 - 橙色
    'middle': (0, 255, 255),    # 中指 - 黄色
    'ring': (0, 255, 0),        # 无名指 - 绿色
    'pinky': (255, 255, 0),     # 小指 - 青色
}

# 手指连线关系
HAND_CONNECTIONS = [
    (0, 1),  # wrist -> thumb
    (0, 2),  # wrist -> index
    (0, 3),  # wrist -> middle
    (0, 4),  # wrist -> ring
    (0, 5),  # wrist -> pinky
]

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


def project_3d_to_2d(points_3d, camera_intrinsics):
    """
    将3D点投影到2D图像平面，并应用图像处理变换。
    
    步骤：
    1. 投影到原始 1280x800 图像坐标
    2. 应用 cotrain 变换（crop + resize）到 256x256 空间
    
    Args:
        points_3d: (N, 3) 3D点坐标（相机坐标系）
        camera_intrinsics: 相机内参字典
    
    Returns:
        points_2d: (N, 2) 2D点坐标（256x256 图像空间）
        valid_mask: (N,) 有效点的mask
    """
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid_mask = z > 0.01
    
    # 1. 投影到原始 1280x800 图像坐标
    u_orig = np.where(valid_mask, fx * (x / z) + cx, -1)
    v_orig = np.where(valid_mask, fy * (y / z) + cy, -1)
    
    # 2. 应用 cotrain 变换到 256x256 空间
    points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)
    for i in range(len(points_3d)):
        if valid_mask[i]:
            u_final, v_final = transform_point_cotrain(u_orig[i], v_orig[i])
            points_2d[i] = [u_final, v_final]
        else:
            points_2d[i] = [-1, -1]
    
    return points_2d, valid_mask


def draw_wrist_rotation_arrow(image, wrist_pos_3d, wrist_rotvec, camera_intrinsics, 
                              color, arrow_length=0.08):
    """
    绘制wrist的Z轴方向箭头（红色）
    
    注意：此函数已经通过 project_3d_to_2d 自动应用了图像处理变换
    
    Args:
        wrist_pos_3d: (3,) wrist的3D位置（相机坐标系）
        wrist_rotvec: (3,) wrist的旋转向量（轴角表示）
        camera_intrinsics: 相机内参
        color: 颜色（未使用，统一为红色）
        arrow_length: 箭头长度（米）
    """
    from scipy.spatial.transform import Rotation as R
    
    # 将 rotvec 转换为旋转矩阵
    rot_matrix = R.from_rotvec(wrist_rotvec).as_matrix()
    
    # 取旋转矩阵的Z轴作为方向（表示手腕的"前方"）
    direction = rot_matrix @ np.array([0, 0, 1])  # Z轴方向
    
    # 计算箭头终点
    arrow_end_3d = wrist_pos_3d + direction * arrow_length
    
    # 投影到2D（已自动应用 cotrain 变换到 256x256 空间）
    points_3d = np.array([wrist_pos_3d, arrow_end_3d])
    points_2d, valid_mask = project_3d_to_2d(points_3d, camera_intrinsics)
    
    if valid_mask.all():
        # 绘制红色箭头
        start_pt = tuple(points_2d[0].astype(int))
        end_pt = tuple(points_2d[1].astype(int))
        
        # 检查点是否在图像范围内（256x256）
        h, w = image.shape[:2]
        if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
            0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
            # 统一使用红色绘制Z轴方向箭头
            cv2.arrowedLine(image, start_pt, end_pt, COLOR_Z_AXIS, 
                           2, cv2.LINE_AA, 0, 0.25)


def draw_hand_keypoints(image, keypoints_2d, valid_mask, color, label_prefix, 
                        radius, thickness, show_labels):
    """在图像上绘制手部关键点和连线（每个手指使用独立颜色）"""
    h, w = image.shape[:2]
    kp_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
    
    # 绘制连线（使用手指的颜色）
    for i, j in HAND_CONNECTIONS:
        if valid_mask[i] and valid_mask[j]:
            pt1 = tuple(map(int, keypoints_2d[i]))
            pt2 = tuple(map(int, keypoints_2d[j]))
            
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                # 使用手指的独立颜色（j是手指索引）
                finger_color = FINGER_COLORS[kp_names[j]]
                cv2.line(image, pt1, pt2, finger_color, thickness)
    
    # 绘制关键点（每个手指使用独立颜色）
    for i, (kp_name, pt, valid) in enumerate(zip(kp_names, keypoints_2d, valid_mask)):
        if not valid:
            continue
        
        pt_int = tuple(map(int, pt))
        
        if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
            # 使用每个手指的独立颜色
            point_color = FINGER_COLORS[kp_name]
            cv2.circle(image, pt_int, radius, point_color, -1)
            cv2.circle(image, pt_int, radius + 1, (255, 255, 255), 1)
            
            if show_labels:
                label = f"{label_prefix}{kp_name}"
                cv2.putText(image, label, (pt_int[0] + 8, pt_int[1] - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def parse_keypoints_from_parquet(parquet_path):
    """
    从 lerobot 数据集的 parquet 文件中解析关键点数据
    
    Args:
        parquet_path: parquet 文件路径
    
    Returns:
        keypoints_data: dict, key=(frame_idx,), value=(left_kp_3d, left_rotvec, right_kp_3d, right_rotvec)
            - left_kp_3d: (6, 3) 左手6个关键点 [wrist, thumb, index, middle, ring, pinky]
            - left_rotvec: (3,) 左手腕轴角
            - right_kp_3d: (6, 3) 右手6个关键点
            - right_rotvec: (3,) 右手腕轴角
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
        left_wrist_rotvec = left_keypoints_21[18:21]
        
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
        right_wrist_rotvec = right_keypoints_21[18:21]
        
        right_kp_3d = np.stack([
            right_wrist_xyz,
            right_thumb_tip,
            right_index_tip,
            right_middle_tip,
            right_ring_tip,
            right_pinky_tip
        ], axis=0)  # (6, 3)
        
        # 存储：key 为帧索引，value 为 (left_kp, left_rotvec, right_kp, right_rotvec)
        keypoints_data[frame_idx] = (left_kp_3d, left_wrist_rotvec, right_kp_3d, right_wrist_rotvec)
    
    return keypoints_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='从 lerobot 数据集读取关键点并重投影到视频上',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_keypoint_dataset_reprojector_cli.py \\
      --dataset-root /path/to/dataset \\
      --episode 0 \\
      --chunk 0 \\
      --fps 5 --radius 8
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
    parser.add_argument('--radius', type=int, default=5,
                       help='关键点圆圈半径（默认5）')
    parser.add_argument('--thickness', type=int, default=2,
                       help='连线粗细（默认2）')
    parser.add_argument('--no-labels', action='store_true',
                       help='不显示关键点标签')
    
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
    print("=" * 80)
    print(f"\n数据集根目录: {dataset_root}")
    print(f"Episode: {args.episode:06d} (chunk-{args.chunk:03d})")
    print(f"Parquet 文件: {parquet_path}")
    print(f"视频文件: {video_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"可视化参数: radius={args.radius}, thickness={args.thickness}, labels={not args.no_labels}")
    
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
            left_kp_3d, left_rotvec, right_kp_3d, right_rotvec = keypoints_data[frame_idx]
            
            # 投影关键点到2D（自动应用 cotrain 变换到 256x256 空间）
            left_kp_2d, left_valid = project_3d_to_2d(left_kp_3d, camera_intrinsics)
            right_kp_2d, right_valid = project_3d_to_2d(right_kp_3d, camera_intrinsics)
            
            # 绘制关键点和连线（在 256x256 图像上）
            # 注意：不渲染坐标轴，只渲染关键点和连线
            draw_hand_keypoints(vis_frame, left_kp_2d, left_valid, COLOR_LEFT_HAND, "L_",
                               args.radius, args.thickness, not args.no_labels)
            draw_hand_keypoints(vis_frame, right_kp_2d, right_valid, COLOR_RIGHT_HAND, "R_",
                               args.radius, args.thickness, not args.no_labels)
            
            # 不再绘制wrist的Z轴方向箭头（按用户要求移除）
            # draw_wrist_rotation_arrow(vis_frame, left_kp_3d[0], left_rotvec, 
            #                          camera_intrinsics, None, arrow_length=0.08)
            # draw_wrist_rotation_arrow(vis_frame, right_kp_3d[0], right_rotvec, 
            #                          camera_intrinsics, None, arrow_length=0.08)
            
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
