#!/usr/bin/env python3
"""
Keypoints Reprojection Visualizer (CLI版本)
支持命令行参数配置

Usage:
    读取policy输出的关键点txt文件，并重投影到视频上进行可视化验证
    
Example:
    python eval_retarget_reprojector_cli.py \
        --video output_video_record/.../video.mp4 \
        --keypoints output_video_record/predicted_keypoints_20260112_110653.txt \
        --fps 5 \
        --radius 8

    python eval_retarget_reprojector_cli.py \
        --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3/58ksteps-modify5/5.mp4 \
        --keypoints /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260127_095208.txt \
        --fps 5 \
        --radius 8
"""

import cv2
import numpy as np
import argparse
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

def project_3d_to_2d(points_3d, camera_intrinsics):
    """将3D点投影到2D图像平面"""
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    valid_mask = z > 0.01
    
    u = np.where(valid_mask, fx * (x / z) + cx, -1)
    v = np.where(valid_mask, fy * (y / z) + cy, -1)
    
    points_2d = np.stack([u, v], axis=1)
    return points_2d, valid_mask


def draw_wrist_rotation_arrow(image, wrist_pos_3d, wrist_rotvec, camera_intrinsics, 
                              color, arrow_length=0.08):
    """
    绘制wrist的Z轴方向箭头（红色）
    
    Args:
        wrist_pos_3d: (3,) wrist的3D位置
        wrist_rotvec: (3,) wrist的旋转向量（轴角表示）
        arrow_length: 箭头长度（米）
    """
    from scipy.spatial.transform import Rotation as R
    
    # 将 rotvec 转换为旋转矩阵
    rot_matrix = R.from_rotvec(wrist_rotvec).as_matrix()
    
    # 取旋转矩阵的Z轴作为方向（表示手腕的"前方"）
    direction = rot_matrix @ np.array([0, 0, 1])  # Z轴方向
    
    # 计算箭头终点
    arrow_end_3d = wrist_pos_3d + direction * arrow_length
    
    # 投影到2D
    points_3d = np.array([wrist_pos_3d, arrow_end_3d])
    points_2d, valid_mask = project_3d_to_2d(points_3d, camera_intrinsics)
    
    if valid_mask.all():
        # 绘制红色箭头
        start_pt = tuple(points_2d[0].astype(int))
        end_pt = tuple(points_2d[1].astype(int))
        
        # 检查点是否在图像范围内
        h, w = image.shape[:2]
        if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
            0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
            # 统一使用红色绘制Z轴方向箭头（参数顺序：img, pt1, pt2, color, thickness, line_type, shift, tipLength）
            cv2.arrowedLine(image, start_pt, end_pt, COLOR_Z_AXIS, 
                           3, cv2.LINE_AA, 0, 0.25)


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


def parse_keypoints_file(filepath):
    """解析关键点txt文件（支持21维格式：6kp_xyz + wrist_rotvec）"""
    keypoints_data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            values = list(map(float, line.split()))
            # 新格式：frame_id(1) + t(1) + left_21(21) + right_21(21) = 44
            # 旧格式：frame_id(1) + t(1) + left_18(18) + right_18(18) = 38
            if len(values) == 44:
                # 新格式：包含 wrist_rotvec
                frame_id = int(values[0])
                t = int(values[1])
                coords = np.array(values[2:], dtype=np.float32)
                
                # 左手21维：wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
                left_kp_18 = coords[:18].reshape(6, 3)      # 6个关键点
                left_rotvec = coords[18:21]                  # wrist旋转向量
                
                # 右手21维
                right_kp_18 = coords[21:39].reshape(6, 3)   # 6个关键点
                right_rotvec = coords[39:42]                 # wrist旋转向量
                
                keypoints_data[(frame_id, t)] = (left_kp_18, left_rotvec, right_kp_18, right_rotvec)
                
            elif len(values) == 38:
                # 旧格式：只有关键点，没有 rotvec
                frame_id = int(values[0])
                t = int(values[1])
                coords = np.array(values[2:], dtype=np.float32)
                
                left_kp = coords[:18].reshape(6, 3)
                right_kp = coords[18:].reshape(6, 3)
                
                keypoints_data[(frame_id, t)] = (left_kp, None, right_kp, None)
            else:
                print(f"Warning: 跳过格式不正确的行 (expected 38 or 44 values, got {len(values)})")
                continue
    
    return keypoints_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将关键点重投影到视频上进行可视化验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_retarget_reprojector_cli.py \\
      --video output_video_record/.../video.mp4 \\
      --keypoints output_video_record/predicted_keypoints.txt \\
      --fps 5 --radius 8
        """
    )
    
    # 必需参数
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--keypoints', type=str, required=True,
                       help='关键点txt文件路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认在输入视频目录生成*_reprojected.mp4）')
    parser.add_argument('--fps', type=int, default=5,
                       help='输出视频fps（默认5）')
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
    
    # 转换路径
    video_path = Path(args.video)
    keypoints_path = Path(args.keypoints)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_reprojected.mp4"
    
    # 相机内参
    camera_intrinsics = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    print("=" * 80)
    print("Keypoints Reprojection Visualizer (CLI)")
    print("=" * 80)
    print(f"\n输入视频: {video_path}")
    print(f"关键点文件: {keypoints_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"可视化参数: radius={args.radius}, thickness={args.thickness}, labels={not args.no_labels}")
    
    # 检查输入文件
    if not video_path.exists():
        print(f"\n❌ 错误: 视频文件不存在: {video_path}")
        return
    
    if not keypoints_path.exists():
        print(f"\n❌ 错误: 关键点文件不存在: {keypoints_path}")
        return
    
    # 1. 解析关键点数据
    print("\n[1/3] 解析关键点数据...")
    keypoints_data = parse_keypoints_file(keypoints_path)
    print(f"✓ 加载了 {len(keypoints_data)} 个时间步的关键点数据")
    
    frame_ids = sorted(set(k[0] for k in keypoints_data.keys()))
    print(f"✓ Chunk范围: {min(frame_ids)} ~ {max(frame_ids)} (共{len(frame_ids)}个chunk)")
    
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
    
    print(f"✓ 视频信息: {width}x{height}, {input_fps:.2f} fps, {total_frames} 帧")
    
    if input_fps <= args.fps:
        frame_skip = 1
        actual_output_fps = input_fps
        print(f"✓ 输入FPS({input_fps:.2f}) <= 输出FPS({args.fps})，保持所有帧")
    else:
        frame_skip = int(input_fps / args.fps)
        actual_output_fps = args.fps
        print(f"✓ 帧采样: 每隔 {frame_skip} 帧取1帧 (从 {input_fps:.2f}fps 降至 {args.fps}fps)")
    
    # 3. 创建输出视频
    print("\n[3/3] 处理视频并重投影关键点...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, actual_output_fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 错误: 无法创建输出视频文件")
        cap.release()
        return
    
    # 处理每一帧
    frame_idx = 0
    written_frames = 0
    current_chunk_idx = 0
    current_timestep = 0
    
    pbar = tqdm(total=total_frames, desc="处理进度")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            vis_frame = frame.copy()
            key = (current_chunk_idx, current_timestep)
            
            if key in keypoints_data:
                # 解包数据：支持新格式（带rotvec）和旧格式（不带rotvec）
                data = keypoints_data[key]
                if len(data) == 4:
                    # 新格式：(left_kp, left_rotvec, right_kp, right_rotvec)
                    left_kp_3d, left_rotvec, right_kp_3d, right_rotvec = data
                else:
                    # 旧格式：(left_kp, right_kp)
                    left_kp_3d, right_kp_3d = data
                    left_rotvec = right_rotvec = None
                
                # 投影关键点到2D
                left_kp_2d, left_valid = project_3d_to_2d(left_kp_3d, camera_intrinsics)
                right_kp_2d, right_valid = project_3d_to_2d(right_kp_3d, camera_intrinsics)
                
                # 绘制关键点和连线
                draw_hand_keypoints(vis_frame, left_kp_2d, left_valid, COLOR_LEFT_HAND, "L_",
                                   args.radius, args.thickness, not args.no_labels)
                draw_hand_keypoints(vis_frame, right_kp_2d, right_valid, COLOR_RIGHT_HAND, "R_",
                                   args.radius, args.thickness, not args.no_labels)
                
                # 绘制wrist的Z轴方向箭头（如果有rotvec数据），统一使用红色
                if left_rotvec is not None:
                    draw_wrist_rotation_arrow(vis_frame, left_kp_3d[0], left_rotvec, 
                                             camera_intrinsics, None, arrow_length=0.08)
                if right_rotvec is not None:
                    draw_wrist_rotation_arrow(vis_frame, right_kp_3d[0], right_rotvec, 
                                             camera_intrinsics, None, arrow_length=0.08)
                
                info_text = f"Chunk {current_chunk_idx} | Step {current_timestep} | Frame {frame_idx}"
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(vis_frame, f"No keypoints for Chunk {current_chunk_idx}, Step {current_timestep}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            out.write(vis_frame)
            written_frames += 1
            
            current_timestep += 1
            if current_timestep >= 16:
                current_timestep = 0
                current_chunk_idx += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ 处理完成!")
    print(f"  输入帧数: {total_frames} ({input_fps:.2f} fps)")
    print(f"  输出帧数: {written_frames} ({actual_output_fps:.2f} fps)")
    print(f"  输出文件: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
