#!/usr/bin/env python3
"""
观测数据重投影可视化工具
读取 policy 输出的原始观测数据（手腕位姿），并重投影到视频上进行可视化

Usage:
    python eval_observation_reprojector_cli.py \
        --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget/58ksteps/test1_58k.mp4 \
        --observation /vla/users/lijiayi/code/groot_retarget/output_video_record/observation_wrist_20260114_150533.txt \
        --fps 5 \
        --radius 8
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

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

# 颜色 (BGR格式)
COLOR_LEFT_HAND = (0, 255, 0)   # 左手 - 绿色
COLOR_RIGHT_HAND = (0, 0, 255)  # 右手 - 红色
COLOR_Z_AXIS = (0, 0, 255)      # Z轴方向 - 红色

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


def draw_wrist_with_rotation(image, wrist_pos_3d, wrist_rotvec, camera_intrinsics, 
                              color, label, radius=8, arrow_length=0.08):
    """
    绘制手腕位置和Z轴方向（红色箭头）
    
    Args:
        wrist_pos_3d: (3,) wrist的3D位置
        wrist_rotvec: (3,) wrist的旋转向量（轴角表示）
        arrow_length: 箭头长度（米）
    """
    # 将 rotvec 转换为旋转矩阵
    rot_matrix = R.from_rotvec(wrist_rotvec).as_matrix()
    
    # 取旋转矩阵的Z轴作为方向（表示手腕的"前方"）
    direction = rot_matrix @ np.array([0, 0, 1])  # Z轴方向
    
    # 计算箭头终点
    arrow_end_3d = wrist_pos_3d + direction * arrow_length
    
    # 投影到2D
    points_3d = np.array([wrist_pos_3d, arrow_end_3d])
    points_2d, valid_mask = project_3d_to_2d(points_3d, camera_intrinsics)
    
    h, w = image.shape[:2]
    
    if valid_mask[0]:
        # 绘制手腕位置
        wrist_pt = tuple(points_2d[0].astype(int))
        if 0 <= wrist_pt[0] < w and 0 <= wrist_pt[1] < h:
            cv2.circle(image, wrist_pt, radius, color, -1)
            cv2.circle(image, wrist_pt, radius + 2, (255, 255, 255), 2)
            # 添加标签
            cv2.putText(image, label, (wrist_pt[0] + 12, wrist_pt[1] - 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if valid_mask.all():
        # 绘制Z轴方向箭头（统一使用红色）
        start_pt = tuple(points_2d[0].astype(int))
        end_pt = tuple(points_2d[1].astype(int))
        
        if (0 <= start_pt[0] < w and 0 <= start_pt[1] < h and
            0 <= end_pt[0] < w and 0 <= end_pt[1] < h):
            cv2.arrowedLine(image, start_pt, end_pt, COLOR_Z_AXIS, 
                           3, cv2.LINE_AA, 0, 0.25)


def parse_observation_file(filepath):
    """
    解析观测数据txt文件
    
    格式：frame_id t L_wrist_xyz(3) L_wrist_rotvec(3) L_finger(6) R_wrist_xyz(3) R_wrist_rotvec(3) R_finger(6)
    总共：2 + 6 + 6 + 6 + 6 = 26 个值
    
    Returns:
        dict: {(frame_id, t): (left_wrist_xyz, left_wrist_rotvec, right_wrist_xyz, right_wrist_rotvec)}
    """
    observation_data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            values = list(map(float, line.split()))
            if len(values) != 26:
                print(f"Warning: 跳过格式不正确的行 (expected 26 values, got {len(values)})")
                continue
            
            frame_id = int(values[0])
            t = int(values[1])
            
            # 解析数据
            left_wrist_xyz = np.array(values[2:5], dtype=np.float32)
            left_wrist_rotvec = np.array(values[5:8], dtype=np.float32)
            # left_finger = values[8:14]  # 暂不使用
            
            right_wrist_xyz = np.array(values[14:17], dtype=np.float32)
            right_wrist_rotvec = np.array(values[17:20], dtype=np.float32)
            # right_finger = values[20:26]  # 暂不使用
            
            observation_data[(frame_id, t)] = (
                left_wrist_xyz, left_wrist_rotvec, 
                right_wrist_xyz, right_wrist_rotvec
            )
    
    return observation_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将观测数据重投影到视频上进行可视化验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_observation_reprojector_cli.py \\
      --video output_video_record/.../video.mp4 \\
      --observation output_video_record/observation_wrist_20260113_xxx.txt \\
      --fps 5 --radius 8
        """
    )
    
    # 必需参数
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--observation', type=str, required=True,
                       help='观测数据txt文件路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认在输入视频目录生成*_obs_reprojected.mp4）')
    parser.add_argument('--fps', type=int, default=5,
                       help='输出视频fps（默认5）')
    parser.add_argument('--radius', type=int, default=8,
                       help='手腕圆圈半径（默认8）')
    parser.add_argument('--arrow-length', type=float, default=0.08,
                       help='旋转箭头长度（米，默认0.08）')
    
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
    observation_path = Path(args.observation)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_obs_reprojected.mp4"
    
    # 相机内参
    camera_intrinsics = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    print("=" * 80)
    print("观测数据重投影可视化工具")
    print("=" * 80)
    print(f"\n输入视频: {video_path}")
    print(f"观测数据文件: {observation_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"可视化参数: radius={args.radius}, arrow_length={args.arrow_length}")
    
    # 检查输入文件
    if not video_path.exists():
        print(f"\n❌ 错误: 视频文件不存在: {video_path}")
        return
    
    if not observation_path.exists():
        print(f"\n❌ 错误: 观测数据文件不存在: {observation_path}")
        return
    
    # 1. 解析观测数据
    print("\n[1/3] 解析观测数据...")
    observation_data = parse_observation_file(observation_path)
    print(f"✓ 加载了 {len(observation_data)} 个时间步的观测数据")
    
    frame_ids = sorted(set(k[0] for k in observation_data.keys()))
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
    print("\n[3/3] 处理视频并重投影观测数据...")
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
            
            if key in observation_data:
                left_wrist_xyz, left_wrist_rotvec, right_wrist_xyz, right_wrist_rotvec = observation_data[key]
                
                # 绘制左右手腕位置和旋转方向
                draw_wrist_with_rotation(vis_frame, left_wrist_xyz, left_wrist_rotvec,
                                        camera_intrinsics, COLOR_LEFT_HAND, "L_wrist",
                                        args.radius, args.arrow_length)
                draw_wrist_with_rotation(vis_frame, right_wrist_xyz, right_wrist_rotvec,
                                        camera_intrinsics, COLOR_RIGHT_HAND, "R_wrist",
                                        args.radius, args.arrow_length)
                
                info_text = f"Chunk {current_chunk_idx} | Step {current_timestep} | Frame {frame_idx}"
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(vis_frame, f"No observation for Chunk {current_chunk_idx}, Step {current_timestep}",
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
