#!/usr/bin/env python3
"""
EE Pose Reprojection Visualizer
将retarget后的手腕eepose（位置+方向）重投影到视频上进行可视化验证

读取retargeted_actions文件，提取每个时间步的左右手wrist pose（xyz + rotvec），
将其投影到2D图像平面，并用坐标轴显示手腕的方向。

Usage:
    python eval_after_retarget_reprojector_cli.py --video VIDEO_PATH --retarget RETARGET_PATH [OPTIONS]
    
Example:
    python eval_after_retarget_reprojector_cli.py \
        --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep/28ksteps-modify11/11_reprojected.mp4 \
        --retarget /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260129_153548.txt \
        --fps 30 \
        --steps-per-chunk 16 \
        --axis-length 0.05
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

# 手腕颜色 (BGR格式)
COLOR_LEFT_WRIST = (0, 255, 0)   # 左手 - 绿色
COLOR_RIGHT_WRIST = (0, 0, 255)  # 右手 - 红色

# Z轴方向颜色 (BGR格式) - 统一使用红色
COLOR_Z_AXIS = (255, 0, 255)   # Z轴方向 - 紫色

# =============================================================================
# 工具函数
# =============================================================================

def project_3d_to_2d(points_3d, camera_intrinsics):
    """
    将3D点投影到2D图像平面
    
    Args:
        points_3d: (N, 3) array of 3D points in camera coordinate system
        camera_intrinsics: dict with keys 'fx', 'fy', 'cx', 'cy'
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        valid_mask: (N,) boolean mask indicating which points are in front of camera
    """
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


def rotvec_to_rotation_matrix(rotvec):
    """
    将旋转向量(axis-angle)转换为旋转矩阵
    
    Args:
        rotvec: (3,) rotation vector
    
    Returns:
        R: (3, 3) rotation matrix
    """
    return R.from_rotvec(rotvec).as_matrix()


def draw_wrist_pose(image, wrist_pos_3d, wrist_rotvec, camera_intrinsics, 
                    wrist_color, axis_length, thickness, label):
    """
    在图像上绘制手腕位置和Z轴方向
    
    Args:
        image: 输入图像 (会被修改)
        wrist_pos_3d: (3,) 手腕位置 [x, y, z]
        wrist_rotvec: (3,) 手腕旋转向量 [rx, ry, rz]
        camera_intrinsics: 相机内参
        wrist_color: 手腕点颜色
        axis_length: Z轴长度（米）
        thickness: 线条粗细
        label: 标签文字
    """
    h, w = image.shape[:2]
    
    # 将手腕位置投影到2D
    wrist_pos_2d, valid = project_3d_to_2d(wrist_pos_3d.reshape(1, 3), camera_intrinsics)
    wrist_pos_2d = wrist_pos_2d[0]
    
    if not valid[0]:
        return
    
    # 检查点是否在图像范围内
    if not (0 <= wrist_pos_2d[0] < w and 0 <= wrist_pos_2d[1] < h):
        return
    
    # 绘制手腕位置点
    wrist_pt = tuple(map(int, wrist_pos_2d))
    cv2.circle(image, wrist_pt, 8, wrist_color, -1)
    cv2.circle(image, wrist_pt, 10, (255, 255, 255), 2)
    
    # 添加标签
    cv2.putText(image, label, (wrist_pt[0] + 12, wrist_pt[1] - 12),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 计算旋转矩阵
    rot_matrix = rotvec_to_rotation_matrix(wrist_rotvec)
    
    # 只绘制Z轴方向（手腕的主要朝向）
    z_axis_local = np.array([0, 0, axis_length])  # Z轴
    
    # 旋转到相机坐标系
    z_axis_cam = rot_matrix @ z_axis_local
    
    # 计算Z轴端点在相机坐标系中的位置
    z_endpoint_3d = wrist_pos_3d + z_axis_cam
    
    # 投影到2D
    z_endpoint_2d, z_valid = project_3d_to_2d(z_endpoint_3d.reshape(1, 3), camera_intrinsics)
    
    # 绘制Z轴方向箭头（红色）
    if z_valid[0]:
        endpoint_pt = tuple(map(int, z_endpoint_2d[0]))
        
        # 检查端点是否在图像范围内
        if 0 <= endpoint_pt[0] < w and 0 <= endpoint_pt[1] < h:
            # 绘制红色箭头（参数顺序：img, pt1, pt2, color, thickness, line_type, shift, tipLength）
            cv2.arrowedLine(image, wrist_pt, endpoint_pt, COLOR_Z_AXIS, 
                           thickness, cv2.LINE_AA, 0, 0.25)


def parse_retarget_actions_file(filepath):
    """
    解析retargeted_actions txt文件
    
    文件格式：
    chunk_id t L_wrist_xyz(3) L_rotvec(3) L_finger_q(6) R_wrist_xyz(3) R_rotvec(3) R_finger_q(6)
    总共：2 + 6 + 6 + 6 + 6 = 26个数值
    
    Returns:
        retarget_data: dict mapping (chunk_id, t) -> (left_pose, right_pose)
                      where left_pose and right_pose are dicts with 'pos' and 'rotvec'
    """
    retarget_data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析数据行
            values = list(map(float, line.split()))
            if len(values) != 26:  # chunk_id + t + 24个数据
                print(f"Warning: 跳过格式不正确的行 (expected 26 values, got {len(values)})")
                continue
            
            chunk_id = int(values[0])
            t = int(values[1])
            
            # 提取左手wrist pose (索引2-7)
            left_wrist_pos = np.array(values[2:5], dtype=np.float32)     # xyz
            left_wrist_rotvec = np.array(values[5:8], dtype=np.float32)  # rotvec
            
            # 提取右手wrist pose (索引14-19)
            right_wrist_pos = np.array(values[14:17], dtype=np.float32)    # xyz
            right_wrist_rotvec = np.array(values[17:20], dtype=np.float32) # rotvec
            
            left_pose = {
                'pos': left_wrist_pos,
                'rotvec': left_wrist_rotvec
            }
            
            right_pose = {
                'pos': right_wrist_pos,
                'rotvec': right_wrist_rotvec
            }
            
            retarget_data[(chunk_id, t)] = (left_pose, right_pose)
    
    return retarget_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='将retarget后的手腕eepose重投影到视频上进行可视化验证',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python eval_after_retarget_reprojector_cli.py \\
      --video output_video_record/.../video.mp4 \\
      --retarget output_video_record/retargeted_actions.txt \\
      --fps 5 --axis-length 0.05
        """
    )
    
    # 必需参数
    parser.add_argument('--video', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--retarget', type=str, required=True,
                       help='retargeted_actions txt文件路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径（默认在输入视频目录生成*_eepose_reprojected.mp4）')
    parser.add_argument('--fps', type=int, default=30,
                       help='输出视频fps（默认30，与substep视频一致）')
    parser.add_argument('--steps-per-chunk', type=int, default=16,
                       help='每个chunk的时间步数（默认16）')
    parser.add_argument('--axis-length', type=float, default=0.05,
                       help='坐标轴长度（米，默认0.05）')
    parser.add_argument('--thickness', type=int, default=3,
                       help='线条粗细（默认3）')
    
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
    retarget_path = Path(args.retarget)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = video_path.parent / f"{video_path.stem}_eepose_reprojected.mp4"
    
    # 相机内参
    camera_intrinsics = {
        'fx': args.fx,
        'fy': args.fy,
        'cx': args.cx,
        'cy': args.cy
    }
    
    print("=" * 80)
    print("EE Pose Reprojection Visualizer (Substep对齐版)")
    print("=" * 80)
    print(f"\n输入视频: {video_path}")
    print(f"Retarget文件: {retarget_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {args.fps}")
    print(f"每个chunk的步数: {args.steps_per_chunk}")
    print(f"可视化参数: axis_length={args.axis_length}m, thickness={args.thickness}")
    
    # 检查输入文件
    if not video_path.exists():
        print(f"\n❌ 错误: 视频文件不存在: {video_path}")
        return
    
    if not retarget_path.exists():
        print(f"\n❌ 错误: Retarget文件不存在: {retarget_path}")
        return
    
    # 1. 解析retarget数据
    print("\n[1/3] 解析retarget数据...")
    retarget_data = parse_retarget_actions_file(retarget_path)
    print(f"✓ 加载了 {len(retarget_data)} 个时间步的eepose数据")
    
    chunk_ids = sorted(set(k[0] for k in retarget_data.keys()))
    print(f"✓ Chunk范围: {min(chunk_ids)} ~ {max(chunk_ids)} (共{len(chunk_ids)}个chunk)")
    
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
    
    # 计算预期的总步数
    total_retarget_steps = len(retarget_data)
    expected_chunks = (total_retarget_steps + args.steps_per_chunk - 1) // args.steps_per_chunk
    
    print(f"✓ Retarget数据: {total_retarget_steps} 个时间步 ({expected_chunks} chunks × {args.steps_per_chunk} steps)")
    print(f"✓ 视频帧数: {total_frames} 帧")
    
    if total_frames != total_retarget_steps:
        print(f"⚠ 警告: 视频帧数({total_frames}) != Retarget步数({total_retarget_steps})，将按帧索引对齐")
    
    # 3. 创建输出视频
    print("\n[3/3] 处理视频并重投影eepose...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, args.fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 错误: 无法创建输出视频文件")
        cap.release()
        return
    
    # 处理每一帧 - 直接一一对应
    # 视频的第N帧对应 chunk_id = N // steps_per_chunk, timestep = N % steps_per_chunk
    frame_idx = 0
    matched_frames = 0
    unmatched_frames = 0
    
    pbar = tqdm(total=total_frames, desc="处理进度")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis_frame = frame.copy()
        
        # 根据帧索引计算对应的 chunk_id 和 timestep
        current_chunk_idx = frame_idx // args.steps_per_chunk
        current_timestep = frame_idx % args.steps_per_chunk
        key = (current_chunk_idx, current_timestep)
        
        if key in retarget_data:
            left_pose, right_pose = retarget_data[key]
            
            # 绘制左手wrist pose
            draw_wrist_pose(
                vis_frame,
                left_pose['pos'],
                left_pose['rotvec'],
                camera_intrinsics,
                COLOR_LEFT_WRIST,
                args.axis_length,
                args.thickness,
                "L_wrist"
            )
            
            # 绘制右手wrist pose
            draw_wrist_pose(
                vis_frame,
                right_pose['pos'],
                right_pose['rotvec'],
                camera_intrinsics,
                COLOR_RIGHT_WRIST,
                args.axis_length,
                args.thickness,
                "R_wrist"
            )
            
            matched_frames += 1
        else:
            unmatched_frames += 1
        
        # 添加信息文本（右上角显示）
        info_text = f"Chunk {current_chunk_idx} | Step {current_timestep} | Frame {frame_idx}"
        (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(vis_frame, info_text, (width - text_w - 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if key not in retarget_data:
            no_data_text = "No retarget data"
            (nd_w, _), _ = cv2.getTextSize(no_data_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(vis_frame, no_data_text,
                       (width - nd_w - 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 写入输出视频
        out.write(vis_frame)
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    out.release()
    
    print(f"\n✓ 处理完成!")
    print(f"  总帧数: {frame_idx}")
    print(f"  匹配成功: {matched_frames} 帧")
    print(f"  匹配失败: {unmatched_frames} 帧")
    print(f"  输出FPS: {args.fps}")
    print(f"  输出文件: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
