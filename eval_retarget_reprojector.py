#!/usr/bin/env python3
"""
Keypoints Reprojection Visualizer
将保存的关键点坐标重新投影到视频上，用于验证模型输出是否正确

Usage:
    python eval_retarget_reprojector.py
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# 配置参数
# =============================================================================
# 输入文件路径
VIDEO_PATH = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget/74ksteps/d77fd305-7431-41e5-bcca-bb1baa24c22b_success0.mp4")
KEYPOINTS_PATH = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260112_110653.txt")

# 输出文件路径
OUTPUT_PATH = VIDEO_PATH.parent / f"{VIDEO_PATH.stem}_reprojected.mp4"

# 视频参数
OUTPUT_FPS = 5  # 输出视频fps (可调)

# 相机内参 (来自GR1RetargetConfig)
CAMERA_INTRINSICS = {
    'fx': 502.8689,
    'fy': 502.8689,
    'cx': 640.0,
    'cy': 400.0
}

# 可视化参数
KEYPOINT_RADIUS = 5  # 关键点圆圈半径
LINE_THICKNESS = 2   # 连线粗细

# 关键点颜色 (BGR格式)
COLOR_LEFT_HAND = (0, 255, 0)   # 左手 - 绿色
COLOR_RIGHT_HAND = (0, 0, 255)  # 右手 - 红色
COLOR_WRIST = (255, 255, 0)     # 手腕 - 青色

# 手指连线关系 (索引对应关键点在6点中的位置)
# [wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
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
    
    # 提取xyz坐标
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    
    # 投影公式：u = fx * (x/z) + cx, v = fy * (y/z) + cy
    # 只投影z>0的点（在相机前方）
    valid_mask = z > 0.01  # 设置最小深度阈值
    
    u = np.where(valid_mask, fx * (x / z) + cx, -1)
    v = np.where(valid_mask, fy * (y / z) + cy, -1)
    
    points_2d = np.stack([u, v], axis=1)
    return points_2d, valid_mask


def draw_hand_keypoints(image, keypoints_2d, valid_mask, color, label_prefix):
    """
    在图像上绘制手部关键点和连线
    
    Args:
        image: 输入图像 (会被修改)
        keypoints_2d: (6, 2) array of 2D keypoint coordinates
        valid_mask: (6,) boolean mask
        color: BGR color tuple
        label_prefix: 标签前缀 (e.g., "L_" for left hand)
    """
    h, w = image.shape[:2]
    
    # 关键点名称
    kp_names = ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']
    
    # 绘制连线
    for i, j in HAND_CONNECTIONS:
        if valid_mask[i] and valid_mask[j]:
            pt1 = tuple(map(int, keypoints_2d[i]))
            pt2 = tuple(map(int, keypoints_2d[j]))
            
            # 检查点是否在图像范围内
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(image, pt1, pt2, color, LINE_THICKNESS)
    
    # 绘制关键点
    for i, (kp_name, pt, valid) in enumerate(zip(kp_names, keypoints_2d, valid_mask)):
        if not valid:
            continue
        
        pt_int = tuple(map(int, pt))
        
        # 检查点是否在图像范围内
        if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
            # 手腕用特殊颜色
            point_color = COLOR_WRIST if i == 0 else color
            
            # 绘制圆圈
            cv2.circle(image, pt_int, KEYPOINT_RADIUS, point_color, -1)
            cv2.circle(image, pt_int, KEYPOINT_RADIUS + 1, (255, 255, 255), 1)
            
            # 绘制标签 (只标注手腕和指尖)
            label = f"{label_prefix}{kp_name}"
            cv2.putText(image, label, (pt_int[0] + 8, pt_int[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def parse_keypoints_file(filepath):
    """
    解析关键点txt文件
    
    Returns:
        keypoints_data: dict mapping (frame_id, t) -> (left_kp, right_kp)
                       where left_kp and right_kp are (6, 3) arrays
    """
    keypoints_data = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue
            
            # 解析数据行
            values = list(map(float, line.split()))
            if len(values) != 38:  # frame_id + t + 36个坐标值
                print(f"Warning: 跳过格式不正确的行 (expected 38 values, got {len(values)})")
                continue
            
            frame_id = int(values[0])
            t = int(values[1])
            coords = np.array(values[2:], dtype=np.float32)
            
            # 左手18维 + 右手18维
            left_kp = coords[:18].reshape(6, 3)   # [wrist, thumb, index, middle, ring, pinky]
            right_kp = coords[18:].reshape(6, 3)  # [wrist, thumb, index, middle, ring, pinky]
            
            keypoints_data[(frame_id, t)] = (left_kp, right_kp)
    
    return keypoints_data


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 80)
    print("Keypoints Reprojection Visualizer")
    print("=" * 80)
    print(f"\n输入视频: {VIDEO_PATH}")
    print(f"关键点文件: {KEYPOINTS_PATH}")
    print(f"输出视频: {OUTPUT_PATH}")
    print(f"输出FPS: {OUTPUT_FPS}")
    
    # 检查输入文件是否存在
    if not VIDEO_PATH.exists():
        print(f"\n❌ 错误: 视频文件不存在: {VIDEO_PATH}")
        return
    
    if not KEYPOINTS_PATH.exists():
        print(f"\n❌ 错误: 关键点文件不存在: {KEYPOINTS_PATH}")
        return
    
    # 1. 解析关键点数据
    print("\n[1/3] 解析关键点数据...")
    keypoints_data = parse_keypoints_file(KEYPOINTS_PATH)
    print(f"✓ 加载了 {len(keypoints_data)} 个时间步的关键点数据")
    
    # 统计chunk信息
    frame_ids = sorted(set(k[0] for k in keypoints_data.keys()))
    print(f"✓ Chunk范围: {min(frame_ids)} ~ {max(frame_ids)} (共{len(frame_ids)}个chunk)")
    
    # 2. 打开输入视频
    print("\n[2/3] 打开输入视频...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件")
        return
    
    # 获取视频属性
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ 视频信息: {width}x{height}, {input_fps:.2f} fps, {total_frames} 帧")
    
    # 计算帧采样间隔
    if input_fps <= OUTPUT_FPS:
        # 如果输入fps已经小于等于输出fps，不进行采样
        frame_skip = 1
        actual_output_fps = input_fps
        print(f"✓ 输入FPS({input_fps:.2f}) <= 输出FPS({OUTPUT_FPS})，保持所有帧")
    else:
        frame_skip = int(input_fps / OUTPUT_FPS)
        actual_output_fps = OUTPUT_FPS
        print(f"✓ 帧采样: 每隔 {frame_skip} 帧取1帧 (从 {input_fps:.2f}fps 降至 {OUTPUT_FPS}fps)")
    
    # 3. 创建输出视频
    print("\n[3/3] 处理视频并重投影关键点...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, actual_output_fps, (width, height))
    
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
        
        # 每隔frame_skip帧处理一次
        if frame_idx % frame_skip == 0:
            # 制作frame的副本用于绘制
            vis_frame = frame.copy()
            
            # 查找对应的关键点数据
            # 假设关键点数据按照chunk和timestep顺序排列
            # 每个chunk有16个timestep (t=0~15)
            key = (current_chunk_idx, current_timestep)
            
            if key in keypoints_data:
                left_kp_3d, right_kp_3d = keypoints_data[key]
                
                # 投影到2D
                left_kp_2d, left_valid = project_3d_to_2d(left_kp_3d, CAMERA_INTRINSICS)
                right_kp_2d, right_valid = project_3d_to_2d(right_kp_3d, CAMERA_INTRINSICS)
                
                # 绘制左手关键点
                draw_hand_keypoints(vis_frame, left_kp_2d, left_valid, COLOR_LEFT_HAND, "L_")
                
                # 绘制右手关键点
                draw_hand_keypoints(vis_frame, right_kp_2d, right_valid, COLOR_RIGHT_HAND, "R_")
                
                # 添加信息文本
                info_text = f"Chunk {current_chunk_idx} | Step {current_timestep} | Frame {frame_idx}"
                cv2.putText(vis_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # 如果找不到对应的关键点数据，在图像上标注
                cv2.putText(vis_frame, f"No keypoints for Chunk {current_chunk_idx}, Step {current_timestep}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 写入输出视频
            out.write(vis_frame)
            written_frames += 1
            
            # 更新chunk和timestep索引
            current_timestep += 1
            if current_timestep >= 16:  # 假设每个chunk有16个timestep
                current_timestep = 0
                current_chunk_idx += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"\n✓ 处理完成!")
    print(f"  输入帧数: {total_frames} ({input_fps:.2f} fps)")
    print(f"  输出帧数: {written_frames} ({actual_output_fps:.2f} fps)")
    print(f"  输出文件: {OUTPUT_PATH}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
