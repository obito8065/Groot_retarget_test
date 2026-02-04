#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检查RoboCasa手部关节关键点脚本

功能：
1. 读取txt文件中的关节角度（L_arm_q1-7, L_finger_q1-6, R_arm_q1-7, R_finger_q1-6, waist_q1-3）
2. 使用完整机器人FK计算6个关键点（wrist + 5 tips）（与body_retarget_robocasa_eepose_keypoints_v3.py一致）
3. 重投影到视频上

使用方法：
python check_robocasa_hand_joint_keypoint.py \
    --action-txt /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/robocasa_action_20260202_152421.txt \
    --video-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/4_reprojected_eepose_reprojected.mp4 \
    --robot-urdf /vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf \
    --output-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/output_reprojected2.mp4 \
    --fps 30.0

注意：现在使用完整机器人FK，不再需要单独的hand URDF文件
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import pinocchio as pin

# 不再需要导入独立的hand FK，直接使用完整机器人FK


# =============================================================================
# 配置参数
# =============================================================================

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

def set_joint_q(model: pin.Model, q: np.ndarray, joint_name: str, value: float):
    """根据关节名字安全地写入 q 向量中对应 DOF"""
    joint_id = model.getJointId(joint_name)
    if joint_id == 0:
        raise ValueError(f"关节 {joint_name} 的 ID 为 0(universe)，没有 DOF")
    q_index = model.idx_qs[joint_id]
    nqs = model.nqs[joint_id]
    if nqs != 1:
        raise ValueError(f"当前代码假定 {joint_name} 只有 1 个 DOF, 实际 nqs={nqs}")
    q[q_index] = value


def build_full_joint_array(model: pin.Model, action_vector: np.ndarray) -> np.ndarray:
    """
    将 robocasa 的 44 维动作向量，按照 URDF 中的关节名字映射到 Pinocchio 的 q 向量
    """
    q = pin.neutral(model).copy()

    # Left leg: Action[13-18]
    set_joint_q(model, q, "left_hip_roll_joint",   action_vector[13])
    set_joint_q(model, q, "left_hip_yaw_joint",    action_vector[14])
    set_joint_q(model, q, "left_hip_pitch_joint",  action_vector[15])
    set_joint_q(model, q, "left_knee_pitch_joint", action_vector[16])
    set_joint_q(model, q, "left_ankle_pitch_joint",action_vector[17])
    set_joint_q(model, q, "left_ankle_roll_joint", action_vector[18])

    # Right leg: Action[35-40]
    set_joint_q(model, q, "right_hip_roll_joint",   action_vector[35])
    set_joint_q(model, q, "right_hip_yaw_joint",    action_vector[36])
    set_joint_q(model, q, "right_hip_pitch_joint", action_vector[37])
    set_joint_q(model, q, "right_knee_pitch_joint", action_vector[38])
    set_joint_q(model, q, "right_ankle_pitch_joint",action_vector[39])
    set_joint_q(model, q, "right_ankle_roll_joint", action_vector[40])

    # Waist: Action[41-43]
    set_joint_q(model, q, "waist_yaw_joint",   action_vector[41])
    set_joint_q(model, q, "waist_pitch_joint", action_vector[42])
    set_joint_q(model, q, "waist_roll_joint",  action_vector[43])

    # Neck: Action[19-21]
    set_joint_q(model, q, "head_roll_joint",  action_vector[20])
    set_joint_q(model, q, "head_pitch_joint", action_vector[21])
    set_joint_q(model, q, "head_yaw_joint",   action_vector[19])

    # Left arm: Action[0-6]
    set_joint_q(model, q, "left_shoulder_pitch_joint", action_vector[0])
    set_joint_q(model, q, "left_shoulder_roll_joint",  action_vector[1])
    set_joint_q(model, q, "left_shoulder_yaw_joint",   action_vector[2])
    set_joint_q(model, q, "left_elbow_pitch_joint",    action_vector[3])
    set_joint_q(model, q, "left_wrist_yaw_joint",      action_vector[4])
    set_joint_q(model, q, "left_wrist_roll_joint",     action_vector[5])
    set_joint_q(model, q, "left_wrist_pitch_joint",    action_vector[6])

    # Right arm: Action[22-28]
    set_joint_q(model, q, "right_shoulder_pitch_joint", action_vector[22])
    set_joint_q(model, q, "right_shoulder_roll_joint",  action_vector[23])
    set_joint_q(model, q, "right_shoulder_yaw_joint",   action_vector[24])
    set_joint_q(model, q, "right_elbow_pitch_joint",    action_vector[25])
    set_joint_q(model, q, "right_wrist_yaw_joint",      action_vector[26])
    set_joint_q(model, q, "right_wrist_roll_joint",     action_vector[27])
    set_joint_q(model, q, "right_wrist_pitch_joint",    action_vector[28])

    # 左手指的 6 个 active 关节
    # left_hand: Action[7:13] 共 6 维
    # 数据集中是[index, middle, ring, pinky, thumb_pitch, thumb_yaw] (与body_retarget_robocasa_eepose_keypoints_v3.py一致)
    set_joint_q(model, q, "L_index_proximal_joint",       -action_vector[7])
    set_joint_q(model, q, "L_middle_proximal_joint",      -action_vector[8])
    set_joint_q(model, q, "L_ring_proximal_joint",        -action_vector[9])
    set_joint_q(model, q, "L_pinky_proximal_joint",       -action_vector[10])
    set_joint_q(model, q, "L_thumb_proximal_yaw_joint",   -action_vector[12])
    set_joint_q(model, q, "L_thumb_proximal_pitch_joint", -action_vector[11])

    # 右手指的 6 个 active 关节
    # right_hand: Action[29:35] 共 6 维
    set_joint_q(model, q, "R_index_proximal_joint",       -action_vector[29])
    set_joint_q(model, q, "R_middle_proximal_joint",      -action_vector[30])
    set_joint_q(model, q, "R_ring_proximal_joint",        -action_vector[31])
    set_joint_q(model, q, "R_pinky_proximal_joint",       -action_vector[32])
    set_joint_q(model, q, "R_thumb_proximal_yaw_joint",   -action_vector[34])
    set_joint_q(model, q, "R_thumb_proximal_pitch_joint", -action_vector[33])

    return q


class BodyRetargeterFK:
    """
    使用完整机器人模型进行FK，计算手腕和手指关键点（相机坐标系）
    复用 body_retarget_robocasa_eepose_keypoints_v3.py 中验证成功的链路
    """

    def __init__(self, urdf_path: Path):
        print("正在初始化 BodyRetargeterFK (使用完整机器人URDF)...")
        
        # 使用Pinocchio构建完整机器人模型
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()

        # 获取关键link的frame ID
        self.frame_id_torso = self.model.getFrameId("torso_link", pin.BODY)
        self.frame_id_head = self.model.getFrameId("head_pitch_link", pin.BODY)
        self.frame_id_left_hand = self.model.getFrameId("left_hand_pitch_link", pin.BODY)
        self.frame_id_right_hand = self.model.getFrameId("right_hand_pitch_link", pin.BODY)

        # 左右手指尖 frame
        self.left_fingertip_frame_names = [
            "L_thumb_tip_link",
            "L_index_tip_link",
            "L_middle_tip_link",
            "L_ring_tip_link",
            "L_pinky_tip_link",
        ]
        self.right_fingertip_frame_names = [
            "R_thumb_tip_link",
            "R_index_tip_link",
            "R_middle_tip_link",
            "R_ring_tip_link",
            "R_pinky_tip_link",
        ]
        self.left_fingertip_frame_ids = [
            self.model.getFrameId(name) for name in self.left_fingertip_frame_names
        ]
        self.right_fingertip_frame_ids = [
            self.model.getFrameId(name) for name in self.right_fingertip_frame_names
        ]
        
        # 定义相机在头部坐标系中的固定变换 (外参)
        t_cam_in_head = [2.650 - 2.65017178 + 0.23, -1.944 + 2.174 - 0.23, 1.538 - 1.4475]
        q_cam_in_head = [-0.205, 0.676, -0.676, 0.205]  # (w, x, y, z)
        T_cam_in_head = self._create_transform(t_cam_in_head, q_cam_in_head)
        self.T_head_to_cam = np.linalg.inv(T_cam_in_head)
        
        print("初始化完成。")
        print(f"模型总关节数: {self.model.nq}, 总link数: {len(self.model.frames)}")

    def _create_transform(self, t: List[float], q: List[float]) -> np.ndarray:
        """创建4x4齐次变换矩阵"""
        T = np.eye(4)
        # q 是 (w, x, y, z), scipy 需要 (x, y, z, w)
        T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        T[:3, 3] = t
        return T

    def _get_link_pose(self, frame_id: int) -> np.ndarray:
        """获取指定link的位姿（4x4齐次变换矩阵）"""
        return self.data.oMf[frame_id].homogeneous

    def compute_keypoints(self, action_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        从44维关节角计算左右手关键点（相机坐标系）
        
        返回：
            left_keypoints: (21,) - wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
            right_keypoints: (21,) - wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        """
        # 构建完整关节配置
        q_full = build_full_joint_array(self.model, action_vector)
        
        # FK计算所有位姿
        pin.framesForwardKinematics(self.model, self.data, q_full)

        # 获取 world 坐标系下的 torso/head/左右手
        T_world_to_torso = self._get_link_pose(self.frame_id_torso)
        T_world_to_head = self._get_link_pose(self.frame_id_head)
        T_world_to_left_hand = self._get_link_pose(self.frame_id_left_hand)
        T_world_to_right_hand = self._get_link_pose(self.frame_id_right_hand)

        # 计算相对于torso的变换
        T_torso_to_head = np.linalg.inv(T_world_to_torso) @ T_world_to_head
        T_torso_to_left_hand = np.linalg.inv(T_world_to_torso) @ T_world_to_left_hand
        T_torso_to_right_hand = np.linalg.inv(T_world_to_torso) @ T_world_to_right_hand

        # 坐标系转换 torso -> head -> camera
        T_inv_torso_to_head = np.linalg.inv(T_torso_to_head)
        T_head_to_left_hand = T_inv_torso_to_head @ T_torso_to_left_hand
        T_head_to_right_hand = T_inv_torso_to_head @ T_torso_to_right_hand

        T_cam_to_left_hand = self.T_head_to_cam @ T_head_to_left_hand
        T_cam_to_right_hand = self.T_head_to_cam @ T_head_to_right_hand

        # 提取手腕位置和旋转
        left_hand_pos = np.asarray(T_cam_to_left_hand[:3, 3]).flatten()
        right_hand_pos = np.asarray(T_cam_to_right_hand[:3, 3]).flatten()
        left_hand_rot = np.asarray(T_cam_to_left_hand[:3, :3], dtype=np.float64)
        right_hand_rot = np.asarray(T_cam_to_right_hand[:3, :3], dtype=np.float64)
        
        # 转换为轴角表示
        left_hand_rotvec = R.from_matrix(left_hand_rot).as_rotvec()
        right_hand_rotvec = R.from_matrix(right_hand_rot).as_rotvec()

        # 计算左右手指尖在相机系下的位置
        left_fingertips: List[np.ndarray] = []
        for fid in self.left_fingertip_frame_ids:
            T_world_to_tip = self._get_link_pose(fid)
            T_torso_to_tip = np.linalg.inv(T_world_to_torso) @ T_world_to_tip
            T_head_to_tip = T_inv_torso_to_head @ T_torso_to_tip
            T_cam_to_tip = self.T_head_to_cam @ T_head_to_tip
            pos = np.asarray(T_cam_to_tip[:3, 3]).flatten()
            left_fingertips.append(pos)

        right_fingertips: List[np.ndarray] = []
        for fid in self.right_fingertip_frame_ids:
            T_world_to_tip = self._get_link_pose(fid)
            T_torso_to_tip = np.linalg.inv(T_world_to_torso) @ T_world_to_tip
            T_head_to_tip = T_inv_torso_to_head @ T_torso_to_tip
            T_cam_to_tip = self.T_head_to_cam @ T_head_to_tip
            pos = np.asarray(T_cam_to_tip[:3, 3]).flatten()
            right_fingertips.append(pos)

        # 组装 21维关键点：wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        left_keypoints = np.concatenate([left_hand_pos] + left_fingertips + [left_hand_rotvec])
        right_keypoints = np.concatenate([right_hand_pos] + right_fingertips + [right_hand_rotvec])

        return left_keypoints, right_keypoints


def parse_action_txt(filepath: Path) -> List[Dict]:
    """
    解析action txt文件
    
    格式：chunk_id t L_arm_q1 ... L_arm_q7 L_finger_q1 ... L_finger_q6 
          R_arm_q1 ... R_arm_q7 R_finger_q1 ... R_finger_q6 waist_q1 waist_q2 waist_q3
    
    返回：
        data: List of dicts, each containing chunk_id, t, and joint angles
    """
    data = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if not line or line.startswith('#'):
                continue
            
            values = list(map(float, line.split()))
            if len(values) < 2:
                continue
            
            chunk_id = int(values[0])
            t = int(values[1])
            
            # 提取关节角度（跳过chunk_id和t）
            joint_angles = np.array(values[2:], dtype=np.float32)
            
            # 构建44维action向量
            # 顺序：left_arm(7), left_hand(6), left_leg(6), neck(3), 
            #       right_arm(7), right_hand(6), right_leg(6), waist(3)
            # txt文件顺序：L_arm_q1-7, L_finger_q1-6, R_arm_q1-7, R_finger_q1-6, waist_q1-3
            # txt文件中L_finger顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            # 44维action中left_hand顺序: [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
            # 需要填充left_leg, neck, right_leg为0
            action_44d = np.zeros(44, dtype=np.float32)
            
            # Left arm: [0:7]
            action_44d[0:7] = joint_angles[0:7]
            
            # Left hand: [7:13] - 需要转换顺序
            # txt: [pinky, ring, middle, index, thumb_pitch, thumb_yaw] -> [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
            left_hand_txt = joint_angles[7:13]  # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            action_44d[7] = left_hand_txt[3]   # index
            action_44d[8] = left_hand_txt[2]   # middle
            action_44d[9] = left_hand_txt[1]   # ring
            action_44d[10] = left_hand_txt[0]  # pinky
            action_44d[11] = left_hand_txt[4]  # thumb_pitch
            action_44d[12] = left_hand_txt[5]  # thumb_yaw
            
            # Left leg: [13:19] - 用0填充
            action_44d[13:19] = 0.0
            
            # Neck: [19:22] - 用0填充
            action_44d[19:22] = 0.0
            
            # Right arm: [22:29]
            action_44d[22:29] = joint_angles[13:20]
            
            # Right hand: [29:35] - 需要转换顺序
            # txt: [pinky, ring, middle, index, thumb_pitch, thumb_yaw] -> [index, middle, ring, pinky, thumb_pitch, thumb_yaw]
            right_hand_txt = joint_angles[20:26]  # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            action_44d[29] = right_hand_txt[3]   # index
            action_44d[30] = right_hand_txt[2]   # middle
            action_44d[31] = right_hand_txt[1]   # ring
            action_44d[32] = right_hand_txt[0]  # pinky
            action_44d[33] = right_hand_txt[4]  # thumb_pitch
            action_44d[34] = right_hand_txt[5]  # thumb_yaw
            
            # Right leg: [35:41] - 用0填充
            action_44d[35:41] = 0.0
            
            # Waist: [41:44]
            action_44d[41:44] = joint_angles[26:29]
            
            data.append({
                'chunk_id': chunk_id,
                't': t,
                'action_44d': action_44d,
            })
    
    return data


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


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='检查RoboCasa手部关节关键点并重投影到视频')
    parser.add_argument('--action-txt', type=str, required=True,
                       help='输入的action txt文件路径')
    parser.add_argument('--video-path', type=str, required=True,
                       help='输入视频文件路径')
    parser.add_argument('--robot-urdf', type=str, required=True,
                       help='完整机器人URDF文件路径')
    parser.add_argument('--left-hand-urdf', type=str, required=False,
                       help='左手URDF文件路径（已废弃，不再需要）')
    parser.add_argument('--right-hand-urdf', type=str, required=False,
                       help='右手URDF文件路径（已废弃，不再需要）')
    parser.add_argument('--output-path', type=str, required=True,
                       help='输出视频文件路径')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='输出视频FPS (默认: 5.0)')
    
    args = parser.parse_args()
    
    action_txt_path = Path(args.action_txt)
    video_path = Path(args.video_path)
    robot_urdf_path = Path(args.robot_urdf)
    output_path = Path(args.output_path)
    output_fps = args.fps
    
    print("=" * 80)
    print("RoboCasa手部关节关键点检查脚本")
    print("=" * 80)
    print(f"\n输入action文件: {action_txt_path}")
    print(f"输入视频: {video_path}")
    print(f"输出视频: {output_path}")
    print(f"输出FPS: {output_fps}")
    
    # 检查输入文件
    if not action_txt_path.exists():
        print(f"\n❌ 错误: Action文件不存在: {action_txt_path}")
        return
    
    if not video_path.exists():
        print(f"\n❌ 错误: 视频文件不存在: {video_path}")
        return
    
    if not robot_urdf_path.exists():
        print(f"\n❌ 错误: 机器人URDF文件不存在: {robot_urdf_path}")
        return
    
    # 1. 解析action txt文件
    print("\n[1/4] 解析action txt文件...")
    action_data = parse_action_txt(action_txt_path)
    print(f"✓ 加载了 {len(action_data)} 个时间步的数据")
    
    # 2. 初始化FK计算器（使用完整机器人FK）
    print("\n[2/4] 初始化FK计算器...")
    fk = BodyRetargeterFK(robot_urdf_path)
    print("✓ FK计算器初始化完成")
    
    # 3. 计算所有时间步的关键点
    print("\n[3/4] 计算关键点...")
    keypoints_data = {}  # {(chunk_id, t): (left_kp_6, right_kp_6)}
    
    for item in tqdm(action_data, desc="计算关键点"):
        chunk_id = item['chunk_id']
        t = item['t']
        action_44d = item['action_44d']
        
        # 使用完整机器人FK直接计算所有关键点（与body_retarget_robocasa_eepose_keypoints_v3.py一致）
        left_keypoints_21d, right_keypoints_21d = fk.compute_keypoints(action_44d)
        
        # 提取6个关键点xyz（wrist + 5 tips）
        # 顺序：wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
        left_kp_6 = np.zeros((6, 3), dtype=np.float32)
        left_kp_6[0] = left_keypoints_21d[0:3]  # wrist
        left_kp_6[1] = left_keypoints_21d[3:6]  # thumb_tip
        left_kp_6[2] = left_keypoints_21d[6:9]  # index_tip
        left_kp_6[3] = left_keypoints_21d[9:12]  # middle_tip
        left_kp_6[4] = left_keypoints_21d[12:15]  # ring_tip
        left_kp_6[5] = left_keypoints_21d[15:18]  # pinky_tip
        
        right_kp_6 = np.zeros((6, 3), dtype=np.float32)
        right_kp_6[0] = right_keypoints_21d[0:3]  # wrist
        right_kp_6[1] = right_keypoints_21d[3:6]  # thumb_tip
        right_kp_6[2] = right_keypoints_21d[6:9]  # index_tip
        right_kp_6[3] = right_keypoints_21d[9:12]  # middle_tip
        right_kp_6[4] = right_keypoints_21d[12:15]  # ring_tip
        right_kp_6[5] = right_keypoints_21d[15:18]  # pinky_tip
        
        keypoints_data[(chunk_id, t)] = (left_kp_6, right_kp_6)
    
    print(f"✓ 计算了 {len(keypoints_data)} 个时间步的关键点")
    
    # 4. 处理视频并重投影关键点
    print("\n[4/4] 处理视频并重投影关键点...")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 错误: 无法打开视频文件")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ 原始视频信息: {width}x{height}, {input_fps:.2f} fps, {total_frames} 帧")
    
    # 计算帧采样率
    frame_skip = max(1, int(input_fps / output_fps))
    print(f"✓ 帧采样: 每隔 {frame_skip} 帧取1帧 (从 {input_fps:.2f}fps 降至 {output_fps}fps)")
    
    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
    
    frame_idx = 0
    current_chunk_idx = 0
    current_timestep = 0
    written_frames = 0
    steps_per_chunk = 50  # 每个chunk有16个时间步
    
    pbar = tqdm(total=total_frames, desc="处理进度")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔frame_skip帧处理一次
        if frame_idx % frame_skip == 0:
            # 制作frame的副本用于绘制
            vis_frame = frame.copy()
            
            # 直接从frame_idx计算对应的chunk和timestep
            # 假设视频帧和action数据是一一对应的（每个视频帧对应一个时间步）
            # frame_idx就是全局时间步索引
            global_step_idx = frame_idx
            
            # 计算chunk和timestep
            current_chunk_idx = global_step_idx // steps_per_chunk
            current_timestep = global_step_idx % steps_per_chunk
            
            # 查找对应的关键点数据
            key = (current_chunk_idx, current_timestep)
            
            if key in keypoints_data:
                left_kp_6, right_kp_6 = keypoints_data[key]
                
                # 投影到2D
                left_kp_2d, left_valid = project_3d_to_2d(left_kp_6, CAMERA_INTRINSICS)
                right_kp_2d, right_valid = project_3d_to_2d(right_kp_6, CAMERA_INTRINSICS)
                
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
                cv2.putText(vis_frame, f"No keypoints for Chunk {current_chunk_idx}, Step {current_timestep} (Frame {frame_idx})",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 写入输出视频
            out.write(vis_frame)
            written_frames += 1
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    out.release()
    pbar.close()
    
    print(f"\n✓ 处理完成!")
    print(f"  输入帧数: {total_frames} ({input_fps:.2f} fps)")
    print(f"  输出帧数: {written_frames} ({output_fps:.2f} fps)")
    print(f"  输出文件: {output_path}")


if __name__ == "__main__":
    main()
