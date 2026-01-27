#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fourier Hand FK v2 - 输出45维格式（与训练数据对齐）

输出格式（45维）:
- left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- right_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- waist(3)

与 body_retarget_robocasa_eepose_keypoints_v2.py 的输出格式完全对齐。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


@dataclass(frozen=True)
class MimicSpec:
    mimic_joint: str
    source_joint: str
    multiplier: float
    offset: float


class FourierHandFK:
    """
    Fourier 单手 FK：
      - 输入：wrist pose（cam下 pos+rotvec） + finger6（6个"主动关节"）
      - 输出：21维关键点：wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
    """

    def __init__(self, urdf_path: Path, side: str):
        self.urdf_path = Path(urdf_path).resolve()
        assert side in ("L", "R"), "side 必须是 'L' 或 'R'"
        self.side = side

        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF 不存在: {self.urdf_path}")

        # Pinocchio model（手的根 link 在 URDF 中就是 *_hand_base_link）
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()

        # 6个输入关节名
        self.finger_joint_names_6 = [
            f"{side}_index_proximal_joint",
            f"{side}_middle_proximal_joint",
            f"{side}_ring_proximal_joint",
            f"{side}_pinky_proximal_joint",
            f"{side}_thumb_proximal_yaw_joint",
            f"{side}_thumb_proximal_pitch_joint",
        ]

        # 5个指尖 frame 名
        self.tip_frame_names_5 = [
            f"{side}_thumb_tip_link",
            f"{side}_index_tip_link",
            f"{side}_middle_tip_link",
            f"{side}_ring_tip_link",
            f"{side}_pinky_tip_link",
        ]
        self.tip_frame_ids_5 = [self.model.getFrameId(n) for n in self.tip_frame_names_5]

        # 解析 mimic 关系
        self.mimics: List[MimicSpec] = self._parse_mimics_from_urdf(self.urdf_path)

        # 校验关节存在
        for jn in self.finger_joint_names_6:
            jid = self.model.getJointId(jn)
            if jid == 0:
                raise ValueError(f"[{side}] URDF 里找不到关节 {jn}")

    @staticmethod
    def _parse_mimics_from_urdf(urdf_path: Path) -> List[MimicSpec]:
        tree = ET.parse(str(urdf_path))
        root = tree.getroot()

        mimics: List[MimicSpec] = []
        for joint in root.findall("joint"):
            jname = joint.attrib.get("name", "")
            mimic = joint.find("mimic")
            if mimic is None:
                continue
            src = mimic.attrib.get("joint", "")
            mult = float(mimic.attrib.get("multiplier", "1.0"))
            off = float(mimic.attrib.get("offset", "0.0"))
            if jname and src:
                mimics.append(MimicSpec(jname, src, mult, off))
        return mimics

    def _set_1dof_joint(self, q: np.ndarray, joint_name: str, value: float) -> None:
        jid = self.model.getJointId(joint_name)
        if jid == 0:
            raise ValueError(f"关节不存在或无 DOF: {joint_name}")
        idx_q = int(self.model.idx_qs[jid])
        nqs = int(self.model.nqs[jid])
        if nqs != 1:
            raise ValueError(f"当前实现仅支持 1DOF 关节: {joint_name}, nqs={nqs}")
        q[idx_q] = float(value)

    def _apply_mimics(self, q: np.ndarray) -> None:
        """应用 mimic 关节"""
        for _ in range(2):
            for m in self.mimics:
                mid = self.model.getJointId(m.mimic_joint)
                sid = self.model.getJointId(m.source_joint)
                if mid == 0 or sid == 0:
                    continue
                midx = int(self.model.idx_qs[mid])
                sidx = int(self.model.idx_qs[sid])
                q[midx] = float(m.multiplier) * float(q[sidx]) + float(m.offset)

    def forward_keypoints_cam(
        self,
        wrist_pos_cam: np.ndarray,      # (3,)
        wrist_rotvec_cam: np.ndarray,   # (3,)
        finger6: np.ndarray,            # (6,)
    ) -> np.ndarray:
        """
        返回 21维关键点：
          wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3) = 21维
        顺序：wrist, thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, wrist_rotvec
        """
        wrist_pos_cam = np.asarray(wrist_pos_cam, dtype=np.float64).copy().reshape(3)
        wrist_rotvec_cam = np.asarray(wrist_rotvec_cam, dtype=np.float64).copy().reshape(3)
        finger6 = np.asarray(finger6, dtype=np.float64).copy().reshape(6)

        # 1) 手部在"自身根坐标系"（hand_base_link）下做 FK
        q = pin.neutral(self.model).copy()

        # 写入 6 个主动关节
        for jn, v in zip(self.finger_joint_names_6, finger6):
            self._set_1dof_joint(q, jn, v)

        # 写入 mimic 关节
        self._apply_mimics(q)

        # FK + frame placements
        pin.framesForwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        # 2) 变换到 camera 坐标系：p_cam = R * p_hand + t
        R_cam = R.from_rotvec(wrist_rotvec_cam).as_matrix()
        t_cam = wrist_pos_cam

        tips_cam: List[np.ndarray] = []
        for fid in self.tip_frame_ids_5:
            p_hand = np.asarray(self.data.oMf[fid].translation, dtype=np.float64).reshape(3)
            p_cam = R_cam @ p_hand + t_cam
            tips_cam.append(p_cam)

        # 组装 21维：wrist_xyz(3) + 5tips(15) + wrist_rotvec(3)
        key_points = np.concatenate([wrist_pos_cam] + tips_cam + [wrist_rotvec_cam])
        return key_points


class PolicyFourierHandKeypoints:
    """
    用于 policy 的 Fourier 手关键点 FK v2：
    
    输入（来自 policy 输出的 state）：
      - left_arm/right_arm: wrist_pose，最后一维 >=6
            [pos_x, pos_y, pos_z, rotvec_x, rotvec_y, rotvec_z, ...]
      - left_hand/right_hand: finger6，最后一维 ==6
            [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
      - waist: 腰部参数，最后一维 ==3

    输出格式（45维）：
      - left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
      - right_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
      - waist(3)
      
    支持形状：
      - (B, T, D) 批量时序
      - (B, D) 批量单帧
      - (D,) 单样本
      
    """

    def __init__(self, left_urdf: Path, right_urdf: Path):
        self.fk_L = FourierHandFK(left_urdf, side="L")
        self.fk_R = FourierHandFK(right_urdf, side="R")

    def compute_state_45d(
        self,
        left_arm: np.ndarray,    # (..., >=6)
        left_hand: np.ndarray,   # (..., 6)
        right_arm: np.ndarray,   # (..., >=6)
        right_hand: np.ndarray,  # (..., 6)
        waist: np.ndarray,       # (..., 3)
    ) -> np.ndarray:
        """
        计算45维state格式
        
        Returns:
            output: (..., 45) 与输入的batch/time维度对齐
        """
        la = np.asarray(left_arm, dtype=np.float64)
        lh = np.asarray(left_hand, dtype=np.float64)
        ra = np.asarray(right_arm, dtype=np.float64)
        rh = np.asarray(right_hand, dtype=np.float64)
        w = np.asarray(waist, dtype=np.float64)
        
        # 验证形状
        if la.shape[:-1] != lh.shape[:-1] or la.shape[:-1] != ra.shape[:-1] or \
           la.shape[:-1] != rh.shape[:-1] or la.shape[:-1] != w.shape[:-1]:
            raise ValueError(
                f"batch/time维度不匹配: la={la.shape}, lh={lh.shape}, "
                f"ra={ra.shape}, rh={rh.shape}, waist={w.shape}"
            )
        
        if la.shape[-1] < 6 or ra.shape[-1] < 6:
            raise ValueError(f"arm需要>=6维: la={la.shape}, ra={ra.shape}")
        if lh.shape[-1] != 6 or rh.shape[-1] != 6:
            raise ValueError(f"hand需要6维: lh={lh.shape}, rh={rh.shape}")
        if w.shape[-1] != 3:
            raise ValueError(f"waist需要3维: w={w.shape}")
        
        # 保存原始形状
        orig_shape = la.shape[:-1]
        
        # 展平到 (N, D)
        la_flat = la.reshape(-1, la.shape[-1])
        lh_flat = lh.reshape(-1, 6)
        ra_flat = ra.reshape(-1, ra.shape[-1])
        rh_flat = rh.reshape(-1, 6)
        w_flat = w.reshape(-1, 3)
        
        N = la_flat.shape[0]
        output = np.zeros((N, 45), dtype=np.float32)
        
        # 计算每个样本的FK
        for i in range(N):
            # 左手
            lw_pos = la_flat[i, 0:3]
            lw_rot = la_flat[i, 3:6]
            left_21d = self.fk_L.forward_keypoints_cam(lw_pos, lw_rot, lh_flat[i])
            
            # 右手
            rw_pos = ra_flat[i, 0:3]
            rw_rot = ra_flat[i, 3:6]
            right_21d = self.fk_R.forward_keypoints_cam(rw_pos, rw_rot, rh_flat[i])
            
            # 组装45维
            output[i] = np.concatenate([left_21d, right_21d, w_flat[i]])
        
        # 恢复原始形状
        output = output.reshape(*orig_shape, 45)
        return output

    def batch_process(
        self,
        states: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        方便的批量处理接口
        
        Args:
            states: 字典，包含 'left_arm', 'left_hand', 'right_arm', 'right_hand', 'waist'
        
        Returns:
            output: (..., 45)
        """
        return self.compute_state_45d(
            left_arm=states['left_arm'],
            left_hand=states['left_hand'],
            right_arm=states['right_arm'],
            right_hand=states['right_hand'],
            waist=states['waist']
        )


def test_single_sample():
    """测试单样本"""
    print("=" * 80)
    print("测试单样本FK")
    print("=" * 80)
    
    left_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_left_hand.urdf")
    right_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_right_hand.urdf")
    
    fk = PolicyFourierHandKeypoints(left_urdf, right_urdf)
    
    # 构造测试输入
    left_arm = np.array([-0.23888783, 0.13361477, 0.21128708, -1.33076850, 1.17501027, 1.17894613])
    left_hand = np.array([-0.00052188, -0.00058938, -0.00064615, -0.00051951, -0.00367418, -0.00115043])
    right_arm = np.array([0.24146207, 0.14099238, 0.21498004, -1.17017198, 1.31527550, 1.25513116])
    right_hand = np.array([-0.00192741, -0.00207026, -0.00212422, -0.00173645, -0.01637430, 0.00044878])
    waist = np.array([0.0, 0.0, 0.0])
    
    output = fk.compute_state_45d(left_arm, left_hand, right_arm, right_hand, waist)
    
    print(f"输出形状: {output.shape}")
    print(f"输出维度: {output.shape[-1]}")
    print(f"\n左手关键点 (21维):")
    print(f"  wrist_xyz:       {output[0:3]}")
    print(f"  thumb_tip_xyz:   {output[3:6]}")
    print(f"  index_tip_xyz:   {output[6:9]}")
    print(f"  middle_tip_xyz:  {output[9:12]}")
    print(f"  ring_tip_xyz:    {output[12:15]}")
    print(f"  pinky_tip_xyz:   {output[15:18]}")
    print(f"  wrist_rotvec:    {output[18:21]}")
    print(f"\n右手关键点 (21维):")
    print(f"  wrist_xyz:       {output[21:24]}")
    print(f"  thumb_tip_xyz:   {output[24:27]}")
    print(f"  index_tip_xyz:   {output[27:30]}")
    print(f"  middle_tip_xyz:  {output[30:33]}")
    print(f"  ring_tip_xyz:    {output[33:36]}")
    print(f"  pinky_tip_xyz:   {output[36:39]}")
    print(f"  wrist_rotvec:    {output[39:42]}")
    print(f"\nwaist (3维):       {output[42:45]}")


def test_batch():
    """测试批量处理"""
    print("\n" + "=" * 80)
    print("测试批量处理")
    print("=" * 80)
    
    left_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_left_hand.urdf")
    right_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_right_hand.urdf")
    
    fk = PolicyFourierHandKeypoints(left_urdf, right_urdf)
    
    # 构造批量输入 (B=3, T=5)
    B, T = 3, 5
    left_arm = np.random.randn(B, T, 6)
    left_hand = np.random.randn(B, T, 6) * 0.01
    right_arm = np.random.randn(B, T, 6)
    right_hand = np.random.randn(B, T, 6) * 0.01
    waist = np.random.randn(B, T, 3) * 0.1
    
    output = fk.compute_state_45d(left_arm, left_hand, right_arm, right_hand, waist)
    
    print(f"输入形状: left_arm={left_arm.shape}, left_hand={left_hand.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输出维度: {output.shape[-1]}")
    print(f"验证: {'✓' if output.shape == (B, T, 45) else '✗'}")


def test_dict_interface():
    """测试字典接口"""
    print("\n" + "=" * 80)
    print("测试字典接口")
    print("=" * 80)
    
    left_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_left_hand.urdf")
    right_urdf = Path("/vla/users/lijiayi/code/robot_retarget_for_fourier/retarget/assets/robots/hands/fourier_hand/fourier_right_hand.urdf")
    
    fk = PolicyFourierHandKeypoints(left_urdf, right_urdf)
    
    # 使用字典接口
    states = {
        'left_arm': np.random.randn(2, 6),
        'left_hand': np.random.randn(2, 6) * 0.01,
        'right_arm': np.random.randn(2, 6),
        'right_hand': np.random.randn(2, 6) * 0.01,
        'waist': np.random.randn(2, 3) * 0.1,
    }
    
    output = fk.batch_process(states)
    
    print(f"输入: 字典包含 {list(states.keys())}")
    print(f"输出形状: {output.shape}")
    print(f"验证: {'✓' if output.shape == (2, 45) else '✗'}")




if __name__ == "__main__":
    # 运行测试
    test_single_sample()
    test_batch()
    test_dict_interface()
    
    print("\n" + "=" * 80)
    print("所有测试完成！")
    print("=" * 80)
