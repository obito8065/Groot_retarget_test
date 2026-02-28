"""
测试脚本：使用 hand_base_link 作为末端进行 IK 求解
输入：左手末端位姿（相机坐标系下的位置和轴角表示）
输出：7自由度关节角度
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import PyKDL as kdl
from gr1_pos_transform import BodyRetargeter


class BodyRetargeterHandBase(BodyRetargeter):
    """修改版本：使用 hand_base_link 作为末端链接"""
    
    def __init__(self, urdf_path: Path, camera_intrinsics: dict):
        print("正在初始化 BodyRetargeterHandBase (使用 hand_base_link 作为末端)...")
        self.camera_intrinsics = camera_intrinsics
        
        robot_urdf = URDF.from_xml_file(str(urdf_path))
        tree = kdl_tree_from_urdf_model(robot_urdf)
        base_link = "torso_link"
        
        # 修改：使用 hand_base_link 作为末端
        self.kin_head = KDLKinematics(robot_urdf, base_link, "head_pitch_link", tree)
        self.kin_left_arm = KDLKinematics(robot_urdf, base_link, "L_hand_base_link", tree)
        self.kin_right_arm = KDLKinematics(robot_urdf, base_link, "R_hand_base_link", tree)
        
        # 直接保存 PyKDL chain 以便 IK 求解使用
        self.chain_head = tree.getChain(base_link, "head_pitch_link")
        self.chain_left_arm = tree.getChain(base_link, "L_hand_base_link")
        self.chain_right_arm = tree.getChain(base_link, "R_hand_base_link")
        
        # 定义相机在头部坐标系中的固定变换 (外参)
        t_cam_in_head = [2.650 - 2.65017178 + 0.23, -1.944 + 2.174 - 0.23, 1.538 - 1.4475]
        q_cam_in_head = [-0.205, 0.676, -0.676, 0.205]
        T_cam_in_head = self._create_transform(t_cam_in_head, q_cam_in_head)
        self.T_head_to_cam = np.linalg.inv(T_cam_in_head)
        
        # 工作空间和IK相关参数
        self._chain_max_reach = {
            "head": self._compute_chain_max_reach(self.chain_head),
            "left_arm": self._compute_chain_max_reach(self.chain_left_arm),
            "right_arm": self._compute_chain_max_reach(self.chain_right_arm),
        }
        self._workspace_margin = 0.05
        self._ik_local_margin = np.deg2rad(90.0)
        self._ik_last_solution = {}
        self._ik_damping = 1e-3
        
        # 使用 URDF 关节限制
        self._use_urdf_joint_limits = True
        self._joint_limits = self._extract_joint_limits_from_urdf(robot_urdf)
        self._chain_joint_limits = {
            "left_arm": self._get_chain_joint_limits(self.kin_left_arm),
            "right_arm": self._get_chain_joint_limits(self.kin_right_arm),
            "head": self._get_chain_joint_limits(self.kin_head),
        }
        
        # 打印关节限制信息
        print("\n=== URDF 关节限制 (使用 hand_base_link) ===")
        joint_names = self.kin_left_arm.get_joint_names()
        q_min, q_max = self._chain_joint_limits["left_arm"]
        print(f"\nleft_arm (7自由度):")
        for i, name in enumerate(joint_names):
            print(f"  {name}: [{np.rad2deg(q_min[i]):.1f}°, {np.rad2deg(q_max[i]):.1f}°]")
        print("=" * 50)
        
        print("初始化完成。")
    
    def test_left_hand_ik(self, 
                          left_hand_pos: np.ndarray,
                          left_hand_axisangle: np.ndarray,
                          q_init_left: np.ndarray = None):
        """
        测试左手 IK 求解
        
        参数:
            left_hand_pos: 左手位置 (3,) - 相机坐标系
            left_hand_axisangle: 左手轴角表示 (3,) - 相机坐标系
            q_init_left: 初始关节角度猜测，如果为None则使用零角度
        
        返回:
            q_left_arm: 7自由度关节角度
        """
        print("\n" + "="*60)
        print("测试左手 IK 求解 (使用 hand_base_link)")
        print("="*60)
        
        # 打印输入信息
        print(f"\n输入位姿 (相机坐标系):")
        print(f"  位置: [{left_hand_pos[0]:.6f}, {left_hand_pos[1]:.6f}, {left_hand_pos[2]:.6f}]")
        print(f"  轴角: [{left_hand_axisangle[0]:.6f}, {left_hand_axisangle[1]:.6f}, {left_hand_axisangle[2]:.6f}]")
        
        # 将轴角转换为旋转矩阵
        left_hand_rot = R.from_rotvec(left_hand_axisangle).as_matrix()
        
        # 构建相机坐标系下的手部变换矩阵
        T_cam_to_left_hand = np.eye(4)
        T_cam_to_left_hand[:3, :3] = left_hand_rot
        T_cam_to_left_hand[:3, 3] = left_hand_pos
        
        print(f"\n变换矩阵 T_cam_to_left_hand:")
        print(T_cam_to_left_hand)
        
        # 假设 waist 和 neck 为零角度（简化测试）
        num_head_joints = len(self.kin_head.get_joint_names())
        q_head = [0.0] * num_head_joints
        
        # 计算 torso 到 head 的变换
        T_torso_to_head = self.kin_head.forward(q_head)
        
        # 坐标系转换（相机 -> head -> torso）
        T_cam_to_head = np.linalg.inv(self.T_head_to_cam)
        T_head_to_left_hand = T_cam_to_head @ T_cam_to_left_hand
        T_torso_to_left_hand = T_torso_to_head @ T_head_to_left_hand
        
        print(f"\n变换矩阵 T_torso_to_left_hand:")
        print(T_torso_to_left_hand)
        
        # 准备初始猜测
        num_left_joints = len(self.kin_left_arm.get_joint_names())
        if q_init_left is None:
            q_init_left = [0.0] * num_left_joints
        else:
            q_init_left = q_init_left.tolist() if isinstance(q_init_left, np.ndarray) else q_init_left
        
        print(f"\n初始关节角度猜测:")
        joint_names = self.kin_left_arm.get_joint_names()
        for i, name in enumerate(joint_names):
            print(f"  {name}: {np.rad2deg(q_init_left[i]):.2f}°")
        
        # IK 求解
        print(f"\n开始 IK 求解...")
        q_left_arm = self._solve_ik(
            self.chain_left_arm, 
            T_torso_to_left_hand, 
            q_init_left,
            chain_name="left_arm", 
            env_idx=0
        )
        
        if q_left_arm is not None:
            print(f"\n✓ IK 求解成功！")
            print(f"\n输出关节角度 (7自由度):")
            for i, name in enumerate(joint_names):
                print(f"  {name}: {np.rad2deg(q_left_arm[i]):.4f}° ({q_left_arm[i]:.6f} rad)")
            
            # 验证：使用 FK 验证结果
            print(f"\n验证：使用 FK 验证结果...")
            T_torso_to_left_hand_verified = self.kin_left_arm.forward(q_left_arm)
            
            # 转换回相机坐标系
            T_head_to_left_hand_verified = np.linalg.inv(T_torso_to_head) @ T_torso_to_left_hand_verified
            T_cam_to_left_hand_verified = self.T_head_to_cam @ T_head_to_left_hand_verified
            
            pos_verified = T_cam_to_left_hand_verified[:3, 3]
            rot_verified = T_cam_to_left_hand_verified[:3, :3]
            
            pos_error = np.linalg.norm(left_hand_pos - pos_verified)
            rot_error = R.from_matrix(left_hand_rot.T @ rot_verified).magnitude()
            
            print(f"\n验证结果:")
            print(f"  位置误差: {pos_error:.6f} m")
            print(f"  姿态误差: {np.rad2deg(rot_error):.4f}°")
            
            if pos_error < 0.01 and np.rad2deg(rot_error) < 5.0:
                print(f"  ✓ 验证通过！")
            else:
                print(f"  ⚠ 验证警告：误差较大")
            
            return np.array(q_left_arm)
        else:
            print(f"\n✗ IK 求解失败！")
            return None


def main():
    """主测试函数"""
    # URDF 路径
    urdf_path = Path("/vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf")
    
    # 相机内参
    camera_intrinsics = {
        'fx': 502.8689,
        'fy': 502.8689,
        'cx': 640.0,
        'cy': 400.0
    }
    
    # 创建修改后的 retargeter
    retargeter = BodyRetargeterHandBase(urdf_path, camera_intrinsics)
    
    # 测试输入：左手末端位姿（相机坐标系）
    # 位置: -0.117341 0.226418 0.424539
    # 轴角: -1.215335 2.056101 1.160733
    left_hand_pos = np.array([-0.117341, 0.226418, 0.424539], dtype=np.float64)
    left_hand_axisangle = np.array([-1.215335, 2.056101, 1.160733], dtype=np.float64)
    
    # 执行 IK 求解
    q_left_arm = retargeter.test_left_hand_ik(
        left_hand_pos=left_hand_pos,
        left_hand_axisangle=left_hand_axisangle,
        q_init_left=None  # 使用零角度作为初始猜测
    )
    
    if q_left_arm is not None:
        print("\n" + "="*60)
        print("最终输出：7自由度关节角度")
        print("="*60)
        joint_names = retargeter.kin_left_arm.get_joint_names()
        print("\n关节角度 (弧度):")
        for i, name in enumerate(joint_names):
            print(f"  {name}: {q_left_arm[i]:.6f}")
        
        print("\n关节角度 (度):")
        for i, name in enumerate(joint_names):
            print(f"  {name}: {np.rad2deg(q_left_arm[i]):.4f}")
        
        print("\n数组格式 (可直接复制):")
        print(f"  {q_left_arm.tolist()}")


if __name__ == "__main__":
    main()
