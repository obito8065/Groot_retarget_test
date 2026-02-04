#!/usr/bin/env python3
"""
Fourier Hand Retarget API v2 - 严格按照原始retarget脚本实现
包含warmup和完整的retarget流程

输入格式（45维）:
(1)读取v4数据集中的state/action字段，格式为：
- left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- right_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- waist(3)
(2)读取原数据集中的state/action字段，格式为：
- left_arm(7)
- left_hand(6) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
- left_leg(6)
- neck(3)
- right_arm(7)
- right_hand(6)
- right_leg(6)
- waist(3)

输出格式（保持不变）:
- left_wrist_pose: (6,) [pos(3), rotvec(3)]
- left_finger_joints: (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]

测试结果：
   1. 输出手腕xyz+rotvec与原数据集的差距 6*2子图
   2. 输出手指关节与原数据集的差距  6*2子图

测试脚本用法：
```bash
python /vla/users/lijiayi/code/groot_retarget/gr00t/eval/fourier_hand_retarget_api_hand_joint_test.py \
    --parquet_file  /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v3/data/chunk-000/episode_000000.parquet \
    --original_parquet_file /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300/data/chunk-000/episode_000000.parquet \
    --action_key action \
    --output_plot /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v4 \
    --max_frames 1000
```
observation.state
参数说明：
- --parquet_file: Parquet文件路径（训练数据集）
- --action_key: Action字段的key名称，默认'action'
- --output_plot: 输出plot图片的路径，如果不指定则不保存
- --max_frames: 最大处理帧数，如果不指定则处理所有帧

API用法示例：
```python
from gr00t.eval.fourier_hand_retarget_api_test import FourierHandRetargetAPI

# 初始化（只需初始化一次）
retargeter = FourierHandRetargetAPI()

# 在每个episode开始时reset
retargeter.reset()

# 处理每一帧（输入45维）
state_45d = np.array([...])  # (45,) 从模型输出
result = retargeter.retarget_from_45d(state_45d)

# 使用结果
left_wrist_pose = result['left']['wrist_pose']  # (6,): [pos(3), rotvec(3)]
left_finger_joints = result['left']['finger_joints']  # (6,): [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
```
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, List
from scipy.spatial.transform import Rotation as R

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置 SAPIEN 环境变量（headless模式）
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 添加retarget路径（指向robot_retarget_for_anything）
retarget_src_path = Path("/vla/users/lijiayi/code/robot_retarget_for_anything/retarget/src")
if retarget_src_path.exists():
    sys.path.insert(0, str(retarget_src_path))
else:
    # 备用路径
    retarget_src_path = Path(__file__).parent.parent.parent.parent / "retarget" / "src"
    sys.path.insert(0, str(retarget_src_path))

# 导入核心retargeting组件
from dex_retargeting.constants import HandType, RetargetingType, RobotName, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting import yourdfpy as urdf
from pytransform3d import rotations

# 导入 SAPIEN（用于创建 robot 和计算 FK）
import sapien


class FourierHandRetargetAPI:
    """
    Fourier灵巧手Retarget API v2 - 严格按照原始retarget脚本实现
    
    关键特性:
    1. ✅ 包含warmup处理（episode开始的前几帧）
    2. ✅ 支持45维输入格式（与训练数据对齐）
    3. ✅ 严格遵循原始retarget脚本的处理流程
    4. ✅ 输出格式保持不变
    
    输入格式（45维）:
        - left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        - right_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        - waist(3)
    
    输出格式:
        {
            'left': {
                'wrist_pose': (6,) [pos_xyz(3), rotvec_xyz(3)],
                'finger_joints': (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            },
            'right': {...}
        }
        
    注意: finger_joints顺序为6个主动关节: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
          这是retarget配置文件中target_joint_names的顺序，可直接用于仿真控制
    """
    
    def __init__(
        self, 
        robot_name: str = "fourier",
        hand_sides: List[str] = ["left", "right"],
        wrist_enhance_weight: float = 2.0,
        warm_up_steps: int = 1,
    ):
        """
        初始化Retarget API v2
        
        Args:
            robot_name: 机器人名称，默认"fourier"
            hand_sides: 手部列表，默认["left", "right"]
            wrist_enhance_weight: 手腕优化权重，默认2.0
            warm_up_steps: warmup的帧数，默认1（即第一帧进行warmup）
        """
        # 转换robot_name为RobotName enum
        if hasattr(RobotName, robot_name):
            self.robot_name_enum = getattr(RobotName, robot_name)
        else:
            raise ValueError(f"Unknown robot: {robot_name}")
        
        self.hand_sides = hand_sides
        self.warm_up_steps = warm_up_steps
        self.retargetings = {}
        self.retarget2target = {}  # 已废弃，保留以兼容
        self.desired_joint_indices = {}  # 存储desired finger joints的索引
        
        # 用于跟踪episode状态
        self._episode_frame_count = {side: 0 for side in hand_sides}
        self._is_warmed_up = {side: False for side in hand_sides}
        
        # 用于轴角连续性处理：存储上一帧的四元数和轴角
        self._last_quaternion = {side: None for side in hand_sides}
        self._last_rotvec = {side: None for side in hand_sides}
        
        # 创建 SAPIEN scene 和 robots（用于 FK 计算，与 hand_robot_viewer_fourier.py 保持一致）
        self.scene = sapien.Scene()
        self.sapien_robots = {}  # 存储每个 side 的 SAPIEN robot 对象
        self.hand_base_link_indices = {}  # 存储 hand_base_link 的索引（通常是7）
        
        # 为每个手侧初始化retargeting
        for side in hand_sides:
            hand_type = getattr(HandType, side)
            
            # 加载配置
            config_path = get_default_config_path(
                self.robot_name_enum, 
                RetargetingType.position, 
                hand_type
            )
            config = RetargetingConfig.load_from_file(config_path)
            
            # 使用config的build方法创建retargeting实例
            retargeting = config.build()
            
            # 设置优化权重（与原始脚本一致）
            retargeting.optimizer.set_retarget_weight(wrist_enhance_weight=wrist_enhance_weight)
            
            # 定义6个主动finger joints（不包括dummy joints和mimic joints）
            # 这些是仿真需要的控制量，顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            prefix = 'L_' if side == 'left' else 'R_'
            desired_finger_joint_names = [
                f'{prefix}pinky_proximal_joint',
                f'{prefix}ring_proximal_joint',
                f'{prefix}middle_proximal_joint',
                f'{prefix}index_proximal_joint',
                f'{prefix}thumb_proximal_pitch_joint',
                f'{prefix}thumb_proximal_yaw_joint',
            ]
            
            # 计算这6个finger joints在retargeting.joint_names中的索引
            desired_joint_indices = []
            for joint_name in desired_finger_joint_names:
                try:
                    idx = retargeting.joint_names.index(joint_name)
                    desired_joint_indices.append(idx)
                except ValueError:
                    # 如果找不到，打印错误
                    print(f"Error: Cannot find joint {joint_name} in retargeting.joint_names")
                    print(f"  Available joints: {retargeting.joint_names}")
                    raise
            
            self.retargetings[side] = retargeting
            self.retarget2target[side] = None  # 不再需要这个映射
            self.desired_joint_indices[side] = np.array(desired_joint_indices, dtype=int)
            
            # 创建 SAPIEN robot（与 hand_robot_viewer_fourier.py 保持一致）
            loader = self.scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True
            
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
            robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
            temp_path = f"{temp_dir}/{urdf_name}"
            robot_urdf.write_xml_file(temp_path)
            
            sapien_robot = loader.load(temp_path)
            sapien_robot.set_name(f"{self.robot_name_enum.name}_{side}")
            self.sapien_robots[side] = sapien_robot
            
            # 获取 hand_base_link 的索引（通常是7，与 hand_robot_viewer_fourier.py 一致）
            # 查找 hand_base_link 在 links 中的索引
            links = sapien_robot.get_links()
            hand_base_link_idx = None
            for i, link in enumerate(links):
                if 'hand_base' in link.name.lower() or 'base' in link.name.lower():
                    hand_base_link_idx = i
                    break
            if hand_base_link_idx is None:
                # 如果找不到，使用索引7（与注释一致）
                hand_base_link_idx = 7
                print(f"Warning: Cannot find hand_base_link by name, using index {hand_base_link_idx}")
            self.hand_base_link_indices[side] = hand_base_link_idx
            
            # 调试信息
            print(f"\n[{side.upper()} hand]")
            print(f"  Desired 6 finger joints: {desired_finger_joint_names}")
            print(f"  Indices in qpos: {self.desired_joint_indices[side]}")
            print(f"  Total joints in qpos: {len(retargeting.joint_names)}")
            print(f"  Hand base link index: {hand_base_link_idx} (link name: {links[hand_base_link_idx].name})")
        
        print(f"[FourierHandRetargetAPI] Initialized successfully")
        print(f"  Robot: {robot_name}, Sides: {hand_sides}")
        print(f"  Wrist enhance weight: {wrist_enhance_weight}")
        print(f"  Warmup steps: {warm_up_steps}")
    
    def _ensure_axisangle_continuity(self, current_rotvec: np.ndarray, side: str) -> np.ndarray:
        """
        确保轴角表示的连续性，避免等价表示之间的跳变。
        
        处理两种等价表示：
        1. 四元数 q 和 -q 表示同一个旋转
        2. 轴角 r 和 -r + 2πk 表示同一个旋转
        
        参数:
            current_rotvec: 当前帧的轴角 (3,)
            side: 'left' 或 'right'
        
        返回:
            修正后的轴角
        """
        # 将当前轴角转换为四元数 (scipy格式: x, y, z, w)
        current_quat = R.from_rotvec(current_rotvec).as_quat()
        
        # 如果没有历史记录，直接保存四元数和轴角并返回原轴角
        if self._last_quaternion[side] is None:
            self._last_quaternion[side] = current_quat.copy()
            self._last_rotvec[side] = current_rotvec.copy()
            return current_rotvec
        
        last_quat = self._last_quaternion[side]
        last_rotvec = self._last_rotvec[side]
        
        # 第一步：处理四元数符号跳变（q 和 -q）
        quat_dot = np.dot(current_quat, last_quat)
        if quat_dot < 0:
            current_quat = -current_quat
            current_rotvec = R.from_quat(current_quat).as_rotvec()
        
        # 第二步：处理轴角等价表示跳变（r 和 -r + 2πk）
        # 计算直接差值
        direct_diff = np.linalg.norm(current_rotvec - last_rotvec)
        
        # 计算实际旋转角度（通过旋转矩阵）
        R_last = R.from_rotvec(last_rotvec).as_matrix()
        R_curr = R.from_rotvec(current_rotvec).as_matrix()
        R_diff = R_last.T @ R_curr
        actual_angle = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
        
        # 如果直接差值很大但实际旋转角度很小，说明是等价表示跳变
        if direct_diff > 2.0 and actual_angle < direct_diff * 0.5:
            # 尝试 -r + 2πk 的等价表示
            angle = np.linalg.norm(current_rotvec)
            if angle > 1e-6:
                axis = current_rotvec / angle
                # 尝试 -r + 2πk 表示
                alternative_angle = 2 * np.pi - angle
                alternative_rotvec = -axis * alternative_angle
                
                # 检查哪个更接近上一帧
                diff_original = np.linalg.norm(current_rotvec - last_rotvec)
                diff_alternative = np.linalg.norm(alternative_rotvec - last_rotvec)
                
                if diff_alternative < diff_original:
                    current_rotvec = alternative_rotvec
                    # 更新四元数
                    current_quat = R.from_rotvec(current_rotvec).as_quat()
        
        # 更新历史记录
        self._last_quaternion[side] = current_quat.copy()
        self._last_rotvec[side] = current_rotvec.copy()
        
        return current_rotvec
        
    def reset(self, env_idx: Optional[int] = None):
        """
        重置API状态（新episode或新环境开始时调用）
        
        重要: 每个新episode开始前必须调用此方法！
        这会重置warmup状态和last_qpos缓存，确保新episode的前几帧会进行正确的warmup处理。
        
        Args:
            env_idx: 并行环境索引（与policy.py的reset_ik_cache保持一致）
                - None: 重置所有环境
                - int: 仅重置指定环境的状态（目前实现为重置所有，因为retarget是无状态的）
        
        注意：
            - Position retargeting每次都是独立优化，但warmup状态需要在新episode时重置
            - env_idx参数主要用于接口一致性，实际上每次reset都会重置所有状态
        """
        # 重置帧计数和warmup状态
        for side in self.hand_sides:
            self._episode_frame_count[side] = 0
            self._is_warmed_up[side] = False
            self._last_quaternion[side] = None  # 重置四元数历史
            self._last_rotvec[side] = None  # 重置轴角历史
        
        # 重置debug信息
        if hasattr(self, '_debug_info'):
            self._debug_info = {}
        
        # 如果未来需要缓存last_qpos，在此根据env_idx清理
        # if env_idx is not None:
        #     清理指定环境的缓存
        # else:
        #     清理所有环境的缓存
        
        print(f"[FourierHandRetargetAPI] Reset for new episode (env_idx={env_idx})")
    
    def _warmup(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray, side: str):
        """
        执行warmup（与原始脚本的multi_robot_warmup一致）
        
        Args:
            wrist_pos: 手腕位置 (3,)
            wrist_quat: 手腕四元数旋转 (4,) [w, x, y, z]
            side: 'left' 或 'right'
        
        注意:
            - 这对应于原始脚本的 multi_robot_warmup() 和 warm_start()
            - warmup使用手腕的wrist position作为joint输入
            - is_mano_convention=True 表示使用MANO坐标系约定
        """
        hand_type = getattr(HandType, side)
        retargeting = self.retargetings[side]
        
        # 调用warm_start（与原始脚本line 526-531完全一致）
        warmup_qpos6d = retargeting.warm_start(
            wrist_pos=wrist_pos,         # 手腕3D位置
            wrist_quat=wrist_quat,       # 手腕四元数旋转 [w, x, y, z]
            hand_type=hand_type,         # 左手或右手
            # is_mano_convention=True,     # 使用MANO坐标系约定
            is_mano_convention=False,     # 不使用MANO坐标系约定，默认相机坐标系
        )
        
        self._is_warmed_up[side] = True
        return warmup_qpos6d
    
    def retarget_from_45d(
        self,
        state_45d: np.ndarray,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        从45维state进行retargeting
        
        Args:
            state_45d: (45,) 格式：
                [0:21]   - left_key_points: wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
                [21:42]  - right_key_points: wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
                [42:45]  - waist(3)
        
        Returns:
            {
                'left': {
                    'wrist_pose': (6,) [pos_xyz(3), euler_xyz(3)],
                    'finger_joints': (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                },
                'right': {...}
            }
        
        流程说明（严格按照原始retarget脚本）:
            1. 提取左右手的关键点数据
            2. 如果是episode的前N帧（warm_up_steps），执行warmup
            3. 执行retargeting
            4. 转换输出格式
        """
        state_45d = np.asarray(state_45d, dtype=np.float32).flatten()
        if state_45d.shape[0] != 45:
            raise ValueError(f"输入必须是45维，得到: {state_45d.shape}")
        
        result = {}
        
        # 处理左右手
        offsets = {
            'left': 0,   # [0:21]
            'right': 21, # [21:42]
        }
        
        for side in self.hand_sides:
            offset = offsets[side]
            
            # 1. 从45维中提取21维关键点
            key_points_21 = state_45d[offset:offset+21]
            
            # 解析21维格式：wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
            wrist_xyz = key_points_21[0:3]           # 手腕位置
            thumb_tip = key_points_21[3:6]           # 拇指尖
            index_tip = key_points_21[6:9]           # 食指尖
            middle_tip = key_points_21[9:12]         # 中指尖
            ring_tip = key_points_21[12:15]          # 无名指尖
            pinky_tip = key_points_21[15:18]         # 小指尖
            wrist_rotvec = key_points_21[18:21]      # 手腕轴角旋转
            
            # 2. 组装6个关键点 [wrist, thumb, index, middle, ring, pinky]
            keypoints_6x3 = np.stack([
                wrist_xyz,
                thumb_tip,
                index_tip,
                middle_tip,
                ring_tip,
                pinky_tip,
            ], axis=0)  # (6, 3)
            
            # 3. Warmup处理（与原始脚本line 561-566一致）
            if self._episode_frame_count[side] < self.warm_up_steps:
                # 将rotvec转换为四元数（与原始脚本line 564一致）
                wrist_quat = R.from_rotvec(wrist_rotvec).as_quat()  # [x, y, z, w]
                
                # scipy返回的是[x,y,z,w]，需要转换为[w,x,y,z]
                wrist_quat_wxyz = np.array([
                    wrist_quat[3],  # w
                    wrist_quat[0],  # x
                    wrist_quat[1],  # y
                    wrist_quat[2],  # z
                ])
                
                # 执行warmup（使用手腕位置作为joint）
                self._warmup(
                    wrist_pos=wrist_xyz,      # 手腕3D位置
                    wrist_quat=wrist_quat_wxyz, # [w,x,y,z]
                    side=side
                )
            
            # 4. 重排序关键点以匹配MANO格式
            # 从 [wrist, thumb, index, middle, ring, pinky] 
            # 到 [thumb, index, middle, ring, pinky, wrist]
            kp_reordered = keypoints_6x3[[1, 2, 3, 4, 5, 0]]  # (6, 3)
            
            # 5. 构造21点MANO数组（只填充我们有的6个点）
            hand_21 = np.zeros((21, 3), dtype=np.float32)
            # MANO 21点索引：[4, 8, 12, 16, 20, 0] 对应 [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, wrist]
            hand_21[[4, 8, 12, 16, 20, 0]] = kp_reordered
            
            # 6. 根据target_link_human_indices提取需要的关键点
            retargeting = self.retargetings[side]
            human_indices = retargeting.optimizer.target_link_human_indices
            human_keypoints = hand_21[human_indices, :]  # (N, 3)
            
            # 7. Retarget（与原始脚本line 568一致）
            # 返回完整的qpos（包含所有关节）
            # qpos格式: [dummy_joints(6), ...其他关节(N)...]
            # dummy_joints = [x_trans, y_trans, z_trans, x_rot, y_rot, z_rot]
            
            # Debug: 记录retarget前的关键点（wrist位置）
            # target_link_human_indices的最后一个索引对应wrist
            wrist_idx_in_human_keypoints = len(human_keypoints) - 1  # 假设wrist是最后一个
            target_wrist_kp = human_keypoints[wrist_idx_in_human_keypoints]  # 输入到retargeting的wrist位置
            
            qpos_full = retargeting.retarget(human_keypoints)
            
            # 8. 通过 SAPIEN FK 计算 hand_base_link 的实际位姿（与 hand_robot_viewer_fourier.py 保持一致）
            # 因为 dummy joint 给出的是 root_link 的位姿，不是 hand_base_link 的位姿
            sapien_robot = self.sapien_robots[side]
            sapien_robot.set_qpos(qpos_full)
            hand_base_link_idx = self.hand_base_link_indices[side]
            link_pose = sapien_robot.get_links()[hand_base_link_idx].entity_pose  # sapien.Pose
            
            # 提取位置和旋转（与 hand_robot_viewer_fourier.py line 505-507 一致）
            wrist_pos = link_pose.p  # (3,) 位置
            wrist_rotvec_raw = rotations.compact_axis_angle_from_quaternion(link_pose.q)  # (3,) 轴角表示
            
            # Debug: 检查retarget后的wrist位置是否匹配输入
            wrist_pos_error = np.linalg.norm(wrist_pos - target_wrist_kp)
            
            # 应用轴角连续性处理，避免等价表示跳变
            wrist_rotvec = self._ensure_axisangle_continuity(wrist_rotvec_raw, side)
            
            # Debug: 存储debug信息（如果需要在外部访问）
            if not hasattr(self, '_debug_info'):
                self._debug_info = {}
            if side not in self._debug_info:
                self._debug_info[side] = []
            
            self._debug_info[side].append({
                'frame': self._episode_frame_count[side],
                'input_wrist_xyz': wrist_xyz.copy(),  # 原始输入的wrist位置
                'target_wrist_kp': target_wrist_kp.copy(),  # 输入到retargeting的wrist位置
                'output_wrist_pos': wrist_pos.copy(),  # retargeting后FK计算的wrist位置
                'wrist_pos_error': wrist_pos_error,
                'input_rotvec': wrist_rotvec.copy(),  # 原始输入的rotvec
                'output_rotvec_raw': wrist_rotvec_raw.copy(),  # FK计算的原始rotvec
                'output_rotvec_continuous': wrist_rotvec.copy(),  # 连续性处理后的rotvec
            })
            
            # 组合成 6D wrist pose: [pos(3), rotvec(3)]
            wrist_pose = np.concatenate([wrist_pos, wrist_rotvec]).astype(np.float32)
            
            # finger_joints: 从qpos_full中提取6个主动关节
            # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            # 这是config文件中target_joint_names的顺序，直接用于仿真控制
            desired_indices = self.desired_joint_indices[side]
            finger_joints = qpos_full[desired_indices]  # (6,)
            
            # 符号修正：为了与控制量统一，需要对某些关节取负号
            # [pinky, ring, middle, index, thumb_pitch, thumb_yaw] 中
            # pinky, ring, middle, index, thumb_yaw 需要取负号
            # thumb_pitch 不需要取负号
            finger_joints_corrected = finger_joints.copy()
            finger_joints_corrected[0] = -finger_joints[0]  # pinky
            finger_joints_corrected[1] = -finger_joints[1]  # ring
            finger_joints_corrected[2] = -finger_joints[2]  # middle
            finger_joints_corrected[3] = -finger_joints[3]  # index
            # finger_joints_corrected[4] = finger_joints[4]  # thumb_pitch (保持不变)
            finger_joints_corrected[5] = -finger_joints[5]  # thumb_yaw
            
            # 确保shape正确
            assert wrist_pose.shape == (6,), f"{side} wrist_pose shape错误: {wrist_pose.shape}, 期望(6,)"
            assert finger_joints_corrected.shape == (6,), f"{side} finger_joints shape错误: {finger_joints_corrected.shape}, 期望(6,). desired_indices={desired_indices}, qpos_full.shape={qpos_full.shape}"
            
            result[side] = {
                'wrist_pose': wrist_pose,
                'finger_joints': finger_joints_corrected,  # [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (已符号修正)
            }
            
            # 增加帧计数
            self._episode_frame_count[side] += 1
        
        return result



# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试Fourier Hand Retarget API")
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Parquet文件路径（训练数据集）"
    )
    parser.add_argument(
        "--action_key",
        type=str,
        default="action",
        help="Action字段的key名称，默认'action'"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="输出plot图片的路径，如果不指定则不保存"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="最大处理帧数，如果不指定则处理所有帧"
    )
    parser.add_argument(
        "--original_parquet_file",
        type=str,
        default=None,
        help="原始数据集Parquet文件路径（44维关节格式），用于对比手指关节。如果不指定，则只生成手腕对比图"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing FourierHandRetargetAPI with Parquet Dataset")
    print("=" * 80)
    
    # 初始化
    print("\n[1] 初始化 FourierHandRetargetAPI...")
    api = FourierHandRetargetAPI(warm_up_steps=1)
    
    # 模拟episode开始
    print("\n[2] 开始新episode（调用reset）...")
    api.reset()
    
    # 读取parquet文件
    parquet_path = Path(args.parquet_file)
    print(f"\n[3] 读取Parquet文件: {parquet_path}")
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet文件不存在: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"   成功读取 {len(df)} 行数据")
    
    # 检查action字段
    if args.action_key not in df.columns:
        raise ValueError(f"Action字段 '{args.action_key}' 不存在于parquet文件中。可用字段: {df.columns.tolist()}")
    
    # 提取action数据
    action_data = df[args.action_key].values
    print(f"   Action数据类型: {type(action_data[0])}")
    
    # 处理action数据：可能是字典、列表或numpy数组
    actions_list = []
    for i, act in enumerate(action_data):
        if isinstance(act, dict):
            # 如果是字典，尝试提取45维数组
            # 可能的格式：{'left_key_points': [...], 'right_key_points': [...], 'waist': [...]}
            # 或者已经flatten的格式
            if 'left_key_points' in act and 'right_key_points' in act:
                left_kp = np.array(act['left_key_points']).flatten()
                right_kp = np.array(act['right_key_points']).flatten()
                waist = np.array(act.get('waist', [0, 0, 0])).flatten()
                combined = np.concatenate([left_kp, right_kp, waist])
            else:
                # 尝试将所有值flatten
                combined = np.concatenate([np.array(v).flatten() for v in act.values()])
            actions_list.append(combined)
        elif isinstance(act, (list, np.ndarray)):
            actions_list.append(np.array(act).flatten())
        else:
            raise ValueError(f"无法处理action数据类型: {type(act)} at index {i}")
    
    actions = np.array(actions_list)
    print(f"   Action shape: {actions.shape}")
    
    if len(actions.shape) != 2 or actions.shape[1] != 45:
        raise ValueError(f"Action维度错误: 期望(N, 45)，实际为{actions.shape}")
    
    # 限制处理帧数
    num_frames = len(actions)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    actions = actions[:num_frames]
    
    print(f"   将处理 {num_frames} 帧数据")
    
    # 存储retarget前后的数据
    # 格式: {side: {dim: [values...]}}
    # dim: 'x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z'
    before_retarget = {
        'left': {'x': [], 'y': [], 'z': [], 'rotvec_x': [], 'rotvec_y': [], 'rotvec_z': []},
        'right': {'x': [], 'y': [], 'z': [], 'rotvec_x': [], 'rotvec_y': [], 'rotvec_z': []}
    }
    after_retarget = {
        'left': {'x': [], 'y': [], 'z': [], 'rotvec_x': [], 'rotvec_y': [], 'rotvec_z': []},
        'right': {'x': [], 'y': [], 'z': [], 'rotvec_x': [], 'rotvec_y': [], 'rotvec_z': []}
    }
    
    # 存储手指关节数据（用于功能2）
    # 格式: {side: {joint_name: [values...]}}
    # joint_name: 'pinky', 'ring', 'middle', 'index', 'thumb_pitch', 'thumb_yaw'
    original_finger_joints = {
        'left': {'pinky': [], 'ring': [], 'middle': [], 'index': [], 'thumb_pitch': [], 'thumb_yaw': []},
        'right': {'pinky': [], 'ring': [], 'middle': [], 'index': [], 'thumb_pitch': [], 'thumb_yaw': []}
    }
    retargeted_finger_joints = {
        'left': {'pinky': [], 'ring': [], 'middle': [], 'index': [], 'thumb_pitch': [], 'thumb_yaw': []},
        'right': {'pinky': [], 'ring': [], 'middle': [], 'index': [], 'thumb_pitch': [], 'thumb_yaw': []}
    }
    
    # 读取原始数据集（如果提供）
    original_df = None
    original_actions = None
    has_original_data = False
    if args.original_parquet_file:
        original_parquet_path = Path(args.original_parquet_file)
        if not original_parquet_path.exists():
            print(f"警告: 原始数据集文件不存在: {original_parquet_path}，将跳过手指关节对比")
        else:
            print(f"\n[3.5] 读取原始数据集Parquet文件: {original_parquet_path}")
            original_df = pd.read_parquet(original_parquet_path)
            print(f"   成功读取 {len(original_df)} 行数据")
            
            # 检查action字段
            if args.action_key not in original_df.columns:
                print(f"警告: Action字段 '{args.action_key}' 不存在于原始数据集中，将跳过手指关节对比")
            else:
                # 提取原始action数据（44维格式）
                original_action_data = original_df[args.action_key].values
                
                # 处理原始action数据
                original_actions_list = []
                for i, act in enumerate(original_action_data):
                    if isinstance(act, (list, np.ndarray)):
                        original_actions_list.append(np.array(act).flatten())
                    elif isinstance(act, dict):
                        # 如果是字典，尝试提取所有值
                        combined = np.concatenate([np.array(v).flatten() for v in act.values()])
                        original_actions_list.append(combined)
                    else:
                        print(f"警告: 无法处理原始action数据类型: {type(act)} at index {i}")
                        break
                
                if len(original_actions_list) > 0:
                    original_actions = np.array(original_actions_list)
                    print(f"   原始Action shape: {original_actions.shape}")
                    
                    if len(original_actions.shape) == 2 and original_actions.shape[1] == 44:
                        has_original_data = True
                        print(f"   ✅ 成功加载原始44维数据，将进行手指关节对比")
                    else:
                        print(f"警告: 原始Action维度错误: 期望(N, 44)，实际为{original_actions.shape}，将跳过手指关节对比")
                else:
                    print(f"警告: 无法解析原始action数据，将跳过手指关节对比")
    
    # 对每个时间步进行retarget
    print("\n[4] 对每个时间步进行retarget...")
    print("=" * 80)
    
    # Debug统计信息
    debug_stats = {
        'left': {
            'max_pos_error': 0.0,
            'max_rotvec_error': 0.0,
            'max_actual_rot_error': 0.0,
            'frames_with_large_error': []
        },
        'right': {
            'max_pos_error': 0.0,
            'max_rotvec_error': 0.0,
            'max_actual_rot_error': 0.0,
            'frames_with_large_error': []
        }
    }
    
    for t in range(num_frames):
        if (t + 1) % 100 == 0:
            print(f"  处理进度: {t+1}/{num_frames}")
        
        state_45d = actions[t].astype(np.float32)
        
        # 提取retarget前的数据（从输入中提取）
        # left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        # right_key_points(21): 同上
        left_wrist_xyz_before = state_45d[0:3]
        left_wrist_rotvec_before = state_45d[18:21]
        right_wrist_xyz_before = state_45d[21:24]
        right_wrist_rotvec_before = state_45d[39:42]
        
        # 记录retarget前的数据
        before_retarget['left']['x'].append(left_wrist_xyz_before[0])
        before_retarget['left']['y'].append(left_wrist_xyz_before[1])
        before_retarget['left']['z'].append(left_wrist_xyz_before[2])
        before_retarget['left']['rotvec_x'].append(left_wrist_rotvec_before[0])
        before_retarget['left']['rotvec_y'].append(left_wrist_rotvec_before[1])
        before_retarget['left']['rotvec_z'].append(left_wrist_rotvec_before[2])
        
        before_retarget['right']['x'].append(right_wrist_xyz_before[0])
        before_retarget['right']['y'].append(right_wrist_xyz_before[1])
        before_retarget['right']['z'].append(right_wrist_xyz_before[2])
        before_retarget['right']['rotvec_x'].append(right_wrist_rotvec_before[0])
        before_retarget['right']['rotvec_y'].append(right_wrist_rotvec_before[1])
        before_retarget['right']['rotvec_z'].append(right_wrist_rotvec_before[2])
        
        # 执行retarget
        result = api.retarget_from_45d(state_45d)
        
        # 记录retarget后的数据并进行debug分析
        for side in ['left', 'right']:
            if side in result:
                wrist_pose = result[side]['wrist_pose']
                wrist_xyz = wrist_pose[0:3]
                wrist_rotvec = wrist_pose[3:6]
                
                # 获取输入数据
                if side == 'left':
                    input_xyz = left_wrist_xyz_before
                    input_rotvec = left_wrist_rotvec_before
                else:
                    input_xyz = right_wrist_xyz_before
                    input_rotvec = right_wrist_rotvec_before
                
                # 计算位置误差
                pos_error = np.linalg.norm(wrist_xyz - input_xyz)
                debug_stats[side]['max_pos_error'] = max(debug_stats[side]['max_pos_error'], pos_error)
                
                # 计算轴角数值误差
                rotvec_error = np.linalg.norm(wrist_rotvec - input_rotvec)
                debug_stats[side]['max_rotvec_error'] = max(debug_stats[side]['max_rotvec_error'], rotvec_error)
                
                # 计算实际旋转角度误差（通过旋转矩阵）
                R_input = R.from_rotvec(input_rotvec).as_matrix()
                R_output = R.from_rotvec(wrist_rotvec).as_matrix()
                R_diff = R_input.T @ R_output  # 相对旋转
                actual_rotvec_diff = R.from_matrix(R_diff).as_rotvec()
                actual_rot_error = np.linalg.norm(actual_rotvec_diff)
                actual_rot_error_deg = np.degrees(actual_rot_error)
                debug_stats[side]['max_actual_rot_error'] = max(debug_stats[side]['max_actual_rot_error'], actual_rot_error_deg)
                
                # 如果误差较大，记录详细信息
                if pos_error > 0.05 or actual_rot_error_deg > 10.0:  # 位置误差>5cm或旋转误差>10度
                    debug_stats[side]['frames_with_large_error'].append({
                        'frame': t,
                        'pos_error': pos_error,
                        'rotvec_error': rotvec_error,
                        'actual_rot_error_deg': actual_rot_error_deg,
                        'input_xyz': input_xyz.copy(),
                        'output_xyz': wrist_xyz.copy(),
                        'input_rotvec': input_rotvec.copy(),
                        'output_rotvec': wrist_rotvec.copy(),
                    })
                
                # 每50帧打印一次详细debug信息
                if t % 50 == 0 or (pos_error > 0.05 or actual_rot_error_deg > 10.0):
                    print(f"\n[DEBUG Frame {t}] {side.upper()} hand:")
                    print(f"  Position:")
                    print(f"    Input:  [{input_xyz[0]:.4f}, {input_xyz[1]:.4f}, {input_xyz[2]:.4f}]")
                    print(f"    Output: [{wrist_xyz[0]:.4f}, {wrist_xyz[1]:.4f}, {wrist_xyz[2]:.4f}]")
                    print(f"    Error:  {pos_error:.4f} m ({pos_error*100:.2f} cm)")
                    print(f"  Rotation (axis-angle):")
                    print(f"    Input:  [{input_rotvec[0]:.4f}, {input_rotvec[1]:.4f}, {input_rotvec[2]:.4f}]")
                    print(f"    Output: [{wrist_rotvec[0]:.4f}, {wrist_rotvec[1]:.4f}, {wrist_rotvec[2]:.4f}]")
                    print(f"    Rotvec norm error: {rotvec_error:.4f}")
                    print(f"    Actual rotation error: {actual_rot_error_deg:.2f}°")
                    
                    # 检查warmup状态
                    is_warmed = api._is_warmed_up.get(side, False)
                    frame_count = api._episode_frame_count.get(side, 0)
                    print(f"  Warmup status: {is_warmed} (frame_count={frame_count})")
                    
                    # 检查retargeting内部debug信息
                    if hasattr(api, '_debug_info') and side in api._debug_info and len(api._debug_info[side]) > 0:
                        debug_info = api._debug_info[side][-1]  # 最新的一帧
                        if 'wrist_pos_error' in debug_info:
                            print(f"  Retargeting内部wrist位置误差: {debug_info['wrist_pos_error']:.4f} m")
                            print(f"    输入到retargeting的wrist位置: [{debug_info['target_wrist_kp'][0]:.4f}, "
                                  f"{debug_info['target_wrist_kp'][1]:.4f}, {debug_info['target_wrist_kp'][2]:.4f}]")
                            print(f"    FK计算的wrist位置: [{debug_info['output_wrist_pos'][0]:.4f}, "
                                  f"{debug_info['output_wrist_pos'][1]:.4f}, {debug_info['output_wrist_pos'][2]:.4f}]")
                
                after_retarget[side]['x'].append(wrist_xyz[0])
                after_retarget[side]['y'].append(wrist_xyz[1])
                after_retarget[side]['z'].append(wrist_xyz[2])
                after_retarget[side]['rotvec_x'].append(wrist_rotvec[0])
                after_retarget[side]['rotvec_y'].append(wrist_rotvec[1])
                after_retarget[side]['rotvec_z'].append(wrist_rotvec[2])
                
                # 记录retarget后的手指关节（功能2）
                finger_joints = result[side]['finger_joints']  # (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                retargeted_finger_joints[side]['pinky'].append(finger_joints[0])
                retargeted_finger_joints[side]['ring'].append(finger_joints[1])
                retargeted_finger_joints[side]['middle'].append(finger_joints[2])
                retargeted_finger_joints[side]['index'].append(finger_joints[3])
                retargeted_finger_joints[side]['thumb_pitch'].append(finger_joints[4])
                retargeted_finger_joints[side]['thumb_yaw'].append(finger_joints[5])
        
        # 记录原始手指关节（功能2）
        if has_original_data and t < len(original_actions):
            original_action_44d = original_actions[t].astype(np.float32)
            # 原数据集格式：left_hand(6)在[7:13], right_hand(6)在[29:35]
            # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            # 注意：原数据集中的手指关节顺序与retarget输出一致
            if len(original_action_44d) >= 35:
                left_hand_original = original_action_44d[7:13]
                right_hand_original = original_action_44d[29:35]
                
                original_finger_joints['left']['pinky'].append(left_hand_original[0])
                original_finger_joints['left']['ring'].append(left_hand_original[1])
                original_finger_joints['left']['middle'].append(left_hand_original[2])
                original_finger_joints['left']['index'].append(left_hand_original[3])
                original_finger_joints['left']['thumb_pitch'].append(left_hand_original[4])
                original_finger_joints['left']['thumb_yaw'].append(left_hand_original[5])
                
                original_finger_joints['right']['pinky'].append(right_hand_original[0])
                original_finger_joints['right']['ring'].append(right_hand_original[1])
                original_finger_joints['right']['middle'].append(right_hand_original[2])
                original_finger_joints['right']['index'].append(right_hand_original[3])
                original_finger_joints['right']['thumb_pitch'].append(right_hand_original[4])
                original_finger_joints['right']['thumb_yaw'].append(right_hand_original[5])
            else:
                # 如果数据长度不够，填充NaN
                for joint_name in ['pinky', 'ring', 'middle', 'index', 'thumb_pitch', 'thumb_yaw']:
                    original_finger_joints['left'][joint_name].append(np.nan)
                    original_finger_joints['right'][joint_name].append(np.nan)
        elif has_original_data:
            # 如果原始数据帧数不足，填充NaN以保持长度一致
            for joint_name in ['pinky', 'ring', 'middle', 'index', 'thumb_pitch', 'thumb_yaw']:
                original_finger_joints['left'][joint_name].append(np.nan)
                original_finger_joints['right'][joint_name].append(np.nan)
    
    print(f"  完成！处理了 {num_frames} 帧数据")
    
    # 打印debug统计信息
    print("\n" + "=" * 80)
    print("[DEBUG] Retarget误差统计")
    print("=" * 80)
    for side in ['left', 'right']:
        stats = debug_stats[side]
        print(f"\n{side.upper()} hand:")
        print(f"  最大位置误差: {stats['max_pos_error']:.4f} m ({stats['max_pos_error']*100:.2f} cm)")
        print(f"  最大轴角数值误差: {stats['max_rotvec_error']:.4f}")
        print(f"  最大实际旋转误差: {stats['max_actual_rot_error']:.2f}°")
        print(f"  误差较大的帧数: {len(stats['frames_with_large_error'])}")
        
        if len(stats['frames_with_large_error']) > 0:
            print(f"\n  前10个误差较大的帧:")
            for i, err_info in enumerate(stats['frames_with_large_error'][:10]):
                print(f"    Frame {err_info['frame']}: pos_error={err_info['pos_error']:.4f}m, "
                      f"rot_error={err_info['actual_rot_error_deg']:.2f}°")
                print(f"      Input rotvec:  [{err_info['input_rotvec'][0]:.4f}, "
                      f"{err_info['input_rotvec'][1]:.4f}, {err_info['input_rotvec'][2]:.4f}]")
                print(f"      Output rotvec: [{err_info['output_rotvec'][0]:.4f}, "
                      f"{err_info['output_rotvec'][1]:.4f}, {err_info['output_rotvec'][2]:.4f}]")
    
    print("\n" + "=" * 80)
    
    # 生成对比图
    print("\n[5] 生成对比图...")
    
    # 功能1: 创建12个子图：左右手各6个维度（手腕位置和轴角）
    fig1, axes1 = plt.subplots(6, 2, figsize=(16, 20))
    fig1.suptitle('功能1: Retarget前后对比 - 左右手腕位置和轴角', fontsize=16, fontweight='bold')
    
    dim_names = ['x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z']
    dim_labels = ['X_position', 'Y_position', 'Z_position', 'X_rotvec', 'Y_rotvec', 'Z_rotvec']
    sides = ['left', 'right']
    side_labels = ['left_hand', 'right_hand']
    colors = {'before': 'red', 'after': 'blue'}
    
    time_steps = np.arange(num_frames)
    
    for row, (dim, dim_label) in enumerate(zip(dim_names, dim_labels)):
        for col, (side, side_label) in enumerate(zip(sides, side_labels)):
            ax = axes1[row, col]
            
            # 绘制retarget前后的数据
            ax.scatter(time_steps, before_retarget[side][dim], 
                      c=colors['before'], label='Retarget input', alpha=0.6, s=4)
            ax.scatter(time_steps, after_retarget[side][dim], 
                      c=colors['after'], label='Retarget output', alpha=0.6, s=4)
            
            ax.set_xlabel('timestep')
            ax.set_ylabel(dim_label)
            ax.set_title(f'{side_label} - {dim_label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存功能1的图片
    if args.output_plot:
        output_path = Path(args.output_plot)
        
        # 处理路径：如果是目录或没有扩展名，自动添加文件名和扩展名
        if output_path.is_dir() or not output_path.suffix:
            # 如果路径是目录或没有扩展名，使用默认文件名
            if output_path.is_dir():
                output_dir = output_path
                base_name = "retarget_comparison"
            else:
                # 路径存在但没有扩展名
                output_dir = output_path.parent if output_path.parent != Path('.') else Path('.')
                base_name = output_path.name if output_path.name else "retarget_comparison"
            
            # 确保目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 如果提供了原始数据，保存为不同的文件名
            if has_original_data:
                wrist_output_path = output_dir / f"{base_name}_wrist.png"
            else:
                wrist_output_path = output_dir / f"{base_name}.png"
        else:
            # 路径是文件路径，直接使用
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if has_original_data:
                wrist_output_path = output_path.parent / f"{output_path.stem}_wrist{output_path.suffix}"
            else:
                wrist_output_path = output_path
        
        plt.savefig(wrist_output_path, dpi=150, bbox_inches='tight')
        print(f"   功能1图片已保存到: {wrist_output_path}")
    else:
        plt.show()
    
    # 功能2: 创建12个子图：左右手各6个手指关节对比
    if has_original_data:
        print("\n[6] 生成手指关节对比图（功能2）...")
        
        fig2, axes2 = plt.subplots(6, 2, figsize=(16, 20))
        fig2.suptitle('功能2: Retarget前后对比 - 左右手指关节', fontsize=16, fontweight='bold')
        
        joint_names = ['pinky', 'ring', 'middle', 'index', 'thumb_pitch', 'thumb_yaw']
        joint_labels = ['Pinky', 'Ring', 'Middle', 'Index', 'Thumb_Pitch', 'Thumb_Yaw']
        
        for row, (joint_name, joint_label) in enumerate(zip(joint_names, joint_labels)):
            for col, (side, side_label) in enumerate(zip(sides, side_labels)):
                ax = axes2[row, col]
                
                # 确保数据长度一致
                min_len = min(len(original_finger_joints[side][joint_name]), 
                             len(retargeted_finger_joints[side][joint_name]))
                
                if min_len > 0:
                    orig_data = original_finger_joints[side][joint_name][:min_len]
                    retarget_data = retargeted_finger_joints[side][joint_name][:min_len]
                    time_steps_joint = np.arange(min_len)
                    
                    # 绘制原始和retarget后的数据
                    ax.scatter(time_steps_joint, orig_data, 
                              c=colors['before'], label='Original dataset', alpha=0.6, s=4)
                    ax.scatter(time_steps_joint, retarget_data, 
                              c=colors['after'], label='Retarget output', alpha=0.6, s=4)
                    
                    # 计算并显示平均误差
                    if len(orig_data) == len(retarget_data):
                        errors = np.abs(np.array(orig_data) - np.array(retarget_data))
                        mean_error = np.mean(errors)
                        max_error = np.max(errors)
                        ax.text(0.02, 0.98, f'Mean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}', 
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel('timestep')
                ax.set_ylabel(f'{joint_label} Joint Angle (rad)')
                ax.set_title(f'{side_label} - {joint_label}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存功能2的图片
        if args.output_plot:
            output_path = Path(args.output_plot)
            
            # 处理路径：如果是目录或没有扩展名，自动添加文件名和扩展名
            if output_path.is_dir() or not output_path.suffix:
                # 如果路径是目录或没有扩展名，使用默认文件名
                if output_path.is_dir():
                    output_dir = output_path
                    base_name = "retarget_comparison"
                else:
                    # 路径存在但没有扩展名
                    output_dir = output_path.parent if output_path.parent != Path('.') else Path('.')
                    base_name = output_path.name if output_path.name else "retarget_comparison"
                
                # 确保目录存在
                output_dir.mkdir(parents=True, exist_ok=True)
                finger_output_path = output_dir / f"{base_name}_finger_joints.png"
            else:
                # 路径是文件路径，直接使用
                output_path.parent.mkdir(parents=True, exist_ok=True)
                finger_output_path = output_path.parent / f"{output_path.stem}_finger_joints{output_path.suffix}"
            
            plt.savefig(finger_output_path, dpi=150, bbox_inches='tight')
            print(f"   功能2图片已保存到: {finger_output_path}")
        else:
            plt.show()
        
        # 打印手指关节误差统计
        print("\n" + "=" * 80)
        print("[DEBUG] 手指关节误差统计")
        print("=" * 80)
        for side in ['left', 'right']:
            print(f"\n{side.upper()} hand finger joints:")
            for joint_name, joint_label in zip(joint_names, joint_labels):
                if len(original_finger_joints[side][joint_name]) > 0 and \
                   len(retargeted_finger_joints[side][joint_name]) > 0:
                    min_len = min(len(original_finger_joints[side][joint_name]), 
                                 len(retargeted_finger_joints[side][joint_name]))
                    orig = np.array(original_finger_joints[side][joint_name][:min_len])
                    retarget = np.array(retargeted_finger_joints[side][joint_name][:min_len])
                    errors = np.abs(orig - retarget)
                    mean_error = np.mean(errors)
                    max_error = np.max(errors)
                    print(f"  {joint_label:15s}: Mean Error = {mean_error:.6f} rad, Max Error = {max_error:.6f} rad")
    else:
        print("\n[6] 跳过手指关节对比图（未提供原始数据集）")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
    print("\n说明:")
    print("  - 第0帧会执行warmup（因为warm_up_steps=1）")
    print("  - warmup使用wrist_xyz和wrist_rotvec信息")
    print("  - 后续帧直接进行retarget")
    print("  - 功能1（手腕对比）:")
    print("    * 红色点：Retarget前的数据（输入）")
    print("    * 蓝色点：Retarget后的数据（输出）")
    print("    * 输出格式：wrist_pose(6) = [xyz(3), rotvec(3)]")
    if has_original_data:
        print("  - 功能2（手指关节对比）:")
        print("    * 红色点：原始数据集中的手指关节（44维格式）")
        print("    * 蓝色点：Retarget输出的手指关节")
        print("    * 关节顺序：[pinky, ring, middle, index, thumb_pitch, thumb_yaw]")
    else:
        print("  - 功能2（手指关节对比）: 未生成（需要提供--original_parquet_file参数）")
