#!/usr/bin/env python3
"""
Fourier Hand Retarget API v2 - 严格按照原始retarget脚本实现
包含warmup和完整的retarget流程

输入格式（45维）:
- left_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- right_key_points(21): wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
- waist(3)

输出格式:
- left_wrist_pose: (6,) [pos(3), rotvec(3)]
- left_finger_joints: (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (MuJoCo控制量)
  - MuJoCo范围: 握紧[1.5,1.5,1.5,1.5,3,3], 放松[-1.5,-1.5,-1.5,-1.5,-3,-3]

用法示例：
```python
from gr00t.eval.fourier_hand_retarget_api_v2 import FourierHandRetargetAPIV2

# 初始化（只需初始化一次）
retargeter = FourierHandRetargetAPIV2()

# 在每个episode开始时reset
retargeter.reset()

# 处理每一帧（输入45维）
state_45d = np.array([...])  # (45,) 从模型输出
result = retargeter.retarget_from_45d(state_45d)

# 使用结果
left_wrist_pose = result['left']['wrist_pose']  # (6,): [pos(3), rotvec(3)]
left_finger_joints = result['left']['finger_joints']  # (6,): MuJoCo控制量 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
```
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from scipy.spatial.transform import Rotation as R

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
                'finger_joints': (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (MuJoCo控制量)
            },
            'right': {...}
        }
        
    注意: finger_joints顺序为6个主动关节: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
          - 输出的是MuJoCo控制量，已从URDF角度映射
          - MuJoCo范围: 握紧[1.5,1.5,1.5,1.5,3,3], 放松[-1.5,-1.5,-1.5,-1.5,-3,-3]
          - 可直接用于MuJoCo仿真控制
    """
    
    def __init__(
        self, 
        robot_name: str = "fourier",
        hand_sides: List[str] = ["left", "right"],
        wrist_enhance_weight: float = 2.0,
        warm_up_steps: int = 5,
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
        
        # URDF到MuJoCo的关节角度映射配置
        # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        # URDF全部握紧: [-1.57, -1.74, -1.57, -1.57, 0, -1.74]
        # URDF全部松开: [0, 0, 0, 0, 1.22, 0]
        # MuJoCo全部握紧: [1.5, 1.5, 1.5, 1.5, 3, 3]
        # MuJoCo全部放松: [-1.5, -1.5, -1.5, -1.5, -3, -3]
        self.urdf_min = np.array([-1.57, -1.74, -1.57, -1.57, 0, -1.74], dtype=np.float32)  # URDF握紧
        self.urdf_max = np.array([0, 0, 0, 0, 1.22, 0], dtype=np.float32)  # URDF松开
        self.mujoco_min = np.array([-1.5, -1.5, -1.5, -1.5, -3, -3], dtype=np.float32)  # MuJoCo放松
        self.mujoco_max = np.array([1.5, 1.5, 1.5, 1.5, 3, 3], dtype=np.float32)  # MuJoCo握紧
        
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
    
    def _map_urdf_to_mujoco(self, urdf_joints: np.ndarray) -> np.ndarray:
        """
        将URDF关节角度映射到MuJoCo控制量
        
        Args:
            urdf_joints: (6,) URDF关节角度 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        
        Returns:
            mujoco_joints: (6,) MuJoCo控制量 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        
        映射规则:
            - URDF全部握紧: [-1.57, -1.74, -1.57, -1.57, 0, -1.74] -> MuJoCo全部握紧: [1.5, 1.5, 1.5, 1.5, 3, 3]
            - URDF全部松开: [0, 0, 0, 0, 1.22, 0] -> MuJoCo全部放松: [-1.5, -1.5, -1.5, -1.5, -3, -3]
            使用线性插值进行映射
        """
        urdf_joints = np.asarray(urdf_joints, dtype=np.float32)
        if urdf_joints.shape != (6,):
            raise ValueError(f"urdf_joints必须是6维，得到: {urdf_joints.shape}")
        
        # 反向线性映射: URDF的"握紧"（较小值）对应MuJoCo的"握紧"（较大值）
        # 映射公式: mujoco = (1 - normalized) * (mujoco_max - mujoco_min) + mujoco_min
        # 其中 normalized = (urdf - urdf_min) / (urdf_max - urdf_min)
        # 这样: urdf_min -> normalized=0 -> mujoco_max, urdf_max -> normalized=1 -> mujoco_min
        mujoco_joints = np.zeros(6, dtype=np.float32)
        for i in range(6):
            urdf_range = self.urdf_max[i] - self.urdf_min[i]
            if abs(urdf_range) < 1e-6:
                # 如果范围太小，使用端点值（反向映射）
                if abs(urdf_joints[i] - self.urdf_min[i]) < abs(urdf_joints[i] - self.urdf_max[i]):
                    mujoco_joints[i] = self.mujoco_max[i]  # URDF接近最小值 -> MuJoCo最大值
                else:
                    mujoco_joints[i] = self.mujoco_min[i]  # URDF接近最大值 -> MuJoCo最小值
            else:
                # 归一化到[0, 1]
                normalized = (urdf_joints[i] - self.urdf_min[i]) / urdf_range
                # 反向映射到MuJoCo范围: (1 - normalized)使得urdf_min -> mujoco_max, urdf_max -> mujoco_min
                mujoco_joints[i] = (1.0 - normalized) * (self.mujoco_max[i] - self.mujoco_min[i]) + self.mujoco_min[i]
        
        return mujoco_joints
    
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
                    'wrist_pose': (6,) [pos_xyz(3), rotvec_xyz(3)],
                    'finger_joints': (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (MuJoCo控制量)
                },
                'right': {...}
            }
        
        流程说明（严格按照原始retarget脚本）:
            1. 提取左右手的关键点数据
            2. 如果是episode的前N帧（warm_up_steps），执行warmup
            3. 执行retargeting，得到URDF关节角度
            4. 将URDF角度映射到MuJoCo控制量
            5. 转换输出格式
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
            
            # 应用轴角连续性处理，避免等价表示跳变
            wrist_rotvec = self._ensure_axisangle_continuity(wrist_rotvec_raw, side)
            
            # 组合成 6D wrist pose: [pos(3), rotvec(3)]
            wrist_pose = np.concatenate([wrist_pos, wrist_rotvec]).astype(np.float32)
            
            # finger_joints: 从qpos_full中提取6个主动关节
            # 顺序: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            # 这是config文件中target_joint_names的顺序
            desired_indices = self.desired_joint_indices[side]
            finger_joints_urdf = qpos_full[desired_indices]  # (6,) URDF角度
            
            # 将URDF角度映射到MuJoCo控制量
            finger_joints = self._map_urdf_to_mujoco(finger_joints_urdf)  # (6,) MuJoCo控制量
            
            # 确保shape正确
            assert wrist_pose.shape == (6,), f"{side} wrist_pose shape错误: {wrist_pose.shape}, 期望(6,)"
            assert finger_joints.shape == (6,), f"{side} finger_joints shape错误: {finger_joints.shape}, 期望(6,). desired_indices={desired_indices}, qpos_full.shape={qpos_full.shape}"
            
            result[side] = {
                'wrist_pose': wrist_pose,
                'finger_joints': finger_joints,  # (6,) MuJoCo控制量 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
            }
            
            # 增加帧计数
            self._episode_frame_count[side] += 1
        
        return result



# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Testing FourierHandRetargetAPI")
    print("=" * 80)
    
    # 初始化
    print("\n[1] 初始化 FourierHandRetargetAPI...")
    api = FourierHandRetargetAPI(warm_up_steps=1)
    
    # 模拟episode开始
    print("\n[2] 开始新episode（调用reset）...")
    api.reset()
    
    # 读取数据文件
    data_file = Path("/vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_eepose/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000_keypoints_v2/data/chunk-000/episode_000000_actions_first20.txt")
    print(f"\n[3] 读取数据文件: {data_file}")
    
    frames_data = []
    with open(data_file, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx >= 3:  # 只读取前3帧（Frame 0, 1, 2）
                break
            # 解析格式: "Frame N: num1 num2 ... num45"
            parts = line.strip().split(':')
            if len(parts) != 2:
                continue
            frame_num = int(parts[0].split()[1])
            numbers = parts[1].strip().split()
            if len(numbers) != 45:
                print(f"警告: Frame {frame_num} 的数据维度不是45维，实际为 {len(numbers)}")
                continue
            state_45d = np.array([float(x) for x in numbers], dtype=np.float32)
            frames_data.append((frame_num, state_45d))
    
    print(f"   成功读取 {len(frames_data)} 帧数据")
    
    # 测试前3帧（包含warmup）
    print("\n[4] 对前3帧进行retarget...")
    print("=" * 80)
    for frame_num, state_45d in frames_data:
        print(f"\n--- Frame {frame_num} ---")
        
        result = api.retarget_from_45d(state_45d)
        
        # 打印每个时间步的retarget输出：手腕xyz + 轴角
        for side in ['left', 'right']:
            if side in result:
                wrist_pose = result[side]['wrist_pose']
                wrist_xyz = wrist_pose[0:3]  # 前3维是xyz位置
                wrist_rotvec = wrist_pose[3:6]  # 后3维是轴角旋转
                
                print(f"\n{side.capitalize()}手:")
                print(f"  手腕位置 (xyz): [{wrist_xyz[0]:.6f}, {wrist_xyz[1]:.6f}, {wrist_xyz[2]:.6f}]")
                print(f"  手腕轴角 (rotvec): [{wrist_rotvec[0]:.6f}, {wrist_rotvec[1]:.6f}, {wrist_rotvec[2]:.6f}]")
                
                # 显示warmup状态
                is_warmed = api._is_warmed_up[side]
                print(f"  Warmed up: {is_warmed}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)
    print("\n说明:")
    print("  - 第0帧会执行warmup（因为warm_up_steps=1）")
    print("  - warmup使用wrist_xyz和wrist_rotvec信息")
    print("  - 后续帧直接进行retarget")
    print("  - 输出格式：wrist_pose(6) = [xyz(3), rotvec(3)]")
