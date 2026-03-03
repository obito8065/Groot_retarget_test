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
- left_finger_joints: (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
  - 已进行符号修正：pinky, ring, middle, index, thumb_yaw 取负号，thumb_pitch 保持不变
  - 可直接用于MuJoCo仿真控制

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
from typing import Dict, List, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass
import logging

# 仅在首次创建 SAPIEN scene 时调用；若未设置则设置为 egl（headless 模式）
def _ensure_sapien_environment():
    if "PYOPENGL_PLATFORM" not in os.environ:
        os.environ["PYOPENGL_PLATFORM"] = "egl"

# 添加 retarget 路径（使用直接下载到本地的包）
_RETARGET_SRC_PATH = Path(__file__).resolve().parent / "retarget" / "src"
if _RETARGET_SRC_PATH.exists():
    if str(_RETARGET_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_RETARGET_SRC_PATH))
else:
    raise ImportError(
        f"dex_retargeting not found at {_RETARGET_SRC_PATH}. "
        "Ensure gr00t/eval/retarget/ is properly set up."
    )


# 导入核心retargeting组件
from dex_retargeting.constants import (
    HandType, 
    RetargetingType, 
    RobotName, 
    get_default_config_path )
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting import yourdfpy as urdf
from pytransform3d import rotations

logger = logging.getLogger(__name__)

# 定义所有的常量：
INPUT_DIM = 45
KEYPOINTS_PER_HAND = 21
KEYPOINTS_6DIM = 6  # wrist + 5 tips
OUTPUT_WRIST_POSE_DIM = 6
OUTPUT_FINGER_JOINTS_DIM = 6

# 45维输入中左右手的偏移
STATE_OFFSETS: Dict[str, int] = {"left": 0, "right": 21}

# MANO 21点索引：[thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip, wrist]
MANO_KEYPOINT_INDICES = np.array([4, 8, 12, 16, 20, 0])

# 21维关键点内顺序: [wrist, thumb, index, middle, ring, pinky] -> [thumb..wrist] 的 MANO 顺序
KEYPOINT_REORDER = np.array([1, 2, 3, 4, 5, 0])

# 手指关节符号修正系数 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
# pinky, ring, middle, index, thumb_yaw 取负并缩放，thumb_pitch 保持不变
FINGER_JOINT_CORRECTION_SCALE = np.array(
    [-1.64, -1.64, -1.64, -1.64, 1.0, -1.0],
    dtype=np.float32,
)

# 轴角连续性检测中的阈值，抽成常量便于调参
AXISANGLE_JUMP_THRESHOLD = 2.0
AXISANGLE_RATIO_THRESHOLD = 0.5
AXISANGLE_EPS = 1e-6

# 结构化定义retarget输出
@dataclass
class RetargetOutput:
    wrist_pose: np.ndarray  # (6,) [pos_xyz(3), rotvec_xyz(3)]
    finger_joints: np.ndarray  # (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw]



class FourierHandRetargetAPI:
    """
    Fourier灵巧手Retarget API 

    关键特性:
    1.  包含warmup处理（episode开始的前几帧）
    2.  支持45维输入格式（与训练数据对齐）
    3.  严格遵循原始retarget脚本的处理流程
    
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
        
    注意: 
        - finger_joints顺序为6个主动关节: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        - robocasa需要进行符号修正：pinky, ring, middle, index, thumb_yaw 取负号，thumb_pitch 保持不变
        - 可直接用于MuJoCo仿真控制
    """

    _FINGER_JOINT_NAMES = {
        "left": [
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_middle_proximal_joint",
            "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_proximal_yaw_joint",
        ],
        "right": [
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_middle_proximal_joint",
            "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
        ],
    }
    
    def __init__(
        self, 
        robot_name: str = "fourier",
        hand_sides: Optional[List[str]] = None,
        wrist_enhance_weight: float = 2.0,
        warm_up_steps: int = 5,
        finger_correction_scale: Optional[np.ndarray] = None,
    ):
        """
        初始化Retarget API
        
        Args:
            robot_name: 机器人名称，默认"fourier"
            hand_sides: 手部列表，默认["left", "right"]
            wrist_enhance_weight: 手腕优化权重，默认2.0
            warm_up_steps: warmup的帧数，默认1（即第一帧进行warmup）
            finger_correction_scale: 手指关节符号修正系数 (6,)，None 则用默认值
        """
        # 转换robot_name为RobotName enum
        hand_sides = hand_sides or ["left", "right"]
        self.robot_name_enum = self._get_robot_name_enum(robot_name)

        self.hand_sides = hand_sides
        self.warm_up_steps = warm_up_steps

        self.finger_correction_scale = (
            finger_correction_scale if finger_correction_scale is not None else FINGER_JOINT_CORRECTION_SCALE.copy()
        )
        self.retargetings: Dict[str, object] = {}
        self.desired_joint_indices: Dict[str, np.ndarray] = {}
        self.sapien_robots: Dict[str, object] = {}
        self.hand_base_link_indices: Dict[str, int] = {}

        self._episode_frame_count: Dict[str, int] = {s: 0 for s in hand_sides}
        self._is_warmed_up: Dict[str, bool] = {s: False for s in hand_sides}
        self._last_quaternion: Dict[str, Optional[np.ndarray]] = {s: None for s in hand_sides}
        self._last_rotvec: Dict[str, Optional[np.ndarray]] = {s: None for s in hand_sides}

        _ensure_sapien_environment()
        import sapien
        
        # 创建 SAPIEN scene 和 robots（用于 FK 计算，与 hand_robot_viewer_fourier.py 保持一致）
        self.scene = sapien.Scene()

        
        # 为每个手侧初始化retargeting
        for side in hand_sides:
            self._init_hand_side(side, wrist_enhance_weight, robot_name)
            
    def _get_robot_name_enum(self, robot_name: str) -> RobotName:
        """转换robot_name为RobotName enum"""
        if hasattr(RobotName, robot_name):
            return getattr(RobotName, robot_name)
        else:
            raise ValueError(f"Unknown robot: {robot_name}")
    
    def _init_hand_side(self, side: str, wrist_enhance_weight: float, robot_name: str):
        """初始化单个手侧"""
        hand_type = getattr(HandType, side)
        config_path = get_default_config_path(
            self.robot_name_enum, 
            RetargetingType.position, 
            hand_type
        )
        config = RetargetingConfig.load_from_file(config_path)
        retargeting = config.build()
        retargeting.optimizer.set_retarget_weight(wrist_enhance_weight=wrist_enhance_weight)
        
        joint_names = self._FINGER_JOINT_NAMES[side]
        desired_indices = []
        for joint_name in joint_names:
            try:
                idx = retargeting.joint_names.index(joint_name)
                desired_indices.append(idx)
            except ValueError:
                print(f"Error: Cannot find joint {joint_name} in retargeting.joint_names")
                print(f"  Available joints: {retargeting.joint_names}")
                raise
        self.retargetings[side] = retargeting
        self.desired_joint_indices[side] = np.array(desired_indices, dtype=np.int32)

        sapien_robot = self._create_sapien_robot(config, retargeting, side)
        self.sapien_robots[side] = sapien_robot
        self.hand_base_link_indices[side] = self._find_hand_base_link_index(sapien_robot)


    def _create_sapien_robot(self, config: object, retargeting: object, side: str): 
        """创建 SAPIEN robot"""
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
        return sapien_robot

    def _find_hand_base_link_index(self, sapien_robot: object):
        """查找 hand_base_link 的索引，通常是7"""
        links = sapien_robot.get_links()

        for i, link in enumerate(links):
            if 'hand_base' in link.name.lower() or 'base' in link.name.lower():
                return i
        logger.warning(f"hand_base_link not found by name, using index 7")
        return 7


    def _validate_input(self, state_45d: np.ndarray) -> np.ndarray:
        """ 验证输入是否为45维 """
        state = np.asarray(state_45d, dtype=np.float32)
        if state.size != INPUT_DIM:
            raise ValueError(
                f"Expected input dimension {INPUT_DIM}, got {state.size}. "
                "Use state_45d.flatten() if needed."
            )
        return state.flatten()
    
    def _extract_keypoints_from_45d(
        self, state_45d: np.ndarray, side: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从 45 维提取单手数据的逻辑独立，用常量 STATE_OFFSETS 替代硬编码 0/21。"""
        offset = STATE_OFFSETS[side]
        kp = state_45d[offset : offset + KEYPOINTS_PER_HAND]
        wrist_xyz = kp[0:3]
        tips = kp[3:18].reshape(5, 3)  # thumb, index, middle, ring, pinky
        wrist_rotvec = kp[18:21]
        keypoints_6x3 = np.vstack([wrist_xyz, tips]).astype(np.float32)
        return keypoints_6x3, wrist_rotvec

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
        if direct_diff > AXISANGLE_JUMP_THRESHOLD  and actual_angle < direct_diff * AXISANGLE_RATIO_THRESHOLD:
            # 尝试 -r + 2πk 的等价表示
            angle = np.linalg.norm(current_rotvec)
            if angle > AXISANGLE_EPS:
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
        
        
        print(f"[FourierHandRetargetAPI] Reset for new episode (env_idx={env_idx})")
    
    def _maybe_warmup(
        self, keypoints_6x3: np.ndarray, wrist_rotvec: np.ndarray, side: str
    ) -> None:
        """warmup 条件判断独立成方法，retarget_from_45d 主流程更清晰。"""
        if self._episode_frame_count[side] >= self.warm_up_steps:
            return
        wrist_xyz = keypoints_6x3[0]
        quat_xyzw = R.from_rotvec(wrist_rotvec).as_quat()
        wrist_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]) # scipy返回的是[x,y,z,w]，需要转换为[w,x,y,z]
        self._warmup(wrist_xyz, wrist_quat_wxyz, side)

    def _warmup(self, wrist_pos: np.ndarray, wrist_quat: np.ndarray, side: str):
        """
        执行warmup
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
            is_mano_convention=False,     # 不使用MANO坐标系约定，默认相机坐标系
        )
        
        self._is_warmed_up[side] = True
        return warmup_qpos6d

    def _build_mano_hand_21(self, keypoints_6x3: np.ndarray) -> np.ndarray:
        """MANO 格式转换用常量 KEYPOINT_REORDER、MANO_KEYPOINT_INDICES，避免魔法索引。"""
        kp_reordered = keypoints_6x3[KEYPOINT_REORDER]
        hand_21 = np.zeros((21, 3), dtype=np.float32)
        hand_21[MANO_KEYPOINT_INDICES] = kp_reordered
        return hand_21

    def _run_retarget(self, hand_21: np.ndarray, side: str) -> np.ndarray:
        """调用 dex_retargeting 的核心逻辑抽离"""
        retargeting = self.retargetings[side]
        human_indices = retargeting.optimizer.target_link_human_indices
        human_keypoints = hand_21[human_indices, :]
        return retargeting.retarget(human_keypoints)
    
    def _compute_output_from_qpos(self, qpos_full: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算retargeting后的输出:
            通过 SAPIEN FK 计算 hand_base_link 的实际位姿,因为 dummy joint 给出的是 root_link 的位姿,不是 hand_base_link 的位姿
        """
        sapien_robot = self.sapien_robots[side]
        sapien_robot.set_qpos(qpos_full)
        link_idx = self.hand_base_link_indices[side]
        link_pose = sapien_robot.get_links()[link_idx].entity_pose

        wrist_pos = np.array(link_pose.p, dtype=np.float32)
        wrist_rotvec_raw = rotations.compact_axis_angle_from_quaternion(link_pose.q)
        wrist_rotvec = self._ensure_axisangle_continuity(wrist_rotvec_raw, side)
        wrist_pose = np.concatenate([wrist_pos, wrist_rotvec]).astype(np.float32)

        desired_indices = self.desired_joint_indices[side]
        finger_joints = qpos_full[desired_indices]
        # 向量化符号修正：finger_joints * self.finger_correction_scale, robocasa的实际控制量不够，手动进行double控制量
        finger_joints_corrected = (
            finger_joints * self.finger_correction_scale
        ).astype(np.float32)

        return wrist_pose, finger_joints_corrected
    
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
                    'finger_joints': (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (已符号修正)
                },
                'right': {...}
            }
        
        流程说明（严格按照原始retarget脚本）:
            1. 提取左右手的关键点数据
            2. 如果是episode的前N帧（warm_up_steps），执行warmup
            3. 执行retargeting
            4. 转换输出格式
        """
        state_45d = self._validate_input(state_45d)

        result = {}
        
        
        for side in self.hand_sides:
            keypoints_6x3, wrist_rotvec = self._extract_keypoints_from_45d(state_45d, side)
            self._maybe_warmup(keypoints_6x3, wrist_rotvec, side)
            hand_21 = self._build_mano_hand_21(keypoints_6x3)
            qpos_full = self._run_retarget(hand_21, side)
            wrist_pose, finger_correct_joints = self._compute_output_from_qpos(qpos_full, side)

            assert wrist_pose.shape == (OUTPUT_WRIST_POSE_DIM,)
            assert finger_correct_joints.shape == (OUTPUT_FINGER_JOINTS_DIM,)


            result[side] = {
                'wrist_pose': wrist_pose,
                'finger_joints': finger_correct_joints,  # (6,) [pinky, ring, middle, index, thumb_pitch, thumb_yaw] (已符号修正)
            }
            
            # 增加帧计数
            self._episode_frame_count[side] += 1
        
        return result



def _load_frames_from_file(
    data_file: Path, max_frames: int = 3
) -> List[Tuple[int, np.ndarray]]:
    """从数据集文件中加载数据，验证retarget得效果"""
    frames = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= max_frames:
                break
            parts = line.strip().split(":")
            if len(parts) != 2:
                continue
            frame_num = int(parts[0].split()[1])
            numbers = parts[1].strip().split()
            if len(numbers) != INPUT_DIM:
                logger.warning("Frame %d: expected %d dims, got %d", frame_num, INPUT_DIM, len(numbers))
                continue
            state = np.array([float(x) for x in numbers], dtype=np.float32)
            frames.append((frame_num, state))
    return frames


# ============================================================================
# 测试
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test FourierHandRetargetAPI")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path(
            "/vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_eepose/"
            "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000_keypoints_v2/"
            "data/chunk-000/episode_000000_actions_first20.txt"
        ),
        help="Path to episode actions file",
    )
    parser.add_argument("--max-frames", type=int, default=3, help="Max frames to test")
    args = parser.parse_args()

    print("=" * 60)
    print("Testing FourierHandRetargetAPI")
    print("=" * 60)

    api = FourierHandRetargetAPI(warm_up_steps=1)
    api.reset()
    frames = _load_frames_from_file(args.data_file, args.max_frames)
    print(f"Loaded {len(frames)} frames from {args.data_file}")

    for frame_num, state_45d in frames:
        print(f"\n--- Frame {frame_num} ---")
        result = api.retarget_from_45d(state_45d)
        for side in ["left", "right"]:
            if side in result:
                wp = result[side]["wrist_pose"]
                fj = result[side]["finger_joints"]
                print(f"  {side}: wrist_xyz={wp[:3].round(4)} | fingers={fj.round(4)} | warmed={api._is_warmed_up[side]}")

    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)
