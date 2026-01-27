# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
import os


import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

from gr00t.data.dataset import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.schema import DatasetMetadata
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.model.gr00t_n1 import GR00T_N1_5


COMPUTE_DTYPE = torch.bfloat16


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to get the action for a given state.

        Args:
            observations: The observations from the environment.

        Returns:
            The action to take in the environment in dictionary format.
        """
        raise NotImplementedError

    @abstractmethod
    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Return the modality config of the policy.
        """
        raise NotImplementedError


class Gr00tPolicy(BasePolicy):
    """
    A wrapper for Gr00t model checkpoints that handles loading the model, applying transforms,
    making predictions, and unapplying transforms. This loads some custom configs, stats
    and metadata related to the model checkpoints used
    in the Gr00t model.
    """

    def __init__(
        self,
        model_path: str,
        embodiment_tag: Union[str, EmbodimentTag],
        modality_config: Dict[str, ModalityConfig],
        modality_transform: ComposedModalityTransform,
        denoising_steps: Optional[int] = None,
        device: Union[int, str] = "cuda" if torch.cuda.is_available() else "cpu",
        action_horizon=None,
        enable_latent_embedding=False,
        backbone_type=None,
        backbone_model_name_or_path=None,
        use_dino=False,
        use_time_aware_action_head=False,
        use_eepose=False,
        use_fourier_hand_retarget=False,
    ):
        """
        Initialize the Gr00tPolicy.

        Args:
            model_path (str): Path to the model checkpoint directory or the huggingface hub id.
            modality_config (Dict[str, ModalityConfig]): The modality config for the model.
            modality_transform (ComposedModalityTransform): The modality transform for the model.
            embodiment_tag (Union[str, EmbodimentTag]): The embodiment tag for the model.
            denoising_steps: Number of denoising steps to use for the action head.
            device (Union[int, str]): Device to run the model on.
        """
        try:
            # NOTE(YL) this returns the local path to the model which is normally
            # saved in ~/.cache/huggingface/hub/
            model_path = snapshot_download(model_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {model_path}"
            )

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        self._modality_transform.eval()  # set this to eval mode
        self.model_path = Path(model_path)
        self.device = device

        # Convert string embodiment tag to EmbodimentTag enum if needed
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        # Load model
        self._load_model(model_path, 
                         backbone_type=backbone_type, 
                         enable_latent_alignment=enable_latent_embedding, 
                         use_dino=use_dino,
                         use_time_aware_action_head=use_time_aware_action_head,
                         )

        if action_horizon is not None:
            self.model.update_action_horizon(action_horizon)

        # Load transforms
        self._load_metadata(self.model_path / "experiment_cfg")
        # Load horizons
        self._load_horizons()

        if denoising_steps is not None:
            if hasattr(self.model, "action_head") and hasattr(
                self.model.action_head, "num_inference_timesteps"
            ):
                self.model.action_head.num_inference_timesteps = denoising_steps
                print(f"Set action denoising steps to {denoising_steps}")

        self.use_eepose = use_eepose
        print(f"debug policy : is use eepose :{self.use_eepose}")
        if self.use_eepose:
            
            if "robocasa" in self.embodiment_tag.value:
                from gr00t.eval.gr1_pos_transform import BodyRetargeter, GR1RetargetConfig
                gr1_config = GR1RetargetConfig()
                
                # 然后，通过这个实例来访问属性
                self.body_retargeter = BodyRetargeter(
                    urdf_path=Path(gr1_config.urdf_path), 
                    camera_intrinsics=gr1_config.camera_intrinsics
                )
                print("Enabled Robocasa EEPose processing in Gr00tPolicy.")
            elif "robotwin" in self.embodiment_tag.value:
                from gr00t.eval.robotwin_aloha_transform_kdl import AlohaRetargeter, RobotwinRetargetConfig
                robotwin_config = RobotwinRetargetConfig()
                self.body_retargeter = AlohaRetargeter(
                    urdf_path=Path(robotwin_config.urdf_path),                )
                print("Enabled Robotwin EEPose processing in Gr00tPolicy.")

        # ==========================================================
        # Fourier Hand FK & Retarget 初始化
        # 当use_eepose和use_fourier_hand_retarget同时为True时初始化
        # ==========================================================
        self.use_fourier_hand_retarget = use_fourier_hand_retarget
        print(f"debug policy : is use fourier_hand_retarget :{self.use_fourier_hand_retarget}")
        
        # 只在同时使用eepose和fourier_hand_retarget时初始化FK和Retarget模块
        if self.use_eepose and self.use_fourier_hand_retarget:
            if "robocasa" in self.embodiment_tag.value:
                # -----------------------------------------------------
                # 初始化 Fourier Hand FK (joint angles -> 6 keypoints)
                # 用于Step 1: 将wrist pose + finger joints转换为hand keypoints
                # -----------------------------------------------------
                # FK 输入 finger6 顺序: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
                # FK 输出 keypoints 顺序: [wrist, thumb, index, middle, ring, pinky] (6, 3)
                from gr00t.eval.gr1_hand_fk import PolicyFourierHandKeypoints
                self.policy_fourier_hand_keypoints = PolicyFourierHandKeypoints(
                    left_urdf=Path("gr00t/eval/robot_assets/fourier_hand/fourier_left_hand.urdf"), 
                    right_urdf=Path("gr00t/eval/robot_assets/fourier_hand/fourier_right_hand.urdf")
                )
                print("✓ Initialized Fourier Hand FK for Step 1 (input processing).")
                
                # -----------------------------------------------------
                # 初始化 Fourier Hand Retargeter (6 keypoints -> joint angles)
                # 用于Step 3: 将预测的hand keypoints转换回wrist pose + finger joints
                # -----------------------------------------------------
                # Retarget 输入 keypoints 顺序: [wrist, thumb, index, middle, ring, pinky] (6, 3)
                # Retarget 输出：
                #   - wrist_pose: [wrist_xyz(3), rotvec(3)] = 6维
                #   - finger_joints: [pinky, ring, middle, index, thumb_pitch, thumb_yaw] = 6维 (6个主动关节)
                # 注意: finger_joints为仿真所需的6个主动关节，不包含mimic关节
                from gr00t.eval.fourier_hand_retarget_api import FourierHandRetargetAPI
                self.fourier_hand_retargeter = FourierHandRetargetAPI()
                print("✓ Initialized Fourier Hand Retargeter for Step 3 (output processing).")
            else:
                print(f"Warning: use_eepose={self.use_eepose} and use_fourier_hand_retarget={self.use_fourier_hand_retarget}, "
                      f"but embodiment_tag={self.embodiment_tag.value} is not robocasa. Skipping initialization.")

        elif self.use_fourier_hand_retarget and not self.use_eepose:
            if "robocasa" in self.embodiment_tag.value:
                # 如果只使用fourier_hand_retarget（不使用eepose）
                # 这种情况下只需要Retarget模块，不需要FK模块
                from gr00t.eval.fourier_hand_retarget_api import FourierHandRetargetAPI
                self.fourier_hand_retargeter = FourierHandRetargetAPI()
                print("✓ Initialized Fourier Hand Retargeter (without eepose mode).")
            else:
                print(f"Warning: use_fourier_hand_retarget=True but embodiment_tag={self.embodiment_tag.value} is not robocasa. Skipping.")


    def apply_transforms(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply transforms to the observation.

        Args:
            obs (Dict[str, Any]): The observation to transform.

        Returns:
            Dict[str, Any]: The transformed observation.
        """
        # Ensure correct dimensions before applying transforms
        return self._modality_transform(obs)

    def unapply_transforms(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unapply transforms to the action.

        Args:
            action (Dict[str, Any]): The action to unapply transforms to.

        Returns:
            Dict[str, Any]: The untransformed action.
        """
        return self._modality_transform.unapply(action)
    
    def reset_ik_cache(self, env_idx: Optional[int] = None):
        """
        清空 Robocasa/GR1 EEPose IK 的历史缓存，以及 Fourier Hand Retarget 的 last_qpos 缓存。
        
        在每个新 episode 开始时调用，用于重置：
        1. Body IK 的 last_qpos 缓存（body_retargeter）
        2. Fourier Hand Retarget 的 warmup 状态和 last_qpos（fourier_hand_retargeter）
        
        注意：
        - FK（policy_fourier_hand_keypoints）是无状态的，不需要reset
        - 只有IK和Retarget需要清空历史qpos缓存
        
        Args:
            env_idx: 并行环境索引
                - None: 清空所有环境的缓存
                - int: 仅清空指定环境的缓存
        """
        # 1. 重置 Body IK 缓存（用于 arm 的 IK，包含 last_qpos）
        if hasattr(self, "body_retargeter") and hasattr(self.body_retargeter, "reset_ik_cache"):
            self.body_retargeter.reset_ik_cache(env_idx)
        
        # 2. 重置 Fourier Hand Retarget 缓存（用于 hand keypoints -> joint angles）
        #    这会清空 warmup 状态和优化器的 last_qpos 历史
        if hasattr(self, "fourier_hand_retargeter") and hasattr(self.fourier_hand_retargeter, "reset"):
            self.fourier_hand_retargeter.reset(env_idx)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction with the model.
        Args:
            obs (Dict[str, Any]): The observation to make a prediction for.

        e.g. obs = {
            "video.<>": np.ndarray,  # (T, H, W, C)
            "state.<>": np.ndarray, # (T, D)
            "annotation.<>": np.ndarray, # (T, )
        }

        or with batched input:
        e.g. obs = {
            "video.<>": np.ndarray,, # (B, T, H, W, C)
            "state.<>": np.ndarray, # (B, T, D)
            "annotation.<>": np.ndarray, # (B, T, )
        }

        Returns:
            Dict[str, Any]: The predicted action.
        """
        # Create a copy to avoid mutating input
        obs_copy = observations.copy()
        left_arm_state = None
        right_arm_state = None
        # full_44dof_vector = None
        full_action_vector = None

        # ==========================================================
        # 新增：从 obs 里读取 reset 标记，清掉对应 env_idx 的 IK bucket
        # 约定：
        # - obs_copy["meta.reset_mask"] 或 obs_copy["reset_mask"]
        # - 形状为 (B,) 的 bool/int；True 表示该 slot 刚 reset（新 episode 开始）
        # ==========================================================
        reset_mask = None
        if "meta.reset_mask" in obs_copy:
            reset_mask = obs_copy.pop("meta.reset_mask", None)


        if reset_mask is not None:
            rm = np.asarray(reset_mask).astype(bool)
            if rm.ndim == 0:
                if bool(rm):
                    print("[Gr00tPolicy] reset_ik_cache env_idx=0 (scalar reset_mask)", flush=True)
                    self.reset_ik_cache(env_idx=0)
            else:
                rm = rm.reshape(-1)
                reset_envs = np.where(rm)[0].tolist()
                if reset_envs:
                    print(f"[Gr00tPolicy] reset_ik_cache env_idx={reset_envs}", flush=True)
                for env_idx, flag in enumerate(rm):
                    if bool(flag):
                        self.reset_ik_cache(env_idx=env_idx)


        """
        Robocasa:
            --- Debugging 'obs' for Robocasa arm_qpos ---
            Key: 'annotation.human.coarse_action', Type: <class 'list'>
            Key: 'state.left_arm', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 7), Dtype: float64
            Key: 'state.left_hand', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64，[pinky，ring，middle，index, thumb_pitch, thumb_yaw]
            Key: 'state.right_arm', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 7), Dtype: float64
            Key: 'state.right_hand', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64
            Key: 'state.waist', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 3), Dtype: float64
            Key: 'video.ego_view_pad_res256_freq20', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 256, 256, 3), Dtype: uint8
            Key: 'video.ego_view', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 256, 256, 3), Dtype: uint8
            ------------------------------------------------

            --- Debugging 'obs' for eepose ---
            Key: 'annotation.human.coarse_action', Type: <class 'list'>
            Key: 'state.left_arm', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64
            Key: 'state.left_hand', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64
            Key: 'state.right_arm', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64
            Key: 'state.right_hand', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 6), Dtype: float64
            Key: 'state.waist', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 3), Dtype: float64
            Key: 'video.ego_view_pad_res256_freq20', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 256, 256, 3), Dtype: uint8
            Key: 'video.ego_view', Type: <class 'numpy.ndarray'>, Shape: (5, 1, 256, 256, 3), Dtype: uint8
            ------------------------------------------------
        
        Robotwin:
            --- Debugging 'obs' for eepose ---
            video.image_high: (1, 1, 480, 640, 3)
            video.image_left_wrist: (1, 1, 480, 640, 3)
            video.image_right_wrist: (1, 1, 480, 640, 3)
            state.left_arm: (1, 1, 6)
            state.left_gripper: (1, 1, 1)
            state.right_arm: (1, 1, 6)
            state.right_gripper: (1, 1, 1)
            annotation.human.task_description: (1,)
            ------------------------------------------------
        
        """

   
        is_batch = self._check_state_is_batched(obs_copy)

        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        
        
        # print("=== action dict shapes ===")
        # for k, v in obs_copy.items():
        #     try:
        #         arr = np.asarray(v)
        #         print(f"{k}: {arr.shape}")
        #     except Exception:
        #         # 极少数不是 array 的就打印类型
        #         print(f"{k}: type={type(v)}")

            
        # Convert to numpy arrays
        for k, v in obs_copy.items():
            if not isinstance(v, np.ndarray):
                # 如果值是元组 (通常来自并行环境的文本观测)
                # 我们需要显式指定 dtype，以避免创建 object 类型的数组
                if isinstance(v, tuple):
                    obs_copy[k] = np.array(v, dtype=str)
                    # print(f"Converted tuple observation for key {k} from tuple to {obs_copy[k].dtype} with dtype str.")

                else:
                    obs_copy[k] = np.array(v)
        


        # FK -> EEpose
        if self.use_eepose:
            if "robocasa" in self.embodiment_tag.value:
                # 1. 使用 BodyRetargeter 将 EEpose 转换为标准状态表示
                full_action_vector = self._build_full_44dof_vector(obs_copy)
                (left_hand_positions, left_hand_axisangles), (right_hand_positions, right_hand_axisangles), (left_qpos_states, right_qpos_states) = self.body_retargeter.process_frame_kinematics_axisangle(full_action_vector)
                # print(f"Left hand positions shape: {left_hand_positions.shape}, axis-angles shape: {left_hand_axisangles.shape}")
                # 2. 将转换后的状态添加回 obs_copy
                left_arm_state = obs_copy.get("state.left_arm", None)
                right_arm_state = obs_copy.get("state.right_arm", None)
                # 拼接两个 (bs, 3) 的数组，得到一个 (bs, 6) 的二维数组
                left_eepose_2d = np.concatenate((left_hand_positions, left_hand_axisangles), axis=-1)
                right_eepose_2d = np.concatenate((right_hand_positions, right_hand_axisangles), axis=-1)

                # 使用 np.newaxis 恢复时间维度，将其从 (bs, 6) 变为 (bs, 1, 6)
                obs_copy["state.left_arm"] = left_eepose_2d[:, np.newaxis, :]
                obs_copy["state.right_arm"] = right_eepose_2d[:, np.newaxis, :]
            
            elif "robotwin" in self.embodiment_tag.value:
                # print("Robotwin EE Pose processing in Gr00tPolicy.")
                # 单帧 14d
                full_action_vector = self._build_single_14dof_vector_robotwin(obs_copy)  # (14,)

                # FK -> 相机系 (pos, axis-angle)
                (left_pos, left_axis), (right_pos, right_axis), (left_q, right_q) = \
                    self.body_retargeter.process_frame_kinematics_axisangle(full_action_vector)

                # 保存原始 qpos（用于 IK seed）；这里 robotwin 是 (B,T,6)
                left_arm_state = obs_copy.get("state.left_arm", None)
                right_arm_state = obs_copy.get("state.right_arm", None)

                left_eepose_2d = np.concatenate((left_pos, left_axis), axis=-1)     # (B,6)
                right_eepose_2d = np.concatenate((right_pos, right_axis), axis=-1)  # (B,6)
                # print(f"Left eepose 2d: {left_eepose_2d}, Right eepose 2d: {right_eepose_2d}")

                obs_copy["state.left_arm"] = left_eepose_2d[:, np.newaxis, :]   # (B,1,6)
                obs_copy["state.right_arm"] = right_eepose_2d[:, np.newaxis, :] # (B,1,6)
                # print(f"Robotwin EE Pose processed: Left EE Pose shape: {obs_copy['state.left_arm'].shape},  Right EE Pose shape: {obs_copy['state.right_arm'].shape}")

        # ==========================================================
        # 第1步：输入处理 - 当use_eepose和use_fourier_hand_retarget同时为True时
        # 将wrist pose + finger joints 转换为 hand keypoints
        # ==========================================================
        # 输入格式（经过use_eepose处理后）：
        #   - obs_copy["state.left_arm"]: (B, T, 6) = [L_wrist_xyz(3), L_rotvec(3)]
        #   - obs_copy["state.left_hand"]: (B, T, 6) = [L_finger_q1~6] (数据集格式: pinky, ring, middle, index, thumb_pitch, thumb_yaw)
        #   - obs_copy["state.right_arm"]: (B, T, 6) = [R_wrist_xyz(3), R_rotvec(3)]
        #   - obs_copy["state.right_hand"]: (B, T, 6) = [R_finger_q1~6]
        # 
        # 输出格式（转换后）：
        #   - obs_copy["state.left_key_points"]: (B, T, 18) = [wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz]
        #   - obs_copy["state.right_key_points"]: (B, T, 18) = 同上
        # ==========================================================
        if (self.use_eepose and self.use_fourier_hand_retarget and 
            "robocasa" in self.embodiment_tag.value):
            
            # 此时obs_copy中应该已经有eepose格式的arm和hand数据
            if ("state.left_arm" in obs_copy and "state.left_hand" in obs_copy):
                
                # 获取当前的wrist pose和finger joints
                # 输入：
                #   - left_arm_orig: (B, T, 6) = [wrist_xyz(3), rotvec(3)]
                #   - left_hand_orig: (B, T, 6) = [finger joints in dataset format]
                left_arm_orig = obs_copy.get("state.left_arm", None)
                right_arm_orig = obs_copy.get("state.right_arm", None)
                left_hand_orig = obs_copy.get("state.left_hand", None)
                right_hand_orig = obs_copy.get("state.right_hand", None)
                
                if (left_hand_orig is not None and right_hand_orig is not None and
                    left_arm_orig is not None and right_arm_orig is not None):
                    
                    B, T, D_arm = left_arm_orig.shape
                    
                    print(f"[Policy Step1] Input shapes: left_arm={left_arm_orig.shape}, left_hand={left_hand_orig.shape}")
                    
                    # ==========================================================
                    # 调试功能：保存原始观测数据到txt文件（用于可视化投影）
                    # ==========================================================
                    if not hasattr(self, '_observation_log_initialized'):
                        from datetime import datetime
                        import os
                        
                        # 创建日志文件
                        log_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self._observation_log_file = log_dir / f"observation_wrist_{timestamp}.txt"
                        self._observation_frame_counter = 0
                        
                        # 写入文件头
                        with open(self._observation_log_file, 'w') as f:
                            f.write("# 原始观测数据：左右手腕位姿和手指关节\n")
                            f.write("# 格式：frame_id t L_wrist_x L_wrist_y L_wrist_z L_wrist_rotvec_x L_wrist_rotvec_y L_wrist_rotvec_z L_finger0 L_finger1 L_finger2 L_finger3 L_finger4 L_finger5 R_wrist_x R_wrist_y R_wrist_z R_wrist_rotvec_x R_wrist_rotvec_y R_wrist_rotvec_z R_finger0 R_finger1 R_finger2 R_finger3 R_finger4 R_finger5\n")
                            f.write("# 左手：wrist_xyz(3) + wrist_rotvec(3) + finger_joints(6)\n")
                            f.write("# 右手：wrist_xyz(3) + wrist_rotvec(3) + finger_joints(6)\n")
                            f.write("# 手指关节顺序（仿真格式）：[pinky, ring, middle, index, thumb_pitch, thumb_yaw]\n")
                            f.write("# batch=0的数据，每个时间步(t=0~T)记录一行\n")
                            f.write("#\n")
                        
                        self._observation_log_initialized = True
                        print(f"[Observation Logger] 创建日志文件: {self._observation_log_file}")
                    
                    # 记录原始观测数据到文件（只保存batch=0的数据）
                    if hasattr(self, '_observation_log_file'):
                        try:
                            with open(self._observation_log_file, 'a') as f:
                                # 遍历时间步（T维度）
                                for t in range(T):
                                    # 提取batch=0的数据
                                    left_arm_t = left_arm_orig[0, t, :]      # (6,) = [wrist_xyz, wrist_rotvec]
                                    right_arm_t = right_arm_orig[0, t, :]    # (6,)
                                    left_hand_t = left_hand_orig[0, t, :]    # (6,) = finger joints
                                    right_hand_t = right_hand_orig[0, t, :]  # (6,)
                                    
                                    # 拼接成一行：frame_id, t, left_arm(6), left_hand(6), right_arm(6), right_hand(6)
                                    # 总共：2 + 6 + 6 + 6 + 6 = 26 个值
                                    line_data = [self._observation_frame_counter, t]
                                    line_data.extend(left_arm_t.tolist())
                                    line_data.extend(left_hand_t.tolist())
                                    line_data.extend(right_arm_t.tolist())
                                    line_data.extend(right_hand_t.tolist())
                                    
                                    # 写入文件
                                    line = ' '.join(f'{v:.6f}' for v in line_data)
                                    f.write(line + '\n')
                            
                            self._observation_frame_counter += 1
                            
                        except Exception as e:
                            print(f"[Observation Logger] 写入日志失败: {e}")
                    # ==========================================================
                    
                    # 数据集格式和FK期望格式一致，不需要重排序
                    # Robocasa数据集: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    # FK期望:        [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
                    # 只交换最后两个维度
                    left_hand_fk = left_hand_orig[..., [3, 2, 1, 0, 5, 4]]
                    right_hand_fk = right_hand_orig[..., [3, 2, 1, 0, 5, 4]]
                                        
                    # 执行 FK 计算
                    # 输入：
                    #   - left_arm_orig: (B, T, 6) = [wrist_xyz(3), rotvec(3)]
                    #   - left_hand_fk: (B, T, 6) = [index, middle, ring, pinky, thumb_yaw, thumb_pitch]
                    #   - waist: (B, T, 3) = 腰部位置（用0填充）
                    # 输出：
                    #   - state_45d: (B, T, 45) = [left_21d, right_21d, waist_3d]
                    waist = np.zeros((B, T, 3), dtype=np.float32)
                    state_45d = self.policy_fourier_hand_keypoints.compute_state_45d(
                        left_arm=left_arm_orig,
                        left_hand=left_hand_fk,
                        right_arm=right_arm_orig,
                        right_hand=right_hand_fk,
                        waist=waist,
                    )
                    
                    # 从45维中提取完整的21维关键点：
                    # 输入对齐数据集输入：wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz, wrist_rotvec
                    # [0:21] = left: wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
                    # [21:42] = right: wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
                    left_keypoints_21d = state_45d[..., 0:21]    # (B, T, 21)
                    right_keypoints_21d = state_45d[..., 21:42]  # (B, T, 21)
                    
                    obs_copy["state.left_key_points"] = left_keypoints_21d
                    obs_copy["state.right_key_points"] = right_keypoints_21d
                    
                    print(f"[Policy Step1] Converted to keypoints:")
                    print(f"  left_key_points: {obs_copy['state.left_key_points'].shape}")
                    print(f"  right_key_points: {obs_copy['state.right_key_points'].shape}")
                    
                    # 保存原始的wrist pose，用于后续retarget时的参考
                    left_arm_state = left_arm_orig.copy()
                    right_arm_state = right_arm_orig.copy()
                    
                    # 移除原始的arm和hand数据（模型只需要keypoints）
                    obs_copy.pop("state.left_arm", None)
                    obs_copy.pop("state.left_hand", None)
                    obs_copy.pop("state.right_arm", None)
                    obs_copy.pop("state.right_hand", None)
        
        

        # ==========================================================
        # 第2步：模型推理
        # ==========================================================
        # 输入格式（经过第1步转换后）：
        #   - obs_copy["state.left_key_points"]: (B, T, 18) = [wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz]
        #   - obs_copy["state.right_key_points"]: (B, T, 18) = 同上
        #   - obs_copy["state.waist"]: (B, T, 3) = [waist joints]
        #   - obs_copy["video.xxx"]: 视频数据
        #   - obs_copy["annotation.xxx"]: 任务描述等
        # 
        # 输出格式：
        #   - unnormalized_action["action.left_key_points"]: (B, horizon, 18) = [预测的wrist_xyz, thumb_tip_xyz, ...]
        #   - unnormalized_action["action.right_key_points"]: (B, horizon, 18) = 同上
        #   - unnormalized_action["action.waist"]: (B, horizon, 3) = [预测的waist joints]
        # ==========================================================
        
        # Apply transforms (normalization, etc.)
        normalized_input = self.apply_transforms(obs_copy)
        
        # Get action from model
        normalized_action = self._get_action_from_normalized_input(normalized_input)
        
        # Unapply transforms (denormalization, etc.)
        unnormalized_action = self._get_unnormalized_action(normalized_action)
        


        if not is_batch and "robocasa" in self.embodiment_tag.value:
            unnormalized_action = squeeze_dict_values(unnormalized_action) # 因为robocasa支持并行环境评测，而robotwin不支持并行环境评测
        else:
            pass

        # ==========================================================
        # 第3步：输出处理 - 当use_eepose和use_fourier_hand_retarget同时为True时
        # 将模型输出的 hand keypoints 转换回 wrist pose + finger joints
        # ==========================================================
        # 输入格式（模型输出）：
        #   - unnormalized_action["action.left_key_points"]: (B, horizon, 18) = [wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz]
        #   - unnormalized_action["action.right_key_points"]: (B, horizon, 18) = 同上
        # 
        # 输出格式（转换后）：
        #   - unnormalized_action["action.left_arm"]: (B, horizon, 6) = [L_wrist_xyz(3), L_rotvec(3)]
        #   - unnormalized_action["action.left_hand"]: (B, horizon, 6) = [L_finger_q1~6] (数据集格式: pinky, ring, middle, index, thumb_pitch, thumb_yaw)
        #   - unnormalized_action["action.right_arm"]: (B, horizon, 6) = [R_wrist_xyz(3), R_rotvec(3)]
        #   - unnormalized_action["action.right_hand"]: (B, horizon, 6) = [R_finger_q1~6]
        # ==========================================================
        if (self.use_eepose and self.use_fourier_hand_retarget and 
            "robocasa" in self.embodiment_tag.value and
            "action.left_key_points" in unnormalized_action and 
            "action.right_key_points" in unnormalized_action):
            
            # ==========================================================
            # 调试功能：保存模型预测的关键点坐标到txt文件
            # ==========================================================
            if not hasattr(self, '_keypoints_log_initialized'):
                from datetime import datetime
                import os
                
                # 创建日志文件
                log_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._keypoints_log_file = log_dir / f"predicted_keypoints_{timestamp}.txt"
                self._keypoints_frame_counter = 0
                
                # 写入文件头
                with open(self._keypoints_log_file, 'w') as f:
                    f.write("# 模型预测的手部关键点坐标\n")
                    f.write("# 格式：frame_id t L_wrist_x L_wrist_y L_wrist_z L_thumb_tip_x L_thumb_tip_y L_thumb_tip_z L_index_tip_x L_index_tip_y L_index_tip_z L_middle_tip_x L_middle_tip_y L_middle_tip_z L_ring_tip_x L_ring_tip_y L_ring_tip_z L_pinky_tip_x L_pinky_tip_y L_pinky_tip_z L_wrist_rotvec_x L_wrist_rotvec_y L_wrist_rotvec_z R_wrist_x R_wrist_y R_wrist_z R_thumb_tip_x R_thumb_tip_y R_thumb_tip_z R_index_tip_x R_index_tip_y R_index_tip_z R_middle_tip_x R_middle_tip_y R_middle_tip_z R_ring_tip_x R_ring_tip_y R_ring_tip_z R_pinky_tip_x R_pinky_tip_y R_pinky_tip_z R_wrist_rotvec_x R_wrist_rotvec_y R_wrist_rotvec_z\n")
                    f.write("# 左手6个关键点: 手腕(L_hand_base_link) + 5个指尖(thumb/index/middle/ring/pinky)\n")
                    f.write("# 右手6个关键点: 手腕(R_hand_base_link) + 5个指尖(thumb/index/middle/ring/pinky)\n")
                    f.write("# batch=0的数据，每个时间步(t=0~15)记录一行\n")
                    f.write("#\n")
                
                self._keypoints_log_initialized = True
                print(f"[Keypoints Logger] 创建日志文件: {self._keypoints_log_file}")
            
            # 获取模型输出的keypoints
            # 输入：
            #   - pred_left_keypoints_seq: (B, horizon, 18) = flatten的6个关键点
            #   - pred_right_keypoints_seq: (B, horizon, 18) = 同上
            pred_left_keypoints_seq = unnormalized_action.get("action.left_key_points", None)
            pred_right_keypoints_seq = unnormalized_action.get("action.right_key_points", None)
            
            if pred_left_keypoints_seq is not None and pred_right_keypoints_seq is not None:
                batch_size, horizon, D_kp = pred_left_keypoints_seq.shape
                
                print(f"[Policy Step3] Input keypoints shape: left={pred_left_keypoints_seq.shape}, right={pred_right_keypoints_seq.shape}")
                
                # ==========================================================
                # 保存关键点数据到txt文件（只保存batch=0的数据）
                # ==========================================================
                if hasattr(self, '_keypoints_log_file'):
                    try:
                        with open(self._keypoints_log_file, 'a') as f:
                            # 遍历时间步（horizon维度）
                            for t in range(horizon):
                                # 提取batch=0的数据
                                left_kp = pred_left_keypoints_seq[0, t, :]  # (D_kp,) =  21
                                right_kp = pred_right_keypoints_seq[0, t, :] # (D_kp,)
                                
                                # 拼接左右手关键点
                                full_kp = np.concatenate([left_kp, right_kp])
                                
                                # 写入一行：frame_id t D_kp*2个坐标值
                                line = f"{self._keypoints_frame_counter} {t}"
                                for val in full_kp:
                                    line += f" {val:.6f}"
                                line += "\n"
                                f.write(line)
                        
                        self._keypoints_frame_counter += 1
                        
                        # 每10帧打印一次进度
                        if self._keypoints_frame_counter % 10 == 0:
                            print(f"[Keypoints Logger] 已保存 {self._keypoints_frame_counter} 帧数据")
                    except Exception as e:
                        print(f"[Keypoints Logger] 保存失败: {e}")
                # ==========================================================
                
                # 检查维度：支持 21维（新格式，包含wrist_rotvec）
                if D_kp == 21:
                    # 初始化存储结果的数组
                    # 输出格式：
                    #   - arm: (B, horizon, 6) = [wrist_xyz(3), rotvec(3)]
                    #   - hand: (B, horizon, 6) = [finger joints in dataset format]
                    retarget_left_arm_seq = np.zeros((batch_size, horizon, 6), dtype=np.float32)
                    retarget_right_arm_seq = np.zeros((batch_size, horizon, 6), dtype=np.float32)
                    retarget_left_hand_seq = np.zeros((batch_size, horizon, 6), dtype=np.float32)
                    retarget_right_hand_seq = np.zeros((batch_size, horizon, 6), dtype=np.float32)
                    
                    # 逐帧进行retarget转换
                    for t in range(horizon):
                        for b in range(batch_size):
                            
                            # 新格式：使用完整的45维（left_21 + right_21 + waist_3）
                            # 这样可以利用wrist_rotvec进行warmup
                            left_21 = pred_left_keypoints_seq[b, t, :]   # (21,)
                            right_21 = pred_right_keypoints_seq[b, t, :] # (21,)
                            waist_3 = np.zeros(3, dtype=np.float32)       # 占位
                            state_45d = np.concatenate([left_21, right_21, waist_3])  # (45,)
                            
                            # 调用新的retarget API（支持warmup）
                            result = self.fourier_hand_retargeter.retarget_from_45d(state_45d)
                        
                            # 提取左手的wrist pose和finger joints
                            if 'left' in result:
                                # wrist_pose: [wrist_xyz(3), rotvec(3)] = 6维
                                retarget_left_arm_seq[b, t] = result['left']['wrist_pose']
                                # finger_joints: [pinky, ring, middle, index, thumb_pitch, thumb_yaw] = 6维 (6个主动关节)
                                retarget_left_hand_seq[b, t] = result['left']['finger_joints']
                            
                            # 提取右手的wrist pose和finger joints
                            if 'right' in result:
                                retarget_right_arm_seq[b, t] = result['right']['wrist_pose']
                                retarget_right_hand_seq[b, t] = result['right']['finger_joints']
                    
                    # 将转换后的结果赋值到action字典中
                    # 输出格式：
                    #   - action.left_arm: (B, horizon, 6) = [L_wrist_xyz(3), L_rotvec(3)]
                    #   - action.left_hand: (B, horizon, 6) = [L_finger_q1~6] pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    #   - action.right_arm: (B, horizon, 6) = [R_wrist_xyz(3), R_rotvec(3)]
                    #   - action.right_hand: (B, horizon, 6) = [R_finger_q1~6] (数据集格式)

                    # 重排序finger joints: Retarget API输出格式 -> RoboCasa数据集格式
                    # API输出格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    # 数据集格式: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    
                    unnormalized_action["action.left_arm"] = retarget_left_arm_seq
                    # 只有索引0,1,2,3,5需要取负号，索引4(thumb_pitch)不需要
                    left_hand_processed = retarget_left_hand_seq.copy()
                    left_hand_processed[..., [0, 1, 2, 3, 5]] = -left_hand_processed[..., [0, 1, 2, 3, 5]]
                    unnormalized_action["action.left_hand"] = left_hand_processed # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    
                    unnormalized_action["action.right_arm"] = retarget_right_arm_seq
                    right_hand_processed = retarget_right_hand_seq.copy()
                    right_hand_processed[..., [0, 1, 2, 3, 5]] = -right_hand_processed[..., [0, 1, 2, 3, 5]]
                    unnormalized_action["action.right_hand"] = right_hand_processed # 直接使用Retarget API的输出 [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                    # ==========================================================
                    # 调试功能：保存每个chunk中每个时间步的retarget输出（wrist pose + finger joints）
                    # ==========================================================
                    if not hasattr(self, '_retarget_log_initialized'):
                        from datetime import datetime
                        
                        # 创建日志文件
                        log_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
                        log_dir.mkdir(parents=True, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self._retarget_log_file = log_dir / f"retargeted_actions_{timestamp}.txt"
                        self._retarget_chunk_counter = 0
                        
                        # 写入文件头
                        with open(self._retarget_log_file, 'w') as f:
                            f.write("# Retarget后的动作输出（每个chunk的每个时间步）\n")
                            f.write("# 格式：chunk_id t L_wrist_x L_wrist_y L_wrist_z L_rotvec_x L_rotvec_y L_rotvec_z L_finger_q1 L_finger_q2 L_finger_q3 L_finger_q4 L_finger_q5 L_finger_q6 R_wrist_x R_wrist_y R_wrist_z R_rotvec_x R_rotvec_y R_rotvec_z R_finger_q1 R_finger_q2 R_finger_q3 R_finger_q4 R_finger_q5 R_finger_q6\n")
                            f.write("# L_finger_joint_names_6: pinky, ring, middle, index, thumb_pitch, thumb_yaw]\n")
                            f.write("# batch=0的数据，每个chunk的16个时间步(t=0~15)各记录一行\n")
                            f.write("#\n")
                        
                        self._retarget_log_initialized = True
                        print(f"[Retarget Logger] 创建日志文件: {self._retarget_log_file}")
                    
                    # 保存当前chunk的retarget数据（只保存batch=0）
                    if hasattr(self, '_retarget_log_file'):
                        try:
                            # API已返回正确顺序的finger joints: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                            # 可直接用于仿真控制
                            
                            with open(self._retarget_log_file, 'a') as f:
                                # 遍历当前chunk的所有时间步（horizon维度，通常16步）
                                for t in range(horizon):
                                    # 提取batch=0的数据
                                    left_wrist_pose = retarget_left_arm_seq[0, t, :]     # (6,) = [wrist_xyz(3), rotvec(3)]
                                    left_finger_joints = retarget_left_hand_seq[0, t, :] # (6,) = [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                                    right_wrist_pose = retarget_right_arm_seq[0, t, :]   # (6,)
                                    right_finger_joints = retarget_right_hand_seq[0, t, :] # (6,) 
                                    
                                    # 拼接完整动作：[left_wrist(6), left_finger(6), right_wrist(6), right_finger(6)] = 24维
                                    full_action = np.concatenate([
                                        left_wrist_pose,        # L_wrist_xyz + L_rotvec
                                        left_finger_joints,     # L_finger_q1~6: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                                        right_wrist_pose,       # R_wrist_xyz + R_rotvec
                                        right_finger_joints     # R_finger_q1~6: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                                    ])
                                    
                                    # 写入一行：chunk_id t 24个数值
                                    line = f"{self._retarget_chunk_counter} {t}"
                                    for val in full_action:
                                        line += f" {val:.6f}"
                                    line += "\n"
                                    f.write(line)
                            
                            self._retarget_chunk_counter += 1
                            
                            # 每10个chunk打印一次进度
                            if self._retarget_chunk_counter % 10 == 0:
                                print(f"[Retarget Logger] 已保存 {self._retarget_chunk_counter} 个chunk数据 (共 {self._retarget_chunk_counter * horizon} 个时间步)")
                        except Exception as e:
                            print(f"[Retarget Logger] 保存失败: {e}")
                            import traceback
                            traceback.print_exc()
                    # ==========================================================
                    
                    
                    # 移除keypoints数据（客户端不需要）
                    unnormalized_action.pop("action.left_key_points", None)
                    unnormalized_action.pop("action.right_key_points", None)
                    
                    print(f"[Policy Step3] Converted keypoints to joint angles:")
                    print(f"  left_arm (wrist pose): (B={batch_size}, H={horizon}, 6)")
                    print(f"  left_hand (finger joints): (B={batch_size}, H={horizon}, 6)")
                    print(f"  right_arm (wrist pose): (B={batch_size}, H={horizon}, 6)")
                    print(f"  right_hand (finger joints): (B={batch_size}, H={horizon}, 6)")
                else:
                    print(f"[Policy Step3] Warning: action.left_key_points dim={D_kp}, expected 21. Skipping retarget.")
        
       

        # ==========================================================
        # 第4步（可选）：use_eepose的IK处理 - 将wrist pose转换为arm joint angles
        # 当use_eepose=True时，将第3步输出的6维wrist pose通过IK转换为7维arm joint angles
        # ==========================================================
        # 输入格式（如果经过第3步转换）：
        #   - unnormalized_action["action.left_arm"]: (B, horizon, 6) = [L_wrist_xyz(3), L_rotvec(3)]
        #   - unnormalized_action["action.right_arm"]: (B, horizon, 6) = [R_wrist_xyz(3), R_rotvec(3)]
        # 
        # 输出格式（经过IK转换后）：
        #   - unnormalized_action["action.left_arm"]: (B, horizon, 7) = [L_arm_joint_angles(7)]
        #   - unnormalized_action["action.right_arm"]: (B, horizon, 7) = [R_arm_joint_angles(7)]
        # ==========================================================
        if self.use_eepose:
            if "robocasa" in self.embodiment_tag.value:
                # 从模型输出的6-DoF EE Pose动作中提取 pos 和 axis-angle
                # 如果use_fourier_hand_retarget=True，这里的数据来自第3步的retarget输出
                # 如果use_fourier_hand_retarget=False，这里的数据来自模型直接输出
                pred_left_eepose_seq = unnormalized_action["action.left_arm"]
                pred_right_eepose_seq = unnormalized_action["action.right_arm"]
                
                batch_size, horizon, _ = pred_left_eepose_seq.shape
                
                # 初始化用于存储IK结果的数组
                q_left_arm_seq = np.zeros((batch_size, horizon, 7)) # 目标是7-DoF
                q_right_arm_seq = np.zeros((batch_size, horizon, 7)) # 目标是7-DoF

                # 使用输入时的原始7-DoF手臂状态作为第一个时间步的IK初始猜测
                # 形状从 (B, 1, 7) 变为 (B, 7)
                q_init_left = left_arm_state[:, -1, :] if left_arm_state is not None else None
                q_init_right = right_arm_state[:, -1, :] if right_arm_state is not None else None

                # 遍历动作序列的每一个时间步 (从 0 到 15)
                for t in range(horizon):
                    # 提取当前时间步 t 的EE Pose动作，形状为 (B, 6)
                    left_eepose_t = pred_left_eepose_seq[:, t, :]
                    right_eepose_t = pred_right_eepose_seq[:, t, :]
                    # print(f"Time step {t}: Left EE Pose shape: {left_eepose_t.shape}, Right EE Pose shape: {right_eepose_t.shape}")

                    # 将EE Pose分解为位置和轴角
                    left_hand_pos = left_eepose_t[:, :3]
                    left_hand_axisangle = left_eepose_t[:, 3:6]
                    right_hand_pos = right_eepose_t[:, :3]
                    right_hand_axisangle = right_eepose_t[:, 3:6]

                    # 执行IK计算，输入是 (B, 3)，输出是 (B, 7)
                    q_left_arm_t, q_right_arm_t = self.body_retargeter.inverse_kinematics_from_camera_axisangle(
                        left_hand_pos=left_hand_pos,
                        left_hand_axisangle=left_hand_axisangle,
                        right_hand_pos=right_hand_pos,
                        right_hand_axisangle=right_hand_axisangle,
                        current_action_vector=full_action_vector,
                        q_init_left=q_init_left,
                        q_init_right=q_init_right
                    )

                    # 将计算出的关节角存储到结果序列中
                    if q_left_arm_t is not None:
                        q_left_arm_seq[:, t, :] = q_left_arm_t
                    if q_right_arm_t is not None:
                        q_right_arm_seq[:, t, :] = q_right_arm_t
                    
                    # 使用当前步的IK解作为下一步的初始猜测，以保证动作的连续性
                    q_init_left = q_left_arm_t
                    q_init_right = q_right_arm_t

                    # 打印每个chunk的值（按照data_config顺序：right_arm, right_hand, left_arm, left_hand）
                    batch_idx = 0  # 假设只有一个batch
                    right_arm = q_right_arm_seq[batch_idx, t, :]  # shape: (6,)
                    left_arm = q_left_arm_seq[batch_idx, t, :]    # shape: (6,)
                    

                # 将完整的关节角序列更新回 unnormalized_action 字典
                unnormalized_action["action.left_arm"] = q_left_arm_seq
                unnormalized_action["action.right_arm"] = q_right_arm_seq

            elif "robotwin" in self.embodiment_tag.value:
                # 从模型输出的6-DoF EE Pose动作中提取 pos 和 axis-angle
                pred_left_eepose_seq = unnormalized_action["action.left_arm"]
                pred_right_eepose_seq = unnormalized_action["action.right_arm"]
                # print(f"Pred Left EE Pose shape: {pred_left_eepose_seq.shape}, Pred Right EE Pose shape: {pred_right_eepose_seq.shape}")
                
                batch_size, horizon, _ = pred_left_eepose_seq.shape
                # print(f"Batch size: {batch_size}, Horizon: {horizon}")
                
                # --- 关键：robotwin/aloha 手臂是 6DoF；robocasa/gr1 仍按 7DoF ---
                q_out_dim = 6 
                # 初始化用于存储IK结果的数组
                q_left_arm_seq = np.zeros((batch_size, horizon, q_out_dim)) # 目标是q_out_dim
                q_right_arm_seq = np.zeros((batch_size, horizon, q_out_dim)) 

                # 使用输入时的原始7-DoF手臂状态作为第一个时间步的IK初始猜测
                # 形状从 (B, 1, 7) 变为 (B, 7)
                q_init_left = left_arm_state[:, -1, :q_out_dim] if left_arm_state is not None else None
                q_init_right = right_arm_state[:, -1, :q_out_dim] if right_arm_state is not None else None

                # 遍历动作序列的每一个时间步 (从 0 到 15)
                for t in range(horizon):
                    # 提取当前时间步 t 的EE Pose动作，形状为 (B, 6)
                    left_eepose_t = pred_left_eepose_seq[:, t, :]
                    right_eepose_t = pred_right_eepose_seq[:, t, :]
                    # print(f"Time step {t}: Left EE Pose shape: {left_eepose_t.shape}, Right EE Pose shape: {right_eepose_t.shape}")
                    # print(f"Left EE Pose: {left_eepose_t}, Right EE Pose: {right_eepose_t}")
                    # 将EE Pose分解为位置和轴角
                    left_hand_pos = left_eepose_t[:, :3]
                    left_hand_axisangle = left_eepose_t[:, 3:6]
                    right_hand_pos = right_eepose_t[:, :3]
                    right_hand_axisangle = right_eepose_t[:, 3:6]
                    # print(f"Time {t}: Left hand pos: {left_hand_pos}, Right hand pos: {right_hand_pos}")
                    # print(f"Time {t}: Left hand axisangle: {left_hand_axisangle}, Right hand axisangle: {right_hand_axisangle}")

                    # 执行IK计算，输入是 (B, 3)，输出是 (B, 7)
                    q_left_arm_t, q_right_arm_t = self.body_retargeter.inverse_kinematics_from_camera_axisangle(
                        left_hand_pos=left_hand_pos,
                        left_hand_axisangle=left_hand_axisangle,
                        right_hand_pos=right_hand_pos,
                        right_hand_axisangle=right_hand_axisangle,
                        current_action_vector=full_action_vector,
                        q_init_left=q_init_left,
                        q_init_right=q_init_right
                    )

                    # 将计算出的关节角存储到结果序列中
                    if q_left_arm_t is not None:
                        q_left_arm_seq[:, t, :] = q_left_arm_t[:, :q_out_dim] if q_left_arm_t.ndim == 2 else q_left_arm_t[:q_out_dim]
                    if q_right_arm_t is not None:
                        q_right_arm_seq[:, t, :] = q_right_arm_t[:, :q_out_dim] if q_right_arm_t.ndim == 2 else q_right_arm_t[:q_out_dim]
                    
                    # 使用当前步的IK解作为下一步的初始猜测，以保证动作的连续性
                    q_init_left = q_left_arm_t
                    q_init_right = q_right_arm_t
                    
                # 将完整的关节角序列更新回 unnormalized_action 字典
                unnormalized_action["action.left_arm"] = q_left_arm_seq
                unnormalized_action["action.right_arm"] = q_right_arm_seq
                # Unnormalized Action: Left Arm shape: (1, 16, 6), Right Arm shape: (1, 16, 6)

            

            # ==========================================================
            # 新增功能：保存unnormalized_action到robocasa_action日志
            # 用于对比验证action数据是否正确传递
            # ==========================================================
            if not hasattr(self, '_robocasa_action_log_initialized'):
                from datetime import datetime
                
                # 创建日志文件
                log_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
                log_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._robocasa_action_log_file = log_dir / f"robocasa_action_{timestamp}.txt"
                self._robocasa_action_chunk_counter = 0
                
                # 写入文件头
                with open(self._robocasa_action_log_file, 'w') as f:
                    f.write("# RoboCasa unnormalized_action数据（每个chunk的每个时间步）\n")
                    f.write("# 格式：chunk_id t L_arm_q1 L_arm_q2 L_arm_q3 L_arm_q4 L_arm_q5 L_arm_q6 L_arm_q7 L_finger_q1 L_finger_q2 L_finger_q3 L_finger_q4 L_finger_q5 L_finger_q6 R_arm_q1 R_arm_q2 R_arm_q3 R_arm_q4 R_arm_q5 R_arm_q6 R_arm_q7 R_finger_q1 R_finger_q2 R_finger_q3 R_finger_q4 R_finger_q5 R_finger_q6 waist_q1 waist_q2 waist_q3\n")
                    f.write("# L_finger_joint_names_6: pinky, ring, middle, index, thumb_pitch, thumb_yaw]\n")
                    f.write("# batch=0的数据，每个chunk的16个时间步(t=0~15)各记录一行\n")
                    f.write("#\n")
                
                self._robocasa_action_log_initialized = True
                print(f"[RoboCasa Action Logger] 创建日志文件: {self._robocasa_action_log_file}")
            
            # 保存当前chunk的unnormalized_action数据（只保存batch=0）
            if hasattr(self, '_robocasa_action_log_file'):
                try:
                    with open(self._robocasa_action_log_file, 'a') as f:
                        # 遍历当前chunk的所有时间步（horizon维度，通常16步）
                        for t in range(horizon):
                            # 从unnormalized_action中提取batch=0的数据
                            left_arm = unnormalized_action["action.left_arm"][0, t, :]      # (7,) - arm joint angles
                            left_hand = unnormalized_action["action.left_hand"][0, t, :]    # (6,) - finger joints
                            right_arm = unnormalized_action["action.right_arm"][0, t, :]    # (7,) - arm joint angles
                            right_hand = unnormalized_action["action.right_hand"][0, t, :]  # (6,) - finger joints

                            waist = unnormalized_action["action.waist"][0, t, :]          # (3,) - waist joint angles
                            
                            # 拼接完整动作：[left_arm(6), left_hand(6), right_arm(6), right_hand(6)] = 24维
                            full_action = np.concatenate([
                                left_arm,     # L_arm_q1-q7
                                left_hand,    # L_finger_q1~6: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
                                right_arm,    # R_arm_q1-q7
                                right_hand,    # R_finger_q1~6: 
                                waist        # waist joint angles
                            ])
                            
                            # 写入一行：chunk_id t 26个数值 (7+6+7+6)
                            line = f"{self._robocasa_action_chunk_counter} {t}"
                            for val in full_action:
                                line += f" {val:.6f}"
                            line += "\n"
                            f.write(line)
                    
                    self._robocasa_action_chunk_counter += 1
                    
                    # 每10个chunk打印一次进度
                    if self._robocasa_action_chunk_counter % 10 == 0:
                        print(f"[RoboCasa Action Logger] 已保存 {self._robocasa_action_chunk_counter} 个chunk数据")
                except Exception as e:
                    print(f"[RoboCasa Action Logger] 保存失败: {e}")
                    import traceback
                    traceback.print_exc()
            # ==========================================================

        return unnormalized_action

    # Gr00t policy inference
    def _get_action_from_normalized_input(self, normalized_input: Dict[str, Any]) -> torch.Tensor:
        # Set up autocast context if needed
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            model_pred = self.model.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(self, normalized_action: torch.Tensor) -> Dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()})

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """
        Get the modality config for the model, overrides the base class method
        """
        return self._modality_config

    @property
    def modality_config(self) -> Dict[str, ModalityConfig]:
        return self._modality_config

    @property
    def modality_transform(self) -> ComposedModalityTransform:
        return self._modality_transform

    @property
    def video_delta_indices(self) -> np.ndarray:
        """Get the video delta indices."""
        return self._video_delta_indices

    @property
    def state_delta_indices(self) -> np.ndarray | None:
        """Get the state delta indices."""
        return self._state_delta_indices

    @property
    def denoising_steps(self) -> int:
        """Get the number of denoising steps."""
        return self.model.action_head.num_inference_timesteps

    @denoising_steps.setter
    def denoising_steps(self, value: int):
        """Set the number of denoising steps."""
        self.model.action_head.num_inference_timesteps = value

    def _check_state_is_batched(self, obs: Dict[str, Any]) -> bool:
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:  # (B, Time, Dim)
                return False
        return True

    def _load_model(self, model_path, 
                    backbone_type=None, 
                    enable_latent_alignment=None, 
                    use_dino=False, 
                    use_time_aware_action_head=False
                    ):
        if use_time_aware_action_head:
            model = GR00T_N1_5.from_pretrained(model_path, backbone_type=backbone_type, torch_dtype=COMPUTE_DTYPE,
                                            enable_latent_alignment=enable_latent_alignment,
                                            use_dino=use_dino,
                                            use_time_aware_action_head=use_time_aware_action_head,
                                            load_pretrained=True)
        else:
            model = GR00T_N1_5.from_pretrained(model_path, backbone_type=backbone_type, torch_dtype=COMPUTE_DTYPE,
                                            enable_latent_alignment=enable_latent_alignment,
                                            use_dino=use_dino,
                                            load_pretrained=False)

        model.eval()  # Set model to eval mode
        model.to(device=self.device)  # type: ignore

        self.model = model

    def _load_metadata(self, exp_cfg_dir: Path):
        """Load the transforms for the model."""
        # Load metadata for normalization stats
        metadata_path = exp_cfg_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        # Get metadata for the specific embodiment
        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}",
                f"make sure the metadata.json file is present at {metadata_path}",
            )

        metadata = DatasetMetadata.model_validate(metadata_dict)

        self._modality_transform.set_metadata(metadata)
        self.metadata = metadata

    def _load_horizons(self):
        """Load the horizons needed for the model."""
        # Get modality configs
        # Video horizons
        self._video_delta_indices = np.array(self._modality_config["video"].delta_indices)
        self._assert_delta_indices(self._video_delta_indices)
        self._video_horizon = len(self._video_delta_indices)
        # State horizons (if used)
        if "state" in self._modality_config:
            self._state_delta_indices = np.array(self._modality_config["state"].delta_indices)
            self._assert_delta_indices(self._state_delta_indices)
            self._state_horizon = len(self._state_delta_indices)
        else:
            self._state_horizon = None
            self._state_delta_indices = None

    def _assert_delta_indices(self, delta_indices: np.ndarray):
        """Assert that the delta indices are valid."""
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    # 组成44DOF向量的辅助函数
    @staticmethod
    def _build_full_44dof_vector(obs_dict):
        """
        从包含批处理数据的观测字典中构建一个 (batch_size, 44) 的完整状态向量。
        
        Args:
            obs_dict (Dict[str, np.ndarray]): 观测字典，其中 state 的形状为 (B, T, D)。
                                              B 是批次大小 (环境数量), T 是时间步, D 是特征维度。

        Returns:
            np.ndarray: 形状为 (B, 44) 的状态向量。
        """
        # 定义44-DoF向量中每个部分的起始和结束索引
        layout_44dof = {
            "left_arm": (0, 7), "left_hand": (7, 13), "left_leg": (13, 19),
            "neck": (19, 22), "right_arm": (22, 29), "right_hand": (29, 35),
            "right_leg": (35, 41), "waist": (41, 44),
        }

        # 从任意一个存在的状态键确定批次大小
        batch_size = 0
        for key in obs_dict:
            if key.startswith("state."):
                batch_size = obs_dict[key].shape[0]
                break
        
        if batch_size == 0:
            # 如果没有找到任何 state key，无法确定批次大小，返回空数组或抛出错误
            # 这里我们假设至少会有一个 state key
            # 如果可能完全没有state，则需要根据具体情况处理
            # 例如，可以尝试从 "video" key 获取 batch_size
            if "video.ego_view" in obs_dict:
                 batch_size = obs_dict["video.ego_view"].shape[0]
            else: # 默认返回一个 (1, 44) 的零向量
                return np.zeros((1, 44), dtype=np.float64)


        # 初始化一个 (batch_size, 44) 的零矩阵
        full_vector = np.zeros((batch_size, 44), dtype=np.float64)

        # 遍历布局，填充 full_vector
        for part_name, (start, end) in layout_44dof.items():
            obs_key = f"state.{part_name}"
            
            if obs_key in obs_dict:
                # 提取数据，形状为 (B, T, D)
                data = np.asarray(obs_dict[obs_key])
                
                # 我们只关心最后一个时间步的数据，其形状为 (B, D)
                last_time_step_data = data[:, -1, :]
                
                # 将数据填充到 full_vector 的正确位置
                full_vector[:, start:end] = last_time_step_data
            # 如果 obs_key 不在字典中，则该部分将保持为零，符合要求

        return full_vector
    
    # 组成Robotwin/Aloha 14DOF向量的辅助函数（单帧）
    @staticmethod
    def _build_single_14dof_vector_robotwin(obs_dict) -> np.ndarray:
        """
        从观测字典中构建一个 (14,) 的 Robotwin/Aloha state 向量（单帧）。

        约定输出 layout：
        [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]

        兼容输入 state 形状：
        - (B, T, D) 取最后一步 -> (B, D)，再取 batch=0 变成单帧
        - (T, D) / (D,) 也做兼容
        """
        layout_14dof = {
            "left_arm": (0, 6),
            "left_gripper": (6, 7),
            "right_arm": (7, 13),
            "right_gripper": (13, 14),
        }

        def _get_last_step(key: str):
            if key not in obs_dict:
                return None
            arr = np.asarray(obs_dict[key])

            # (B,T,D) -> (B,D)
            if arr.ndim == 3:
                arr = arr[:, -1, :]
                arr = arr[0]  # 取单帧 (D,)
            # (T,D) -> (D,)
            elif arr.ndim == 2:
                arr = arr[-1, :]
            # (D,) 保持
            elif arr.ndim == 1:
                pass
            else:
                raise ValueError(f"{key} 维度不支持: shape={arr.shape}")

            return np.asarray(arr, dtype=np.float64).reshape(-1)

        full_vec = np.zeros((14,), dtype=np.float64)

        # arm 必须存在
        la = _get_last_step("state.left_arm")
        ra = _get_last_step("state.right_arm")
        if la is None or ra is None:
            raise KeyError("robotwin 需要至少包含 state.left_arm 和 state.right_arm 来拼 14d state")

        # gripper：优先专用 key；否则退回用 hand 的第0维；都没有就置0
        lg = _get_last_step("state.left_gripper")
        rg = _get_last_step("state.right_gripper")
        if lg is None:
            lh = _get_last_step("state.left_hand")
            lg = lh[:1] if lh is not None else np.zeros((1,), dtype=np.float64)
        else:
            lg = lg[:1]

        if rg is None:
            rh = _get_last_step("state.right_hand")
            rg = rh[:1] if rh is not None else np.zeros((1,), dtype=np.float64)
        else:
            rg = rg[:1]

        # 填充
        full_vec[layout_14dof["left_arm"][0]:layout_14dof["left_arm"][1]] = la[:6]
        full_vec[layout_14dof["right_arm"][0]:layout_14dof["right_arm"][1]] = ra[:6]
        full_vec[layout_14dof["left_gripper"][0]:layout_14dof["left_gripper"][1]] = lg
        full_vec[layout_14dof["right_gripper"][0]:layout_14dof["right_gripper"][1]] = rg

        return full_vec
        

#######################################################################################################




# Helper functions
def unsqueeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unsqueeze the values of a dictionary.
    This converts the data to be batched of size 1.
    """
    unsqueezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            unsqueezed_data[k] = np.expand_dims(v, axis=0)
        elif isinstance(v, list):
            unsqueezed_data[k] = np.array(v)
        elif isinstance(v, torch.Tensor):
            unsqueezed_data[k] = v.unsqueeze(0)
        else:
            unsqueezed_data[k] = v
    return unsqueezed_data


def squeeze_dict_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Squeeze the values of a dictionary. This removes the batch dimension.
    """
    squeezed_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            squeezed_data[k] = np.squeeze(v)
        elif isinstance(v, torch.Tensor):
            squeezed_data[k] = v.squeeze()
        else:
            squeezed_data[k] = v
    return squeezed_data
