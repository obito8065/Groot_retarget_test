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
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)
from gr00t.model.policy import BasePolicy

import multiprocessing as mp
import random
from gymnasium import Wrapper


class ResetSeedFromQueueWrapper(Wrapper):
    def __init__(self, env, seed_queue, env_idx: int):
        super().__init__(env)
        self.seed_queue = seed_queue
        self.env_idx = env_idx

    def reset(self, *, seed=None, options=None):
        # 用 timeout 防止队列空时“悄悄卡死”
        try:
            s = self.seed_queue.get(timeout=60)
        except Exception as e:
            raise RuntimeError(f"[Env{self.env_idx}] seed_queue 取 seed 超时/失败，可能 seed 不够用或 Manager 已退出: {e}")

        print(f"[Env{self.env_idx}] reset(seed={int(s)})", flush=True)
        obs, info = self.env.reset(seed=int(s), options=options)
        # 额外把 seed 写进 info，便于你在主进程也能看到（初始 reset 时最有用）
        try:
            info = dict(info)
            info["reset_seed"] = int(s)
        except Exception:
            pass
        return obs, info

def _seed_worker(seed: int):
    # 覆盖 worker 进程内可能用到的 python/numpy 全局 RNG
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440
    # 新增：保存带有 chunk/step 标注的 substep 视频（用于调试和验证）
    save_substep_video: bool = False


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)
    episode_seed_start: int = 0  # 新增仿真评测任务的每个episode的随机种子起始值





class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555, use_msgpack: bool=True):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port, use_msgpack=use_msgpack)
        self.env = None


    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call_endpoint("get_action", observations)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # 只在每次 run_simulation 构建 env 时创建一次队列（每个 task 独立）
        self._seed_manager = mp.Manager()     
        seed_queue = self._seed_manager.Queue() 

        # 放入本次评测要用的全部 seeds：seed_start..seed_start+n_episodes-1
        # for s in range(config.episode_seed_start, config.episode_seed_start + config.n_episodes + config.n_envs): # 随机种子给一点冗余防止堵塞
        for s in range(config.episode_seed_start, config.episode_seed_start + config.n_episodes): 
            seed_queue.put(int(s))
            
        # 给每个 env fn 传入同一个 seed_queue（共享消耗）
        env_fns = [
            partial(_create_single_env, config=config, idx=i, seed_queue=seed_queue)
            for i in range(config.n_envs)
        ]

        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""

        if config.n_envs > config.n_episodes:
            raise ValueError(f"n_envs({config.n_envs}) 不能大于 n_episodes({config.n_episodes})，否则 seed_queue 会被取空并阻塞。")


        start_time = time.time()

        # 释放 seed 队列资源
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # if use_eepose:
        #     eepose_config = EEPoseConfig()
        #     retarget = BodyRetargeter(urdf_path=eepose_config.urdf_path, camera_intrinsics=eepose_config.camera_intrinsics)
        #     print("EEPose transformation enabled for GR1.")
        #     pass

        # Set up the environment
        self.env = self.setup_environment(config)
        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        # Initial environment reset
        obs, _ = self.env.reset()

        # 第一次 rollout：强制让 server 清掉所有 env slot 的 IK 历史缓存
        obs["meta.reset_mask"] = np.ones((config.n_envs,), dtype=bool)

       
        # if use_eepose:
        #     action_vector = np.concatenate(obs["state.left_arm"][0], axis=0)


        # Main simulation loop
        while completed_episodes < config.n_episodes:
            # Process observations and get actions from the server
            actions = self._get_actions_from_server(obs)
            
            # print("\n--- Debugging actions received from server ---")
            # for key, value in actions.items():
            #     if isinstance(value, np.ndarray):
            #         print(f"  Key: '{key}', Type: {type(value)}, Shape: {value.shape}")
            #     elif isinstance(value, list) and len(value) > 0:
            #         print(f"  Key: '{key}', Type: list, Length: {len(value)}, Shape of element 0: {getattr(value[0], 'shape', 'N/A')}")
            #     else:
            #         print(f"  Key: '{key}', Type: {type(value)}")
            # print("--------------------------------------------\n")
            
            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)

            # 把“本步结束的 env slot”作为下一步 obs 的 reset 信号发给 server
            reset_mask_next = (np.asarray(terminations) | np.asarray(truncations)).astype(bool)
            next_obs["meta.reset_mask"] = reset_mask_next
            ended_envs = np.where(reset_mask_next)[0].tolist()
            if ended_envs:
                print(f"[Client: Simulation] episode ended -> next obs will reset IK for env_idx={ended_envs}", flush=True)

            # Update episode tracking
            for env_idx in range(config.n_envs):
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            obs = next_obs
        # Clean up
        # self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_server(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions


def _create_single_env(config: SimulationConfig, idx: int, seed_queue: mp.Queue) -> gym.Env:
    """Create a single environment with appropriate wrappers."""

    # worker 进程初始化阶段先定一个基础 seed（防止 env 构建阶段随机）
    _seed_worker(int(config.episode_seed_start) + int(idx))

    # Create base environment
    env = gym.make(config.env_name, 
                    enable_render=True,
              
                    ) # 为False回导致视频无法录制
    # 拦截 reset，从队列取 seed
    env = ResetSeedFromQueueWrapper(env, seed_queue, env_idx=idx)

    # Add video recording wrapper if needed (only for the first environment)x
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
        save_substep_video=config.multistep.save_substep_video,
        video_dir=config.video.video_dir,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port, use_msgpack=True)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
