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

from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
import cv2

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def stack_repeated(x, n, loc):
    return np.repeat(np.expand_dims(x, axis=loc), n, axis=loc)


def repeated_box(box_space, n, loc):
    return spaces.Box(
        low=stack_repeated(box_space.low, n, loc),
        high=stack_repeated(box_space.high, n, loc),
        shape=box_space.shape[:loc] + (n,) + box_space.shape[loc:],
        dtype=box_space.dtype,
    )


def repeated_space(space, n, loc=0):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n, loc)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n, loc)
        return result_space
    elif isinstance(space, spaces.Discrete):
        return spaces.MultiDiscrete([[space.n] for _ in range(n)])
    elif isinstance(space, spaces.Text):  # For language, we don't repeat and only keep the last one
        return space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    if method == "max":
        # equivalent to any
        return np.max(data)
    elif method == "min":
        # equivalent to all
        return np.min(data)
    elif method == "mean":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    else:
        raise NotImplementedError()


class MultiStepWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_delta_indices,
        state_delta_indices,
        n_action_steps,
        max_episode_steps=None,
        reward_agg_method="max",
        save_substep_video: bool = False,
        video_dir: str = None,
    ):
        """
        video_delta_indices: np.ndarray[int], please check `assert_delta_indices` to see the requirements
        state_delta_indices: np.ndarray[int] | None, please check `assert_delta_indices` to see the requirements
          if None, it means the model is vision-only
        save_substep_video: bool, if True, save a video with chunk/step annotations
        video_dir: str, directory to save substep video
        """
        super().__init__(env)
        # Assign action space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        
        # === 新增：保存带有 chunk/step 标注的 substep 视频 ===
        self.save_substep_video = save_substep_video
        self.video_dir = video_dir
        self._substep_counter = 0  # 全局 substep 计数器
        self._chunk_counter = 0    # chunk 计数器
        self._episode_counter = 0  # episode 计数器
        self._video_writer = None  # 视频写入器
        self._video_fps = 30       # 视频帧率
        
        if self.save_substep_video and self.video_dir:
            self._video_dir_path = Path(self.video_dir)
            self._video_dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[MultiStepWrapper] 将保存 substep 视频到: {self._video_dir_path}")

        # Assign delta indices and horizons
        self.video_delta_indices = video_delta_indices
        self.video_horizon = len(video_delta_indices)
        self.assert_delta_indices(self.video_delta_indices, self.video_horizon)
        if state_delta_indices is not None:
            self.state_delta_indices = state_delta_indices
            self.state_horizon = len(state_delta_indices)
            self.assert_delta_indices(self.state_delta_indices, self.state_horizon)
        else:
            self.state_horizon = None
            self.state_delta_indices = None

        # Assign observation space
        self._observation_space = self.convert_observation_space(
            self.observation_space,
            self.video_horizon,
            self.state_horizon,
        )

        # Assign other attributes
        self.max_episode_steps = max_episode_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.max_steps_needed = self.get_max_steps_needed()

        self.obs = deque(maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.max_steps_needed + 1))

    def convert_observation_space(self, observation_space, video_horizon, state_horizon):
        """
        For video, the observation space will be (video_horizon,) + original shape
        For state (if not None), the observation space will be (state_horizon,) + original shape
        """
        new_observation_space = {}
        for k in observation_space.keys():
            if k.startswith("video"):
                box = observation_space[k]
                horizon = video_horizon
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("state"):
                box = observation_space[k]
                if state_horizon is not None:
                    horizon = state_horizon
                else:
                    # Don't include the state in the observation space
                    continue
                new_observation_space[k] = repeated_space(box, horizon)
            elif k.startswith("annotation"):
                text = observation_space[k]
                new_observation_space[k] = text
            else:
                raise ValueError(f"Unknown key: {k}")  # NOTE: We might add "language" in the future

        return spaces.Dict(new_observation_space)

    def get_max_steps_needed(self):
        """
        Get the maximum number of steps that we need to cache.
        """
        video_max_steps_needed = (
            np.max(self.video_delta_indices) - np.min(self.video_delta_indices) + 1
        )
        if self.state_delta_indices is not None:
            state_max_steps_needed = (
                np.max(self.state_delta_indices) - np.min(self.state_delta_indices) + 1
            )
        else:
            state_max_steps_needed = 0
        return int(max(video_max_steps_needed, state_max_steps_needed))

    def assert_delta_indices(self, delta_indices: np.ndarray, horizon: int):
        # Check the length
        # (In this wrapper, this seems redundant because we get the horizon from the delta indices. But in the policy, the horizon is not derived from the delta indices but we need to make it consistent. To make the function consistent, we keep the check here.)
        assert len(delta_indices) == horizon, f"{delta_indices=}, {horizon=}"
        # All delta indices should be non-positive because there's no way to get the future observations
        assert np.all(delta_indices <= 0), f"{delta_indices=}"
        # The last delta index should be 0 because it doesn't make sense to not use the latest observation
        assert delta_indices[-1] == 0, f"{delta_indices=}"
        if len(delta_indices) > 1:
            # The step is consistent (because in real robot experiments, we actually use the dt to get the observations, which requires the step to be consistent)
            assert np.all(
                np.diff(delta_indices) == delta_indices[1] - delta_indices[0]
            ), f"{delta_indices=}"
            # And the step is positive
            assert (delta_indices[1] - delta_indices[0]) > 0, f"{delta_indices=}"

    def _init_video_writer(self, frame_shape):
        """
        初始化视频写入器。
        
        Args:
            frame_shape: 帧的形状 (H, W, C)
        """
        if self._video_writer is not None:
            self._release_video_writer()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"substep_video_ep{self._episode_counter:03d}_{timestamp}.mp4"
        video_path = self._video_dir_path / video_filename
        
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(str(video_path), fourcc, self._video_fps, (w, h))
        print(f"[MultiStepWrapper] 开始录制视频: {video_path}")
    
    def _release_video_writer(self):
        """
        释放视频写入器，保存视频文件。
        """
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            print(f"[MultiStepWrapper] 视频录制完成")
    
    def _save_substep_image(self, observation: dict, chunk_id: int, substep_id: int):
        """
        保存每个 substep 的 observation 图像到视频，并在左上角标注 chunk 和 step 信息。
        
        Args:
            observation: 当前 step 的 observation 字典
            chunk_id: 当前 chunk 的 ID
            substep_id: 当前 substep 的 ID (0 ~ n_action_steps-1)
        """
        try:
            # 优先使用原始 1280x800 图像 (video.egoview)
            # 如果没有则使用处理后的 256x256 图像
            if 'video.egoview' in observation:
                img = observation['video.egoview']
            else:
                # 回退到其他 video key
                ego_view_key = None
                for key in observation.keys():
                    if 'ego_view' in key.lower() or 'video' in key.lower():
                        ego_view_key = key
                        break
                
                if ego_view_key is None:
                    return
                
                img = observation[ego_view_key]
            
            # 处理可能的维度问题
            if img.ndim == 4:  # (B, H, W, C) 或 (T, H, W, C)
                img = img[0]
            
            # 确保是 uint8 格式
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 如果是 RGB，转换为 BGR（OpenCV 使用 BGR）
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 初始化视频写入器（第一帧时）
            if self._video_writer is None:
                self._init_video_writer(img.shape)
            
            # 在图像左上角添加文字标注
            img_with_text = img.copy()
            text = f"Chunk: {chunk_id:04d}  Step: {substep_id:02d}"
            
            # 根据图像尺寸调整字体大小
            h, w = img_with_text.shape[:2]
            font_scale = max(0.5, min(h, w) / 400)  # 根据图像尺寸自适应
            thickness = max(1, int(font_scale * 2))
            
            # 添加黑色背景矩形，使文字更清晰
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img_with_text, (5, 5), (15 + text_w, 15 + text_h + baseline), (0, 0, 0), -1)
            
            # 添加白色文字
            cv2.putText(
                img_with_text, 
                text, 
                (10, 10 + text_h), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                thickness
            )
            
            # 写入视频帧
            self._video_writer.write(img_with_text)
            
        except Exception as e:
            print(f"[MultiStepWrapper] 保存 substep 图像失败: {e}")

    def reset(self, seed=None, options=None):
        """Resets the environment using kwargs."""
        obs, info = super().reset(seed=seed, options=options)
        
        # 保存上一个 episode 的视频（如果有的话）
        if self.save_substep_video and self._video_writer is not None:
            self._release_video_writer()
            self._episode_counter += 1
        
        # 重置 chunk 计数器（新 episode 开始）
        self._chunk_counter = 0
        self._substep_counter = 0

        self.obs = deque([obs] * (self.max_steps_needed + 1), maxlen=self.max_steps_needed + 1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.max_steps_needed + 1))

        obs = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        info = {k: [v] for k, v in info.items()}
        return obs, info

    def step(self, action):
        """
        action: dict: key-value pairs where the values are of shape (n_action_steps,) + action_shape
        """
        states = []
        rewards = []
        dones = []
        for step in range(self.n_action_steps):
            act = {}
            for key, value in action.items():
                act[key] = value[step, :]
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, truncated, info = super().step(act)
            
            # === 保存每个 substep 到视频 ===
            if self.save_substep_video and self.video_dir:
                self._save_substep_image(observation, self._chunk_counter, step)
            
            env_state = {"states": [], "model": []}
            states.append(env_state["states"])
            rewards.append(reward)
            dones.append(done)
            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) and (
                len(self.reward) >= self.max_episode_steps
            ):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)
            self._substep_counter += 1
        
        # 增加 chunk 计数器
        self._chunk_counter += 1

        observation = self._get_obs(self.video_delta_indices, self.state_delta_indices)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, "max")
        info = dict_take_last_n(self.info, self.max_steps_needed)
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        info["states"] = states
        info["rewards"] = rewards
        info["model"] = env_state["model"]
        info["actions"] = action
        info["dones"] = dones
        return observation, reward, done, truncated, info

    def _get_obs(self, video_delta_indices, state_delta_indices):
        """
        Output:
        For video: (video_horizon,) + obs_shape
        For state (if not None): (state_horizon,) + obs_shape
        """
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                if key.startswith("video"):
                    """
                    NOTE:
                      We need to subtract 1 because video_delta_indices is 0-indexed.
                      E.g., video_delta_indices = np.array([-4, -3, -2, -1, 0])
                      Then when we select the observation,
                        it should be [obs[-5], obs[-4], obs[-3], obs[-2], obs[-1]]
                      (i.e., the latest observation is at the last index)
                    """
                    delta_indices = video_delta_indices - 1
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("state"):
                    if state_delta_indices is not None:
                        delta_indices = state_delta_indices - 1
                    else:
                        raise ValueError(
                            f"state_delta_indices is None but `state` is still in the {self.observation_space=}"
                        )
                    this_obs = [self.obs[i][key] for i in delta_indices]
                    result[key] = np.stack(this_obs, axis=0)
                elif key.startswith("annotation"):
                    result[key] = self.obs[-1][key]
                else:
                    raise ValueError(f"Unknown key: {key}")
            return result
        else:
            raise RuntimeError(f"Unsupported space type: {type(self.observation_space)=}")

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)

    def get_rewards(self):
        return self.reward

    def get_attr(self, name):
        return getattr(self, name)

    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result
    
    def close(self):
        """关闭环境并释放资源。"""
        # 保存最后一个 episode 的视频
        if self.save_substep_video and self._video_writer is not None:
            self._release_video_writer()
        super().close()