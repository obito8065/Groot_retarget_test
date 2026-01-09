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

from dataclasses import dataclass
from typing import Literal
import os 
import sys
# Add current directory to Python path
SOURCE_PATH = os.getcwd()
if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)

# Also add to PYTHONPATH environment variable
os.environ['PYTHONPATH'] = f"{SOURCE_PATH}:{os.environ.get('PYTHONPATH', '')}"

import numpy as np
import tyro

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient, RobotInferenceServer
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

import random
import torch

# 模型推理时固定随机种子
def set_seed_everywhere(seed: int):
    print(f"server policy set_seed_everywhere: {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["INFERENCE_SEED"] = str(seed) # flowmatching的噪声随机种子固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 添加 CUDA 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ArgsConfig:
    """Command line arguments for the inference service."""

    model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path to the model checkpoint directory."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """The embodiment tag for the model."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_waist"
    """The name of the data config to use."""

    port: int = 5555
    """The port number for the server."""

    host: str = "localhost"
    """The host address for the server."""

    server: bool = False
    """Whether to run the server."""

    client: bool = False
    """Whether to run the client."""

    denoising_steps: int = 4
    """The number of denoising steps to use."""

    api_token: str = None
    """API token for authentication. If not provided, authentication is disabled."""

    ########### qwen2.5-vl ##############
    backbone_type: str = None
    """VLM backbone. [eagle, qwen2_5_vl]"""

    use_eepose: bool = False
    """Whether to use EEPose for input and output."""

    use_fourier_hand_retarget: bool = False
    """Whether to use Fourier Hand Retarget for input and output."""

    seed: int = 0
    """The seed for the model."""

    use_msgpack: bool = False
    """Whether to use msgpack for the server."""


#####################################################################################


def main(args: ArgsConfig):
    
    ####### run server  ########
    if args.server:
        # Create a policy
        # The `Gr00tPolicy` class is being used to create a policy object that encapsulates
        # the model path, transform name, embodiment tag, and denoising steps for the robot
        # inference system. This policy object is then utilized in the server mode to start
        # the Robot Inference Server for making predictions based on the specified model and
        # configuration.

        # we will use an existing data config to create the modality config and transform
        # if a new data config is specified, this expect user to
        # construct your own modality config and transform
        # see gr00t/utils/data.py for more details

        set_seed_everywhere(args.seed)  #固定随机种子


        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        print(f"debug infer use_eepose:{args.use_eepose}")
        print(f"debug infer port: {args.port}")
        policy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            backbone_type=args.backbone_type,
            use_eepose=args.use_eepose,
            use_fourier_hand_retarget=args.use_fourier_hand_retarget,
        )

        # Start the server
        server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token, use_msgpack=True)
        server.run()

    elif args.client:
        import time

        # In this mode, we will send a random observation to the server and get an action back
        # This is useful for testing the server and client connection
        # Create a policy wrapper
        set_seed_everywhere(args.seed)

        policy_client = RobotInferenceClient(
            host=args.host, port=args.port, api_token=args.api_token
        )

        print("Available modality config available:")
        modality_configs = policy_client.get_modality_config()
        print(modality_configs.keys())

        # Making prediction...
        # - obs: video.ego_view: (1, 256, 256, 3)
        # - obs: state.left_arm: (1, 7)
        # - obs: state.right_arm: (1, 7)
        # - obs: state.left_hand: (1, 6)
        # - obs: state.right_hand: (1, 6)
        # - obs: state.waist: (1, 3)

        # - action: action.left_arm: (16, 7)
        # - action: action.right_arm: (16, 7)
        # - action: action.left_hand: (16, 6)
        # - action: action.right_hand: (16, 6)
        # - action: action.waist: (16, 3)
        obs = {
            "video.ego_view": np.random.randint(0, 256, (1, 256, 256, 3), dtype=np.uint8),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["do your thing!"],
        }
        print(f"Obs.state.left_arm: {obs['state.left_arm']}")
        print(f"Obs.state.right_arm: {obs['state.right_arm']}")
        print(f"Obs.state.left_hand: {obs['state.left_hand']}")
        print(f"Obs.state.right_hand: {obs['state.right_hand']}")
        print(f"Obs.state.waist: {obs['state.waist']}")

        time_start = time.time()
        action = policy_client.get_action(obs)
        print(f"Total time taken to get action from server: {time.time() - time_start} seconds")
        print(f"Action: {action}")

        for key, value in action.items():
            print(f"Action: {key}: {value.shape}")
        
        print(f"Action: {action}")

    else:
        raise ValueError("Please specify either --server or --client")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config)

"""

python3 scripts/inference_service.py --server \
    --model_path /mnt/workspace/users/lijiayi/checkpoints/GR00T-N1.5-3B \
    --data_config fourier_gr1_arms_waist 

python scripts/inference_service.py --client \


# robotwin:
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA
python3 scripts/inference_service.py --server \
    --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robotwin_ckpt_10tasks_sample/n1.5_nopretrain_finetuneALL_on_robotwin_eepose_v0.1/checkpoint-7740 \
    --data_config robotwin\
    --embodiment_tag robotwin \
    --port 8030 \
    --use_eepose

"""