"""
    脚本功能： pi0_simulation_service.py
    1. 启动一个服务器，定义robocasa的仿真环境（直接定义modality_config在服务端）
    2. pi0服务端只需要定一个get_action接口
"""

import argparse

import numpy as np

from gr00t.eval.robot import RobotInferenceServer
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from dataclasses import dataclass
import tyro

@dataclass
class ArgsConfig:
    client: bool = False
    """Whether to run the client."""
    
    data_config: str = "fourier_gr1_arms_waist"
    """The name of the data config to use."""
    
    env_name: str = "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    """Name of the environment to run."""
    
    port : int = 8070
    """The port number for the server."""
    
    host: str = "localhost"
    """The host address for the server."""
    
    video_dir: str = "./videos"
    """Directory to save videos."""
    
    n_episodes: int = 2
    """Number of episodes to run."""
    
    n_envs: int = 1
    """Number of parallel environments."""
    
    n_action_steps: int = 16
    """Number of action steps per environment step."""
    
    max_episode_steps: int = 1440
    """Maximum number of steps per episode."""
    
#####################################################################################

    
def main(args: ArgsConfig):
    if args.client:
        
        simulation_client = SimulationInferenceClient(host=args.host, port=args.port)
        print("Available modality configs")
        
        data_config = DATA_CONFIG_MAP[args.data_config]
        modality_configs = data_config.modality_config()
        print(modality_configs.keys())
        
        # Create simulation configuration
        config = SimulationConfig(
            env_name=args.env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=args.video_dir),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
            ),
        )
        
        # Run the simulation
        print(f"Running simulation for {args.env_name}...")
        env_name, episode_successes = simulation_client.run_simulation(config)

        # Print results
        print(f"Results for {env_name}:")
        print(f"Success rate: {np.mean(episode_successes):.2f}")

    else:
        raise ValueError("Please specify --client")
        
if __name__ == "__main__":
    args = tyro.cli(ArgsConfig)
    main(args)