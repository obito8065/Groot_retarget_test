import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory, calc_mse_for_rm_trajectory

from gr00t.inference.connect_with_grpc import interact_with_grpc
from transforms3d.euler import quat2axangle, euler2quat
from transforms3d.quaternions import mat2quat
from transforms3d.axangles import axangle2mat


warnings.simplefilter("ignore", category=FutureWarning)

@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "localhost"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    plot: bool = False
    """Whether to plot the images."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data config to use."""

    steps: int = 15
    """Number of steps to evaluate."""

    trajs: int = 1
    """Number of trajectories to evaluate."""

    action_horizon: int = 16
    """Action horizon to evaluate."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for various codec options. h264: decord or av: torchvision_av"""

    dataset_path: str = "demo_data/robot_sim.PickNPlace/"
    """Path to the dataset."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    model_path: str = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    ########## CONFIGS ###########
    obs_camera_name: List[str] = field(default_factory=lambda: ["image_head_left"]) # image_head_right
    """GRPC传入的图像"""

    simulate: bool = False
    """Whether to simulate grpc."""

    prompt: str = "Take the cup"


def obs_to_datapoint(image, obs, prompt):
    """
    [
    'video.cam_head_left',  (1, 720, 1280, 3)
    'state.arm_right_position_state',  (1, 3)
    'state.arm_right_axangle_state',  (1, 3)
    'state.hand_right_pose_state',    (1, 6)
    'action.arm_right_position_action', 'action.arm_right_axangle_action', 'action.hand_right_pose_action',
    'annotation.human.action.task_description' ['Take the cup and raise it']
    ]
    """

    data_point = {}

    data_point["video.cam_head_left"] = image['image_head_left'][None, ...]
    data_point["state.arm_right_position_state"] = obs['right_arm_pose'][None, :3]

    ############### 四元数转轴角 ################
    eef_pos = obs['right_arm_pose']
    rot_quat = [eef_pos[6], eef_pos[3], eef_pos[4], eef_pos[5]]
    rot_axisangle = quat2axangle(rot_quat)
    rot_axisangle_new = rot_axisangle[0] * rot_axisangle[1]
    axis_angle = np.array([rot_axisangle_new[0],rot_axisangle_new[1],rot_axisangle_new[2]])
    data_point["state.arm_right_axangle_state"] = axis_angle[None, :3]

    data_point["state.hand_right_pose_state"] = obs['right_hand_angle'][None, :]
    data_point["annotation.human.action.task_description"] = [prompt]

    return data_point

def pred_to_realman(raw_action):
    action = {}
    action["world_vector"] = raw_action["world_vector"]
    action_rotation_delta = np.asarray(
        raw_action["rotation_delta"], dtype=np.float64
    )

    # 轴角->四元数
    angle = np.linalg.norm(action_rotation_delta)
    axis = action_rotation_delta / (angle + 1e-6)
    action["quaternion"] = mat2quat(axangle2mat(axis, angle))

    # 欧拉->四元数
    # roll, pitch, yaw = action_rotation_delta
    # action["quaternion"] = euler2quat(roll, pitch, yaw)

    action["gripper"] = raw_action["open_gripper"].astype(int)
    return action

def main(args: ArgsConfig):
    data_config = DATA_CONFIG_MAP[args.data_config]
    if args.model_path is not None:
        import torch

        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        policy: BasePolicy = Gr00tPolicy(
            model_path=args.model_path,
            modality_config=modality_config,
            modality_transform=modality_transform,
            embodiment_tag=args.embodiment_tag,
            denoising_steps=args.denoising_steps,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        policy: BasePolicy = RobotInferenceClient(host=args.host, port=args.port)

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    print("Current modality config: \n", modality)

    running = True
    prompt = str(args.prompt)
    steps = args.steps

    while running:
        # key = input("||| [start model] ||| please input key: ")
        # if key == "q":
        #     break

        import ipdb; ipdb.set_trace()
        args.prompt = prompt
        args.steps = steps

        interact = interact_with_grpc()
        if args.simulate:
            image, obs = interact.get_state_grpc(camera_name=args.obs_camera_name, simulate=True)
        else:
            image, obs = interact.get_state_grpc(camera_name=args.obs_camera_name)

        # 四元数转轴角
        data_point = obs_to_datapoint(image, obs, args.prompt)

        for step_count in range(args.steps):
            # import ipdb; ipdb.set_trace()
            print("inferencing at step: ", step_count)
            raw_action_chunk = policy.get_action(data_point)

            # print(raw_action_chunk)

            """
            un-normalized actions:
            'action.arm_right_position_action':  (16, 3)
            'action.arm_right_axangle_action':   (16, 3)
            'action.hand_right_pose_action':     (16, 6)
            """
            assert args.action_horizon > 0
            obs_origin = None
            for j in range(args.action_horizon):
                if j < 1:
                    # action chunk process
                    continue
                raw_action = {
                    "world_vector": np.array(raw_action_chunk['action.arm_right_position_action'][j]),
                    "rotation_delta": np.array(raw_action_chunk['action.arm_right_axangle_action'][j]),
                    "open_gripper": np.array(
                        raw_action_chunk['action.hand_right_pose_action'][j]
                    ),
                }
                # 轴角->四元数
                action = pred_to_realman(raw_action)

                keys_to_concatenate = ['world_vector', 'quaternion', 'gripper']
                arrays_to_concatenate = [action[key] for key in keys_to_concatenate]
                action = np.concatenate(arrays_to_concatenate)

                if args.simulate:
                    obs_origin = interact.single_step_pub_action(
                        action=action,
                        camera_name=args.obs_camera_name,
                        simulate=True,
                    )
                else:
                    # import ipdb; ipdb.set_trace()
                    obs_origin = interact.single_step_pub_action(
                        action=action,
                        camera_name=args.obs_camera_name
                    )

            # pub action end, get observations
            if args.simulate:
                image, obs = interact.get_simulate_data(camera_name=args.obs_camera_name)
            else:
                image, obs = interact.get_obs(obs_origin=obs_origin, camera_name=args.obs_camera_name)

            # 四元数转轴角
            data_point = obs_to_datapoint(image, obs, args.prompt)

    print("Done")
    exit()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)