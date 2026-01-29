#!/usr/bin/env python3
"""
在训练集上评估模型，保存输入和输出的关键点数据

使用方法:


python test_evaluation_on_traindataset.py \
    --dataset-path /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_300_keypoints_v3 \
    --model-path  /vla/users/lijiayi/code/groot_retarget/output_ckpt/n1.5_nopretrain_finetuneALL_on_robocasa_task1_retarget_v3_bs384/checkpoint-28050 \
    --data-config robocasa_retarget \
    --embodiment-tag robocasa \
    --traj-id 7 \
    --steps 360 \
    --action-horizon 16


"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
from typing import Dict, Any

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP

warnings.simplefilter("ignore", category=FutureWarning)


def test_evaluation_on_traindataset(
    dataset_path: str,
    model_path: str,
    data_config_name: str,
    embodiment_tag: str,
    traj_id: int = 0,
    steps: int = 150,
    action_horizon: int = 16,
    device: str = "cuda",
):
    """
    在训练集上评估模型，保存输入和输出的关键点数据
    
    Args:
        dataset_path: 数据集路径
        model_path: 模型路径
        data_config_name: 数据配置名称
        embodiment_tag: 机器人类型标签
        traj_id: 轨迹ID
        steps: 评估步数
        action_horizon: 动作预测horizon
        device: 设备（cuda/cpu）
    """
    print("=" * 80)
    print("训练集评估脚本")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print(f"模型路径: {model_path}")
    print(f"数据配置: {data_config_name}")
    print(f"机器人类型: {embodiment_tag}")
    print(f"轨迹ID: {traj_id}")
    print(f"评估步数: {steps}")
    print(f"动作Horizon: {action_horizon}")
    print("=" * 80)
    
    # 1. 加载数据配置
    data_config = DATA_CONFIG_MAP[data_config_name]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    # 2. 加载数据集
    print("\n[1/4] 加载数据集...")
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend="decord",
        video_backend_kwargs=None,
        transforms=None,  # 不使用transforms，policy会处理
        embodiment_tag=EmbodimentTag(embodiment_tag),
    )
    print(f"数据集加载完成，总轨迹数: {len(dataset.trajectory_lengths)}")
    print(f"轨迹 {traj_id} 的长度: {dataset.trajectory_lengths[traj_id]}")
    
    # 3. 加载模型（禁用eepose和retarget，只保留关键点）
    print("\n[2/4] 加载模型...")
    policy = Gr00tPolicy(
        model_path=model_path,
        embodiment_tag=EmbodimentTag(embodiment_tag),
        modality_config=modality_config,
        modality_transform=modality_transform,
        device=device,
        use_eepose=False,  # 禁用eepose处理，直接使用关键点
        use_fourier_hand_retarget=False,  # 禁用retarget处理
    )
    print("模型加载完成")
    
    # 4. 创建输出日志文件
    print("\n[3/4] 创建输出日志文件...")
    log_dir = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_eval_keypoints_traj{traj_id}_{timestamp}.txt"
    
    # 写入文件头
    with open(log_file, 'w') as f:
        f.write("# 训练集评估：输入和输出的关键点数据\n")
        f.write("# 格式：global_step chunk_id t_in_chunk ")
        f.write("L_input_wrist_x L_input_wrist_y L_input_wrist_z ")
        f.write("L_input_thumb_tip_x L_input_thumb_tip_y L_input_thumb_tip_z ")
        f.write("L_input_index_tip_x L_input_index_tip_y L_input_index_tip_z ")
        f.write("L_input_middle_tip_x L_input_middle_tip_y L_input_middle_tip_z ")
        f.write("L_input_ring_tip_x L_input_ring_tip_y L_input_ring_tip_z ")
        f.write("L_input_pinky_tip_x L_input_pinky_tip_y L_input_pinky_tip_z ")
        f.write("R_input_wrist_x R_input_wrist_y R_input_z ")
        f.write("R_input_thumb_tip_x R_input_thumb_tip_y R_input_thumb_tip_z ")
        f.write("R_input_index_tip_x R_input_index_tip_y R_input_index_tip_z ")
        f.write("R_input_middle_tip_x R_input_middle_tip_y R_input_middle_tip_z ")
        f.write("R_input_ring_tip_x R_input_ring_tip_y R_input_ring_tip_z ")
        f.write("R_input_pinky_tip_x R_input_pinky_tip_y R_input_pinky_tip_z ")
        f.write("L_output_wrist_x L_output_wrist_y L_output_wrist_z ")
        f.write("L_output_thumb_tip_x L_output_thumb_tip_y L_output_thumb_tip_z ")
        f.write("L_output_index_tip_x L_output_index_tip_y L_output_index_tip_z ")
        f.write("L_output_middle_tip_x L_output_middle_tip_y L_output_middle_tip_z ")
        f.write("L_output_ring_tip_x L_output_ring_tip_y L_output_ring_tip_z ")
        f.write("L_output_pinky_tip_x L_output_pinky_tip_y L_output_pinky_tip_z ")
        f.write("R_output_wrist_x R_output_wrist_y R_output_wrist_z ")
        f.write("R_output_thumb_tip_x R_output_thumb_tip_y R_output_thumb_tip_z ")
        f.write("R_output_index_tip_x R_output_index_tip_y R_output_index_tip_z ")
        f.write("R_output_middle_tip_x R_output_middle_tip_y R_output_middle_tip_z ")
        f.write("R_output_ring_tip_x R_output_ring_tip_y R_output_ring_tip_z ")
        f.write("R_output_pinky_tip_x R_output_pinky_tip_y R_output_pinky_tip_z\n")
        f.write("# 输入关键点：来自训练集的state.left_key_points和state.right_key_points\n")
        f.write("# 输出关键点：模型预测的action.left_key_points和action.right_key_points\n")
        f.write("# 每个关键点包含3个坐标值(x, y, z)\n")
        f.write("# 左手6个关键点：wrist + thumb_tip + index_tip + middle_tip + ring_tip + pinky_tip = 18维\n")
        f.write("# 右手6个关键点：wrist + thumb_tip + index_tip + middle_tip + ring_tip + pinky_tip = 18维\n")
        f.write("# global_step: 全局时间步（从0开始）\n")
        f.write("# chunk_id: chunk编号（每次inference产生一个chunk）\n")
        f.write("# t_in_chunk: chunk内的时间步（0到action_horizon-1）\n")
        f.write("#\n")
    
    print(f"日志文件: {log_file}")
    
    # 5. 遍历轨迹并评估
    print("\n[4/4] 开始评估...")
    global_step = 0
    chunk_id = 0
    
    # 检查关键点维度（可能是18维或21维）
    # 18维: [wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz]
    # 21维: [wrist_xyz, thumb_tip_xyz, index_tip_xyz, middle_tip_xyz, ring_tip_xyz, pinky_tip_xyz, wrist_rotvec]
    
    for step_count in range(steps):
        # 检查是否超出轨迹长度
        if step_count >= dataset.trajectory_lengths[traj_id]:
            print(f"\n警告: 步数 {step_count} 超出轨迹长度 {dataset.trajectory_lengths[traj_id]}，停止评估")
            break
        
        # 获取当前步的观测数据
        data_point = dataset.get_step_data(traj_id, step_count)
        
        # 检查是否有关键点数据
        if "state.left_key_points" not in data_point or "state.right_key_points" not in data_point:
            print(f"\n警告: 步数 {step_count} 没有关键点数据，跳过")
            continue
        
        # 在每个action_horizon的倍数处进行推理
        if step_count % action_horizon == 0:
            print(f"推理步数: {step_count} (chunk {chunk_id})")
            
            # 调用policy获取动作
            try:
                action_chunk = policy.get_action(data_point)
            except Exception as e:
                print(f"错误: 在步数 {step_count} 处推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # 检查输出是否包含关键点
            if "action.left_key_points" not in action_chunk or "action.right_key_points" not in action_chunk:
                print(f"警告: chunk {chunk_id} 的输出不包含关键点，跳过")
                chunk_id += 1
                continue
            
            # 获取输入关键点（来自训练集）
            input_left_kp = data_point["state.left_key_points"]  # 形状可能是 (B, T, 18) 或 (B, T, 21)
            input_right_kp = data_point["state.right_key_points"]
            
            # 获取输出关键点（来自模型预测）
            output_left_kp = action_chunk["action.left_key_points"]  # 形状 (B, horizon, 18) 或 (B, horizon, 21)
            output_right_kp = action_chunk["action.right_key_points"]
            
            # 确保是numpy数组
            if not isinstance(input_left_kp, np.ndarray):
                input_left_kp = np.array(input_left_kp)
            if not isinstance(input_right_kp, np.ndarray):
                input_right_kp = np.array(input_right_kp)
            if not isinstance(output_left_kp, np.ndarray):
                output_left_kp = np.array(output_left_kp)
            if not isinstance(output_right_kp, np.ndarray):
                output_right_kp = np.array(output_right_kp)
            
            # 处理维度：确保有batch维度
            # 输入可能是 (B, T, D) 或 (T, D)
            if input_left_kp.ndim == 2:
                # 如果是2维，添加batch维度: (T, D) -> (1, T, D)
                input_left_kp = input_left_kp[np.newaxis, :, :]
                input_right_kp = input_right_kp[np.newaxis, :, :]
            
            # 输出可能是 (B, horizon, D) 或 (horizon, D)
            if output_left_kp.ndim == 2:
                # 如果是2维，添加batch维度: (horizon, D) -> (1, horizon, D)
                output_left_kp = output_left_kp[np.newaxis, :, :]
                output_right_kp = output_right_kp[np.newaxis, :, :]
            
            # 现在应该都是3维了
            B_in, T_in, D_in = input_left_kp.shape
            B_out, horizon, D_out = output_left_kp.shape
            
            # 只取前18维（关键点坐标），忽略可能的rotvec
            D_kp = min(18, D_in, D_out)
            
            # 保存当前chunk的所有时间步
            with open(log_file, 'a') as f:
                for t_in_chunk in range(horizon):
                    # 获取输入关键点（当前chunk的输入，即step_count步的输入）
                    # 输入是 (B, T, D)，我们取最后一个时间步 T-1（这是当前观测）
                    input_left = input_left_kp[0, -1, :D_kp]  # (D_kp,)
                    input_right = input_right_kp[0, -1, :D_kp]  # (D_kp,)
                    
                    # 获取输出关键点（chunk内的时间步t_in_chunk）
                    output_left = output_left_kp[0, t_in_chunk, :D_kp]  # (D_kp,)
                    output_right = output_right_kp[0, t_in_chunk, :D_kp]  # (D_kp,)
                    
                    # 计算全局时间步：当前chunk的起始步 + chunk内的时间步
                    current_global_step = step_count + t_in_chunk
                    
                    # 写入一行数据
                    line = f"{current_global_step} {chunk_id} {t_in_chunk}"
                    
                    # 输入关键点（左手18维 + 右手18维 = 36维）
                    for val in input_left:
                        line += f" {val:.6f}"
                    for val in input_right:
                        line += f" {val:.6f}"
                    
                    # 输出关键点（左手18维 + 右手18维 = 36维）
                    for val in output_left:
                        line += f" {val:.6f}"
                    for val in output_right:
                        line += f" {val:.6f}"
                    
                    line += "\n"
                    f.write(line)
            
            chunk_id += 1
    
    print(f"\n评估完成！")
    print(f"总chunk数: {chunk_id}")
    print(f"日志文件: {log_file}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="在训练集上评估模型，保存输入和输出的关键点数据"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="数据集路径"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="fourier_gr1_arms_only",
        help="数据配置名称"
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="gr1",
        help="机器人类型标签"
    )
    parser.add_argument(
        "--traj-id",
        type=int,
        default=0,
        help="轨迹ID"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="评估步数"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help="动作预测horizon"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备（cuda/cpu）"
    )
    
    args = parser.parse_args()
    
    test_evaluation_on_traindataset(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        data_config_name=args.data_config,
        embodiment_tag=args.embodiment_tag,
        traj_id=args.traj_id,
        steps=args.steps,
        action_horizon=args.action_horizon,
        device=args.device,
    )


if __name__ == "__main__":
    main()

