#!/usr/bin/env python3
"""
检查数据集中手腕轴角的连续性，检测是否有正负跳变

用法：
python check_dataset_rot.py \
    --dataset-root /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_300_keypoints_v4_1

python check_dataset_rot.py \
    --dataset-root /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v4_1 

    
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from typing import List, Tuple, Dict


def detect_axisangle_jump(
    rotvec_seq: np.ndarray, 
    threshold: float = 1.0
) -> List[Tuple[int, float, float, bool]]:
    """
    检测轴角序列中的跳变
    
    参数:
        rotvec_seq: (T, 3) 轴角序列
        threshold: 跳变阈值（弧度）
    
    返回:
        List of (frame_idx, direct_diff, actual_angle, is_equiv_jump)
        - frame_idx: 跳变位置的索引（相对于序列开始）
        - direct_diff: 直接差值（L2 norm）
        - actual_angle: 实际旋转角度
        - is_equiv_jump: 是否为等价表示跳变
    """
    if len(rotvec_seq) < 2:
        return []
    
    jumps = []
    
    for i in range(1, len(rotvec_seq)):
        # 将轴角转换为旋转矩阵
        R_prev = R.from_rotvec(rotvec_seq[i-1]).as_matrix()
        R_curr = R.from_rotvec(rotvec_seq[i]).as_matrix()
        
        # 计算相对旋转
        R_diff = R_prev.T @ R_curr
        rotvec_diff = R.from_matrix(R_diff).as_rotvec()
        actual_angle = np.linalg.norm(rotvec_diff)
        
        # 直接计算轴角的差值
        direct_diff = np.linalg.norm(rotvec_seq[i] - rotvec_seq[i-1])
        
        # 检查是否是等价表示跳变
        # 轴角 r 和 -r + 2πk 表示同一个旋转
        # 如果直接差值很大，但旋转矩阵相同（或非常接近），说明是等价表示跳变
        is_equiv_jump = False
        if direct_diff > threshold:
            # 检查旋转矩阵是否相同
            if np.allclose(R_prev, R_curr, atol=1e-3):
                is_equiv_jump = True
            # 或者实际旋转角度远小于直接差值
            elif actual_angle < direct_diff * 0.7:  # 实际旋转小于直接差值的70%
                is_equiv_jump = True
        
        if is_equiv_jump or direct_diff > threshold * 2:
            jumps.append((i, direct_diff, actual_angle, is_equiv_jump))
    
    return jumps


def check_episode_rotvec(
    df: pd.DataFrame,
    episode_path: Path
) -> Dict[str, List]:
    """
    检查单个episode的轴角跳变
    
    返回:
        {
            'state_left_jumps': [...],
            'state_right_jumps': [...],
            'action_left_jumps': [...],
            'action_right_jumps': [...],
        }
    """
    results = {
        'state_left_jumps': [],
        'state_right_jumps': [],
        'action_left_jumps': [],
        'action_right_jumps': [],
    }
    
    # 提取state和action数据
    states = []
    actions = []
    
    for _, row in df.iterrows():
        state = row.get("observation.state", None)
        action = row.get("action", None)
        
        if state is not None:
            if isinstance(state, (list, np.ndarray)):
                state = np.array(state).flatten()
            else:
                continue
            if len(state) == 45:
                states.append(state)
        
        if action is not None:
            if isinstance(action, (list, np.ndarray)):
                action = np.array(action).flatten()
            else:
                continue
            if len(action) == 45:
                actions.append(action)
    
    if len(states) == 0 and len(actions) == 0:
        return results
    
    # 检查state的轴角
    if len(states) > 1:
        states_array = np.array(states)
        
        # 提取左右手腕轴角
        # left_key_points: [0:21], wrist_rotvec在[18:21]
        # right_key_points: [21:42], wrist_rotvec在[39:42]
        left_state_rotvec = states_array[:, 18:21]  # (T, 3)
        right_state_rotvec = states_array[:, 39:42]  # (T, 3)
        
        # 检测跳变
        left_jumps = detect_axisangle_jump(left_state_rotvec)
        right_jumps = detect_axisangle_jump(right_state_rotvec)
        
        if left_jumps:
            results['state_left_jumps'] = [
                (episode_path, frame_idx, direct_diff, actual_angle, is_equiv)
                for frame_idx, direct_diff, actual_angle, is_equiv in left_jumps
            ]
        
        if right_jumps:
            results['state_right_jumps'] = [
                (episode_path, frame_idx, direct_diff, actual_angle, is_equiv)
                for frame_idx, direct_diff, actual_angle, is_equiv in right_jumps
            ]
    
    # 检查action的轴角
    if len(actions) > 1:
        actions_array = np.array(actions)
        
        # 提取左右手腕轴角
        left_action_rotvec = actions_array[:, 18:21]  # (T, 3)
        right_action_rotvec = actions_array[:, 39:42]  # (T, 3)
        
        # 检测跳变
        left_jumps = detect_axisangle_jump(left_action_rotvec)
        right_jumps = detect_axisangle_jump(right_action_rotvec)
        
        if left_jumps:
            results['action_left_jumps'] = [
                (episode_path, frame_idx, direct_diff, actual_angle, is_equiv)
                for frame_idx, direct_diff, actual_angle, is_equiv in left_jumps
            ]
        
        if right_jumps:
            results['action_right_jumps'] = [
                (episode_path, frame_idx, direct_diff, actual_angle, is_equiv)
                for frame_idx, direct_diff, actual_angle, is_equiv in right_jumps
            ]
    
    return results


def check_dataset(dataset_root: Path, threshold: float = 1.0):
    """
    检查整个数据集的轴角跳变
    """
    data_root = dataset_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    
    # 收集所有parquet文件
    parquet_files = []
    chunk_dirs = sorted(p for p in data_root.iterdir() if p.is_dir())
    for chunk in chunk_dirs:
        parquet_files.extend(sorted(chunk.glob("*.parquet")))
    
    if not parquet_files:
        print(f"未找到任何parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    print(f"开始检查轴角跳变（阈值: {threshold} rad）...")
    print("=" * 80)
    
    # 统计所有跳变
    all_state_left_jumps = []
    all_state_right_jumps = []
    all_action_left_jumps = []
    all_action_right_jumps = []
    
    # 处理每个episode
    for parquet_path in tqdm(parquet_files, desc="检查episodes"):
        try:
            df = pd.read_parquet(parquet_path)
            results = check_episode_rotvec(df, parquet_path)
            
            all_state_left_jumps.extend(results['state_left_jumps'])
            all_state_right_jumps.extend(results['state_right_jumps'])
            all_action_left_jumps.extend(results['action_left_jumps'])
            all_action_right_jumps.extend(results['action_right_jumps'])
        except Exception as e:
            print(f"\n警告: 处理 {parquet_path} 时出错: {e}")
            continue
    
    # 打印结果
    print("\n" + "=" * 80)
    print("检查结果汇总")
    print("=" * 80)
    
    total_jumps = (
        len(all_state_left_jumps) + len(all_state_right_jumps) +
        len(all_action_left_jumps) + len(all_action_right_jumps)
    )
    
    print(f"\n总共发现 {total_jumps} 个轴角跳变:")
    print(f"  - observation.state 左手: {len(all_state_left_jumps)} 个")
    print(f"  - observation.state 右手: {len(all_state_right_jumps)} 个")
    print(f"  - action 左手: {len(all_action_left_jumps)} 个")
    print(f"  - action 右手: {len(all_action_right_jumps)} 个")
    
    # 打印详细信息
    if total_jumps > 0:
        print("\n" + "=" * 80)
        print("详细信息:")
        print("=" * 80)
        
        if all_state_left_jumps:
            print("\n[observation.state] 左手腕轴角跳变:")
            for episode_path, frame_idx, direct_diff, actual_angle, is_equiv in all_state_left_jumps[:20]:  # 只显示前20个
                jump_type = "等价表示跳变" if is_equiv else "大角度跳变"
                print(f"  {episode_path.name} Frame {frame_idx}: "
                      f"直接差值={direct_diff:.4f} rad, "
                      f"实际角度={actual_angle:.4f} rad ({np.degrees(actual_angle):.2f}°), "
                      f"类型={jump_type}")
            if len(all_state_left_jumps) > 20:
                print(f"  ... 还有 {len(all_state_left_jumps) - 20} 个跳变未显示")
        
        if all_state_right_jumps:
            print("\n[observation.state] 右手腕轴角跳变:")
            for episode_path, frame_idx, direct_diff, actual_angle, is_equiv in all_state_right_jumps[:20]:
                jump_type = "等价表示跳变" if is_equiv else "大角度跳变"
                print(f"  {episode_path.name} Frame {frame_idx}: "
                      f"直接差值={direct_diff:.4f} rad, "
                      f"实际角度={actual_angle:.4f} rad ({np.degrees(actual_angle):.2f}°), "
                      f"类型={jump_type}")
            if len(all_state_right_jumps) > 20:
                print(f"  ... 还有 {len(all_state_right_jumps) - 20} 个跳变未显示")
        
        if all_action_left_jumps:
            print("\n[action] 左手腕轴角跳变:")
            for episode_path, frame_idx, direct_diff, actual_angle, is_equiv in all_action_left_jumps[:20]:
                jump_type = "等价表示跳变" if is_equiv else "大角度跳变"
                print(f"  {episode_path.name} Frame {frame_idx}: "
                      f"直接差值={direct_diff:.4f} rad, "
                      f"实际角度={actual_angle:.4f} rad ({np.degrees(actual_angle):.2f}°), "
                      f"类型={jump_type}")
            if len(all_action_left_jumps) > 20:
                print(f"  ... 还有 {len(all_action_left_jumps) - 20} 个跳变未显示")
        
        if all_action_right_jumps:
            print("\n[action] 右手腕轴角跳变:")
            for episode_path, frame_idx, direct_diff, actual_angle, is_equiv in all_action_right_jumps[:20]:
                jump_type = "等价表示跳变" if is_equiv else "大角度跳变"
                print(f"  {episode_path.name} Frame {frame_idx}: "
                      f"直接差值={direct_diff:.4f} rad, "
                      f"实际角度={actual_angle:.4f} rad ({np.degrees(actual_angle):.2f}°), "
                      f"类型={jump_type}")
            if len(all_action_right_jumps) > 20:
                print(f"  ... 还有 {len(all_action_right_jumps) - 20} 个跳变未显示")
    else:
        print("\n✅ 未发现轴角跳变！数据集中的轴角是连续的。")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="检查数据集中手腕轴角的连续性，检测是否有正负跳变"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="数据集根目录（包含data子目录）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="跳变检测阈值（弧度），默认1.0"
    )
    
    args = parser.parse_args()
    
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")
    
    check_dataset(dataset_root, threshold=args.threshold)


if __name__ == "__main__":
    main()
