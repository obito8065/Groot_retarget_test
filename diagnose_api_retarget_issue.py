#!/usr/bin/env python3
"""
诊断fourier_hand_retarget_api.py的retarget问题

对比:
1. 从predicted_keypoints.txt读取的wrist_rotvec
2. 从正常retarget脚本重新retarget的结果
3. 分析坐标系转换问题
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def load_predicted_keypoints(txt_path):
    """
    加载predicted_keypoints.txt
    
    格式: frame_id t L_wrist_xyz(3) L_5tips_xyz(15) L_wrist_rotvec(3) 
                      R_wrist_xyz(3) R_5tips_xyz(15) R_wrist_rotvec(3)
    """
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = list(map(float, line.split()))
            if len(values) == 44:  # frame_id + t + 21*2
                data.append(values)
    
    data = np.array(data)
    
    # 提取wrist数据
    result = {
        'left': {
            'wrist_xyz': data[:, 2:5],      # L_wrist_xyz
            'wrist_rotvec': data[:, 20:23],  # L_wrist_rotvec
        },
        'right': {
            'wrist_xyz': data[:, 23:26],     # R_wrist_xyz
            'wrist_rotvec': data[:, 41:44],  # R_wrist_rotvec
        }
    }
    
    return result, len(data)

def load_retargeted_actions(txt_path):
    """
    加载retargeted_actions.txt
    
    格式: chunk_id t L_wrist_xyz(3) L_rotvec(3) L_finger(6) 
                      R_wrist_xyz(3) R_rotvec(3) R_finger(6)
    """
    data = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = list(map(float, line.split()))
            if len(values) == 26:
                data.append(values)
    
    data = np.array(data)
    
    result = {
        'left': {
            'wrist_xyz': data[:, 2:5],
            'wrist_rotvec': data[:, 5:8],
            'finger': data[:, 8:14],
        },
        'right': {
            'wrist_xyz': data[:, 14:17],
            'wrist_rotvec': data[:, 17:20],
            'finger': data[:, 20:26],
        }
    }
    
    return result, len(data)

def compare_rotations(rot1, rot2, name1="Rotation 1", name2="Rotation 2"):
    """对比两个旋转的差异"""
    # 都转换为旋转矩阵
    if rot1.shape == (3,):
        R1 = R.from_rotvec(rot1).as_matrix()
    else:
        R1 = rot1
    
    if rot2.shape == (3,):
        R2 = R.from_rotvec(rot2).as_matrix()
    else:
        R2 = rot2
    
    # 计算相对旋转
    R_diff = R1.T @ R2
    angle_diff = np.linalg.norm(R.from_matrix(R_diff).as_rotvec())
    
    print(f"\n{name1} vs {name2}:")
    print(f"  旋转差异: {np.degrees(angle_diff):.2f}°")
    print(f"  {name1} rotvec: {R.from_matrix(R1).as_rotvec()}")
    print(f"  {name2} rotvec: {R.from_matrix(R2).as_rotvec()}")
    
    return np.degrees(angle_diff)

def main():
    # 路径
    predicted_path = "/vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260114_150533.txt"
    retargeted_path = "/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260114_150534.txt"
    
    print("="*80)
    print("诊断 Fourier Hand Retarget API 问题")
    print("="*80)
    
    # 加载数据
    print("\n1. 加载数据...")
    predicted_data, num_predicted = load_predicted_keypoints(predicted_path)
    retargeted_data, num_retargeted = load_retargeted_actions(retargeted_path)
    
    print(f"  Predicted frames: {num_predicted}")
    print(f"  Retargeted frames: {num_retargeted}")
    
    # 对比前几帧
    print("\n2. 对比前5帧的wrist rotation...")
    
    for side in ['left', 'right']:
        print(f"\n{'='*80}")
        print(f"{side.upper()} Hand")
        print(f"{'='*80}")
        
        for i in range(min(5, num_predicted, num_retargeted)):
            print(f"\n--- Frame {i} ---")
            
            # 从predicted_keypoints读取的rotvec（模型预测）
            pred_rotvec = predicted_data[side]['wrist_rotvec'][i]
            
            # API retarget后输出的rotvec
            retarget_rotvec = retargeted_data[side]['wrist_rotvec'][i]
            
            # 对比
            print(f"Predicted rotvec (相机系):  {pred_rotvec}")
            print(f"Retargeted rotvec (输出):   {retarget_rotvec}")
            
            # 计算差异
            angle_diff = compare_rotations(
                pred_rotvec, 
                retarget_rotvec,
                "Predicted (input)",
                "Retargeted (output)"
            )
            
            if angle_diff > 10:
                print(f"  ⚠️ 差异较大！可能存在坐标系转换或retarget问题")
            elif angle_diff > 1:
                print(f"  ⚠️ 有一定差异，这可能是正常的retarget调整")
            else:
                print(f"  ✓ 差异很小，基本一致")
    
    # 分析wrist position
    print("\n" + "="*80)
    print("3. 对比wrist位置...")
    print("="*80)
    
    for side in ['left', 'right']:
        print(f"\n{side.upper()} Hand:")
        for i in range(min(3, num_predicted, num_retargeted)):
            pred_xyz = predicted_data[side]['wrist_xyz'][i]
            retarget_xyz = retargeted_data[side]['wrist_xyz'][i]
            
            diff = np.linalg.norm(pred_xyz - retarget_xyz)
            print(f"  Frame {i}: diff = {diff:.4f}m")
            print(f"    Predicted:  {pred_xyz}")
            print(f"    Retargeted: {retarget_xyz}")
    
    print("\n" + "="*80)
    print("诊断完成")
    print("="*80)
    
    print("\n关键发现:")
    print("1. 如果wrist rotation差异很大（>10°），说明存在坐标系转换问题")
    print("2. predicted_keypoints.txt中的rotvec是相机坐标系")
    print("3. API应该在warmup前进行坐标系转换")
    print("\n建议修复:")
    print("在fourier_hand_retarget_api.py的retarget_from_45d函数中，")
    print("在使用wrist_rotvec之前添加从相机系到机器人基座系的坐标转换")

if __name__ == "__main__":
    main()
