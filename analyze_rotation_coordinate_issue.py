#!/usr/bin/env python3
"""
分析Retarget后旋转方向错位的问题

问题分析：
===========
1. 模型预测输出（predicted_keypoints）中的数据都在**相机坐标系**下：
   - wrist_xyz: 相机坐标系下的手腕位置
   - tips_xyz: 相机坐标系下的指尖位置
   - wrist_rotvec: 相机坐标系下的手腕旋转向量

2. Retarget过程中：
   - 位置信息（wrist_xyz, tips_xyz）通过IK优化，会自动转换到机器人基座坐标系
   - 但是 wrist_rotvec 在warmup时直接使用，**没有进行坐标系转换**

3. 结果：
   - Retarget输出的wrist位置是正确的（在base_link坐标系）
   - 但wrist的旋转方向是错误的（仍然基于相机坐标系的orientation）

解决方案：
===========
需要在使用wrist_rotvec之前，将其从相机坐标系转换到机器人基座坐标系。

坐标系转换公式：
- R_base_wrist = R_base_camera @ R_camera_wrist
- rotvec_base = R.from_matrix(R_base_wrist).as_rotvec()

其中：
- R_camera_wrist: 从wrist_rotvec得到的旋转矩阵（相机坐标系）
- R_base_camera: 相机到基座的旋转矩阵（从相机外参得到）
- R_base_wrist: 转换后的手腕旋转矩阵（基座坐标系）
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def load_predicted_keypoints(filepath):
    """加载预测的关键点数据"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = list(map(float, line.split()))
            if len(values) == 44:  # frame_id + t + 21*2
                data.append(values)
    return np.array(data)


def load_retargeted_actions(filepath):
    """加载retarget后的动作数据"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            values = list(map(float, line.split()))
            if len(values) == 26:  # chunk_id + t + 12*2
                data.append(values)
    return np.array(data)


def analyze_rotation_difference():
    """分析旋转向量的差异"""
    
    # 加载数据
    pred_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260113_171107.txt"
    retarget_file = "/vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260113_171107.txt"
    
    pred_data = load_predicted_keypoints(pred_file)
    retarget_data = load_retargeted_actions(retarget_file)
    
    print("=" * 80)
    print("旋转向量坐标系问题分析")
    print("=" * 80)
    
    # 分析前几帧的数据
    n_frames = min(5, len(pred_data), len(retarget_data))
    
    for i in range(n_frames):
        frame_id = int(pred_data[i, 0])
        t = int(pred_data[i, 1])
        
        # 提取左手的wrist rotvec
        # predicted: [frame_id, t, left_21, right_21]
        # left_21: wrist_xyz(3) + 5tips_xyz(15) + wrist_rotvec(3)
        left_wrist_xyz_pred = pred_data[i, 2:5]
        left_wrist_rotvec_pred = pred_data[i, 20:23]  # 相机坐标系
        
        # retargeted: [chunk_id, t, left_12, right_12]
        # left_12: wrist_xyz(3) + rotvec(3) + finger_joints(6)
        left_wrist_xyz_retarget = retarget_data[i, 2:5]
        left_wrist_rotvec_retarget = retarget_data[i, 5:8]  # 基座坐标系（错误）
        
        print(f"\n帧 {frame_id}, 时间步 {t}:")
        print(f"  左手腕位置 (预测): {left_wrist_xyz_pred}")
        print(f"  左手腕位置 (retarget): {left_wrist_xyz_retarget}")
        print(f"  位置差异: {np.linalg.norm(left_wrist_xyz_pred - left_wrist_xyz_retarget):.6f} m")
        print()
        print(f"  左手腕rotvec (预测/相机系): {left_wrist_rotvec_pred}")
        print(f"  左手腕rotvec (retarget/基座系): {left_wrist_rotvec_retarget}")
        
        # 计算旋转角度差异
        R_pred = R.from_rotvec(left_wrist_rotvec_pred)
        R_retarget = R.from_rotvec(left_wrist_rotvec_retarget)
        R_diff = R_retarget * R_pred.inv()
        angle_diff = np.linalg.norm(R_diff.as_rotvec())
        
        print(f"  旋转角度差异: {np.degrees(angle_diff):.2f}°")
        print()
        
        # 将rotvec转换为欧拉角以便理解
        euler_pred = R_pred.as_euler('xyz', degrees=True)
        euler_retarget = R_retarget.as_euler('xyz', degrees=True)
        print(f"  预测欧拉角 (相机系): {euler_pred}")
        print(f"  Retarget欧拉角 (基座系): {euler_retarget}")
    
    print("\n" + "=" * 80)
    print("问题总结：")
    print("=" * 80)
    print("""
1. 预测的wrist_rotvec是在相机坐标系下的
2. Retarget过程直接使用了这个rotvec进行warmup，没有坐标系转换
3. 导致retarget后的方向是基于错误的坐标系

解决方案：
在fourier_hand_retarget_api.py中，warmup之前需要进行坐标系转换：

```python
# 需要添加：相机到基座的旋转矩阵
R_base_camera = ...  # 从相机外参获取

# 在warmup之前转换rotvec
R_camera_wrist = R.from_rotvec(wrist_rotvec)  # 相机坐标系
R_base_wrist = R_base_camera * R_camera_wrist  # 转到基座坐标系
wrist_rotvec_base = R_base_wrist.as_rotvec()  # 基座坐标系的rotvec

# 然后使用转换后的rotvec进行warmup
wrist_quat = R.from_rotvec(wrist_rotvec_base).as_quat()
```

注意：需要确定相机外参中的旋转矩阵表示的是哪个方向的转换
    """)


if __name__ == "__main__":
    analyze_rotation_difference()
