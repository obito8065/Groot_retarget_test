#!/usr/bin/env python3
"""
FK检查脚本
功能：
1. 读取 robocasa_action_20260120_101716.txt 文件
2. 提取每行的 left_arm (7-DoF), right_arm (7-DoF), waist (3-DoF), left_finger (6-DoF), right_finger (6-DoF)
3. neck关节设置为0（3-DoF）
4. 使用 BodyRetargeter.process_frame_kinematics_axisangle 进行FK
5. 输出wrist pose (xyz + rotvec) 和 finger joints 到 FK_check.txt
6. 输出格式与 retargeted_actions_20260120_101716.txt 一致

用途：通过FK验证IK的准确性
"""

import numpy as np
from pathlib import Path
from gr00t.eval.gr1_pos_transform import BodyRetargeter

def build_44dof_vector(left_arm, right_arm, waist, left_finger, right_finger):
    """
    构建44-DoF的动作向量
    
    布局：
    - left_arm: 0-7 (7-DoF)
    - left_hand: 7-13 (6-DoF)
    - left_leg: 13-19 (6-DoF, 填充为0)
    - neck: 19-22 (3-DoF, 填充为0)
    - right_arm: 22-29 (7-DoF)
    - right_hand: 29-35 (6-DoF)
    - right_leg: 35-41 (6-DoF, 填充为0)
    - waist: 41-44 (3-DoF)
    
    参数:
        left_arm: (7,) array
        right_arm: (7,) array
        waist: (3,) array
        left_finger: (6,) array
        right_finger: (6,) array
    
    返回:
        (44,) array
    """
    vector = np.zeros(44, dtype=np.float64)
    
    # 填充数据
    vector[0:7] = left_arm      # left_arm
    vector[7:13] = left_finger  # left_hand
    # vector[13:19] 保持为0 (left_leg)
    # vector[19:22] 保持为0 (neck)
    vector[22:29] = right_arm   # right_arm
    vector[29:35] = right_finger # right_hand
    # vector[35:41] 保持为0 (right_leg)
    vector[41:44] = waist       # waist
    
    return vector


def main():
    # 文件路径
    input_file = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260126_171642.txt")
    output_file = Path("/vla/users/lijiayi/code/groot_retarget/output_video_record/FK_check.txt")
    urdf_path = Path("/vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf")
    
    print(f"读取输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    # 初始化 BodyRetargeter
    print("初始化 BodyRetargeter...")
    camera_intrinsics = {'fx': 502.8689, 'fy': 502.8689, 'cx': 640.0, 'cy': 400.0}
    retargeter = BodyRetargeter(urdf_path=urdf_path, camera_intrinsics=camera_intrinsics)
    print("初始化完成\n")
    
    # 读取输入文件
    data_lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            data_lines.append(line)
    
    print(f"读取到 {len(data_lines)} 行数据")
    
    # 准备输出文件
    with open(output_file, 'w') as f_out:
        # 写入文件头
        f_out.write("# FK验证输出（从robocasa_action通过FK得到的wrist pose和finger joints）\n")
        f_out.write("# 格式：chunk_id t L_wrist_x L_wrist_y L_wrist_z L_rotvec_x L_rotvec_y L_rotvec_z L_finger_q1 L_finger_q2 L_finger_q3 L_finger_q4 L_finger_q5 L_finger_q6 R_wrist_x R_wrist_y R_wrist_z R_rotvec_x R_rotvec_y R_rotvec_z R_finger_q1 R_finger_q2 R_finger_q3 R_finger_q4 R_finger_q5 R_finger_q6\n")
        f_out.write("# L_finger_joint_names_6: [index, middle, ring, pinky, thumb_yaw, thumb_pitch]\n")
        f_out.write("#\n")
        
        # 处理每一行数据
        for line_idx, line in enumerate(data_lines):
            # 解析数据行
            # 格式: chunk_id t L_arm(7) L_finger(6) R_arm(7) R_finger(6) waist(3)
            values = list(map(float, line.split()))
            
            if len(values) != 31:  # 2 + 7 + 6 + 7 + 6 + 3 = 31
                print(f"警告: 第 {line_idx+1} 行数据格式错误（期望31个值，实际{len(values)}个），跳过")
                continue
            
            # 提取各部分数据（根据robocasa_action.txt的格式）
            chunk_id = int(values[0])
            t = int(values[1])
            
            # L_arm_q1~q7, L_finger_q1~q6, R_arm_q1~q7, R_finger_q1~q6, waist_q1~q3
            left_arm = np.array(values[2:9], dtype=np.float64)      # 7-DoF
            left_finger = np.array(values[9:15], dtype=np.float64)  # 6-DoF
            right_arm = np.array(values[15:22], dtype=np.float64)   # 7-DoF
            right_finger = np.array(values[22:28], dtype=np.float64) # 6-DoF
            waist = np.array(values[28:31], dtype=np.float64)       # 3-DoF
            
            # 构建44-DoF向量（neck设为0）
            full_vector = build_44dof_vector(left_arm, right_arm, waist, left_finger, right_finger)
            
            # 执行FK
            (left_hand_pos, left_hand_axisangle), (right_hand_pos, right_hand_axisangle), _ = \
                retargeter.process_frame_kinematics_axisangle(full_vector)
            
            # 组装输出行
            # chunk_id t L_wrist(3) L_rotvec(3) L_finger(6) R_wrist(3) R_rotvec(3) R_finger(6)
            output_values = [chunk_id, t]
            output_values.extend(left_hand_pos.tolist())        # L_wrist_x, y, z
            output_values.extend(left_hand_axisangle.tolist())  # L_rotvec_x, y, z
            output_values.extend(left_finger.tolist())          # L_finger_q1~q6
            output_values.extend(right_hand_pos.tolist())       # R_wrist_x, y, z
            output_values.extend(right_hand_axisangle.tolist()) # R_rotvec_x, y, z
            output_values.extend(right_finger.tolist())         # R_finger_q1~q6
            
            # 格式化输出（保持与retargeted_actions.txt一致的格式）
            output_line = f"{chunk_id} {t}"
            for val in output_values[2:]:
                output_line += f" {val:.6f}"
            
            f_out.write(output_line + "\n")
            
            # 打印进度
            if (line_idx + 1) % 50 == 0:
                print(f"已处理 {line_idx + 1}/{len(data_lines)} 行")
    
    print(f"\n处理完成！")
    print(f"输出文件已保存到: {output_file}")
    print(f"\n说明：")
    print(f"- 输入文件包含 RoboCasa 的关节角度（arm + finger + waist）")
    print(f"- 通过FK计算得到相机坐标系下的wrist pose（位置+轴角）")
    print(f"- finger joints直接从输入复制（FK不改变手指关节）")
    print(f"- neck关节固定为0")
    print(f"- 可以将此文件与 retargeted_actions.txt 对比，验证IK的准确性")


if __name__ == "__main__":
    main()
