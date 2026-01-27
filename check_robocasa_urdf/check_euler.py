import numpy as np
from scipy.spatial.transform import Rotation

# Retarget输出的内旋XYZ欧拉角
intrinsic_xyz = np.array([-2.221956, 0.232139, 1.606730])

# 转换为旋转矩阵
R = Rotation.from_euler('xyz', intrinsic_xyz, degrees=False)

# 转换为URDF的外旋XYZ（rpy）
rpy = R.as_euler('XYZ', degrees=False)  # 大写表示外旋

print(f"URDF rpy: {rpy[0]} {rpy[1]} {rpy[2]}")