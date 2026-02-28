import torch

# axis sequences for Euler angles (保持为 Python 常量)
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# 使用 PyTorch 常量替代 TensorFlow 的 eps
_EPS4 = torch.tensor(1e-12, dtype=torch.float32)  # 调整精度根据实际情况


def axangle2mat_batch(axes, angles=None, is_normalized=False):
    """
    PyTorch 版本的轴角转旋转矩阵 (支持批量)

    Parameters
    ----------
    axes : torch.Tensor, shape=(N, 3)
    angles : torch.Tensor, shape=(N,) 或 标量
    is_normalized : bool

    Returns
    -------
    mats : torch.Tensor, shape=(N, 3, 3)
    """
    # axes = torch.as_tensor(axes, dtype=torch.float32)
    # if angles is not None:
    #     angles = torch.as_tensor(angles, dtype=torch.float32)

    # 处理形状
    if len(axes.shape) == 1:
        axes = axes.unsqueeze(0)
    N = axes.shape[0]

    # 归一化
    if not is_normalized:
        norms = torch.norm(axes, dim=1, keepdim=True)
        axes = axes / norms
        if angles is None:
            angles = norms.squeeze(-1)

    # 处理角度输入
    if angles is None:
        raise ValueError("Angles must be provided if is_normalized=True")
    if angles.dim() == 0:
        angles = angles.expand(N)

    # 三角函数计算
    c = torch.cos(angles)
    s = torch.sin(angles)
    C = 1.0 - c

    # 分解轴分量
    x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]

    # 中间计算项
    xs = x * s
    ys = y * s
    zs = z * s
    xC = x * C
    yC = y * C
    zC = z * C

    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    # 构造旋转矩阵 (批量)
    row0 = torch.stack([x * xC + c, xyC - zs, zxC + ys], dim=1)
    row1 = torch.stack([xyC + zs, y * yC + c, yzC - xs], dim=1)
    row2 = torch.stack([zxC - ys, yzC + xs, z * zC + c], dim=1)
    mats = torch.stack([row0, row1, row2], dim=1)

    return mats


def mat2euler_batch(mats, axes='sxyz'):
    """PyTorch 版本的旋转矩阵转欧拉角 (批量)"""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    # mats = torch.as_tensor(mats, dtype=torch.float32)
    N = mats.shape[0]
    M = mats[:, :3, :3]

    # 初始化输出张量
    ax = torch.zeros(N, dtype=mats.dtype, device=mats.device)
    ay = torch.zeros(N, dtype=mats.dtype, device=mats.device)
    az = torch.zeros(N, dtype=mats.dtype, device=mats.device)

    if repetition:
        # 计算 sy = sqrt(M[i,j]^2 + M[i,k]^2)
        term1 = torch.square(M[:, i, j])
        term2 = torch.square(M[:, i, k])
        sy = torch.sqrt(term1 + term2)
        mask = sy > _EPS4

        # 处理 sy > EPS4
        ax_masked = torch.atan2(M[:, i, j], M[:, i, k])
        ay_masked = torch.atan2(sy, M[:, i, i])
        az_masked = torch.atan2(M[:, j, i], -M[:, k, i])

        # 处理 sy <= EPS4
        ax_unmasked = torch.atan2(-M[:, j, k], M[:, j, j])
        ay_unmasked = torch.atan2(sy, M[:, i, i])
        az_unmasked = torch.zeros_like(ax_unmasked)

        # 合并结果
        ax = torch.where(mask, ax_masked, ax_unmasked)
        ay = torch.where(mask, ay_masked, ay_unmasked)
        az = torch.where(mask, az_masked, az_unmasked)
    else:
        # 计算 cy = sqrt(M[i,i]^2 + M[j,i]^2)
        term1 = torch.square(M[:, i, i])
        term2 = torch.square(M[:, j, i])
        cy = torch.sqrt(term1 + term2)
        mask = cy > _EPS4

        # 处理 cy > EPS4
        ax_masked = torch.atan2(M[:, k, j], M[:, k, k])
        ay_masked = torch.atan2(-M[:, k, i], cy)
        az_masked = torch.atan2(M[:, j, i], M[:, i, i])

        # 处理 cy <= EPS4
        ax_unmasked = torch.atan2(-M[:, j, k], M[:, j, j])
        ay_unmasked = torch.atan2(-M[:, k, i], cy)
        az_unmasked = torch.zeros_like(ax_unmasked)

        ax = torch.where(mask, ax_masked, ax_unmasked)
        ay = torch.where(mask, ay_masked, ay_unmasked)
        az = torch.where(mask, az_masked, az_unmasked)

    # 处理奇偶性和坐标系
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax

    return torch.stack([ax, ay, az], dim=1)


def axangle2euler_batch(vectors, thetas=None, axes='sxyz'):
    # 转换为 PyTorch 张量
    # vectors = torch.as_tensor(vectors)
    # if thetas is not None:
    #     thetas = torch.as_tensor(thetas, dtype=torch.float32)

    # 轴角 -> 旋转矩阵
    mats = axangle2mat_batch(vectors, thetas)

    # 旋转矩阵 -> 欧拉角
    return mat2euler_batch(mats, axes)