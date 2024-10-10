import torch

import numpy as np
from scipy.spatial.transform import Rotation as R

def build_rotation_np(r):
    norm = np.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = np.zeros((q.shape[0], 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    return R


def build_rotation(q):
    """
    Convert a quaternion into a rotation matrix.
    q: tensor of shape [batch_size, 4] representing [r, x, y, z]
    """
    # Normalizing the quaternion to ensure it represents a valid rotation.
    q = q / q.norm(dim=1, keepdim=True)

    r, x, y, z = q.unbind(dim=1)

    # Precompute repeated terms
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    rx = r * x
    ry = r * y
    rz = r * z

    rot = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - rz), 2 * (xz + ry),
        2 * (xy + rz), 1 - 2 * (xx + zz), 2 * (yz - rx),
        2 * (xz - ry), 2 * (yz + rx), 1 - 2 * (xx + yy)
    ], dim=1).view(-1, 3, 3)
    
    return rot


def extract_rotation_scipy(rotation_matrix):
    batch_size, _, _ = rotation_matrix.shape
    rotation_matrices_np = rotation_matrix.reshape(batch_size, 3, 3)  # 转换为 NumPy 数组
    quaternions = R.from_matrix(rotation_matrices_np).as_quat()  # 从旋转矩阵中提取四元数
    
    # 注意，Scipy 返回的四元数格式为 (x, y, z, w)，可能需要重新排序为 (w, x, y, z)
    quaternions = quaternions[:, [3, 0, 1, 2]]
    
    return quaternions

def extract_rotation_torch(rotation_matrix):
    """
    Extracts quaternion from rotation matrix using PyTorch operations.
    Input:
        rotation_matrix: Tensor of shape (batch_size, 3, 3)
    Output:
        quaternions: Tensor of shape (batch_size, 4)
    """
    batch_size = rotation_matrix.shape[0]
    # Trace of the rotation matrix
    tr = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]

    # Initialize quaternion components to zeros
    qw = torch.zeros(batch_size, dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    qx = torch.zeros_like(qw)
    qy = torch.zeros_like(qw)
    qz = torch.zeros_like(qw)

    # Conditions
    cond1 = tr > 0
    cond2 = (~cond1) & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 1, 1]) & (rotation_matrix[:, 0, 0] > rotation_matrix[:, 2, 2])
    cond3 = (~cond1) & (~cond2) & (rotation_matrix[:, 1, 1] > rotation_matrix[:, 2, 2])
    
    # Case 1: If trace is positive
    t0 = torch.sqrt(tr[cond1] + 1.0) * 2
    qw[cond1] = 0.25 * t0
    qx[cond1] = (rotation_matrix[cond1, 2, 1] - rotation_matrix[cond1, 1, 2]) / t0
    qy[cond1] = (rotation_matrix[cond1, 0, 2] - rotation_matrix[cond1, 2, 0]) / t0
    qz[cond1] = (rotation_matrix[cond1, 1, 0] - rotation_matrix[cond1, 0, 1]) / t0

    # Other cases where a diagonal element is the largest entry of the matrix...
    # Case 2: The largest element is rotation_matrix[:, 0, 0]
    t1 = torch.sqrt(1.0 + rotation_matrix[cond2, 0, 0] - rotation_matrix[cond2, 1, 1] - rotation_matrix[cond2, 2, 2]) * 2
    qw[cond2] = (rotation_matrix[cond2, 2, 1] - rotation_matrix[cond2, 1, 2]) / t1
    qx[cond2] = 0.25 * t1
    qy[cond2] = (rotation_matrix[cond2, 0, 1] + rotation_matrix[cond2, 1, 0]) / t1
    qz[cond2] = (rotation_matrix[cond2, 0, 2] + rotation_matrix[cond2, 2, 0]) / t1
    
    # Case 3: The largest element is rotation_matrix[:, 1, 1]
    t2 = torch.sqrt(1.0 + rotation_matrix[cond3, 1, 1] - rotation_matrix[cond3, 0, 0] - rotation_matrix[cond3, 2, 2]) * 2
    qw[cond3] = (rotation_matrix[cond3, 0, 2] - rotation_matrix[cond3, 2, 0]) / t2
    qx[cond3] = (rotation_matrix[cond3, 0, 1] + rotation_matrix[cond3, 1, 0]) / t2
    qy[cond3] = 0.25 * t2
    qz[cond3] = (rotation_matrix[cond3, 1, 2] + rotation_matrix[cond3, 2, 1]) / t2

    # Case 4: The largest element is rotation_matrix[:, 2, 2]
    cond4 = (~cond1) & (~cond2) & (~cond3)
    t3 = torch.sqrt(1.0 + rotation_matrix[cond4, 2, 2] - rotation_matrix[cond4, 0, 0] - rotation_matrix[cond4, 1, 1]) * 2
    qw[cond4] = (rotation_matrix[cond4, 1, 0] - rotation_matrix[cond4, 0, 1]) / t3
    qx[cond4] = (rotation_matrix[cond4, 0, 2] + rotation_matrix[cond4, 2, 0]) / t3
    qy[cond4] = (rotation_matrix[cond4, 1, 2] + rotation_matrix[cond4, 2, 1]) / t3
    qz[cond4] = 0.25 * t3

    # Combine quaternion components
    quaternions = torch.stack((qw, qx, qy, qz), dim=1)

    # Normalize quaternions
    quaternions = quaternions / quaternions.norm(p=2, dim=1, keepdim=True)

    return quaternions


def euler_angles_to_rotation_matrix(angles):
    """Convert Euler angles to rotation matrix.
       Angles: tensor of shape [3], representing [roll, pitch, yaw] in radians.
    """
    roll, pitch, yaw = angles

    # 计算单一轴的旋转矩阵
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    Rx = torch.stack([
        torch.stack([torch.tensor(1.0, device=angles.device), torch.tensor(0.0, device=angles.device), torch.tensor(0.0, device=angles.device)]),
        torch.stack([torch.tensor(0.0, device=angles.device), cos_r, -sin_r]),
        torch.stack([torch.tensor(0.0, device=angles.device), sin_r, cos_r])
    ])

    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    Ry = torch.stack([
        torch.stack([cos_p, torch.tensor(0.0, device=angles.device), sin_p]),
        torch.stack([torch.tensor(0.0, device=angles.device), torch.tensor(1.0, device=angles.device), torch.tensor(0.0, device=angles.device)]),
        torch.stack([-sin_p, torch.tensor(0.0, device=angles.device), cos_p])
    ])

    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    Rz = torch.stack([
        torch.stack([cos_y, -sin_y, torch.tensor(0.0, device=angles.device)]),
        torch.stack([sin_y, cos_y, torch.tensor(0.0, device=angles.device)]),
        torch.stack([torch.tensor(0.0, device=angles.device), torch.tensor(0.0, device=angles.device), torch.tensor(1.0, device=angles.device)])
    ])

    # 旋转矩阵顺序：Rz * Ry * Rx
    rotation_matrix = torch.matmul(Rz, torch.matmul(Ry, Rx))

    return rotation_matrix