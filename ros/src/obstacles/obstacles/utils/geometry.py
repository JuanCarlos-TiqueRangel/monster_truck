import math
import numpy as np
import torch

# ========================================================
# Helpers: quaternion -> R + pitch, angle unwrap
# ========================================================
@staticmethod
def quat_to_R_and_pitch(qw, qx, qy, qz):
    R00 = 1 - 2 * (qy * qy + qz * qz)
    R01 = 2 * (qx * qy - qw * qz)
    R02 = 2 * (qx * qz + qw * qy)

    R10 = 2 * (qx * qy + qw * qz)
    R11 = 1 - 2 * (qx * qx + qz * qz)
    R12 = 2 * (qy * qz - qw * qx)

    R20 = 2 * (qx * qz - qw * qy)
    R21 = 2 * (qy * qz + qw * qx)
    R22 = 1 - 2 * (qx * qx + qy * qy)

    pitch = -math.asin(max(-1.0, min(1.0, R20)))
    R = np.array([[R00, R01, R02],
                    [R10, R11, R12],
                    [R20, R21, R22]], dtype=float)
    return R, pitch


@staticmethod
def quat_to_euler_xyz(w, x, y, z):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)
    using the roll-pitch-yaw convention (XYZ intrinsic).
    
    Returns: (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp to handle numerical drift outside [-1, 1]
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


@staticmethod
def unwrap_angle(prev_angle, prev_unwrapped, angle):
    if prev_angle is None:
        return angle, angle
    d = angle - prev_angle
    if d > math.pi:
        angle_unwrapped = prev_unwrapped + (d - 2 * math.pi)
    elif d < -math.pi:
        angle_unwrapped = prev_unwrapped + (d + 2 * math.pi)
    else:
        angle_unwrapped = prev_unwrapped + d
    return angle, angle_unwrapped


# ========================================================
# Torch helpers: angdiff, stage cost, GP step
# ========================================================
@staticmethod
def angdiff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi