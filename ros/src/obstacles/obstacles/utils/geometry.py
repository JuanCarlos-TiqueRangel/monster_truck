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
def quat_to_euler_xyz(w, x, y, z, wx, wy, wz):
    # normalize quaternion
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    w /= n
    x /= n
    y /= n
    z /= n

    # Roll
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    p = wx
    q = wy
    r = wz

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)

    eps = 1e-8
    if abs(cp) < eps:
        cp = eps if cp >= 0.0 else -eps

    tp = math.sin(pitch) / cp

    roll_dot  = p + q * sr * tp + r * cr * tp
    pitch_dot = q * cr - r * sr
    yaw_dot   = (q * sr + r * cr) / cp

    return roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot


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