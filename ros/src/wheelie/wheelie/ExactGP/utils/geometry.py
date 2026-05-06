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
def up_and_updot_from_quat_gyro(
    qw: float, qx: float, qy: float, qz: float,
    wx: float, wy: float, wz: float
):
    """
    Compute:
      up    = R[2,2] = world_z dot body_z_world in [-1,1]
      updot = d(up)/dt computed from rigid body kinematics + gyro.

    Uses:
      updot = (e3 x b3) dot w_world
      where b3 = body z-axis in world, w_world = R * w_body
    """
    R, _= quat_to_R_and_pitch(qw, qx, qy, qz)

    # body z axis in world
    b3 = R[:, 2]
    up = float(b3[2])  # = R[2,2]

    w_body = np.array([wx, wy, wz], dtype=np.float64)
    w_world = R @ w_body

    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    updot = float(np.dot(np.cross(e3, b3), w_world))
    return up, updot, R


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



@staticmethod
def wrap_to_pi(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@staticmethod
def continuous_angle(prev_unwrapped: float | None, new_wrapped: float) -> float:
    """
    Convert a wrapped angle into a continuous/unwrapped angle.

    prev_unwrapped:
        previous continuous angle, or None for first sample

    new_wrapped:
        new angle from atan2, usually in [-pi, pi]
    """
    if prev_unwrapped is None:
        return new_wrapped

    delta = wrap_to_pi(new_wrapped - prev_unwrapped)
    return prev_unwrapped + delta


@staticmethod
def quat_to_wheelie_pitch(qw, qx, qy, qz):
    """
    Quaternion -> wheelie pitch angle without the Euler pitch singularity.

    For pure pitch motion:
        0 rad        = car flat
        pi/2 rad     = car vertical, 90 degrees
        pi rad       = car upside down

    This does NOT fold at +/-90 degrees like Euler pitch.
    """

    # Normalize quaternion
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n == 0.0:
        return 0.0

    qw /= n
    qx /= n
    qy /= n
    qz /= n

    # Rotation matrix terms needed for pitch around body/world y direction
    # This gives atan2(sin(theta), cos(theta)) instead of asin(sin(theta)).
    sin_pitch = 2.0 * (qw*qy - qx*qz)
    cos_pitch = 1.0 - 2.0 * (qy*qy + qz*qz)

    return math.atan2(sin_pitch, cos_pitch)


@staticmethod
def quat_to_wheelie_state(
    qw, qx, qy, qz,
    wx, wy, wz,
    prev_pitch_unwrapped=None,
    pitch_rate_sign=1.0,
):
    """
    Quaternion + gyro -> wheelie pitch and pitch_dot.

    Returns:
        pitch_unwrapped, pitch_dot

    pitch_rate_sign:
        Use +1.0 first.
        If your pitch increases but pitch_dot has the wrong sign, use -1.0.
    """

    pitch_wrapped = quat_to_wheelie_pitch(qw, qx, qy, qz)
    pitch_unwrapped = continuous_angle(prev_pitch_unwrapped, pitch_wrapped)

    # For a wheelie, pitch rate is usually gyro y.
    # If sign is wrong in your robot, change pitch_rate_sign to -1.0.
    pitch_dot = float(pitch_rate_sign) * float(wy)

    return pitch_unwrapped, pitch_dot