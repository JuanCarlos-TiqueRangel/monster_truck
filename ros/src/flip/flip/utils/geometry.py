import numpy as np

def quat_to_R(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Quaternion -> rotation matrix (body->world).
    Assumes quaternion is (w, x, y, z).
    """
    R00 = 1 - 2 * (qy * qy + qz * qz)
    R01 = 2 * (qx * qy - qw * qz)
    R02 = 2 * (qx * qz + qw * qy)

    R10 = 2 * (qx * qy + qw * qz)
    R11 = 1 - 2 * (qx * qx + qz * qz)
    R12 = 2 * (qy * qz - qw * qx)

    R20 = 2 * (qx * qz - qw * qy)
    R21 = 2 * (qy * qz + qw * qx)
    R22 = 1 - 2 * (qx * qx + qy * qy)

    return np.array([[R00, R01, R02],
                     [R10, R11, R12],
                     [R20, R21, R22]], dtype=np.float64)


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
    R = quat_to_R(qw, qx, qy, qz)

    # body z axis in world
    b3 = R[:, 2]
    up = float(b3[2])  # = R[2,2]

    w_body = np.array([wx, wy, wz], dtype=np.float64)
    w_world = R @ w_body

    e3 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    updot = float(np.dot(np.cross(e3, b3), w_world))
    return up, updot, R