"""Geometry helpers for the LQR wheelie controller."""

import math
from typing import Optional


def clip(value: float, lower: float, upper: float) -> float:
    """Clip a scalar value to [lower, upper]."""
    return min(max(float(value), float(lower)), float(upper))


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def continuous_angle(
    prev_unwrapped: Optional[float],
    new_wrapped: float,
) -> float:
    """Convert a wrapped angle measurement into a continuous angle."""
    if prev_unwrapped is None:
        return float(new_wrapped)
    return float(prev_unwrapped) + wrap_to_pi(
        float(new_wrapped) - float(prev_unwrapped)
    )


def quat_to_wheelie_pitch(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
) -> float:
    """Return singularity-free wheelie pitch from a quaternion."""
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0:
        return 0.0

    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm

    sin_pitch = 2.0 * (qw * qy - qx * qz)
    cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(sin_pitch, cos_pitch)


def quat_to_wheelie_state(
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    wx: float,
    wy: float,
    wz: float,
    prev_pitch_unwrapped: Optional[float] = None,
    pitch_rate_sign: float = 1.0,
) -> tuple[float, float]:
    """Return continuous wheelie pitch and pitch rate from IMU data."""
    del wx, wz
    pitch_wrapped = quat_to_wheelie_pitch(qw, qx, qy, qz)
    pitch_unwrapped = continuous_angle(prev_pitch_unwrapped, pitch_wrapped)
    pitch_dot = float(pitch_rate_sign) * float(wy)
    return pitch_unwrapped, pitch_dot
