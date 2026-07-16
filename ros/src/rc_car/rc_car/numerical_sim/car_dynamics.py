from dataclasses import dataclass
import math

import numpy as np


@dataclass
class WheelieParams:
    """Physical parameters and simulation settings for the wheelie model."""

    m: float = 5.1
    l: float = 0.2
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.081
    g: float = 9.81
    c_v: float = 9.0
    tau_min: float = -8.0
    tau_max: float = 8.0
    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(90.0)
    omega_min: float = -5.0
    omega_max: float = 5.0
    v_min: float = -5.0
    v_max: float = 5.0
    pitch_ref: float = 70.0
    sim_time: float = 5.0
    sim_dt: float = 0.01

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


def continuous_dynamics_np(
    state: np.ndarray,
    tau: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    """Evaluate the continuous-time numerical plant model."""
    x, v, theta, omega = state
    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v
    theta_dot = omega
    omega_dot = (-tau + p.m * p.g * p.l * np.cos(theta)) / p.I_eff

    if plant_has_mismatch:
        v_dot += 0.10 * np.sin(theta) + 0.15 * tau * np.cos(theta)
        omega_dot += 0.5 * omega + 1.0 * v + 3.0 * np.sin(theta)

    return np.array([x_dot, v_dot, theta_dot, omega_dot], dtype=float)


def rk4_step_np(
    state: np.ndarray,
    tau: float,
    dt: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    """Advance the numerical plant by one fourth-order Runge-Kutta step."""
    k1 = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
    k2 = continuous_dynamics_np(state + 0.5 * dt * k1, tau, p, plant_has_mismatch)
    k3 = continuous_dynamics_np(state + 0.5 * dt * k2, tau, p, plant_has_mismatch)
    k4 = continuous_dynamics_np(state + dt * k3, tau, p, plant_has_mismatch)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
