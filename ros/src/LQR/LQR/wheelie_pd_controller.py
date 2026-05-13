"""
wheelie_pd_controller.py

Simple wheelie pitch controller using feedback linearization + PD.
Model:
    x_dot     = v
    v_dot     = tau / (m*r)
    theta_dot = omega
    omega_dot = (-tau + m*g*l*cos(theta)) / I_eff

Control law from feedback linearization:
    tau = m*g*l*cos(theta) + I_eff*(kp*(theta-theta_ref) + kd*omega)

This makes the closed-loop pitch error approximately:
    e_ddot + kd*e_dot + kp*e = 0

Run:
    python wheelie_pd_controller.py
"""

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class WheelieParams:
    m: float = 2.0          # kg
    l: float = 0.18         # m, rear axle to COM distance
    I_body: float = 0.04    # kg*m^2, body inertia about COM
    r: float = 0.05         # m, rear wheel radius
    g: float = 9.81         # m/s^2
    tau_min: float = -4.0   # N*m
    tau_max: float = 4.0    # N*m
    pitch_ref: float = 90.0
    c_v: float = 9.0 

    sim_time: float = 5.0
    sim_dt: float = 0.1

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class PDGains:
    kp: float = 30.0
    kd: float = 8.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def wheelie_dynamics(state: np.ndarray, tau: float, p: WheelieParams) -> np.ndarray:
    """Continuous-time dynamics."""
    x, v, theta, omega = state

    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v #tau / (p.m * p.r)
    theta_dot = omega
    omega_dot = (-tau + p.m * p.g * p.l * math.cos(theta)) / p.I_eff
    dynamics = np.array([x_dot, v_dot, theta_dot, omega_dot], dtype=float)

    return dynamics


def rk4_step(state: np.ndarray, tau: float, dt: float, p: WheelieParams) -> np.ndarray:
    """Fourth-order Runge-Kutta integration."""
    k1 = wheelie_dynamics(state, tau, p)
    k2 = wheelie_dynamics(state + 0.5 * dt * k1, tau, p)
    k3 = wheelie_dynamics(state + 0.5 * dt * k2, tau, p)
    k4 = wheelie_dynamics(state + dt * k3, tau, p)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def pd_feedback_linearizing_controller(
    theta: float,
    omega: float,
    theta_ref: float,
    p: WheelieParams,
    gains: PDGains,
) -> float:
    """
    Feedback linearization + PD.

    Original pitch dynamics:
        I_eff*theta_ddot = -tau + m*g*l*cos(theta)

    Choose desired virtual acceleration:
        theta_ddot = -kp*(theta-theta_ref) - kd*omega

    Then solve for tau:
        tau = m*g*l*cos(theta) + I_eff*(kp*(theta-theta_ref) + kd*omega)
    """
    error_pitch = theta - theta_ref
    error_dot = omega
    tau = p.m * p.g * p.l * math.cos(theta) + p.I_eff * (gains.kp * error_pitch + gains.kd * error_dot)
    return clamp(tau, p.tau_min, p.tau_max)


def simulate() -> None:
    p = WheelieParams()
    gains = PDGains()

    # Reference wheelie angle.
    theta_ref_deg = p.pitch_ref
    theta_ref = math.radians(theta_ref_deg)

    # Initial state: x, v, theta, omega.
    state = np.array([0.0, 0.0, math.radians(0.0), 0.0], dtype=float)

    dt = p.sim_dt
    T = p.sim_time
    steps = int(T / dt)

    history = np.zeros((steps, 6))

    for k in range(steps):
        t = k * dt
        x, v, theta, omega = state

        tau = pd_feedback_linearizing_controller(theta, omega, theta_ref, p, gains)

        history[k] = [t, x, v, theta, omega, tau]
        state = rk4_step(state, tau, dt, p)

    t = history[:, 0]
    theta_deg = np.rad2deg(history[:, 3])
    omega = history[:, 4]
    tau = history[:, 5]
    x = history[:, 1]
    v = history[:, 2]

    print(f"Final theta: {theta_deg[-1]:.2f} deg")
    print(f"Final omega: {omega[-1]:.3f} rad/s")
    print(f"Final x: {x[-1]:.2f} m")
    print(f"Final v: {v[-1]:.2f} m/s")

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

    axs[0].plot(t, theta_deg, label="theta")
    axs[0].axhline(theta_ref_deg, linestyle="--", label="theta_ref", color="tab:orange")
    axs[0].set_ylabel("pitch angle [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, tau)
    axs[1].set_ylabel("rear axle torque [N m]")
    axs[1].grid(True)

    axs[2].plot(t, v)
    axs[2].set_xlabel("time [s]")
    axs[2].set_ylabel("forward velocity [m/s]")
    axs[2].grid(True)

    fig.suptitle("Wheelie PD Closed-Loop Response")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate()
