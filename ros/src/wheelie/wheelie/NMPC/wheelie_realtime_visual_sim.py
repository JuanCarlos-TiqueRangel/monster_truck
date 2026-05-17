#!/usr/bin/env python3
"""
wheelie_realtime_visual_sim.py

Real-time 2D visualization for the wheelie pitch controllers.

Expected files in the same folder:
    wheelie_pd_controller.py
    wheelie_nmpc_casadi.py

Examples
--------
PD controller:
    python3 wheelie_realtime_visual_sim.py --controller pd --theta-ref-deg 90

NMPC controller:
    python3 wheelie_realtime_visual_sim.py --controller nmpc --theta-ref-deg 90

NMPC with a different initial angle:
    python3 wheelie_realtime_visual_sim.py --controller nmpc --theta-ref-deg 75 --theta0-deg 0

Notes
-----
- The animation uses the same dynamics functions from your controller files.
- For NMPC, IPOPT may be slower than real time. If the animation is laggy, use:
    python3 wheelie_realtime_visual_sim.py --controller nmpc --precompute
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Polygon


State = np.ndarray


@dataclass
class SimConfig:
    controller: str
    theta_ref_deg: float
    theta0_deg: float
    v0: float
    omega0: float
    x0: float
    sim_time: float
    realtime_factor: float
    precompute: bool
    v_ref: float


@dataclass
class SimData:
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    tau: np.ndarray
    theta_ref: np.ndarray


class ControllerWrapper:
    """
    Small adapter that makes the PD and NMPC files look the same to the visualizer.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.controller_type = cfg.controller.lower()

        if self.controller_type == "pd":
            import wheelie_pd_controller as pd_mod

            self.mod = pd_mod
            self.p = pd_mod.WheelieParams()
            self.gains = pd_mod.PDGains()
            self.dt = float(self.p.sim_dt)
            self.theta_ref = math.radians(cfg.theta_ref_deg)

            self.p.pitch_ref = cfg.theta_ref_deg
            self.p.sim_time = cfg.sim_time

            self.state_step: Callable[[State, float], State] = (
                lambda state, tau: pd_mod.rk4_step(state, tau, self.dt, self.p)
            )

        elif self.controller_type == "nmpc":
            import wheelie_nmpc_casadi as nmpc_mod

            self.mod = nmpc_mod
            self.p = nmpc_mod.WheelieParams()
            self.mpc_cfg = nmpc_mod.MPCConfig()
            self.controller = nmpc_mod.WheelieNMPC(self.p, self.mpc_cfg)
            self.dt = float(self.p.sim_dt)
            self.theta_ref = math.radians(cfg.theta_ref_deg)
            self.tau_prev = 0.0

            self.p.pitch_ref = cfg.theta_ref_deg
            self.p.sim_time = cfg.sim_time

            self.state_step = (
                lambda state, tau: nmpc_mod.rk4_step_np(state, tau, self.dt, self.p)
            )

        else:
            raise ValueError("controller must be either 'pd' or 'nmpc'")

    def control(self, state: State) -> float:
        theta = float(state[2])
        omega = float(state[3])

        if self.controller_type == "pd":
            return float(
                self.mod.pd_feedback_linearizing_controller(
                    theta,
                    omega,
                    self.theta_ref,
                    self.p,
                    self.gains,
                )
            )

        ref = np.array([0.0, self.cfg.v_ref, self.theta_ref, 0.0], dtype=float)
        tau_cmd, info = self.controller.solve(state, ref, self.tau_prev)
        self.tau_prev = float(tau_cmd)

        if not info.get("success", False):
            print("[WARN] NMPC failed; using fallback torque.")

        return float(tau_cmd)

    def step(self, state: State, tau: float) -> State:
        return self.state_step(state, tau)


def simulate_trajectory(wrapper: ControllerWrapper, cfg: SimConfig) -> SimData:
    dt = wrapper.dt
    steps = int(cfg.sim_time / dt) + 1

    t_hist = np.zeros(steps)
    x_hist = np.zeros(steps)
    v_hist = np.zeros(steps)
    theta_hist = np.zeros(steps)
    omega_hist = np.zeros(steps)
    tau_hist = np.zeros(steps)
    theta_ref_hist = np.zeros(steps)

    state = np.array(
        [
            cfg.x0,
            cfg.v0,
            math.radians(cfg.theta0_deg),
            cfg.omega0,
        ],
        dtype=float,
    )

    for k in range(steps):
        t = k * dt
        tau = wrapper.control(state)

        t_hist[k] = t
        x_hist[k] = state[0]
        v_hist[k] = state[1]
        theta_hist[k] = state[2]
        omega_hist[k] = state[3]
        tau_hist[k] = tau
        theta_ref_hist[k] = wrapper.theta_ref

        state = wrapper.step(state, tau)

    return SimData(
        t=t_hist,
        x=x_hist,
        v=v_hist,
        theta=theta_hist,
        omega=omega_hist,
        tau=tau_hist,
        theta_ref=theta_ref_hist,
    )


def make_car_polygon(x_rear: float, theta: float, body_length: float, body_height: float) -> np.ndarray:
    """
    Rectangular body attached to the rear axle.

    theta = 0 deg  -> body horizontal
    theta = 90 deg -> body vertical
    """
    rear = np.array([x_rear, 0.0])
    e_body = np.array([math.cos(theta), math.sin(theta)])
    e_norm = np.array([-math.sin(theta), math.cos(theta)])

    rear_lower = rear - 0.45 * body_height * e_norm
    rear_upper = rear + 0.55 * body_height * e_norm
    front_upper = rear + body_length * e_body + 0.55 * body_height * e_norm
    front_lower = rear + body_length * e_body - 0.45 * body_height * e_norm

    return np.vstack([rear_lower, front_lower, front_upper, rear_upper])


def animate(data: Optional[SimData], wrapper: ControllerWrapper, cfg: SimConfig) -> None:
    dt = wrapper.dt

    if data is not None:
        steps = len(data.t)
    else:
        steps = int(cfg.sim_time / dt) + 1

    live_state = np.array(
        [
            cfg.x0,
            cfg.v0,
            math.radians(cfg.theta0_deg),
            cfg.omega0,
        ],
        dtype=float,
    )

    live_t = []
    live_theta = []
    live_tau = []
    live_v = []

    body_length = 0.55
    body_height = 0.12
    wheel_radius = float(wrapper.p.r)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1.0], height_ratios=[1, 1, 1])

    ax_car = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_tau = fig.add_subplot(gs[1, 1])
    ax_v = fig.add_subplot(gs[2, 1])

    ground_line, = ax_car.plot([], [], linewidth=2)
    body_patch = Polygon(np.zeros((4, 2)), closed=True, alpha=0.85)
    rear_wheel = Circle((0.0, 0.0), wheel_radius, fill=False, linewidth=2)
    front_wheel = Circle((0.0, 0.0), wheel_radius, fill=False, linewidth=2)
    rear_axle_dot, = ax_car.plot([], [], "ko", markersize=4)
    com_dot, = ax_car.plot([], [], "o", markersize=5)
    ref_text = ax_car.text(0.02, 0.95, "", transform=ax_car.transAxes, va="top")
    state_text = ax_car.text(0.02, 0.88, "", transform=ax_car.transAxes, va="top")

    ax_car.add_patch(body_patch)
    ax_car.add_patch(rear_wheel)
    ax_car.add_patch(front_wheel)

    ax_car.set_aspect("equal", adjustable="box")
    ax_car.set_xlabel("x [m]")
    ax_car.set_ylabel("height [m]")
    ax_car.set_title(f"Wheelie real-time visualization ({cfg.controller.upper()})")
    ax_car.grid(True)

    theta_line, = ax_theta.plot([], [], label="theta", color="C0")
    theta_ref_line, = ax_theta.plot([], [], "--", label="theta_ref", color="C1")
    tau_line, = ax_tau.plot([], [], label="tau", color="C2")
    v_line, = ax_v.plot([], [], label="v", color="C3")

    ax_theta.set_ylabel("pitch [deg]")
    ax_tau.set_ylabel("torque [N m]")
    ax_v.set_ylabel("velocity [m/s]")
    ax_v.set_xlabel("time [s]")

    for ax in [ax_theta, ax_tau, ax_v]:
        ax.grid(True)
        ax.legend(loc="upper right")

    ax_theta.set_xlim(0.0, cfg.sim_time)
    ax_tau.set_xlim(0.0, cfg.sim_time)
    ax_v.set_xlim(0.0, cfg.sim_time)

    ax_theta.set_ylim(-10.0, max(110.0, cfg.theta_ref_deg + 20.0))
    ax_tau.set_ylim(wrapper.p.tau_min - 0.5, wrapper.p.tau_max + 0.5)
    ax_v.set_ylim(-5.0, 5.0)

    def update(frame: int):
        nonlocal live_state

        if data is not None:
            t = float(data.t[frame])
            x = float(data.x[frame])
            v = float(data.v[frame])
            theta = float(data.theta[frame])
            omega = float(data.omega[frame])
            tau = float(data.tau[frame])

            t_plot = data.t[: frame + 1]
            theta_plot = np.rad2deg(data.theta[: frame + 1])
            theta_ref_plot = np.rad2deg(data.theta_ref[: frame + 1])
            tau_plot = data.tau[: frame + 1]
            v_plot = data.v[: frame + 1]

        else:
            t = frame * dt
            tau = wrapper.control(live_state)
            x, v, theta, omega = [float(val) for val in live_state]

            live_t.append(t)
            live_theta.append(math.degrees(theta))
            live_tau.append(tau)
            live_v.append(v)

            live_state = wrapper.step(live_state, tau)

            t_plot = np.asarray(live_t)
            theta_plot = np.asarray(live_theta)
            theta_ref_plot = np.full_like(t_plot, cfg.theta_ref_deg, dtype=float)
            tau_plot = np.asarray(live_tau)
            v_plot = np.asarray(live_v)

        rear_x = x
        rear_y = wheel_radius
        theta_for_drawing = theta

        poly = make_car_polygon(rear_x, theta_for_drawing, body_length, body_height)
        poly[:, 1] += rear_y
        body_patch.set_xy(poly)

        front_center = np.array(
            [
                rear_x + body_length * math.cos(theta_for_drawing),
                rear_y + body_length * math.sin(theta_for_drawing),
            ]
        )

        rear_wheel.center = (rear_x, wheel_radius)
        front_wheel.center = (front_center[0], max(wheel_radius, front_center[1] - 0.04))

        rear_axle_dot.set_data([rear_x], [rear_y])

        com_pos = np.array(
            [
                rear_x + float(wrapper.p.l) * math.cos(theta_for_drawing),
                rear_y + float(wrapper.p.l) * math.sin(theta_for_drawing),
            ]
        )
        com_dot.set_data([com_pos[0]], [com_pos[1]])

        x_center = rear_x + 0.3
        ax_car.set_xlim(x_center - 1.0, x_center + 1.0)
        ax_car.set_ylim(-0.05, 1.1)
        ground_line.set_data([x_center - 2.0, x_center + 2.0], [0.0, 0.0])

        ref_text.set_text(f"reference pitch: {cfg.theta_ref_deg:.1f} deg")
        state_text.set_text(
            f"t={t:.2f} s | theta={math.degrees(theta):.2f} deg | "
            f"omega={omega:.2f} rad/s\n"
            f"v={v:.2f} m/s | tau={tau:.2f} N m"
        )

        theta_line.set_data(t_plot, theta_plot)
        theta_ref_line.set_data(t_plot, theta_ref_plot)
        tau_line.set_data(t_plot, tau_plot)
        v_line.set_data(t_plot, v_plot)

        if len(v_plot) > 2:
            v_min = min(-1.0, float(np.min(v_plot)) - 0.5)
            v_max = max(1.0, float(np.max(v_plot)) + 0.5)
            ax_v.set_ylim(v_min, v_max)

        fig.suptitle(
            f"Wheelie Pitch Control | {cfg.controller.upper()} | "
            f"dt={dt:.2f} s | real-time factor={cfg.realtime_factor:.2f}"
        )

        return (
            ground_line,
            body_patch,
            rear_wheel,
            front_wheel,
            rear_axle_dot,
            com_dot,
            ref_text,
            state_text,
            theta_line,
            theta_ref_line,
            tau_line,
            v_line,
        )

    interval_ms = 1000.0 * dt / max(cfg.realtime_factor, 1e-6)

    _anim = FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()


def parse_args() -> SimConfig:
    parser = argparse.ArgumentParser(description="Real-time wheelie controller visualization.")
    parser.add_argument("--controller", choices=["pd", "nmpc"], default="pd")
    parser.add_argument("--theta-ref-deg", type=float, default=90.0)
    parser.add_argument("--theta0-deg", type=float, default=0.0)
    parser.add_argument("--v0", type=float, default=0.0)
    parser.add_argument("--omega0", type=float, default=0.0)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--sim-time", type=float, default=5.0)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--v-ref", type=float, default=0.0)

    args = parser.parse_args()

    return SimConfig(
        controller=args.controller,
        theta_ref_deg=args.theta_ref_deg,
        theta0_deg=args.theta0_deg,
        v0=args.v0,
        omega0=args.omega0,
        x0=args.x0,
        sim_time=args.sim_time,
        realtime_factor=args.realtime_factor,
        precompute=args.precompute,
        v_ref=args.v_ref,
    )


def main() -> None:
    cfg = parse_args()
    wrapper = ControllerWrapper(cfg)

    data = None
    if cfg.precompute:
        print("[INFO] Precomputing trajectory...")
        data = simulate_trajectory(wrapper, cfg)
        print("[INFO] Done. Replaying animation.")

    animate(data, wrapper, cfg)


if __name__ == "__main__":
    main()
