#!/usr/bin/env python3
"""
wheelie_realtime_visual_sim_v3.py

Improved real-time 2D visualization for the wheelie pitch controllers.

Expected files in the same folder:
    wheelie_pd_controller.py
    wheelie_nmpc_casadi.py

Examples
--------
PD controller:
    python3 wheelie_realtime_visual_sim_v3.py --controller pd --theta-ref-deg 90

NMPC controller:
    python3 wheelie_realtime_visual_sim_v3.py --controller nmpc --theta-ref-deg 90

NMPC with precompute:
    python3 wheelie_realtime_visual_sim_v3.py --controller nmpc --theta-ref-deg 75 --precompute
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
        [cfg.x0, cfg.v0, math.radians(cfg.theta0_deg), cfg.omega0],
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


def transform_points(local_pts: np.ndarray, origin: np.ndarray, theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return local_pts @ R.T + origin


def animate(data: Optional[SimData], wrapper: ControllerWrapper, cfg: SimConfig) -> None:
    dt = wrapper.dt

    if data is not None:
        steps = len(data.t)
    else:
        steps = int(cfg.sim_time / dt) + 1

    live_state = np.array(
        [cfg.x0, cfg.v0, math.radians(cfg.theta0_deg), cfg.omega0],
        dtype=float,
    )

    live_t = []
    live_theta = []
    live_tau = []
    live_v = []

    # ----------------------------
    # Visual-only geometry/scaling
    # ----------------------------
    # We decouple the visual size from the physical parameters so the truck
    # looks more like a monster truck while the dynamics remain unchanged.
    vis_wheel_radius = 0.18
    vis_front_wheel_radius = 0.165
    body_length = 0.92
    rear_overhang = 0.14
    body_scale = vis_wheel_radius / max(float(wrapper.p.r), 1e-6)
    vis_com_offset = float(wrapper.p.l) * body_scale

    front_axle_local = np.array([0.62, 0.01])

    # Side-view body silhouette inspired by a monster truck body
    body_local = np.array([
        [-rear_overhang, 0.02],   # rear bumper lower
        [-0.10, 0.14],            # rear bed rise
        [ 0.05, 0.20],            # rear cabin
        [ 0.40, 0.23],            # roof back
        [ 0.60, 0.23],            # roof front
        [ 0.71, 0.18],            # windshield top
        [ 0.84, 0.14],            # hood top
        [ 0.88, 0.09],            # front nose upper
        [ 0.86, 0.04],            # front nose lower
        [ 0.67, 0.03],            # hood lower
        [ 0.52, 0.03],            # front body lower
        [ 0.28, 0.02],            # middle lower
        [ 0.08, 0.02],            # near rear lower
    ])

    window_local = np.array([
        [0.47, 0.205],
        [0.62, 0.205],
        [0.69, 0.165],
        [0.54, 0.155],
    ])

    hood_flame_local = np.array([
        [0.02, 0.08],
        [0.12, 0.12],
        [0.26, 0.10],
        [0.40, 0.16],
        [0.60, 0.13],
        [0.76, 0.10],
    ])

    side_flame_local = np.array([
        [0.12, 0.05],
        [0.28, 0.12],
        [0.45, 0.09],
        [0.59, 0.16],
        [0.75, 0.12],
    ])

    # Figure layout
    fig = plt.figure(figsize=(13.8, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.65, 1.0], height_ratios=[1, 1, 1])

    ax_car = fig.add_subplot(gs[:, 0])
    ax_theta = fig.add_subplot(gs[0, 1])
    ax_tau = fig.add_subplot(gs[1, 1])
    ax_v = fig.add_subplot(gs[2, 1])

    # ----------------------------
    # Car artists
    # ----------------------------
    ground_line, = ax_car.plot([], [], color="#444444", linewidth=2.5)
    axle_line, = ax_car.plot([], [], color="#202020", linewidth=4)
    shock_line_1, = ax_car.plot([], [], color="#303030", linewidth=2)
    shock_line_2, = ax_car.plot([], [], color="#303030", linewidth=2)

    body_patch = Polygon(np.zeros((4, 2)), closed=True,
                         facecolor="#0a1220", edgecolor="black", linewidth=2.2)
    shadow_body_patch = Polygon(np.zeros((4, 2)), closed=True,
                                facecolor="#101820", edgecolor="none", alpha=0.35)
    window_patch = Polygon(np.zeros((4, 2)), closed=True,
                           facecolor="#e6eef7", edgecolor="white", linewidth=1.5)
    flame_hood_line, = ax_car.plot([], [], color="#4a7cff", linewidth=3.0)
    flame_side_line, = ax_car.plot([], [], color="#77aaff", linewidth=2.5)
    body_trim_line, = ax_car.plot([], [], color="#5d84ff", linewidth=1.5)

    rear_tire = Circle((0.0, 0.0), vis_wheel_radius, facecolor="#1c1c1c",
                       edgecolor="black", linewidth=2.3)
    front_tire = Circle((0.0, 0.0), vis_front_wheel_radius, facecolor="#1c1c1c",
                        edgecolor="black", linewidth=2.3)

    rear_rim = Circle((0.0, 0.0), vis_wheel_radius * 0.42, facecolor="#a8c8ff",
                      edgecolor="#4b6fff", linewidth=2.0)
    front_rim = Circle((0.0, 0.0), vis_front_wheel_radius * 0.42, facecolor="#a8c8ff",
                       edgecolor="#4b6fff", linewidth=2.0)

    rear_hub = Circle((0.0, 0.0), vis_wheel_radius * 0.10, facecolor="#d9d9d9",
                      edgecolor="#666666", linewidth=1.0)
    front_hub = Circle((0.0, 0.0), vis_front_wheel_radius * 0.10, facecolor="#d9d9d9",
                       edgecolor="#666666", linewidth=1.0)

    rear_axle_dot, = ax_car.plot([], [], "ko", markersize=4)
    com_dot, = ax_car.plot([], [], "o", color="red", markersize=6)

    ref_text = ax_car.text(0.02, 0.97, "", transform=ax_car.transAxes, va="top",
                           fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    state_text = ax_car.text(0.02, 0.88, "", transform=ax_car.transAxes, va="top",
                             fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    for patch in [
        rear_tire, front_tire, rear_rim, front_rim, rear_hub, front_hub,
        shadow_body_patch, body_patch, window_patch
    ]:
        ax_car.add_patch(patch)

    ax_car.set_aspect("equal", adjustable="box")
    ax_car.set_xlabel("x [m]")
    ax_car.set_ylabel("height [m]")
    ax_car.set_title(f"Wheelie real-time visualization ({cfg.controller.upper()})")
    ax_car.grid(True, alpha=0.35)

    # ----------------------------
    # Time-series plots
    # ----------------------------
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
        ax.set_xlim(0.0, cfg.sim_time)

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

        rear_center = np.array([x, vis_wheel_radius])
        body_pts = transform_points(body_local, rear_center, theta)
        shadow_pts = body_pts.copy()
        shadow_pts[:, 1] -= 0.02
        window_pts = transform_points(window_local, rear_center, theta)
        hood_flame_pts = transform_points(hood_flame_local, rear_center, theta)
        side_flame_pts = transform_points(side_flame_local, rear_center, theta)

        front_center = transform_points(front_axle_local.reshape(1, 2), rear_center, theta)[0]
        front_center[1] = max(vis_front_wheel_radius, front_center[1])

        body_patch.set_xy(body_pts)
        shadow_body_patch.set_xy(shadow_pts)
        window_patch.set_xy(window_pts)
        flame_hood_line.set_data(hood_flame_pts[:, 0], hood_flame_pts[:, 1])
        flame_side_line.set_data(side_flame_pts[:, 0], side_flame_pts[:, 1])

        trim_local = np.array([[0.04, 0.055], [0.78, 0.055]])
        trim_pts = transform_points(trim_local, rear_center, theta)
        body_trim_line.set_data(trim_pts[:, 0], trim_pts[:, 1])

        rear_tire.center = tuple(rear_center)
        rear_rim.center = tuple(rear_center)
        rear_hub.center = tuple(rear_center)

        front_tire.center = tuple(front_center)
        front_rim.center = tuple(front_center)
        front_hub.center = tuple(front_center)

        rear_axle_dot.set_data([rear_center[0]], [rear_center[1]])

        chassis_rear = rear_center + np.array([0.02, 0.05])
        chassis_front = front_center + np.array([-0.03, 0.02])
        axle_line.set_data([chassis_rear[0], chassis_front[0]],
                           [chassis_rear[1], chassis_front[1]])

        body_anchor_front = transform_points(np.array([[0.54, 0.03]]), rear_center, theta)[0]
        body_anchor_mid = transform_points(np.array([[0.15, 0.03]]), rear_center, theta)[0]
        shock_line_1.set_data([front_center[0], body_anchor_front[0]],
                              [front_center[1] + 0.04, body_anchor_front[1]])
        shock_line_2.set_data([rear_center[0] + 0.02, body_anchor_mid[0]],
                              [rear_center[1] + 0.04, body_anchor_mid[1]])

        com_pos = rear_center + np.array([vis_com_offset * math.cos(theta),
                                          vis_com_offset * math.sin(theta)])
        com_dot.set_data([com_pos[0]], [com_pos[1]])

        x_center = x + 0.25
        ax_car.set_xlim(x_center - 1.15, x_center + 1.35)
        ax_car.set_ylim(-0.05, 1.25)
        ground_line.set_data([x_center - 2.5, x_center + 2.5], [0.0, 0.0])

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
            ground_line, axle_line, shock_line_1, shock_line_2,
            body_patch, shadow_body_patch, window_patch,
            flame_hood_line, flame_side_line, body_trim_line,
            rear_tire, rear_rim, rear_hub,
            front_tire, front_rim, front_hub,
            rear_axle_dot, com_dot,
            ref_text, state_text,
            theta_line, theta_ref_line, tau_line, v_line,
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
