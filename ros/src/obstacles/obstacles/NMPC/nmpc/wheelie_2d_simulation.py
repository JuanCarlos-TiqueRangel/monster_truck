import math

import numpy as np
import matplotlib.pyplot as plt

from params import WheelieParams, MPCConfig
from rls import nominal_rls_parameters, rls_update
from nmpc import NMPC


# ============================================================
# Plant dynamics for plain Python simulation
# ============================================================

def continuous_dynamics_np(
    state: np.ndarray,
    tau: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    x, v, theta, omega = state

    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v
    theta_dot = omega
    omega_dot = (-tau + p.m * p.g * p.l * np.cos(theta)) / p.I_eff

    if plant_has_mismatch:
        # Artificial unknown dynamics for testing RLS.
        # For real MuJoCo/hardware, this is replaced by measured next state.
        v_dot += 0.50 * np.sin(theta) + 0.25 * tau * np.cos(theta)
        omega_dot += 0.5 * omega + 5.0 * v + 3.0 * np.sin(theta)

    return np.array([x_dot, v_dot, theta_dot, omega_dot], dtype=float)


def rk4_step_np(
    state: np.ndarray,
    tau: float,
    dt: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    k1 = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
    k2 = continuous_dynamics_np(state + 0.5 * dt * k1, tau, p, plant_has_mismatch)
    k3 = continuous_dynamics_np(state + 0.5 * dt * k2, tau, p, plant_has_mismatch)
    k4 = continuous_dynamics_np(state + dt * k3, tau, p, plant_has_mismatch)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# Simulation
# ============================================================

def simulate_closed_loop() -> None:
    p = WheelieParams()
    mpc_cfg = MPCConfig()
    controller = NMPC(p, mpc_cfg)

    theta_ref = math.radians(p.pitch_ref)
    ref = np.array([0.0, 0.0, theta_ref, 0.0], dtype=float)

    state = np.array([0.0, 0.0, math.radians(0.0), 0.0], dtype=float)
    tau_prev = 0.0
    tau_cmd = 0.0

    sim_dt = p.sim_dt
    steps = int(p.sim_time / sim_dt)
    mpc_period_steps = max(1, int(mpc_cfg.dt / sim_dt))

    forgetting_factor = 0.999
    initial_covariance = 3.0
    derivative_alpha = 0.0
    clip_parameters = False

    sigma_v_dot = 2.0
    sigma_omega_dot = 5.0

    a_nom = nominal_rls_parameters(p)
    a_rls = a_nom.copy()
    P_rls = initial_covariance * np.eye(a_nom.shape[0])   # 10 RLS params
    filtered_y_dot = None
    n_rls = a_nom.shape[0]

    plant_has_mismatch = True

    # Columns:
    # t, x, v, theta, omega, tau, theta_ref,
    # v_dot_meas, v_dot_hat, v_dot_err,
    # omega_dot_meas, omega_dot_hat, omega_dot_err,   (13 logs)
    # then the RLS coefficients (n_rls = 11)
    history = np.zeros((steps, 13 + n_rls))

    for k in range(steps):
        t = k * sim_dt
        state_prev = state.copy()

        if k % mpc_period_steps == 0:
            tau_cmd, info = controller.solve(state_prev, ref, tau_prev, a_rls)
            tau_prev = tau_cmd

            if not info["success"]:
                print(f"[WARN] NMPC failed at t={t:.2f}, using fallback torque")

        state_next = rk4_step_np(
            state_prev,
            tau_cmd,
            sim_dt,
            p,
            plant_has_mismatch=plant_has_mismatch,
        )

        a_rls, P_rls, filtered_y_dot, rls_info = rls_update(
            state_prev=state_prev,
            tau=tau_cmd,
            state_next=state_next,
            dt=sim_dt,
            a=a_rls,
            P=P_rls,
            filtered_y_dot=filtered_y_dot,
            forgetting_factor=forgetting_factor,
            derivative_alpha=derivative_alpha,
            sigma_v_dot=sigma_v_dot,
            sigma_omega_dot=sigma_omega_dot,
            clip_parameters=clip_parameters,
            p=p,
        )

        history[k, :] = np.concatenate([
            np.array(
                [
                    t,
                    state_prev[0],
                    state_prev[1],
                    state_prev[2],
                    state_prev[3],
                    tau_cmd,
                    ref[2],
                    rls_info["v_dot_measured"],
                    rls_info["v_dot_hat"],
                    rls_info["v_dot_error"],
                    rls_info["omega_dot_measured"],
                    rls_info["omega_dot_hat"],
                    rls_info["omega_dot_error"],
                ],
                dtype=float,
            ),
            a_rls,
        ])

        state = state_next

    print_results(history, a_rls, a_nom)
    plot_results(history, a_nom)


def print_results(history: np.ndarray, a_rls: np.ndarray, a_nom: np.ndarray) -> None:
    theta_deg = np.rad2deg(history[:, 3])
    omega = history[:, 4]
    x = history[:, 1]
    v = history[:, 2]

    print("\n========== Final result ==========")
    print(f"Final theta: {theta_deg[-1]:.2f} deg")
    print(f"Final omega: {omega[-1]:.3f} rad/s")
    print(f"Final x:     {x[-1]:.2f} m")
    print(f"Final v:     {v[-1]:.2f} m/s")
    print(f"Final v_dot error:     {history[-1, 9]: .6f}")
    print(f"Final omega_dot error: {history[-1, 12]: .6f}")

    names = [
        "b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
        "a_g", "a_tau", "a_omega", "a_v", "a_0",
    ]

    print("\n========== RLS coefficients ==========")
    print("v_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0")
    print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    for i, name in enumerate(names):
        print(f"{name:10s} learned: {a_rls[i]: .6f} | nominal: {a_nom[i]: .6f}")
    print("==================================\n")


def plot_results(history: np.ndarray, a_nom: np.ndarray) -> None:
    t = history[:, 0]
    theta_deg = np.rad2deg(history[:, 3])
    theta_ref_deg = np.rad2deg(history[:, 6])
    omega = history[:, 4]
    tau = history[:, 5]

    v_dot_measured = history[:, 7]
    v_dot_hat = history[:, 8]
    v_dot_error = history[:, 9]

    omega_dot_measured = history[:, 10]
    omega_dot_hat = history[:, 11]
    omega_dot_error = history[:, 12]

    coeffs = history[:, 13:]

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(12, 12))

    axs[0].plot(t, theta_deg, label="theta")
    axs[0].plot(t, theta_ref_deg, linestyle="--", label="theta_ref")
    axs[0].set_ylabel("pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, omega)
    axs[1].set_ylabel("omega [rad/s]")
    axs[1].grid(True)

    axs[2].plot(t, tau)
    axs[2].set_ylabel("tau [Nm]")
    axs[2].grid(True)

    axs[3].plot(t, v_dot_measured, label="measured v_dot")
    axs[3].plot(t, v_dot_hat, linestyle="--", label="RLS v_dot")
    axs[3].plot(t, v_dot_error, linestyle=":", label="v_dot error")
    axs[3].set_ylabel("v_dot [m/s^2]")
    axs[3].grid(True)
    axs[3].legend()

    axs[4].plot(t, omega_dot_measured, label="measured omega_dot")
    axs[4].plot(t, omega_dot_hat, linestyle="--", label="RLS omega_dot")
    axs[4].plot(t, omega_dot_error, linestyle=":", label="omega_dot error")
    axs[4].set_ylabel("omega_dot [rad/s^2]")
    axs[4].grid(True)
    axs[4].legend()

    labels = [
        "b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
        "a_g", "a_tau", "a_omega", "a_v", "a_0",
    ]
    for i, label in enumerate(labels):
        axs[5].plot(t, coeffs[:, i], label=label)
        axs[5].axhline(a_nom[i], linestyle="--", linewidth=0.8)
    axs[5].set_ylabel("RLS coeffs")
    axs[5].set_xlabel("time [s]")
    axs[5].grid(True)
    axs[5].legend(ncol=5, fontsize=8)

    fig.suptitle("Wheelie NMPC + Two-Output Full-Dynamics RLS")
    fig.tight_layout()
    plt.show()

    fig.savefig("images/wheelie_fullDynamics_2.png", dpi=200)
    print("Saved figure:", "images/wheelie_fullDynamics_2.png")

if __name__ == "__main__":
    simulate_closed_loop()
