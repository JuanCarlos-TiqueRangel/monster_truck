import math
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    from .car_dynamics import WheelieParams, rk4_step_np
    from .mppi import MPPIConfig, WheelieMPPITorch
    from .rls import nominal_rls_parameters, rls_update
except ImportError:
    from car_dynamics import WheelieParams, rk4_step_np
    from mppi import MPPIConfig, WheelieMPPITorch
    from rls import nominal_rls_parameters, rls_update


def simulate_closed_loop() -> None:
    p = WheelieParams()
    cfg = MPPIConfig()
    controller = WheelieMPPITorch(p, cfg)
    ref = np.array([0.0, 0.0, math.radians(p.pitch_ref), 0.0], dtype=float)
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    tau_prev = 0.0
    tau_cmd = 0.0

    steps = int(p.sim_time / p.sim_dt)
    control_period_steps = max(1, int(cfg.dt / p.sim_dt))
    a_nom = nominal_rls_parameters(p)
    a_rls = a_nom.copy()
    P_rls = 3.0 * np.eye(10)
    plant_has_mismatch = False

    history = np.zeros((steps, 25), dtype=float)
    last_info = {"cost_min": np.nan, "effective_sample_size": np.nan}

    for k in range(steps):
        t = k * p.sim_dt
        state_prev = state.copy()
        if k % control_period_steps == 0:
            tau_cmd, last_info = controller.solve(state_prev, ref, tau_prev, a_rls)
            tau_prev = tau_cmd

        state_next = rk4_step_np(
            state_prev, tau_cmd, p.sim_dt, p, plant_has_mismatch
        )
        a_rls, P_rls, rls_info = rls_update(
            state_prev,
            tau_cmd,
            state_next,
            p.sim_dt,
            a_rls,
            P_rls,
            forgetting_factor=0.999,
            sigma_v_dot=2.0,
            sigma_omega_dot=5.0,
        )

        history[k, :] = np.concatenate(
            [
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
                    ]
                ),
                a_rls,
                np.array(
                    [last_info["cost_min"], last_info["effective_sample_size"]]
                ),
            ]
        )
        state = state_next

    print_results(history, a_rls, a_nom)
    plot_results(history, a_nom)


def print_results(history: np.ndarray, a_rls: np.ndarray, a_nom: np.ndarray) -> None:
    theta_deg = np.rad2deg(history[:, 3])
    print("\n========== Final result ==========")
    print(f"Final theta: {theta_deg[-1]:.2f} deg")
    print(f"Final omega: {history[-1, 4]:.3f} rad/s")
    print(f"Final x:     {history[-1, 1]:.2f} m")
    print(f"Final v:     {history[-1, 2]:.2f} m/s")
    print(f"Final measured v_dot:     {history[-1, 7]: .6f} m/s^2")
    print(f"Final measured omega_dot: {history[-1, 10]: .6f} rad/s^2")
    print(f"Final v_dot error:     {history[-1, 9]: .6f}")
    print(f"Final omega_dot error: {history[-1, 12]: .6f}")

    names = [
        "b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
        "a_g", "a_tau", "a_omega", "a_v", "a_0",
    ]
    print("\n========== RLS coefficients ==========")
    for i, name in enumerate(names):
        print(f"{name:10s} learned: {a_rls[i]: .6f} | nominal: {a_nom[i]: .6f}")
    print("==================================\n")


def plot_results(history: np.ndarray, a_nom: np.ndarray) -> None:
    t = history[:, 0]
    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(12, 14))

    axs[0].plot(t, np.rad2deg(history[:, 3]), label="theta")
    axs[0].plot(t, np.rad2deg(history[:, 6]), "--", label="theta_ref")
    axs[0].set_ylabel("pitch [deg]")
    axs[0].legend()
    axs[1].plot(t, history[:, 4])
    axs[1].set_ylabel("omega [rad/s]")
    axs[2].plot(t, history[:, 5])
    axs[2].set_ylabel("tau [Nm]")

    axs[3].plot(t, history[:, 7], label="measured v_dot")
    axs[3].plot(t, history[:, 8], "--", label="RLS v_dot")
    axs[3].plot(t, history[:, 9], ":", label="v_dot error")
    axs[3].set_ylabel("v_dot [m/s^2]")
    axs[3].legend()

    axs[4].plot(t, history[:, 10], label="measured omega_dot")
    axs[4].plot(t, history[:, 11], "--", label="RLS omega_dot")
    axs[4].plot(t, history[:, 12], ":", label="omega_dot error")
    axs[4].set_ylabel("omega_dot [rad/s^2]")
    axs[4].legend()

    labels = [
        "b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
        "a_g", "a_tau", "a_omega", "a_v", "a_0",
    ]
    for i, label in enumerate(labels):
        axs[5].plot(t, history[:, 13 + i], label=label)
        axs[5].axhline(a_nom[i], linestyle="--", linewidth=0.8)
    axs[5].set_ylabel("RLS coeffs")
    axs[5].legend(ncol=5, fontsize=8)

    axs[6].plot(t, history[:, 23], label="min cost")
    axs[6].plot(t, history[:, 24], label="ESS")
    axs[6].set_ylabel("MPPI debug")
    axs[6].set_xlabel("time [s]")
    axs[6].legend()

    for ax in axs:
        ax.grid(True)
    fig.suptitle("PyTorch MPPI with Two-Output RLS")
    fig.tight_layout()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "wheelie_mppi_torch_rls_nmpc_cost.png")
    fig.savefig(out, dpi=200)
    print("Saved figure:", out)
    plt.show()


if __name__ == "__main__":
    simulate_closed_loop()
