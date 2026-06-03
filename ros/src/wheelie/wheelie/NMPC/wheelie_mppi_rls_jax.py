import math
import os
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


# ============================================================
# Parameters
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1
    l: float = 0.2
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.085
    g: float = 9.81
    c_v: float = 9.0

    tau_min: float = -8.0
    tau_max: float = 12.0

    pitch_ref: float = 80.0
    sim_time: float = 5.0
    sim_dt: float = 0.1

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPPIConfig:
    dt: float = 0.1
    N: int = 30
    num_samples: int = 4096

    # MPPI parameters
    temperature: float = 20.6
    noise_sigma: float = 3.0

    # Same cost weights as the original NMPCConfig
    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 1300.0
    q_omega: float = 15.0

    r_tau: float = 0.1
    r_dtau: float = 0.01

    q_terminal_theta: float = 700.0
    q_terminal_omega: float = 100.0


# ============================================================
# Plant dynamics used for simulation
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
# Standard two-output RLS, no derivative filter
# ============================================================

def nominal_rls_parameters(p: WheelieParams) -> np.ndarray:
    return np.array(
        [
            1.0 / (p.m * p.r),  # tau -> v_dot
            -p.c_v,             # v -> v_dot
            0.0,                # |v|v -> v_dot
            0.0,                # tau*cos(theta) -> v_dot
            0.0,                # bias -> v_dot
            p.m * p.g * p.l / p.I_eff,  # cos(theta) -> omega_dot
            -1.0 / p.I_eff,             # tau -> omega_dot
            0.0,                        # omega -> omega_dot
            0.0,                        # v -> omega_dot
            0.0,                        # bias -> omega_dot
        ],
        dtype=float,
    )


def rls_update(
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    a: np.ndarray,
    P: np.ndarray,
    forgetting_factor: float = 0.999,
    sigma_v_dot: float = 2.0,
    sigma_omega_dot: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    # Raw finite-difference derivatives. No filtering.
    y = np.array(
        [
            (float(v_next) - float(v_prev)) / dt,
            (float(omega_next) - float(omega_prev)) / dt,
        ],
        dtype=float,
    )

    phi_v = np.array(
        [
            tau,
            v_prev,
            abs(v_prev) * v_prev,
            tau * np.cos(theta_prev),
            1.0,
        ],
        dtype=float,
    )

    phi_w = np.array(
        [
            np.cos(theta_prev),
            tau,
            omega_prev,
            v_prev,
            1.0,
        ],
        dtype=float,
    )

    H = np.zeros((2, 10), dtype=float)
    H[0, 0:5] = phi_v
    H[1, 5:10] = phi_w

    y_hat_before = H @ a
    error = y - y_hat_before

    R = np.diag([sigma_v_dot**2, sigma_omega_dot**2])
    P_pred = P / forgetting_factor
    S = H @ P_pred @ H.T + R

    if np.linalg.cond(S) > 1e12:
        info = {
            "v_dot_measured": float(y[0]),
            "omega_dot_measured": float(y[1]),
            "v_dot_hat": float(y_hat_before[0]),
            "omega_dot_hat": float(y_hat_before[1]),
            "v_dot_error": float(y[0] - y_hat_before[0]),
            "omega_dot_error": float(y[1] - y_hat_before[1]),
            "skipped": True,
        }
        return a, P, info

    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    # Joseph-form covariance update for numerical stability.
    I = np.eye(10)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    y_hat_after = H @ a
    info = {
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat_after[0]),
        "omega_dot_hat": float(y_hat_after[1]),
        "v_dot_error": float(y[0] - y_hat_after[0]),
        "omega_dot_error": float(y[1] - y_hat_after[1]),
        "skipped": False,
    }
    return a, P, info


# ============================================================
# JAX dynamics used by MPPI
# ============================================================

def rls_dynamics_jax(state: jnp.ndarray, tau: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    x = state[..., 0]
    v = state[..., 1]
    theta = state[..., 2]
    omega = state[..., 3]

    x_dot = v
    v_dot = (
        a[0] * tau
        + a[1] * v
        + a[2] * jnp.abs(v) * v
        + a[3] * tau * jnp.cos(theta)
        + a[4]
    )
    theta_dot = omega
    omega_dot = (
        a[5] * jnp.cos(theta)
        + a[6] * tau
        + a[7] * omega
        + a[8] * v
        + a[9]
    )

    return jnp.stack([x_dot, v_dot, theta_dot, omega_dot], axis=-1)


def rk4_step_jax(state: jnp.ndarray, tau: jnp.ndarray, dt: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
    k1 = rls_dynamics_jax(state, tau, a)
    k2 = rls_dynamics_jax(state + 0.5 * dt * k1, tau, a)
    k3 = rls_dynamics_jax(state + 0.5 * dt * k2, tau, a)
    k4 = rls_dynamics_jax(state + dt * k3, tau, a)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@jax.jit
def mppi_update_jax(
    state: jnp.ndarray,
    ref: jnp.ndarray,
    tau_prev: jnp.ndarray,
    a_rls: jnp.ndarray,
    u_nominal: jnp.ndarray,
    noise: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Standard MPPI update using the same cost structure as the original NMPC.

    Stage cost:
        e_k^T Q e_k + r_tau*tau_k^2 + r_dtau*(tau_k - tau_{k-1})^2

    Terminal cost:
        e_N^T Qf e_N

    params = [dt, tau_min, tau_max, temperature,
              q_x, q_v, q_theta, q_omega,
              r_tau, r_dtau,
              q_terminal_theta, q_terminal_omega]
    """
    (
        dt,
        tau_min,
        tau_max,
        temperature,
        q_x,
        q_v,
        q_theta,
        q_omega,
        r_tau,
        r_dtau,
        q_terminal_theta,
        q_terminal_omega,
    ) = params

    K = noise.shape[0]
    u_samples = jnp.clip(u_nominal[None, :] + noise, tau_min, tau_max)

    states0 = jnp.repeat(state[None, :], K, axis=0)
    costs0 = jnp.zeros((K,), dtype=state.dtype)
    tau_prev0 = jnp.full((K,), tau_prev, dtype=state.dtype)

    def scan_step(carry, tau_k):
        states, costs, tau_previous = carry

        e = states - ref[None, :]
        du = tau_k - tau_previous

        stage_cost = (
            q_x * e[:, 0] ** 2
            + q_v * e[:, 1] ** 2
            + q_theta * e[:, 2] ** 2
            + q_omega * e[:, 3] ** 2
            + r_tau * tau_k**2
            + r_dtau * du**2
        )

        next_states = rk4_step_jax(states, tau_k, dt, a_rls)
        return (next_states, costs + stage_cost, tau_k), None

    (final_states, running_costs, _), _ = lax.scan(
        scan_step,
        (states0, costs0, tau_prev0),
        u_samples.T,
    )

    eN = final_states - ref[None, :]
    terminal_cost = (
        q_x * eN[:, 0] ** 2
        + q_v * eN[:, 1] ** 2
        + q_terminal_theta * eN[:, 2] ** 2
        + q_terminal_omega * eN[:, 3] ** 2
    )

    costs = jnp.nan_to_num(
        running_costs + terminal_cost,
        nan=1.0e20,
        posinf=1.0e20,
        neginf=1.0e20,
    )

    beta = jnp.min(costs)
    weights = jax.nn.softmax(-(costs - beta) / jnp.maximum(temperature, 1.0e-9))

    # Direct weighted average of sampled control sequences. No filtering.
    u_new = jnp.sum(weights[:, None] * u_samples, axis=0)
    u_new = jnp.clip(u_new, tau_min, tau_max)

    return u_new, costs, weights


class WheelieMPPIJAX:
    def __init__(self, p: WheelieParams, cfg: MPPIConfig, seed: int = 0):
        self.p = p
        self.cfg = cfg
        self.key = jax.random.PRNGKey(seed)
        self.u_nominal = np.zeros(cfg.N, dtype=float)

        self.params = jnp.array(
            [
                cfg.dt,
                p.tau_min,
                p.tau_max,
                cfg.temperature,
                cfg.q_x,
                cfg.q_v,
                cfg.q_theta,
                cfg.q_omega,
                cfg.r_tau,
                cfg.r_dtau,
                cfg.q_terminal_theta,
                cfg.q_terminal_omega,
            ],
            dtype=jnp.float64,
        )

        print("JAX devices:", jax.devices())

    def solve(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        tau_prev: float,
        a_rls: np.ndarray,
    ) -> tuple[float, dict]:
        cfg = self.cfg
        p = self.p

        self.key, subkey = jax.random.split(self.key)
        noise = cfg.noise_sigma * jax.random.normal(
            subkey,
            shape=(cfg.num_samples, cfg.N),
            dtype=jnp.float64,
        )

        u_new_jax, costs_jax, weights_jax = mppi_update_jax(
            jnp.asarray(state, dtype=jnp.float64),
            jnp.asarray(ref, dtype=jnp.float64),
            jnp.asarray(tau_prev, dtype=jnp.float64),
            jnp.asarray(a_rls, dtype=jnp.float64),
            jnp.asarray(self.u_nominal, dtype=jnp.float64),
            noise,
            self.params,
        )

        u_new = np.asarray(u_new_jax.block_until_ready(), dtype=float)
        costs = np.asarray(costs_jax, dtype=float)
        weights = np.asarray(weights_jax, dtype=float)

        tau_cmd = float(np.clip(u_new[0], p.tau_min, p.tau_max))

        # Receding horizon shift. This is not a filter; it is standard MPC/MPPI warm start.
        self.u_nominal = np.concatenate([u_new[1:], np.array([0.0])])

        info = {
            "cost_min": float(np.min(costs)),
            "cost_mean": float(np.mean(costs)),
            "effective_sample_size": float(1.0 / np.sum(weights**2)),
            "tau_cmd": tau_cmd,
        }
        return tau_cmd, info


# ============================================================
# Simulation
# ============================================================

def simulate_closed_loop() -> None:
    p = WheelieParams()
    cfg = MPPIConfig()
    controller = WheelieMPPIJAX(p, cfg)

    theta_ref = math.radians(p.pitch_ref)
    ref = np.array([0.0, 0.0, theta_ref, 0.0], dtype=float)

    state = np.array([0.0, 0.0, math.radians(0.0), 0.0], dtype=float)
    tau_prev = 0.0
    tau_cmd = 0.0

    steps = int(p.sim_time / p.sim_dt)
    control_period_steps = max(1, int(cfg.dt / p.sim_dt))

    a_nom = nominal_rls_parameters(p)
    a_rls = a_nom.copy()
    P_rls = 3.0 * np.eye(10)

    plant_has_mismatch = True

    # Columns:
    # t, x, v, theta, omega, tau, theta_ref,
    # v_dot_meas, v_dot_hat, v_dot_err,
    # omega_dot_meas, omega_dot_hat, omega_dot_err,
    # ten RLS coefficients,
    # MPPI cost_min, MPPI ESS
    history = np.zeros((steps, 25), dtype=float)
    last_info = {"cost_min": np.nan, "effective_sample_size": np.nan}

    for k in range(steps):
        t = k * p.sim_dt
        state_prev = state.copy()

        if k % control_period_steps == 0:
            tau_cmd, last_info = controller.solve(state_prev, ref, tau_prev, a_rls)
            tau_prev = tau_cmd

        state_next = rk4_step_np(
            state_prev,
            tau_cmd,
            p.sim_dt,
            p,
            plant_has_mismatch=plant_has_mismatch,
        )

        a_rls, P_rls, rls_info = rls_update(
            state_prev=state_prev,
            tau=tau_cmd,
            state_next=state_next,
            dt=p.sim_dt,
            a=a_rls,
            P=P_rls,
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
                    ],
                    dtype=float,
                ),
                a_rls,
                np.array([last_info["cost_min"], last_info["effective_sample_size"]], dtype=float),
            ]
        )

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
        "b_tau",
        "b_v",
        "b_abs_v",
        "b_tau_cos",
        "b_0",
        "a_g",
        "a_tau",
        "a_omega",
        "a_v",
        "a_0",
    ]

    print("\n========== RLS coefficients ==========")
    print("v_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*cos(theta) + b_0")
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

    coeffs = history[:, 13:23]
    cost_min = history[:, 23]
    ess = history[:, 24]

    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(12, 14))

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
        "b_tau",
        "b_v",
        "b_abs_v",
        "b_tau_cos",
        "b_0",
        "a_g",
        "a_tau",
        "a_omega",
        "a_v",
        "a_0",
    ]
    for i, label in enumerate(labels):
        axs[5].plot(t, coeffs[:, i], label=label)
        axs[5].axhline(a_nom[i], linestyle="--", linewidth=0.8)
    axs[5].set_ylabel("RLS coeffs")
    axs[5].grid(True)
    axs[5].legend(ncol=5, fontsize=8)

    axs[6].plot(t, cost_min, label="min cost")
    axs[6].plot(t, ess, label="ESS")
    axs[6].set_ylabel("MPPI debug")
    axs[6].set_xlabel("time [s]")
    axs[6].grid(True)
    axs[6].legend()

    fig.suptitle("Standard MPPI with NMPC Cost + Two-Output RLS")
    fig.tight_layout()

    os.makedirs("images", exist_ok=True)
    out = "images/wheelie_mppi_jax_rls_nmpc_cost.png"
    fig.savefig(out, dpi=200)
    print("Saved figure:", out)
    plt.show()


if __name__ == "__main__":
    simulate_closed_loop()