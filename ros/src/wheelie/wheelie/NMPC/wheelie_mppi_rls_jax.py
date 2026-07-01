import math
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
from jax import lax
import matplotlib
# Use an interactive backend when a display is available so the figure window
# actually pops up when the script finishes. Fall back to headless Agg (e.g. on
# a remote/CI box with no display) so that saving the PNG still works.
if os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
    pass  # let matplotlib pick its default interactive backend (Qt/Tk/MacOSX)
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)


# ============================================================
# Parameters (unchanged)
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
    tau_max: float = 8.0
    pitch_ref: float = 80.0
    sim_time: float = 5.0
    sim_dt: float = 0.1

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPPIConfig:
    dt: float = 0.05
    N: int = 30
    num_samples: int = 1024
    temperature: float = 20.6
    noise_sigma: float = 3.0
    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 1300.0
    q_omega: float = 15.0
    r_tau: float = 0.1
    r_dtau: float = 0.01
    q_terminal_theta: float = 700.0
    q_terminal_omega: float = 100.0


def nominal_rls_parameters(p: WheelieParams) -> np.ndarray:
    return np.array([
        1.0 / (p.m * p.r), -p.c_v, 0.0, 0.0, 0.0,
        p.m * p.g * p.l / p.I_eff, -1.0 / p.I_eff, 0.0, 0.0, 0.0,
    ], dtype=float)


# ============================================================
# Build the single fused, jitted simulator.
# All scalar config is closed over (baked as compile-time constants).
# ============================================================
def build_simulator(p: WheelieParams, cfg: MPPIConfig,
                    forgetting_factor: float = 0.999,
                    sigma_v_dot: float = 2.0,
                    sigma_omega_dot: float = 5.0,
                    compute_dtype=jnp.float64,
                    rollout_order: int = 4):
    # --- static scalars ---
    m, l, r, g, c_v = p.m, p.l, p.r, p.g, p.c_v
    I_eff = p.I_eff
    tau_min, tau_max = p.tau_min, p.tau_max
    sim_dt = p.sim_dt
    steps = int(p.sim_time / p.sim_dt)
    control_period_steps = max(1, int(cfg.dt / p.sim_dt))

    dt = cfg.dt
    N = cfg.N
    num_samples = cfg.num_samples
    noise_sigma = cfg.noise_sigma
    temp_safe = max(cfg.temperature, 1.0e-9)
    q_x, q_v, q_theta, q_omega = cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega
    r_tau, r_dtau = cfg.r_tau, cfg.r_dtau
    q_tT, q_tW = cfg.q_terminal_theta, cfg.q_terminal_omega

    R = jnp.diag(jnp.array([sigma_v_dot**2, sigma_omega_dot**2], dtype=jnp.float64))
    I10 = jnp.eye(10, dtype=jnp.float64)
    cdt = compute_dtype  # dtype for the heavy MPPI rollout

    # ---------- true plant (with mismatch), single 4-vector ----------
    def plant_dynamics(s, tau):
        v, theta, omega = s[1], s[2], s[3]
        x_dot = v
        v_dot = tau / (m * r) - c_v * v + 0.50 * jnp.sin(theta) + 0.25 * tau * jnp.cos(theta)
        theta_dot = omega
        omega_dot = (-tau + m * g * l * jnp.cos(theta)) / I_eff + 0.5 * omega + 5.0 * v + 3.0 * jnp.sin(theta)
        return jnp.stack([x_dot, v_dot, theta_dot, omega_dot])

    def plant_rk4(s, tau):
        k1 = plant_dynamics(s, tau)
        k2 = plant_dynamics(s + 0.5 * sim_dt * k1, tau)
        k3 = plant_dynamics(s + 0.5 * sim_dt * k2, tau)
        k4 = plant_dynamics(s + sim_dt * k3, tau)
        return s + (sim_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ---------- learned RLS model (batched over samples) ----------
    def model_dynamics(states, tau, a):
        v = states[:, 1]; theta = states[:, 2]; omega = states[:, 3]
        x_dot = v
        v_dot = a[0]*tau + a[1]*v + a[2]*jnp.abs(v)*v + a[3]*tau*jnp.cos(theta) + a[4]
        theta_dot = omega
        omega_dot = a[5]*jnp.cos(theta) + a[6]*tau + a[7]*omega + a[8]*v + a[9]
        return jnp.stack([x_dot, v_dot, theta_dot, omega_dot], axis=-1)

    def model_rk4(states, tau, a):
        if rollout_order == 1:           # explicit Euler (1 dynamics eval)
            k1 = model_dynamics(states, tau, a)
            return states + dt * k1
        if rollout_order == 2:           # Heun / RK2 (2 dynamics evals)
            k1 = model_dynamics(states, tau, a)
            k2 = model_dynamics(states + dt * k1, tau, a)
            return states + (dt / 2.0) * (k1 + k2)
        k1 = model_dynamics(states, tau, a)          # classic RK4 (4 evals)
        k2 = model_dynamics(states + 0.5*dt*k1, tau, a)
        k3 = model_dynamics(states + 0.5*dt*k2, tau, a)
        k4 = model_dynamics(states + dt*k3, tau, a)
        return states + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    # ---------- MPPI core: returns u_new, cost_min, ess ----------
    def mppi_core(state, ref, tau_prev, a, u_nominal, noise):
        K = num_samples
        u_samples = jnp.clip(u_nominal[None, :] + noise, tau_min, tau_max)  # (K, N)

        # cast the heavy rollout to compute_dtype (float64 by default)
        st = state.astype(cdt)
        rf = ref.astype(cdt)
        ac = a.astype(cdt)
        us = u_samples.astype(cdt)
        tp = jnp.asarray(tau_prev, cdt)

        states0 = jnp.broadcast_to(st, (K, 4))
        costs0 = jnp.zeros((K,), dtype=cdt)
        tau_prev0 = jnp.full((K,), tp, dtype=cdt)

        def rollout(carry, tau_k):
            states, costs, tau_previous = carry
            e = states - rf
            du = tau_k - tau_previous
            stage = (q_x*e[:, 0]**2 + q_v*e[:, 1]**2 + q_theta*e[:, 2]**2
                     + q_omega*e[:, 3]**2 + r_tau*tau_k**2 + r_dtau*du**2)
            states_next = model_rk4(states, tau_k, ac)
            return (states_next, costs + stage, tau_k), None

        (final_states, running, _), _ = lax.scan(rollout, (states0, costs0, tau_prev0), us.T)

        eN = final_states - rf
        terminal = (q_x*eN[:, 0]**2 + q_v*eN[:, 1]**2 + q_tT*eN[:, 2]**2 + q_tW*eN[:, 3]**2)
        costs = jnp.nan_to_num(running + terminal, nan=1.0e20, posinf=1.0e20, neginf=1.0e20)

        beta = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - beta) / temp_safe)
        u_new = jnp.sum(weights[:, None] * us, axis=0)
        u_new = jnp.clip(u_new, tau_min, tau_max).astype(jnp.float64)

        cost_min = jnp.min(costs).astype(jnp.float64)
        ess = (1.0 / jnp.sum(weights**2)).astype(jnp.float64)
        return u_new, cost_min, ess

    # ---------- two-output RLS update (float64), returns a, P, info(6,) ----------
    def rls_update(s_prev, tau, s_next, a, P):
        v_prev, theta_prev, omega_prev = s_prev[1], s_prev[2], s_prev[3]
        v_next, omega_next = s_next[1], s_next[3]

        y = jnp.array([(v_next - v_prev) / sim_dt, (omega_next - omega_prev) / sim_dt])

        phi_v = jnp.array([tau, v_prev, jnp.abs(v_prev) * v_prev, tau * jnp.cos(theta_prev), 1.0])
        phi_w = jnp.array([jnp.cos(theta_prev), tau, omega_prev, v_prev, 1.0])
        z5 = jnp.zeros(5)
        H = jnp.stack([jnp.concatenate([phi_v, z5]), jnp.concatenate([z5, phi_w])])  # (2,10)

        y_hat_before = H @ a
        error = y - y_hat_before
        P_pred = P / forgetting_factor
        S = H @ P_pred @ H.T + R

        def apply_update(_):
            Kg = P_pred @ H.T @ jnp.linalg.inv(S)
            a_new = a + Kg @ error
            P_new = (I10 - Kg @ H) @ P_pred @ (I10 - Kg @ H).T + Kg @ R @ Kg.T
            P_new = 0.5 * (P_new + P_new.T)
            yh = H @ a_new
            info = jnp.array([y[0], yh[0], y[0]-yh[0], y[1], yh[1], y[1]-yh[1]])
            return a_new, P_new, info

        def skip_update(_):
            yh = y_hat_before
            info = jnp.array([y[0], yh[0], y[0]-yh[0], y[1], yh[1], y[1]-yh[1]])
            return a, P, info

        well_cond = jnp.linalg.cond(S) <= 1.0e12
        return lax.cond(well_cond, apply_update, skip_update, operand=None)

    # ---------- one simulation step (= scan body) ----------
    def step(carry, k):
        state, tau_app, u_nominal, a_rls, P_rls, key, last_cmin, last_ess = carry

        is_solve = (k % control_period_steps) == 0

        def solve_branch(op):
            key, u_nom, tau_app = op
            key, subkey = jax.random.split(key)
            noise = noise_sigma * jax.random.normal(subkey, (num_samples, N), dtype=jnp.float64)
            u_new, cmin, ess = mppi_core(state, ref, tau_app, a_rls, u_nom, noise)
            tau_cmd = jnp.clip(u_new[0], tau_min, tau_max)
            u_nom_next = jnp.concatenate([u_new[1:], jnp.zeros(1)])
            return key, u_nom_next, tau_cmd, cmin, ess

        def hold_branch(op):
            key, u_nom, tau_app = op
            return key, u_nom, tau_app, last_cmin, last_ess

        key, u_nominal, tau_cmd, cmin, ess = lax.cond(
            is_solve, solve_branch, hold_branch, operand=(key, u_nominal, tau_app))
        tau_app = tau_cmd

        state_next = plant_rk4(state, tau_app)
        a_new, P_new, info = rls_update(state, tau_app, state_next, a_rls, P_rls)

        t = k.astype(jnp.float64) * sim_dt
        log = jnp.concatenate([
            jnp.array([t, state[0], state[1], state[2], state[3], tau_app, ref[2]]),
            info,        # 6
            a_new,       # 10
            jnp.array([cmin, ess]),
        ])
        new_carry = (state_next, tau_app, u_nominal, a_new, P_new, key, cmin, ess)
        return new_carry, log

    # ref is closed over (theta_ref fixed)
    ref = jnp.array([0.0, 0.0, math.radians(p.pitch_ref), 0.0], dtype=jnp.float64)

    @jax.jit
    def run(initial_state, a_nom, P0, key):
        carry0 = (initial_state, jnp.asarray(0.0), jnp.zeros(N),
                  a_nom, P0, key,
                  jnp.asarray(jnp.nan), jnp.asarray(jnp.nan))
        carry_f, logs = lax.scan(step, carry0, jnp.arange(steps))
        a_f = carry_f[3]
        P_f = carry_f[4]
        return logs, a_f, P_f

    return run, steps


# ============================================================
# Driver
# ============================================================
def simulate_closed_loop(do_plot=True, save_npy=True, fast32=False,
                         rollout_order=4, num_samples=None, horizon=None,
                         reps=3):
    p = WheelieParams()
    cfg = MPPIConfig()
    if num_samples is not None:
        cfg.num_samples = num_samples
    if horizon is not None:
        cfg.N = horizon
    dtype = jnp.float32 if fast32 else jnp.float64
    run, steps = build_simulator(p, cfg, compute_dtype=dtype, rollout_order=rollout_order)

    initial_state = jnp.array([0.0, 0.0, math.radians(0.0), 0.0], dtype=jnp.float64)
    a_nom = jnp.asarray(nominal_rls_parameters(p))
    P0 = 3.0 * jnp.eye(10, dtype=jnp.float64)
    key = jax.random.PRNGKey(0)

    print(f"JAX devices: {jax.devices()} | dtype={dtype.__name__} "
          f"rollout_order={rollout_order} samples={cfg.num_samples} N={cfg.N}")

    # --- compile + first execution ---
    t0 = time.perf_counter()
    logs, a_f, P_f = run(initial_state, a_nom, P0, key)
    logs.block_until_ready()
    t1 = time.perf_counter()

    # --- pure execution (already compiled), best of `reps` ---
    best = float("inf")
    for _ in range(reps):
        t2 = time.perf_counter()
        logs2, _, _ = run(initial_state, a_nom, P0, key)
        logs2.block_until_ready()
        best = min(best, time.perf_counter() - t2)

    print(f"[OPT] compile + first run: {t1 - t0:.4f} s")
    print(f"[OPT] run only (compiled): {best*1000:.3f} ms  ({best/steps*1000:.3f} ms / step)")

    history = np.asarray(logs)
    a_rls = np.asarray(a_f)
    a_nom_np = nominal_rls_parameters(p)

    if save_npy:
        suffix = "_f32" if fast32 else ""
        np.save(f"hist_optimized{suffix}.npy", history)
        np.save(f"arls_optimized{suffix}.npy", a_rls)

    if do_plot:
        print_results(history, a_rls, a_nom_np)
        plot_results(history, a_nom_np, fast32)
    return history, a_rls


def print_results(history, a_rls, a_nom):
    theta_deg = np.rad2deg(history[:, 3]); omega = history[:, 4]
    x = history[:, 1]; v = history[:, 2]
    print("\n========== Final result ==========")
    print(f"Final theta: {theta_deg[-1]:.2f} deg")
    print(f"Final omega: {omega[-1]:.3f} rad/s")
    print(f"Final x:     {x[-1]:.2f} m")
    print(f"Final v:     {v[-1]:.2f} m/s")
    print(f"Final v_dot error:     {history[-1, 9]: .6f}")
    print(f"Final omega_dot error: {history[-1, 12]: .6f}")
    names = ["b_tau","b_v","b_abs_v","b_tau_cos","b_0","a_g","a_tau","a_omega","a_v","a_0"]
    print("\n========== RLS coefficients ==========")
    for i, name in enumerate(names):
        print(f"{name:10s} learned: {a_rls[i]: .6f} | nominal: {a_nom[i]: .6f}")
    print("==================================\n")


def plot_results(history, a_nom, fast32=False):
    t = history[:, 0]
    theta_deg = np.rad2deg(history[:, 3]); theta_ref_deg = np.rad2deg(history[:, 6])
    omega = history[:, 4]; tau = history[:, 5]
    v_dot_measured = history[:, 7]; v_dot_hat = history[:, 8]; v_dot_error = history[:, 9]
    omega_dot_measured = history[:, 10]; omega_dot_hat = history[:, 11]; omega_dot_error = history[:, 12]
    coeffs = history[:, 13:23]; cost_min = history[:, 23]; ess = history[:, 24]
    fig, axs = plt.subplots(7, 1, sharex=True, figsize=(12, 14))
    axs[0].plot(t, theta_deg, label="theta"); axs[0].plot(t, theta_ref_deg, "--", label="theta_ref")
    axs[0].set_ylabel("pitch [deg]"); axs[0].grid(True); axs[0].legend()
    axs[1].plot(t, omega); axs[1].set_ylabel("omega [rad/s]"); axs[1].grid(True)
    axs[2].plot(t, tau); axs[2].set_ylabel("tau [Nm]"); axs[2].grid(True)
    axs[3].plot(t, v_dot_measured, label="measured v_dot"); axs[3].plot(t, v_dot_hat, "--", label="RLS v_dot")
    axs[3].plot(t, v_dot_error, ":", label="v_dot error"); axs[3].set_ylabel("v_dot [m/s^2]"); axs[3].grid(True); axs[3].legend()
    axs[4].plot(t, omega_dot_measured, label="measured omega_dot"); axs[4].plot(t, omega_dot_hat, "--", label="RLS omega_dot")
    axs[4].plot(t, omega_dot_error, ":", label="omega_dot error"); axs[4].set_ylabel("omega_dot [rad/s^2]"); axs[4].grid(True); axs[4].legend()
    labels = ["b_tau","b_v","b_abs_v","b_tau_cos","b_0","a_g","a_tau","a_omega","a_v","a_0"]
    for i, label in enumerate(labels):
        axs[5].plot(t, coeffs[:, i], label=label); axs[5].axhline(a_nom[i], linestyle="--", linewidth=0.8)
    axs[5].set_ylabel("RLS coeffs"); axs[5].grid(True); axs[5].legend(ncol=5, fontsize=8)
    axs[6].plot(t, cost_min, label="min cost"); axs[6].plot(t, ess, label="ESS")
    axs[6].set_ylabel("MPPI debug"); axs[6].set_xlabel("time [s]"); axs[6].grid(True); axs[6].legend()
    fig.suptitle("Fused single-dispatch MPPI + Two-Output RLS" + (" [float32]" if fast32 else ""))
    fig.tight_layout()
    os.makedirs("images", exist_ok=True)
    out = "images/wheelie_mppi_fused.png"
    fig.savefig(out, dpi=200)
    print("Saved figure:", out)
    # Pop up the interactive window (no-op on the headless Agg fallback).
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fused single-dispatch MPPI + RLS wheelie sim. "
                    "Defaults reproduce the original float64/RK4/4096 result exactly.")
    parser.add_argument("--fast32", action="store_true",
                        help="cast the heavy MPPI rollout to float32 (RLS/plant stay float64)")
    parser.add_argument("--rollout", type=int, default=4, choices=[1, 2, 4],
                        help="MPPI prediction integrator: 1=Euler, 2=RK2, 4=RK4 (default). "
                             "Plant integration is always RK4.")
    parser.add_argument("--samples", type=int, default=None,
                        help="number of MPPI rollout samples (default 4096)")
    parser.add_argument("--horizon", type=int, default=None,
                        help="MPPI prediction horizon N (default 30)")
    parser.add_argument("--reps", type=int, default=3,
                        help="timed repetitions of the compiled run (best is reported)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="skip figure generation")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="skip saving history/coeff .npy files")
    args = parser.parse_args()

    simulate_closed_loop(
        do_plot=args.plot,
        save_npy=args.save,
        fast32=args.fast32,
        rollout_order=args.rollout,
        num_samples=args.samples,
        horizon=args.horizon,
        reps=args.reps,
    )