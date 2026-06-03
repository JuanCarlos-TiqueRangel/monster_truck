import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca


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

    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(90.0)

    omega_min: float = -8.0
    omega_max: float = 8.0

    v_min: float = -5.0
    v_max: float = 5.0

    pitch_ref: float = 80.0
    sim_time: float = 5.0
    sim_dt: float = 0.1

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.1
    N: int = 30

    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 300.0
    q_omega: float = 15.0

    r_tau: float = 0.1
    r_dtau: float = 0.01

    q_terminal_theta: float = 100.0
    q_terminal_omega: float = 100.0

    ipopt_max_iter: int = 50


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
# Full-dynamics two-output RLS
# ============================================================

# Model:
#   v_dot     = phi_v^T a_v
#   omega_dot = phi_w^T a_w
#
# Feature vectors:
#   phi_v = [tau, v, |v|v, tau*cos(theta), 1]
#   phi_w = [cos(theta), tau, omega, v, 1]
#
# Stacked parameter vector:
#   a = [a_v(5), a_w(5)]
#
# Block regression:
#   y = [v_dot, omega_dot]
#   y = H a


def nominal_rls_parameters(p: WheelieParams) -> np.ndarray:
    return np.array(
        [
            # v_dot = b_tau*tau + b_v*v + b_quad*|v|v + b_tau_theta*tau*cos(theta) + b_0
            1.0 / (p.m * p.r),
            -p.c_v,
            0.0,
            0.0,
            0.0,

            # omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
            p.m * p.g * p.l / p.I_eff,
            -1.0 / p.I_eff,
            0.0,
            0.0,
            0.0,
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
    filtered_y_dot: np.ndarray | None,
    forgetting_factor: float = 0.999,
    derivative_alpha: float = 0.85,
    sigma_v_dot: float = 2.0,
    sigma_omega_dot: float = 5.0,
    clip_parameters: bool = True,
    p: WheelieParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    # 1) Measured derivatives.
    v_dot_raw = (float(v_next) - float(v_prev)) / dt
    omega_dot_raw = (float(omega_next) - float(omega_prev)) / dt
    y_raw = np.array([v_dot_raw, omega_dot_raw], dtype=float)

    # 2) Filter derivatives.
    if filtered_y_dot is None:
        filtered_y_dot = y_raw.copy()
    else:
        filtered_y_dot = (
            derivative_alpha * filtered_y_dot
            + (1.0 - derivative_alpha) * y_raw
        )

    y = filtered_y_dot.copy()

    # 3) Different feature vectors.
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

    # 4) Block regression matrix H.
    H = np.zeros((2, 10), dtype=float)
    H[0, 0:5] = phi_v
    H[1, 5:10] = phi_w

    # 5) Prediction and error.
    y_hat_before = H @ a
    error = y - y_hat_before

    # 6) Weighted/Joseph-form RLS.
    R = np.diag([sigma_v_dot**2, sigma_omega_dot**2])
    P_pred = P / forgetting_factor
    S = H @ P_pred @ H.T + R

    if np.linalg.cond(S) > 1e12:
        info = {
            "v_dot_raw": float(v_dot_raw),
            "omega_dot_raw": float(omega_dot_raw),
            "v_dot_measured": float(y[0]),
            "omega_dot_measured": float(y[1]),
            "v_dot_hat": float(y_hat_before[0]),
            "omega_dot_hat": float(y_hat_before[1]),
            "v_dot_error": float(y[0] - y_hat_before[0]),
            "omega_dot_error": float(y[1] - y_hat_before[1]),
            "skipped": True,
        }
        return a, P, filtered_y_dot, info

    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    I = np.eye(10)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    # 7) Optional projection/clipping.
    if clip_parameters and p is not None:
        a_nom = nominal_rls_parameters(p)

        # v_dot coefficients.
        a[0] = np.clip(a[0], 0.25 * a_nom[0], 2.50 * a_nom[0])
        a[1] = np.clip(a[1], 2.50 * a_nom[1], 0.25 * a_nom[1])
        a[2] = np.clip(a[2], -20.0, 20.0)
        a[3] = np.clip(a[3], -20.0, 20.0)
        a[4] = np.clip(a[4], -20.0, 20.0)

        # omega_dot coefficients.
        a[5] = np.clip(a[5], 0.25 * a_nom[5], 2.50 * a_nom[5])
        a[6] = np.clip(a[6], 3.00 * a_nom[6], 0.10 * a_nom[6])
        a[7] = np.clip(a[7], -30.0, 30.0)
        a[8] = np.clip(a[8], -30.0, 30.0)
        a[9] = np.clip(a[9], -80.0, 80.0)

    y_hat_after = H @ a

    info = {
        "v_dot_raw": float(v_dot_raw),
        "omega_dot_raw": float(omega_dot_raw),
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat_after[0]),
        "omega_dot_hat": float(y_hat_after[1]),
        "v_dot_error": float(y[0] - y_hat_after[0]),
        "omega_dot_error": float(y[1] - y_hat_after[1]),
        "skipped": False,
    }

    return a, P, filtered_y_dot, info


# ============================================================
# NMPC Controller
# ============================================================

class WheelieNMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self.n_rls = 10
        self.last_solution = None
        self._build_solver()

    def _f_ca(self, x, u, a_rls):
        # State x = [position, velocity, pitch, pitch_rate]
        x_dot = x[1]

        v_dot = (
            a_rls[0] * u[0]
            + a_rls[1] * x[1]
            + a_rls[2] * ca.fabs(x[1]) * x[1]
            + a_rls[3] * u[0] * ca.cos(x[2])
            + a_rls[4]
        )

        theta_dot = x[3]

        omega_dot = (
            a_rls[5] * ca.cos(x[2])
            + a_rls[6] * u[0]
            + a_rls[7] * x[3]
            + a_rls[8] * x[1]
            + a_rls[9]
        )

        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u, a_rls):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u, a_rls)
        k2 = self._f_ca(x + 0.5 * dt * k1, u, a_rls)
        k3 = self._f_ca(x + 0.5 * dt * k2, u, a_rls)
        k4 = self._f_ca(x + dt * k3, u, a_rls)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector:
        # state [x, v, theta, omega]
        # ref   [x_ref, v_ref, theta_ref, omega_ref]
        # tau_prev
        # RLS coefficients [v_dot coeffs 5, omega_dot coeffs 5]
        P = ca.SX.sym("P", 9 + self.n_rls)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        a_rls = P[9:19]

        obj = 0
        g = []
        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            obj += ca.mtimes([e.T, Q, e])
            obj += cfg.r_tau * uk[0] ** 2
            obj += cfg.r_dtau * du ** 2

            x_next = self._rk4_ca(xk, uk, a_rls)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        nlp = {"f": obj, "x": opt_vars, "g": g, "p": P}

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.tol": 1e-4,
            "print_time": 0,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        nX = self.nx * (N + 1)

        lbx = []
        ubx = []

        for _ in range(N + 1):
            lbx += [-ca.inf, p.v_min, p.theta_min, p.omega_min]
            ubx += [ca.inf, p.v_max, p.theta_max, p.omega_max]

        for _ in range(N):
            lbx += [p.tau_min]
            ubx += [p.tau_max]

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)

        self.lbg = np.zeros(self.nx * (N + 1))
        self.ubg = np.zeros(self.nx * (N + 1))

        self.nX = nX

    def _initial_guess(self, state: np.ndarray, tau_prev: float) -> np.ndarray:
        N = self.cfg.N

        if self.last_solution is not None:
            sol = self.last_solution.copy()

            X_sol = sol[:self.nX].reshape((self.nx, N + 1), order="F")
            U_sol = sol[self.nX:].reshape((self.nu, N), order="F")

            X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

            X_guess[:, 0] = state
        else:
            X_guess = np.tile(state.reshape(-1, 1), (1, N + 1))
            U_guess = np.full((self.nu, N), tau_prev)

        return np.concatenate([
            X_guess.reshape(-1, order="F"),
            U_guess.reshape(-1, order="F"),
        ])

    def solve(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        tau_prev: float,
        a_rls: np.ndarray,
    ) -> tuple[float, dict]:
        params = np.concatenate([
            state,
            ref,
            np.array([tau_prev], dtype=float),
            np.asarray(a_rls, dtype=float).reshape(-1),
        ])

        x_init = self._initial_guess(state, tau_prev)

        try:
            sol = self.solver(
                x0=x_init,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )

            w = np.array(sol["x"]).flatten()
            self.last_solution = w

            U_opt = w[self.nX:].reshape((self.nu, self.cfg.N), order="F")
            tau = float(U_opt[0, 0])
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

            return tau, {"success": True, "cost": float(sol["f"])}

        except RuntimeError as exc:
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "error": str(exc)}


# ============================================================
# Simulation
# ============================================================

def simulate_closed_loop() -> None:
    p = WheelieParams()
    mpc_cfg = MPCConfig()
    controller = WheelieNMPC(p, mpc_cfg)

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
    P_rls = initial_covariance * np.eye(10)
    filtered_y_dot = None

    plant_has_mismatch = True

    # Columns:
    # t, x, v, theta, omega, tau, theta_ref,
    # v_dot_meas, v_dot_hat, v_dot_err,
    # omega_dot_meas, omega_dot_hat, omega_dot_err,
    # ten RLS coefficients
    history = np.zeros((steps, 23))

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