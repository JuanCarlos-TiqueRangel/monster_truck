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
    N: int = 35

    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 1300.0
    q_omega: float = 15.0

    r_tau: float = 0.1
    r_dtau: float = 0.01

    q_terminal_theta: float = 400.0
    q_terminal_omega: float = 100.0

    ipopt_max_iter: int = 50


# ============================================================
# Nominal dynamics
# ============================================================
def continuous_dynamics_np(
    state: np.ndarray,
    tau: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    """
    Simulated plant.

    For real MuJoCo/hardware, replace this with the measured next state.
    """
    x, v, theta, omega = state

    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v
    theta_dot = omega
    omega_dot =  (-tau + p.m * p.g * p.l * np.cos(theta)) / p.I_eff 

    if plant_has_mismatch:
        # Artificial unknown dynamics for testing RLS only.
        unknown = 0.5 * omega + 5.0 * v + 3.0 * np.sin(theta)
        omega_dot = omega_dot + unknown

    return np.array([x_dot, v_dot, theta_dot, omega_dot], dtype=float)


def rk4_step_np(state: np.ndarray, tau: float, dt: float, p: WheelieParams, plant_has_mismatch: bool = False,) -> np.ndarray:
    k1 = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
    k2 = continuous_dynamics_np(state + 0.5 * dt * k1, tau, p, plant_has_mismatch)
    k3 = continuous_dynamics_np(state + 0.5 * dt * k2, tau, p, plant_has_mismatch)
    k4 = continuous_dynamics_np(state + dt * k3, tau, p, plant_has_mismatch)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================
# RLS in one function
# ============================================================

def rls_update(
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    p: WheelieParams,
    a: np.ndarray,
    P: np.ndarray,
    filtered_omega_dot: float | None,
    forgetting_factor: float = 0.999,
    derivative_alpha: float = 0.85,
    clip_parameters: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, dict]:

    _, v_prev, theta_prev, omega_prev = state_prev
    omega_next = float(state_next[3])

    # 1) Measured angular acceleration
    omega_dot_raw = (omega_next - float(omega_prev)) / dt

    # 2) Filter measured angular acceleration FIRST
    if filtered_omega_dot is None:
        filtered_omega_dot = omega_dot_raw
    else:
        filtered_omega_dot = (
            derivative_alpha * filtered_omega_dot
            + (1.0 - derivative_alpha) * omega_dot_raw
        )

    omega_dot_measured = float(filtered_omega_dot)

    # 3) Nominal angular acceleration
    omega_dot_nominal = (
        -tau + p.m * p.g * p.l * np.cos(theta_prev)
    ) / p.I_eff

    # 4) Residual target
    residual_target = omega_dot_measured - omega_dot_nominal

    # 5) Feature vector: phi = [cos(theta), tau, omega, v, 1]
    phi = np.array(
        [np.cos(theta_prev), tau, omega_prev, v_prev, 1.0],
        dtype=float,
    )

    # 6) RLS prediction of residual
    residual_hat_before = float(phi @ a)
    error_before = residual_target - residual_hat_before

    # 7) RLS gain
    P_phi = P @ phi
    denom = forgetting_factor + float(phi @ P_phi)

    if abs(denom) < 1e-12:
        omega_dot_hat = omega_dot_nominal + residual_hat_before
        info = {
            "omega_dot_raw": float(omega_dot_raw),
            "y": float(omega_dot_measured),
            "y_hat": float(omega_dot_hat),
            "residual_target": float(residual_target),
            "residual_hat": float(residual_hat_before),
            "error": float(omega_dot_measured - omega_dot_hat),
            "skipped": True,
        }
        return a, P, float(filtered_omega_dot), info

    K = P_phi / denom

    # 8) Update residual parameters
    a = a + K * error_before

    # 9) Covariance update
    I = np.eye(5)
    P = ((I - np.outer(K, phi)) @ P) / forgetting_factor
    P = 0.5 * (P + P.T)

    # 10) Clip residual parameters around zero
    if clip_parameters:
        a_g_nom = p.m * p.g * p.l / p.I_eff
        a_tau_nom = -1.0 / p.I_eff

        a[0] = np.clip(a[0], -0.5 * abs(a_g_nom), 0.5 * abs(a_g_nom))
        a[1] = np.clip(a[1], -0.5 * abs(a_tau_nom), 0.5 * abs(a_tau_nom))
        a[2] = np.clip(a[2], -5.0, 5.0)
        a[3] = np.clip(a[3], -1.0, 1.0)
        a[4] = np.clip(a[4], -10.0, 10.0)

    residual_hat_after = float(phi @ a)
    omega_dot_hat = omega_dot_nominal + residual_hat_after

    info = {
        "omega_dot_raw": float(omega_dot_raw),
        "y": float(omega_dot_measured),
        "y_hat": float(omega_dot_hat),
        "residual_target": float(residual_target),
        "residual_hat": float(residual_hat_after),
        "error": float(omega_dot_measured - omega_dot_hat),
        "skipped": False,
    }

    return a, P, float(filtered_omega_dot), info


# ============================================================
# NMPC Controller
# ============================================================

class WheelieNMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self.n_rls = 5
        self.last_solution = None
        self._build_solver()

    def _omega_dot_ca(self, x, u, a_rls):
        p = self.p

        omega_dot_nominal = (-u[0] + p.m * p.g * p.l * ca.cos(x[2])) / p.I_eff

        omega_dot_rls = (
            a_rls[0] * ca.cos(x[2])
            + a_rls[1] * u[0]
            + a_rls[2] * x[3]
            + a_rls[3] * x[1]
            + a_rls[4]
        )

        return omega_dot_nominal + omega_dot_rls

    def _f_ca(self, x, u, a_rls):
        p = self.p
        # State x = [position, velocity, pitch, pitch_rate]
        x_dot = x[1]
        v_dot = u[0] / (p.m * p.r) - p.c_v * x[1]
        theta_dot = x[3]
        omega_dot = self._omega_dot_ca(x, u, a_rls)

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
        # RLS coefficients [a_g, a_tau, a_omega, a_v, a_0]
        P = ca.SX.sym("P", 9 + self.n_rls)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        a_rls = P[9:14]

        obj = 0
        g = []

        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        # for k in range(N):
        #     xk = X[:, k]
        #     uk = U[:, k]

        #     theta_error = xk[2] - ref[2]

        #     if k == 0:
        #         du = uk[0] - tau_prev
        #     else:
        #         du = uk[0] - U[0, k - 1]

        #     tau_eq = p.m * p.g * p.l * ca.cos(ref[2])

        #     # Angle tracking only
        #     obj += cfg.q_theta * theta_error**2

        #     # Keep torque effort
        #     obj += cfg.r_tau * (uk[0] - tau_eq)**2

        #     # Keep torque-rate smoothing
        #     obj += cfg.r_dtau * du**2

        #     x_next = self._rk4_ca(xk, uk, a_rls)
        #     g.append(X[:, k + 1] - x_next)

        # # Terminal angle tracking only
        # theta_error_N = X[2, N] - ref[2]
        # obj += cfg.q_terminal_theta * theta_error_N**2

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            tau_eq = p.m * p.g * p.l * ca.cos(ref[2])

            obj += ca.mtimes([e.T, Q, e])
            obj += cfg.r_tau * (uk[0] - tau_eq) ** 2
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
            ubx += [ ca.inf, p.v_max, p.theta_max, p.omega_max]

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
        a_rls: np.ndarray,) -> tuple[float, dict]:
        
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

    # RLS settings.
    forgetting_factor = 0.999
    initial_covariance = 5.0

    # For clean simulation, 0.0 is okay.
    # For MuJoCo or real hardware, try 0.7 to 0.9.
    derivative_alpha = 0.0

    clip_parameters = False

    # Initial RLS parameters from nominal physics.
    a_nom = np.array(
        [
            p.m * p.g * p.l / p.I_eff,
            -1.0 / p.I_eff,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    #a_rls = a_nom.copy()
    a_rls = np.zeros(5)
    P_rls = initial_covariance * np.eye(5)
    filtered_omega_dot = None

    # Set True only if you want to test whether RLS learns artificial mismatch.
    # For nominal simulation, keep False.
    plant_has_mismatch = True

    # Columns:
    # t, x, v, theta, omega, tau, theta_ref,
    # omega_dot_measured, omega_dot_rls, rls_error,
    # a_g, a_tau, a_omega, a_v, a_0
    history = np.zeros((steps, 15))

    for k in range(steps):
        t = k * sim_dt
        state_prev = state.copy()

        if k % mpc_period_steps == 0:
            tau_cmd, info = controller.solve(state_prev, ref, tau_prev, a_rls)
            tau_prev = tau_cmd

            if not info["success"]:
                print(f"[WARN] NMPC failed at t={t:.2f}, using fallback torque")

        # Simulate one step.
        # In MuJoCo/real hardware, replace this with your measured next state.
        state_next = rk4_step_np(state_prev, tau_cmd, sim_dt, p, plant_has_mismatch=plant_has_mismatch)

        # RLS update OUTSIDE the NMPC solver.
        a_rls, P_rls, filtered_omega_dot, rls_info = rls_update(
            state_prev=state_prev,
            tau=tau_cmd,
            state_next=state_next,
            dt=sim_dt,
            p=p,
            a=a_rls,
            P=P_rls,
            filtered_omega_dot=filtered_omega_dot,
            forgetting_factor=forgetting_factor,
            derivative_alpha=derivative_alpha,
            clip_parameters=clip_parameters,
        )

        history[k, :] = [
            t,
            state_prev[0],
            state_prev[1],
            state_prev[2],
            state_prev[3],
            tau_cmd,
            ref[2],
            rls_info["y"],
            rls_info["y_hat"],
            rls_info["error"],
            a_rls[0],
            a_rls[1],
            a_rls[2],
            a_rls[3],
            a_rls[4],
        ]

        state = state_next

    print_results(history, p, a_rls, a_nom)
    plot_results(history, a_nom)


def print_results(history: np.ndarray, p: WheelieParams, a_rls: np.ndarray, a_nom: np.ndarray) -> None:
    theta_deg = np.rad2deg(history[:, 3])
    omega = history[:, 4]
    x = history[:, 1]
    v = history[:, 2]

    print("\n========== Final result ==========")
    print(f"Final theta: {theta_deg[-1]:.2f} deg")
    print(f"Final omega: {omega[-1]:.3f} rad/s")
    print(f"Final x:     {x[-1]:.2f} m")
    print(f"Final v:     {v[-1]:.2f} m/s")

    print("\n========== RLS coefficients ==========")
    print("Model: omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    print(f"a_g     learned: {a_rls[0]: .4f} | nominal: {a_nom[0]: .4f}")
    print(f"a_tau   learned: {a_rls[1]: .4f} | nominal: {a_nom[1]: .4f}")
    print(f"a_omega learned: {a_rls[2]: .4f} | nominal: {a_nom[2]: .4f}")
    print(f"a_v     learned: {a_rls[3]: .4f} | nominal: {a_nom[3]: .4f}")
    print(f"a_0     learned: {a_rls[4]: .4f} | nominal: {a_nom[4]: .4f}")
    print("==================================\n")


def plot_results(history: np.ndarray, a_nom: np.ndarray) -> None:
    t = history[:, 0]
    theta_deg = np.rad2deg(history[:, 3])
    theta_ref_deg = np.rad2deg(history[:, 6])
    omega = history[:, 4]
    tau = history[:, 5]

    omega_dot_measured = history[:, 7]
    omega_dot_rls = history[:, 8]
    rls_error = history[:, 9]

    a_g = history[:, 10]
    a_tau = history[:, 11]
    a_omega = history[:, 12]
    a_v = history[:, 13]
    a_0 = history[:, 14]

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(11, 10))

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

    axs[3].plot(t, omega_dot_measured, label="measured omega_dot")
    axs[3].plot(t, omega_dot_rls, linestyle="--", label="RLS omega_dot")
    axs[3].plot(t, rls_error, linestyle=":", label="RLS error")
    axs[3].set_ylabel("omega_dot [rad/s^2]")
    axs[3].grid(True)
    axs[3].legend()

    axs[4].plot(t, a_g, label="a_g")
    axs[4].plot(t, a_tau, label="a_tau")
    axs[4].plot(t, a_omega, label="a_omega")
    axs[4].plot(t, a_v, label="a_v")
    axs[4].plot(t, a_0, label="a_0")
    axs[4].axhline(a_nom[0], linestyle="--", linewidth=1, label="a_g nominal")
    axs[4].axhline(a_nom[1], linestyle="--", linewidth=1, label="a_tau nominal")
    axs[4].set_ylabel("RLS coeffs")
    axs[4].set_xlabel("time [s]")
    axs[4].grid(True)
    axs[4].legend(ncol=3)

    fig.suptitle("Wheelie NMPC + RLS in One Function")
    fig.tight_layout()
    plt.show()

    fig.savefig("images/wheelie_residual.png", dpi=200)
    print("Saved figure:", "images/wheelie_residual.png")


if __name__ == "__main__":
    simulate_closed_loop()