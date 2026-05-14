"""
wheelie_nmpc_casadi.py

Nonlinear MPC for wheelie pitch control using CasADi + IPOPT.

Model:
    x_dot     = v
    v_dot     = tau / (m*r)
    theta_dot = omega
    omega_dot = (-tau + m*g*l*cos(theta)) / I_eff

Install dependencies:
    pip install casadi numpy matplotlib

Run:
    python wheelie_nmpc_casadi.py

Notes:
- This script uses CasADi/IPOPT because acados requires code generation and
  an installed acados environment.
- The MPC tracks theta_ref and optionally v_ref while respecting torque,
  pitch, and angular-rate limits.
"""

import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

try:
    import casadi as ca
except ImportError as exc:
    raise ImportError(
        "CasADi is required. Install it with: pip install casadi"
    ) from exc


@dataclass
class WheelieParams:
    m: float = 5.1          # total mass of the vehicle
    l: float = 0.18         # rear axle to COM DISTANCE
    I_body: float = (1/12)*(m)*(0.53**2 + 0.30**2)    #Inertia calculation I_body = 1/12 * mass(Lenght^2 + Height^2) 0.04
    r: float = 0.085         #
    g: float = 9.81         #
    c_v: float = 9.0        #
    tau_min: float = -8.0
    tau_max: float = 12.0
    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(100.0)
    omega_min: float = -8.0
    omega_max: float = 8.0
    v_min: float = -4.0
    v_max: float = 4.0

    pitch_ref: float = 90.0
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
    q_theta: float = 1000.0
    q_omega: float = 15.0
    r_tau: float = 0.1
    r_dtau: float = 0.01
    q_terminal_theta: float = 400.0
    q_terminal_omega: float = 100.0
    ipopt_max_iter: int = 50


def continuous_dynamics_np(state: np.ndarray, tau: float, p: WheelieParams) -> np.ndarray:
    x, v, theta, omega = state

    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v
    pitch_dot = omega
    omega_dot = (-tau + p.m * p.g * p.l * math.cos(theta)) / p.I_eff
    dynamics = np.array([x_dot, v_dot, pitch_dot, omega_dot], dtype=float)

    return dynamics

    # return np.array([
    #     v,
    #     tau / (p.m * p.r) - p.c_v * v,
    #     omega,
    #     (-tau + p.m * p.g * p.l * math.cos(theta)) / p.I_eff,
    # ], dtype=float)


def rk4_step_np(state: np.ndarray, tau: float, dt: float, p: WheelieParams) -> np.ndarray:
    k1 = continuous_dynamics_np(state, tau, p)
    k2 = continuous_dynamics_np(state + 0.5 * dt * k1, tau, p)
    k3 = continuous_dynamics_np(state + 0.5 * dt * k2, tau, p)
    k4 = continuous_dynamics_np(state + dt * k3, tau, p)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class WheelieNMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self._build_solver()
        self.last_solution = None

    def _f_ca(self, x, u):
        p = self.p
        return ca.vertcat(
            x[1],
            u[0] / (p.m * p.r) - p.c_v * x[1],
            x[3],
            (-u[0] + p.m * p.g * p.l * ca.cos(x[2])) / p.I_eff,
        )

    def _rk4_ca(self, x, u):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u)
        k2 = self._f_ca(x + 0.5 * dt * k1, u)
        k3 = self._f_ca(x + 0.5 * dt * k2, u)
        k4 = self._f_ca(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N

        # Decision variables.
        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector:
        # current state [x, v, theta, omega], reference [x_ref, v_ref, theta_ref, omega_ref], previous tau
        P = ca.SX.sym("P", 9)
        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]

        obj = 0
        g = []

        # Initial condition constraint.
        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        R = cfg.r_tau
        Rd = cfg.r_dtau
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            tau_eq = p.m * p.g * p.l * ca.cos(ref[2])
            obj += ca.mtimes([e.T, Q, e]) + R * (uk[0] - tau_eq)**2 + Rd * du**2

            x_next = self._rk4_ca(xk, uk)
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

        # Bounds on decision variables.
        nX = self.nx * (N + 1)
        nU = self.nu * N
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

        # Equality constraints are all zero.
        self.lbg = np.zeros(self.nx * (N + 1))
        self.ubg = np.zeros(self.nx * (N + 1))

        self.nX = nX
        self.nU = nU

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

    def solve(self, state: np.ndarray, ref: np.ndarray, tau_prev: float) -> tuple[float, dict]:
        params = np.concatenate([state, ref, np.array([tau_prev])])
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
            info = {"success": True, "cost": float(sol["f"])}
            return tau, info
        except RuntimeError as exc:
            # Safe fallback: keep previous torque clipped.
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            info = {"success": False, "error": str(exc)}
            return tau, info


def simulate_closed_loop() -> None:
    p = WheelieParams()
    cfg = MPCConfig()
    controller = WheelieNMPC(p, cfg)

    theta_ref_deg = p.pitch_ref
    theta_ref = math.radians(theta_ref_deg)

    # Reference state: x_ref, v_ref, theta_ref, omega_ref.
    # x_ref is not important because q_x=0 by default.
    ref = np.array([0.0, 0.0, theta_ref, 0.0], dtype=float)

    state = np.array([0.0, 0.0, math.radians(0.0), 0.0], dtype=float)
    tau_prev = 0.0

    sim_dt = p.sim_dt
    T = p.sim_time
    steps = int(T / sim_dt)
    mpc_period_steps = max(1, int(cfg.dt / sim_dt))

    history = np.zeros((steps, 7))
    tau_cmd = 0.0

    for k in range(steps):
        t = k * sim_dt

        if k % mpc_period_steps == 0:
            tau_cmd, info = controller.solve(state, ref, tau_prev)
            tau_prev = tau_cmd
            if not info["success"]:
                print(f"[WARN] NMPC failed at t={t:.2f}, using fallback torque")

        history[k] = [t, state[0], state[1], state[2], state[3], tau_cmd, ref[2]]
        state = rk4_step_np(state, tau_cmd, sim_dt, p)

    t = history[:, 0]
    theta_deg = np.rad2deg(history[:, 3])
    theta_ref_deg_vec = np.rad2deg(history[:, 6])
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
    axs[0].plot(t, theta_ref_deg_vec, linestyle="--", label="theta_ref")
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

    fig.suptitle("Wheelie NMPC Closed-Loop Response")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    simulate_closed_loop()
