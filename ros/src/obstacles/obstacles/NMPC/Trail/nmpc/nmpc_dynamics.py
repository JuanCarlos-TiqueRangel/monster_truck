
import math

import numpy as np
import casadi as ca

from params_nmpc import WheelieParams, MPCConfig
import matplotlib.pyplot as plt

# ============================================================
# NMPC Controller
# ============================================================

class NMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig, live_plot: bool = False):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self.last_solution = None

        self.live_plot = bool(live_plot)
        self.plot_obstacle_span = None
        self.plot = None
        self.plot_off = False

        self._build_solver()


    def _f_ca(self, x, u):
        p = self.p

        x_pos = x[0]
        v = x[1]
        theta = x[2]
        omega = x[3]
        tau = u[0]

        x_dot = v
        v_dot = tau / (p.m * p.r)
        theta_dot = omega

        omega_dot_free = (-tau + p.m * p.g * p.l * ca.cos(theta)) / p.I_eff

        on_ground = ca.logic_and(theta >= 0.0, omega_dot_free > 0.0)
        omega_dot = ca.if_else(on_ground, 0.0, omega_dot_free)

        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)


    def _rk4_ca(self, x, u):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u)
        k2 = self._f_ca(x + 0.5 * dt * k1, u)
        k3 = self._f_ca(x + 0.5 * dt * k2, u)
        k4 = self._f_ca(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector: state(4), ref(4), tau_prev(1)
        P = ca.SX.sym("P", 9)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]

        obj = 0
        g = []
        g.append(X[:, 0] - x0)

        #Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            theta_deg = xk[2] * (180.0 / math.pi)
            ref_theta_deg = ref[2] * (180.0 / math.pi)
            e_theta_deg = theta_deg - ref_theta_deg

            # obj += (cfg.q_x * e[0] ** 2 
            #         + cfg.q_v * e[1] ** 2
            #         + cfg.q_theta * xk[2] ** 2 
            #         + cfg.q_omega * e[3] ** 2)

            obj += cfg.r_tau * uk[0] ** 2
            obj += cfg.r_dtau * du ** 2

            obj += (
                cfg.q_x * e[0] ** 2
                + cfg.q_v * e[1] ** 2
                + cfg.q_theta * e_theta_deg ** 2
                #+ cfg.q_omega * e[3] ** 2
            )

            x_next = self._rk4_ca(xk, uk)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        terminal_theta_deg = X[2, N] * (180.0 / math.pi)
        terminal_ref_theta_deg = ref[2] * (180.0 / math.pi)
        terminal_e_theta_deg = terminal_theta_deg - terminal_ref_theta_deg

        obj += (
            cfg.q_x * eN[0] ** 2
            + cfg.q_v * eN[1] ** 2
            + cfg.q_terminal_theta * terminal_e_theta_deg ** 2
            #+ cfg.q_terminal_omega * eN[3] ** 2
        )

        # obj += ca.mtimes([eN.T, Qf, eN])

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

        # opt_vars 
        inf = ca.inf
        self.lbx = np.array(
            [-inf, -inf, -inf, -inf]
            + [-inf, p.v_min, p.theta_min, p.omega_min] * N
            + [p.tau_min] * N,
            dtype=float,
        )
        self.ubx = np.array(
            [inf, inf, inf, inf]
            + [inf, p.v_max, p.theta_max, p.omega_max] * N
            + [p.tau_max] * N,
            dtype=float,
        )

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


    def solve(self, state: np.ndarray, ref: np.ndarray, tau_prev: float) -> tuple[float, dict]:
        params = np.concatenate([
            np.asarray(state, dtype=float).reshape(-1),
            np.asarray(ref, dtype=float).reshape(-1),
            np.array([tau_prev], dtype=float),
        ])

        x_init = self._initial_guess(np.asarray(state, dtype=float), tau_prev)

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

            X_opt = w[:self.nX].reshape((self.nx, self.cfg.N + 1), order="F")
            self.plot_plan(X_opt, ref, state)

            U_opt = w[self.nX:].reshape((self.nu, self.cfg.N), order="F")
            tau = float(U_opt[0, 0])
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

            stats = self.solver.stats()
            success = bool(stats.get("success", False))

            return tau, {
                "success": success,
                "cost": float(sol["f"]),
                "status": stats.get("return_status", ""),
            }

        except RuntimeError as exc:
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "cost": np.inf, "error": str(exc)}
        

    def plot_plan(self, X_opt, ref, state):
        if not self.live_plot or self.plot_off:
            return

        try:
            # Red line: NMPC predicted trajectory
            xs = X_opt[0, :].copy()
            theta_deg = np.rad2deg(X_opt[2, :]).copy()

            # Blue dot/trail: real measured robot state
            x_now = float(state[0])
            theta_now = math.degrees(float(state[2]))

            # Force prediction to visually start from the measured robot state
            xs[0] = x_now
            theta_deg[0] = theta_now

            goal_x = float(ref[0])

            if self.plot is None:
                plt.ion()
                fig, ax = plt.subplots()

                ax.set(
                    xlabel="x [m]",
                    ylabel="theta [deg]",
                    xlim=(-2.0, goal_x + 2.0),
                    ylim=(-120.0, 120.0),
                    title="NMPC planned trajectory (x vs theta)",
                )

                if self.plot_obstacle_span is not None:
                    ax.axvspan(*self.plot_obstacle_span, color="0.85", zorder=0)

                ax.grid(True, alpha=0.3)

                opt_line, = ax.plot([], [], "C3-o", lw=2.0, ms=3, label="optimal plan")
                trail_line, = ax.plot([], [], "C0-", lw=1.3, alpha=0.85, label="realized")
                car_point, = ax.plot([], [], "o", ms=12, mfc="blue", mec="k", mew=1.0, zorder=5, label="car")

                ax.legend(loc="upper left")

                self.plot = {
                    "fig": fig,
                    "ax": ax,
                    "opt": opt_line,
                    "trail": trail_line,
                    "car": car_point,
                    "trail_x": [],
                    "trail_theta": [],
                }

            plot = self.plot

            if plot["trail_x"] and plot["trail_x"][-1] - x_now > 0.5:
                plot["trail_x"].clear()
                plot["trail_theta"].clear()

            plot["trail_x"].append(x_now)
            plot["trail_theta"].append(theta_now)

            plot["opt"].set_data(xs, theta_deg)
            plot["trail"].set_data(plot["trail_x"], plot["trail_theta"])
            plot["car"].set_data([x_now], [theta_now])

            plot["fig"].canvas.draw_idle()
            plot["fig"].canvas.flush_events()
            plt.pause(0.001)

        except Exception:
            self.plot_off = True