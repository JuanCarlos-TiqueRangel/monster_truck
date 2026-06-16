import math

import numpy as np
import casadi as ca

from params_mujoco import WheelieParams, MPCConfig


def casadi_gp_mean(z_sym, Z_sym, alpha_sym, lengthscales, sf2):
    """Symbolic ARD-RBF sparse-GP mean
        mu(z) = sum_i alpha_i * sf2 * exp(-0.5 ||(z - Z_i)/l||^2)
    over the inducing set (Z, alpha) the streaming SSGP supplies each solve.
    Kept here so the NMPC depends only on SSGP.py (Z, alpha, lengthscales, sf2)."""
    M = Z_sym.shape[0]
    l = ca.DM(np.asarray(lengthscales, dtype=float).reshape(-1))
    out = 0
    for i in range(M):
        diff = (z_sym - Z_sym[i, :].T) / l
        out += alpha_sym[i] * (sf2 * ca.exp(-0.5 * ca.dot(diff, diff)))
    return out


# ============================================================
# NMPC Controller
# ============================================================

class NMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig, gp_cfg):
        self.p = p
        self.cfg = cfg
        self.gp_cfg = gp_cfg
        self.nx = 4
        self.nu = 1
        self.n_rls = 10        # a = [a_v(5), a_w(5)]
        self.M = gp_cfg.max_points                  # GP inducing points
        self.d = len(gp_cfg.lengthscales)           # GP feature dim (= 5: x,theta,omega,v,tau)
        self.gp_l = np.asarray(gp_cfg.lengthscales, dtype=float)
        self.last_solution = None
        self._build_solver()

    def _f_ca(self, x, u, a_rls, Z, alpha_v_dot, alpha_omega_dot):
        # State x = [position, velocity, pitch, pitch_rate]
        # GP residual feature z = [x, theta] (position + pitch). The obstacle is a
        # function of WHERE you are and HOW reared you are; evaluating gp_v_dot at the
        # PREDICTED (x, theta) along the horizon lets the NMPC see the blockage coming
        # and rear BEFORE contact -- the wheelie emerges, no pitch reference.
        z = ca.vertcat(x[0], x[2])
        gp_v_dot = casadi_gp_mean(z, Z, alpha_v_dot, self.gp_l, self.gp_cfg.sf2)
        gp_omega_dot = casadi_gp_mean(z, Z, alpha_omega_dot, self.gp_l, self.gp_cfg.sf2)

        x_dot = x[1]

        v_dot = (
            a_rls[0] * u[0]                          # drive torque -> traction force
            + a_rls[1] * x[1]                        # linear (viscous/rolling) drag
            + a_rls[2] * ca.fabs(x[1]) * x[1]        # quadratic aero drag
            + a_rls[3] * u[0] * (ca.cos(x[2]) - 1.0) # traction roll-off when reared (0 at flat)
            + a_rls[4]                               # constant bias
            + gp_v_dot
        )

        theta_dot = x[3]

        omega_dot = (
            a_rls[5] * ca.cos(x[2])                  # gravity restoring torque (pendulum)
            + a_rls[6] * u[0]                        # wheel reaction torque that pops the wheelie
            + a_rls[7] * x[3]                        # pitch-rate damping
            + a_rls[8] * x[1]                        # weight-transfer/speed coupling
            + a_rls[9]                               # constant bias
            + gp_omega_dot
        )

        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u, a_rls, Z, alpha_v_dot, alpha_omega_dot):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u, a_rls, Z, alpha_v_dot, alpha_omega_dot)
        k2 = self._f_ca(x + 0.5 * dt * k1, u, a_rls, Z, alpha_v_dot, alpha_omega_dot)
        k3 = self._f_ca(x + 0.5 * dt * k2, u, a_rls, Z, alpha_v_dot, alpha_omega_dot)
        k4 = self._f_ca(x + dt * k3, u, a_rls, Z, alpha_v_dot, alpha_omega_dot)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N
        M, d = self.M, self.d

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector:
        #   state(4) ref(4) tau_prev(1) a_rls(n_rls=10) | Z(M*d) alpha_v_dot(M) alpha_omega_dot(M)
        n_base = 9 + self.n_rls
        P = ca.SX.sym("P", n_base + M * d + 2 * M)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        i = 9
        a_rls = P[i:i + self.n_rls];            i += self.n_rls
        Z = ca.reshape(P[i:i + M * d], M, d);   i += M * d
        alpha_v_dot = P[i:i + M];                   i += M
        alpha_omega_dot = P[i:i + M]

        obj = 0
        g = []
        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        # Smooth flip penalty. The goal-reaching cost makes the truck want forward
        # progress; with no pitch reference (q_theta=0) nothing stops it rearing past
        # the ~90 deg tip-over while chasing the goal. These one-sided, differentiable
        # barriers (smooth so IPOPT converges -- a fmax/fabs kink stalls it) make
        # over-rearing expensive WITHOUT forbidding the wheelie needed to climb.
        theta_soft = math.radians(cfg.theta_soft_deg)
        theta_climb = math.radians(cfg.theta_climb_deg)

        def _smooth_relu(s):
            return 0.5 * (s + ca.sqrt(s * s + 1e-4))      # smooth max(0, s)

        def flip_pen(xk):
            # barrier on pitch beyond theta_soft: q_flip * relu(theta^2 - theta_soft^2)^2
            return cfg.q_flip * _smooth_relu(xk[2] ** 2 - theta_soft ** 2) ** 2

        def flipw_pen(xk):
            # penalise rotating FURTHER up (omega>0) while already reared past
            # theta_climb -> stops the tip-over but allows a held/recovering wheelie.
            return cfg.q_flipw * _smooth_relu(xk[2] - theta_climb) * _smooth_relu(xk[3])

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
            obj += flip_pen(xk) + flipw_pen(xk)

            x_next = self._rk4_ca(xk, uk, a_rls, Z, alpha_v_dot, alpha_omega_dot)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])
        obj += flip_pen(X[:, N]) + flipw_pen(X[:, N])

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

        # opt_vars order is [all states; all controls], so the bounds are laid out
        # the same way: stage-0 state | stages 1..N state | N controls.
        # Stage 0 is left free (the equality X[:,0]-x0=0 pins it) so a measured
        # state outside the box bounds cannot make the NLP infeasible.
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

    def solve(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        tau_prev: float,
        a_rls: np.ndarray,
        gp_params: np.ndarray,
    ) -> tuple[float, dict]:
        params = np.concatenate([
            np.asarray(state, dtype=float).reshape(-1),
            np.asarray(ref, dtype=float).reshape(-1),
            np.array([tau_prev], dtype=float),
            np.asarray(a_rls, dtype=float).reshape(-1),
            np.asarray(gp_params, dtype=float).reshape(-1),
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

            U_opt = w[self.nX:].reshape((self.nu, self.cfg.N), order="F")
            tau = float(U_opt[0, 0])
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

            return tau, {"success": True, "cost": float(sol["f"])}

        except RuntimeError as exc:
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "error": str(exc)}
