
import math

import numpy as np
import casadi as ca

from params_nmpc import WheelieParams, MPCConfig

# The GP residual's closed-form mean lives in gp_kernel (shared with GP.py / the MPPI).
# casadi_gp_mean builds the symbolic kernel sum; the kernel SHAPE is baked in at BUILD
# time from gp_cfg.kernel (casadi can't branch on a runtime kernel id).
from gp_kernel import casadi_gp_mean, kernel_id_from_name


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
        self.n_rls = 20        # a = [a_v(5), a_w(5)]
        self.M = gp_cfg.max_points                  # GP inducing points (rollout set size)
        self.d = gp_cfg.n_features                  # GP feature dim (= 5: x, v, theta, omega, tau)
        # kernel SHAPE baked into the symbolic graph (must match the GP model's kernel).
        self.kid = kernel_id_from_name(getattr(gp_cfg, "kernel", "matern32"))
        self.last_solution = None
        self._build_solver()

    def _f_ca(self, x, u, a_rls, gp):
        # State x = [position, velocity, pitch, pitch_rate], control u = [tau].
        # GP residual feature z = [x, v, theta, omega, tau] (d = 5) -- the FULL system
        # input, matching GP.py. Evaluating the residual at the PREDICTED (x,v,theta,
        # omega,tau) along the horizon lets the NMPC see the blockage coming and rear
        # BEFORE contact -- the wheelie emerges, no pitch reference.
        (Z, alpha_v_dot, alpha_omega_dot, x_mean, x_std,
         ell_v, ell_w, sf2_v, sf2_w, ymean_v, ymean_w) = gp
        z = ca.vertcat(x[0], x[1], x[2], x[3], u[0])
        # Both residual channels are exported and can drive the rollout. The omega target is
        # CLIPPED before observe (OMEGA_RES_CLIP in the sim) so contact impulses don't inflate
        # the GP's fitted noise; the lengthscales/noise are LEARNED by fit() (not bounded). If
        # gp_omega_dot still feathers the pitch (it does in the MPPI variant), set
        # GP_OMEGA_IN_ROLLOUT=False in the sim -- mpc_params() then zeros this channel, so
        # gp_omega_dot is 0 here while the v_dot blockage still drives the wheelie.
        gp_v_dot = casadi_gp_mean(z, Z, alpha_v_dot, x_mean, x_std, ell_v, sf2_v, ymean_v, self.kid)
        gp_omega_dot = casadi_gp_mean(z, Z, alpha_omega_dot, x_mean, x_std, ell_w, sf2_w, ymean_w, self.kid)

        # x_dot = x[1]
        # v_dot = (
        #     a_rls[0] * u[0]                          # drive torque -> traction force
        #     + a_rls[1] * x[1]                        # linear (viscous/rolling) drag
        #     + a_rls[2] * ca.fabs(x[1]) * x[1]        # quadratic aero drag
        #     + a_rls[3] * u[0] * (ca.cos(x[2]) - 1.0) # traction roll-off when reared (0 at flat)
        #     + a_rls[4]                               # constant bias
        #     + gp_v_dot
        # )
        # theta_dot = x[3]
        # omega_dot = (
        #     a_rls[5] * ca.cos(x[2])                  # gravity restoring torque (pendulum)
        #     + a_rls[6] * u[0]                        # wheel reaction torque that pops the wheelie
        #     + a_rls[7] * x[3]                        # pitch-rate damping
        #     + a_rls[8] * x[1]                        # weight-transfer/speed coupling
        #     + a_rls[9]                               # constant bias
        #     + gp_omega_dot
        # )

        # x_dot = x[1]

        # v_dot = (
        #     a_rls[0] * u[0]                          # drive torque -> traction force
        #     + a_rls[1] * x[1]                        # linear (viscous/rolling) drag
        #     + a_rls[2] * ca.fabs(x[1]) * x[1]        # quadratic aero drag
        #     + a_rls[3] * u[0] * (ca.cos(x[2]))       # traction roll-off when reared (0 at flat)
        #     + a_rls[4] * ca.tanh(x[1]/0.05)
        #     + a_rls[5] * x[3]**2 * (ca.cos(x[2]))
        #     + a_rls[6]                               # constant bias
        #     + gp_v_dot
        # )

        # theta_dot = x[3]

        # omega_dot = (
        #     a_rls[7] * ca.cos(x[2])                  # gravity restoring torque (pendulum)
        #     + a_rls[8] * u[0]                        # wheel reaction torque that pops the wheelie
        #     + a_rls[9] * x[3]                        # pitch-rate damping
        #     + a_rls[10] * x[1]                        # weight-transfer/speed coupling
        #     + a_rls[11] * ca.fabs(x[3]) * x[3]
        #     + a_rls[12] * x[1] * x[3]
        #     + a_rls[13] * ca.fabs(x[1]) * x[3]
        #     + a_rls[14] * ca.cos(x[2]) * x[3]
        #     + a_rls[15] * ca.sin(x[2]) * x[1]
        #     + a_rls[16] * ca.sin(x[2])
        #     + a_rls[17] * ca.cos(x[2]) * u[0]
        #     + a_rls[18] * u[0] * x[1]
        #     + a_rls[19]                               # constant bias
        #     + gp_omega_dot
        # )

        x_dot = x[1]
        v_dot = (u[0]/(5.1 * 0.081)) + gp_v_dot
        theta_dot = x[3]
        omega_dot = ((-u[0] + 5.1*9.81*0.2*ca.cos(x[2]))/((1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2))) + gp_omega_dot


        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u, a_rls, gp):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u, a_rls, gp)
        k2 = self._f_ca(x + 0.5 * dt * k1, u, a_rls, gp)
        k3 = self._f_ca(x + 0.5 * dt * k2, u, a_rls, gp)
        k4 = self._f_ca(x + dt * k3, u, a_rls, gp)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N
        M, d = self.M, self.d

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector. The GP block is GP.mpc_params():
        #   state(4) ref(4) tau_prev(1) a_rls(n_rls=10)
        #   | Z(M*d) alpha_v(M) alpha_w(M) x_mean(d) x_std(d) ell_v(d) ell_w(d) sf2_v sf2_w ymean_v ymean_w kernel_id
        # The trailing kernel_id is consumed for SIZE only (the kernel shape is baked into
        # this graph at build time from gp_cfg.kernel -- casadi can't branch on it at solve).
        n_base = 9 + self.n_rls
        gp_size = M * d + 2 * M + 4 * d + 4 + 1
        P = ca.SX.sym("P", n_base + gp_size)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        i = 9
        a_rls = P[i:i + self.n_rls];            i += self.n_rls
        Z = ca.reshape(P[i:i + M * d], M, d);   i += M * d
        alpha_v_dot = P[i:i + M];               i += M
        alpha_omega_dot = P[i:i + M];           i += M
        x_mean = P[i:i + d];                    i += d
        x_std = P[i:i + d];                     i += d
        ell_v = P[i:i + d];                     i += d
        ell_w = P[i:i + d];                     i += d
        sf2_v = P[i];                           i += 1
        sf2_w = P[i];                           i += 1
        ymean_v = P[i];                         i += 1
        ymean_w = P[i];                         i += 1
        gp = (Z, alpha_v_dot, alpha_omega_dot, x_mean, x_std,
              ell_v, ell_w, sf2_v, sf2_w, ymean_v, ymean_w)

        obj = 0
        g = []
        g.append(X[:, 0] - x0)

        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        # Smooth flip penalty. The goal-reaching cost makes the truck want forward
        # progress; with only a mild flat preference (small q_theta) nothing else stops
        # it rearing past the ~90 deg tip-over while chasing the goal. This one-sided,
        # differentiable barrier (smooth so IPOPT converges -- a fmax/fabs kink stalls it)
        # makes over-rearing expensive WITHOUT forbidding the wheelie needed to climb.
        theta_soft = math.radians(cfg.theta_soft_deg)

        def _smooth_relu(s):
            return 0.5 * (s + ca.sqrt(s * s + 1e-4))      # smooth max(0, s)

        def flip_pen(xk):
            # barrier on pitch beyond theta_soft: q_flip * relu(theta^2 - theta_soft^2)^2
            return cfg.q_flip * _smooth_relu(xk[2] ** 2 - theta_soft ** 2) ** 2

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            # NO pitch reference: q_theta is only a mild FLAT preference (penalise pitch on
            # open ground). The wheelie is NOT commanded -- it EMERGES because w_progress
            # rewards forward speed and the learned GP model knows that rearing reduces the
            # obstacle blockage, so rearing at the obstacle is the speed-maximising plan. The
            # controller discovers WHEN and HOW MUCH to climb on its own (no angle/location).
            obj += (cfg.q_x * e[0] ** 2 + cfg.q_v * e[1] ** 2
                    + cfg.q_theta * xk[2] ** 2 + cfg.q_omega * e[3] ** 2)
            # MBRL reward term: maximise forward progress (-> minimise time). The planner
            # trades this against the obstacle deceleration the SSGP predicts, so it discovers
            # the speed-maximising maneuver (a wheelie at the obstacle) on its own.
            obj += cfg.r_tau * uk[0] ** 2
            obj += cfg.r_dtau * du ** 2
            obj += flip_pen(xk)

            x_next = self._rk4_ca(xk, uk, a_rls, gp)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])
        obj += flip_pen(X[:, N])

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