#!/usr/bin/env python3
"""
nmpc.py
-------
Nonlinear Model Predictive Controller for the wheelie / obstacle-climb task.

The predictor used inside the MPC is:

    x_dot     = v
    v_dot     = b . phi_v(theta, v, tau)        + gp_v(z)
    theta_dot = omega
    omega_dot = a . phi_w(theta, omega, v, tau) + gp_omega(z)

where
  * a, b               are the RLS-estimated linear/angular coefficients,
                       passed in as PARAMETERS so they update online without
                       rebuilding the solver,
  * gp_v, gp_omega     are the sparse-GP residual means (contact dynamics),
                       also passed as parameters (inducing points + weights),
  * phi_v, phi_w       are the shared regressors defined in rls.py, so the
                       MPC model is identical to what RLS identifies.

This file owns the model parameters, the cost function and the IPOPT solver.
It deliberately knows nothing about MuJoCo or the mission supervisor.
"""

import math
from dataclasses import dataclass

import numpy as np
import casadi as ca

from rls import omega_regressor, v_regressor, N_OMEGA_FEATURES, N_V_FEATURES
from gp_residual import casadi_gp_mean


# ============================================================
# Physical model parameters and actuator / state limits
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
    theta_max: float = math.radians(100.0)
    omega_min: float = -8.0
    omega_max: float = 8.0
    v_min: float = -5.0
    v_max: float = 5.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.05
    N: int = 10
    q_x: float = 15.0
    q_v: float = 55.0
    q_theta: float = 540.0
    q_omega: float = 0.1
    r_tau: float = 0.5
    r_dtau: float = 2.5
    q_terminal_theta: float = 250.0
    q_terminal_omega: float = 0.1
    ipopt_max_iter: int = 50


def nominal_rls_seeds(p: WheelieParams):
    """
    Physics-based initial guesses for the two RLS estimators.

    Returns (a0, b0):
        a0 : angular model seed for phi_w = [cos(theta), tau, omega, v, 1]
        b0 : linear  model seed for phi_v = [tau, v, sin(theta), 1]
    """
    a0 = np.array([p.m * p.g * p.l / p.I_eff,   # cos(theta) -> gravity torque
                   -1.0 / p.I_eff,              # tau        -> drive reaction
                   0.0,                         # omega
                   0.0,                         # v
                   0.0])                        # offset
    b0 = np.array([1.0 / (p.m * p.r),           # tau        -> drive accel
                   -p.c_v,                       # v          -> drag
                   0.0,                          # sin(theta) -> gravity proj.
                   0.0])                         # offset
    return a0, b0


# ============================================================
# NMPC controller
# ============================================================

class WheelieNMPC:
    """
    Builds the IPOPT NLP once; each step only the numeric parameters change
    (current state, reference, RLS coefficients, GP dictionary).
    """

    # Physical bounds on the GP residual contribution (safety clamp).
    # Keeps a mis-learned or extrapolating GP from injecting unphysical
    # accelerations into the prediction model.
    GP_V_CLAMP = 1.5 * 9.81      # max |longitudinal residual accel|  [m/s^2]
    GP_OMEGA_CLAMP = 8.0         # max |pitch residual accel|         [rad/s^2]

    def __init__(self, p: WheelieParams, cfg: MPCConfig, gp_cfg):
        self.p = p
        self.cfg = cfg
        self.gp_cfg = gp_cfg
        self.nx = 4
        self.nu = 1
        self.n_a = N_OMEGA_FEATURES          # angular coefficients
        self.n_b = N_V_FEATURES              # linear  coefficients
        self.M = gp_cfg.max_points           # GP inducing points
        self.d = 4                           # GP feature dimension
        self.gp_l = np.asarray(gp_cfg.lengthscales, dtype=float)
        self.last_solution = None
        self._build_solver()

    # ── Dynamics ─────────────────────────────────────────────────────────────

    @staticmethod
    def _soft_clip(val, lim):
        """Smooth, differentiable clamp to [-lim, lim] (tanh saturation)."""
        return lim * ca.tanh(val / lim)

    def _f_ca(self, x, u, a, b, Z, alpha_v, alpha_w):
        """Continuous-time dynamics x_dot = f(x, u) with RLS model + GP residual."""
        theta, omega, v, tau = x[2], x[3], x[1], u[0]

        # GP residual feature z = [theta, omega, v, tau] (matches GPResidual)
        z = ca.vertcat(theta, omega, v, tau)
        gp_v = self._soft_clip(
            casadi_gp_mean(z, Z, alpha_v, self.gp_l, self.gp_cfg.sf2),
            self.GP_V_CLAMP)
        gp_w = self._soft_clip(
            casadi_gp_mean(z, Z, alpha_w, self.gp_l, self.gp_cfg.sf2),
            self.GP_OMEGA_CLAMP)

        phi_v = ca.vertcat(*v_regressor(theta, v, tau, sin=ca.sin))
        phi_w = ca.vertcat(*omega_regressor(theta, omega, v, tau, cos=ca.cos))

        x_dot = v
        v_dot = ca.dot(b, phi_v) + gp_v
        theta_dot = omega
        omega_dot = ca.dot(a, phi_w) + gp_w
        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u, a, b, Z, alpha_v, alpha_w):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u, a, b, Z, alpha_v, alpha_w)
        k2 = self._f_ca(x + 0.5 * dt * k1, u, a, b, Z, alpha_v, alpha_w)
        k3 = self._f_ca(x + 0.5 * dt * k2, u, a, b, Z, alpha_v, alpha_w)
        k4 = self._f_ca(x + dt * k3, u, a, b, Z, alpha_v, alpha_w)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ── Solver construction ──────────────────────────────────────────────────

    def _build_solver(self):
        cfg, p, N = self.cfg, self.p, self.cfg.N
        M, d = self.M, self.d

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter layout:
        #   state(4) ref(4) tau_prev(1) a(n_a) b(n_b) | Z(M*d) alpha_v(M) alpha_w(M)
        n_base = 9 + self.n_a + self.n_b
        P = ca.SX.sym("P", n_base + M * d + 2 * M)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        i = 9
        a = P[i:i + self.n_a];           i += self.n_a
        b = P[i:i + self.n_b];           i += self.n_b
        Z = ca.reshape(P[i:i + M * d], M, d);  i += M * d
        alpha_v = P[i:i + M];            i += M
        alpha_w = P[i:i + M]

        obj = 0
        g = [X[:, 0] - x0]

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v,
                                cfg.q_terminal_theta, cfg.q_terminal_omega))

        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            e = xk - ref
            du = (uk[0] - tau_prev) if k == 0 else (uk[0] - U[0, k - 1])
            obj += ca.mtimes([e.T, Q, e]) + cfg.r_tau * uk[0]**2 + cfg.r_dtau * du**2
            g.append(X[:, k + 1] - self._rk4_ca(xk, uk, a, b, Z, alpha_v, alpha_w))

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {"f": obj, "x": opt_vars, "g": ca.vertcat(*g), "p": P}
        opts = {"ipopt.print_level": 0, "ipopt.max_iter": cfg.ipopt_max_iter,
                "ipopt.tol": 1e-4, "print_time": 0}
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        self.nX = self.nx * (N + 1)
        lbx, ubx = [], []
        for _ in range(N + 1):
            lbx += [-ca.inf, p.v_min, p.theta_min, p.omega_min]
            ubx += [ca.inf, p.v_max, p.theta_max, p.omega_max]
        for _ in range(N):
            lbx += [p.tau_min]; ubx += [p.tau_max]
        self.lbx = np.array(lbx); self.ubx = np.array(ubx)
        self.lbg = np.zeros(self.nx * (N + 1))
        self.ubg = np.zeros(self.nx * (N + 1))

    # ── Warm start ───────────────────────────────────────────────────────────

    def _initial_guess(self, state, tau_prev):
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
        return np.concatenate([X_guess.reshape(-1, order="F"),
                               U_guess.reshape(-1, order="F")])

    # ── Solve ────────────────────────────────────────────────────────────────

    def solve(self, state, ref, tau_prev, a, b, gp_params):
        """
        Parameters
        ----------
        state    : (4,) current [x, v, theta, omega]
        ref      : (4,) target  [x, v, theta, omega]
        tau_prev : float, last applied torque (for du penalty / warm start)
        a        : (n_a,) angular RLS coefficients
        b        : (n_b,) linear  RLS coefficients
        gp_params: flat GP parameter vector from GPResidual.mpc_params()

        Returns (tau, info).
        """
        params = np.concatenate([
            np.asarray(state, dtype=float).reshape(-1),
            np.asarray(ref, dtype=float).reshape(-1),
            np.array([tau_prev], dtype=float),
            np.asarray(a, dtype=float).reshape(-1),
            np.asarray(b, dtype=float).reshape(-1),
            np.asarray(gp_params, dtype=float).reshape(-1),
        ])
        x_init = self._initial_guess(np.asarray(state, dtype=float), tau_prev)
        try:
            sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx,
                              lbg=self.lbg, ubg=self.ubg, p=params)
            w = np.array(sol["x"]).flatten()
            self.last_solution = w
            U_opt = w[self.nX:].reshape((self.nu, self.cfg.N), order="F")
            tau = float(np.clip(U_opt[0, 0], self.p.tau_min, self.p.tau_max))
            return tau, {"success": True, "cost": float(sol["f"])}
        except RuntimeError as exc:
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "error": str(exc)}
