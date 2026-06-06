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


def casadi_gp_mean(z_sym, Z_sym, alpha_sym, lengthscales, sf2):
    """Symbolic RBF/ARD GP mean
        mu(z) = sum_i alpha_i * sf2 * exp(-0.5 ||(z - Z_i)/l||^2)
    over the inducing set (Z, alpha) the streaming sparse GP supplies each solve.
    Defined here so the NMPC has NO gp_residual dependency (uses only the SSGP)."""
    M = Z_sym.shape[0]
    l = ca.DM(np.asarray(lengthscales, dtype=float).reshape(-1))
    out = 0
    for i in range(M):
        diff = (z_sym - Z_sym[i, :].T) / l
        out += alpha_sym[i] * (sf2 * ca.exp(-0.5 * ca.dot(diff, diff)))
    return out


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

    # Goal-distance velocity reference (same idea as the MPPI): the velocity cost
    # tracks  v_ref = clip(v_ref_gain*(goal_x - x), +-v_cruise)  instead of 0, so
    # the truck has authority to move/climb far from the goal but is driven to
    # v=0 AT the goal -> it brakes to a stop instead of overshooting.
    #
    # OFF by default (v_ref_gain=0 -> old pure-setpoint cost q_v*v^2). Reason:
    # enabling it makes the truck want forward speed, and THIS NMPC has no smooth
    # flip penalty -- only a hard theta_max=100deg bound (above the ~90deg
    # tip-over). So it pops a wheelie chasing v_ref and flips, even on flat ground
    # (the MPPI's flip_penalty stops that; the NMPC has nothing equivalent).
    # To use it cleanly the NMPC needs a smooth pitch/flip penalty + retune --
    # see notes in wheelie_gp_climb.py. The MECHANISM is here and correct.
    v_ref_gain: float = 0.0
    v_cruise: float = 1.2

    # Smooth flip penalty (the piece the NMPC lacked vs the MPPI). A one-sided,
    # differentiable barrier on pitch:  q_flip * max(0, |theta| - theta_soft)^2,
    # so popping a wheelie past theta_soft gets expensive -> the solver won't rear
    # over the tip-over point chasing v_ref. q_flip=0 -> off.
    q_flip: float = 0.0
    theta_soft_deg: float = 80.0

    # Smooth velocity barrier (the MPPI has one; the NMPC only had a hard plan
    # bound that reality violates after a contact launch). q_vbar*max(0,v^2-v_hard^2)^2
    # -> the cost itself fights overspeed, killing the post-obstacle runaway.
    q_vbar: float = 0.0
    v_hard: float = 2.5

    # omega-aware flip penalty: penalise ROTATING FURTHER UP while already reared,
    #   q_flipw * max(0, theta - theta_climb) * max(0, omega)
    # -> a tall wheelie that is HELD or RECOVERING (omega<=0) is allowed (needed to
    # climb), but accelerating the rotation toward a tip-over is expensive. This
    # decouples "high wheelie to climb" from "flip", unlike the static theta cap.
    q_flipw: float = 0.0
    theta_climb_deg: float = 55.0

    # warm_start=False -> COLD-START IPOPT every step. A warm start (re-using the
    # previous plan) traps the local solver in the "keep cruising" minimum: once
    # the truck overshoots the goal it never finds the brake/reverse trajectory
    # (the cost screams to reverse, but the solver can't walk there). Cold-starting
    # lets it reverse -> it brakes to a stop, and it's also FASTER here (~10ms).
    warm_start: bool = False


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

        # GP residual feature z = [theta, omega, v, tau] (streaming GP feature order)
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

        def with_vref(xk):
            # replace the velocity-error component (v - ref_v) with (v - v_ref).
            # v_ref saturates to +-v_cruise via tanh (SMOOTH, so IPOPT converges --
            # a clip(fmax/fmin) here stalls it just like a non-smooth penalty):
            #   v_ref = v_cruise * tanh(v_ref_gain*(goal_x - x)/v_cruise)
            vc = cfg.v_cruise
            v_ref = vc * ca.tanh(cfg.v_ref_gain * (ref[0] - xk[0]) / vc) if cfg.v_ref_gain > 0 else 0.0
            return ca.vertcat(xk[0] - ref[0], xk[1] - v_ref,
                              xk[2] - ref[2], xk[3] - ref[3])

        theta_soft = math.radians(cfg.theta_soft_deg)

        def _smooth_relu(s):
            return 0.5 * (s + ca.sqrt(s * s + 1e-4))      # smooth max(0, s)

        v_hard2 = cfg.v_hard ** 2

        def flip_pen(xk):
            # one-sided barrier on theta^2 beyond theta_soft, made SMOOTH so IPOPT
            # (a smooth-NLP solver) converges -- fmax/fabs kinks stall it.
            return cfg.q_flip * _smooth_relu(xk[2] ** 2 - theta_soft ** 2) ** 2

        def vbar_pen(xk):
            # one-sided barrier on v^2 beyond v_hard (anti-runaway), smooth.
            return cfg.q_vbar * _smooth_relu(xk[1] ** 2 - v_hard2) ** 2

        theta_climb = math.radians(cfg.theta_climb_deg)

        def flipw_pen(xk):
            # penalise up-rotation (omega>0) while reared past theta_climb -> stops
            # the tip-over but allows a held/recovering tall wheelie. Smooth.
            return (cfg.q_flipw * _smooth_relu(xk[2] - theta_climb)
                    * _smooth_relu(xk[3]))

        for k in range(N):
            xk, uk = X[:, k], U[:, k]
            e = with_vref(xk)
            du = (uk[0] - tau_prev) if k == 0 else (uk[0] - U[0, k - 1])
            obj += (ca.mtimes([e.T, Q, e]) + cfg.r_tau * uk[0]**2
                    + cfg.r_dtau * du**2
                    + flip_pen(xk) + vbar_pen(xk) + flipw_pen(xk))
            g.append(X[:, k + 1] - self._rk4_ca(xk, uk, a, b, Z, alpha_v, alpha_w))

        eN = with_vref(X[:, N])
        obj += (ca.mtimes([eN.T, Qf, eN]) + flip_pen(X[:, N])
                + vbar_pen(X[:, N]) + flipw_pen(X[:, N]))

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
        if self.cfg.warm_start and self.last_solution is not None:
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
        gp_params: flat GP parameter vector from the streaming GP's mpc_params()

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
