#!/usr/bin/env python3

import math
import numpy as np
from numba import njit, prange

# The GP residual's closed-form mean lives in gp_kernel (shared with GP.py / the NMPC),
# so the kernel math is NOT re-implemented here. `kid` (kernel id, from the GP export)
# selects the kernel shape, so swapping the GP's kernel flows through automatically.
from gp_kernel import gp_residual_mean as _gp


@njit(fastmath=True, cache=True)
def _dyn(x, v, th, om, tau, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
         gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
         gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid):

    residual_v_dot = _gp(
        x, v, th, om, tau,
        gp_points,
        gp_vdot_weights,
        gp_input_mean,
        gp_input_std,
        gp_vdot_lengthscale,
        gp_vdot_signal_var,
        gp_vdot_mean,
        kid,
    ) # RESIDUAL LINEAR ACCELERATION V_DOT

    residual_omega_dot = _gp(
        x, v, th, om, tau,
        gp_points,
        gp_omegadot_weights,
        gp_input_mean,
        gp_input_std,
        gp_omegadot_lengthscale,
        gp_omegadot_signal_var,
        gp_omegadot_mean,
        kid,
    ) # RESIDUAL ANGULAR ACCLERATION OMEGA_DOT

    xd = v

    v_dot = (
        a[0] * tau
        + a[1] * v
        + a[2] * abs(v) * v
        + a[3] * tau * (math.cos(th) - 1.0)
        + a[4]
        + residual_v_dot
    )

    td = om

    omega_dot = (
        a[5] * math.cos(th)
        + a[6] * tau
        + a[7] * om
        + a[8] * v
        + a[9]
        + residual_omega_dot
    )

    return xd, v_dot, td, omega_dot


@njit(fastmath=True, cache=True)
def _rk4(x, v, th, om, tau, dt, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
         gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
         gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid):

    k1 = _dyn(x, v, th, om, tau, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
              gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
              gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid)

    k2 = _dyn(
        x + 0.5 * dt * k1[0],
        v + 0.5 * dt * k1[1],
        th + 0.5 * dt * k1[2],
        om + 0.5 * dt * k1[3],
        tau, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
        gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
        gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid,
    )

    k3 = _dyn(
        x + 0.5 * dt * k2[0],
        v + 0.5 * dt * k2[1],
        th + 0.5 * dt * k2[2],
        om + 0.5 * dt * k2[3],
        tau, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
        gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
        gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid,
    )

    k4 = _dyn(
        x + dt * k3[0],
        v + dt * k3[1],
        th + dt * k3[2],
        om + dt * k3[3],
        tau, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
        gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
        gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid,
    )

    x += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    v += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    th += dt / 6.0 * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    om += dt / 6.0 * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])

    v = min(max(v, -10.0), 10.0)
    om = min(max(om, -30.0), 30.0)

    return x, v, th, om


@njit(fastmath=True, cache=True)
def _cost(x, v, th, om, tau, tau_prev, ref,
          q_x, q_v, q_th, q_om, r_tau, r_dtau,
          q_flip, th_soft, q_flipw, th_climb, th_max):

    ex = x - ref[0]
    ev = v - ref[1]
    eth = th - ref[2]
    eom = om - ref[3]
    du = tau - tau_prev

    J = (
        q_x * ex * ex
        + q_v * ev * ev
        + q_th * eth * eth
        + q_om * eom * eom
        + r_tau * tau * tau
        + r_dtau * du * du
    )

    b = th * th - th_soft * th_soft
    if b > 0.0:
        J += q_flip * b * b

    if q_flipw > 0.0:
        over = th - th_climb
        if over > 0.0 and om > 0.0:
            J += q_flipw * over * om

    if abs(th) > th_max:
        J += 1e4

    return J


@njit(parallel=True, fastmath=True, cache=True)
def _rollout(U, s0, ref, a, gp_points, gp_vdot_weights, gp_omegadot_weights,
             gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
             gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid,
             dt, q_x, q_v, q_th, q_om, r_tau, r_dtau,
             q_flip, th_soft, q_flipw, th_climb, th_max, tau_prev):

    K, N = U.shape
    S = np.zeros(K)

    for k in prange(K):
        x, v, th, om = s0[0], s0[1], s0[2], s0[3]
        prev = tau_prev

        for n in range(N):
            tau = U[k, n]

            x, v, th, om = _rk4(
                x, v, th, om, tau, dt,
                a, gp_points, gp_vdot_weights, gp_omegadot_weights,
                gp_input_mean, gp_input_std, gp_vdot_lengthscale, gp_omegadot_lengthscale,
                gp_vdot_signal_var, gp_omegadot_signal_var, gp_vdot_mean, gp_omegadot_mean, kid,
            )

            S[k] += _cost(
                x, v, th, om, tau, prev, ref,
                q_x, q_v, q_th, q_om, r_tau, r_dtau,
                q_flip, th_soft, q_flipw, th_climb, th_max,
            )

            prev = tau

        ex = x - ref[0]
        ev_ = v - ref[1]
        S[k] += q_x * ex * ex + q_v * ev_ * ev_

    return S


class MPPI:
    K = 1024
    SIGMA = 10.0
    LAM = 1.0
    BETA = 0.8
    SEED = 0

    def __init__(self, p, cfg, gp_cfg):
        self.p = p
        self.cfg = cfg
        self.gp_cfg = gp_cfg

        self.M = int(gp_cfg.max_points)
        self.d = int(gp_cfg.n_features)

        self.last_solution = None
        self.rng = np.random.default_rng(self.SEED)

    def solve(self, state, ref, tau_prev, a_rls, gp_params):
        p = self.p
        cfg = self.cfg

        N = int(cfg.N)
        dt = float(cfg.dt)

        tau_prev = float(tau_prev)
        tau_min = float(p.tau_min)
        tau_max = float(p.tau_max)

        s0 = np.ascontiguousarray(state, dtype=float).reshape(-1)
        ref = np.ascontiguousarray(ref, dtype=float).reshape(-1)
        a = np.ascontiguousarray(a_rls, dtype=float).reshape(-1)

        (
            gp_points,
            gp_vdot_weights,
            gp_omegadot_weights,
            gp_input_mean,
            gp_input_std,
            gp_vdot_lengthscale,
            gp_omegadot_lengthscale,
            gp_vdot_signal_var,
            gp_omegadot_signal_var,
            gp_vdot_mean,
            gp_omegadot_mean,
            kid,
        ) = self._unpack_gp(gp_params)

        U_nom = self._warm_start(N, tau_prev)
        U = self._sample_controls(U_nom, tau_min, tau_max)
        eps = U - U_nom[None, :]

        S = _rollout(
            U, s0, ref, a,
            gp_points,
            gp_vdot_weights,
            gp_omegadot_weights,
            gp_input_mean,
            gp_input_std,
            gp_vdot_lengthscale,
            gp_omegadot_lengthscale,
            gp_vdot_signal_var,
            gp_omegadot_signal_var,
            gp_vdot_mean,
            gp_omegadot_mean,
            kid,
            dt,
            float(cfg.q_x),
            float(cfg.q_v),
            float(cfg.q_theta),
            float(cfg.q_omega),
            float(cfg.r_tau),
            float(cfg.r_dtau),
            float(cfg.q_flip),
            math.radians(float(cfg.theta_soft_deg)),
            float(getattr(cfg, "q_flipw", 0.0)),
            math.radians(float(getattr(cfg, "theta_climb_deg", 90.0))),
            float(p.theta_max),
            tau_prev,
        )

        S = np.nan_to_num(S, nan=1e12, posinf=1e12, neginf=1e12)

        rho = S.min()
        scale = max(np.median(S) - rho, 1e-6)

        w = np.exp(-(S - rho) / (self.LAM * scale))
        w /= np.sum(w) + 1e-12

        U_opt = U_nom + w @ eps
        U_opt = np.clip(np.nan_to_num(U_opt, nan=tau_prev), tau_min, tau_max)

        self.last_solution = U_opt

        return float(U_opt[0]), {"success": True, "cost": float(rho)}

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return np.full(N, tau_prev)

        return np.concatenate([self.last_solution[1:], self.last_solution[-1:]])

    def _sample_controls(self, U_nom, tau_min, tau_max):
        K = self.K
        N = U_nom.size

        raw = self.rng.normal(0.0, self.SIGMA, size=(K, N))
        eps = np.empty((K, N))

        eps[:, 0] = raw[:, 0]

        c = math.sqrt(1.0 - self.BETA ** 2)

        for n in range(1, N):
            eps[:, n] = self.BETA * eps[:, n - 1] + c * raw[:, n]

        U = U_nom[None, :] + eps
        return np.ascontiguousarray(np.clip(U, tau_min, tau_max))

    def _unpack_gp(self, gp_params):
        g = np.asarray(gp_params, dtype=float).reshape(-1)

        M = self.M
        d = self.d
        i = 0

        gp_points = np.ascontiguousarray(g[i:i + M * d].reshape(M, d)); i += M * d
        gp_vdot_weights = np.ascontiguousarray(g[i:i + M]); i += M
        gp_omegadot_weights = np.ascontiguousarray(g[i:i + M]); i += M
        gp_input_mean = np.ascontiguousarray(g[i:i + d]); i += d
        gp_input_std = np.ascontiguousarray(g[i:i + d]); i += d
        gp_vdot_lengthscale = np.ascontiguousarray(g[i:i + d]); i += d
        gp_omegadot_lengthscale = np.ascontiguousarray(g[i:i + d]); i += d

        gp_vdot_signal_var = float(g[i]); i += 1
        gp_omegadot_signal_var = float(g[i]); i += 1
        gp_vdot_mean = float(g[i]); i += 1
        gp_omegadot_mean = float(g[i]); i += 1
        kid = int(round(g[i]))                    # trailing kernel_id (see GP.mpc_params)

        return (
            gp_points,
            gp_vdot_weights,
            gp_omegadot_weights,
            gp_input_mean,
            gp_input_std,
            gp_vdot_lengthscale,
            gp_omegadot_lengthscale,
            gp_vdot_signal_var,
            gp_omegadot_signal_var,
            gp_vdot_mean,
            gp_omegadot_mean,
            kid,
        )