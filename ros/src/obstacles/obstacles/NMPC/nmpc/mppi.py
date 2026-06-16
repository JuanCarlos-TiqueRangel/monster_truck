#!/usr/bin/env python3
"""
mppi.py
-------
Sampling-based MPPI controller -- a DROP-IN replacement for nmpc.NMPC. Same
constructor (p, cfg, gp_cfg), same solve(state, ref, tau_prev, a_rls, gp_params)
-> (tau, info), same `last_solution` attribute (set to None to reset the warm start).

It uses the SAME predictor as the NMPC so the two are directly comparable:

    x_dot     = v
    v_dot     = b . phi_v(tau, v, theta) + gp_v(x, theta)
    theta_dot = omega
    omega_dot = a . phi_w(theta, omega, v, tau) + gp_w(x, theta)

with the RLS coefficients a_rls passed in and the GP residual from the streaming
sparse GP (gp_params = [Z, alpha_v_dot, alpha_omega_dot] from gp.mpc_params()). The stage/terminal
cost matches the NMPC's (goal-reaching Q + control + smooth flip penalties), so the
ONLY difference is the solver: information-theoretic MPPI (sample K control sequences,
roll them out, exponentially weight by cost) instead of IPOPT.

Everything is vectorised over the K samples in numpy (no casadi here).
"""

import math

import numpy as np


class MPPI:
    # MPPI hyperparameters (tune here).
    K = 256               # number of sampled control sequences (raise for quality)
    SIGMA = 1.5           # exploration std on tau [Nm]
    LAM = 1.0             # temperature, RELATIVE to the (robust) cost spread
    BETA = 0.8            # noise time-correlation (0=white; high=smooth coherent maneuvers)
    SEED = 0

    def __init__(self, p, cfg, gp_cfg):
        self.p = p
        self.cfg = cfg
        self.gp_cfg = gp_cfg
        self.M = gp_cfg.max_points
        self.d = len(gp_cfg.lengthscales)
        self.gp_l = np.asarray(gp_cfg.lengthscales, dtype=float)
        self.sf2 = float(gp_cfg.sf2)
        self.last_solution = None             # nominal control seq (N,), for warm start
        self.rng = np.random.default_rng(self.SEED)

    # ── GP residual mean, vectorised over the K rollout samples ──────────────
    def _gp(self, x, theta, Z, alpha):
        # x, theta: (K,)   Z: (M, 2)   alpha: (M,)  ->  (K,)
        zq = np.stack([x, theta], axis=1)                      # (K, 2)
        diff = (zq[:, None, :] - Z[None, :, :]) / self.gp_l    # (K, M, 2)
        Kqz = self.sf2 * np.exp(-0.5 * np.sum(diff * diff, axis=2))   # (K, M)
        return Kqz @ alpha                                     # (K,)

    def _dyn(self, x, v, th, om, tau, a, Z, alpha_v_dot, alpha_omega_dot):
        gp_v_dot = self._gp(x, th, Z, alpha_v_dot)
        gp_omega_dot = self._gp(x, th, Z, alpha_omega_dot)
        xd = v
        # v_dot: traction(tau) + linear drag(v) + quad drag(|v|v) + traction roll-off(tau*(cos-1)) + bias
        vd = a[0] * tau + a[1] * v + a[2] * np.abs(v) * v + a[3] * tau * (np.cos(th) - 1.0) + a[4] + gp_v_dot
        td = om
        # omega_dot: gravity(cos) + reaction torque(tau) + pitch damping(om) + speed coupling(v) + bias
        od = a[5] * np.cos(th) + a[6] * tau + a[7] * om + a[8] * v + a[9] + gp_omega_dot
        return xd, vd, td, od

    def _rk4(self, x, v, th, om, tau, a, Z, alpha_v_dot, alpha_omega_dot, dt):
        k1 = self._dyn(x, v, th, om, tau, a, Z, alpha_v_dot, alpha_omega_dot)
        k2 = self._dyn(x + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1],
                       th + 0.5 * dt * k1[2], om + 0.5 * dt * k1[3], tau, a, Z, alpha_v_dot, alpha_omega_dot)
        k3 = self._dyn(x + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1],
                       th + 0.5 * dt * k2[2], om + 0.5 * dt * k2[3], tau, a, Z, alpha_v_dot, alpha_omega_dot)
        k4 = self._dyn(x + dt * k3[0], v + dt * k3[1],
                       th + dt * k3[2], om + dt * k3[3], tau, a, Z, alpha_v_dot, alpha_omega_dot)
        x = x + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        v = v + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        th = th + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        om = om + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        return x, v, th, om

    def solve(self, state, ref, tau_prev, a_rls, gp_params):
        p, cfg, K = self.p, self.cfg, self.K
        N, dt = cfg.N, cfg.dt
        a = np.asarray(a_rls, dtype=float).reshape(-1)
        state = np.asarray(state, dtype=float).reshape(-1)
        ref = np.asarray(ref, dtype=float).reshape(-1)
        tau_prev = float(tau_prev)

        # unpack GP params (= gp.mpc_params(): [Z(M*d), alpha_v_dot(M), alpha_omega_dot(M)])
        gp = np.asarray(gp_params, dtype=float).reshape(-1)
        M, d = self.M, self.d
        Z = gp[:M * d].reshape(M, d)
        alpha_v_dot = gp[M * d:M * d + M]
        alpha_omega_dot = gp[M * d + M:M * d + 2 * M]

        # nominal control: warm-start by shifting the previous plan
        if self.last_solution is not None:
            U_nom = np.concatenate([self.last_solution[1:], self.last_solution[-1:]])
        else:
            U_nom = np.full(N, tau_prev)

        # sample K perturbed control sequences with TIME-CORRELATED noise (AR(1)), so
        # each sample is a smooth, coherent maneuver -- white per-step noise almost
        # never produces the sustained rear needed to climb a box.
        raw = self.rng.normal(0.0, self.SIGMA, size=(K, N))
        eps = np.empty((K, N))
        eps[:, 0] = raw[:, 0]
        a_c = np.sqrt(1.0 - self.BETA**2)
        for k in range(1, N):
            eps[:, k] = self.BETA * eps[:, k - 1] + a_c * raw[:, k]
        U = np.clip(U_nom[None, :] + eps, p.tau_min, p.tau_max)
        eps = U - U_nom[None, :]                 # actual perturbation after clipping

        # roll out all K samples, accumulate cost (same cost as the NMPC)
        x = np.full(K, state[0]); v = np.full(K, state[1])
        th = np.full(K, state[2]); om = np.full(K, state[3])
        S = np.zeros(K)
        theta_soft = math.radians(cfg.theta_soft_deg)
        theta_climb = math.radians(cfg.theta_climb_deg)
        prev_tau = np.full(K, tau_prev)
        for k in range(N):
            tau = U[:, k]
            x, v, th, om = self._rk4(x, v, th, om, tau, a, Z, alpha_v_dot, alpha_omega_dot, dt)
            # clamp the rollout state so a divergent sample (e.g. quadratic drag with a
            # wrong-sign RLS coeff) can't blow up to inf/NaN -- it just gets a high cost.
            v = np.clip(v, -10.0, 10.0); om = np.clip(om, -30.0, 30.0)
            ex, ev, et, eo = x - ref[0], v - ref[1], th - ref[2], om - ref[3]
            du = tau - prev_tau
            S += cfg.q_x * ex**2 + cfg.q_v * ev**2 + cfg.q_theta * et**2 + cfg.q_omega * eo**2
            S += cfg.r_tau * tau**2 + cfg.r_dtau * du**2
            S += cfg.q_flip * np.maximum(0.0, th**2 - theta_soft**2)**2          # flip barrier
            S += cfg.q_flipw * np.maximum(0.0, th - theta_climb) * np.maximum(0.0, om)
            S += 1e4 * (np.abs(th) > p.theta_max)                                # hard flip bound
            prev_tau = tau
        # terminal (driver sets terminal theta/omega weights to 0 -> position + velocity)
        S += cfg.q_x * (x - ref[0])**2 + cfg.q_v * (v - ref[1])**2

        # sanitise: any non-finite rollout cost -> huge (that sample gets ~0 weight)
        S = np.nan_to_num(S, nan=1e12, posinf=1e12, neginf=1e12)

        # information-theoretic weighting. Scale the temperature by the spread of the
        # BETTER-than-median samples (robust to flip-cost outliers, which would
        # otherwise inflate the scale and wash the weights out toward uniform).
        rho = S.min()
        scale = max(np.median(S) - rho, 1e-6)
        w = np.exp(-(S - rho) / (self.LAM * scale))
        w /= w.sum() + 1e-12

        U_opt = U_nom + (w[:, None] * eps).sum(axis=0)
        U_opt = np.nan_to_num(U_opt, nan=tau_prev)
        U_opt = np.clip(U_opt, p.tau_min, p.tau_max)
        self.last_solution = U_opt

        tau0 = float(np.clip(U_opt[0], p.tau_min, p.tau_max))
        return tau0, {"success": True, "cost": float(rho)}
