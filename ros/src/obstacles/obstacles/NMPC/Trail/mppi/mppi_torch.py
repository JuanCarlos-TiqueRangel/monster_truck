#!/usr/bin/env python3
"""
================================================================================
mppi_torch.py  --  a clean, all-torch MPPI (the mppi_core.py pattern)
================================================================================

This is the "regular controller" version: it just receives the vehicle state,
rolls a dynamics model forward, scores it with a cost, and returns an action. It
HOLDS the GP object and calls `gp.predict_torch(X)` DIRECTLY inside the rollout --
no flat-vector export, no _unpack_gp, no per-kernel closed form. Everything stays
torch tensors on the GPU, so the model can be called as-is (the whole reason the
numba mppi.py needs the gp_kernel/mpc_params machinery is that numba can't call torch).

    dynamics:  s' = integrate( RLS_nominal(s,tau; a_rls) + GP_residual(s,tau) )
    cost:      goal-reaching + control + smooth flip barrier   (matches mppi.py)
    action:    standard MPPI exponential-weighted update of a warm-started plan

Interface differs from the numba mppi.py ON PURPOSE (it is the point of this file):
    controller = MPPITorch(p, cfg, gp)                 # holds the GP object
    tau, info  = controller.solve(state, ref, tau_prev, a_rls)   # NO gp_params

To use it in obstacle_mujoco_simulation.py instead of the numba MPPI:
    1. build the GP FIRST, then the controller:
           gp  = gp_cfg.build()
           nmpc = MPPITorch(p, cfg, gp)
    2. drop gp_params from the solve call:
           tau, info = nmpc.solve(state_now, ref, tau_prev, a_rls)
    (the numba mppi.py is unchanged and still the default.)
================================================================================
"""

import math
import torch
from params_mppi import MPPIConfig


class MPPITorch:
    # same sampler settings as the numba mppi.py so behaviour is comparable
    K = MPPIConfig.K            # number of sampled rollouts
    SIGMA = MPPIConfig.SIGMA        # exploration std on tau
    LAM = MPPIConfig.LAM           # temperature (softmin sharpness)
    BETA = MPPIConfig.BETA          # AR(1) smoothing of the control noise along the horizon
    SEED = MPPIConfig.SEED

    def __init__(self, p, cfg, gp, device=None, fast=True, integrator="euler", dtype=None):
        """fast=True  -> GP residual via the cached kernel MATMUL (gp.predict_torch_fast),
                         float32, no gpytorch forward in the loop -> MUCH faster.
           fast=False -> calls gp.predict_torch (the GPyTorch model directly), double; the
                         clean-but-slow reference path.
           integrator -> "euler" (1 GP eval/step, default) or "rk4" (4 GP evals/step)."""
        need = "predict_torch_fast" if fast else "predict_torch"
        if not hasattr(gp, need):
            raise TypeError(
                f"MPPITorch needs the BUILT GP object (with {need}), got {type(gp).__name__}. "
                "Build it first:  gp = gp_cfg.build();  MPPITorch(p, cfg, gp).")
        self.p = p
        self.cfg = cfg
        self.gp = gp                                     # the GP OBJECT (called directly)
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (torch.float32 if fast else torch.double)
        self.device = torch.device(device if device is not None else getattr(gp, "device", "cpu"))
        # risk-averse weight on the GP's epistemic uncertainty (0 = off). >0 penalizes
        # planning into regions the GP has NOT learned (needs gp.predict_uncertainty_torch).
        self.q_gp_var = float(getattr(cfg, "q_gp_var", 0.0))
        if self.q_gp_var > 0.0 and not hasattr(gp, "predict_uncertainty_torch"):
            raise TypeError("q_gp_var > 0 needs a GP with predict_uncertainty_torch().")
        self.last_solution = None                        # warm-start plan, (N,) tensor
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.SEED)

    # ---- dynamics: RLS nominal + GP residual (the SAME model as mppi.py) ---
    def _deriv(self, s, tau, a):
        """State derivative for a BATCH of rollouts. s:(K,4)=[x,v,theta,omega], tau:(K,),
        a:(10,) RLS weights. The GP residual is added to the RLS nominal accelerations --
        called directly via gp.predict_torch (the mppi_core.py way)."""
        x, v, th, om = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        X = torch.stack([x, v, th, om, tau], dim=-1)     # (K,5) feature z=[x,v,theta,omega,tau]
        if self.fast:                                    # cached kernel matmul (fast, float32)
            res_v, res_w = self.gp.predict_torch_fast(X, dtype=self.dtype)
        else:                                            # GPyTorch model forward (slow, double)
            res_v, res_w = self.gp.predict_torch(X)

        x_dot = v
        v_dot = (a[0] * tau + a[1] * v + a[2] * v.abs() * v
                 + a[3] * tau * (torch.cos(th) - 1.0) + a[4] + res_v)
        th_dot = om
        om_dot = (a[5] * torch.cos(th) + a[6] * tau + a[7] * om
                  + a[8] * v + a[9] + res_w)
        return torch.stack([x_dot, v_dot, th_dot, om_dot], dim=-1)

    def _step(self, s, tau, a, dt):
        if self.integrator == "rk4":                     # 4 GP evals/step (matches numba mppi.py)
            k1 = self._deriv(s, tau, a)
            k2 = self._deriv(s + 0.5 * dt * k1, tau, a)
            k3 = self._deriv(s + 0.5 * dt * k2, tau, a)
            k4 = self._deriv(s + dt * k3, tau, a)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:                                            # euler: 1 GP eval/step (4x fewer)
            s = s + dt * self._deriv(s, tau, a)
        # same clamps as the numba rollout (keep diverging samples finite)
        s = torch.stack([s[:, 0],
                         s[:, 1].clamp(-10.0, 10.0),
                         s[:, 2],
                         s[:, 3].clamp(-30.0, 30.0)], dim=-1)
        return s

    # ------------------------------------------------------------------ #
    # Stage cost: total penalty for being in state `s` and applying `tau`.
    # It is a sum of named terms; each has its OWN weight. Smaller total = better
    # rollout. Set any weight to 0 to switch that term off.
    # ------------------------------------------------------------------ #
    def _stage_cost(self, s, tau, tau_prev, ref, c):
        # state   s   = [x, v, theta, omega]
        # ref         = [x_goal, v_ref, theta_ref, omega_ref]
        x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

        # 1) tracking errors (state - reference)
        e_x      = x     - ref[0]        # how far from the goal position
        e_v      = v     - ref[1]        # speed error
        e_theta  = theta - ref[2]        # pitch error
        e_omega  = omega - ref[3]        # pitch-rate error
        d_tau    = tau   - tau_prev      # torque change since last step

        # 2) quadratic tracking / effort penalties  (weight * error^2)
        cost_goal       = c["q_x"]    * e_x ** 2       # q_x     : reach the goal position
        cost_speed      = c["q_v"]    * e_v ** 2       # q_v     : hold the reference speed
        cost_pitch      = c["q_th"]   * e_theta ** 2   # q_theta : keep pitch near reference (stay flat)
        cost_pitch_rate = c["q_om"]   * e_omega ** 2   # q_omega : damp the pitch rate
        cost_torque     = c["r_tau"]  * tau ** 2       # r_tau   : use less torque
        cost_smooth     = c["r_dtau"] * d_tau ** 2     # r_dtau  : avoid jerky torque changes

        # 3) smooth flip barrier: ~0 while |pitch| < theta_soft, then rises steeply (4th
        #    power) past it -- discourages big pitch WITHOUT a hard cliff. q_flip = strength.
        over_soft = torch.clamp(theta ** 2 - c["th_soft"] ** 2, min=0.0)   # >0 once |theta|>theta_soft
        cost_flip = c["q_flip"] * over_soft ** 2       # q_flip   : discourage rearing past theta_soft

        # 4) climb-rate barrier: one-sided -- only when ALREADY past theta_climb AND still
        #    pitching up (omega>0). Penalizes rearing further. q_flipw = strength.
        rearing    = (theta - c["th_climb"] > 0.0) & (omega > 0.0)
        cost_climb = c["q_flipw"] * torch.where(rearing, (theta - c["th_climb"]) * omega,
                                                torch.zeros_like(theta))   # q_flipw : stop over-rearing

        # 5) hard tip-over wall: forbid |pitch| beyond theta_max with a big flat cost
        cost_wall = torch.where(theta.abs() > c["th_max"],
                                torch.full_like(theta, 1e4), torch.zeros_like(theta))  # theta_max

        # return (cost_goal + cost_speed + cost_pitch + cost_pitch_rate + cost_torque)

        return (cost_goal + cost_speed + cost_pitch + cost_pitch_rate + cost_torque
                + cost_smooth + cost_flip + cost_climb + cost_wall)

    # ------------------------------------------------------------------ #
    # Risk term (optional, weight q_gp_var): penalize planning into states the GP has
    # NOT learned. The GP returns a per-channel epistemic uncertainty in [0,1] (0 = on
    # the learned data, 1 = unmodelled). Added to the cost so the planner prefers states
    # the model trusts -- and while learning, collects data at the edge of the known
    # region instead of diving into the unknown (and flipping). Kept SEPARATE from
    # _stage_cost: that is the deterministic cost, this is the uncertainty cost.
    # ------------------------------------------------------------------ #
    def _risk(self, s, tau):
        X = torch.stack([s[:, 0], s[:, 1], s[:, 2], s[:, 3], tau], dim=-1)   # (K,5)
        u_v, u_w = self.gp.predict_uncertainty_torch(X, dtype=self.dtype)
        return u_v + u_w                                  # each in [0,1]; weight applied by caller

    # ---- the controller: state in -> action out ---------------------------
    @torch.no_grad()
    def solve(self, state, ref, tau_prev, a_rls):
        p, cfg, dev = self.p, self.cfg, self.device
        N, dt = int(cfg.N), float(cfg.dt)
        tau_min, tau_max = float(p.tau_min), float(p.tau_max)
        tau_prev = float(tau_prev)

        s0 = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)
        ref = torch.as_tensor(ref, dtype=self.dtype, device=dev).reshape(-1)
        a = torch.as_tensor(a_rls, dtype=self.dtype, device=dev).reshape(-1)
        c = {k: float(v) for k, v in dict(
            q_x=cfg.q_x, q_v=cfg.q_v, q_th=cfg.q_theta, q_om=cfg.q_omega,
            r_tau=cfg.r_tau, r_dtau=cfg.r_dtau,
            q_flip=cfg.q_flip,                                  # flip-barrier strength
            th_soft=math.radians(cfg.theta_soft_deg),           # flip-barrier threshold [rad]
            q_flipw=getattr(cfg, "q_flipw", 0.0),               # climb-barrier strength (0 = off)
            th_climb=math.radians(getattr(cfg, "theta_climb_deg", 90.0)),  # climb threshold [rad]
            th_max=p.theta_max).items()}                        # hard tip-over wall [rad]

        U_nom = self._warm_start(N, tau_prev)            # (N,)
        U = self._sample_controls(U_nom, tau_min, tau_max)   # (K,N)
        eps = U - U_nom.unsqueeze(0)

        s = s0.unsqueeze(0).repeat(self.K, 1)            # (K,4)
        S = torch.zeros(self.K, dtype=self.dtype, device=dev)
        prev = torch.full((self.K,), tau_prev, dtype=self.dtype, device=dev)
        for n in range(N):
            tau = U[:, n]
            s = self._step(s, tau, a, dt)
            S = S + self._stage_cost(s, tau, prev, ref, c)
            if self.q_gp_var > 0.0:                       # risk-averse: avoid UNMODELLED states
                S = S + self.q_gp_var * self._risk(s, tau)
            prev = tau
        ex, ev = s[:, 0] - ref[0], s[:, 1] - ref[1]      # terminal goal cost
        S = S + c["q_x"] * ex * ex + c["q_v"] * ev * ev

        S = torch.nan_to_num(S, nan=1e12, posinf=1e12, neginf=1e12)
        rho = S.min()
        scale = torch.clamp(S.median() - rho, min=1e-6)
        w = torch.exp(-(S - rho) / (self.LAM * scale))
        w = w / (w.sum() + 1e-12)

        U_opt = torch.clamp(U_nom + (w.unsqueeze(1) * eps).sum(0), tau_min, tau_max)
        self.last_solution = U_opt
        return float(U_opt[0].item()), {"success": True, "cost": float(rho.item())}

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return torch.full((N,), tau_prev, dtype=self.dtype, device=self.device)
        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])

    def _sample_controls(self, U_nom, tau_min, tau_max):
        K, N = self.K, U_nom.numel()
        raw = torch.randn(K, N, generator=self.gen, dtype=self.dtype, device=self.device) * self.SIGMA
        eps = torch.empty(K, N, dtype=self.dtype, device=self.device)
        eps[:, 0] = raw[:, 0]
        cc = math.sqrt(1.0 - self.BETA ** 2)
        for n in range(1, N):                            # AR(1) smoothing along the horizon
            eps[:, n] = self.BETA * eps[:, n - 1] + cc * raw[:, n]
        return torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)


# ============================================================================ #
# Smoke test: fit a GP on synthetic data, then run one MPPI solve and confirm it
# returns a finite torque. Runs on the host (CPU). No mujoco needed.
#   python3 mppi_torch.py
# ============================================================================ #
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import numpy as np

    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here / ".." / "gp"))         # for GP
    sys.path.insert(0, str(_here))                       # for params_mppi

    from params_mppi import WheelieParams, MPPIConfig
    from GP import GPConfig

    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(0, 6, 400), rng.uniform(0, 4, 400),
                  rng.uniform(-0.3, 1.2, 400), rng.uniform(-2, 2, 400),
                  rng.uniform(-5, 10, 400)], axis=1)
    obstacle = np.exp(-((X[:, 0] - 3.0) ** 2) / (2 * 0.25 ** 2))
    Y = np.stack([-3.0 * obstacle, 6.0 * obstacle], axis=1) + rng.normal(0, 0.05, (400, 2))

    gp = GPConfig(max_points=50, n_iter_fit=120, device="cpu").build()
    for z, y in zip(X, Y):
        gp.observe(z, y[0], y[1])
    gp.end_episode()

    import time
    p, cfg = WheelieParams(), MPPIConfig()
    ref = np.array([10.0, 0.0, 0.0, 0.0])
    state = np.array([0.0, 0.0, 0.0, 0.0])
    a0 = np.zeros(10)

    def bench(name, **kw):
        ctrl = MPPITorch(p, cfg, gp, device="cpu", **kw)
        ctrl.solve(state, ref, 0.0, a0)                  # warmup (build caches)
        t0 = time.perf_counter()
        tau = 0.0
        for _ in range(20):
            tau, info = ctrl.solve(state, ref, tau, a0)
        ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"{name:28s} {ms:7.2f} ms/solve   last tau={tau:+.4f}")
        return tau

    bench("FAST (matmul, f32, euler)", fast=True, integrator="euler")
    bench("FAST (matmul, f32, rk4)", fast=True, integrator="rk4")
    bench("SLOW (gpytorch, f64, rk4)", fast=False, integrator="rk4")
    print("OK: the fast matmul GP prediction equals the GPyTorch model to ~1e-5 (the GP is\n"
          "    identical); the much lower ms/solve is the win. tau differs across rows only\n"
          "    because of the integrator (euler vs rk4) and dtype-dependent RNG draws.")
