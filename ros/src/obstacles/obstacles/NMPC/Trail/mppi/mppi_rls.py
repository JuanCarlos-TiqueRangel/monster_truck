#!/usr/bin/env python3

import math
import torch
from params_mppi import MPPIConfig
import matplotlib.pyplot as plt
import numpy as np

class MPPITorch:
    # same sampler settings as the numba mppi.py so behaviour is comparable
    K = MPPIConfig.K            # number of sampled rollouts
    SIGMA = MPPIConfig.SIGMA        # exploration std on tau
    LAM = MPPIConfig.LAM           # temperature (softmin sharpness)
    SEED = MPPIConfig.SEED

    def __init__(self, p, cfg, gp, device=None, fast=True, integrator="euler", dtype=None, live_plot=False):
        """fast=True  -> GP residual via the cached kernel MATMUL (gp.predict_torch_fast),
                         float32, no gpytorch forward in the loop -> MUCH faster.
           fast=False -> calls gp.predict_torch (the GPyTorch model directly), double; the
                         clean-but-slow reference path.
           integrator -> "euler" (1 GP eval/step, default) or "rk4" (4 GP evals/step)."""

        self.p = p
        self.cfg = cfg
        self.gp = gp                                     # the GP OBJECT (called directly)
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (torch.float32 if fast else torch.double)
        self.device = torch.device("cuda")
        # risk-averse weight on the GP's epistemic uncertainty (0 = off). >0 penalizes
        # planning into regions the GP has NOT learned (needs gp.predict_uncertainty_torch).
        self.q_gp_var = float(getattr(cfg, "q_gp_var", 0.0))
        if self.q_gp_var > 0.0 and not hasattr(gp, "predict_uncertainty_torch"):
            raise TypeError("q_gp_var > 0 needs a GP with predict_uncertainty_torch().")
        self.last_solution = None                        # warm-start plan, (N,) tensor
        self.plot_hook = None                            # set to a fn(self,s0,U,S,U_opt,a,dt) to live-plot; None = zero overhead
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(self.SEED)

        # Plot
        # --- live plan plot (optional; one figure, updated each solve) ---
        self.live_plot = bool(live_plot)     # draw the planned-trajectory figure each solve
        self.plot_obstacle_span = None       # (x0, x1) to shade an obstacle in x; None = no shade
        self._plot = None                    # figure + line handles, built lazily on first draw
        self._plot_off = False               # set True if matplotlib/display is unavailable


    def _plot_plan(self, s0, U, S, U_opt, a, dt, ref, n_best=100):
        if self._plot_off:
            return
        try:
            N = int(U.shape[1])

            with torch.no_grad():
                # n_best cheapest sampled rollouts (one batched re-roll)
                idx = torch.argsort(S)[:n_best]
                sb = s0.unsqueeze(0).repeat(idx.numel(), 1)
                xs, ths = [sb[:, 0].clone()], [sb[:, 2].clone()]
                
                for n in range(N):
                    sb = self._step(sb, U[idx, n], a, dt)
                    xs.append(sb[:, 0].clone()); ths.append(sb[:, 2].clone())
                
                xs = torch.stack(xs, 1).cpu().numpy()
                ths = np.degrees(torch.stack(ths, 1).cpu().numpy())

                # the optimal plan
                so = s0.unsqueeze(0)
                xo, tho = [so[:, 0].clone()], [so[:, 2].clone()]

                for n in range(N):
                    so = self._step(so, U_opt[n:n + 1], a, dt)
                    xo.append(so[:, 0].clone()); tho.append(so[:, 2].clone())
                
                xo = torch.stack(xo, 1).cpu().numpy().ravel()
                tho = np.degrees(torch.stack(tho, 1).cpu().numpy().ravel())

            if self._plot is None:                 # first call: build the figure once
                self._plot_setup(plt, ref, n_best)
            pl = self._plot

            for line, x, t in zip(pl.samples, xs, ths):
                line.set_data(x, t)
            pl.opt.set_data(xo, tho)

            # realized trail = the true (x, theta) the truck has passed through. s0 each
            # solve IS the current real state; a big backward jump in x = episode reset.
            xr, tr = float(s0[0]), math.degrees(float(s0[2]))
            if pl.trail_x and pl.trail_x[-1] - xr > 0.5:
                pl.trail_x.clear(); pl.trail_t.clear()
            pl.trail_x.append(xr); pl.trail_t.append(tr)
            pl.trail.set_data(pl.trail_x, pl.trail_t)
            pl.car.set_data([xr], [tr])            # blue circle = current (x, theta)

            pl.fig.canvas.draw_idle(); pl.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:                   # no display / matplotlib -> disable, don't crash
            print(f"[plot_plan] disabled ({exc})")
            self._plot_off = True

    def _plot_setup(self, plt, ref, n_best):
        """Build the persistent figure and empty line artists (called once, on first draw)."""
        from types import SimpleNamespace
        goal_x = float(ref[0])
        plt.ion()
        fig, ax = plt.subplots()
        ax.set(xlabel="x [m]", ylabel="theta [deg]",
               xlim=(-2.0, goal_x + 2.0), ylim=(-120, 120),
               title="MPPI planned trajectories (x vs theta)")
        if self.plot_obstacle_span is not None:
            ax.axvspan(*self.plot_obstacle_span, color="0.85", zorder=0)
        ax.grid(True, alpha=0.3)
        samples = [ax.plot([], [], color="0.7", lw=0.6, alpha=0.5)[0] for _ in range(n_best)]
        opt,   = ax.plot([], [], "C3-o", lw=2.0, ms=3, label="optimal plan")
        trail, = ax.plot([], [], "C0-",  lw=1.3, alpha=0.85, label="realized")
        car,   = ax.plot([], [], "o", ms=12, mfc="blue", mec="k", mew=1.0, zorder=5, label="car")
        ax.legend(loc="upper left")
        self._plot = SimpleNamespace(fig=fig, ax=ax, samples=samples, opt=opt,
                                     trail=trail, car=car, trail_x=[], trail_t=[])

    # ---- dynamics: nominal dynamics  ---
    def _deriv(self, s, tau):

        x, v, th, om = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

        # DYNAMIC MODEL
        x_dot = v
        v_dot = tau/(5.1 * 0.081) 
        th_dot = om
        I_eff = ((1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2))
        om_dot = ((-tau + 5.1*9.81*0.2*torch.cos(th))/(I_eff))

        return torch.stack([x_dot, v_dot, th_dot, om_dot], dim=-1)


    def _step(self, s, tau, dt):
        if self.integrator == "rk4":                     
            k1 = self._deriv(s, tau)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:                                            # euler: 1 GP eval/step (4x fewer)
            s = s + dt * self._deriv(s, tau)
        # same clamps as the numba rollout (keep diverging samples finite)
        s = torch.stack([s[:, 0],
                         s[:, 1].clamp(-10.0, 10.0),
                         s[:, 2],
                         s[:, 3].clamp(-30.0, 30.0)], dim=-1)
        return s


    def _stage_cost(self, s, tau, tau_prev, ref):
            cfg, p = self.cfg, self.p
            x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

            e_x     = x     - ref[0]      # position error
            e_v     = v     - ref[1]      # speed error
            e_theta = theta - ref[2]      # pitch error
            e_omega = omega - ref[3]      # pitch-rate error
            d_tau   = tau   - tau_prev    # torque change since last step

            cost_goal       = cfg.q_x     * e_x     ** 2
            cost_speed      = cfg.q_v     * e_v     ** 2
            cost_pitch      = cfg.q_theta * e_theta ** 2
            cost_pitch_rate = cfg.q_omega * e_omega ** 2
            cost_torque     = cfg.r_tau   * tau     ** 2
            cost_smooth     = cfg.r_dtau  * d_tau   ** 2

            return cost_goal
            #return (cost_goal + cost_speed + cost_pitch + cost_pitch_rate + cost_torque)

    
    # ---- the controller: state in -> action out ---------------------------
    #  Information-theoretic MPPI:
    #   1. sample K control sequences   V^m = U_nom + eps^m,  eps^m ~ N(0, sigma^2) i.i.d.
    #   2. roll each through F, summing the cost J(X^m, V^m)                 (1),(2)
    #   3. weight them  w^m = exp(-J^m/lambda) / sum_j exp(-J^j/lambda)      (3),(4)
    #   4. update  U* = U_nom + sum_m w^m eps^m,  apply the first action     (5)
    @torch.no_grad()
    def solve(self, state, ref, tau_prev, a_rls):
        p, cfg, dev = self.p, self.cfg, self.device
        # k=Samples, N=Horizon, dt=Sample time
        K, N, dt = self.K, int(cfg.N), float(cfg.dt)
        # lam=temperature, sigma=exploration std
        lam, sigma = float(self.LAM), float(self.SIGMA)
        tau_min, tau_max = float(p.tau_min), float(p.tau_max)
        tau_prev = float(tau_prev)

        s0  = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)   # (4,)
        ref = torch.as_tensor(ref,   dtype=self.dtype, device=dev).reshape(-1)   # (4,)

        # 1. sample K control sequences around the warm-started nominal
        U_nom = self._warm_start(N, tau_prev)                                    # (N,)  u_t
        eps = sigma * torch.randn(K, N, generator=self.gen,
                                  dtype=self.dtype, device=dev)                  # (K,N) ~ N(0, sigma^2)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)             # (K,N) V^m
        eps = U - U_nom.unsqueeze(0)                                            # effective noise after clamp

        # 2. roll every sequence through the dynamics and accumulate its cost J
        s = s0.unsqueeze(0).repeat(K, 1)                                         # (K,4) all start at x_0
        J = torch.zeros(K, dtype=self.dtype, device=dev)
        prev = torch.full((K,), tau_prev, dtype=self.dtype, device=dev)

        # Running Cost
        for n in range(N):
            tau = U[:, n]                                                        # u_n for every sample
            J = J + self._stage_cost(s, tau, prev, ref)                       # running cost l(x_n, u_n)
            s = self._step(s, tau, dt)                                        # x_{n+1} = F(x_n, u_n)
            prev = tau
        
        # Terminal Cost
        e = s - ref                                                             # terminal cost phi(x_N)
        J = J + cfg.q_x * e[:, 0]**2
        # J = J + (cfg.q_x * e[:, 0]**2 + cfg.q_v * e[:, 1]**2
        #          + cfg.q_theta * e[:, 2]**2 + cfg.q_omega * e[:, 3]**2)
        J = torch.nan_to_num(J, nan=1e12, posinf=1e12, neginf=1e12)

        # 3. costs -> weights  (subtract min cost rho first; it cancels but avoids overflow)
        rho = J.min()
        w = torch.exp(-(J - rho) / lam)                                          # exp(-(J - rho)/lambda)
        w = w / w.sum()                                                          # normalize (sum >= 1)

        # 4. weighted update; U_opt is the new plan, U_opt[0] is the action applied
        U_opt = torch.clamp(U_nom + (w.unsqueeze(1) * eps).sum(0), tau_min, tau_max)
        self.last_solution = U_opt                                              # warm start next call

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, a, dt, ref)
        return float(U_opt[0].item()), {"success": True, "cost": float(rho.item())}

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return torch.full((N,), tau_prev, dtype=self.dtype, device=self.device)
        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])
