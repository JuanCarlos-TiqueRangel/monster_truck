#!/usr/bin/env python3

import math
import torch
from params_mppi import MPPIConfig
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

class MPPITorch:
    # same sampler settings as the numba mppi.py so behaviour is comparable
    K = MPPIConfig.K            # number of sampled rollouts
    SIGMA = MPPIConfig.SIGMA        # exploration std on tau
    LAM = MPPIConfig.LAM           # temperature (softmin sharpness)
    SEED = MPPIConfig.SEED

    def __init__(self, p, cfg, gp, device=None, fast=True, integrator="euler", dtype=None, live_plot=False):

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


    def _plot_plan(self, s0, U, S, U_opt, dt, ref, n_best=50):
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
                    sb = self._step(sb, U[idx, n], dt)
                    xs.append(sb[:, 0].clone()); ths.append(sb[:, 2].clone())
                
                xs = torch.stack(xs, 1).cpu().numpy()
                ths = np.degrees(torch.stack(ths, 1).cpu().numpy())

                # the optimal plan
                so = s0.unsqueeze(0)
                xo, tho = [so[:, 0].clone()], [so[:, 2].clone()]

                for n in range(N):
                    so = self._step(so, U_opt[n:n + 1], dt)
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
    # def _deriv(self, s, tau):
    #     x, v, th, om = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

    #     m = 5.1
    #     r = 0.081
    #     g = 9.81
    #     l = 0.2
    #     L_car = 0.53
    #     H_body = 0.30

    #     x_dot = v
    #     v_dot = tau / (m * r)
    #     th_dot = om

    #     I_body = (1.0 / 12.0) * m * (L_car**2 + H_body**2)
    #     I_eff = I_body + m * l**2

    #     print("[DEBUG]: ", I_eff)

    #     om_dot = (-tau + m * g * l * torch.sin(th)) / I_eff

    #     return torch.stack([x_dot, v_dot, th_dot, om_dot], dim=-1)

    def _deriv(self, s, tau):

        M, R, G = 5.1, 0.081, 9.81
        L_R = 0.20                                   # CoM -> rear axle, horizontal [m]
        H   = 0.16                                   # CoM height when flat [m]
        RHO   = math.hypot(L_R, H - R)               # 0.215 m
        ALPHA = math.atan2(H - R, L_R)               # 0.377 rad
        I_C   = (1.0 / 12.0) * M * (0.53**2 + 0.30**2)
        I_A   = I_C + M * RHO**2                     # 0.394 kg m^2, about the rear contact
        C_OMEGA = 0.2                                # small pitch damping, tune to MuJoCo

        x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        beta = ALPHA - theta
        lever_drive   = R + RHO * torch.sin(beta)        # CoM height, grows with rear-up
        lever_gravity = RHO * torch.cos(beta)            # horizontal offset, shrinks
        omega_dot = (-(tau / R) * lever_drive + M * G * lever_gravity) / I_A - C_OMEGA * omega
        on_ground = (theta >= 0.0) & (omega_dot > 0.0)   # floor absorbs nose-down push
        omega_dot = torch.where(on_ground, torch.zeros_like(omega_dot), omega_dot)
        x_dot = v
        v_dot = tau / (M * R)
        theta_dot = omega
        return torch.stack([x_dot, v_dot, theta_dot, omega_dot], dim=-1)



    def _step(self, s, tau, dt):
        if self.integrator == "rk4":                     
            k1 = self._deriv(s, tau)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:                                            # euler: 1 GP eval/step (4x fewer)
            s = s + dt * self._deriv(s, tau)

        landing = (s[:, 2] >= 0.0) & (s[:, 3] > 0.0)
        s = torch.stack([s[:, 0],
                        s[:, 1].clamp(-10.0, 10.0),
                        s[:, 2].clamp(max=0.0),
                        torch.where(landing, torch.zeros_like(s[:, 3]), s[:, 3].clamp(-30.0, 30.0))],
                        dim=-1)

        return s


    def _stage_cost(self, s, tau, tau_prev, ref):
            cfg, p = self.cfg, self.p
            x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

            e_x     = x     - ref[0]      # position error
            e_v     = v     - ref[1]      # speed error
            e_theta = theta - ref[2]      # pitch error
            e_omega = omega - ref[3]      # pitch-rate error
            d_tau   = tau   - tau_prev    # torque change since last step

            theta_deg = torch.rad2deg(theta)
            e_theta_deg = theta_deg - ref[2]

            cost_goal       = cfg.q_x     * e_x     ** 2
            cost_speed      = cfg.q_v     * e_v     ** 2
            cost_pitch      = cfg.q_theta * e_theta ** 2
            #cost_pitch      = cfg.q_theta * e_theta_deg ** 2
            cost_pitch_rate = cfg.q_omega * e_omega ** 2
            cost_torque     = cfg.r_tau   * tau     ** 2
            cost_smooth     = cfg.r_dtau  * d_tau   ** 2

            # cg = cost_goal.mean().item()
            # cp = cost_pitch.mean().item()
            # print(f"cost_goal={cg:.2f}  cost_pitch={cp:.2f}  ratio={cp/max(cg,1e-9):.3f}")

            return cost_goal + cost_pitch

    
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

        # receding waypoint: track a point ~one horizon ahead, capped at the goal.
        # bounds cost magnitude -> lambda/sigma stay valid at any goal distance.
        # ref= [xpos, velocity, pitch, pitch_dot]
        LOOKAHEAD = 6.0                                   # ~ v_max * N * dt [m]
        ref = ref.clone()
        ref[0] = torch.minimum(ref[0], s0[0] + LOOKAHEAD)

        # 1. sample K control sequences around the warm-started nominal
        U_nom = self._warm_start(N, tau_prev)                                    # (N,)  u_t
        eps = sigma * torch.randn(K, N, generator=self.gen,
                                  dtype=self.dtype, device=dev)                # (K,N) ~ N(0, sigma^2)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)             # (K,N) V^m
        eps = U - U_nom.unsqueeze(0)                                          # effective noise after clamp

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
        J = J + cfg.q_x * e[:, 0]**2 + cfg.q_theta * e[:, 2]**2
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

        # ess = 1.0 / (w ** 2).sum()
        # print(f"ESS = {ess.item():.0f} / {K}")

        # good = J[J < J.median()]                      # the samples that actually get weight
        # spread = (good.max() - good.min()).item()
        # ess = 1.0 / (w ** 2).sum()
        # print(f"J_min={rho.item():.0f}  spread={spread:.0f}  J_max={J.max().item():.0f}  ESS={ess.item():.0f}/{K}")

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, dt, ref)

        U_opt = float(U_opt[0].item())
        info = {"success": True, "cost": float(rho.item())}

        #return float(U_opt[0].item()), {"success": True, "cost": float(rho.item())}
        return U_opt, info

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return torch.full((N,), tau_prev, dtype=self.dtype, device=self.device)
        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])
