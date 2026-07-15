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

    def __init__(self, p, cfg, gp, device=None, fast=True, integrator="euler", dtype=None, live_plot=False):

        self.p = p
        self.cfg = cfg
        self.gp = gp
        self.a_rls = None
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

        # Plot
        # --- live plan plot (optional; one figure, updated each solve) ---
        self.live_plot = bool(live_plot)     # draw the planned-trajectory figure each solve
        self.plot_obstacle_span = None       # (x0, x1) to shade an obstacle in x; None = no shade
        self._plot = None                    # figure + line handles, built lazily on first draw
        self._plot_off = False               # set True if matplotlib/display is unavailable


    def _plot_plan(self, s0, U, S, U_opt, dt, ref, n_best=100):
        if self._plot_off:
            return
        try:
            N = int(U.shape[1])
            goal_x = ref[0]
            direction = 1.0 if float(goal_x - s0[0]) >= 0.0 else -1.0

            with torch.no_grad():
                idx = torch.argsort(S)[:n_best]
                sb = s0.unsqueeze(0).repeat(idx.numel(), 1)
                active = direction * (goal_x - sb[:, 0]) > 0.0

                xs = [sb[:, 0].clone()]
                ths = [sb[:, 2].clone()]

                for n in range(N):
                    active_before = active.clone()

                    if torch.any(active):
                        rows = torch.where(active)[0]
                        sb_next = self._step(sb[rows], U[idx[rows], n], dt)
                        reached = direction * (goal_x - sb_next[:, 0]) <= 0.0
                        sb_next[reached, 0] = goal_x
                        sb[rows] = sb_next
                        active[rows[reached]] = False

                    nan_x = torch.full_like(sb[:, 0], torch.nan)
                    nan_theta = torch.full_like(sb[:, 2], torch.nan)
                    xs.append(torch.where(active_before, sb[:, 0], nan_x))
                    ths.append(torch.where(active_before, sb[:, 2], nan_theta))

                xs = torch.stack(xs, 1).cpu().numpy()
                ths = np.degrees(torch.stack(ths, 1).cpu().numpy())

                so = s0.unsqueeze(0)
                active_opt = bool(direction * float(goal_x - so[0, 0]) > 0.0)
                xo = [so[:, 0].clone()]
                tho = [so[:, 2].clone()]

                for n in range(N):
                    if not active_opt:
                        xo.append(torch.full_like(so[:, 0], torch.nan))
                        tho.append(torch.full_like(so[:, 2], torch.nan))
                        continue

                    so = self._step(so, U_opt[n:n + 1], dt)
                    reached = direction * (goal_x - so[0, 0]) <= 0.0

                    if reached:
                        so[0, 0] = goal_x
                        active_opt = False

                    xo.append(so[:, 0].clone())
                    tho.append(so[:, 2].clone())

                xo = torch.stack(xo, 1).cpu().numpy().ravel()
                tho = np.degrees(torch.stack(tho, 1).cpu().numpy().ravel())

            if self._plot is None:
                self._plot_setup(plt, ref, n_best)
            pl = self._plot

            for line, x, theta in zip(pl.samples, xs, ths):
                line.set_data(x, theta)
            pl.opt.set_data(xo, tho)

            xr, tr = float(s0[0]), math.degrees(float(s0[2]))
            if pl.trail_x and pl.trail_x[-1] - xr > 0.5:
                pl.trail_x.clear()
                pl.trail_t.clear()
            pl.trail_x.append(xr)
            pl.trail_t.append(tr)
            pl.trail.set_data(pl.trail_x, pl.trail_t)
            pl.car.set_data([xr], [tr])

            pl.fig.canvas.draw_idle()
            pl.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as exc:
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


    # ---- dynamics: RLS full model + GP residual ---
    def _deriv(self, s, tau):
        x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        a = self.a_rls

        x_dot = v
        v_dot = (
            a[0] * tau
            + a[1] * v
            + a[2] * torch.abs(v) * v
            + a[3] * tau * torch.cos(theta)
            + a[4]
        )

        theta_dot = omega
        omega_dot = (
            a[5] * torch.cos(theta)
            + a[6] * tau
            + a[7] * omega
            + a[8] * v
            + a[9]
        )

        if self.gp is not None and self.gp.ready:
            X = torch.stack([x, v, theta, omega, tau], dim=-1)
            residual = self.gp.predict_torch_fast(X)
            v_dot = v_dot + residual[:, 0]
            omega_dot = omega_dot + residual[:, 1]

        on_ground = (theta >= 0.0) & (omega_dot > 0.0)
        omega_dot = torch.where(on_ground, torch.zeros_like(omega_dot), omega_dot)

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


    # def _stage_cost(self, s, tau, tau_prev, ref):
    #         cfg, p = self.cfg, self.p
    #         x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

    #         e_x     = x     - ref[0]      # position error
    #         e_v     = v     - ref[1]      # speed error
    #         e_theta = theta - ref[2]      # pitch error
    #         e_omega = omega - ref[3]      # pitch-rate error
    #         d_tau   = tau   - tau_prev    # torque change since last step

    #         theta_deg = torch.rad2deg(theta)
    #         e_theta_deg = theta_deg - torch.rad2deg(ref[2])

    #         cost_goal       = cfg.q_x     * e_x     ** 2
    #         cost_speed      = cfg.q_v     * e_v     ** 2
    #         cost_pitch      = cfg.q_theta * e_theta ** 2
    #         cost_pitch_deg      = cfg.q_theta * e_theta_deg ** 2
    #         cost_pitch_rate = cfg.q_omega * e_omega ** 2
    #         cost_torque     = cfg.r_tau   * tau     ** 2
    #         cost_smooth     = cfg.r_dtau  * d_tau   ** 2

    #         return cost_goal


    def _stage_cost(self, s, tau, tau_prev, ref):
        cfg = self.cfg
        x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

        e_x = x - ref[0]
        e_v = v - ref[1]
        d_tau = tau - tau_prev

        theta_deg = torch.rad2deg(theta)

        too_much_wheelie = torch.relu((-theta_deg) - 75.0)
        nose_down = torch.relu(theta_deg)

        cost_goal = cfg.q_x * e_x**2
        cost_speed = cfg.q_v * e_v**2
        cost_pitch = cfg.q_theta * (too_much_wheelie**2 + nose_down**2)
        cost_pitch_rate = cfg.q_omega * omega**2
        cost_torque = cfg.r_tau * tau**2
        cost_smooth = cfg.r_dtau * d_tau**2

        return (
            cost_goal
            #+ cost_speed
            + cost_pitch
            + cost_pitch_rate
            #+ cost_torque
            + cost_smooth
        )

    
    # ---- the controller: state in -> action out ---------------------------
    #  Information-theoretic MPPI:
    #   1. sample K control sequences   V^m = U_nom + eps^m,  eps^m ~ N(0, sigma^2) i.i.d.
    #   2. roll each through F, summing the cost J(X^m, V^m)                 (1),(2)
    #   3. weight them  w^m = exp(-J^m/lambda) / sum_j exp(-J^j/lambda)      (3),(4)
    #   4. update  U* = U_nom + sum_m w^m eps^m,  apply the first action     (5)
    @torch.no_grad()
    def solve(self, state, ref, tau_prev, a_rls):
        p, cfg, dev = self.p, self.cfg, self.device
        K, N, dt = self.K, int(cfg.N), float(cfg.dt)
        lam, sigma = float(self.LAM), float(self.SIGMA)
        tau_min, tau_max = float(p.tau_min), float(p.tau_max)
        tau_prev = float(tau_prev)

        s0 = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)
        ref = torch.as_tensor(ref, dtype=self.dtype, device=dev).reshape(-1)
        self.a_rls = torch.as_tensor(a_rls, dtype=self.dtype, device=dev).reshape(10)

        U_nom = self._warm_start(N, tau_prev)
        eps = sigma * torch.randn(K, N, dtype=self.dtype, device=dev)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)
        eps = U - U_nom.unsqueeze(0)

        s = s0.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=dev)
        prev = torch.full((K,), tau_prev, dtype=self.dtype, device=dev)
        used = torch.zeros(K, N, dtype=self.dtype, device=dev)

        goal_x = ref[0]
        direction = 1.0 if float(goal_x - s0[0]) >= 0.0 else -1.0
        active = direction * (goal_x - s[:, 0]) > 0.0

        for n in range(N):
            if not torch.any(active):
                break

            rows = torch.where(active)[0]
            tau = U[rows, n]

            J[rows] += self._stage_cost(s[rows], tau, prev[rows], ref)
            s_next = self._step(s[rows], tau, dt)

            reached = direction * (goal_x - s_next[:, 0]) <= 0.0
            s_next[reached, 0] = goal_x

            s[rows] = s_next
            prev[rows] = tau
            used[rows, n] = 1.0
            active[rows[reached]] = False

        e = s - ref
        J += cfg.q_terminal_x * e[:, 0]**2
        J += cfg.q_terminal_theta * e[:, 2]**2
        J += cfg.q_terminal_omega * e[:, 3]**2
        J += -cfg.q_progress * s[:, 0]
        J = torch.nan_to_num(J, nan=1e12, posinf=1e12, neginf=1e12)

        rho = J.min()
        w = torch.exp(-(J - rho) / lam)
        w_sum = w.sum() + 1e-8

        du = (w.unsqueeze(1) * eps * used).sum(0) / w_sum
        U_opt = torch.clamp(U_nom + du, tau_min, tau_max)
        self.last_solution = U_opt

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, dt, ref)

        return float(U_opt[0].item()), {
            "success": True,
            "cost": float(rho.item()),
        }

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return torch.full((N,), tau_prev, dtype=self.dtype, device=self.device)
        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])