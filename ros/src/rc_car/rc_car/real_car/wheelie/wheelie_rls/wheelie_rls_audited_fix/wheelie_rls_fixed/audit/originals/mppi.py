#!/usr/bin/env python3

import math
import torch
from params_mppi import MPPIConfig
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from rls import rls_dynamics_torch
import multiprocessing as mp
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PLOT_OFSET = 7.0


def _plot_worker(plot_queue, obstacle_span, n_best):

    goal_x = 1.0
    fig, ax = plt.subplots()

    ax.set(
        xlabel="x [m]",
        ylabel="theta [deg]",
        xlim=(-PLOT_OFSET, goal_x + PLOT_OFSET),
        ylim=(-120.0, 120.0),
        title="MPPI planned trajectories (x vs theta)",
    )

    if obstacle_span is not None:
        ax.axvspan(*obstacle_span, color="0.85", zorder=0)

    ax.grid(True, alpha=0.3)

    samples = [
        ax.plot([], [], color="0.7", lw=0.6, alpha=0.5)[0]
        for _ in range(n_best)
    ]

    opt, = ax.plot([], [], "C3-o", lw=2.0, ms=3, label="optimal plan")
    trail, = ax.plot([], [], "C0-", lw=1.3, alpha=0.85, label="realized")
    car, = ax.plot([], [], "o", ms=12, mfc="blue", mec="k", mew=1.0, zorder=5, label="car")

    ax.legend(loc="upper left")

    trail_x = []
    trail_t = []
    closed = False

    def update(_):
        nonlocal closed
        latest = None
        while True:
            try:
                packet = plot_queue.get_nowait()
                if packet is None:
                    closed = True
                    break

                latest = packet

            except queue.Empty:
                break

        if closed:
            plt.close(fig)
            return tuple(samples) + (opt, trail, car)

        if latest is None:
            return tuple(samples) + (opt, trail, car)

        xs = latest["xs"]
        ths = latest["ths"]
        xo = latest["xo"]
        tho = latest["tho"]
        xr = latest["xr"]
        tr = latest["tr"]

        for line, x, theta in zip(samples, xs, ths):
            line.set_data(x, theta)

        opt.set_data(xo, tho)

        if trail_x and trail_x[-1] - xr > 0.5:
            trail_x.clear()
            trail_t.clear()

        trail_x.append(xr)
        trail_t.append(tr)

        trail.set_data(trail_x, trail_t)
        car.set_data([xr], [tr])

        return tuple(samples) + (opt, trail, car)

    animation = FuncAnimation(
        fig,
        update,
        interval=100,
        cache_frame_data=False,
        blit=False,
    )

    plt.show()




class MPPITorch:
    # same sampler settings as the numba mppi.py so behaviour is comparable
    # K = MPPIConfig.K            # number of sampled rollouts
    # SIGMA = MPPIConfig.SIGMA        # exploration std on u
    # LAM = MPPIConfig.LAM           # temperature (softmin sharpness)

    def __init__(self, p, cfg, rls_parameters, device=None, fast=True, integrator="euler", dtype=None, live_plot=False):

        self.p = p
        self.cfg = cfg                                   # the GP OBJECT (called directly)
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (torch.float32 if fast else torch.double)

        self.device = torch.device("cuda")

        self.rls_parameters = torch.as_tensor(rls_parameters, dtype=self.dtype, device=self.device).clone()
        self.last_solution = None


        self.live_plot = bool(live_plot)
        self.plot_obstacle_span = None

        self._plot_process = None
        self._plot_queue = None
        self._plot_n_best = 50

        self.K = int(cfg.K)
        self.SIGMA = float(cfg.SIGMA)
        self.LAM = float(cfg.LAM)        

        if self.live_plot:
            ctx = mp.get_context("spawn")
            self._plot_queue = ctx.Queue(maxsize=1)


    def _start_plot_process(self):
        if not self.live_plot:
            return

        if self._plot_process is not None and self._plot_process.is_alive():
            return

        ctx = mp.get_context("spawn")

        self._plot_process = ctx.Process(
            target=_plot_worker,
            args=(
                self._plot_queue,
                self.plot_obstacle_span,
                self._plot_n_best,
            ),
            daemon=True,
        )

        self._plot_process.start()


    def _plot_plan(self, s0, U, S, U_opt, dt, ref, n_best=100):
        if not self.live_plot:
            return

        self._plot_n_best = n_best
        self._start_plot_process()

        N = int(U.shape[1])

        with torch.no_grad():

            idx = torch.argsort(S)[:n_best]
            sb = s0.unsqueeze(0).repeat(idx.numel(), 1)
            xs = [sb[:, 0].clone()]
            ths = [sb[:, 2].clone()]

            for n in range(N):
                sb = self._step(sb, U[idx, n], dt)
                xs.append(sb[:, 0].clone())
                ths.append(sb[:, 2].clone())

            xs = torch.stack(xs, 1).cpu().numpy()
            ths = np.degrees(torch.stack(ths, 1).cpu().numpy())
            so = s0.unsqueeze(0)
            xo = [so[:, 0].clone()]
            tho = [so[:, 2].clone()]

            for n in range(N):
                so = self._step(so, U_opt[n:n + 1], dt)
                xo.append(so[:, 0].clone())
                tho.append(so[:, 2].clone())

            xo = torch.stack(xo, 1).cpu().numpy().ravel()
            tho = np.degrees(torch.stack(tho, 1).cpu().numpy().ravel())

        packet = {
            "xs": xs,
            "ths": ths,
            "xo": xo,
            "tho": tho,
            "xr": float(s0[0]),
            "tr": math.degrees(float(s0[2])),
        }

        try:
            self._plot_queue.put_nowait(packet)

        except queue.Full:
            try:
                self._plot_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._plot_queue.put_nowait(packet)
            except queue.Full:
                pass


    def close_plot(self):
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(None)

            except queue.Full:
                try:
                    self._plot_queue.get_nowait()
                except queue.Empty:
                    pass

                try:
                    self._plot_queue.put_nowait(None)
                except queue.Full:
                    pass

        if self._plot_process is not None:
            self._plot_process.join(timeout=0.5)

            if self._plot_process.is_alive():
                self._plot_process.terminate()

        self._plot_process = None


    def set_rls_parameters(self, parameters):
        """Copy the latest CPU-side RLS estimate to the rollout device."""
        self.rls_parameters.copy_(
            torch.as_tensor(parameters, dtype=self.dtype, device=self.device)
        )

    # ---- learned RLS dynamics ---
    def _deriv(self, s, u):
        return rls_dynamics_torch(s, u, self.rls_parameters)



    def _step(self, state: torch.Tensor, u: torch.Tensor, dt) -> torch.Tensor:
        """
        Advances the batch state forward by self.cfg.dt using RK4 integration
        with safety clamping on pitch, velocities, and numerical finiteness.

        state shape: (K, 4) -> [x, x_dot, pitch, pitch_dot]
        u shape:     (K, 1) or (K,)
        """
        dt = self.cfg.dt

        # RK4 Integration steps
        k1 = self._deriv(state, u)
        k2 = self._deriv(state + 0.5 * dt * k1, u)
        k3 = self._deriv(state + 0.5 * dt * k2, u)
        k4 = self._deriv(state + dt * k3, u)

        next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # ------------------------------------------------------------------
        # STATE SAFETY CLAMPING (Prevents Rollout Cost Explosions)
        # ------------------------------------------------------------------
        # 1. Clamp velocity v [m/s]
        next_state[:, 1] = torch.clamp(next_state[:, 1], min=-15.0, max=15.0)

        # 2. Clamp pitch angle theta [rad] (e.g. +/- 2*pi rad)
        next_state[:, 2] = torch.clamp(
            next_state[:, 2], 
            min=-2.0 * math.pi, 
            max=2.0 * math.pi
        )

        # 3. Clamp pitch rate omega [rad/s]
        next_state[:, 3] = torch.clamp(next_state[:, 3], min=-50.0, max=50.0)

        # 4. Replace non-finite values (NaN / Inf) with bounded values
        next_state = torch.nan_to_num(
            next_state, 
            nan=0.0, 
            posinf=10.0, 
            neginf=-10.0
        )

        return next_state


    def _stage_cost(self, s, u, u_prev, ref):
            cfg, p = self.cfg, self.p
            x, v, theta, omega = s[:, 0], s[:, 1], s[:, 2], s[:, 3]

            e_x     = x     - ref[0]      # position error
            e_v     = v     - ref[1]      # speed error
            e_theta = theta - ref[2]      # pitch error
            e_omega = omega - ref[3]      # pitch-rate error
            d_u   = u   - u_prev    # torque change since last step

            # theta_deg = torch.rad2deg(theta)
            # e_theta_deg = theta_deg - torch.rad2deg(ref[2])


            cost_goal       = cfg.q_x     * e_x     ** 2
            cost_speed      = cfg.q_v     * e_v     ** 2
            cost_pitch      = cfg.q_theta * e_theta ** 2
            cost_pitch_rate = cfg.q_omega * e_omega ** 2
            cost_torque     = cfg.r_u   * u     ** 2
            cost_smooth     = cfg.r_du  * d_u   ** 2
            #cost_pitch_deg = cfg.q_theta * e_theta_deg ** 2

            return (#cost_goal 
                    #+ cost_pitch_deg 
                    + cost_pitch
                    + cost_smooth 
                    #+ cost_speed 
                    + cost_pitch_rate)

    
    # ---- the controller: state in -> action out ---------------------------
    @torch.no_grad()
    def solve(self, state, ref, u_prev):
        p, cfg, dev = self.p, self.cfg, self.device
        # k=Samples, N=Horizon, dt=Sample time
        K, N, dt = self.K, int(cfg.N), float(cfg.dt)
        # lam=temperature, sigma=exploration std
        lam, sigma = float(self.LAM), float(self.SIGMA)
        u_min, u_max = float(p.u_min), float(p.u_max)
        u_prev = float(u_prev)

        s0  = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)   # (4,)
        ref = torch.as_tensor(ref,   dtype=self.dtype, device=dev).reshape(-1)   # (4,)

        if not bool(torch.isfinite(s0).all()) or not bool(torch.isfinite(ref).all()) or not math.isfinite(u_prev):
            self.last_solution = None
            return 0.0, {"success": False, "cost": float("inf")}

        U_nom = self._warm_start(N, u_prev)
        eps = sigma * torch.randn(K, N, dtype=self.dtype, device=dev)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, u_min, u_max)
        eps = U - U_nom.unsqueeze(0)

        s = s0.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=dev)
        prev = torch.full((K,), u_prev, dtype=self.dtype, device=dev)
        for n in range(N):
            u = U[:, n]
            J = J + self._stage_cost(s, u, prev, ref)
            s = self._step(s, u, dt)
            prev = u

        e = s - ref
        goal_terminal_cost = cfg.q_x * e[:, 0] ** 2
        speed_terminal_cost = cfg.q_terminal_v * e[:, 1] ** 2
        pitch_terminal_cost = cfg.q_terminal_theta * e[:, 2] ** 2
        pitch_rate_terminal_cost = cfg.q_terminal_omega * e[:, 3] ** 2
        


        J = J + (#goal_terminal_cost
                 #+ speed_terminal_cost
                 + pitch_terminal_cost
                 + pitch_rate_terminal_cost
        )


        # J = torch.nan_to_num(J, nan=1e12, posinf=1e12, neginf=1e12)
        valid = torch.isfinite(J)

        if not bool(valid.any()):
            self.last_solution = None
            return 0.0, {"success": False, "cost": float("inf")}

        J = J.masked_fill(~valid, float("inf"))


        #rho = J.min()

        best_idx = torch.argmin(J)
        rho = J[best_idx]

        w = torch.exp(-(J - rho) / lam)
        du = (w.unsqueeze(1) * eps).sum(0) / (w.sum() + 1e-8)
        U_opt = torch.clamp(U_nom + du, u_min, u_max)
        self.last_solution = U_opt                                              # warm start next call

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, dt, ref)

        U_opt = float(U_opt[0].item())
        # info = {"success": True, "cost": float(rho.item())}
        info = {
            "success": True,
            "cost": float(rho.item()),
            "pitch_cost_now": float((cfg.q_theta * (s0[2] - ref[2])**2).item()),
            "best_theta_N_deg": math.degrees(float(s[best_idx, 2].item())),
            "best_omega_N": float(s[best_idx, 3].item()),
            "best_v_N": float(s[best_idx, 1].item()),
        }

        #return float(U_opt[0].item()), {"success": True, "cost": float(rho.item())}
        return U_opt, info

    def _warm_start(self, N, u_prev):
        if self.last_solution is None:
            return torch.full((N,), u_prev, dtype=self.dtype, device=self.device)
        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])
