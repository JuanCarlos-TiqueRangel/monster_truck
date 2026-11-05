#!/usr/bin/env python3

import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from rls_history_mppi import RLSHistoryDynamics
import multiprocessing as mp
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PLOT_OFSET = 5.0


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

    def __init__(self, p, cfg, model_path, device=None, fast=True, integrator="euler", dtype=None, live_plot=False):

        self.p = p
        self.cfg = cfg
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (torch.float32 if fast else torch.double)

        self.device = torch.device("cuda")
        self.model = RLSHistoryDynamics(model_path, device=self.device, dtype=self.dtype)
        if self.integrator != "euler":
            raise ValueError("This history adapter uses one Euler step per history sample; set integrator='euler'.")
        if not math.isclose(float(cfg.dt), self.model.nominal_dt, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"cfg.dt={cfg.dt} must match the model history period {self.model.nominal_dt}.")
        if int(cfg.N) < 1 or int(cfg.K) < 1 or float(cfg.LAM) <= 0.0 or float(cfg.SIGMA) < 0.0:
            raise ValueError("Require positive N, K and LAM, and nonnegative SIGMA.")
        if not float(p.u_min) <= 0.0 <= float(p.u_max):
            raise ValueError("Input bounds must include the zero-input fallback.")
        self.last_solution = None


        self.live_plot = bool(live_plot)
        self.plot_obstacle_span = None

        self._plot_process = None
        self._plot_queue = None
        self._plot_n_best = 200

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


    def _plot_plan(self, s0, U, S, U_opt, dt, ref, past_z, past_ages, n_best=100):
        if not self.live_plot:
            return

        self._plot_n_best = n_best
        self._start_plot_process()

        N = int(U.shape[1])

        with torch.no_grad():

            idx = torch.argsort(S)[:n_best]
            sb = s0.unsqueeze(0).repeat(idx.numel(), 1)
            past = past_z.unsqueeze(0).repeat(idx.numel(), 1, 1)
            ages = past_ages.unsqueeze(0).repeat(idx.numel(), 1)
            xs = [sb[:, 0].clone()]
            ths = [sb[:, 2].clone()]

            for n in range(N):
                sb, past, ages = self._step(sb, U[idx, n], dt, past, ages)
                xs.append(sb[:, 0].clone())
                ths.append(sb[:, 2].clone())

            xs = torch.stack(xs, 1).cpu().numpy()
            ths = np.degrees(torch.stack(ths, 1).cpu().numpy())
            so = s0.unsqueeze(0)
            past = past_z.unsqueeze(0).clone()
            ages = past_ages.unsqueeze(0).clone()
            xo = [so[:, 0].clone()]
            tho = [so[:, 2].clone()]

            for n in range(N):
                so, past, ages = self._step(so, U_opt[n:n + 1], dt, past, ages)
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


    def _step(self, s, u, dt, past, ages):
        # Acceleration inference is frozen. Each sampled trajectory shifts its own history.
        s, past, ages = self.model.step(s, u, past, ages, dt=dt, project_ground=False)

        # Retain your existing rollout projections; these are NOT learned dynamics.
        landing = (s[:, 2] >= 0.0) & (s[:, 3] > 0.0)
        s = torch.stack([s[:, 0], s[:, 1].clamp(-10.0, 10.0), s[:, 2].clamp(max=0.0),
                         torch.where(landing, torch.zeros_like(s[:, 3]), s[:, 3].clamp(-30.0, 30.0))], dim=-1)
        return s, past, ages


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
    #  Information-theoretic MPPI:
    #   1. sample K control sequences   V^m = U_nom + eps^m,  eps^m ~ N(0, sigma^2) i.i.d.
    #   2. roll each through F, summing the cost J(X^m, V^m)                 (1),(2)
    #   3. weight them  w^m = exp(-J^m/lambda) / sum_j exp(-J^j/lambda)      (3),(4)
    #   4. update  U* = U_nom + sum_m w^m eps^m,  apply the first action     (5)
    @torch.no_grad()
    def solve(self, state, ref, u_prev, past_z, past_ages):
        """past_z=(4,4), newest first, excluding now; rows=[u_applied,v,theta,omega].

        past_ages=(4,) gives elapsed seconds from the current state to each row.
        Supply fresh measured state/history on EVERY real control iteration.
        The returned first command is not added to the real history here.
        """
        p, cfg, dev = self.p, self.cfg, self.device
        # k=Samples, N=Horizon, dt=Sample time
        K, N, dt = self.K, int(cfg.N), float(cfg.dt)
        # lam=temperature, sigma=exploration std
        lam, sigma = float(self.LAM), float(self.SIGMA)
        u_min, u_max = float(p.u_min), float(p.u_max)
        u_prev = float(u_prev)

        s0  = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)   # (4,)
        ref = torch.as_tensor(ref,   dtype=self.dtype, device=dev).reshape(-1)   # (4,)

        past_z = torch.as_tensor(past_z, dtype=self.dtype, device=dev)
        past_ages = torch.as_tensor(past_ages, dtype=self.dtype, device=dev)
        if s0.shape != (4,) or ref.shape != (4,) or past_z.shape != (4, 4) or past_ages.shape != (4,):
            raise ValueError("Expected state/ref (4,), past_z (4,4), and past_ages (4,).")
        finite = torch.isfinite(torch.cat((s0, ref, past_z.flatten(), past_ages))).all()
        ordered = (past_ages[0] > 0.0) & torch.all(past_ages[1:] > past_ages[:-1])
        if not bool(finite & ordered) or not math.isfinite(u_prev):
            self.last_solution = None
            return 0.0, {"success": False, "cost": float("inf")}

        U_nom = self._warm_start(N, u_prev)
        eps = sigma * torch.randn(K, N, dtype=self.dtype, device=dev)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, u_min, u_max)

        if K < 3:
            raise ValueError("This diagnostic requires at least three sampled trajectories.")

        U[0] = torch.clamp(U_nom, u_min, u_max)
        U[1].fill_(u_min)
        U[2].fill_(u_max)
        eps = U - U_nom.unsqueeze(0)
        

        s = s0.unsqueeze(0).repeat(K, 1)
        past = past_z.unsqueeze(0).repeat(K, 1, 1)
        ages = past_ages.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=dev)
        prev = torch.full((K,), u_prev, dtype=self.dtype, device=dev)
        for n in range(N):
            u = U[:, n]
            J = J + self._stage_cost(s, u, prev, ref)
            s, past, ages = self._step(s, u, dt, past, ages)
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
            self._plot_plan(s0, U, J, U_opt, dt, ref, past_z, past_ages)

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
