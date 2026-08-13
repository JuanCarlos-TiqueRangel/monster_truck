#!/usr/bin/env python3

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from params_mppi import MPPIConfig


class MPPITorch:
    K = MPPIConfig.K
    SIGMA = MPPIConfig.SIGMA
    LAM = MPPIConfig.LAM

    def __init__(
        self,
        p,
        cfg,
        gp,
        device=None,
        fast=True,
        integrator="euler",
        dtype=None,
        live_plot=False,
    ):
        self.p = p
        self.cfg = cfg
        self.gp = gp
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (
            torch.float32 if fast else torch.float64
        )
        self.device = torch.device("cuda" if device is None else device)

        self.q_gp_var = float(getattr(cfg, "q_gp_var", 0.0))
        if (
            self.q_gp_var > 0.0
            and gp is not None
            and not hasattr(gp, "predict_uncertainty_torch")
        ):
            raise TypeError(
                "q_gp_var > 0 needs a GP with "
                "predict_uncertainty_torch()."
            )

        self.last_solution = None

        self.live_plot = bool(live_plot)
        self.plot_obstacle_span = None
        self._plot = None
        self._plot_off = False

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _rollout_for_plot(self, s0, controls, dt, goal_x):
        """Roll out each trajectory only until it reaches the forward x goal."""
        s = s0.clone()
        M, N = controls.shape
        goal_x = torch.as_tensor(goal_x, dtype=s.dtype, device=s.device)
        active = s[:, 0] < goal_x

        xs = [s[:, 0].clone()]
        thetas = [s[:, 2].clone()]

        for n in range(N):
            idx = torch.nonzero(active, as_tuple=False).squeeze(1)

            x_draw = torch.full(
                (M,), torch.nan, dtype=s.dtype, device=s.device
            )
            theta_draw = torch.full(
                (M,), torch.nan, dtype=s.dtype, device=s.device
            )

            if idx.numel() > 0:
                s_next, reached = self._step_to_goal(
                    s[idx], controls[idx, n], dt, goal_x
                )
                s[idx] = s_next

                x_draw[idx] = s_next[:, 0]
                theta_draw[idx] = s_next[:, 2]
                active[idx[reached]] = False

            xs.append(x_draw)
            thetas.append(theta_draw)

        return torch.stack(xs, 1), torch.stack(thetas, 1)

    def _plot_plan(self, s0, U, S, U_opt, dt, ref, n_best=100):
        if self._plot_off:
            return

        try:
            n_best = min(int(n_best), int(U.shape[0]))

            with torch.no_grad():
                idx = torch.argsort(S)[:n_best]
                sb = s0.unsqueeze(0).repeat(idx.numel(), 1)
                xs, ths = self._rollout_for_plot(
                    sb, U[idx], dt, ref[0]
                )
                xs = xs.cpu().numpy()
                ths = np.degrees(ths.cpu().numpy())

                so = s0.unsqueeze(0)
                xo, tho = self._rollout_for_plot(
                    so, U_opt.unsqueeze(0), dt, ref[0]
                )
                xo = xo.cpu().numpy().ravel()
                tho = np.degrees(tho.cpu().numpy().ravel())

            if self._plot is None:
                self._plot_setup(plt, ref, n_best)

            pl = self._plot

            for line, x, theta in zip(pl.samples, xs, ths):
                line.set_data(x, theta)

            for line in pl.samples[len(xs):]:
                line.set_data([], [])

            pl.opt.set_data(xo, tho)
            goal_x = float(ref[0].item())
            pl.goal.set_xdata([goal_x, goal_x])

            xr = float(s0[0].item())
            tr = math.degrees(float(s0[2].item()))

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
        goal_x = float(ref[0].item())

        plt.ion()
        fig, ax = plt.subplots()
        ax.set(
            xlabel="x [m]",
            ylabel="theta [deg]",
            xlim=(-2.0, 12.0),
            ylim=(-120.0, 120.0),
            title="MPPI planned trajectories (x vs theta)",
        )

        if self.plot_obstacle_span is not None:
            ax.axvspan(*self.plot_obstacle_span, color="0.85", zorder=0)

        ax.grid(True, alpha=0.3)

        samples = [
            ax.plot([], [], color="0.7", lw=0.6, alpha=0.5)[0]
            for _ in range(n_best)
        ]
        opt, = ax.plot(
            [], [], "C3-o", lw=2.0, ms=3, label="optimal plan"
        )
        trail, = ax.plot(
            [], [], "C0-", lw=1.3, alpha=0.85, label="realized"
        )
        car, = ax.plot(
            [], [], "o", ms=12, mfc="blue", mec="k", mew=1.0,
            zorder=5, label="car"
        )
        goal = ax.axvline(
            goal_x, linestyle="--", lw=1.5, label="x goal"
        )

        ax.legend(loc="upper left")
        self._plot = SimpleNamespace(
            fig=fig,
            ax=ax,
            samples=samples,
            opt=opt,
            trail=trail,
            car=car,
            goal=goal,
            trail_x=[],
            trail_t=[],
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _deriv(self, s, tau):
        x = s[:, 0]
        v = s[:, 1]
        theta = s[:, 2]
        omega = s[:, 3]

        if self.gp is not None and self.gp.ready:
            X = torch.stack([x, v, theta, omega, tau], dim=-1)
            res = self.gp.predict_torch_fast(X)
            res_v_dot = res[:, 0]
            res_w_dot = res[:, 1]
        else:
            res_v_dot = torch.zeros_like(v)
            res_w_dot = torch.zeros_like(omega)

        m = 5.1
        r = 0.081
        g = 9.81
        l = 0.2
        L_car = 0.53
        H_body = 0.30

        x_dot = v
        v_dot = tau / (m * r) + res_v_dot
        theta_dot = omega

        I_body = (1.0 / 12.0) * m * (L_car**2 + H_body**2)
        I_eff = I_body + m * l**2

        omega_dot = (
            (-tau + m * g * l * torch.cos(theta)) / I_eff
            + res_w_dot
        )

        on_ground = (theta >= 0.0) & (omega_dot > 0.0)
        omega_dot = torch.where(
            on_ground, torch.zeros_like(omega_dot), omega_dot
        )

        return torch.stack(
            [x_dot, v_dot, theta_dot, omega_dot], dim=-1
        )

    def _step(self, s, tau, dt):
        if self.integrator == "rk4":
            k1 = self._deriv(s, tau)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
        else:
            s = s + dt * self._deriv(s, tau)

        landing = (s[:, 2] >= 0.0) & (s[:, 3] > 0.0)

        return torch.stack(
            [
                s[:, 0],
                s[:, 1].clamp(-10.0, 10.0),
                s[:, 2].clamp(max=0.0),
                torch.where(
                    landing,
                    torch.zeros_like(s[:, 3]),
                    s[:, 3].clamp(-30.0, 30.0),
                ),
            ],
            dim=-1,
        )

    def _step_to_goal(self, s, tau, dt, goal_x):
        """Propagate one step and terminate exactly at the forward x goal."""
        goal_x = torch.as_tensor(goal_x, dtype=s.dtype, device=s.device)
        candidate = self._step(s, tau, dt)
        reached = candidate[:, 0] >= goal_x

        dx = candidate[:, 0] - s[:, 0]
        safe_dx = torch.where(
            dx.abs() > 1e-9, dx, torch.ones_like(dx)
        )
        alpha = ((goal_x - s[:, 0]) / safe_dx).clamp(0.0, 1.0)
        at_goal = s + alpha.unsqueeze(1) * (candidate - s)

        at_goal = torch.stack(
            [
                torch.full_like(at_goal[:, 0], goal_x),
                at_goal[:, 1],
                at_goal[:, 2],
                at_goal[:, 3],
            ],
            dim=-1,
        )

        s_next = torch.where(
            reached.unsqueeze(1), at_goal, candidate
        )
        return s_next, reached

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _stage_cost(self, s, tau, tau_prev, ref):
        cfg = self.cfg

        x = s[:, 0]
        v = s[:, 1]
        theta = s[:, 2]
        omega = s[:, 3]

        e_x = x - ref[0]
        e_v = v - ref[1]
        d_tau = tau - tau_prev

        theta_deg = torch.rad2deg(theta)
        too_much_wheelie = torch.relu((-theta_deg) - 75.0)
        nose_down = torch.relu(theta_deg)

        cost_goal = cfg.q_x * e_x**2
        cost_speed = cfg.q_v * e_v**2
        cost_pitch = cfg.q_theta * (
            too_much_wheelie**2 + nose_down**2
        )
        cost_pitch_rate = cfg.q_omega * omega**2
        cost_torque = cfg.r_tau * tau**2
        cost_smooth = cfg.r_dtau * d_tau**2

        return (
            cost_goal
            #+ cost_speed
            + cost_pitch
            + cost_pitch_rate
            # + cost_torque
            + cost_smooth
        )

    # ------------------------------------------------------------------
    # MPPI
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, state, ref, tau_prev):
        p = self.p
        cfg = self.cfg
        dev = self.device

        K = int(self.K)
        N = int(cfg.N)
        dt = float(cfg.dt)
        lam = float(self.LAM)
        sigma = float(self.SIGMA)
        tau_min = float(p.tau_min)
        tau_max = float(p.tau_max)
        tau_prev = float(tau_prev)

        s0 = torch.as_tensor(state, dtype=self.dtype, device=dev).reshape(-1)
        ref = torch.as_tensor(ref, dtype=self.dtype, device=dev).reshape(-1)
        goal_x = ref[0]

        # The controller is for forward motion. If the measured car has
        # reached or passed x_goal, do not generate another rollout batch.
        if bool((s0[0] >= goal_x).item()):
            stop_tau = min(max(0.0, tau_min), tau_max)
            self.last_solution = torch.full(
                (N,), stop_tau, dtype=self.dtype, device=dev
            )
            return stop_tau, {
                "success": True,
                "cost": 0.0,
                "goal_reached": True,
                "reached_fraction": 1.0,
                "mean_rollout_steps": 0.0,
            }

        # 1. Sample control sequences.
        U_nom = self._warm_start(N, tau_prev)
        eps = sigma * torch.randn(K, N, dtype=self.dtype, device=dev)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)
        eps = U - U_nom.unsqueeze(0)

        # 2. Roll out only active trajectories.
        s = s0.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=dev)
        prev = torch.full((K,), tau_prev, dtype=self.dtype, device=dev)
        active = s[:, 0] < goal_x
        control_used = torch.zeros(K, N, dtype=torch.bool, device=dev)

        goal_x = ref[0]
        direction = 1.0 if float(goal_x - s0[0]) >= 0.0 else -1.0
        active = direction * (goal_x - s[:, 0]) > 0.0

        for n in range(N):
            idx = torch.nonzero(active, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                break

            s_active = s[idx]
            tau_active = U[idx, n]
            prev_active = prev[idx]
            control_used[idx, n] = True

            J[idx] += self._stage_cost(
                s_active, tau_active, prev_active, ref
            )

            s_next, reached = self._step_to_goal(
                s_active, tau_active, dt, goal_x
            )
            s[idx] = s_next
            prev[idx] = tau_active
            active[idx[reached]] = False

        # 3. Terminal cost at the first goal crossing or at horizon N.
        e = s - ref
        terminal_cost_x = cfg.q_terminal_x * e[:, 0] ** 2
        
        terminal_cost_v = float(
            getattr(cfg, "q_terminal_v", 0.0)
        ) * e[:, 1] ** 2

        #terminal_cost_theta = cfg.q_terminal_theta * e[:, 2] ** 2

        terminal_cost_omega = cfg.q_terminal_omega * e[:, 3] ** 2
        
        cost_progress_terminal = -cfg.q_progress * s[:, 0]

        theta_deg = torch.rad2deg(s[:, 2])
        too_much_wheelie = torch.relu((-theta_deg) - 75.0)
        nose_down = torch.relu(theta_deg)
        terminal_cost_pitch = cfg.q_theta * (
            too_much_wheelie**2 + nose_down**2
        )

        J += (
            terminal_cost_x
            #+ terminal_cost_v
            + terminal_cost_pitch
            #+ terminal_cost_omega
            + cost_progress_terminal
        )

        J = torch.nan_to_num(
            J, nan=1e12, posinf=1e12, neginf=1e12
        )

        # 4. MPPI weights.
        rho = J.min()
        w = torch.exp(-(J - rho) / lam)

        # A rollout contributes at n only if it was still active at n.
        used = control_used.to(self.dtype)
        weighted_used = w.unsqueeze(1) * used
        numerator = (weighted_used * eps).sum(0)
        denominator = weighted_used.sum(0)

        du = torch.where(
            denominator > 1e-8,
            numerator / denominator.clamp_min(1e-8),
            torch.zeros_like(numerator),
        )

        U_opt = torch.clamp(
            U_nom + du, tau_min, tau_max
        )

        # Controls after all rollouts have terminated are not used.
        zero_tau = min(max(0.0, tau_min), tau_max)
        U_opt = torch.where(
            control_used.any(0),
            U_opt,
            torch.full_like(U_opt, zero_tau),
        )

        self.last_solution = U_opt

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, dt, ref)

        reached_fraction = float((~active).float().mean().item())
        mean_rollout_steps = float(
            control_used.sum(1).float().mean().item()
        )

        return float(U_opt[0].item()), {
            "success": True,
            "cost": float(rho.item()),
            "goal_reached": False,
            "reached_fraction": reached_fraction,
            "mean_rollout_steps": mean_rollout_steps,
        }

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            return torch.full(
                (N,), tau_prev, dtype=self.dtype, device=self.device
            )

        return torch.cat(
            [self.last_solution[1:], self.last_solution[-1:]]
        )