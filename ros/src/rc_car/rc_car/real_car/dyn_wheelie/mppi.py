#!/usr/bin/env python3

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from params_mppi import MPPIConfig


class MPPITorch:
    # Same sampler structure as the previous MuJoCo implementation.
    # K = MPPIConfig.K
    # SIGMA = MPPIConfig.SIGMA
    # LAM = MPPIConfig.LAM

    def __init__(
        self,
        p,
        cfg,
        device=None,
        fast=True,
        integrator="rk4",
        dtype=None,
        live_plot=False,
    ):

        self.p = p
        self.cfg = cfg

        self.fast = bool(fast)
        self.integrator = str(integrator).lower()

        self.dtype = (
            dtype
            if dtype is not None
            else (
                torch.float32
                if fast
                else torch.double
            )
        )

        self.device = torch.device("cuda")

        self.last_solution = None

        # ----------------------------------------------------
        # Optional plan plot
        # ----------------------------------------------------

        self.live_plot = bool(live_plot)
        self.plot_obstacle_span = None
        self._plot = None
        self._plot_off = False

        print(
            "PyTorch device:",
            self.device,
        )

        if self.device.type == "cuda":
            print(
                "CUDA device:",
                torch.cuda.get_device_name(
                    self.device
                ),
            )

    # ========================================================
    # OPTIONAL LIVE PLAN PLOT
    # ========================================================

    def _plot_plan(
        self,
        s0,
        U,
        J,
        U_opt,
        dt,
        ref,
        n_best=50,
    ):

        if self._plot_off:
            return

        try:
            N = int(U.shape[1])

            with torch.no_grad():
                idx = torch.argsort(J)[:n_best]

                sb = (
                    s0.unsqueeze(0)
                    .repeat(idx.numel(), 1)
                )

                xs = [
                    sb[:, 0].clone()
                ]

                ths = [
                    sb[:, 2].clone()
                ]

                for n in range(N):
                    sb = self._step(
                        sb,
                        U[idx, n],
                        dt,
                    )

                    xs.append(
                        sb[:, 0].clone()
                    )

                    ths.append(
                        sb[:, 2].clone()
                    )

                xs = (
                    torch.stack(xs, 1)
                    .cpu()
                    .numpy()
                )

                ths = np.degrees(
                    torch.stack(ths, 1)
                    .cpu()
                    .numpy()
                )

                # Optimal plan
                so = s0.unsqueeze(0)

                xo = [
                    so[:, 0].clone()
                ]

                tho = [
                    so[:, 2].clone()
                ]

                for n in range(N):
                    so = self._step(
                        so,
                        U_opt[n:n + 1],
                        dt,
                    )

                    xo.append(
                        so[:, 0].clone()
                    )

                    tho.append(
                        so[:, 2].clone()
                    )

                xo = (
                    torch.stack(xo, 1)
                    .cpu()
                    .numpy()
                    .ravel()
                )

                tho = np.degrees(
                    torch.stack(tho, 1)
                    .cpu()
                    .numpy()
                    .ravel()
                )

            if self._plot is None:
                self._plot_setup(
                    plt,
                    ref,
                    n_best,
                )

            pl = self._plot

            for line, x, theta in zip(
                pl.samples,
                xs,
                ths,
            ):
                line.set_data(
                    x,
                    theta,
                )

            pl.opt.set_data(
                xo,
                tho,
            )

            xr = float(s0[0])
            tr = math.degrees(
                float(s0[2])
            )

            if (
                pl.trail_x
                and pl.trail_x[-1] - xr > 0.5
            ):
                pl.trail_x.clear()
                pl.trail_t.clear()

            pl.trail_x.append(xr)
            pl.trail_t.append(tr)

            pl.trail.set_data(
                pl.trail_x,
                pl.trail_t,
            )

            pl.car.set_data(
                [xr],
                [tr],
            )

            pl.fig.canvas.draw_idle()
            pl.fig.canvas.flush_events()
            plt.pause(0.001)

        except Exception as exc:
            print(
                f"[plot_plan] disabled ({exc})"
            )
            self._plot_off = True

    def _plot_setup(
        self,
        plt,
        ref,
        n_best,
    ):

        plt.ion()

        fig, ax = plt.subplots()

        ax.set(
            xlabel="x [m]",
            ylabel="theta [deg]",
            xlim=(-2.0, 10.0),
            ylim=(-100.0, 10.0),
            title="MPPI planned trajectories (x vs theta)",
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        samples = [
            ax.plot(
                [],
                [],
                color="0.7",
                lw=0.6,
                alpha=0.5,
            )[0]
            for _ in range(n_best)
        ]

        opt, = ax.plot(
            [],
            [],
            "C3-o",
            lw=2.0,
            ms=3,
            label="optimal plan",
        )

        trail, = ax.plot(
            [],
            [],
            "C0-",
            lw=1.3,
            alpha=0.85,
            label="realized",
        )

        car, = ax.plot(
            [],
            [],
            "o",
            ms=12,
            mfc="blue",
            mec="k",
            mew=1.0,
            zorder=5,
            label="car",
        )

        ax.legend(
            loc="upper left"
        )

        self._plot = SimpleNamespace(
            fig=fig,
            ax=ax,
            samples=samples,
            opt=opt,
            trail=trail,
            car=car,
            trail_x=[],
            trail_t=[],
        )

    # ========================================================
    # PHYSICAL DYNAMICS
    # ========================================================

    def _deriv(self, s, tau):
        """
        State:
            [x, v, theta, omega]

        Real-car convention:
            negative theta = nose up
            negative omega = rotating nose up

        tau is physical rear-wheel torque [N m].
        """
        p = self.p

        v = s[..., 1]
        theta = s[..., 2]
        omega = s[..., 3]

        x_dot = v
        v_dot = (tau / (p.m * p.r) - p.c_v * v)
        theta_dot = omega

        gravity_torque = p.m * p.g * (
            p.x_com * torch.cos(theta)
            + p.h_com * torch.sin(theta)
        )

        omega_dot = (-tau + gravity_torque) / p.I_eff

        #omega_dot = (-tau + p.gravity_torque * torch.cos(theta)) / p.I_eff

        # At level ground, positive angular acceleration
        # would rotate nose-down through the floor.
        on_ground = (
            (theta >= 0.0)
            & (omega_dot > 0.0)
        )

        omega_dot = torch.where(
            on_ground,
            torch.zeros_like(
                omega_dot
            ),
            omega_dot,
        )

        return torch.stack(
            [
                x_dot,
                v_dot,
                theta_dot,
                omega_dot,
            ],
            dim=-1,
        )

    # ========================================================
    # INTEGRATOR
    # ========================================================

    def _step(
        self,
        s,
        tau,
        dt,
    ):

        if self.integrator == "rk4":

            k1 = self._deriv(s, tau,)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        else:

            s = (s + dt * self._deriv(s, tau))

        landing = (
            (s[:, 2] >= 0.0)
            & (s[:, 3] > 0.0)
        )

        s = torch.stack(
            [
                s[:, 0],

                s[:, 1].clamp(
                    self.p.v_min,
                    self.p.v_max,
                ),

                s[:, 2].clamp(
                    min=self.p.theta_min,
                    max=self.p.theta_max,
                ),

                torch.where(
                    landing,
                    torch.zeros_like(
                        s[:, 3]
                    ),
                    s[:, 3].clamp(
                        self.p.omega_min,
                        self.p.omega_max,
                    ),
                ),
            ],
            dim=-1,
        )

        return s

    # ========================================================
    # STAGE COST
    # ========================================================

    def _stage_cost(self, s, tau, tau_prev, ref):

        cfg = self.cfg
        x = s[:, 0]
        v = s[:, 1]
        theta = s[:, 2]
        omega = s[:, 3]

        e_x = x - ref[0]
        e_v = v - ref[1]
        e_theta = theta - ref[2]
        e_omega = omega - ref[3]
        d_tau = tau - tau_prev

        # Keep the same degree-based pitch cost structure
        # used by your previous controller.
        e_theta_deg = torch.rad2deg(e_theta)

        cost_goal = (cfg.q_x * e_x**2)
        cost_speed = (cfg.q_v * e_v**2)
        # cost_pitch_deg = (cfg.q_theta * e_theta_deg**2)
        cost_pitch = (cfg.q_theta * e_theta**2)
        cost_pitch_rate = (cfg.q_omega * e_omega**2)
        cost_torque = (cfg.r_tau * tau**2)
        cost_smooth = (cfg.r_dtau * d_tau**2)

        return (
            #cost_goal
            #+ cost_speed
            + cost_pitch
            + cost_pitch_rate
            + cost_torque
            + cost_smooth
        )

    # ========================================================
    # MPPI SOLVE
    # ========================================================

    @torch.no_grad()
    def solve(self, state, ref, tau_prev,):

        K = int(self.cfg.K)
        N = int(self.cfg.N)
        dt = float(self.cfg.dt)

        lam = float(self.cfg.LAM)
        sigma = float(self.cfg.SIGMA)

        p = self.p
        cfg = self.cfg
        dev = self.device

        # K = self.K
        # N = int(cfg.N)
        # dt = float(cfg.dt)
        # lam = float(self.LAM)
        # sigma = float(self.SIGMA)

        tau_min = float(
            p.tau_min
        )

        tau_max = float(
            p.tau_max
        )

        tau_prev = float(
            tau_prev
        )

        s0 = torch.as_tensor(state, dtype=self.dtype, device=dev,).reshape(-1)
        ref = torch.as_tensor(ref, dtype=self.dtype, device=dev, ).reshape(-1)
        U_nom = self._warm_start(N, tau_prev,)

        eps = (
            sigma
            * torch.randn(
                K,
                N,
                dtype=self.dtype,
                device=dev,
            )
        )

        U = torch.clamp(
            U_nom.unsqueeze(0)
            + eps,
            tau_min,
            tau_max,
        )

        # Actual perturbation after clipping.
        eps = U - U_nom.unsqueeze(0)
        s = s0.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=dev,)
        prev = torch.full(
            (K,),
            tau_prev,
            dtype=self.dtype,
            device=dev,
        )

        for n in range(N):
            tau = U[:, n]

            J = (
                J
                + self._stage_cost(
                    s,
                    tau,
                    prev,
                    ref,
                )
            )

            # IMPORTANT:
            # physical dynamics directly.
            # No RLS parameters.
            s = self._step(
                s,
                tau,
                dt,
            )

            prev = tau

        # ----------------------------------------------------
        # Terminal cost
        # ----------------------------------------------------
        e = (s - ref)
        e_theta_deg = torch.rad2deg(e[:, 2])

        J = J + (
            # cfg.q_terminal_x * e[:, 0]**2
            + cfg.q_terminal_v * e[:, 1]**2
            + cfg.q_terminal_theta * e[:, 2]**2
            + cfg.q_terminal_omega * e[:, 3]**2
        )

        J = torch.nan_to_num(
            J,
            nan=1e12,
            posinf=1e12,
            neginf=1e12,
        )

        rho = J.min()
        w = torch.exp(-(J - rho) / lam)
        du = ((w.unsqueeze(1) * eps).sum(0) / (w.sum() + 1e-8))

        U_opt = torch.clamp(U_nom + du, tau_min, tau_max,)

        self.last_solution = (U_opt.detach().clone())

        # ----------------------------------------------------
        # Optimized prediction for diagnostics
        # ----------------------------------------------------

        predicted = [s0.clone()]

        s_opt = s0.unsqueeze(0)

        for n in range(N):
            s_opt = self._step(
                s_opt,
                U_opt[n:n + 1],
                dt,
            )

            predicted.append(
                s_opt[0].clone()
            )

        predicted = torch.stack(
            predicted,
            dim=0,
        )

        if self.live_plot:
            self._plot_plan(s0, U, J, U_opt, dt, ref,)

        info = {
            "success": True,
            "cost": float(
                rho.item()
            ),
            "predicted_states": (
                predicted
                .cpu()
                .numpy()
            ),
            "u_plan": (
                U_opt
                .cpu()
                .numpy()
            ),
        }

        return (
            float(
                U_opt[0].item()
            ),
            info,
        )

    # ========================================================
    # WARM START
    # ========================================================
    def _warm_start(self, N, tau_prev):

        if self.last_solution is None:

            return torch.full(
                (N,),
                self.p.tau_acquire,
                dtype=self.dtype,
                device=self.device,
            )

        return torch.cat(
            [
                self.last_solution[1:],
                self.last_solution[-1:],
            ]
        )
