#!/usr/bin/env python3

import numpy as np
import torch

from rls import rls_dynamics_torch


class MPPIRLSTorch:
    def __init__(self, p, cfg, rls_parameters, device=None, fast=True, integrator="rk4", dtype=None):
        self.p = p
        self.cfg = cfg
        self.fast = bool(fast)
        self.integrator = str(integrator).lower()
        self.dtype = dtype if dtype is not None else (torch.float32 if fast else torch.double)

        self.device = torch.device("cuda")

        self.rls_parameters = torch.as_tensor(rls_parameters, dtype=self.dtype, device=self.device).clone()
        self.last_solution = None

        print("PyTorch device:", self.device)
        if self.device.type == "cuda":
            print("CUDA device:", torch.cuda.get_device_name(self.device))

    def set_rls_parameters(self, parameters):
        self.rls_parameters.copy_(torch.as_tensor(parameters, dtype=self.dtype, device=self.device))

    def _deriv(self, s, tau):
        return rls_dynamics_torch(s, tau, self.rls_parameters)

    def _step(self, s, tau, dt):
        if self.integrator == "rk4":
            k1 = self._deriv(s, tau)
            k2 = self._deriv(s + 0.5 * dt * k1, tau)
            k3 = self._deriv(s + 0.5 * dt * k2, tau)
            k4 = self._deriv(s + dt * k3, tau)
            s = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            s = s + dt * self._deriv(s, tau)

        landing = (s[:, 2] >= 0.0) & (s[:, 3] > 0.0)

        return torch.stack([
            s[:, 0],
            s[:, 1].clamp(self.p.v_min, self.p.v_max),
            s[:, 2].clamp(self.p.theta_min, self.p.theta_max),
            torch.where(
                landing,
                torch.zeros_like(s[:, 3]),
                s[:, 3].clamp(self.p.omega_min, self.p.omega_max),
            ),
        ], dim=-1)


    def _stage_cost(self, s, tau, tau_prev, ref):
        cfg = self.cfg
        e_x = s[:, 0] - ref[0]
        e_v = s[:, 1] - ref[1]
        e_theta = s[:, 2] - ref[2]
        e_omega = s[:, 3] - ref[3]
        d_tau = tau - tau_prev

        return (
            cfg.q_x * e_x.square()
            + cfg.q_v * e_v.square()
            + cfg.q_theta * e_theta.square()
            + cfg.q_omega * e_omega.square()
            + cfg.r_tau * tau.square()
            + cfg.r_dtau * d_tau.square()
        )

    @torch.no_grad()
    def solve(self, state, ref, tau_prev):
        p, cfg = self.p, self.cfg
        K, N, dt = int(cfg.K), int(cfg.N), float(cfg.dt)
        sigma, lam = float(cfg.SIGMA), float(cfg.LAM)
        tau_min, tau_max, tau_prev = float(p.tau_min), float(p.tau_max), float(tau_prev)

        s0 = torch.as_tensor(state, dtype=self.dtype, device=self.device).reshape(4)
        ref_t = torch.as_tensor(ref, dtype=self.dtype, device=self.device).reshape(4)

        U_nom = self._warm_start(N, tau_prev)
        eps = sigma * torch.randn(K, N, dtype=self.dtype, device=self.device)
        U = torch.clamp(U_nom.unsqueeze(0) + eps, tau_min, tau_max)
        eps = U - U_nom.unsqueeze(0)

        s = s0.unsqueeze(0).repeat(K, 1)
        J = torch.zeros(K, dtype=self.dtype, device=self.device)
        prev = torch.full((K,), tau_prev, dtype=self.dtype, device=self.device)

        for n in range(N):
            tau = U[:, n]
            J += self._stage_cost(s, tau, prev, ref_t)
            s = self._step(s, tau, dt)
            prev = tau

        e = s - ref_t
        J += (
            cfg.q_terminal_x * e[:, 0].square()
            + cfg.q_terminal_v * e[:, 1].square()
            + cfg.q_terminal_theta * e[:, 2].square()
            + cfg.q_terminal_omega * e[:, 3].square()
        )

        J = torch.nan_to_num(J, nan=1e12, posinf=1e12, neginf=1e12)

        rho = J.min()
        w = torch.exp(-(J - rho) / max(lam, 1e-9))
        w = w / (w.sum() + 1e-8)

        du = torch.sum(w.unsqueeze(1) * eps, dim=0)
        U_opt = torch.clamp(U_nom + du, tau_min, tau_max)
        self.last_solution = U_opt.detach().clone()

        predicted = [s0.clone()]
        s_opt = s0.unsqueeze(0)

        for n in range(N):
            s_opt = self._step(s_opt, U_opt[n:n + 1], dt)
            predicted.append(s_opt[0].clone())

        predicted = torch.stack(predicted, dim=0)
        ess = 1.0 / (torch.sum(w.square()) + 1e-12)

        info = {
            "success": True,
            "cost": float(rho.item()),
            "cost_mean": float(J.mean().item()),
            "ess": float(ess.item()),
            "predicted_states": predicted.cpu().numpy(),
            "u_plan": U_opt.cpu().numpy(),
        }

        return float(U_opt[0].item()), info

    def _warm_start(self, N, tau_prev):
        if self.last_solution is None:
            initial_tau = float(getattr(self.p, "tau_acquire", tau_prev))
            return torch.full((N,), initial_tau, dtype=self.dtype, device=self.device)

        return torch.cat([self.last_solution[1:], self.last_solution[-1:]])
