#!/usr/bin/env python3
"""
mppi.py
-------
Model Predictive Path Integral (MPPI) controller for the wheelie / obstacle
task. Drop-in alternative to the CasADi NMPC: same state, same goal, same
gray-box model (RLS + sparse GP), same cost weights -- only the optimiser
differs (sampling vs. gradient/IPOPT).

Why MPPI here:
  * no smoothness / differentiability requirement -> we can add a hard,
    non-smooth FLIP penalty that IPOPT cannot use (this is MPPI's real edge);
  * global sampling escapes the local minima / knife-edge tuning the gradient
    NMPC suffered from on the wheelie maneuver;
  * the rollout is one batched GPU kernel -> real-time on a Jetson-class board,
    using only the learned model (no simulator) -> deployable on the 1:8 car.

solve() mirrors WheelieNMPC.solve(state, ref, tau_prev, a, b, gp_params) so the
same simulation harness drives either controller.
"""

from dataclasses import dataclass

import numpy as np
import torch

from mppi_dynamics import rk4_batch


@dataclass
class MPPIConfig:
    dt: float = 0.05
    N: int = 25                  # horizon steps
    num_samples: int = 2048      # K rollouts (GPU-parallel)
    temperature: float = 10.0    # lambda; lower = greedier toward best samples
    noise_sigma: float = 4.0     # torque exploration std [N.m]

    # Cost weights (same meaning as MPCConfig, so a fair comparison)
    q_x: float = 15.0
    q_v: float = 8.0
    q_theta: float = 6.0
    q_omega: float = 60.0
    r_tau: float = 0.05
    r_dtau: float = 5.0
    q_terminal_theta: float = 6.0
    q_terminal_omega: float = 60.0

    # MPPI-only, non-smooth terms (impossible/awkward for the gradient NMPC)
    flip_threshold_deg: float = 85.0    # |pitch| beyond this counts as a flip
    flip_penalty: float = 5.0e4         # flat cost added to any rollout that flips
    v_barrier: float = 50.0             # soft penalty weight for |v| > p.v_max

    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    seed: int = 0


class WheelieMPPI:
    def __init__(self, p, cfg: MPPIConfig, gp_cfg):
        self.p = p
        self.cfg = cfg
        self.gp_cfg = gp_cfg
        self.M = gp_cfg.max_points
        self.d = 4

        dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(dev)
        self.dtype = cfg.dtype

        # Fixed GP kernel constants on device
        l = np.asarray(gp_cfg.lengthscales, dtype=float).reshape(-1)
        self.inv_l2 = torch.tensor(1.0 / l**2, dtype=self.dtype, device=self.device)
        self.sf2 = float(gp_cfg.sf2)

        self.flip_threshold = np.radians(cfg.flip_threshold_deg)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.u_nominal = torch.zeros(cfg.N, dtype=self.dtype, device=self.device)
        self.last_solution = None  # kept for interface symmetry with the NMPC

        if self.device.type == "cuda":
            print("MPPI on CUDA:", torch.cuda.get_device_name(self.device))
        else:
            print("MPPI on CPU")

    def reset(self):
        self.u_nominal.zero_()

    @torch.no_grad()
    def solve(self, state, ref, tau_prev, a, b, gp_params):
        """
        state    : (4,) [x, v, theta, omega]
        ref      : (4,) goal [x, v, theta, omega]
        tau_prev : float
        a, b     : RLS coefficients (5,), (4,)
        gp_params: flat [Z(M*d), alpha_v(M), alpha_w(M)]  (== GPResidual.mpc_params())
        Returns (tau, info).
        """
        cfg, p, M, d = self.cfg, self.p, self.M, self.d
        dev, dt = self.device, self.cfg.dt

        st = torch.as_tensor(state, dtype=self.dtype, device=dev)
        goal = torch.as_tensor(ref, dtype=self.dtype, device=dev)
        a_t = torch.as_tensor(np.asarray(a), dtype=self.dtype, device=dev)
        b_t = torch.as_tensor(np.asarray(b), dtype=self.dtype, device=dev)

        gpp = torch.as_tensor(np.asarray(gp_params), dtype=self.dtype, device=dev)
        Z = gpp[:M * d].reshape(M, d)
        alpha_v = gpp[M * d: M * d + M]
        alpha_w = gpp[M * d + M: M * d + 2 * M]

        # --- sample K control sequences (warm-started nominal + noise) ---
        K, N = cfg.num_samples, cfg.N
        noise = cfg.noise_sigma * torch.randn((K, N), dtype=self.dtype,
                                              device=dev, generator=self.gen)
        u = torch.clamp(self.u_nominal[None, :] + noise, p.tau_min, p.tau_max)

        # --- batched rollout + cost ---
        states = st[None, :].repeat(K, 1)            # (K, 4)
        cost = torch.zeros(K, dtype=self.dtype, device=dev)
        tau_prev_k = torch.full((K,), float(tau_prev), dtype=self.dtype, device=dev)
        flip_thr = self.flip_threshold
        vmax = p.v_max

        for k in range(N):
            tau_k = u[:, k]
            e = states - goal[None, :]
            du = tau_k - tau_prev_k
            v = states[:, 1]
            theta = states[:, 2]

            cost = cost + (
                cfg.q_x * e[:, 0] ** 2
                + cfg.q_v * e[:, 1] ** 2
                + cfg.q_theta * e[:, 2] ** 2
                + cfg.q_omega * e[:, 3] ** 2
                + cfg.r_tau * tau_k ** 2
                + cfg.r_dtau * du ** 2
                # MPPI-only non-smooth terms:
                + cfg.flip_penalty * (theta.abs() > flip_thr).to(self.dtype)
                + cfg.v_barrier * torch.relu(v.abs() - vmax) ** 2
            )
            states = rk4_batch(states, tau_k, dt, a_t, b_t,
                               Z, alpha_v, alpha_w, self.inv_l2, self.sf2)
            tau_prev_k = tau_k

        eN = states - goal[None, :]
        cost = cost + (
            cfg.q_x * eN[:, 0] ** 2
            + cfg.q_v * eN[:, 1] ** 2
            + cfg.q_terminal_theta * eN[:, 2] ** 2
            + cfg.q_terminal_omega * eN[:, 3] ** 2
        )
        cost = torch.nan_to_num(cost, nan=1e20, posinf=1e20, neginf=1e20)

        # --- path-integral weighting ---
        beta = torch.min(cost)
        w = torch.softmax(-(cost - beta) / max(cfg.temperature, 1e-9), dim=0)

        u_new = torch.clamp((w[:, None] * u).sum(0), p.tau_min, p.tau_max)
        tau = float(u_new[0].item())

        # receding-horizon warm start
        self.u_nominal = torch.cat([u_new[1:], u_new[-1:].clone()])

        ess = float((1.0 / torch.sum(w ** 2)).item())   # effective sample size
        info = {"success": True, "cost": float(beta.item()), "ess": ess}
        return tau, info
