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

# ---------------------------------------------------------------------------
# Batched (GPU) rollout model -- merged in from the former mppi_dynamics.py.
# Same gray-box model the CasADi NMPC uses (RLS linear part + sparse-GP residual
# mean), vectorised over K rollouts so one MPPI step is a single GPU kernel:
#     x_dot     = v
#     v_dot     = b . [tau, v, sin(theta), 1]        + gp_v(z)
#     theta_dot = omega
#     omega_dot = a . [cos(theta), tau, omega, v, 1] + gp_omega(z)
#     z = [theta, omega, v, tau]
# Regressor order MUST match rls.py (omega=[cos th, tau, om, v, 1],
#                                     v   =[tau, v, sin th, 1]).
# ---------------------------------------------------------------------------

# Physical clamps on the GP residual contribution (identical to nmpc.py), so a
# mis-learned / extrapolating GP cannot inject unphysical accelerations.
GP_V_CLAMP = 1.5 * 9.81      # max |longitudinal residual accel|  [m/s^2]
GP_OMEGA_CLAMP = 8.0         # max |pitch residual accel|         [rad/s^2]


def _soft_clip(val, lim):
    """Smooth, differentiable clamp to [-lim, lim] (matches nmpc.py)."""
    return lim * torch.tanh(val / lim)


def gp_mean_batch(z, Z, alpha_v, alpha_w, inv_l2, sf2):
    """Batched sparse-GP mean for both channels. z:(K,4) [theta,omega,v,tau],
    Z:(M,4) inducing, alpha_*:(M,), inv_l2:(4,), sf2:float -> (gp_v, gp_w) each
    (K,). Inactive dictionary slots carry alpha=0 so they contribute nothing."""
    diff = z[:, None, :] - Z[None, :, :]                 # (K, M, 4)
    d2 = (diff * diff * inv_l2[None, None, :]).sum(-1)   # (K, M)
    k = sf2 * torch.exp(-0.5 * d2)                       # (K, M)
    gp_v = _soft_clip(k @ alpha_v, GP_V_CLAMP)
    gp_w = _soft_clip(k @ alpha_w, GP_OMEGA_CLAMP)
    return gp_v, gp_w


def f_batch(state, tau, a, b, Z, alpha_v, alpha_w, inv_l2, sf2):
    """Continuous-time batched dynamics x_dot=f(x,u). state:(K,4)=[x,v,theta,
    omega], tau:(K,), a:(5,) angular RLS, b:(4,) linear RLS -> x_dot:(K,4)."""
    v = state[:, 1]
    theta = state[:, 2]
    omega = state[:, 3]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    z = torch.stack([theta, omega, v, tau], dim=-1)      # (K, 4)
    gp_v, gp_w = gp_mean_batch(z, Z, alpha_v, alpha_w, inv_l2, sf2)
    # v_dot = b . [tau, v, sin(theta), 1] + gp_v
    v_dot = b[0] * tau + b[1] * v + b[2] * sin_t + b[3] + gp_v
    # omega_dot = a . [cos(theta), tau, omega, v, 1] + gp_w
    omega_dot = a[0] * cos_t + a[1] * tau + a[2] * omega + a[3] * v + a[4] + gp_w
    return torch.stack([v, v_dot, omega, omega_dot], dim=-1)


def rk4_batch(state, tau, dt, a, b, Z, alpha_v, alpha_w, inv_l2, sf2):
    """One RK4 integration step, batched over K."""
    args = (a, b, Z, alpha_v, alpha_w, inv_l2, sf2)
    k1 = f_batch(state, tau, *args)
    k2 = f_batch(state + 0.5 * dt * k1, tau, *args)
    k3 = f_batch(state + 0.5 * dt * k2, tau, *args)
    k4 = f_batch(state + dt * k3, tau, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


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

    # Goal-distance velocity reference: the velocity cost tracks
    #   v_ref = clip(v_ref_gain*(goal_x - x), +-v_cruise)
    # so it cruises toward the goal and v_ref -> 0 AT the goal, making the
    # optimiser brake to a stop instead of overshooting. v_ref_gain=0 recovers
    # the old q_v*v^2 cost (which does not stop).
    v_ref_gain: float = 0.6
    v_cruise: float = 1.2

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
        gp_params: flat [Z(M*d), alpha_v(M), alpha_w(M)]  (streaming GP's mpc_params())
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

        gx = goal[0]
        vc = cfg.v_cruise
        for k in range(N):
            tau_k = u[:, k]
            e = states - goal[None, :]
            du = tau_k - tau_prev_k
            v = states[:, 1]
            theta = states[:, 2]
            # goal-distance velocity reference: brake to a stop at the goal
            v_ref = torch.clamp(cfg.v_ref_gain * (gx - states[:, 0]), -vc, vc)

            cost = cost + (
                cfg.q_x * e[:, 0] ** 2
                + cfg.q_v * (v - v_ref) ** 2
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
        v_refN = torch.clamp(cfg.v_ref_gain * (gx - states[:, 0]), -vc, vc)
        cost = cost + (
            cfg.q_x * eN[:, 0] ** 2
            + cfg.q_v * (states[:, 1] - v_refN) ** 2
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
