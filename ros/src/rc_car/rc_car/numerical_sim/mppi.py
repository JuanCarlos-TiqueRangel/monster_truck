from dataclasses import dataclass

import numpy as np
import torch

try:
    from .rls import nominal_rls_parameters, rk4_step_torch
except ImportError:
    from rls import nominal_rls_parameters, rk4_step_torch


@dataclass
class MPPIConfig:
    dt: float = 0.05
    N: int = 10
    num_samples: int = 4096
    noise_sigma: float = 2.5
    temperature: float = 20.0

    q_x: float = 0.0
    q_v: float = 0.001
    q_theta: float = 50.0
    q_omega: float = 0.3
    r_tau: float = 0.0003
    r_dtau: float = 0.025
    q_terminal_x: float = 0.0
    q_terminal_theta: float = 0.4
    q_terminal_omega: float = 0.08

    dtype: torch.dtype = torch.float32
    device: str = "cuda"


@torch.no_grad()
def mppi_update_torch(
    state: torch.Tensor,
    ref: torch.Tensor,
    tau_prev: torch.Tensor,
    a_rls: torch.Tensor,
    u_nominal: torch.Tensor,
    noise: torch.Tensor,
    cfg: MPPIConfig,
    p,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = noise.shape[0]
    u_samples = torch.clamp(u_nominal[None, :] + noise, p.tau_min, p.tau_max)
    effective_noise = u_samples - u_nominal[None, :]
    states = state[None, :].repeat(K, 1)
    costs = torch.zeros(K, dtype=state.dtype, device=state.device)
    tau_previous = torch.full(
        (K,), tau_prev.item(), dtype=state.dtype, device=state.device
    )

    for k in range(cfg.N):
        tau_k = u_samples[:, k]
        e = states - ref[None, :]
        du = tau_k - tau_previous
        pitch_error = e[:, 2]
        pitch_below_min = torch.relu(p.theta_min - states[:, 2])
        pitch_above_max = torch.relu(states[:, 2] - p.theta_max)
        costs += (
            cfg.q_x * e[:, 0] ** 2
            + cfg.q_v * e[:, 1] ** 2
            + cfg.q_theta * pitch_error**2
            + cfg.q_theta * (pitch_below_min**2 + pitch_above_max**2)
            + cfg.q_omega * e[:, 3] ** 2
            + cfg.r_tau * tau_k**2
            + cfg.r_dtau * du**2
        )
        states = rk4_step_torch(states, tau_k, cfg.dt, a_rls)
        tau_previous = tau_k

    e_terminal = states - ref[None, :]
    costs += (
        cfg.q_terminal_x * e_terminal[:, 0] ** 2
        + cfg.q_v * e_terminal[:, 1] ** 2
        + cfg.q_terminal_theta * e_terminal[:, 2] ** 2
        + cfg.q_terminal_omega * e_terminal[:, 3] ** 2
    )
    costs = torch.nan_to_num(costs, nan=1.0e20, posinf=1.0e20, neginf=1.0e20)

    beta = torch.min(costs)
    weights = torch.exp(-(costs - beta) / max(cfg.temperature, 1.0e-9))
    weights = weights / (torch.sum(weights) + 1.0e-8)
    control_update = torch.sum(weights[:, None] * effective_noise, dim=0)
    u_new = u_nominal + control_update
    return torch.clamp(u_new, p.tau_min, p.tau_max), costs, weights


class WheelieMPPITorch:
    def __init__(self, p, cfg: MPPIConfig, rls_parameters=None):
        self.p = p
        self.cfg = cfg

        if cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        self.u_nominal = torch.zeros(cfg.N, dtype=self.dtype, device=self.device)
        if rls_parameters is None:
            rls_parameters = nominal_rls_parameters(p)
        self.rls_parameters = torch.as_tensor(
            rls_parameters, dtype=self.dtype, device=self.device
        ).clone()

        print("PyTorch device:", self.device)
        if self.device.type == "cuda":
            print("CUDA device:", torch.cuda.get_device_name(self.device))

    def solve(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        tau_prev: float,
        a_rls: np.ndarray | None = None,
    ) -> tuple[float, dict]:
        cfg, p = self.cfg, self.p
        if not p.theta_min <= float(ref[2]) <= p.theta_max:
            raise ValueError(
                "Pitch reference must lie between p.theta_min and p.theta_max"
            )

        state_t = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        ref_t = torch.as_tensor(ref, dtype=self.dtype, device=self.device)
        tau_prev_t = torch.tensor(tau_prev, dtype=self.dtype, device=self.device)
        if a_rls is not None:
            self.set_rls_parameters(a_rls)
        noise = cfg.noise_sigma * torch.randn(
            (cfg.num_samples, cfg.N), dtype=self.dtype, device=self.device
        )

        u_new, costs, weights = mppi_update_torch(
            state_t,
            ref_t,
            tau_prev_t,
            self.rls_parameters,
            self.u_nominal,
            noise,
            cfg,
            p,
        )
        tau_cmd = float(u_new[0].detach().cpu().item())

        self.u_nominal = torch.cat(
            [u_new[1:].detach(), u_new[-1:].detach()], dim=0
        )
        info = {
            "cost_min": float(torch.min(costs).detach().cpu().item()),
            "cost_mean": float(torch.mean(costs).detach().cpu().item()),
            "effective_sample_size": float(
                (1.0 / torch.sum(weights**2)).detach().cpu().item()
            ),
            "tau_cmd": tau_cmd,
        }
        return tau_cmd, info

    def set_rls_parameters(self, parameters: np.ndarray) -> None:
        """Copy the latest learned RLS coefficients to the rollout device."""
        self.rls_parameters.copy_(
            torch.as_tensor(parameters, dtype=self.dtype, device=self.device)
        )


# Same controller naming used by rc_car/mujoco/wheelie/mppi.py.
MPPITorch = WheelieMPPITorch
