# utils/mppi_core.py
import math
import torch

from utils import geometry

class MPPICore:
    """
    Holds MPPI warm-start plan + implements:
      - stage_cost_torch
      - gp_step_batch_torch
      - mppi_action
    """

    def __init__(
        self,
        cfg,
        device,
        gp_xpos,
        gp_xpos_dot,
        gp_pitch,
        gp_pitch_dot,
        model_lock,
        logger=None,
        feature_map_torch=None
    ):
        # Initial setup configurations
        self.cfg = cfg
        self.device = device
        self.logger = logger

        # GP managers (hot-swapped by node)
        self.gp_xpos = gp_xpos
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch = gp_pitch
        self.gp_pitch_dot = gp_pitch_dot


        self.model_lock = model_lock

        self.pitch_target_t = torch.tensor(
            float(cfg.pitch_target), dtype=torch.float32, device=self.device
        )

        # MPPI warm start
        self.plan = None

        # Expose mean cost for logging/plotting
        self.last_mean_cost = 0.0

        self.feature_map_torch = feature_map_torch



    def feature_map_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        PALSGP-lite feature map φ(s,u) default 5D.
        """
        xpos = states[:, 0]
        xdot = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]
        uvals = actions.view(-1)
        return torch.stack([xpos, xdot, pitch, pitch_dot, uvals], dim=-1)

    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        GP input feature map φ(s,u).

        Default is backward compatible with your current GP input:
        X = [x, xdot, pitch, pitch_dot, u]  (D=5)

        states: [B,4]
        actions: [B] or [B,1]
        returns: [B,5]
        """
        if actions.ndim == 2:
            actions = actions.view(-1)
        if self.feature_map_torch is None:
            # default: match existing models (5D)
            return torch.stack(
                [states[:, 0], states[:, 1], states[:, 2], states[:, 3], actions],
                dim=-1,
            )
        # user-provided feature map should accept actions [B,1]
        return self.feature_map_torch(states, actions.view(-1, 1))




    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot):
        self.gp_xpos = gp_xpos
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch = gp_pitch
        self.gp_pitch_dot = gp_pitch_dot


    # COST FUNCTION FOR OBSTACLED

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        xpos = states[:, 0]
        pitch = states[:, 2]
        u = actions

        pitch_err = pitch - float(self.cfg.pitch_target)
        goal_err = xpos - float(self.cfg.goal_x)

        cost_pitch = float(self.cfg.w_pitch) * (pitch_err ** 2)
        cost_goal = float(self.cfg.w_goal) * (goal_err ** 2)
        cost_u = float(self.cfg.w_u) * (u ** 2)

        return cost_pitch + cost_goal + cost_u



    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [x, vx, pitch, pitch_dot]
        xpos    = states[:, 0]
        xpos_dot   = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]

        #dt = float(self.cfg.ctrl_dt)

        # X: (K,5) = [xpos, xpos_dot, pitch, pitch_dot, u]
        X = torch.stack([xpos, xpos_dot, pitch, pitch_dot, actions], dim=-1)

        dxpos_mean   = self.gp_xpos.predict_torch(X)
        dxpos_dot_mean   = self.gp_xpos_dot.predict_torch(X)
        dpitch_mean    = self.gp_pitch.predict_torch(X)
        dpitch_dot_mean    = self.gp_pitch_dot.predict_torch(X)

        next_states = torch.empty_like(states)
        next_states[:, 0] = xpos        + dxpos_mean
        next_states[:, 1] = xpos_dot    + dxpos_dot_mean
        next_states[:, 2] = pitch       + dpitch_mean
        next_states[:, 3] = pitch_dot   + dpitch_dot_mean

        # reasonable clamps (tune)
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)

        return next_states


    @torch.no_grad()
    def action(self, x0_np):
        cfg = self.cfg
        H, K = cfg.horizon, cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        if x0.shape != (4,):
            raise ValueError(f"x0 must be shape (4,), got {tuple(x0.shape)}")

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        eps = torch.randn(K, H, device=self.device) * cfg.sigma
        U = torch.clamp(u_init.unsqueeze(0) + eps, cfg.u_min, cfg.u_max)
        delta = U - u_init.unsqueeze(0)          # <-- important

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            costs = costs + self.stage_cost_torch(states, u_t)
            states = self.gp_step_batch_torch(states, u_t)

        self.last_mean_cost = float(costs.mean().item())

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / cfg.lambda_)
        wsum = w.sum() + 1e-8

        #du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
        du = (w.unsqueeze(1) * delta).sum(dim=0) / wsum

        u_new = torch.clamp(u_init + du, cfg.u_min, cfg.u_max)

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        return float(u_new[0].detach().cpu())