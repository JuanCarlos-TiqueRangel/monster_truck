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

        self._custom_feature_map = feature_map_torch

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
        if self._custom_feature_map is None:
            # default: match existing models (5D)
            return torch.stack(
                [states[:, 0], states[:, 1], states[:, 2], states[:, 3], actions],
                dim=-1,
            )
        # user-provided feature map should accept actions [B,1]
        return self._custom_feature_map(states, actions.view(-1, 1))




    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot):
        self.gp_xpos = gp_xpos
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch = gp_pitch
        self.gp_pitch_dot = gp_pitch_dot


    # COST FUNCTION FOR OBSTACLED

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor,
                         uncertainty: torch.Tensor | None = None) -> torch.Tensor:
        # xpos, xpos_dot, pitch, pitch_dot
        xpos = states[:, 0]
        xpos_dot = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]
        u = actions

        pitch_err = pitch - float(self.cfg.pitch_target)
        goal_err = xpos - float(self.cfg.goal_x)

        #cost_pitch = float(self.cfg.w_pitch) * (pitch_err ** 2)
        cost_pitch = float(self.cfg.w_pitch) * (pitch ** 2)
        cost_pitch_dot = float(self.cfg.w_pitch_dot) * (pitch_dot ** 2)
        cost_goal = float(self.cfg.w_goal) * (goal_err ** 2)
        cost_xpos_dot = float(self.cfg.w_xpos_dot) * (xpos_dot ** 2)
        progress = -float(self.cfg.w_xpos_dot) * xpos_dot * torch.sign(float(self.cfg.goal_x) - xpos)
        cost_u = float(self.cfg.w_u) * (u ** 2)

        cost = cost_goal + cost_u + cost_pitch_dot + cost_pitch

        if uncertainty is not None:
            w_unc = float(getattr(self.cfg, 'w_uncertainty', 50.0))
            cost = cost + w_unc * uncertainty

        return cost



    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [x, vx, pitch, pitch_dot]
        xpos    = states[:, 0]
        xpos_dot   = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]

        # X: (K,5) = [xpos, xpos_dot, pitch, pitch_dot, u]
        X = torch.stack([xpos, xpos_dot, pitch, pitch_dot, actions], dim=-1)

        dxpos_mean, dxpos_var       = self.gp_xpos.predict_torch(X)
        dxpos_dot_mean, dxpos_dot_var = self.gp_xpos_dot.predict_torch(X)
        dpitch_mean, dpitch_var     = self.gp_pitch.predict_torch(X)
        dpitch_dot_mean, dpitch_dot_var = self.gp_pitch_dot.predict_torch(X)

        beta = float(getattr(self.cfg, 'beta_safety', 3.0))

        dpitch_std = torch.sqrt(dpitch_var)
        dpitch_dot_std = torch.sqrt(dpitch_dot_var)

        next_states = torch.empty_like(states)
        next_states[:, 0] = xpos        + dxpos_mean
        next_states[:, 1] = xpos_dot    + dxpos_dot_mean
        next_states[:, 2] = pitch       + dpitch_mean + beta * dpitch_std
        next_states[:, 3] = pitch_dot   + dpitch_dot_mean + beta * dpitch_dot_std

        # reasonable clamps (tune)
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)

        total_std = (torch.sqrt(dxpos_var) + torch.sqrt(dxpos_dot_var) +
                     dpitch_std + dpitch_dot_std)

        return next_states, total_std


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
        delta = U - u_init.unsqueeze(0)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            next_states, step_var = self.gp_step_batch_torch(states, u_t)
            costs = costs + self.stage_cost_torch(states, u_t, uncertainty=step_var)
            states = next_states

        # --- control smoothness penalty ---
        u_diff = U[:, 1:] - U[:, :-1]            # (K, H-1)
        costs = costs + float(cfg.w_du) * (u_diff ** 2).sum(dim=1)

        self.last_mean_cost = float(costs.mean().item())

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / cfg.lambda_)
        wsum = w.sum() + 1e-8

        u_update = (w.unsqueeze(1) * delta).sum(dim=0) / wsum

        u_new = torch.clamp(u_init + u_update, cfg.u_min, cfg.u_max)

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        return float(u_new[0].detach().cpu())