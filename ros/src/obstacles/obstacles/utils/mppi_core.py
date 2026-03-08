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
        gp_pose_x,
        gp_vx,
        gp_flip,
        gp_rate,
        model_lock,
        logger=None,
    ):
        self.cfg = cfg
        self.device = device
        self.logger = logger

        # GP managers (hot-swapped by node)
        self.gp_pose_x = gp_pose_x
        self.gp_vx = gp_vx
        self.gp_flip = gp_flip
        self.gp_rate = gp_rate

        self.model_lock = model_lock

        self.pitch_target_t = torch.tensor(
            float(cfg.pitch_target), dtype=torch.float32, device=self.device
        )

        # MPPI warm start
        self.plan = None

        # Expose mean cost for logging/plotting
        self.last_mean_cost = 0.0

    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_pose_x, gp_vx, gp_flip, gp_rate):
        self.gp_pose_x = gp_pose_x
        self.gp_vx = gp_vx
        self.gp_flip = gp_flip
        self.gp_rate = gp_rate


    # COST FUNCTION FOR OBSTACLED

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor, obs_x: torch.Tensor) -> torch.Tensor:
        # states: (K,4) = [x, vx, pitch, rate]
        x     = states[:, 0]
        vx    = states[:, 1]
        pitch = states[:, 2]
        rate  = states[:, 3]
        u     = actions

        # ----------------------------
        # 1) progress: push x forward until goal_x
        # (use relu so no penalty after crossing)
        # ----------------------------
        goal_x = float(self.cfg.goal_x)
        dist_to_goal = torch.relu(goal_x - x)
        cost_progress = float(self.cfg.w_goal) * (dist_to_goal ** 2)

        # ----------------------------
        # 2) safety: pitch limit (prevents flips)
        # ----------------------------
        pitch_lim = float(self.cfg.pitch_limit)
        cost_pitch_lim = float(self.cfg.w_pitch_limit) * torch.relu(torch.abs(pitch) - pitch_lim) ** 2

        # (optional) penalize high pitch rate
        cost_rate = float(self.cfg.w_rate) * (rate ** 2)

        # ----------------------------
        # 3) don't go backward + control regularization
        # ----------------------------
        w_back = 150.0
        cost_back = w_back * torch.relu(-vx) ** 2

        cost_u = float(self.cfg.w_u) * (u ** 2)

        # return cost_progress + cost_pitch_lim + cost_rate + cost_back + cost_u
        return cost_progress #+ cost_u + cost_rate #+ cost_back
    





    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [x, vx, pitch, rate]
        x    = states[:, 0]
        vx   = states[:, 1]
        pitch= states[:, 2]
        rate = states[:, 3]

        # X: (K,5) = [x, vx, pitch, rate, u]
        X = torch.stack([x, vx, pitch, rate, actions], dim=-1)

        with self.model_lock:
            if self.cfg.entropy_beta <= 0.0:
                dx_mean, _    = self.gp_pose_x.predict_torch(X)
                dvx_mean, _   = self.gp_vx.predict_torch(X)
                dp_mean, _    = self.gp_flip.predict_torch(X)
                dr_mean, _    = self.gp_rate.predict_torch(X)
                dx_var = dvx_var = dp_var = dr_var = None
            else:
                dx_mean,  dx_var  = self.gp_pose_x.predict_torch(X)
                dvx_mean, dvx_var = self.gp_vx.predict_torch(X)
                dp_mean,  dp_var  = self.gp_flip.predict_torch(X)
                dr_mean,  dr_var  = self.gp_rate.predict_torch(X)

        dt = float(self.cfg.ctrl_dt)

        next_states = torch.empty_like(states)
        next_states[:, 0] = x     + dx_mean  * dt
        next_states[:, 1] = vx    + dvx_mean * dt
        next_states[:, 2] = pitch + dp_mean  * dt
        next_states[:, 3] = rate  + dr_mean  * dt

        # reasonable clamps (tune)
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)

        if self.cfg.entropy_beta <= 0.0:
            return next_states, None

        # entropy (optional)
        dx_var  = torch.clamp(dx_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dvx_var = torch.clamp(dvx_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dp_var  = torch.clamp(dp_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dr_var  = torch.clamp(dr_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)

        if self.cfg.entropy_dt_scale:
            var_next = torch.stack([dx_var, dvx_var, dp_var, dr_var], dim=-1) * (dt * dt)
        else:
            var_next = torch.stack([dx_var, dvx_var, dp_var, dr_var], dim=-1)

        if self.cfg.entropy_use_log:
            entropy = 0.5 * torch.log(var_next).sum(dim=-1)
        else:
            entropy = var_next.sum(dim=-1)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return next_states, entropy

    @torch.no_grad()
    def action(self, x0_np, obs_pos_x):
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

        beta = float(cfg.entropy_beta)

        # obs_x for rollout gating
        if obs_pos_x is None:
            obs_x0 = torch.full((K,), 1e6, dtype=torch.float32, device=self.device)
        else:
            obs_x0 = torch.full((K,), float(obs_pos_x), dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t, obs_x0)
            states, ent = self.gp_step_batch_torch(states, u_t)

            if not torch.isfinite(states).all():
                if self.logger is not None:
                    self.logger.error("Rollout produced non-finite states (GP output likely NaN/Inf).")
                break

            if ent is not None:
                stage = stage - beta * ent
            costs = costs + stage

        self.last_mean_cost = float(costs.mean().item())

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / cfg.lambda_)
        wsum = w.sum() + 1e-8

        #du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
        du = (w.unsqueeze(1) * delta).sum(dim=0) / wsum

        u_new = torch.clamp(u_init + du, cfg.u_min, cfg.u_max)

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        return float(u_new[0].detach().cpu())
