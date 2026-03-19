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
    ):
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

    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot):
        self.gp_xpos = gp_xpos
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch = gp_pitch
        self.gp_pitch_dot = gp_pitch_dot


    # COST FUNCTION FOR OBSTACLED

    # def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor, obs_x: torch.Tensor) -> torch.Tensor:
    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: (K,4) = [x, xpos_dot, pitch, pitch_dot]
        xpos     = states[:, 0]
        xpos_dot    = states[:, 1]
        pitch = states[:, 2]
        pitch_dot  = states[:, 3]
        u     = actions

        # ----------------------------
        # 1) progress: push x forward until goal_x
        # (use relu so no penalty after crossing)
        # ----------------------------
        goal_x = float(self.cfg.goal_x)
        dist_to_goal = torch.relu(goal_x - xpos)
        cost_progress = float(self.cfg.w_goal) * (dist_to_goal ** 2)

        # ----------------------------
        # 3) don't go backward + control regularization
        # ----------------------------
        w_back = 150.0
        cost_back = w_back * torch.relu(-xpos_dot) ** 2

        cost_xpos_dot =     float(self.cfg.w_pitch_dot) *   (xpos_dot**2)
        cost_pitch =        float(self.cfg.w_pitch) *       (pitch**2)
        cost_pitch_dot =    float(self.cfg.w_pitch_dot) *   (pitch_dot**2)
        cost_u =            float(self.cfg.w_u) *           (u ** 2)

        # return cost_progress + cost_pitch_lim + cost_pitch_dot + cost_back + cost_u
        return cost_progress + cost_pitch + cost_pitch_dot #+ cost_back
    





    # def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
    #     # states: (K,4) = [x, xpos_dot, pitch, pitch_dot]
    #     xpos    = states[:, 0]
    #     xpos_dot   = states[:, 1]
    #     pitch = states[:, 2]
    #     pitch_dot = states[:, 3]

    #     # X: (K,5) = [x, xpos_dot, pitch, pitch_dot, u]
    #     X = torch.stack([xpos, xpos_dot, pitch, pitch_dot, actions], dim=-1)

    #     with self.model_lock:
    #         if self.cfg.entropy_beta <= 0.0:
    #             dx_mean, _    = self.gp_xpos.predict_torch(X)
    #             dxpos_dot_mean, _   = self.gp_xpos_dot.predict_torch(X)
    #             dp_mean, _    = self.gp_pitch.predict_torch(X)
    #             dr_mean, _    = self.gp_pitch_dot.predict_torch(X)
    #             dx_var = dxpos_dot_var = dp_var = dr_var = None
    #         else:
    #             dx_mean,  dx_var  = self.gp_xpos.predict_torch(X)
    #             dxpos_dot_mean, dxpos_dot_var = self.gp_xpos_dot.predict_torch(X)
    #             dp_mean,  dp_var  = self.gp_pitch.predict_torch(X)
    #             dr_mean,  dr_var  = self.gp_pitch_dot.predict_torch(X)

    #     dt = float(self.cfg.ctrl_dt)

    #     next_states = torch.empty_like(states)
    #     next_states[:, 0] = xpos     + dx_mean  * dt
    #     next_states[:, 1] = xpos_dot    + dxpos_dot_mean * dt
    #     next_states[:, 2] = pitch + dp_mean  * dt
    #     next_states[:, 3] = pitch_dot  + dr_mean  * dt

    #     # reasonable clamps (tune)
    #     next_states[:, 2].clamp_(-math.pi, math.pi)
    #     next_states[:, 3].clamp_(-20.0, 20.0)

    #     if self.cfg.entropy_beta <= 0.0:
    #         return next_states, None

    #     # entropy (optional)
    #     dx_var  = torch.clamp(dx_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
    #     dxpos_dot_var = torch.clamp(dxpos_dot_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
    #     dp_var  = torch.clamp(dp_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
    #     dr_var  = torch.clamp(dr_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)

    #     if self.cfg.entropy_dt_scale:
    #         var_next = torch.stack([dx_var, dxpos_dot_var, dp_var, dr_var], dim=-1) * (dt * dt)
    #     else:
    #         var_next = torch.stack([dx_var, dxpos_dot_var, dp_var, dr_var], dim=-1)

    #     if self.cfg.entropy_use_log:
    #         entropy = 0.5 * torch.log(var_next).sum(dim=-1)
    #     else:
    #         entropy = var_next.sum(dim=-1)

    #     entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    #     return next_states, entropy





    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [x, xpos_dot, pitch, pitch_dot]
        xpos    = states[:, 0]
        xpos_dot   = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]

        # X: (K,5) = [x, xpos_dot, pitch, pitch_dot, u]
        X = torch.stack([xpos, xpos_dot, pitch, pitch_dot, actions], dim=-1)

        dx_mean = self.gp_xpos.predict_mean_torch(X)
        dxpos_dot_mean  = self.gp_xpos_dot.predict_mean_torch(X)
        dp_mean  = self.gp_pitch.predict_mean_torch(X)
        dr_mean = self.gp_pitch_dot.predict_mean_torch(X)

        dt = float(self.cfg.ctrl_dt)

        next_states = torch.empty_like(states)
        next_states[:, 0] = xpos     + dx_mean  * dt
        next_states[:, 1] = xpos_dot    + dxpos_dot_mean * dt
        next_states[:, 2] = pitch + dp_mean  * dt
        next_states[:, 3] = pitch_dot  + dr_mean  * dt

        # reasonable clamps (tune)
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)


        return next_states



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

        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t)
            states = self.gp_step_batch_torch(states, u_t)

            if not torch.isfinite(states).all():
                if self.logger is not None:
                    self.logger.error("Rollout produced non-finite states (GP output likely NaN/Inf).")
                break

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






    # @torch.no_grad()
    # def action(self, x0_np, obs_pos_x):
    #     cfg = self.cfg
    #     H, K = cfg.horizon, cfg.num_rollouts

    #     x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
    #     if x0.shape != (4,):
    #         raise ValueError(f"x0 must be shape (4,), got {tuple(x0.shape)}")

    #     u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan
    #     eps = torch.randn(K, H, device=self.device) * cfg.sigma
    #     U = torch.clamp(u_init.unsqueeze(0) + eps, cfg.u_min, cfg.u_max)
    #     delta = U - u_init.unsqueeze(0)          # <-- important

    #     states = x0.unsqueeze(0).repeat(K, 1)
    #     costs = torch.zeros(K, dtype=torch.float32, device=self.device)

    #     beta = float(cfg.entropy_beta)

    #     # obs_x for rollout gating
    #     # if obs_pos_x is None:
    #     #     obs_x0 = torch.full((K,), 1e6, dtype=torch.float32, device=self.device)
    #     # else:
    #     #     obs_x0 = torch.full((K,), float(obs_pos_x), dtype=torch.float32, device=self.device)

    #     for t in range(H):
    #         u_t = U[:, t]
    #         #stage = self.stage_cost_torch(states, u_t, obs_x0)
    #         #states, ent = self.gp_step_batch_torch(states, u_t)
    #         stage = self.stage_cost_torch(states, u_t)
    #         states = self.gp_step_batch_torch(states, u_t)

    #         if not torch.isfinite(states).all():
    #             if self.logger is not None:
    #                 self.logger.error("Rollout produced non-finite states (GP output likely NaN/Inf).")
    #             break

    #         # if ent is not None:
    #         #     stage = stage - beta * ent
    #         costs = costs + stage

    #     self.last_mean_cost = float(costs.mean().item())

    #     J_min = costs.min()
    #     w = torch.exp(-(costs - J_min) / cfg.lambda_)
    #     wsum = w.sum() + 1e-8

    #     #du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
    #     du = (w.unsqueeze(1) * delta).sum(dim=0) / wsum

    #     u_new = torch.clamp(u_init + du, cfg.u_min, cfg.u_max)

    #     self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
    #     return float(u_new[0].detach().cpu())
