# utils/mppi_core.py
import math
import torch

class MPPICore:
    """
    Flip-only MPPI core.

    State: (K,4) = [x, vx, up, up_dot]
      up     = uprightness = R[2,2] in [-1,1]
      up_dot = time derivative of up (computed from gyro)

    Input feature to GPs:
      X = [x, vx, up, up_dot, a]
    """

    def __init__(self, cfg, device, gp_up_z, gp_up_z_dot, model_lock, logger=None):
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.gp_up_z      = gp_up_z   # now models d(up)/dt
        self.gp_up_z_dot  = gp_up_z_dot   # now models d(up_dot)/dt

        self.model_lock = model_lock

        self.plan = None
        self.last_mean_cost = 0.0

    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_up_z, gp_up_z_dot):
        self.gp_up_z      = gp_up_z   # now models d(up)/dt
        self.gp_up_z_dot  = gp_up_z_dot   # now models d(up_dot)/dt

    # -----------------------------
    # Conservative flip cost
    # -----------------------------
    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # states: (K,4) = [up, up_dot]
        up     = states[:, 0]
        up_dot = states[:, 1]
        a      = actions

        # Conservative: just "be upright" + "don't spin too much" + "don't use huge control"
        cost_up     = float(self.cfg.w_up_z)     * (1.0 - up) ** 2
        cost_updot  = float(self.cfg.w_up_z_dot)  * (up_dot ** 2)
        cost_act    = float(self.cfg.w_u)    * (a ** 2)

        return cost_up + cost_updot + cost_act

    # -----------------------------
    # GP rollout dynamics
    # -----------------------------
    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [up, up_dot]
        up    = states[:, 0]
        updot = states[:, 1]

        X = torch.stack([up, updot, actions], dim=-1)  # (K,2)

        # with self.model_lock:
        #     dud_mean, _ = self.gp_up_z_dot.predict_torch(X)

        # dt = float(self.cfg.ctrl_dt)
        # next_states = torch.empty_like(states)
        # next_states[:, 0] = up    + updot * dt
        # next_states[:, 1] = updot + dud_mean * dt
        # return next_states, None

        with self.model_lock:
            dup_mean, _   = self.gp_up_z.predict_torch(X)   # d(up)/dt
            dud_mean, _   = self.gp_up_z_dot.predict_torch(X)   # d(up_dot)/dt

        dt = float(self.cfg.ctrl_dt)

        next_states = torch.empty_like(states)
        next_states[:, 0] = up    + dup_mean * dt
        next_states[:, 1] = updot + dud_mean * dt

        return next_states, None

    @torch.no_grad()
    def action(self, x0_np):
        cfg = self.cfg
        H, K = int(cfg.horizon), int(cfg.num_rollouts)

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        if x0.shape != (2,):
            raise ValueError(f"x0 must be shape (2,), got {tuple(x0.shape)}")

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        eps = torch.randn(K, H, device=self.device) * float(cfg.sigma)
        U = torch.clamp(u_init.unsqueeze(0) + eps, float(cfg.u_min), float(cfg.u_max))
        delta = U - u_init.unsqueeze(0)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs  = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            a_t = U[:, t]
            stage = self.stage_cost_torch(states, a_t)
            states, _ = self.gp_step_batch_torch(states, a_t)

            if not torch.isfinite(states).all():
                if self.logger:
                    self.logger.error("Rollout produced non-finite states (GP output NaN/Inf).")
                break

            costs = costs + stage

        self.last_mean_cost = float(costs.mean().item())

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / float(cfg.lambda_))
        wsum = w.sum() + 1e-8

        du = (w.unsqueeze(1) * delta).sum(dim=0) / wsum
        u_new = torch.clamp(u_init + du, float(cfg.u_min), float(cfg.u_max))

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        return float(u_new[0].detach().cpu())