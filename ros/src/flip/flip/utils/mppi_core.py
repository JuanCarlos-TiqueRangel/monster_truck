# utils/mppi_core.py
import torch


class MPPICore:
    """
    Fast flip-only MPPI core.

    State: [up, up_dot]
    Input to GP: [up, up_dot, action]
    """

    def __init__(self, cfg, device, gp_up_z, gp_up_z_dot, model_lock, logger=None):
        self.cfg = cfg
        self.device = device
        self.logger = logger

        self.gp_up_z = gp_up_z
        self.gp_up_z_dot = gp_up_z_dot
        self.model_lock = model_lock

        self.plan = None
        self.last_mean_cost = None  # avoid hot-path sync every step

        # reusable buffers
        self._buf_K = None
        self._buf_H = None
        self._X_buf = None
        self._next_states_buf = None

    def reset_plan(self):
        self.plan = None

    def set_models(self, gp_up_z, gp_up_z_dot):
        self.gp_up_z = gp_up_z
        self.gp_up_z_dot = gp_up_z_dot

    def _ensure_buffers(self, K: int):
        if self._buf_K == K and self._X_buf is not None:
            return

        self._buf_K = K
        self._X_buf = torch.empty((K, 3), dtype=torch.float32, device=self.device)
        self._next_states_buf = torch.empty((K, 2), dtype=torch.float32, device=self.device)

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        up = states[:, 0]
        up_dot = states[:, 1]
        a = actions

        cost_up = float(self.cfg.w_up_z) * (1.0 - up) ** 2
        cost_updot = float(self.cfg.w_up_z_dot) * (up_dot ** 2)
        cost_act = float(self.cfg.w_u) * (a ** 2)

        return cost_up #+ cost_updot #+ cost_act

    # def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor, gp_mean_fn, dt: float):
    #     """
    #     Uses reusable buffers and mean-only GP.
    #     """
    #     X = self._X_buf
    #     X[:, 0].copy_(states[:, 0])
    #     X[:, 1].copy_(states[:, 1])
    #     X[:, 2].copy_(actions)

    #     dud_mean = gp_mean_fn(X)

    #     next_states = self._next_states_buf
    #     next_states[:, 0].copy_(states[:, 0]).add_(states[:, 1], alpha=dt)   # up + updot*dt
    #     next_states[:, 1].copy_(states[:, 1]).add_(dud_mean, alpha=dt)        # updot + dud_mean*dt

    #     return next_states



    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor,
                            gp_mean_fn_up, gp_mean_fn_updot):
        X = self._X_buf
        X[:, 0].copy_(states[:, 0])
        X[:, 1].copy_(states[:, 1])
        X[:, 2].copy_(actions)

        d_up = gp_mean_fn_up(X)
        d_updot = gp_mean_fn_updot(X)

        next_states = self._next_states_buf
        next_states[:, 0].copy_(states[:, 0]).add_(d_up)
        next_states[:, 1].copy_(states[:, 1]).add_(d_updot)

        # optional safety clamp
        next_states[:, 0].clamp_(-1.2, 1.2)

        return next_states



    @torch.inference_mode()
    def action(self, x0_np):
        cfg = self.cfg
        H = int(cfg.horizon)
        K = int(cfg.num_rollouts)
        dt = float(cfg.ctrl_dt)

        self._ensure_buffers(K)

        # Snapshot GP reference once.
        # This is much cheaper than holding the lock for the whole rollout.
        with self.model_lock:
            gp_mean_fn_up = self.gp_up_z.predict_mean_torch
            gp_mean_fn_updot = self.gp_up_z_dot.predict_mean_torch

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        if x0.shape != (2,):
            raise ValueError(f"x0 must be shape (2,), got {tuple(x0.shape)}")

        if self.plan is None or self.plan.shape[0] != H:
            u_init = torch.zeros(H, dtype=torch.float32, device=self.device)
        else:
            u_init = self.plan

        # Sample controls
        eps = torch.randn((K, H), dtype=torch.float32, device=self.device)
        eps.mul_(float(cfg.sigma))

        U = u_init.unsqueeze(0) + eps
        U.clamp_(float(cfg.u_min), float(cfg.u_max))

        delta = U - u_init.unsqueeze(0)

        # Initial rollout states
        states = torch.empty((K, 2), dtype=torch.float32, device=self.device)
        states[:, 0].fill_(x0[0].item())
        states[:, 1].fill_(x0[1].item())

        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            a_t = U[:, t]
            costs.add_(self.stage_cost_torch(states, a_t))
            states = self.gp_step_batch_torch(states, a_t, gp_mean_fn_up, gp_mean_fn_updot)

        # Optional: only compute for debugging outside critical path if needed
        # self.last_mean_cost = float(costs.mean().item())

        J_min = torch.min(costs)
        w = torch.exp(-(costs - J_min) / float(cfg.lambda_))
        wsum = torch.sum(w) + 1e-8

        du = torch.sum(w.unsqueeze(1) * delta, dim=0) / wsum
        u_new = torch.clamp(u_init + du, float(cfg.u_min), float(cfg.u_max))

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0)

        # One CPU sync only here
        return float(u_new[0])