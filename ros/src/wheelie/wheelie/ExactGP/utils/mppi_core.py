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
        gp_xpos_dot,
        gp_pitch_dot,
        model_lock,
        logger=None,
        feature_map_torch=None,
    ):
        # Initial setup configurations
        self.cfg = cfg
        self.device = device
        self.logger = logger

        # GP managers (hot-swapped by node)
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch_dot = gp_pitch_dot

        self.model_lock = model_lock
        self.gp_target_mode = getattr(cfg, "gp_target_mode", "next").lower().strip()
        if self.gp_target_mode not in ("derivative", "delta", "next"):
            raise ValueError(
                f"Unsupported gp_target_mode='{self.gp_target_mode}'. "
                "Use derivative|delta|next."
            )
        self.gp_input_keys = tuple(getattr(cfg, "gp_input_keys", ()))
        if not self.gp_input_keys:
            raise ValueError("cfg.gp_input_keys must define the GP feature order.")

        # MPPI warm start
        self.plan = None

        self.x_hold = None
        self.x_hold_t = None

        # Expose mean cost for logging/plotting
        self.last_mean_cost = 0.0

        self._custom_feature_map = feature_map_torch


    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Reduced-order GP input for wheelie.

        Full MPPI state:
            states = [xpos, xpos_dot, pitch, pitch_dot]

        GP input order is configured by cfg.gp_input_keys.
        """

        if actions.ndim == 2:
            actions = actions.view(-1)

        state_columns = {
            "xpos": states[:, 0],
            "xpos_dot": states[:, 1],
            "pitch": states[:, 2],
            "pitch_dot": states[:, 3],
        }

        parts = []
        for key in self.gp_input_keys:
            if key == "u":
                parts.append(actions)
            elif key in state_columns:
                parts.append(state_columns[key])
            else:
                raise KeyError(
                    f"Unsupported MPPI GP input key '{key}'. "
                    f"Available keys: {list(state_columns.keys()) + ['u']}"
                )

        return torch.stack(parts, dim=-1)


    def reset_plan(self):
        self.plan = None
        self.x_hold = None
        self.x_hold_t = None


    def set_models(self, gp_xpos_dot, gp_pitch_dot):
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch_dot = gp_pitch_dot


    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # State = [xpos, xpos_dot, pitch, pitch_dot]
        xpos = states[:, 0]
        xpos_dot = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]
        u = actions

        pitch_target = torch.as_tensor(
            float(self.cfg.pitch_target),
            dtype=torch.float32,
            device=states.device,
        )

        cost_xdot = float(self.cfg.w_xdot) * xpos_dot**2

        pitch_target_abs = abs(float(self.cfg.pitch_target))

        pitch_target_pos = torch.as_tensor(pitch_target_abs, dtype=torch.float32, device=states.device)
        pitch_target_neg = torch.as_tensor(-pitch_target_abs, dtype=torch.float32, device=states.device)

        pitch_err_pos = geometry.angdiff_torch(pitch, pitch_target_pos)
        pitch_err_neg = geometry.angdiff_torch(pitch, pitch_target_neg)

        pitch_err_sq = torch.minimum(pitch_err_pos**2, pitch_err_neg**2)

        cost_pitch = float(self.cfg.w_pitch) * pitch_err_sq


        # Balance objective: angular velocity should go to zero
        cost_pitch_dot = float(self.cfg.w_pitch_dot) * pitch_dot**2

        # Control effort
        cost_u = float(self.cfg.w_u) * u**2

        return cost_pitch + cost_u + cost_xdot # cost_pitch_dot + cost_u


    # # COST FUNCTION FOR WHEELIE
    # def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    #     xpos = states[:, 0]
    #     xpos_dot = states[:, 1]
    #     pitch = states[:, 2]
    #     pitch_dot = states[:, 3]
    #     u = actions

    #     # -------------------------------------------------
    #     # Position hold anchor
    #     # This is automatically set at the beginning of the episode.
    #     # -------------------------------------------------
    #     if self.x_hold_t is None:
    #         x_hold = xpos.detach()
    #     else:
    #         x_hold = self.x_hold_t

    #     # 1. Do not move away from starting position
    #     cost_x_hold = float(self.cfg.w_x_hold) * (xpos - x_hold) ** 2

    #     # 2. Keep forward velocity near zero
    #     cost_xdot = float(self.cfg.w_xdot) * xpos_dot ** 2

    #     # 3. Balance angular velocity
    #     cost_pitch_dot = float(self.cfg.w_pitch_dot) * pitch_dot ** 2

    #     # 4. Keep pitch inside wheelie region
    #     pitch_min = float(self.cfg.pitch_min)
    #     pitch_max = float(self.cfg.pitch_max)

    #     under_pitch = torch.relu(pitch_min - pitch)
    #     over_pitch = torch.relu(pitch - pitch_max)

    #     cost_pitch_region = float(self.cfg.w_pitch_region) * (
    #         under_pitch ** 2 + over_pitch ** 2
    #     )

    #     # 5. Control effort
    #     cost_u = float(self.cfg.w_u) * u ** 2

    #     # 6. Hard backward safety
    #     pitch_hard_max = float(self.cfg.pitch_hard_max)
    #     over_hard = torch.relu(pitch - pitch_hard_max)
    #     cost_hard = float(self.cfg.w_pitch_hard) * over_hard ** 2

    #     return (
    #         cost_x_hold
    #         + cost_xdot
    #         + cost_pitch_dot
    #         + cost_pitch_region
    #         + cost_u
    #         + cost_hard
    #     )




    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Reduced-order wheelie GP dynamics.

        State:
            states = [xpos, xpos_dot, pitch, pitch_dot]

        GP predicts:
            xpos_dot_next
            pitch_dot_next

        Then:
            xpos_next  = xpos  + xpos_dot_next  * dt
            pitch_next = pitch + pitch_dot_next * dt
        """

        xpos = states[:, 0]
        xpos_dot = states[:, 1]
        pitch = states[:, 2]
        pitch_dot = states[:, 3]

        dt = float(self.cfg.ctrl_dt)

        # X follows cfg.gp_input_keys.
        X = self.phi(states, actions)

        pred_xpos_dot = self.gp_xpos_dot.predict_torch(X)
        pred_pitch_dot = self.gp_pitch_dot.predict_torch(X)

        if self.gp_target_mode == "next":
            xpos_dot_next = pred_xpos_dot
            pitch_dot_next = pred_pitch_dot
        elif self.gp_target_mode == "delta":
            xpos_dot_next = xpos_dot + pred_xpos_dot
            pitch_dot_next = pitch_dot + pred_pitch_dot
        else:  # derivative
            xpos_dot_next = xpos_dot + pred_xpos_dot * dt
            pitch_dot_next = pitch_dot + pred_pitch_dot * dt

        next_states = torch.empty_like(states)

        # Integrate the states that are not directly predicted by the GP
        next_states[:, 0] = xpos + xpos_dot_next * dt
        next_states[:, 1] = xpos_dot_next

        next_states[:, 2] = pitch + pitch_dot_next * dt
        next_states[:, 3] = pitch_dot_next

        # Safety clamps
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)

        return next_states


    @torch.no_grad()
    def action(self, x0_np):
        cfg = self.cfg
        H, K = cfg.horizon, cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)

        if self.x_hold is None:
            self.x_hold = float(x0[0].detach().cpu())
            self.x_hold_t = torch.tensor(
                self.x_hold,
                dtype=torch.float32,
                device=self.device,
            )

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
            costs = costs + self.stage_cost_torch(states, u_t)
            states = self.gp_step_batch_torch(states, u_t)

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
