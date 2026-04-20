# utils/mppi_core.py
import math
import torch

from utils import geometry
from utils.palsgp_selector import PALSGPSelectorConfig, select_local_indices_nearest
from utils.palsgp_local_model import LocalModelConfig, build_local_head_from_global

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

        # PALSGP-lite local model cache
        self.selector_cfg = PALSGPSelectorConfig(
            local_num_inducing=int(getattr(cfg, "local_num_inducing", 48)),
            anchor_num_inducing=int(getattr(cfg, "anchor_num_inducing", 12)),
            selector_mode=str(getattr(cfg, "palsgp_local_selector", "nearest")),
        )
        self.local_model_cfg = LocalModelConfig(
            eps_loc=float(getattr(cfg, "eps_loc", 1e-5)),
            cholesky_float64=bool(getattr(cfg, "cholesky_float64", True)),
            build_variance=bool(getattr(cfg, "local_use_uncertainty", False)),
        )

        self.local_heads = None
        self.anchor_idx_by_head = {
            "xpos": None,
            "xpos_dot": None,
            "pitch": None,
            "pitch_dot": None,
        }

        self.anchor_idx = None
        self.local_dirty = True
        self.local_rebuild_counter = 0




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



    def _predict_global_delta_from_features(self, Z_feat: torch.Tensor) -> torch.Tensor:
        d0 = self.gp_xpos.predict_mean_torch(Z_feat)
        d1 = self.gp_xpos_dot.predict_mean_torch(Z_feat)
        d2 = self.gp_pitch.predict_mean_torch(Z_feat)
        d3 = self.gp_pitch_dot.predict_mean_torch(Z_feat)
        return torch.stack([d0, d1, d2, d3], dim=-1)

    def _predict_local_delta_from_features(self, Z_feat: torch.Tensor) -> torch.Tensor:
        if self.local_heads is None:
            return self._predict_global_delta_from_features(Z_feat)

        d0 = self.local_heads["xpos"].predict_mean(Z_feat)
        d1 = self.local_heads["xpos_dot"].predict_mean(Z_feat)
        d2 = self.local_heads["pitch"].predict_mean(Z_feat)
        d3 = self.local_heads["pitch_dot"].predict_mean(Z_feat)
        return torch.stack([d0, d1, d2, d3], dim=-1)


    @torch.no_grad()
    def _build_nominal_rollout_feature_cloud(self, x0: torch.Tensor, u_init: torch.Tensor) -> torch.Tensor:
        """
        Build one nominal rollout cloud in UNNORMALIZED feature space.
        Selection will normalize per head later.
        """
        H = int(self.cfg.horizon)
        s = x0.view(1, -1).clone()
        feats = []

        for t in range(H):
            u_t = u_init[t].view(1)
            z_t = self.feature_map_torch(s, u_t)
            feats.append(z_t.squeeze(0))

            # use global delta model to roll nominal cloud
            ds = self._predict_global_delta_from_features(z_t)
            s = s + ds

            s[:, 2].clamp_(-math.pi, math.pi)
            s[:, 3].clamp_(-20.0, 20.0)

        return torch.stack(feats, dim=0)


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



    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Rollout update: s^+ = s + GP(s,u). Uses local heads if available.
        """
        Z_feat = self.feature_map_torch(states, actions)
        if self.local_heads:
            ds = torch.stack([
                self.local_heads["xpos"].predict_mean(Z_feat),
                self.local_heads["xpos_dot"].predict_mean(Z_feat),
                self.local_heads["pitch"].predict_mean(Z_feat),
                self.local_heads["pitch_dot"].predict_mean(Z_feat)
            ], dim=-1)
        else:
            ds = torch.stack([
                self.gp_xpos.predict_mean_torch(Z_feat),
                self.gp_xpos_dot.predict_mean_torch(Z_feat),
                self.gp_pitch.predict_mean_torch(Z_feat),
                self.gp_pitch_dot.predict_mean_torch(Z_feat)
            ], dim=-1)
        next_states = states + ds
        next_states[:,2].clamp_(-math.pi, math.pi)
        next_states[:,3].clamp_(-20.0, 20.0)
        return next_states




    def gp_step_batch_torch_global(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        GP step using GLOBAL SVGP heads (existing behavior).
        """
        if actions.ndim == 2:
            actions = actions.view(-1)

        X = self.phi(states, actions)

        dx = self.gp_xpos.predict_mean_torch(X)
        dv = self.gp_xpos_dot.predict_mean_torch(X)
        dp = self.gp_pitch.predict_mean_torch(X)
        dr = self.gp_pitch_dot.predict_mean_torch(X)

        ds = torch.stack([dx, dv, dp, dr], dim=-1)

        mode = str(getattr(self.cfg, "gp_target_mode", "derivative")).lower()
        if mode == "derivative":
            next_states = states + ds * float(self.cfg.ctrl_dt)
        else:
            next_states = states + ds

        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)
        return next_states



    @torch.no_grad()
    def maybe_rebuild_local_models(self, x0: torch.Tensor, u_init: torch.Tensor) -> None:
        """
        Rebuild PALSGP-lite local GP heads if enabled.
        """
        if not bool(getattr(self.cfg, "palsgp_use_local", True)):
            self.local_heads = None
            return

        rollout_features = self._build_nominal_rollout_feature_cloud(x0, u_init)

        # Hold model_lock only during extraction of inducing points/posterior
        with self.model_lock:
            heads = {"xpos": self.gp_xpos, "xpos_dot": self.gp_xpos_dot,
                    "pitch": self.gp_pitch, "pitch_dot": self.gp_pitch_dot}
            local_heads = {}

            for name, gp_head in heads.items():
                # Normalize rollout cloud for this head
                Z_glob_n = gp_head.get_inducing_points_normalized()
                rollout_n = (rollout_features - gp_head.X_mean) / gp_head.X_std

                # Select local indices for THIS head
                anchor_idx = self.anchor_idx_by_head.get(name, None)
                idx_loc = select_local_indices_nearest(
                    Z_glob=Z_glob_n,
                    rollout_features=rollout_n,
                    config=self.selector_cfg,
                    anchor_idx=anchor_idx
                )
                # Save anchor subset for next time
                M_anc = min(self.selector_cfg.anchor_num_inducing, idx_loc.numel())
                self.anchor_idx_by_head[name] = idx_loc[:M_anc].clone()

                # Build local head for this GP
                local_heads[name] = build_local_head_from_global(
                    gp_head, idx_loc, self.local_model_cfg
                )
            self.local_heads = local_heads





    @torch.no_grad()
    def action(self, x0_np, obs_pos_x=None):
        cfg = self.cfg
        H, K = cfg.horizon, cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        if x0.shape != (4,):
            raise ValueError(f"x0 must be shape (4,), got {tuple(x0.shape)}")

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        # PALSGP-lite: rebuild local heads for this solve (if enabled)
        if bool(getattr(self.cfg, "palsgp_use_local", True)):
            try:
                self.maybe_rebuild_local_models(x0, u_init)
            except Exception as e:
                # fallback: disable local heads for this solve
                self.local_heads = None
                self.local_dirty = True
                if self.logger is not None:
                    self.logger.warn(f"Local PALSGP rebuild failed; falling back to global SVGP for this step. err={e}")

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