# utils/osgpr_trainablez_retrain_manager.py
import os
import time
import threading
import traceback
import numpy as np
import torch
import gpytorch
import inspect

from gp.svgp_dynamics import SVGPManager


# ------------------------------------------------------------
# Build (X, y) derivative dataset from logged trajectories
# ------------------------------------------------------------
def build_derivative_dataset(pitch, pitch_dot, xpos, xpos_dot, u, ep, dt: float):
    """
    X_t = [xpos, xpos_dot, pitch, pitch_dot, u] at t
    y = (next - current)/dt  (only within same episode)
    """
    if len(pitch) < 2:
        return None

    same_ep = (ep[:-1] == ep[1:])

    xpos0    = xpos[:-1][same_ep]
    xpos_dot0   = xpos_dot[:-1][same_ep]
    pitch0 = pitch[:-1][same_ep]
    pitch_dot0 = pitch_dot[:-1][same_ep]
    u0    = u[:-1][same_ep]

    xpos1    = xpos[1:][same_ep]
    xpos_dot1   = xpos_dot[1:][same_ep]
    pitch1 = pitch[1:][same_ep]
    pitch_dot1 = pitch_dot[1:][same_ep]

    X = np.stack([xpos0, xpos_dot0, pitch0, pitch_dot0, u0], axis=1).astype(np.float32)

    y_dxpos    = ((xpos1    - xpos0)    / dt).astype(np.float32)
    y_dxpos_dot   = ((xpos_dot1   - xpos_dot0)   / dt).astype(np.float32)
    y_dpitch = ((pitch1 - pitch0) / dt).astype(np.float32)
    y_dpitch_dot = ((pitch_dot1 - pitch_dot0) / dt).astype(np.float32)

    # Drop non-finite
    finite = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y_dxpos)
        & np.isfinite(y_dxpos_dot)
        & np.isfinite(y_dpitch)
        & np.isfinite(y_dpitch_dot)
    )
    X = X[finite]
    y_dxpos = y_dxpos[finite]
    y_dxpos_dot = y_dxpos_dot[finite]
    y_dpitch = y_dpitch[finite]
    y_dpitch_dot = y_dpitch_dot[finite]

    if X.shape[0] == 0:
        return None

    return X, y_dxpos, y_dxpos_dot, y_dpitch, y_dpitch_dot



@staticmethod
def build_transition_delta_dataset(pitch, pitch_dot, xpos, xpos_dot, u, ep):
    """
    Build same-episode transition pairs (s_t, u_t, s_{t+1}) for Δs targets.
    """
    if len(pitch) < 2:
        return None
    same_ep = (ep[:-1] == ep[1:])
    if not np.any(same_ep):
        return None

    s0 = np.stack([
        xpos[:-1][same_ep],
        xpos_dot[:-1][same_ep],
        pitch[:-1][same_ep],
        pitch_dot[:-1][same_ep]
    ], axis=1).astype(np.float32)
    s1 = np.stack([
        xpos[1:][same_ep],
        xpos_dot[1:][same_ep],
        pitch[1:][same_ep],
        pitch_dot[1:][same_ep]
    ], axis=1).astype(np.float32)
    u0 = u[:-1][same_ep].reshape(-1,1).astype(np.float32)

    X = np.concatenate([s0, u0], axis=1)
    Y = (s1 - s0).astype(np.float32)  # delta-target
    return X, Y




def build_training_batch_from_transitions(
    s0: np.ndarray, u0: np.ndarray, s1: np.ndarray, dt: float, target_mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert transitions to (X, Y) for GP training.

    Args:
        s0: [N,4]
        u0: [N,1]
        s1: [N,4]
        dt: control dt
        target_mode: "delta" or "derivative"

    Returns:
        X: [N, dx] (feature inputs)
        Y: [N, 4]  (delta or derivative targets)
    """
    if s0.shape[0] == 0:
        raise ValueError("Empty transitions.")

    d = s1 - s0
    if target_mode == "delta":
        Y = d
    elif target_mode == "derivative":
        Y = d / float(dt)
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    return s0, Y  # feature map applied later (needs cfg-provided phi)


# -----------------------------
# Robust access to variational params
# -----------------------------
def _get_variational_param_refs(model):
    """
    Returns references (not copies) to:
      - variational_mean param (M,)
      - chol_variational_covar param (M,M)

    Works across gpytorch versions.
    """
    vs = model.variational_strategy

    # Best case: internal module exists
    vd_mod = getattr(vs, "_variational_distribution", None)
    if vd_mod is not None and hasattr(vd_mod, "variational_mean") and hasattr(vd_mod, "chol_variational_covar"):
        return vd_mod.variational_mean, vd_mod.chol_variational_covar

    # Next: sometimes it's directly stored as a module
    vd_mod = getattr(vs, "variational_distribution", None)
    if isinstance(vd_mod, torch.nn.Module) and hasattr(vd_mod, "variational_mean") and hasattr(vd_mod, "chol_variational_covar"):
        return vd_mod.variational_mean, vd_mod.chol_variational_covar

    # Fallback: find by name in parameters
    vmean = None
    vchol = None
    for name, p in model.named_parameters():
        if "variational_distribution" in name and name.endswith("variational_mean"):
            vmean = p
        elif "variational_distribution" in name and name.endswith("chol_variational_covar"):
            vchol = p

    if vmean is None or vchol is None:
        raise RuntimeError(
            "Could not locate variational params (variational_mean / chol_variational_covar) in model.\n"
            "Tip: print(list(model.state_dict().keys())) and look for 'variational_mean' / 'chol_variational_covar'."
        )

    return vmean, vchol


def _as_safe_lower_chol(L: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Make a safe lower-tri matrix with positive diagonal (for KL computations).
    """
    L = torch.tril(L)
    diag = torch.diagonal(L)
    diag_safe = torch.clamp(diag.abs(), min=eps)
    # in-place diagonal replace (keeps gradients)
    L = L.clone()
    L.diagonal().copy_(diag_safe)
    return L


def _gaussian_kl_cholesky(m: torch.Tensor, L: torch.Tensor, m0: torch.Tensor, L0: torch.Tensor) -> torch.Tensor:
    """
    KL( N(m,S) || N(m0,S0) ) with S = L L^T, S0 = L0 L0^T.
    Uses lower-tri solve. Assumes L and L0 are safe lower choleskies.
    """
    m = m.view(-1)
    m0 = m0.view(-1)
    M = m.numel()

    # log dets
    logdet_S  = 2.0 * torch.log(torch.diagonal(L)).sum()
    logdet_S0 = 2.0 * torch.log(torch.diagonal(L0)).sum()

    # tr(S0^{-1} S): A = L0^{-1} L  => tr = ||A||_F^2
    A = torch.linalg.solve_triangular(L0, L, upper=False)
    tr_term = (A * A).sum()

    # (m-m0)^T S0^{-1} (m-m0)
    dm = (m - m0).unsqueeze(-1)  # (M,1)
    v = torch.linalg.solve_triangular(L0, dm, upper=False)
    quad_term = (v * v).sum()

    return 0.5 * (tr_term + quad_term - M + (logdet_S0 - logdet_S))


def _set_module_trainable(module: torch.nn.Module, trainable: bool):
    for p in module.parameters():
        p.requires_grad_(trainable)


# -----------------------------
# FIXED streaming update (trainable Z)
# -----------------------------
def osgpr_trainablez_stream_update(
    gp,
    X_new,
    y_new,
    steps,
    batch_size,
    lr_theta,
    lr_z,
    anchor_beta,
    z_reg,
    freeze_hypers,
    train_inducing_locations=False,
    grad_clip_max_norm=5.0,
):
    """
    PALSGP-lite compatible update:
      - trains on NEW chunk only
      - delta-target scalar head
      - inducing locations fixed by default
      - finite filtering + grad clipping
    """
    if gp.model is None or gp.likelihood is None:
        raise RuntimeError("SVGPManager missing model/likelihood.")

    if gp.X_mean is None or gp.X_std is None or gp.Y_mean is None or gp.Y_std is None:
        raise RuntimeError("Normalization stats missing in SVGPManager.")

    device = gp.device

    Xt = torch.tensor(X_new, dtype=torch.float32, device=device)
    yt = torch.tensor(y_new, dtype=torch.float32, device=device).flatten()

    Xn = (Xt - gp.X_mean) / gp.X_std
    yn = (yt - gp.Y_mean) / gp.Y_std

    finite = torch.isfinite(Xn).all(dim=1) & torch.isfinite(yn)
    Xn = Xn[finite]
    yn = yn[finite]

    if int(Xn.shape[0]) < 8:
        return

    vmean_param, vchol_param = _get_variational_param_refs(gp.model)

    m0 = vmean_param.detach().clone()
    L0 = _as_safe_lower_chol(vchol_param.detach(), eps=1e-6)

    Z_param = gp.model.variational_strategy.inducing_points
    Z0 = Z_param.detach().clone()

    if freeze_hypers:
        _set_module_trainable(gp.model.mean_module, False)
        _set_module_trainable(gp.model.covar_module, False)
        for p in gp.likelihood.parameters():
            p.requires_grad_(False)

    vmean_param.requires_grad_(True)
    vchol_param.requires_grad_(True)
    Z_param.requires_grad_(bool(train_inducing_locations))

    inducing_id = id(Z_param)
    theta_params = []
    for p in gp.model.parameters():
        if id(p) != inducing_id and p.requires_grad:
            theta_params.append(p)

    lik_params = [p for p in gp.likelihood.parameters() if p.requires_grad]

    param_groups = []
    if len(theta_params) > 0:
        param_groups.append({"params": theta_params, "lr": float(lr_theta)})
    if bool(train_inducing_locations):
        param_groups.append({"params": [Z_param], "lr": float(lr_z)})
    if len(lik_params) > 0:
        param_groups.append({"params": lik_params, "lr": float(lr_theta)})

    optimizer = torch.optim.Adam(param_groups)
    elbo = gpytorch.mlls.VariationalELBO(gp.likelihood, gp.model, num_data=int(Xn.shape[0]))

    gp.model.train()
    gp.likelihood.train()

    B = min(int(batch_size), int(Xn.shape[0]))

    for _ in range(int(steps)):
        idx = torch.randint(0, int(Xn.shape[0]), (B,), device=device)
        xb = Xn[idx]
        yb = yn[idx]

        optimizer.zero_grad(set_to_none=True)
        out = gp.model(xb)
        loss = -elbo(out, yb)

        if float(anchor_beta) > 0.0:
            L = _as_safe_lower_chol(vchol_param, eps=1e-6)
            loss = loss + float(anchor_beta) * _gaussian_kl_cholesky(vmean_param, L, m0, L0)

        if bool(train_inducing_locations) and float(z_reg) > 0.0:
            loss = loss + float(z_reg) * torch.mean((Z_param - Z0) ** 2)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in OSGPR update")

        loss.backward()

        trainable_params = []
        for g in param_groups:
            trainable_params.extend(g["params"])
        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=float(grad_clip_max_norm),
            error_if_nonfinite=True,
        )

        optimizer.step()

        for _, p in gp.model.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                raise RuntimeError("Non-finite model parameter after optimizer step")

    gp.model.eval()
    gp.likelihood.eval()



# ------------------------------------------------------------
# Retrain manager integrated with your DatasetBuffer API
# ------------------------------------------------------------
class OSGPRTrainableZRetrainManager:
    # def __init__(self, cfg, device, model_lock, logger=None):
    def __init__(self, cfg, device, model_lock, logger=None, feature_map_torch=None):
        self.cfg = cfg
        self.device = device
        self.model_lock = model_lock
        self.logger = logger
        self.feature_map_torch = feature_map_torch

        self.training = False
        self.reload_pending = False
        self.train_thread = None

        # last_train_size commits only after successful save
        self.last_train_size = 0

    def maybe_start_retrain_async(self, dataset, episode_id: int, force: bool = False) -> bool:
        if self.training:
            if self.logger:
                self.logger.info("OSGPR(trainableZ) update requested but training is already running; skipping.")
            return False

        n = int(dataset.n_points())

        if not force:
            if n < int(self.cfg.min_points_to_train):
                if self.logger:
                    self.logger.info(f"Not enough data to retrain yet: {n} < {self.cfg.min_points_to_train}")
                return False

            if (n - int(self.last_train_size)) < int(self.cfg.min_new_points_between_trains):
                if self.logger:
                    self.logger.info("Not enough new data since last train; skipping.")
                return False

        pitch, pitch_dot, xpos, xpos_dot, u, ep = dataset.snapshot()
        dataset.save_npz(episode_id, pitch, pitch_dot, xpos, xpos_dot, u, ep)

        # cap to a stable window
        M = int(self.cfg.max_points_for_train)
        pitch, pitch_dot, xpos, xpos_dot, u, ep = dataset.cap_window(M, pitch, pitch_dot, xpos, xpos_dot, u, ep)

        # build “new chunk” slice so we train only on new points since last train
        if force:
            start_idx = 0
        else:
            # include one extra sample before the new chunk for derivative pairing
            start_idx = max(0, int(self.last_train_size) - 1)

        pitch_c = pitch[start_idx:]
        pitch_dot_c = pitch_dot[start_idx:]
        xpos_c    = xpos[start_idx:]
        xpos_dot_c   = xpos_dot[start_idx:]
        u_c    = u[start_idx:]
        ep_c   = ep[start_idx:]

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(pitch_c, pitch_dot_c, xpos_c, xpos_dot_c, u_c, ep_c, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info(
                f"Started OSGPR(trainableZ) update thread | n={n} | last_train_size={self.last_train_size} | start_idx={start_idx}"
            )
        return True


    @staticmethod
    def _build_transition_training_batch_from_arrays(
        pitch,
        pitch_dot,
        xpos,
        xpos_dot,
        u,
        ep,
        dt: float,
        target_mode: str = "delta",
    ):
        """
        Build training data from consecutive same-episode transitions.

        Returns
        -------
        s0 : np.ndarray [N, 4]
            state at time t   = [x, xdot, pitch, pitch_dot]
        u0 : np.ndarray [N, 1]
            action at time t
        Y  : np.ndarray [N, 4]
            target = delta-state or derivative target
        """
        pitch = np.asarray(pitch, dtype=np.float32).reshape(-1)
        pitch_dot = np.asarray(pitch_dot, dtype=np.float32).reshape(-1)
        xpos = np.asarray(xpos, dtype=np.float32).reshape(-1)
        xpos_dot = np.asarray(xpos_dot, dtype=np.float32).reshape(-1)
        u = np.asarray(u, dtype=np.float32).reshape(-1)
        ep = np.asarray(ep).reshape(-1)

        n = len(pitch)
        if not (len(pitch_dot) == len(xpos) == len(xpos_dot) == len(u) == len(ep) == n):
            raise ValueError("Input arrays do not all have the same length.")

        if n < 2:
            return None

        same_ep = (ep[:-1] == ep[1:])
        if not np.any(same_ep):
            return None

        s0 = np.stack(
            [
                xpos[:-1][same_ep],
                xpos_dot[:-1][same_ep],
                pitch[:-1][same_ep],
                pitch_dot[:-1][same_ep],
            ],
            axis=1,
        ).astype(np.float32)

        s1 = np.stack(
            [
                xpos[1:][same_ep],
                xpos_dot[1:][same_ep],
                pitch[1:][same_ep],
                pitch_dot[1:][same_ep],
            ],
            axis=1,
        ).astype(np.float32)

        u0 = u[:-1][same_ep].reshape(-1, 1).astype(np.float32)

        dstate = (s1 - s0).astype(np.float32)

        target_mode = str(target_mode).lower()
        if target_mode == "delta":
            Y = dstate
        elif target_mode == "derivative":
            Y = dstate / max(float(dt), 1e-8)
        else:
            raise ValueError(f"Unknown target_mode={target_mode}")

        return s0, u0, Y

    @staticmethod
    def _gp_all_finite(gp) -> bool:
        if gp is None or gp.model is None:
            return False

        for _, p in gp.model.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                return False

        if getattr(gp, "likelihood", None) is not None:
            for _, p in gp.likelihood.named_parameters():
                if p is not None and not torch.isfinite(p).all():
                    return False

        for attr in ("X_mean", "X_std", "Y_mean", "Y_std"):
            v = getattr(gp, attr, None)
            if v is not None and torch.is_tensor(v) and not torch.isfinite(v).all():
                return False

        return True


    def _train_worker(self, pitch, pitch_dot, xpos, xpos_dot, u, ep, n_at_start: int):
        t0 = time.perf_counter()

        built = build_transition_delta_dataset(pitch, pitch_dot, xpos, xpos_dot, u, ep)
        if built is None:
            if self.logger:
                self.logger.warn("OSGPR: not enough valid consecutive samples to build transition-delta dataset.")
            self.training = False
            return

        X, Y = built
        y_dxpos = Y[:, 0]
        y_dxpos_dot = Y[:, 1]
        y_dpitch = Y[:, 2]
        y_dpitch_dot = Y[:, 3]

        paths = [
            self.cfg.gp_xpos_path,
            self.cfg.gp_xpos_dot_path,
            self.cfg.gp_pitch_path,
            self.cfg.gp_pitch_dot_path,
        ]

        try:
            gp_xpos = SVGPManager.load(paths[0], device=self.device)
            gp_xpos_dot = SVGPManager.load(paths[1], device=self.device)
            gp_pitch = SVGPManager.load(paths[2], device=self.device)
            gp_pitch_dot = SVGPManager.load(paths[3], device=self.device)

            train_Z_online = bool(getattr(self.cfg, "online_train_inducing", False))

            osgpr_trainablez_stream_update(
                gp_xpos, X, y_dxpos,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
                train_inducing_locations=train_Z_online,
            )
            osgpr_trainablez_stream_update(
                gp_xpos_dot, X, y_dxpos_dot,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
                train_inducing_locations=train_Z_online,
            )
            osgpr_trainablez_stream_update(
                gp_pitch, X, y_dpitch,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
                train_inducing_locations=train_Z_online,
            )
            osgpr_trainablez_stream_update(
                gp_pitch_dot, X, y_dpitch_dot,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
                train_inducing_locations=train_Z_online,
            )

            for gp, out_path in zip([gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot], paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            self.last_train_size = int(n_at_start)
            self.reload_pending = True

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"OSGPR delta update finished in {elapsed:.2f}s | "
                    f"N_pairs={X.shape[0]} | train_Z_online={train_Z_online}"
                )

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.error(f"OSGPR delta update failed after {elapsed:.2f}s: {e}")
                self.logger.error(traceback.format_exc())

        finally:
            self.training = False


    def reload_models_if_ready(self):
        if not self.reload_pending or self.training:
            return None

        with self.model_lock:
            gp_pitch   = SVGPManager.load(self.cfg.gp_pitch_path, device=self.device)
            gp_pitch_dot   = SVGPManager.load(self.cfg.gp_pitch_dot_path, device=self.device)
            gp_xpos = SVGPManager.load(self.cfg.gp_xpos_path, device=self.device)
            gp_xpos_dot     = SVGPManager.load(self.cfg.gp_xpos_dot_path, device=self.device)

        self.reload_pending = False
        if self.logger:
            self.logger.info("Reloaded SVGP models after OSGPR(trainableZ) update (hot swap).")

        return gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot
