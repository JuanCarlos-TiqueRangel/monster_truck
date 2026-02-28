# utils/osgpr_trainablez_retrain_manager.py
import os
import time
import threading
import traceback
import numpy as np
import torch
import gpytorch

from gp.svgp_dynamics import SVGPManager


# ------------------------------------------------------------
# Build (X, y) derivative dataset from logged trajectories
# ------------------------------------------------------------
def build_derivative_dataset(up_z, up_z_dot, u, ep, dt: float):
    """
    X_t = [x, vx, up_z, up_z_dot, u] at t
    y = (next - current)/dt  (only within same episode)
    """
    if len(up_z) < 2:
        return None

    same_ep = (ep[:-1] == ep[1:])

    up_z0 = up_z[:-1][same_ep]
    up_z_dot0 = up_z_dot[:-1][same_ep]
    u0    = u[:-1][same_ep]

    up_z1 = up_z[1:][same_ep]
    up_z_dot1 = up_z_dot[1:][same_ep]

    X = np.stack([up_z0, up_z_dot0, u0], axis=1).astype(np.float32)

    #y_dup_z = ((up_z1 - up_z0) / dt).astype(np.float32)
    y_dup_z = up_z_dot0.astype(np.float32)  # instead of finite difference
    y_dup_z_dot = ((up_z_dot1 - up_z_dot0) / dt).astype(np.float32)

    # Drop non-finite
    finite = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y_dup_z)
        & np.isfinite(y_dup_z_dot)
    )
    X = X[finite]
    y_dup_z = y_dup_z[finite]
    y_dup_z_dot = y_dup_z_dot[finite]

    if X.shape[0] == 0:
        return None

    return X, y_dup_z, y_dup_z_dot

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
):
    """
    OSGPR(trainableZ)-style streaming update:
      - trains on NEW chunk only
      - anchor KL on q(u) params (variational mean/chol)
      - trainable inducing points Z with small LR + drift penalty
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

    N = int(Xn.size(0))
    if N < 8:
        return

    # ---- grab variational params robustly (THIS FIXES YOUR ERROR) ----
    vmean_param, vchol_param = _get_variational_param_refs(gp.model)

    # snapshot old q(u)
    m0 = vmean_param.detach().clone()
    L0 = _as_safe_lower_chol(vchol_param.detach(), eps=1e-6)

    # inducing points (trainable Z)
    Z_param = gp.model.variational_strategy.inducing_points
    Z0 = Z_param.detach().clone()

    # freeze hypers if desired (recommended for stability)
    if freeze_hypers:
        _set_module_trainable(gp.model.mean_module, False)
        _set_module_trainable(gp.model.covar_module, False)
        for p in gp.likelihood.parameters():
            p.requires_grad_(False)

    # ensure variational params + Z are trainable
    vmean_param.requires_grad_(True)
    vchol_param.requires_grad_(True)
    Z_param.requires_grad_(True)

    # build optimizer with separate LR for Z
    inducing_id = id(Z_param)
    theta_params = []
    for p in gp.model.parameters():
        if id(p) != inducing_id and p.requires_grad:
            theta_params.append(p)

    lik_params = [p for p in gp.likelihood.parameters() if p.requires_grad]

    param_groups = []
    if len(theta_params) > 0:
        param_groups.append({"params": theta_params, "lr": float(lr_theta)})
    param_groups.append({"params": [Z_param], "lr": float(lr_z)})
    if len(lik_params) > 0:
        param_groups.append({"params": lik_params, "lr": float(lr_theta)})

    optimizer = torch.optim.Adam(param_groups)

    elbo = gpytorch.mlls.VariationalELBO(gp.likelihood, gp.model, num_data=N)

    gp.model.train()
    gp.likelihood.train()

    B = min(int(batch_size), N)

    for _ in range(int(steps)):
        idx = torch.randint(0, N, (B,), device=device)
        xb = Xn[idx]
        yb = yn[idx]

        optimizer.zero_grad(set_to_none=True)
        out = gp.model(xb)
        loss = -elbo(out, yb)

        if float(anchor_beta) > 0.0:
            L = _as_safe_lower_chol(vchol_param, eps=1e-6)
            loss = loss + float(anchor_beta) * _gaussian_kl_cholesky(vmean_param, L, m0, L0)

        if float(z_reg) > 0.0:
            loss = loss + float(z_reg) * torch.mean((Z_param - Z0) ** 2)

        loss.backward()
        optimizer.step()

    gp.model.eval()
    gp.likelihood.eval()



# ------------------------------------------------------------
# Retrain manager integrated with your DatasetBuffer API
# ------------------------------------------------------------
class OSGPRTrainableZRetrainManager:
    def __init__(self, cfg, device, model_lock, logger=None):
        self.cfg = cfg
        self.device = device
        self.model_lock = model_lock
        self.logger = logger

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

        up_z, up_z_dot, u, ep = dataset.snapshot()
        dataset.save_npz(episode_id, up_z, up_z_dot, u, ep)

        # cap to a stable window
        M = int(self.cfg.max_points_for_train)
        up_z, up_z_dot, u, ep = dataset.cap_window(M, up_z, up_z_dot, u, ep)

        # build “new chunk” slice so we train only on new points since last train
        if force:
            start_idx = 0
        else:
            # include one extra sample before the new chunk for derivative pairing
            start_idx = max(0, int(self.last_train_size) - 1)

        up_z_c = up_z[start_idx:]
        up_z_dot_c = up_z_dot[start_idx:]
        u_c    = u[start_idx:]
        ep_c   = ep[start_idx:]

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(up_z_c, up_z_dot_c, u_c, ep_c, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info(
                f"Started OSGPR(trainableZ) update thread | n={n} | last_train_size={self.last_train_size} | start_idx={start_idx}"
            )
        return True

    def _train_worker(self, up_z, up_z_dot, u, ep, n_at_start: int):
        t0 = time.perf_counter()
        dt = float(self.cfg.ctrl_dt)

        built = build_derivative_dataset(up_z, up_z_dot, u, ep, dt)
        if built is None:
            if self.logger:
                self.logger.warn("OSGPR(trainableZ): not enough valid consecutive samples to build derivative dataset.")
            self.training = False
            return

        X, y_dup_z, y_dup_z_dot = built

        paths = [self.cfg.gp_up_z_path, self.cfg.gp_up_z_dot_path]

        try:
            gp_up_z   = SVGPManager.load(paths[0], device=self.device)
            gp_up_z_dot   = SVGPManager.load(paths[1], device=self.device)

            # streaming updates (trainable Z)
            osgpr_trainablez_stream_update(
                gp_up_z, X, y_dup_z,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
            )
            osgpr_trainablez_stream_update(
                gp_up_z_dot, X, y_dup_z_dot,
                steps=int(self.cfg.osgpr_steps),
                batch_size=int(self.cfg.osgpr_batch_size),
                lr_theta=float(self.cfg.osgpr_lr_theta),
                lr_z=float(self.cfg.osgpr_lr_z),
                anchor_beta=float(self.cfg.osgpr_anchor_beta),
                z_reg=float(self.cfg.osgpr_z_reg),
                freeze_hypers=bool(self.cfg.osgpr_freeze_hypers),
            )

            # atomic save
            for gp, out_path in zip([gp_up_z, gp_up_z_dot], paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            # commit last_train_size only after success
            self.last_train_size = int(n_at_start)

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"OSGPR(trainableZ) update finished in {elapsed:.2f}s | N_pairs={X.shape[0]} | steps={self.cfg.osgpr_steps}"
                )

            self.reload_pending = True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.error(f"OSGPR(trainableZ) update failed after {elapsed:.2f}s: {e}")
                self.logger.error(traceback.format_exc())

        finally:
            self.training = False

    def reload_models_if_ready(self):
        if not self.reload_pending or self.training:
            return None

        with self.model_lock:
            gp_up_z   = SVGPManager.load(self.cfg.gp_up_z_path, device=self.device)
            gp_up_z_dot   = SVGPManager.load(self.cfg.gp_up_z_dot_path, device=self.device)

        self.reload_pending = False
        if self.logger:
            self.logger.info("Reloaded SVGP models after OSGPR(trainableZ) update (hot swap).")

        return gp_up_z, gp_up_z_dot
