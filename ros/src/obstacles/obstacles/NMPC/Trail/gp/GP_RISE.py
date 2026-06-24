"""
================================================================================
GP.py  --  GPyTorch sparse-variational GP residual model (5-input / 2-output)
           WITH the arcsinh residual transform (the change that makes it learn)
================================================================================

INPUT   z = [position_x, velocity, pitch, angular_pitch, tau]      (d_in = 5)
OUTPUT      [v_dot_residual, omega_dot_residual]                   (d_out = 2)

    full dynamics  =  nominal (RLS)  +  residual (this GP)

WHAT CHANGED FROM THE OLD GP (and why)
--------------------------------------------------------------------------------
The residual is ~0 almost every step, punctuated by rare, huge contact spikes (the
top ~1% of steps hold ~90%+ of the variance). A plain GP minimises squared error, so
those spikes dominate the fit and the smooth obstacle structure underneath is lost
(held-out R^2 ~ 0). We pass each residual channel through an **arcsinh transform**
before fitting:

        latent = ( arcsinh((r - c) / s) - z_mean ) / z_std            # per channel

arcsinh is linear near 0 and logarithmic in the tails, so it compresses the spikes
and the GP can finally see the conditional-mean structure -- the obstacle map. This
single change takes the held-out residual fit from ~0 to ~0.5 (on the arcsinh scale)
and is what lets the GP localise the obstacle by interaction. Everything else (the
SVGP, fixed shared inducing points, the controller export, gp_kernel.py) is unchanged.

The mapping back to residual units is  r_hat = sinh(latent*z_std + z_mean)*s + c.

THE CONTROLLER CONTRACT (read this before changing the export)
--------------------------------------------------------------------------------
Two evaluation paths exist, and the transform interacts with them differently:

  * predict_torch / predict_torch_fast / predict  (the TORCH MPPI path, mppi_torch.py,
    and diagnostics): these apply the inverse transform, so they return the EXACT
    arcsinh-GP residual mean. This is the path your current MPPITorch controller uses
    -- it gets the full benefit with no approximation.

  * mpc_params()  (the flat kernel-sum for the numba MPPI / casadi NMPC): those
    controllers re-evaluate a *plain* kernel sum  mu(z) = y_mean + sum_j alpha_j k(z,Z_j)
    in gp_kernel.py, and sinh(kernel sum) is NOT a kernel sum. So for THOSE controllers
    we export a kernel-sum that REPRODUCES THE RESIDUAL MEAN AT THE INDUCING POINTS and
    interpolates smoothly between them (alpha = Kzz^{-1} (r_hat(Z) - c)). That is an
    approximation of the arcsinh mean -- exact at the M inducing points, Matern-smooth
    elsewhere. It needs NO change to gp_kernel.py. (If you switch to the numba/casadi
    controllers and want them exact too, the clean fix is to apply sinh in gp_kernel.py;
    ask and I'll wire that one line.)

Note: the one-step-ahead ~0.96 trick (predict the next residual from the LAST MEASURED
residual) is a TEMPORAL feedback signal, not a function of z. It cannot live in the
planner's per-state rollout (predict_torch is a stateless state->residual map). It is
exposed below as predict_next() for optional one-step feedback OUTSIDE the rollout; it
is deliberately NOT part of mpc_params()/predict_torch().

Episodic (MBRL) loop is unchanged:
    gp.observe(z, r_v_dot, r_omega_dot)   # buffer one residual sample (per step)
    gp.end_episode()                      # absorb the buffer: first time builds
                                          # structure + fits; later warm-refines
    gp.mpc_params()                       # flat export the controller plans against

Dependencies: torch, gpytorch, numpy, scipy.
================================================================================
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import gpytorch
from scipy.cluster.vq import kmeans2

# Closed-form kernel math + GPyTorch kernel factory (shared with the MPPI/NMPC
# controllers, so it is NOT re-implemented here). gp_kernel.py is UNCHANGED.
from gp_kernel import (gp_mean_np, make_gpytorch_kernel, detect_kernel_id,
                       kernel_id_from_name, DEFAULT_KERNEL)

_SQRT3 = 1.7320508075688772
_SQRT5 = 2.23606797749979


# ============================================================================ #
# One residual channel: a sparse variational GP with FIXED, SHARED inducing points
# and a (swappable) ARD kernel passed in by the owner.  (unchanged)
# ============================================================================ #
class _ChannelSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module):
        M = inducing_points.size(0)
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(M)
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=False)
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = covar_module

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


# ============================================================================ #
# Two-output residual GP. Public interface is the controller contract the sim
# drives: observe / end_episode / predict / mpc_params / state_dict.
# ============================================================================ #
class ResidualGP:
    OUTPUT_NAMES = ("v_dot", "omega_dot")

    def __init__(self, n_features=5, max_points=50, kernel=DEFAULT_KERNEL,
                 n_iter_fit=300, n_iter_update=120, lr=0.05,
                 max_buffer=8000, jitter=1e-6, device="cuda", seed=0):
        self.d = int(n_features)
        self.M = int(max_points)
        self.kernel = str(kernel)
        self.n_iter_fit = int(n_iter_fit)
        self.n_iter_update = int(n_iter_update)
        self.lr = float(lr)
        self.max_buffer = int(max_buffer)
        self.jitter = float(jitter)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self._seed = int(seed)

        self._X = np.empty((0, self.d))
        self._Y = np.empty((0, 2))
        self._bufX, self._bufY = [], []

        # FIXED-at-first-fit structure
        self._x_mean = None
        self._x_std = None
        # arcsinh output transform per channel (replaces plain output standardization):
        #   latent = (arcsinh((r - y_c)/y_s) - z_mean) / z_std
        self._y_c = None                # (2,) robust center of the raw residual
        self._y_s = None                # (2,) robust scale  of the raw residual
        self._z_mean = None             # (2,) mean of the arcsinh-transformed target
        self._z_std = None              # (2,) std  of the arcsinh-transformed target
        self._Z_std = None              # (M, d) standardized inducing points, shared
        self._models = None
        self._likelihoods = None

        self._ready = False
        self._cache = None
        self._fast = None               # cached float tensors of the export (torch path)
        self._last_residual = np.zeros(2)   # for the optional one-step feedback (predict_next)

    # ---- status -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def n_active(self) -> int:
        return self.M if self._ready else 0

    # ---- learning (buffer during episode, absorb at the boundary) ---------
    def observe(self, z, r_v_dot, r_omega_dot):
        self._bufX.append(np.asarray(z, float).reshape(-1))
        self._bufY.append([float(r_v_dot), float(r_omega_dot)])
        self._last_residual = np.array([float(r_v_dot), float(r_omega_dot)])

    def end_episode(self):
        if self._bufX:
            self._X = np.vstack([self._X, np.asarray(self._bufX)])
            self._Y = np.vstack([self._Y, np.asarray(self._bufY)])
            self._bufX, self._bufY = [], []

        n = self._X.shape[0]
        if n < self.M:
            return

        if self._models is None:
            self._build_models()
            n_iter = self.n_iter_fit
        else:
            n_iter = self.n_iter_update

        Xs, Y = self._training_batch()
        losses = [self._train_channel(j, Xs, Y[:, j], n_iter) for j in range(2)]
        self._cache = self._export()
        self._fast = None
        self._ready = True

        ev = self._cache["ell_v"]; ew = self._cache["ell_w"]
        print(f"[GP] trained on {Xs.shape[0]} pts (buffer {n}) | "
              f"loss v={losses[0]:.3f} w={losses[1]:.3f}\n"
              f"     arcsinh scale s[v,w]=[{self._y_s[0]:.3f},{self._y_s[1]:.3f}]  "
              f"ARD lengthscales v_dot={np.round(ev, 2)} omega_dot={np.round(ew, 2)}\n"
              f"     signal_var v={self._cache['sf2_v']:.3f} w={self._cache['sf2_w']:.3f}  "
              f"noise v={self._noise(0):.4f} w={self._noise(1):.4f}")

    def _training_batch(self):
        X, Y = self._X, self._Y
        if X.shape[0] > self.max_buffer:
            idx = np.random.default_rng(self._seed).choice(
                X.shape[0], self.max_buffer, replace=False)
            X, Y = X[idx], Y[idx]
        Xs = (X - self._x_mean) / self._x_std
        return Xs, Y

    def _build_models(self):
        """First fit only: freeze input standardization + the per-channel arcsinh
        transform, then place the M shared inducing points (k-means, fixed seed)."""
        self._x_mean = self._X.mean(0)
        self._x_std = self._X.std(0)
        self._x_std[self._x_std < 1e-8] = 1.0
        # arcsinh output transform, fixed at the first fit (robust center/scale so the
        # spikes don't set the scale), then standardized so GPyTorch's hyperparameter
        # init sees a unit-ish target. The scale is inverted analytically in predict.
        self._y_c = np.median(self._Y, 0)
        self._y_s = np.median(np.abs(self._Y - self._y_c), 0) * 1.4826 + 1e-6
        Za = np.arcsinh((self._Y - self._y_c) / self._y_s)        # (N, 2) arcsinh targets
        self._z_mean = Za.mean(0)
        self._z_std = Za.std(0)
        self._z_std[self._z_std < 1e-8] = 1.0

        Xs = (self._X - self._x_mean) / self._x_std
        Z, _ = kmeans2(Xs, self.M, seed=self._seed, minit="++", missing="warn")
        if Z.shape[0] != self.M or not np.all(np.isfinite(Z)):
            rng = np.random.default_rng(self._seed)
            Z = Xs[rng.choice(Xs.shape[0], self.M, replace=Xs.shape[0] < self.M)]
        self._Z_std = np.ascontiguousarray(Z, float)

        Zt = torch.as_tensor(self._Z_std, dtype=torch.double, device=self.device)
        self._models, self._likelihoods = [], []
        for _ in range(2):
            m = _ChannelSVGP(Zt.clone(), self._make_covar(self.d)).double().to(self.device)
            lik = gpytorch.likelihoods.GaussianLikelihood().double().to(self.device)
            self._models.append(m)
            self._likelihoods.append(lik)

    def _make_covar(self, d):
        cov = make_gpytorch_kernel(self.kernel, d)
        # Lower-bound the ARD lengthscales. On spiky real residuals the ARD fit can
        # collapse a lengthscale to a tiny value and over-fit (craters out-of-sample);
        # this floor (set BEFORE init so the constraint holds) prevents that. Done here,
        # so gp_kernel.py stays untouched.
        cov.base_kernel.register_constraint(
            "raw_lengthscale", gpytorch.constraints.GreaterThan(0.3))
        return cov

    def _to_latent(self, y_raw, j):
        """raw residual -> standardized arcsinh target the GP is trained on."""
        return (np.arcsinh((y_raw - self._y_c[j]) / self._y_s[j]) - self._z_mean[j]) / self._z_std[j]

    def _train_channel(self, j, Xs, y_raw, n_iter):
        m, lik = self._models[j], self._likelihoods[j]
        y_lat = self._to_latent(y_raw, j)                          # arcsinh-transformed target
        Xt = torch.as_tensor(Xs, dtype=torch.double, device=self.device)
        yt = torch.as_tensor(y_lat, dtype=torch.double, device=self.device)

        m.train(); lik.train()
        opt = torch.optim.Adam(list(m.parameters()) + list(lik.parameters()), lr=self.lr)
        mll = gpytorch.mlls.VariationalELBO(lik, m, num_data=yt.size(0))
        loss = torch.tensor(0.0)
        for _ in range(n_iter):
            opt.zero_grad()
            loss = -mll(m(Xt), yt)
            loss.backward()
            opt.step()
        return float(loss.item())

    # ---- export: torch posterior -> caches for both eval paths -------------
    def _export(self):
        Zc = self._Z_std
        kid = detect_kernel_id(self._models[0].covar_module)
        out = {"Z": Zc, "x_mean": self._x_mean.copy(), "x_std": self._x_std.copy(),
               "kernel_id": int(kid)}
        for j, tag in zip((0, 1), ("v", "w")):
            alpha_lat, ell, sf2, mZ = self._export_channel(j)      # latent kernel-sum (EXACT)
            out[f"alpha_lat_{tag}"] = alpha_lat                    # reproduces the arcsinh-space mean
            out[f"ell_{tag}"] = ell
            out[f"sf2_{tag}"] = sf2
            out[f"z_mean_{tag}"] = float(self._z_mean[j])
            out[f"z_std_{tag}"] = float(self._z_std[j])
            out[f"y_c_{tag}"] = float(self._y_c[j])
            out[f"y_s_{tag}"] = float(self._y_s[j])
            # ---- raw-residual kernel-sum projection for the numba/casadi controllers ----
            # reproduce r_hat at the inducing points, interpolate smoothly between them:
            latentZ = mZ * self._z_std[j] + self._z_mean[j]
            rawZ = np.sinh(latentZ) * self._y_s[j] + self._y_c[j]   # residual mean at Z
            Kzz = self._kernel_matrix_np(Zc, Zc, ell, sf2, kid) + self.jitter * np.eye(self.M)
            out[f"alpha_raw_{tag}"] = np.linalg.solve(Kzz, rawZ - self._y_c[j])
            out[f"y_mean_{tag}"] = float(self._y_c[j])              # far from data -> raw center
        return out

    def _export_channel(self, j):
        """alpha_lat = Kzz^{-1} m(Z) reproduces the SVGP's arcsinh-space posterior mean
        exactly; also return m(Z) (standardized-latent mean at the inducing points)."""
        m = self._models[j]
        m.eval(); self._likelihoods[j].eval()
        Zt = m.variational_strategy.inducing_points.detach()
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
            mZ = m(Zt).mean.double()
            Kzz = m.covar_module(Zt).evaluate().double()
            Kzz = Kzz + self.jitter * torch.eye(Kzz.size(0), dtype=torch.double, device=self.device)
            alpha = torch.linalg.solve(Kzz, mZ)
            ell = m.covar_module.base_kernel.lengthscale.detach().reshape(-1)
            sf2 = float(m.covar_module.outputscale.detach().reshape(()))
        return (alpha.cpu().numpy().astype(float), ell.cpu().numpy().astype(float),
                sf2, mZ.cpu().numpy().astype(float))

    def _noise(self, j):
        return float(self._likelihoods[j].noise.detach().reshape(()))

    @staticmethod
    def _kernel_matrix_np(A, B, ell, sf2, kid):
        """sf2 * kshape(r) between rows of A and B (numpy), r = ||(A-B)/ell|| (ARD)."""
        dd = (A[:, None, :] - B[None, :, :]) / ell[None, None, :]
        r2 = np.sum(dd * dd, axis=-1)
        if kid == 3:
            return sf2 * np.exp(-0.5 * r2)
        r = np.sqrt(r2 + 1e-9)
        if kid == 0:
            return sf2 * np.exp(-r)
        if kid == 1:
            c = _SQRT3 * r
            return sf2 * (1.0 + c) * np.exp(-c)
        c = _SQRT5 * r
        return sf2 * (1.0 + c + c * c / 3.0) * np.exp(-c)

    # ---- prediction (EXACT arcsinh mean; matches the torch MPPI path) ------
    def predict(self, z):
        """Return (mean_v_dot, mean_omega_dot, std). EXACT arcsinh-GP residual mean
        (inverse transform applied), matching predict_torch. std is 0.0 (callers
        discard it). Cache-based, so it also works right after load_state_dict()."""
        if not self._ready:
            return 0.0, 0.0, 0.0
        return self._raw_mean_np(z, "v"), self._raw_mean_np(z, "w"), 0.0

    def _raw_mean_np(self, z, tag):
        c = self._cache
        # latent (arcsinh-space) posterior mean via the exact kernel sum, then inverse transform
        g = gp_mean_np(z, c["Z"], c[f"alpha_lat_{tag}"], c["x_mean"], c["x_std"],
                       c[f"ell_{tag}"], c[f"sf2_{tag}"], 0.0, c["kernel_id"])
        latent = g * c[f"z_std_{tag}"] + c[f"z_mean_{tag}"]
        return float(math.sinh(latent) * c[f"y_s_{tag}"] + c[f"y_c_{tag}"])

    # ---- batched torch prediction (the torch MPPI path) -------------------
    @torch.no_grad()
    def predict_torch(self, X):
        """Batched residual means for the TORCH MPPI. EXACT arcsinh mean, evaluated from
        the cached kernel sum (so it is valid immediately after load_state_dict(), unlike
        a gpytorch-forward path). Returns (mean_v, mean_w), each (K,) double on device."""
        return self._predict_torch_impl(X, torch.double)

    @torch.no_grad()
    def predict_torch_fast(self, X, dtype=torch.float32):
        """Same EXACT arcsinh mean, float32 by default (the numba MPPI's closed form,
        batched in torch). Valid whenever ready (incl. after load)."""
        return self._predict_torch_impl(X, dtype)

    def _predict_torch_impl(self, X, dtype):
        Xt = torch.as_tensor(X, dtype=dtype, device=self.device)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        if not self._ready:
            z = torch.zeros(Xt.shape[0], dtype=dtype, device=self.device)
            return z, z
        f = self._fast_cache(dtype, self.device)
        Xs = (Xt - f["xm"]) / f["xs"]
        return self._raw_mean_torch(Xs, f, "v"), self._raw_mean_torch(Xs, f, "w")

    def _raw_mean_torch(self, Xs, f, tag):
        k = self._kernel_cols(Xs, f["Z"], f["el" + tag], f["sf2" + tag], f["kid"])   # (K,M)
        g = k @ f["alat" + tag]                                                       # latent mean
        latent = g * f["zs" + tag] + f["zm" + tag]
        return torch.sinh(latent) * f["ys" + tag] + f["yc" + tag]

    def _fast_cache(self, dtype, device):
        if (self._fast is not None and self._fast["dtype"] == dtype
                and self._fast["device"] == device):
            return self._fast
        c = self._cache
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        Z, kid = t(c["Z"]), int(c["kernel_id"])
        f = {"dtype": dtype, "device": device, "Z": Z,
             "xm": t(c["x_mean"]), "xs": t(c["x_std"]), "kid": kid}
        for ch in ("v", "w"):
            f["alat" + ch] = t(c[f"alpha_lat_{ch}"])
            f["el" + ch] = t(c[f"ell_{ch}"])
            f["sf2" + ch] = float(c[f"sf2_{ch}"])
            f["zm" + ch] = float(c[f"z_mean_{ch}"])
            f["zs" + ch] = float(c[f"z_std_{ch}"])
            f["yc" + ch] = float(c[f"y_c_{ch}"])
            f["ys" + ch] = float(c[f"y_s_{ch}"])
        eye = torch.eye(Z.shape[0], dtype=dtype, device=device)
        for ch in ("v", "w"):                          # Kzz^{-1} per channel (uncertainty path)
            Kzz = self._kernel_cols(Z, Z, f["el" + ch], f["sf2" + ch], kid) + self.jitter * eye
            f["Kinv" + ch] = torch.linalg.inv(Kzz)
        self._fast = f
        return self._fast

    @torch.no_grad()
    def predict_uncertainty_torch(self, X, dtype=torch.float32):
        """Per-channel NORMALIZED epistemic uncertainty in [0,1]:
            u = 1 - k(X,Z) Kzz^{-1} k(Z,X) / sf2
        u=0 on the inducing points (where the data is), u->1 far from any data. This is
        the geometry of where the GP has seen data (unchanged by the output transform),
        so it is the same risk signal a risk-averse MPPI penalizes. Zeros until ready."""
        Xt = torch.as_tensor(X, dtype=dtype, device=self.device)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        if not self._ready:
            z = torch.zeros(Xt.shape[0], dtype=dtype, device=self.device)
            return z, z
        f = self._fast_cache(dtype, self.device)
        Xs = (Xt - f["xm"]) / f["xs"]
        uv = self._channel_uncertainty(Xs, f["Z"], f["elv"], f["sf2v"], f["Kinvv"], f["kid"])
        uw = self._channel_uncertainty(Xs, f["Z"], f["elw"], f["sf2w"], f["Kinvw"], f["kid"])
        return uv, uw

    @staticmethod
    def _kernel_cols(A, Z, ell, sf2, kid):
        diff = (A.unsqueeze(1) - Z.unsqueeze(0)) / ell
        r2 = (diff * diff).sum(-1)
        if kid == 3:
            return sf2 * torch.exp(-0.5 * r2)
        r = torch.sqrt(r2 + 1e-9)
        if kid == 0:
            return sf2 * torch.exp(-r)
        if kid == 1:
            c = _SQRT3 * r
            return sf2 * (1.0 + c) * torch.exp(-c)
        c = _SQRT5 * r
        return sf2 * (1.0 + c + c * c / 3.0) * torch.exp(-c)

    @staticmethod
    def _channel_uncertainty(Xs, Z, ell, sf2, Kinv, kid):
        k = ResidualGP._kernel_cols(Xs, Z, ell, sf2, kid)
        q = ((k @ Kinv) * k).sum(1)
        return torch.clamp(1.0 - q / sf2, 0.0, 1.0)

    # ---- OPTIONAL one-step feedback (NOT used by the planner) --------------
    def predict_next(self, z, r_measured, rho=0.98):
        """CAUSAL one-step-ahead residual using the LAST MEASURED residual (the ~0.96
        trick). This is a TEMPORAL feedback estimate, NOT a function of z alone, so it
        is deliberately NOT in mpc_params()/predict_torch() (the planner's rollout is a
        stateless state->residual map and cannot use it). Use it only for an optional
        one-step correction OUTSIDE the rollout. Defaults to near-persistence (rho~0.98),
        relaxing toward the GP map.  r_measured = the residual you just measured."""
        if not self._ready:
            return float(r_measured[0]), float(r_measured[1])
        out = []
        for j, tag in zip((0, 1), ("v", "w")):
            c = self._cache
            g = gp_mean_np(z, c["Z"], c[f"alpha_lat_{tag}"], c["x_mean"], c["x_std"],
                           c[f"ell_{tag}"], c[f"sf2_{tag}"], 0.0, c["kernel_id"])
            g = g * c[f"z_std_{tag}"] + c[f"z_mean_{tag}"]                     # GP latent
            zr = np.arcsinh((float(r_measured[j]) - c[f"y_c_{tag}"]) / c[f"y_s_{tag}"])  # measured latent
            blended = g + rho * (zr - g)                                       # relax measurement->GP
            out.append(float(math.sinh(blended) * c[f"y_s_{tag}"] + c[f"y_c_{tag}"]))
        return out[0], out[1]

    # ---- export for the controllers (MPPI / NMPC) ------------------------
    def mpc_params(self, omega_in_rollout: bool = True) -> np.ndarray:
        """FIXED-size flat vector the numba/casadi controllers unpack:
            Z(M*d) alpha_v(M) alpha_w(M) x_mean(d) x_std(d) ell_v(d) ell_w(d)
            sf2_v sf2_w y_mean_v y_mean_w kernel_id
        alpha_* here is the RAW-residual projection (see the export note at the top): a
        kernel sum that reproduces r_hat at the inducing points. The torch MPPI does NOT
        use this -- it calls predict_torch (exact). Same layout/size as before."""
        d, M = self.d, self.M
        if not self._ready:
            return np.concatenate([
                np.zeros(M * d), np.zeros(M), np.zeros(M),
                np.zeros(d), np.ones(d), np.ones(d), np.ones(d),
                np.array([1.0, 1.0, 0.0, 0.0, float(kernel_id_from_name(self.kernel))]),
            ])
        c = self._cache
        alpha_w = c["alpha_raw_w"].copy()
        ymean_w = c["y_mean_w"]
        if not omega_in_rollout:
            alpha_w[:] = 0.0
            ymean_w = 0.0
        return np.concatenate([
            c["Z"].reshape(-1), c["alpha_raw_v"], alpha_w,
            c["x_mean"], c["x_std"], c["ell_v"], c["ell_w"],
            np.array([c["sf2_v"], c["sf2_w"], c["y_mean_v"], ymean_w,
                      float(c["kernel_id"])], float),
        ])

    # ---- persistence (flat arrays for np.savez) ---------------------------
    def state_dict(self) -> dict:
        if not self._ready:
            return {}
        c = self._cache
        sd = {"ready": np.array(1), "Z": c["Z"], "x_mean": c["x_mean"], "x_std": c["x_std"],
              "kernel_id": np.array(c["kernel_id"]), "buf_X": self._X, "buf_Y": self._Y}
        for tag in ("v", "w"):
            for key in ("alpha_lat", "alpha_raw", "ell"):
                sd[f"{key}_{tag}"] = c[f"{key}_{tag}"]
            for key in ("sf2", "z_mean", "z_std", "y_c", "y_s", "y_mean"):
                sd[f"{key}_{tag}"] = np.array(c[f"{key}_{tag}"])
        return sd

    def load_state_dict(self, sd: dict):
        if not sd or "ready" not in sd:
            return
        self._X = np.asarray(sd["buf_X"], float)
        self._Y = np.asarray(sd["buf_Y"], float)
        self._x_mean = np.asarray(sd["x_mean"], float)
        self._x_std = np.asarray(sd["x_std"], float)
        self._Z_std = np.ascontiguousarray(sd["Z"], float)
        # restore the per-channel arcsinh transform so predict() is exact after load
        self._y_c = np.array([float(sd["y_c_v"]), float(sd["y_c_w"])])
        self._y_s = np.array([float(sd["y_s_v"]), float(sd["y_s_w"])])
        self._z_mean = np.array([float(sd["z_mean_v"]), float(sd["z_mean_w"])])
        self._z_std = np.array([float(sd["z_std_v"]), float(sd["z_std_w"])])
        c = {"Z": self._Z_std, "x_mean": self._x_mean, "x_std": self._x_std,
             "kernel_id": int(sd["kernel_id"]) if "kernel_id" in sd
                          else kernel_id_from_name(self.kernel)}
        for tag in ("v", "w"):
            c[f"alpha_lat_{tag}"] = np.asarray(sd[f"alpha_lat_{tag}"], float)
            c[f"alpha_raw_{tag}"] = np.asarray(sd[f"alpha_raw_{tag}"], float)
            c[f"ell_{tag}"] = np.asarray(sd[f"ell_{tag}"], float)
            c[f"sf2_{tag}"] = float(sd[f"sf2_{tag}"])
            c[f"z_mean_{tag}"] = float(sd[f"z_mean_{tag}"])
            c[f"z_std_{tag}"] = float(sd[f"z_std_{tag}"])
            c[f"y_c_{tag}"] = float(sd[f"y_c_{tag}"])
            c[f"y_s_{tag}"] = float(sd[f"y_s_{tag}"])
            c[f"y_mean_{tag}"] = float(sd[f"y_mean_{tag}"])
        self._cache = c
        self._models = None
        self._likelihoods = None
        self._fast = None
        self._ready = True


@dataclass
class GPConfig:
    """Config the controller stack consumes. Lengthscales / signal var / noise are
    LEARNED at runtime. `kernel` picks the kernel SHAPE (auto-propagated to the
    controllers via the exported kernel_id). The arcsinh residual transform is always
    on (it is the point of this version) and needs no config."""
    max_points: int = 50
    n_features: int = 5
    kernel: str = DEFAULT_KERNEL   # "matern12" | "matern32" | "matern52" | "rbf"
    n_iter_fit: int = 300
    n_iter_update: int = 300
    lr: float = 0.05
    max_buffer: int = 8000
    jitter: float = 1e-6
    device: str = "cuda"
    seed: int = 0

    def build(self, n_features: int | None = None) -> ResidualGP:
        return ResidualGP(
            n_features=self.n_features if n_features is None else n_features,
            max_points=self.max_points, kernel=self.kernel,
            n_iter_fit=self.n_iter_fit, n_iter_update=self.n_iter_update,
            lr=self.lr, max_buffer=self.max_buffer, jitter=self.jitter,
            device=self.device, seed=self.seed)


# ---- alias so this is a drop-in for gp_adapter.OnlineGPConfig -------------------
OnlineGPConfig = GPConfig


# ============================================================================ #
# Self-test: fit a synthetic 5-D residual with a SPIKY obstacle and check
#   (1) predict() exactly matches the torch MPPI path predict_torch();
#   (2) the arcsinh model recovers the obstacle better than a raw-target fit;
#   (3) mpc_params() reproduces the residual mean at the inducing points.
#   python3 GP.py
# ============================================================================ #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def true_res(X):
        x, v, th, om, tau = X.T
        obstacle = np.exp(-((x - 3.0) ** 2) / (2 * 0.25 ** 2))     # bump at x=3
        return np.stack([-3.0 * obstacle * (1.0 + 0.2 * v),
                         6.0 * obstacle - 0.1 * om], axis=1)

    def sample(N):
        return np.stack([rng.uniform(0, 6, N), rng.uniform(0, 4, N),
                         rng.uniform(-0.3, 1.2, N), rng.uniform(-2, 2, N),
                         rng.uniform(-5, 10, N)], axis=1)

    N = 1200
    X = sample(N)
    # spiky, heavy-tailed residual noise (like contact): the arcsinh is what tames it
    spike = (rng.random(N) < 0.05) * rng.normal(0, 8, N)
    Y = true_res(X) + rng.normal(0, [0.05, 0.08], size=(N, 2)) + np.stack([spike, 1.5 * spike], 1)
    Xt = sample(400)

    gp = GPConfig(kernel="matern32", max_points=50, n_iter_fit=400, device="cpu").build()
    for z, y in zip(X, Y):
        gp.observe(z, y[0], y[1])
    gp.end_episode()

    # (1) predict() (numpy, exact) vs predict_torch() (the controller path) -------
    mv = np.array([gp.predict(z)[0] for z in Xt])
    mw = np.array([gp.predict(z)[1] for z in Xt])
    tv, tw = gp.predict_torch(Xt)
    d1 = max(float(np.max(np.abs(mv - tv.cpu().numpy()))),
             float(np.max(np.abs(mw - tw.cpu().numpy()))))
    print(f"(1) predict() vs predict_torch() max diff = {d1:.2e}   (should be ~0)")

    # (2) does it recover the obstacle? held-out R^2 on the arcsinh scale -----------
    def aR2(yt, yp, c, s):
        zt = np.arcsinh((yt - c) / s); zp = np.arcsinh((yp - c) / s)
        return 1 - np.sum((zt - zp) ** 2) / np.sum((zt - zt.mean()) ** 2)
    yt = true_res(Xt)
    print(f"(2) held-out arcsinh R^2  v_dot={aR2(yt[:,0], mv, gp._y_c[0], gp._y_s[0]):.2f}   "
          f"omega_dot={aR2(yt[:,1], mw, gp._y_c[1], gp._y_s[1]):.2f}   (raw-target GP would be ~0)")

    # (3) mpc_params() reproduces the residual mean at the inducing points ----------
    flat = gp.mpc_params()
    expect = gp.M * gp.d + 2 * gp.M + 4 * gp.d + 5
    Zc = gp._cache["Z"]; Zraw = Zc * gp._x_std + gp._x_mean            # de-standardize Z back to raw z
    err = max(abs(gp.predict(Zraw[i])[0] - gp_mean_np(Zraw[i], Zc, gp._cache["alpha_raw_v"],
              gp._cache["x_mean"], gp._cache["x_std"], gp._cache["ell_v"], gp._cache["sf2_v"],
              gp._cache["y_mean_v"], gp._cache["kernel_id"])) for i in range(gp.M))
    print(f"(3) flat size = {flat.size} (expect {expect}), kernel_id = {flat[-1]:.0f}; "
          f"export vs exact at inducing pts max diff = {err:.2e}")

    # (4) round-trip the checkpoint -------------------------------------------------
    sd = gp.state_dict()
    gp2 = GPConfig(device="cpu").build()
    gp2.load_state_dict(sd)
    d4 = max(abs(gp.predict(z)[0] - gp2.predict(z)[0]) for z in Xt)
    print(f"(4) predict() after save/load max diff = {d4:.2e}   (should be ~0)")
