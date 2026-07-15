
from dataclasses import dataclass

import numpy as np
import torch
import gpytorch
from scipy.cluster.vq import kmeans2

# The closed-form kernel math lives in gp_kernel (shared with the MPPI/NMPC controllers,
# so it is NOT re-implemented here). predict() uses the numpy version; the GPyTorch
# kernel is built by the factory and its id auto-detected for the controller export.
from gp_kernel import (gp_mean_np, make_gpytorch_kernel, detect_kernel_id,
                       kernel_id_from_name, DEFAULT_KERNEL)

_SQRT3 = 1.7320508075688772
_SQRT5 = 2.23606797749979


# ============================================================================ #
# One residual channel: a sparse variational GP with FIXED, SHARED inducing points
# and a (swappable) ARD kernel passed in by the owner.
# ============================================================================ #
class _ChannelSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module):
        # inducing_points: (M, d) in STANDARDIZED input space. Fixed (not learned)
        # so the two channels keep IDENTICAL Z (the controller wants a single Z).
        M = inducing_points.size(0)
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(M)
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=False)
        super().__init__(var_strat)
        # ZeroMean: we CENTER the targets ourselves (y_mean is exported separately),
        # so far from data the residual reverts to the mean residual
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = covar_module               # built by ResidualGP._make_covar


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
        self.kernel = str(kernel)                    # kernel NAME (see gp_kernel.KERNEL_IDS)
        self.n_iter_fit = int(n_iter_fit)
        self.n_iter_update = int(n_iter_update)
        self.lr = float(lr)
        self.max_buffer = int(max_buffer)
        self.jitter = float(jitter)
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self._seed = int(seed)

        # accumulated training data (raw, across episodes) + this episode's buffer
        self._X = np.empty((0, self.d))
        self._Y = np.empty((0, 2))
        self._bufX, self._bufY = [], []

        # FIXED-at-first-fit structure (so the torch posterior stays valid + warm)
        self._x_mean = None
        self._x_std = None
        self._y_mean = None             # (2,) output centering  (exported as y_mean)
        self._y_scale = None            # (2,) output scaling -> folded into alpha
        self._Z_std = None              # (M, d) standardized inducing points, shared
        self._models = None             # [v_dot model, omega_dot model]
        self._likelihoods = None

        # exported cache (single source of truth for mpc_params + predict)
        self._ready = False
        self._cache = None              # dict of numpy arrays (see _export)
        self._xm_t = self._xs_t = None  # cached standardization tensors (predict_torch)
        self._fast = None               # cached float tensors of the export (predict_torch_fast)

    # ---- status -----------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def n_active(self) -> int:
        return self.M if self._ready else 0

    # ---- learning (buffer during episode, absorb at the boundary) ---------
    def observe(self, z, r_v_dot, r_omega_dot):
        """Buffer one residual sample z=[x,v,theta,omega,tau] -> (r_v_dot, r_omega_dot).
        NOT applied mid-episode: end_episode() absorbs the whole batch so the model the
        controller plans against is frozen within an episode."""
        self._bufX.append(np.asarray(z, float).reshape(-1))
        self._bufY.append([float(r_v_dot), float(r_omega_dot)])

    def end_episode(self):
        """Absorb the episode buffer into the accumulated dataset, then (re)train.
        FIRST time (once we have >= M samples): fix standardization + inducing points
        and build the two SVGPs. Every time: keep optimising on the buffer (warm)."""
        if self._bufX:
            self._X = np.vstack([self._X, np.asarray(self._bufX)])
            self._Y = np.vstack([self._Y, np.asarray(self._bufY)])
            self._bufX, self._bufY = [], []

        n = self._X.shape[0]
        if n < self.M:
            return                                  # too few to place M inducing pts

        if self._models is None:
            self._build_models()                    # first fit: fix structure + stats
            n_iter = self.n_iter_fit
        else:
            n_iter = self.n_iter_update

        Xs, Y = self._training_batch()
        losses = [self._train_channel(j, Xs, Y[:, j], n_iter) for j in range(2)]
        self._cache = self._export()
        self._fast = None                          # invalidate predict_torch_fast cache
        self._ready = True

        ev = self._cache["ell_v"]; ew = self._cache["ell_w"]
        print(f"[GP] trained on {Xs.shape[0]} pts (buffer {n}) | "
              f"loss v={losses[0]:.3f} w={losses[1]:.3f}\n"
              f"     ARD lengthscales [x,v,theta,omega,tau]  "
              f"v_dot={np.round(ev, 2)}  omega_dot={np.round(ew, 2)}\n"
              f"     signal_var v={self._cache['sf2_v']:.3f} w={self._cache['sf2_w']:.3f}  "
              f"noise v={self._noise(0):.4f} w={self._noise(1):.4f}")

    def _training_batch(self):
        """Standardized inputs + RAW outputs for the current accumulated data,
        subsampled to max_buffer (keeps full-track coverage, bounds train cost).
        Output standardization is applied per-channel inside _train_channel."""
        X, Y = self._X, self._Y
        if X.shape[0] > self.max_buffer:
            idx = np.random.default_rng(self._seed).choice(
                X.shape[0], self.max_buffer, replace=False)
            X, Y = X[idx], Y[idx]
        Xs = (X - self._x_mean) / self._x_std
        return Xs, Y

    def _build_models(self):
        """First fit only: freeze input AND output standardization, then place the M
        shared inducing points (k-means on standardized inputs, fixed seed ->
        deterministic + identical across both channels)."""
        self._x_mean = self._X.mean(0)
        self._x_std = self._X.std(0)
        self._x_std[self._x_std < 1e-8] = 1.0
        # Standardize outputs so GPyTorch's default hyperparameter init (unit-ish
        # scale) is appropriate; the scale is folded back into alpha at export.
        self._y_mean = self._Y.mean(0)
        self._y_scale = self._Y.std(0)
        self._y_scale[self._y_scale < 1e-8] = 1.0
        self._xm_t = self._xs_t = None              # invalidate predict_torch cache
        Xs = (self._X - self._x_mean) / self._x_std

        Z, _ = kmeans2(Xs, self.M, seed=self._seed, minit="++", missing="warn")
        # k-means may collapse a cluster on degenerate data; fall back to random rows.
        if Z.shape[0] != self.M or not np.all(np.isfinite(Z)):
            rng = np.random.default_rng(self._seed)
            Z = Xs[rng.choice(Xs.shape[0], self.M, replace=Xs.shape[0] < self.M)]
        self._Z_std = np.ascontiguousarray(Z, float)

        Zt = torch.as_tensor(self._Z_std, dtype=torch.double, device=self.device)
        self._models, self._likelihoods = [], []
        for _ in range(2):
            # one fresh covar module per channel (own hyperparameters), same kernel type.
            m = _ChannelSVGP(Zt.clone(), self._make_covar(self.d)).double().to(self.device)
            lik = gpytorch.likelihoods.GaussianLikelihood().double().to(self.device)
            self._models.append(m)
            self._likelihoods.append(lik)

    def _make_covar(self, d):
        """Build the GPyTorch covariance (ScaleKernel(<kernel>)). Override this in a
        subclass to hard-wire a kernel inline (see GP_gptorch.py); by default it uses
        the `kernel` name via the gp_kernel factory."""
        return make_gpytorch_kernel(self.kernel, d)

    def _train_channel(self, j, Xs, y_raw, n_iter):
        m, lik = self._models[j], self._likelihoods[j]
        y_std = (y_raw - self._y_mean[j]) / self._y_scale[j]      # standardized target
        Xt = torch.as_tensor(Xs, dtype=torch.double, device=self.device)
        yt = torch.as_tensor(y_std, dtype=torch.double, device=self.device)

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

    # ---- export: torch posterior -> the controller's flat Matern representation -
    def _export(self):
        Zc = self._Z_std                                  # (M, d) shared inducing pts
        # auto-detect the kernel id from the trained model -> the controllers use the
        # SAME kernel as whatever GPyTorch kernel was built (the "edit one line" story).
        kid = detect_kernel_id(self._models[0].covar_module)
        out = {"Z": Zc, "x_mean": self._x_mean.copy(), "x_std": self._x_std.copy(),
               "kernel_id": int(kid)}
        for j, tag in zip((0, 1), ("v", "w")):
            alpha, ell, sf2 = self._export_channel(j)
            # Fold the output scale back in: original = y_mean + y_scale * m_std(z),
            # and m_std(z) = sum alpha_std_j k(z,Z_j), so alpha = y_scale * alpha_std.
            out[f"alpha_{tag}"] = self._y_scale[j] * alpha
            out[f"ell_{tag}"] = ell
            out[f"sf2_{tag}"] = sf2                        # signal var in std space (alpha absorbs y_scale)
            out[f"y_mean_{tag}"] = float(self._y_mean[j])
        return out

    def _export_channel(self, j):
        """alpha_std = Kzz^{-1} m(Z): query GPyTorch's OWN posterior mean at the inducing
        points (standardized-output space) and solve one MxM system. With ZeroMean this
        reproduces the SVGP posterior mean exactly via the controller's kernel sum."""
        m = self._models[j]
        m.eval(); self._likelihoods[j].eval()
        Zt = m.variational_strategy.inducing_points.detach()
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
            mZ = m(Zt).mean.double()                       # (M,)
            Kzz = m.covar_module(Zt).evaluate().double()   # (M, M) = sf2 * matern(Z,Z)
            Kzz = Kzz + self.jitter * torch.eye(Kzz.size(0), dtype=torch.double, device=self.device)
            alpha = torch.linalg.solve(Kzz, mZ)
            ell = m.covar_module.base_kernel.lengthscale.detach().reshape(-1)   # (d,)
            sf2 = float(m.covar_module.outputscale.detach().reshape(()))
        return (alpha.cpu().numpy().astype(float),
                ell.cpu().numpy().astype(float), sf2)

    def _noise(self, j):
        return float(self._likelihoods[j].noise.detach().reshape(()))

    # ---- prediction (diagnostics) ----------------------------------------
    def predict(self, z):
        """Return (mean_v_dot, mean_omega_dot, std). Evaluated from the exported cache
        (same math as the controller), so it matches the rollout exactly. std is 0.0:
        the controllers ignore variance and every caller here discards it."""
        if not self._ready:
            return 0.0, 0.0, 0.0
        c = self._cache
        kid = c["kernel_id"]
        mv = gp_mean_np(z, c["Z"], c["alpha_v"], c["x_mean"], c["x_std"],
                        c["ell_v"], c["sf2_v"], c["y_mean_v"], kid)
        mw = gp_mean_np(z, c["Z"], c["alpha_w"], c["x_mean"], c["x_std"],
                        c["ell_w"], c["sf2_w"], c["y_mean_w"], kid)
        return mv, mw, 0.0

    # ---- batched torch prediction (for a torch MPPI, mppi_core.py style) --
    @torch.no_grad()
    def predict_torch(self, X):
        """Batched residual means for a TORCH MPPI: the controller hands the GP a batch
        of raw features X=(K,d) [x,v,theta,omega,tau] and the GP calls its trained
        GPyTorch model DIRECTLY -- no flat-vector export/unpack (that path is only for
        the numba MPPI / casadi NMPC, which can't call torch). Returns (mean_v, mean_w),
        each (K,) double on self.device.

        Zeros until the first fit. NOTE: after load_state_dict() the torch models are not
        rebuilt until the next end_episode(), so predict_torch is zero until then (the
        numba/casadi path uses the loaded cache and needs no retrain)."""
        Xt = torch.as_tensor(X, dtype=torch.double, device=self.device)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        if not self._ready or self._models is None:
            z = torch.zeros(Xt.shape[0], dtype=torch.double, device=self.device)
            return z, z
        if self._xm_t is None:
            self._xm_t = torch.as_tensor(self._x_mean, dtype=torch.double, device=self.device)
            self._xs_t = torch.as_tensor(self._x_std, dtype=torch.double, device=self.device)
        Xs = (Xt - self._xm_t) / self._xs_t
        self._models[0].eval(); self._models[1].eval()
        mv = self._models[0](Xs).mean * self._y_scale[0] + self._y_mean[0]
        mw = self._models[1](Xs).mean * self._y_scale[1] + self._y_mean[1]
        return mv, mw

    def _fast_cache(self, dtype, device):
        """Cache the exported params as torch tensors (+ per-channel Kzz^{-1}, used only by
        the uncertainty path), so the torch predict methods are pure matmuls with no per-call
        allocation. Rebuilt when the GP is re-fit (self._fast set to None). Kzz^{-1} is
        recomputed here from the cache, so this also works after load_state_dict()."""
        if (self._fast is not None and self._fast["dtype"] == dtype
                and self._fast["device"] == device):
            return self._fast
        c = self._cache
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        Z, kid = t(c["Z"]), int(c["kernel_id"])
        f = {
            "dtype": dtype, "device": device,
            "Z": Z, "xm": t(c["x_mean"]), "xs": t(c["x_std"]),
            "av": t(c["alpha_v"]), "aw": t(c["alpha_w"]),
            "elv": t(c["ell_v"]), "elw": t(c["ell_w"]),
            "sf2v": float(c["sf2_v"]), "sf2w": float(c["sf2_w"]),
            "ymv": float(c["y_mean_v"]), "ymw": float(c["y_mean_w"]),
            "kid": kid,
        }
        eye = torch.eye(Z.shape[0], dtype=dtype, device=device)
        for ch in ("v", "w"):                          # Kzz^{-1} per channel (Kzz among Z)
            Kzz = self._kernel_cols(Z, Z, f["el" + ch], f["sf2" + ch], kid) + self.jitter * eye
            f["Kinv" + ch] = torch.linalg.inv(Kzz)
        self._fast = f
        return self._fast

    def _prep(self, X, dtype):
        """Shared input handling for the torch predict paths -> (Xt(K,d), ready?)."""
        Xt = torch.as_tensor(X, dtype=dtype, device=self.device)
        if Xt.ndim == 1:
            Xt = Xt.reshape(1, -1)
        return Xt, self._ready                          # cache-based: valid even after load

    @torch.no_grad()
    def predict_torch_fast(self, X, dtype=torch.float32):
        """FAST batched residual MEAN for a torch MPPI: mean(X) = k(X,Z) @ alpha + y_mean --
        the SAME closed form the numba MPPI uses, batched in torch (float32 by default). No
        gpytorch forward, no FP64. Works whenever ready (incl. after load). X:(K,d) raw."""
        Xt, ready = self._prep(X, dtype)
        if not ready:
            z = torch.zeros(Xt.shape[0], dtype=dtype, device=self.device)
            return z, z
        f = self._fast_cache(dtype, self.device)
        Xs = (Xt - f["xm"]) / f["xs"]                                  # (K,d) standardized
        kv = self._kernel_cols(Xs, f["Z"], f["elv"], f["sf2v"], f["kid"])   # (K,M)
        kw = self._kernel_cols(Xs, f["Z"], f["elw"], f["sf2w"], f["kid"])
        return f["ymv"] + kv @ f["av"], f["ymw"] + kw @ f["aw"]

    @torch.no_grad()
    def predict_uncertainty_torch(self, X, dtype=torch.float32):
        """Per-channel NORMALIZED epistemic uncertainty in [0,1] at X:(K,d):
            u = 1 - k(X,Z) Kzz^{-1} k(Z,X) / sf2
        u=0 right on the data (inducing points), u->1 far from any data (reverts to prior).
        This is the risk signal a risk-averse MPPI penalizes so it won't plan into UNMODELLED
        regions. Cost is O(K * M^2) -- use a modest max_points (~50) when you enable it.
        Returns zeros until ready (incl. right after load, until end_episode runs)."""
        Xt, ready = self._prep(X, dtype)
        if not ready:
            z = torch.zeros(Xt.shape[0], dtype=dtype, device=self.device)
            return z, z
        f = self._fast_cache(dtype, self.device)
        Xs = (Xt - f["xm"]) / f["xs"]
        uv = self._channel_uncertainty(Xs, f["Z"], f["elv"], f["sf2v"], f["Kinvv"], f["kid"])
        uw = self._channel_uncertainty(Xs, f["Z"], f["elw"], f["sf2w"], f["Kinvw"], f["kid"])
        return uv, uw

    @staticmethod
    def _kernel_cols(A, Z, ell, sf2, kid):
        """Kernel matrix sf2 * kshape(r) between rows of A:(n,d) and Z:(M,d) -> (n,M),
        r = ||(A - Z)/ell|| (ARD). kid selects the shape (see gp_kernel.KERNEL_IDS)."""
        diff = (A.unsqueeze(1) - Z.unsqueeze(0)) / ell                 # (n,M,d)
        r2 = (diff * diff).sum(-1)                                     # (n,M)
        if kid == 3:                                                   # RBF
            return sf2 * torch.exp(-0.5 * r2)
        r = torch.sqrt(r2 + 1e-9)
        if kid == 0:                                                  # Matern-1/2
            return sf2 * torch.exp(-r)
        if kid == 1:                                                  # Matern-3/2
            c = _SQRT3 * r
            return sf2 * (1.0 + c) * torch.exp(-c)
        c = _SQRT5 * r                                                # Matern-5/2
        return sf2 * (1.0 + c + c * c / 3.0) * torch.exp(-c)

    @staticmethod
    def _channel_uncertainty(Xs, Z, ell, sf2, Kinv, kid):
        k = ResidualGP._kernel_cols(Xs, Z, ell, sf2, kid)             # (K,M)
        q = ((k @ Kinv) * k).sum(1)                                   # (K,) = k Kzz^{-1} k^T
        return torch.clamp(1.0 - q / sf2, 0.0, 1.0)                   # normalized variance in [0,1]

    # ---- export for the controllers (MPPI / NMPC) ------------------------
    def mpc_params(self, omega_in_rollout: bool = True) -> np.ndarray:
        """FIXED-size flat vector the controllers unpack:
            Z(M*d) alpha_v(M) alpha_w(M) x_mean(d) x_std(d) ell_v(d) ell_w(d)
            sf2_v sf2_w y_mean_v y_mean_w kernel_id
        The trailing kernel_id tells the controllers which kernel SHAPE to evaluate.
        omega_in_rollout=False zeros the omega channel (still learned/logged, just not
        used to drive the rollout -- matches the MPPI variant that drops omega)."""
        d, M = self.d, self.M
        if not self._ready:
            # alpha=0 -> zero GP contribution; x_std=1 keeps (z-x_mean)/x_std finite.
            # kernel_id = the configured kernel (harmless since the weights are zero).
            return np.concatenate([
                np.zeros(M * d), np.zeros(M), np.zeros(M),
                np.zeros(d), np.ones(d), np.ones(d), np.ones(d),
                np.array([1.0, 1.0, 0.0, 0.0, float(kernel_id_from_name(self.kernel))]),
            ])
        c = self._cache
        alpha_w = c["alpha_w"].copy()
        ymean_w = c["y_mean_w"]
        if not omega_in_rollout:
            alpha_w[:] = 0.0
            ymean_w = 0.0
        return np.concatenate([
            c["Z"].reshape(-1), c["alpha_v"], alpha_w,
            c["x_mean"], c["x_std"], c["ell_v"], c["ell_w"],
            np.array([c["sf2_v"], c["sf2_w"], c["y_mean_v"], ymean_w,
                      float(c["kernel_id"])], float),
        ])

    # ---- persistence (flat arrays for np.savez) ---------------------------
    def state_dict(self) -> dict:
        """What save_model() checkpoints. Stores the exported cache (enough for
        mpc_params/predict on reload) plus the raw buffer (so end_episode can keep
        refining). Returns {} until the first fit (nothing to save)."""
        if not self._ready:
            return {}
        c = self._cache
        return {
            "ready": np.array(1),
            "Z": c["Z"], "alpha_v": c["alpha_v"], "alpha_w": c["alpha_w"],
            "x_mean": c["x_mean"], "x_std": c["x_std"],
            "ell_v": c["ell_v"], "ell_w": c["ell_w"],
            "sf2_v": np.array(c["sf2_v"]), "sf2_w": np.array(c["sf2_w"]),
            "y_mean_v": np.array(c["y_mean_v"]), "y_mean_w": np.array(c["y_mean_w"]),
            "kernel_id": np.array(c["kernel_id"]),
            "buf_X": self._X, "buf_Y": self._Y,
        }

    def load_state_dict(self, sd: dict):
        """Restore a checkpoint. The cache is restored for immediate inference; the
        torch models are rebuilt lazily from the restored buffer on the next
        end_episode() (so learning continues, just without the previous warm start)."""
        if not sd or "ready" not in sd:
            return
        self._X = np.asarray(sd["buf_X"], float)
        self._Y = np.asarray(sd["buf_Y"], float)
        self._x_mean = np.asarray(sd["x_mean"], float)
        self._x_std = np.asarray(sd["x_std"], float)
        self._Z_std = np.ascontiguousarray(sd["Z"], float)
        self._cache = {
            "Z": self._Z_std,
            "alpha_v": np.asarray(sd["alpha_v"], float),
            "alpha_w": np.asarray(sd["alpha_w"], float),
            "x_mean": self._x_mean, "x_std": self._x_std,
            "ell_v": np.asarray(sd["ell_v"], float),
            "ell_w": np.asarray(sd["ell_w"], float),
            "sf2_v": float(sd["sf2_v"]), "sf2_w": float(sd["sf2_w"]),
            "y_mean_v": float(sd["y_mean_v"]), "y_mean_w": float(sd["y_mean_w"]),
            # older checkpoints predate kernel_id -> default to the configured kernel.
            "kernel_id": int(sd["kernel_id"]) if "kernel_id" in sd
                         else kernel_id_from_name(self.kernel),
        }
        self._models = None             # rebuilt from buffer on next end_episode()
        self._likelihoods = None
        self._xm_t = self._xs_t = None  # invalidate predict_torch cache
        self._fast = None               # invalidate predict_torch_fast cache
        self._ready = True


@dataclass
class GPConfig:
    """Config the controller stack consumes. The lengthscales / signal var / noise are
    LEARNED at runtime, so they are NOT set here -- the controller only reads max_points
    (inducing-set size M) and n_features (= d). `kernel` picks the kernel SHAPE and is
    auto-propagated to the controllers via the exported kernel_id."""
    max_points: int = 100         # inducing-set size M (= controller rollout M)
    n_features: int = 5           # feature dim d for z = [x, v, theta, omega, tau]
    kernel: str = DEFAULT_KERNEL  # "matern12" | "matern32" | "matern52" | "rbf"
    n_iter_fit: int = 150         # Adam iters on the FIRST fit (build structure)
    n_iter_update: int = 300      # Adam iters per later (warm) episode refine
    lr: float = 0.05              # Adam learning rate
    max_buffer: int = 8000        # cap training points (subsample above this)
    jitter: float = 1e-6          # Kzz solve jitter for the alpha recovery
    device: str = "cuda"          # "cuda" or "cpu"
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
# Self-test: fit on a synthetic 5-D residual and CHECK the exported Matern sum
# reproduces GPyTorch's own posterior mean. Runs without mujoco/casadi.
#   python3 GP.py
# ============================================================================ #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def true_res(X):
        x, v, th, om, tau = X.T
        obstacle = np.exp(-((x - 3.0) ** 2) / (2 * 0.25 ** 2))   # bump at x=3
        return np.stack([-3.0 * obstacle * (1.0 + 0.2 * v),
                         6.0 * obstacle - 0.1 * om], axis=1)

    N = 600
    X = np.stack([rng.uniform(0, 6, N), rng.uniform(0, 4, N),
                  rng.uniform(-0.3, 1.2, N), rng.uniform(-2, 2, N),
                  rng.uniform(-5, 10, N)], axis=1)
    Y = true_res(X) + rng.normal(0, [0.03, 0.05], size=(N, 2))

    Xt = np.stack([rng.uniform(0, 6, 50), rng.uniform(0, 4, 50),
                   rng.uniform(-0.3, 1.2, 50), rng.uniform(-2, 2, 50),
                   rng.uniform(-5, 10, 50)], axis=1)

    # Run each kernel: the exported closed-form mean must reproduce GPyTorch's own mean.
    for kname in ("matern32", "rbf"):
        gp = GPConfig(kernel=kname, max_points=50, n_iter_fit=400, device="cpu").build()
        for z, y in zip(X, Y):
            gp.observe(z, y[0], y[1])
        gp.end_episode()

        Xs = (Xt - gp._x_mean) / gp._x_std
        Tt = torch.as_tensor(Xs, dtype=torch.double)
        max_diff = 0.0
        for j, tag in zip((0, 1), ("v", "w")):
            m = gp._models[j]; m.eval()
            with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
                torch_mean = m(Tt).mean.numpy() * gp._y_scale[j] + gp._y_mean[j]
            export_mean = np.array([gp.predict(z)[j] for z in Xt])
            d = float(np.max(np.abs(torch_mean - export_mean)))
            max_diff = max(max_diff, d)
            rmse = float(np.sqrt(np.mean((true_res(Xt)[:, j] - export_mean) ** 2)))
            print(f"[{kname:8s}] channel {tag}: export-vs-torch max diff = {d:.2e}  "
                  f"| held-out RMSE = {rmse:.3f}")
        flat = gp.mpc_params()
        expect = gp.M * gp.d + 2 * gp.M + 4 * gp.d + 5     # +5 = sf2_v sf2_w ym_v ym_w kernel_id
        print(f"[{kname:8s}] flat size = {flat.size} (expect {expect}), trailing kernel_id = "
              f"{flat[-1]:.0f}  | OVERALL export-vs-torch max diff = {max_diff:.2e}\n")
