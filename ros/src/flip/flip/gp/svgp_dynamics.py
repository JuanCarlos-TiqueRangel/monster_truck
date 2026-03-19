import numpy as np
import torch
import gpytorch

# ------------------------------------------------------------
# SVGP model (single-output)
# ------------------------------------------------------------

class SVGPModel(gpytorch.models.ApproximateGP):
    """
    Sparse Variational GP for 1D output.
    Uses a LinearMean + (Scale * base_kernel) covariance like your ExactGP.
    """
    def __init__(
        self,
        inducing_points: torch.Tensor,        # shape (M, D) in *normalized* space
        kernel: str = "RBF",
        learn_inducing_locations: bool = True,
    ):
        # Variational distribution and strategy
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        input_dim = inducing_points.size(-1)
        self.mean_module = gpytorch.means.LinearMean(input_size=input_dim)

        if kernel == "RBF":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        elif kernel == "Matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        elif kernel == "RQ":
            base_kernel = gpytorch.kernels.RQKernel(ard_num_dims=input_dim)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ------------------------------------------------------------
# SVGP manager (warm-start + frozen normalization + minibatches)
# ------------------------------------------------------------

_device = torch.device("cuda")

def kcenter_greedy_torch(X: torch.Tensor, M: int, seed: int | None = None) -> torch.Tensor:
    """
    Greedy k-center / farthest-point sampling in Euclidean distance.
    Runs fully in torch (CPU or GPU), no numpy copies.

    Args:
      X: (N, D) torch tensor
      M: number of points
      seed: optional seed for the first center

    Returns:
      idx: (M,) long indices into X
    """
    N = X.size(0)
    if M >= N:
        return torch.arange(N, device=X.device, dtype=torch.long)

    # seed only affects the first selected point
    if seed is None:
        first = torch.randint(0, N, (1,), device=X.device).item()
    else:
        g = torch.Generator(device=X.device)
        g.manual_seed(int(seed))
        first = torch.randint(0, N, (1,), generator=g, device=X.device).item()

    idx = torch.empty(M, dtype=torch.long, device=X.device)
    idx[0] = first

    # squared distance to nearest chosen center so far
    d2 = torch.sum((X - X[first:first+1]) ** 2, dim=1)

    for m in range(1, M):
        i = torch.argmax(d2).item()
        idx[m] = i
        di2 = torch.sum((X - X[i:i+1]) ** 2, dim=1)
        d2 = torch.minimum(d2, di2)

    return idx


class SVGPManager:
    def __init__(
        self,
        kernel: str = "RBF",
        lr: float = 0.01,
        iters: int = 300,          # initial gradient steps (fit from scratch)
        batch_size: int = 1024,     # minibatch size
        num_inducing: int = 256,    # M
        learn_inducing_locations: bool = True,
        device: torch.device = _device,
        store_train_data_in_ckpt: bool = False,  # optional
    ):
        self.kernel = kernel
        self.lr = lr
        self.iters = iters
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.learn_inducing_locations = learn_inducing_locations
        self.device = device
        self.store_train_data_in_ckpt = store_train_data_in_ckpt

        self.trained = False

        # Optional raw storage (not required for SVGP inference, but useful for debugging)
        self.X_train: torch.Tensor | None = None
        self.Y_train: torch.Tensor | None = None

        # Normalized replay buffer (what we actually train on)
        self.Xn_train: torch.Tensor | None = None
        self.Yn_train: torch.Tensor | None = None

        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: SVGPModel | None = None

        # Normalization stats (recommend freezing after initial fit)
        self.X_mean: torch.Tensor | None = None
        self.X_std: torch.Tensor | None = None
        self.Y_mean: torch.Tensor | None = None
        self.Y_std: torch.Tensor | None = None
        self.norm_frozen: bool = False

        # Warm-start training objects (persist across updates)
        self.optimizer: torch.optim.Optimizer | None = None
        self.elbo: gpytorch.mlls.VariationalELBO | None = None

    # ----------------------------- #
    # FIT / RETRAIN entrypoints
    # ----------------------------- #
    def fit(self, X: np.ndarray, Y: np.ndarray, freeze_norm: bool = True) -> None:
        """
        Initial training from scratch (creates inducing points, optimizer, etc.)
        After this, you can do warm updates via warm_update().
        """
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device).flatten()

        self.X_train = X.clone()
        self.Y_train = Y.clone()

        self._compute_normalization()
        self.norm_frozen = bool(freeze_norm)

        self.Xn_train = (self.X_train - self.X_mean) / self.X_std
        self.Yn_train = (self.Y_train - self.Y_mean) / self.Y_std

        self._init_model_optimizer_elbo()

        # "Long" initial train
        self._optimize_steps(steps=self.iters, batch_size=self.batch_size)

        self.trained = True

    def retrain(self) -> None:
        """
        Kept for API compatibility.
        For SVGP + warm-start, you usually do NOT want to rebuild from scratch here.

        This implementation:
          - if not trained: raises
          - otherwise: performs a warm update (default: steps=self.iters//10, min 200)
        """
        if not self.trained:
            raise RuntimeError("SVGP has not been fit yet. Call fit() first.")

        warm_steps = max(200, int(self.iters // 10))
        self.warm_update(steps=warm_steps, batch_size=self.batch_size)

    def retrain_from_scratch(self, X: np.ndarray | None = None, Y: np.ndarray | None = None, freeze_norm: bool = True):
        """
        If you ever want a full rebuild, call this explicitly.
        """
        if X is None or Y is None:
            if self.X_train is None or self.Y_train is None:
                raise RuntimeError("No training data available to retrain from scratch.")
            X = self.X_train.detach().cpu().numpy()
            Y = self.Y_train.detach().cpu().numpy()

        # Reset everything
        self.__init__(
            kernel=self.kernel,
            lr=self.lr,
            iters=self.iters,
            batch_size=self.batch_size,
            num_inducing=self.num_inducing,
            learn_inducing_locations=self.learn_inducing_locations,
            device=self.device,
            store_train_data_in_ckpt=self.store_train_data_in_ckpt,
        )
        self.fit(X, Y, freeze_norm=freeze_norm)

    # ----------------------------- #
    # Online data: add + warm update
    # ----------------------------- #
    def add_data(
        self,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        retrain: bool = True,
        warm_steps: int = 200,
        max_points: int = 50_000,
        keep_raw: bool = False,
    ):
        """
        Append new points to the *normalized* replay buffer and (optionally) warm-update.

        Works both:
        - right after fit()
        - right after load() even if the checkpoint did NOT store Xn_train/Yn_train
        """
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        Y_new_t = torch.tensor(Y_new, dtype=torch.float32, device=self.device).flatten()

        # If completely uninitialized, do a first fit.
        if not self.trained:
            self.fit(
                X_new_t.detach().cpu().numpy(),
                Y_new_t.detach().cpu().numpy(),
                freeze_norm=True
            )
            return

        # Optional raw store (debug only)
        if keep_raw:
            if self.X_train is None:
                self.X_train = X_new_t.clone()
                self.Y_train = Y_new_t.clone()
            else:
                self.X_train = torch.cat([self.X_train, X_new_t], dim=0)
                self.Y_train = torch.cat([self.Y_train, Y_new_t], dim=0)

        # Must have normalization stats after fit/load
        if self.X_mean is None or self.X_std is None or self.Y_mean is None or self.Y_std is None:
            raise RuntimeError("Normalization stats missing. This should not happen after fit() or load().")

        # --- Update normalized replay buffer ---
        if not self.norm_frozen:
            # expensive path: recompute stats + renormalize whole buffer
            if self.X_train is None or self.Y_train is None:
                raise RuntimeError(
                    "norm_frozen=False requires raw X_train/Y_train to recompute normalization. "
                    "Set keep_raw=True earlier or freeze_norm=True."
                )
            self._compute_normalization()
            self.Xn_train = (self.X_train - self.X_mean) / self.X_std
            self.Yn_train = (self.Y_train - self.Y_mean) / self.Y_std
        else:
            # fast path: normalize only new points
            Xn_new = (X_new_t - self.X_mean) / self.X_std
            Yn_new = (Y_new_t - self.Y_mean) / self.Y_std

            if self.Xn_train is None:
                self.Xn_train = Xn_new.clone()
                self.Yn_train = Yn_new.clone()
            else:
                self.Xn_train = torch.cat([self.Xn_train, Xn_new], dim=0)
                self.Yn_train = torch.cat([self.Yn_train, Yn_new], dim=0)

        assert self.Xn_train is not None and self.Yn_train is not None

        # Sliding window cap
        if self.Xn_train.size(0) > int(max_points):
            self.Xn_train = self.Xn_train[-int(max_points):].contiguous()
            self.Yn_train = self.Yn_train[-int(max_points):].contiguous()

        # --- Ensure optimizer + ELBO exist (common after load without stored buffer) ---
        if self.optimizer is None or self.elbo is None:
            self._init_optimizer_elbo_only()

        # Always keep ELBO num_data in sync
        self.elbo.num_data = self.Xn_train.size(0)

        if retrain:
            self.warm_update(steps=int(warm_steps), batch_size=self.batch_size)



    def warm_update(self, steps: int = 200, batch_size: int | None = None) -> None:
        """
        Warm-start update: run a small number of gradient steps on minibatches
        without rebuilding the model/inducing/variational distribution.
        """
        if not self.trained or self.model is None or self.likelihood is None:
            raise RuntimeError("SVGP has not been trained yet.")

        if self.Xn_train is None or self.Yn_train is None:
            raise RuntimeError("No replay buffer to train on. Call add_data() or fit().")

        if self.optimizer is None or self.elbo is None:
            # Should not happen, but can happen after a custom load; rebuild safely.
            self._init_optimizer_elbo_only()

        self._optimize_steps(steps=int(steps), batch_size=batch_size or self.batch_size)

    # ----------------------------- #
    # INTERNALS
    # ----------------------------- #
    def _compute_normalization(self) -> None:
        assert self.X_train is not None and self.Y_train is not None

        X_mean = self.X_train.mean(0)
        X_std = self.X_train.std(0)
        Y_mean = self.Y_train.mean()
        Y_std = self.Y_train.std()

        X_std = torch.where(X_std == 0.0, torch.ones_like(X_std), X_std)
        if float(Y_std) == 0.0:
            Y_std = torch.tensor(1.0, device=self.device)

        self.X_mean = X_mean
        self.X_std = X_std
        self.Y_mean = Y_mean
        self.Y_std = Y_std

    def _choose_inducing_points(self, Xn: torch.Tensor) -> torch.Tensor:
        N = Xn.size(0)
        M = min(int(self.num_inducing), N)
        #idx = torch.randperm(N, device=self.device)[:M]

        # k-center inducing init (instead of random subset)
        idx = kcenter_greedy_torch(Xn, M, seed=0)  # change seed if you want
        return Xn[idx].contiguous()

    def _init_model_optimizer_elbo(self) -> None:
        assert self.Xn_train is not None and self.Yn_train is not None

        inducing = self._choose_inducing_points(self.Xn_train)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise_covar.initialize(noise=1e-3)

        self.model = SVGPModel(
            inducing_points=inducing,
            kernel=self.kernel,
            learn_inducing_locations=self.learn_inducing_locations,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr
        )

        self.elbo = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.Xn_train.size(0)
        )

        self.model.eval()
        self.likelihood.eval()

    def _init_optimizer_elbo_only(self) -> None:
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model/likelihood missing; cannot init optimizer/ELBO.")
        if self.Xn_train is None:
            raise RuntimeError("Replay buffer missing; cannot init ELBO num_data.")

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=self.lr
        )
        self.elbo = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=self.Xn_train.size(0)
        )

    def _optimize_steps(self, steps: int, batch_size: int) -> None:
        assert self.model is not None and self.likelihood is not None
        assert self.optimizer is not None and self.elbo is not None
        assert self.Xn_train is not None and self.Yn_train is not None

        self.model.train()
        self.likelihood.train()

        N = self.Xn_train.size(0)
        B = min(int(batch_size), int(N))

        for _ in range(int(steps)):
            idx = torch.randint(0, N, (B,), device=self.device)
            xb = self.Xn_train[idx]
            yb = self.Yn_train[idx].flatten()

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(xb)
            loss = -self.elbo(out, yb)
            loss.backward()
            self.optimizer.step()

        self.model.eval()
        self.likelihood.eval()

    # ----------------------------- #
    # PREDICT (same signature)
    # ----------------------------- #
    # def predict_torch(self, X):
    #     if not self.trained or self.model is None or self.likelihood is None:
    #         raise RuntimeError("SVGP has not been trained yet.")
    #     if self.X_mean is None or self.X_std is None or self.Y_mean is None or self.Y_std is None:
    #         raise RuntimeError("Normalization stats missing.")

    #     X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
    #     Xn = (X - self.X_mean) / self.X_std

    #     with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #         #pred = self.likelihood(self.model(Xn))
    #         pred = self.model(Xn)
    #         mean = pred.mean * self.Y_std + self.Y_mean
    #         var = pred.variance * (self.Y_std ** 2)
    #     return mean, var


    @torch.inference_mode()
    def predict_mean_torch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fast mean-only inference for control.
        Returns denormalized predictive mean only.
        """
        if not self.trained or self.model is None:
            raise RuntimeError("SVGP has not been trained yet.")
        if self.X_mean is None or self.X_std is None or self.Y_mean is None or self.Y_std is None:
            raise RuntimeError("Normalization stats missing.")

        # Assume X is already a torch tensor on the correct device in the hot loop.
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        else:
            if X.device != self.device:
                X = X.to(self.device, non_blocking=True)
            if X.dtype != torch.float32:
                X = X.float()

        Xn = (X - self.X_mean) / self.X_std

        # Mean only. No variance.
        pred = self.model(Xn)
        mean = pred.mean * self.Y_std + self.Y_mean
        return mean



    # ----------------------------- #
    # Convenience
    # ----------------------------- #
    def dataset(self):
        """
        Returns raw dataset if you kept it (may be None).
        """
        if self.X_train is None or self.Y_train is None:
            return None, None
        return self.X_train.detach().cpu().numpy(), self.Y_train.detach().cpu().numpy()

    def buffer_size(self) -> int:
        return 0 if self.Xn_train is None else int(self.Xn_train.size(0))

    # ----------------------------- #
    # SAVE / LOAD
    # ----------------------------- #
    def save(self, path: str) -> None:
        if not self.trained or self.model is None or self.likelihood is None:
            raise RuntimeError("Cannot save an untrained SVGPManager.")
        if self.X_mean is None or self.X_std is None or self.Y_mean is None or self.Y_std is None:
            raise RuntimeError("Normalization stats missing; cannot save.")

        inducing = self.model.variational_strategy.inducing_points.detach().cpu()

        state = {
            "model_type": "SVGPManager",
            "kernel": self.kernel,
            "lr": self.lr,
            "iters": self.iters,
            "batch_size": self.batch_size,
            "num_inducing": self.num_inducing,
            "learn_inducing_locations": self.learn_inducing_locations,
            "norm_frozen": self.norm_frozen,

            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
            "Y_mean": self.Y_mean.detach().cpu(),
            "Y_std": self.Y_std.detach().cpu(),

            "inducing_points": inducing,

            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }

        # Optional: store replay buffer so warm-start can continue after load
        if self.store_train_data_in_ckpt and self.Xn_train is not None and self.Yn_train is not None:
            state["Xn_train"] = self.Xn_train.detach().cpu()
            state["Yn_train"] = self.Yn_train.detach().cpu()

        # Optional raw
        if self.store_train_data_in_ckpt and self.X_train is not None and self.Y_train is not None:
            state["X_train"] = self.X_train.detach().cpu()
            state["Y_train"] = self.Y_train.detach().cpu()

        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "SVGPManager":
        if device is None:
            device = _device

        state = torch.load(path, map_location=device)
        if state.get("model_type", "") != "SVGPManager":
            raise RuntimeError("Checkpoint is not an SVGPManager file.")

        gp = cls(
            kernel=state["kernel"],
            lr=state["lr"],
            iters=state["iters"],
            batch_size=state["batch_size"],
            num_inducing=state["num_inducing"],
            learn_inducing_locations=state["learn_inducing_locations"],
            device=device,
            store_train_data_in_ckpt=("Xn_train" in state or "X_train" in state),
        )
        gp.norm_frozen = bool(state.get("norm_frozen", True))

        gp.X_mean = state["X_mean"].to(device)
        gp.X_std  = state["X_std"].to(device)
        gp.Y_mean = state["Y_mean"].to(device)
        gp.Y_std  = state["Y_std"].to(device)

        inducing = state["inducing_points"].to(device)

        gp.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp.likelihood.load_state_dict(state["likelihood_state_dict"])

        gp.model = SVGPModel(
            inducing_points=inducing,
            kernel=gp.kernel,
            learn_inducing_locations=gp.learn_inducing_locations,
        ).to(device)
        gp.model.load_state_dict(state["model_state_dict"])

        # Restore buffers if present (enables warm-start immediately)
        if "Xn_train" in state and "Yn_train" in state:
            gp.Xn_train = state["Xn_train"].to(device)
            gp.Yn_train = state["Yn_train"].to(device)

        if "X_train" in state and "Y_train" in state:
            gp.X_train = state["X_train"].to(device)
            gp.Y_train = state["Y_train"].to(device)

        gp.model.eval()
        gp.likelihood.eval()
        gp.trained = True

        # Recreate optimizer + ELBO if replay buffer exists (so warm_update works)
        if gp.Xn_train is not None and gp.Yn_train is not None:
            gp._init_optimizer_elbo_only()

        return gp
