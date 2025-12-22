import numpy as np
import torch
import gpytorch

#_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_device = torch.device("cuda")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel: str = "RBF"):
        super().__init__(train_x, train_y, likelihood)

        input_dim = train_x.shape[-1]
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

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPManager:
    def __init__(
        self,
        kernel: str = "RBF",
        lr: float = 0.03,
        iters: int = 300,
        device: torch.device = _device,
    ):
        self.kernel = kernel
        self.lr = lr
        self.iters = iters
        self.device = device

        self.trained = False
        self.X_train: torch.Tensor | None = None
        self.Y_train: torch.Tensor | None = None

        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: ExactGPModel | None = None

        # normalization buffers
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None
        self.Xn = None
        self.Yn = None

    # ----------------------------- #
    #        FIT / INITIAL TRAIN    #
    # ----------------------------- #
    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.float32, device=self.device).flatten()

        self.X_train = X.clone()
        self.Y_train = Y.clone()

        self.retrain()

    def retrain(self) -> None:
        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("No training data set yet.")
        self._compute_normalization()
        self._train_model()

    # ----------------------------- #
    #       ADD NEW DATA POINTS     #
    # ----------------------------- #
    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray, retrain: bool = True):
        X_new = torch.tensor(X_new, dtype=torch.float32, device=self.device)
        Y_new = torch.tensor(Y_new, dtype=torch.float32, device=self.device).flatten()

        if self.X_train is None:
            # no previous data; treat as first fit
            self.fit(X_new.cpu().numpy(), Y_new.cpu().numpy())
            return

        if self.Y_train.ndim > 1:
            self.Y_train = self.Y_train.flatten()

        self.X_train = torch.cat([self.X_train, X_new], dim=0)
        self.Y_train = torch.cat([self.Y_train, Y_new], dim=0)

        if retrain:
            self.retrain()

    # ----------------------------- #
    #         INTERNAL UTILS        #
    # ----------------------------- #
    def _compute_normalization(self) -> None:
        self.X_mean = self.X_train.mean(0)
        self.X_std = self.X_train.std(0)
        self.Y_mean = self.Y_train.mean()
        self.Y_std = self.Y_train.std()

        self.X_std = torch.where(
            self.X_std == 0.0, torch.ones_like(self.X_std), self.X_std
        )
        if self.Y_std == 0.0:
            self.Y_std = torch.tensor(1.0, device=self.device)

        self.Xn = (self.X_train - self.X_mean) / self.X_std
        self.Yn = (self.Y_train - self.Y_mean) / self.Y_std

    def dataset(self):
        X_train = self.X_train.detach().cpu().numpy()
        Y_train = self.Y_train.detach().cpu().numpy()
        return X_train, Y_train

    def _train_model(self) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise_covar.initialize(noise=1e-3)

        self.model = ExactGPModel(
            self.Xn,
            self.Yn,
            self.likelihood,
            kernel=self.kernel,
        ).to(self.device)

        self._optimize_gp(self.model, self.likelihood, self.Xn, self.Yn)
        self.trained = True

    def _optimize_gp(
        self,
        model: ExactGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        model.train()
        likelihood.train()

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(self.iters):
            opt.zero_grad()
            out = model(x)
            loss = -mll(out, y)
            loss.backward()
            opt.step()

        model.eval()
        likelihood.eval()

    # ----- Torch-friendly predict (for MPPI on GPU) -----
    def predict_torch(self, X):
        if not self.trained:
            raise RuntimeError("GP has not been trained yet.")

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Xn = (X - self.X_mean) / self.X_std

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(Xn))
            mean = pred.mean * self.Y_std + self.Y_mean
            var = pred.variance * (self.Y_std**2)
        return mean, var

    # ====================================================
    #           NEW: SAVE / LOAD FOR REUSE
    # ====================================================
    def save(self, path: str) -> None:
        """
        Save trained GP to disk so you can reuse it without retraining.
        Stores:
          - hyperparameters (kernel, lr, iters)
          - model & likelihood state_dict
          - normalization stats
          - training data (needed for exact GP inference)
        """
        if not self.trained or self.model is None or self.likelihood is None:
            raise RuntimeError("Cannot save an untrained GPManager.")

        state = {
            "kernel": self.kernel,
            "lr": self.lr,
            "iters": self.iters,

            "X_train": self.X_train.detach().cpu(),
            "Y_train": self.Y_train.detach().cpu(),

            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
            "Y_mean": self.Y_mean.detach().cpu(),
            "Y_std": self.Y_std.detach().cpu(),

            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "GPManager":
        """
        Load a previously saved GPManager from disk.
        No retraining, just rebuild model + likelihood and load weights.
        """
        if device is None:
            device = _device

        state = torch.load(path, map_location=device)

        gp = cls(
            kernel=state["kernel"],
            lr=state["lr"],
            iters=state["iters"],
            device=device,
        )

        gp.X_train = state["X_train"].to(device)
        gp.Y_train = state["Y_train"].to(device)

        gp.X_mean = state["X_mean"].to(device)
        gp.X_std  = state["X_std"].to(device)
        gp.Y_mean = state["Y_mean"].to(device)
        gp.Y_std  = state["Y_std"].to(device)

        # recompute normalized training data
        gp.Xn = (gp.X_train - gp.X_mean) / gp.X_std
        gp.Yn = (gp.Y_train - gp.Y_mean) / gp.Y_std

        # rebuild likelihood + model and load weights
        gp.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        gp.likelihood.load_state_dict(state["likelihood_state_dict"])

        gp.model = ExactGPModel(
            gp.Xn, gp.Yn, gp.likelihood, kernel=gp.kernel
        ).to(device)
        gp.model.load_state_dict(state["model_state_dict"])

        gp.model.eval()
        gp.likelihood.eval()
        gp.trained = True

        return gp
