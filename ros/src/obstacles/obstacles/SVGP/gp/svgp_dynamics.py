import numpy as np
import torch
import gpytorch

# _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_device = torch.device("cuda")


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: str = "RBF"):
        input_dim = inducing_points.shape[-1]

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

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


class SVGPManager:
    def __init__(
        self,
        kernel: str = "RBF",
        lr: float = 0.03,
        iters: int = 300,
        num_inducing: int = 128,
        batch_size: int | None = None,
        device: torch.device = _device,
    ):
        self.kernel = kernel
        self.lr = lr
        self.iters = iters
        self.num_inducing = num_inducing
        self.batch_size = batch_size
        self.device = device

        self.trained = False
        self.X_train: torch.Tensor | None = None
        self.Y_train: torch.Tensor | None = None

        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.model: SVGPModel | None = None

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


    def _select_inducing_points(self, Xn: torch.Tensor) -> torch.Tensor:
        n = Xn.size(0)
        m = min(self.num_inducing, n)

        if m == n:
            return Xn.clone()

        idx = torch.randperm(n, device=Xn.device)[:m]
        return Xn[idx].clone()


    def dataset(self):
        X_train = self.X_train.detach().cpu().numpy()
        Y_train = self.Y_train.detach().cpu().numpy()
        return X_train, Y_train


    def _train_model(self) -> None:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)).to(self.device)
        
        self.likelihood.noise_covar.initialize(noise=1e-2)

        inducing_points = self._select_inducing_points(self.Xn)

        self.model = SVGPModel(
            inducing_points=inducing_points,
            kernel=self.kernel,
        ).to(self.device)

        self._optimize_gp(self.model, self.likelihood, self.Xn, self.Yn)
        self.trained = True


    def _optimize_gp(
        self,
        model: SVGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        x: torch.Tensor,
        y: torch.Tensor) -> None:
        
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=self.lr,
        )

        num_data = y.size(0)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=num_data)

        with gpytorch.settings.cholesky_jitter(1e-3, 1e-6):
            for _ in range(self.iters):
                if self.batch_size is None or self.batch_size >= num_data:
                    batches = (torch.arange(num_data, device=x.device),)
                else:
                    perm = torch.randperm(num_data, device=x.device)
                    batches = perm.split(self.batch_size)

                for batch_idx in batches:
                    xb = x[batch_idx]
                    yb = y[batch_idx]

                    optimizer.zero_grad()
                    out = model(xb)
                    loss = -mll(out, yb)
                    loss.backward()
                    optimizer.step()

        model.eval()
        likelihood.eval()

    # ----- Torch-friendly predict (for MPPI on GPU) -----
    def predict_torch(self, X):
        if not self.trained:
            raise RuntimeError("GP has not been trained yet.")

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Xn = (X - self.X_mean) / self.X_std

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-3, 1e-6):
            pred = self.likelihood(self.model(Xn))
            mean = pred.mean * self.Y_std + self.Y_mean
            # var = pred.variance * (self.Y_std ** 2)

        return mean  # , var

    # ====================================================
    #           SAVE / LOAD FOR REUSE
    # ====================================================
    def save(self, path: str) -> None:
        """
        Save trained SVGP to disk.
        Stores:
          - hyperparameters
          - model & likelihood state_dict
          - normalization stats
          - training data (useful for retraining / add_data)
          - inducing points
        """
        if not self.trained or self.model is None or self.likelihood is None:
            raise RuntimeError("Cannot save an untrained SVGPManager.")

        state = {
            "kernel": self.kernel,
            "lr": self.lr,
            "iters": self.iters,
            "num_inducing": self.num_inducing,
            "batch_size": self.batch_size,

            "X_train": self.X_train.detach().cpu(),
            "Y_train": self.Y_train.detach().cpu(),

            "X_mean": self.X_mean.detach().cpu(),
            "X_std": self.X_std.detach().cpu(),
            "Y_mean": self.Y_mean.detach().cpu(),
            "Y_std": self.Y_std.detach().cpu(),

            "inducing_points": self.model.variational_strategy.inducing_points.detach().cpu(),

            "model_state_dict": self.model.state_dict(),
            "likelihood_state_dict": self.likelihood.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "SVGPManager":
        """
        Load a previously saved SVGPManager from disk.
        No retraining, just rebuild model + likelihood and load weights.
        """
        if device is None:
            device = _device

        state = torch.load(path, map_location=device)

        gp = cls(
            kernel=state["kernel"],
            lr=state["lr"],
            iters=state["iters"],
            num_inducing=state["num_inducing"],
            batch_size=state["batch_size"],
            device=device,
        )

        gp.X_train = state["X_train"].to(device)
        gp.Y_train = state["Y_train"].to(device)

        gp.X_mean = state["X_mean"].to(device)
        gp.X_std = state["X_std"].to(device)
        gp.Y_mean = state["Y_mean"].to(device)
        gp.Y_std = state["Y_std"].to(device)

        gp.Xn = (gp.X_train - gp.X_mean) / gp.X_std
        gp.Yn = (gp.Y_train - gp.Y_mean) / gp.Y_std

        gp.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        ).to(device)
        gp.likelihood.load_state_dict(state["likelihood_state_dict"])

        inducing_points = state["inducing_points"].to(device)

        gp.model = SVGPModel(
            inducing_points=inducing_points,
            kernel=gp.kernel,
        ).to(device)
        gp.model.load_state_dict(state["model_state_dict"])

        gp.model.eval()
        gp.likelihood.eval()
        gp.trained = True

        return gp