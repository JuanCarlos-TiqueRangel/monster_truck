#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import gpytorch


class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, kernel: str = "matern32", learn_inducing_locations: bool = True):
        n_inducing = inducing_points.shape[0]

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(n_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        input_dim = inducing_points.shape[1]
        kernel_name = str(kernel).lower()

        if kernel_name == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        elif kernel_name == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
        elif kernel_name == "matern12":
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dim)
        else:
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


@dataclass
class SVGPConfig:
    n_inducing: int = 300
    n_iter_fit: int = 300
    n_iter_update: int = 120
    lr: float = 0.03
    kernel: str = "matern32"
    max_train_points: int = 8000
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    jitter: float = 1e-5
    learn_inducing_locations: bool = True


class SVGPRegressor:
    def __init__(self, cfg: SVGPConfig = SVGPConfig()):
        self.cfg = cfg
        self.device = self.select_device(cfg.device)

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        self.models = []
        self.likelihoods = []

        self.X_train = np.empty((0, 0), dtype=np.float32)
        self.Y_train = np.empty((0, 0), dtype=np.float32)
        self.X_buffer = []
        self.Y_buffer = []

        self.ready = False
        self.fast_cache = None

    def select_device(self, device_name: str) -> torch.device:
        if str(device_name).startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_name)
        if str(device_name).startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        x_array = np.asarray(x, dtype=np.float32).reshape(-1)
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)
        self.X_buffer.append(x_array)
        self.Y_buffer.append(y_array)

    def end_episode(self) -> None:
        if len(self.X_buffer) == 0:
            return

        X_new = np.asarray(self.X_buffer, dtype=np.float32)
        Y_new = np.asarray(self.Y_buffer, dtype=np.float32)

        self.X_buffer.clear()
        self.Y_buffer.clear()

        if self.X_train.size == 0:
            self.X_train = X_new
            self.Y_train = Y_new
        else:
            self.X_train = np.vstack((self.X_train, X_new))
            self.Y_train = np.vstack((self.Y_train, Y_new))

        if self.ready:
            n_iter = int(self.cfg.n_iter_update)
            rebuild = False
        else:
            n_iter = int(self.cfg.n_iter_fit)
            rebuild = True

        self.fit(self.X_train, self.Y_train, n_iter=n_iter, rebuild=rebuild)

    def fit(self, X: np.ndarray, Y: np.ndarray, n_iter: int | None = None, rebuild: bool = True) -> None:
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        X, Y = self.select_training_subset(X, Y)

        if rebuild or len(self.models) == 0:
            self.build_models(X, Y)

        Xs = (X - self.x_mean) / self.x_std
        Ys = (Y - self.y_mean) / self.y_std

        Xt = torch.as_tensor(Xs, dtype=self.cfg.dtype, device=self.device)
        Yt = torch.as_tensor(Ys, dtype=self.cfg.dtype, device=self.device)

        parameters = []
        for model, likelihood in zip(self.models, self.likelihoods):
            model.train()
            likelihood.train()
            parameters += list(model.parameters())
            parameters += list(likelihood.parameters())

        optimizer = torch.optim.Adam(parameters, lr=float(self.cfg.lr))

        if n_iter is None:
            total_iterations = int(self.cfg.n_iter_fit)
        else:
            total_iterations = int(n_iter)

        objectives = []
        for model, likelihood in zip(self.models, self.likelihoods):
            objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Xt.shape[0])
            objectives.append(objective)

        for _ in range(total_iterations):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), dtype=self.cfg.dtype, device=self.device)

            for output_index, objective in enumerate(objectives):
                prediction = self.models[output_index](Xt)
                loss = loss - objective(prediction, Yt[:, output_index])

            loss.backward()
            optimizer.step()

        self.X_train = X.copy()
        self.Y_train = Y.copy()
        self.ready = True
        self.update_fast_cache()

    def build_models(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.x_mean = X.mean(axis=0).astype(np.float32)
        self.x_std = X.std(axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-6] = 1.0

        self.y_mean = Y.mean(axis=0).astype(np.float32)
        self.y_std = Y.std(axis=0).astype(np.float32)
        self.y_std[self.y_std < 1e-6] = 1.0

        Xs = (X - self.x_mean) / self.x_std
        n_samples = Xs.shape[0]
        n_inducing = min(int(self.cfg.n_inducing), n_samples)

        indices = np.random.choice(n_samples, size=n_inducing, replace=False)
        inducing_points = torch.as_tensor(Xs[indices], dtype=self.cfg.dtype, device=self.device)

        self.models = []
        self.likelihoods = []

        for _ in range(Y.shape[1]):
            model = SVGPModel(
                inducing_points.clone(),
                kernel=self.cfg.kernel,
                learn_inducing_locations=bool(self.cfg.learn_inducing_locations),
            ).to(self.device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.models.append(model)
            self.likelihoods.append(likelihood)

    def select_training_subset(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_points = int(self.cfg.max_train_points)
        if X.shape[0] <= max_points:
            return X, Y

        indices = np.random.choice(X.shape[0], size=max_points, replace=False)
        return X[indices], Y[indices]

    @torch.no_grad()
    def predict(self, X: np.ndarray, return_std: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.ready:
            mean = np.zeros((X.shape[0], self.output_dim), dtype=np.float32)
            if return_std:
                std = np.zeros_like(mean)
            else:
                std = None
            return mean, std

        Xs = (X - self.x_mean) / self.x_std
        Xt = torch.as_tensor(Xs, dtype=self.cfg.dtype, device=self.device)

        means = []
        stds = []

        with gpytorch.settings.fast_pred_var():
            for output_index, model in enumerate(self.models):
                likelihood = self.likelihoods[output_index]
                model.eval()
                likelihood.eval()

                prediction = likelihood(model(Xt))
                mean = prediction.mean * float(self.y_std[output_index]) + float(self.y_mean[output_index])
                means.append(mean)

                if return_std:
                    std = prediction.stddev * float(self.y_std[output_index])
                    stds.append(std)

        mean_np = torch.stack(means, dim=1).detach().cpu().numpy()

        if not return_std:
            return mean_np, None

        std_np = torch.stack(stds, dim=1).detach().cpu().numpy()
        return mean_np, std_np

    @torch.no_grad()
    def predict_torch(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.ready:
            return torch.zeros(X.shape[0], self.output_dim, dtype=X.dtype, device=X.device)

        original_device = X.device
        original_dtype = X.dtype

        x_mean = torch.as_tensor(self.x_mean, dtype=self.cfg.dtype, device=self.device)
        x_std = torch.as_tensor(self.x_std, dtype=self.cfg.dtype, device=self.device)
        y_mean = torch.as_tensor(self.y_mean, dtype=self.cfg.dtype, device=self.device)
        y_std = torch.as_tensor(self.y_std, dtype=self.cfg.dtype, device=self.device)

        Xs = X.to(device=self.device, dtype=self.cfg.dtype)
        Xs = (Xs - x_mean) / x_std

        outputs = []
        with gpytorch.settings.fast_pred_var():
            for output_index, model in enumerate(self.models):
                model.eval()
                mean = model(Xs).mean
                mean = mean * y_std[output_index] + y_mean[output_index]
                outputs.append(mean)

        output = torch.stack(outputs, dim=1)
        return output.to(device=original_device, dtype=original_dtype)

    @torch.no_grad()
    def update_fast_cache(self) -> None:
        if not self.ready:
            self.fast_cache = None
            return

        dtype = self.cfg.dtype
        device = self.device

        cache = {
            "x_mean": torch.as_tensor(self.x_mean, dtype=dtype, device=device),
            "x_std": torch.as_tensor(self.x_std, dtype=dtype, device=device),
            "y_mean": torch.as_tensor(self.y_mean, dtype=dtype, device=device),
            "y_std": torch.as_tensor(self.y_std, dtype=dtype, device=device),
            "channels": [],
        }

        for model in self.models:
            model.eval()

            Z = model.variational_strategy.inducing_points.detach()
            Z = Z.to(dtype=dtype, device=device)

            prior_mean_Z = model.mean_module.constant.detach().to(dtype=dtype, device=device).reshape(())
            posterior_mean_Z = model(Z).mean.detach().to(dtype=dtype, device=device)

            Kzz = self.dense_matrix(model.covar_module(Z)).detach().to(dtype=dtype, device=device)
            identity = torch.eye(Kzz.shape[0], dtype=dtype, device=device)
            alpha = torch.linalg.solve(Kzz + float(self.cfg.jitter) * identity, posterior_mean_Z - prior_mean_Z)

            lengthscale = model.covar_module.base_kernel.lengthscale.detach().reshape(-1)
            lengthscale = lengthscale.to(dtype=dtype, device=device)
            inverse_lengthscale = 1.0 / lengthscale.clamp_min(1e-8)

            outputscale = model.covar_module.outputscale.detach().to(dtype=dtype, device=device).reshape(())

            cache["channels"].append(
                {
                    "Z": Z,
                    "alpha": alpha,
                    "inverse_lengthscale": inverse_lengthscale,
                    "outputscale": outputscale,
                    "prior_mean": prior_mean_Z,
                }
            )

        self.fast_cache = cache

    def dense_matrix(self, matrix):
        if hasattr(matrix, "to_dense"):
            return matrix.to_dense()
        return matrix.evaluate()

    def kernel_matrix(self, X: torch.Tensor, Z: torch.Tensor, inverse_lengthscale: torch.Tensor, outputscale: torch.Tensor) -> torch.Tensor:
        difference = (X[:, None, :] - Z[None, :, :]) * inverse_lengthscale
        squared_distance = torch.sum(difference * difference, dim=-1).clamp_min(0.0)
        kernel_name = str(self.cfg.kernel).lower()

        if kernel_name == "rbf":
            kernel = torch.exp(-0.5 * squared_distance)
        elif kernel_name == "matern12":
            distance = torch.sqrt(squared_distance + 1e-12)
            kernel = torch.exp(-distance)
        elif kernel_name == "matern52":
            distance = torch.sqrt(squared_distance + 1e-12)
            scaled_distance = 2.23606797749979 * distance
            kernel = (1.0 + scaled_distance + scaled_distance * scaled_distance / 3.0) * torch.exp(-scaled_distance)
        else:
            distance = torch.sqrt(squared_distance + 1e-12)
            scaled_distance = 1.7320508075688772 * distance
            kernel = (1.0 + scaled_distance) * torch.exp(-scaled_distance)

        return outputscale * kernel

    @torch.no_grad()
    def predict_torch_fast(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.ready or self.fast_cache is None:
            return torch.zeros(X.shape[0], self.output_dim, dtype=X.dtype, device=X.device)

        original_device = X.device
        original_dtype = X.dtype

        X_model = X.to(device=self.device, dtype=self.cfg.dtype)
        X_model = (X_model - self.fast_cache["x_mean"]) / self.fast_cache["x_std"]

        outputs = []
        for output_index, channel in enumerate(self.fast_cache["channels"]):
            Kxz = self.kernel_matrix(
                X_model,
                channel["Z"],
                channel["inverse_lengthscale"],
                channel["outputscale"],
            )
            mean_standardized = channel["prior_mean"] + Kxz @ channel["alpha"]
            mean = self.fast_cache["y_mean"][output_index] + self.fast_cache["y_std"][output_index] * mean_standardized
            outputs.append(mean)

        output = torch.stack(outputs, dim=1)
        return output.to(device=original_device, dtype=original_dtype)

    @property
    def input_dim(self) -> int:
        if self.x_mean is not None:
            return int(self.x_mean.size)
        if self.X_train.size != 0:
            return int(self.X_train.shape[1])
        if len(self.X_buffer) > 0:
            return int(self.X_buffer[0].size)
        return 0

    @property
    def output_dim(self) -> int:
        if self.y_mean is not None:
            return int(self.y_mean.size)
        if self.Y_train.size != 0:
            return int(self.Y_train.shape[1])
        if len(self.Y_buffer) > 0:
            return int(self.Y_buffer[0].size)
        return 1

    def save(self, path: str | Path) -> None:
        path = Path(path)

        n_inducing_used = 0
        if self.ready and len(self.models) > 0:
            n_inducing_used = int(
                self.models[0].variational_strategy.inducing_points.shape[0]
            )

        payload = {
            "cfg": self.config_to_dict(),
            "n_inducing_used": torch.tensor(n_inducing_used),
            "x_mean": torch.as_tensor(self.x_mean, dtype=torch.float32),
            "x_std": torch.as_tensor(self.x_std, dtype=torch.float32),
            "y_mean": torch.as_tensor(self.y_mean, dtype=torch.float32),
            "y_std": torch.as_tensor(self.y_std, dtype=torch.float32),
            "X_train": torch.as_tensor(self.X_train, dtype=torch.float32),
            "Y_train": torch.as_tensor(self.Y_train, dtype=torch.float32),
            "ready": bool(self.ready),
            "models": [model.state_dict() for model in self.models],
            "likelihoods": [likelihood.state_dict() for likelihood in self.likelihoods],
        }

        torch.save(payload, path)


    @classmethod
    def load(cls, path: str | Path, map_location: str | None = None) -> "SVGPRegressor":
        try:
            payload = torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=map_location)

        cfg = cls.config_from_dict(payload["cfg"])

        if "n_inducing_used" in payload:
            cfg.n_inducing = int(payload["n_inducing_used"].item())
        elif bool(payload["ready"]) and len(payload["models"]) > 0:
            first_state = payload["models"][0]
            cfg.n_inducing = int(first_state["variational_strategy.inducing_points"].shape[0])

        gp = cls(cfg)

        gp.x_mean = payload["x_mean"].cpu().numpy().astype(np.float32)
        gp.x_std = payload["x_std"].cpu().numpy().astype(np.float32)
        gp.y_mean = payload["y_mean"].cpu().numpy().astype(np.float32)
        gp.y_std = payload["y_std"].cpu().numpy().astype(np.float32)
        gp.X_train = payload["X_train"].cpu().numpy().astype(np.float32)
        gp.Y_train = payload["Y_train"].cpu().numpy().astype(np.float32)
        gp.ready = bool(payload["ready"])

        if gp.ready:
            gp.build_models(gp.X_train, gp.Y_train)

            gp.x_mean = payload["x_mean"].cpu().numpy().astype(np.float32)
            gp.x_std = payload["x_std"].cpu().numpy().astype(np.float32)
            gp.y_mean = payload["y_mean"].cpu().numpy().astype(np.float32)
            gp.y_std = payload["y_std"].cpu().numpy().astype(np.float32)

            for model, state in zip(gp.models, payload["models"]):
                model.load_state_dict(state)

            for likelihood, state in zip(gp.likelihoods, payload["likelihoods"]):
                likelihood.load_state_dict(state)

            gp.update_fast_cache()

        return gp

    def config_to_dict(self) -> dict:
        if self.cfg.dtype == torch.float64:
            dtype_name = "float64"
        else:
            dtype_name = "float32"

        return {
            "n_inducing": int(self.cfg.n_inducing),
            "n_iter_fit": int(self.cfg.n_iter_fit),
            "n_iter_update": int(self.cfg.n_iter_update),
            "lr": float(self.cfg.lr),
            "kernel": str(self.cfg.kernel),
            "max_train_points": int(self.cfg.max_train_points),
            "device": str(self.cfg.device),
            "dtype": dtype_name,
            "jitter": float(self.cfg.jitter),
            "learn_inducing_locations": bool(self.cfg.learn_inducing_locations),
        }

    @classmethod
    def config_from_dict(cls, cfg_dict: dict) -> SVGPConfig:
        dtype_name = str(cfg_dict.get("dtype", "float32"))
        if dtype_name == "float64":
            dtype = torch.float64
        else:
            dtype = torch.float32

        return SVGPConfig(
            n_inducing=int(cfg_dict.get("n_inducing", 50)),
            n_iter_fit=int(cfg_dict.get("n_iter_fit", 300)),
            n_iter_update=int(cfg_dict.get("n_iter_update", 120)),
            lr=float(cfg_dict.get("lr", 0.03)),
            kernel=str(cfg_dict.get("kernel", "matern32")),
            max_train_points=int(cfg_dict.get("max_train_points", 8000)),
            device=str(cfg_dict.get("device", "cuda")),
            dtype=dtype,
            jitter=float(cfg_dict.get("jitter", 1e-5)),
            learn_inducing_locations=bool(cfg_dict.get("learn_inducing_locations", True)),
        )