"""
================================================================================
GP_gptorch.py  --  the "standard GPyTorch way": swap the kernel on ONE line
================================================================================

Same residual model and SAME controller contract as GP.py (5-input -> 2 residual
outputs, exports the flat vector the MPPI/NMPC re-evaluate). The ONLY difference:
here the covariance kernel is written the plain GPyTorch way, inline, so you choose
it by editing a single line -- exactly like:

    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

It reuses the tested machinery in GP.ResidualGP (training, the alpha = Kzz^-1 m(Z)
export, persistence) and overrides only `_make_covar`. The exported `kernel_id`
auto-detects whatever kernel you put on the line, so the **MPPI rollout follows
automatically** -- no other change needed.

Drop-in for the sim: in obstacle_mujoco_simulation.py swap
    from GP import GPConfig as SSGPConfig
for
    from GP_gptorch import GPConfig as SSGPConfig

NOTE on the casadi NMPC: it bakes its kernel in at build time from GPConfig.kernel
(it can't read the runtime kernel_id), so if you change the line below, also set
`kernel=` in GPConfig to match. The numba MPPI needs no such sync.

Self-test:  python3 GP_gptorch.py
================================================================================
"""

from dataclasses import dataclass

import gpytorch

from GP import ResidualGP, GPConfig as _BaseGPConfig


class GPyTorchResidualGP(ResidualGP):
    """ResidualGP whose kernel is chosen the standard GPyTorch way, inline below."""

    def _make_covar(self, d):
        # ====================================================================
        #  >>> CHANGE THIS ONE LINE to swap the kernel (standard GPyTorch) <<<
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=d))
        # ---- other options (uncomment one, comment the line above) ----------
        #   return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=d))
        #   return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=d))
        #   return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=d))
        # Only RBF / Matern(0.5,1.5,2.5) have a controller closed form (gp_kernel.py).
        # To use anything else, add its shape to gp_kernel (numba + numpy + casadi) and
        # to detect_kernel_id. ard_num_dims=d keeps a separate lengthscale per feature.
        # ====================================================================


@dataclass
class GPConfig(_BaseGPConfig):
    """Drop-in config that builds the GPyTorch-style GP. `kernel` here is used ONLY by
    the casadi NMPC (the GPyTorch model + the MPPI follow the line in _make_covar above,
    via the auto-detected kernel_id). Keep it in sync with that line for the NMPC."""
    kernel: str = "rbf"           # MUST match the line in GPyTorchResidualGP._make_covar (for the NMPC)

    def build(self, n_features: int | None = None) -> GPyTorchResidualGP:
        return GPyTorchResidualGP(
            n_features=self.n_features if n_features is None else n_features,
            max_points=self.max_points, kernel=self.kernel,
            n_iter_fit=self.n_iter_fit, n_iter_update=self.n_iter_update,
            lr=self.lr, max_buffer=self.max_buffer, jitter=self.jitter,
            device=self.device, seed=self.seed)


# ---- alias so this is a drop-in for the old config name ------------------------
OnlineGPConfig = GPConfig


# ============================================================================ #
# Self-test: fit with the inline kernel and confirm the exported mean reproduces
# GPyTorch's own mean, and that kernel_id matches the line. No mujoco/casadi.
# ============================================================================ #
if __name__ == "__main__":
    import numpy as np
    import torch

    rng = np.random.default_rng(0)

    def true_res(X):
        x, v, th, om, tau = X.T
        obstacle = np.exp(-((x - 3.0) ** 2) / (2 * 0.25 ** 2))
        return np.stack([-3.0 * obstacle * (1.0 + 0.2 * v),
                         6.0 * obstacle - 0.1 * om], axis=1)

    N = 600
    X = np.stack([rng.uniform(0, 6, N), rng.uniform(0, 4, N),
                  rng.uniform(-0.3, 1.2, N), rng.uniform(-2, 2, N),
                  rng.uniform(-5, 10, N)], axis=1)
    Y = true_res(X) + rng.normal(0, [0.03, 0.05], size=(N, 2))

    gp = GPConfig(max_points=50, n_iter_fit=400, device="cpu").build()
    for z, y in zip(X, Y):
        gp.observe(z, y[0], y[1])
    gp.end_episode()

    Xt = X[:60]
    Xs = (Xt - gp._x_mean) / gp._x_std
    Tt = torch.as_tensor(Xs, dtype=torch.double)
    max_diff = 0.0
    for j, tag in zip((0, 1), ("v", "w")):
        m = gp._models[j]; m.eval()
        with torch.no_grad(), gpytorch.settings.skip_posterior_variances(True):
            torch_mean = m(Tt).mean.numpy() * gp._y_scale[j] + gp._y_mean[j]
        export_mean = np.array([gp.predict(z)[j] for z in Xt])
        max_diff = max(max_diff, float(np.max(np.abs(torch_mean - export_mean))))
    flat = gp.mpc_params()
    print(f"kernel on the line -> exported kernel_id = {flat[-1]:.0f}  (3 = RBF)")
    print(f"flat mpc_params() size = {flat.size}")
    print(f"export-vs-torch max diff = {max_diff:.2e}  (should be small)")
    print("kernel module of model[0]:", type(gp._models[0].covar_module.base_kernel).__name__)
