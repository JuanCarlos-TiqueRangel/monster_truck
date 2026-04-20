import torch
from dataclasses import dataclass


@dataclass
class LocalModelConfig:
    eps_loc: float = 1e-5
    cholesky_float64: bool = True
    build_variance: bool = True


class LocalSparseHead:
    """
    Solve-specific local predictor built from one global GP head.

    Z_loc is stored in normalized GP-input space.
    predict_mean / predict_var accept UNNORMALIZED feature inputs X,
    and normalize internally using gp_head.X_mean / X_std.
    """
    def __init__(
        self,
        Z_loc_n: torch.Tensor,
        L: torch.Tensor,
        mw: torch.Tensor,
        C: torch.Tensor | None,
        gp_head,
    ):
        self.Z_loc_n = Z_loc_n
        self.L = L
        self.mw = mw
        self.C = C
        self.gp_head = gp_head

    def _normalize(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.gp_head.X_mean) / self.gp_head.X_std

    def predict_mean(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=torch.float32, device=self.Z_loc_n.device)
        Xn = self._normalize(X)

        Kzx = self.gp_head.kernel_zz_normalized(self.Z_loc_n, Xn)   # [Mloc, B]
        a = torch.linalg.solve_triangular(self.L, Kzx, upper=False)
        return (a * self.mw[:, None]).sum(dim=0)

    def predict_var(self, X: torch.Tensor) -> torch.Tensor:
        if self.C is None:
            raise RuntimeError("LocalSparseHead was built without variance support.")

        X = torch.as_tensor(X, dtype=torch.float32, device=self.Z_loc_n.device)
        Xn = self._normalize(X)

        Kzx = self.gp_head.kernel_zz_normalized(self.Z_loc_n, Xn)
        a = torch.linalg.solve_triangular(self.L, Kzx, upper=False)

        kdiag = self.gp_head.kernel_diag_normalized(Xn)
        term1 = (a ** 2).sum(dim=0)

        Ct_a = self.C.T @ a
        term2 = (Ct_a ** 2).sum(dim=0)

        return torch.clamp(kdiag - term1 + term2, min=1e-9)


@torch.no_grad()
def build_local_head_from_global(gp_head, idx_loc: torch.Tensor, config: LocalModelConfig) -> LocalSparseHead:
    """
    Build PALSGP-lite local model from one global head.
    """
    device = gp_head.device
    idx_loc = idx_loc.to(device=device, dtype=torch.long)

    Z_glob_n = gp_head.get_inducing_points_normalized()
    Z_loc_n = Z_glob_n[idx_loc]

    K_loc = gp_head.kernel_zz_normalized(Z_loc_n, Z_loc_n)

    if config.cholesky_float64:
        K_loc = K_loc.double()
        eye = torch.eye(K_loc.shape[0], device=device, dtype=K_loc.dtype)
        L = torch.linalg.cholesky(K_loc + float(config.eps_loc) * eye).float()
    else:
        eye = torch.eye(K_loc.shape[0], device=device, dtype=K_loc.dtype)
        L = torch.linalg.cholesky(K_loc + float(config.eps_loc) * eye)

    mu_loc, var_loc = gp_head.predict_latent_mean_var_normalized(Z_loc_n)
    mu_loc = mu_loc.float()
    sigma_loc = torch.sqrt(torch.clamp(var_loc.float(), min=1e-9))

    # mw = L^{-1} mu_loc
    mw = torch.linalg.solve_triangular(L, mu_loc[:, None], upper=False).squeeze(-1)

    C = None
    if config.build_variance:
        C = torch.linalg.solve_triangular(
            L,
            torch.diag(sigma_loc),
            upper=False,
        )

    return LocalSparseHead(
        Z_loc_n=Z_loc_n,
        L=L,
        mw=mw,
        C=C,
        gp_head=gp_head,
    )