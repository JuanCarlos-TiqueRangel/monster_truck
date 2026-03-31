import torch
from obstacles.gp.svgp_dynamics import SVGPManager

from palsgp_local_model import LocalModelConfig, build_local_head_from_global
from palsgp_selector import PALSGPSelectorConfig, select_local_indices_nearest

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GP_DIR   = BASE_DIR / "gp"

gp_xpos_path = str(GP_DIR / "models" / cfg_params.models.xpos)

def test_local_matches_global_mean_reasonably():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gp_path = cfg_params.models.xpos

    gp = SVGPManager.load(gp_xpos_path, device=device)
    Z = gp.get_inducing_points_normalized()  # [M,dx]

    # random query batch in *unnormalized* feature space
    X = torch.randn(256, Z.shape[1], device=device)

    # build local indices using "nearest-to-cloud" where cloud is just X[:20]
    sel_cfg = PALSGPSelectorConfig(local_num_inducing=48, anchor_num_inducing=12)
    idx_loc = select_local_indices_nearest(Z, X[:20].detach(), sel_cfg)

    loc_cfg = LocalModelConfig(build_variance=False, eps_loc=1e-5, cholesky_float64=True)
    local = build_local_head_from_global(gp, idx_loc, loc_cfg)

    mu_g = gp.predict_mean_torch(X).detach()
    mu_l = local.predict_mean_torch(X).detach()

    assert torch.isfinite(mu_l).all()
    # Loose sanity check: local mean shouldn't be wildly off (tune threshold)
    rel = (mu_l - mu_g).abs().mean() / (mu_g.abs().mean() + 1e-6)
    assert float(rel) < 1.0
