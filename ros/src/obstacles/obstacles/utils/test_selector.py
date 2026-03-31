import torch
from palsgp_selector import PALSGPSelectorConfig, select_local_indices_nearest

def test_selector_counts_and_uniqueness():
    torch.manual_seed(0)
    Z = torch.randn(256, 5)
    cloud = torch.randn(20, 5)
    cfg = PALSGPSelectorConfig(local_num_inducing=48, anchor_num_inducing=12)
    idx = select_local_indices_nearest(Z, cloud, cfg)
    assert idx.shape == (48,)
    assert idx.unique().numel() == 48
    assert (idx >= 0).all() and (idx < 256).all()
