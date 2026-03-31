import torch
from dataclasses import dataclass


@dataclass
class PALSGPSelectorConfig:
    local_num_inducing: int = 48
    anchor_num_inducing: int = 12
    selector_mode: str = "nearest"   # "nearest" only for now


def _ordered_unique_long(idx: torch.Tensor) -> torch.Tensor:
    seen = set()
    ordered = []
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return torch.tensor(ordered, dtype=torch.long, device=idx.device)


def farthest_point_indices(Z: torch.Tensor, m: int) -> torch.Tensor:
    """
    Greedy farthest-point selection on rows of Z.
    Z: [M, D]
    returns: [m]
    """
    M = int(Z.shape[0])
    m = min(int(m), M)
    if m <= 0:
        return torch.empty(0, dtype=torch.long, device=Z.device)
    if m >= M:
        return torch.arange(M, dtype=torch.long, device=Z.device)

    first = 0
    selected = [first]

    d2 = torch.sum((Z - Z[first:first+1]) ** 2, dim=1)

    for _ in range(1, m):
        nxt = int(torch.argmax(d2).item())
        selected.append(nxt)
        d2_new = torch.sum((Z - Z[nxt:nxt+1]) ** 2, dim=1)
        d2 = torch.minimum(d2, d2_new)

    return torch.tensor(selected, dtype=torch.long, device=Z.device)


def select_anchor_indices(Z_glob: torch.Tensor, m_anchor: int) -> torch.Tensor:
    return farthest_point_indices(Z_glob, m_anchor)


def select_local_indices_nearest(
    Z_glob: torch.Tensor,
    rollout_features: torch.Tensor,
    config: PALSGPSelectorConfig,
    anchor_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    PALSGP-lite local subset:
      Z_loc = anchors + nearest non-anchor points to rollout cloud

    IMPORTANT:
      Z_glob and rollout_features must already be in the SAME space.
      For your use case that should be normalized GP-input space.
    """
    if config.selector_mode != "nearest":
        raise NotImplementedError(
            f"selector_mode={config.selector_mode!r} is not implemented yet. "
            "Use 'nearest' for PALSGP-lite."
        )

    M = int(Z_glob.shape[0])
    M_loc = min(int(config.local_num_inducing), M)
    M_anchor = min(int(config.anchor_num_inducing), M_loc)

    if anchor_idx is None or int(anchor_idx.numel()) != M_anchor:
        anchor_idx = select_anchor_indices(Z_glob, M_anchor)
    else:
        anchor_idx = anchor_idx.to(device=Z_glob.device, dtype=torch.long)

    if M_loc == M_anchor:
        return anchor_idx.clone()

    mask = torch.ones(M, dtype=torch.bool, device=Z_glob.device)
    mask[anchor_idx] = False
    non_anchor_idx = torch.arange(M, device=Z_glob.device, dtype=torch.long)[mask]
    Z_non = Z_glob[non_anchor_idx]

    # distance from each non-anchor inducing point to rollout cloud
    # [M_non, H]
    dmat = torch.cdist(Z_non, rollout_features)
    dmin = dmat.min(dim=1).values

    k = min(M_loc - M_anchor, int(non_anchor_idx.numel()))
    picked_rel = torch.topk(-dmin, k=k).indices
    picked = non_anchor_idx[picked_rel]

    idx_loc = torch.cat([anchor_idx, picked], dim=0)
    idx_loc = _ordered_unique_long(idx_loc)

    # if duplicates reduced count, fill with nearest remaining non-anchor points
    if int(idx_loc.numel()) < M_loc:
        missing = M_loc - int(idx_loc.numel())
        chosen_mask = torch.ones(non_anchor_idx.numel(), dtype=torch.bool, device=Z_glob.device)
        chosen_mask[picked_rel] = False
        remain_idx = non_anchor_idx[chosen_mask]
        remain_d = dmin[chosen_mask]
        if int(remain_idx.numel()) > 0:
            fill_k = min(missing, int(remain_idx.numel()))
            fill_rel = torch.topk(-remain_d, k=fill_k).indices
            fill = remain_idx[fill_rel]
            idx_loc = _ordered_unique_long(torch.cat([idx_loc, fill], dim=0))

    return idx_loc[:M_loc]