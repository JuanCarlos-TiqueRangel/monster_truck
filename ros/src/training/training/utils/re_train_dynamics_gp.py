#!/usr/bin/env python3
"""
train_dynamics_gp.py

Utilities to:
  - build state–action → delta-state dataset for the MuJoCo flip task
  - select a compact but informative subset of samples
  - train GP dynamics models using GPManager

Typical usage (from another script):

    from train_dynamics_gp import train_dynamics_gp_from_arrays

    gps, X_sel, Y_sel = train_dynamics_gp_from_arrays(
        flip_arr, rate_arr, u_arr, dt=0.1,
        N_target=1000,
        kernel="RQ",
        iters=300
    )

`gps` will be a list of GPManager objects, one per output dim:
  - gps[0]: d(flip)/dt
  - gps[1]: d(rate)/dt
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from gp_dynamics import GPManager


# --------------------------------------------------------------
# 1) Build full state–action / delta-state dataset
#     X_full(t) = [flip_t, rate_t, u_t]
#     Y_full(t) = [d(flip)/dt, d(rate)/dt]
# --------------------------------------------------------------

def build_full_dataset(flip_arr: np.ndarray,
                       rate_arr: np.ndarray,
                       u_arr: np.ndarray,
                       dt: float,
                       episode_id: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Given time-aligned arrays flip, rate, u (length N), build:

      X_full: (N-1, 3)  where X_full[t] = [flip_t, rate_t, u_t]
      Y_full: (N-1, 2)  where Y_full[t] = [d(flip)/dt, d(rate)/dt] at t

    dt: time step between samples (e.g. control period).
    """
    flip_arr = np.asarray(flip_arr, dtype=np.float32)
    rate_arr = np.asarray(rate_arr, dtype=np.float32)
    u_arr    = np.asarray(u_arr,    dtype=np.float32)

    # Make sure all same length and chop to N
    N = min(len(flip_arr), len(rate_arr), len(u_arr))
    flip_arr = flip_arr[:N]
    rate_arr = rate_arr[:N]
    u_arr    = u_arr[:N]

    # valid transitions mask: keep only same-episode t->t+1
    if episode_id is None:
        valid = np.ones(N - 1, dtype=bool)
    else:
        episode_id = np.asarray(episode_id, dtype=np.int64)
        if len(episode_id) < N:
            raise ValueError(f"episode_id shorter than data: {len(episode_id)} < {N}")
        episode_id = episode_id[:N]
        valid = (episode_id[1:] == episode_id[:-1])


    # one-step pairs, filtered
    flip_t  = flip_arr[:-1][valid]
    rate_t  = rate_arr[:-1][valid]
    u_t     = u_arr[:-1][valid]

    flip_tp1 = flip_arr[1:][valid]
    rate_tp1 = rate_arr[1:][valid]

    X_full = np.stack([flip_t, rate_t, u_t], axis=1)
    Y_full = np.stack([(flip_tp1 - flip_t) / dt,
                       (rate_tp1 - rate_t) / dt], axis=1)
    
    if X_full.shape[0] == 0:
        raise ValueError("No valid (t->t+1) transitions after episode_id filtering.")
    
    return X_full, Y_full




def build_full_dataset_with_episode(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as build_full_dataset, but also returns ep_t for each transition row in X_full/Y_full.
    ep_t corresponds to episode_id at time t (for the (t -> t+1) transition).
    """
    flip_arr = np.asarray(flip_arr, dtype=np.float32)
    rate_arr = np.asarray(rate_arr, dtype=np.float32)
    u_arr    = np.asarray(u_arr,    dtype=np.float32)

    N = min(len(flip_arr), len(rate_arr), len(u_arr))
    flip_arr = flip_arr[:N]
    rate_arr = rate_arr[:N]
    u_arr    = u_arr[:N]

    if episode_id is None:
        episode_id = np.zeros(N, dtype=np.int64)
        valid = np.ones(N - 1, dtype=bool)
    else:
        episode_id = np.asarray(episode_id, dtype=np.int64)
        if len(episode_id) < N:
            raise ValueError(f"episode_id shorter than data: {len(episode_id)} < {N}")
        episode_id = episode_id[:N]
        valid = (episode_id[1:] == episode_id[:-1])

    flip_t   = flip_arr[:-1][valid]
    rate_t   = rate_arr[:-1][valid]
    u_t      = u_arr[:-1][valid]
    flip_tp1 = flip_arr[1:][valid]
    rate_tp1 = rate_arr[1:][valid]

    X_full = np.stack([flip_t, rate_t, u_t], axis=1)
    Y_full = np.stack([(flip_tp1 - flip_t) / dt,
                       (rate_tp1 - rate_t) / dt], axis=1)

    ep_t = episode_id[:-1][valid]  # episode label for each transition row

    if X_full.shape[0] == 0:
        raise ValueError("No valid (t->t+1) transitions after episode_id filtering.")

    return X_full, Y_full, ep_t


def _load_seed_npz(seed_npz_path: str, dt: float, seed_episode_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    D = np.load(seed_npz_path)

    for k in ("flip", "rate", "u"):
        if k not in D.files:
            raise KeyError(f"Seed NPZ missing key '{k}'. Found keys: {list(D.files)}")

    # Optional dt consistency check (recommended)
    if "dt" in D.files:
        dt_seed = float(np.asarray(D["dt"]).reshape(()))
        if not np.isclose(dt_seed, float(dt), rtol=1e-3, atol=1e-6):
            raise ValueError(f"Seed dt mismatch: seed dt={dt_seed} vs requested dt={dt}")

    flip_seed = np.asarray(D["flip"], dtype=np.float32).reshape(-1)
    rate_seed = np.asarray(D["rate"], dtype=np.float32).reshape(-1)
    u_seed    = np.asarray(D["u"],    dtype=np.float32).reshape(-1)

    # Force seed to be a distinct episode so boundary transitions are removed
    ep_seed = np.full(len(flip_seed), int(seed_episode_id), dtype=np.int64)

    return flip_seed, rate_seed, u_seed, ep_seed






# --------------------------------------------------------------
# 2) Helper selection functions
# --------------------------------------------------------------

def stratified_by_value_indices(values,
                                n_bins: int = 20,
                                max_per_bin: int = 50,
                                seed: int = 0) -> np.ndarray:
    """
    Uniform-ish coverage over 'values' (e.g., flip_rel).
    Returns a list of indices.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    v_min, v_max = float(values.min()), float(values.max())
    if v_min == v_max:
        # all values the same -> just random pick
        all_idx = np.arange(len(values))
        rng.shuffle(all_idx)
        return all_idx[:max_per_bin]

    bins = np.linspace(v_min, v_max, n_bins + 1)
    chosen = []
    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (values >= bins[i]) & (values <= bins[i + 1])
        else:
            mask = (values >= bins[i]) & (values <  bins[i + 1])

        idx_bin = np.nonzero(mask)[0]
        if len(idx_bin) == 0:
            continue
        rng.shuffle(idx_bin)
        chosen.extend(idx_bin[:max_per_bin])
    return np.array(chosen, dtype=int)


def stratified_by_abs_value_indices(values,
                                    n_bins: int = 10,
                                    max_per_bin: int = 50,
                                    seed: int = 1) -> np.ndarray:
    """
    Uniform-ish coverage over abs(values) (e.g., |rate|).
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    abs_v = np.abs(values)
    
    v_min, v_max = float(abs_v.min()), float(abs_v.max())
    if v_min == v_max:
        all_idx = np.arange(len(values))
        rng.shuffle(all_idx)
        return all_idx[:max_per_bin]

    bins = np.linspace(v_min, v_max, n_bins + 1)
    chosen = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (abs_v >= bins[i]) & (abs_v <= bins[i + 1])
        else:
            mask = (abs_v >= bins[i]) & (abs_v <  bins[i + 1])

        idx_bin = np.nonzero(mask)[0]
        if len(idx_bin) == 0:
            continue
        rng.shuffle(idx_bin)
        chosen.extend(idx_bin[:max_per_bin])

    return np.array(chosen, dtype=int)


def farthest_point_subset(X: np.ndarray, M: int = 500, seed: int = 2) -> np.ndarray:
    """
    Simple greedy farthest-point sampling in feature space.
    Picks M diverse points from X (N,D).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    N = X.shape[0]
    if M >= N:
        return np.arange(N, dtype=int)

    # Start from a random point
    first = rng.integers(0, N)
    chosen = [first]

    # Squared distances to the chosen set (init as +inf)
    d2 = np.full(N, np.inf, dtype=np.float64)

    for _ in range(1, M):
        last_idx = chosen[-1]
        diff = X - X[last_idx]           # (N, D)
        new_d2 = np.sum(diff**2, axis=1) # (N,)
        d2 = np.minimum(d2, new_d2)

        # pick the farthest point from current chosen set
        next_idx = int(np.argmax(d2))
        chosen.append(next_idx)

    return np.array(chosen, dtype=int)


def choose_selection_params(N_full: int, N_target: int = 2500) -> dict:
    """
    Decide how many points each strategy should contribute.
    """
    frac_far    = 0.4
    frac_angle  = 0.25
    frac_rate   = 0.25
    frac_stride = 0.10

    N_far    = int(frac_far    * N_target)
    N_angle  = int(frac_angle  * N_target)
    N_rate   = int(frac_rate   * N_target)
    N_stride = max(1, int(frac_stride * N_target))

    # Farthest-point
    M = N_far

    # Angle bins/limit
    n_bins_angle      = 24
    max_per_bin_angle = max(1, N_angle // n_bins_angle)

    # Rate bins/limit
    n_bins_rate       = 12
    max_per_bin_rate  = max(1, N_rate // n_bins_rate)

    # Temporal stride
    stride = max(1, int(N_full / N_stride))

    return {
        "M": M,
        "n_bins_angle": n_bins_angle,
        "max_per_bin_angle": max_per_bin_angle,
        "n_bins_rate": n_bins_rate,
        "max_per_bin_rate": max_per_bin_rate,
        "stride": stride,
    }


def select_indices(X_full: np.ndarray,
                   flip_all: np.ndarray,
                   rate_all: np.ndarray,
                   N_target: int = 1000) -> np.ndarray:
    """
    Combine:
      - stratified over flip,
      - stratified over |rate|,
      - farthest-point subset in X_full,
      - temporal stride.

    Returns unique index array.
    """
    N_full = X_full.shape[0]
    params = choose_selection_params(N_full, N_target=N_target)

    idx_angle = stratified_by_value_indices(
        flip_all,
        n_bins=params["n_bins_angle"],
        max_per_bin=params["max_per_bin_angle"],
        seed=0,
    )

    idx_rate = stratified_by_abs_value_indices(
        rate_all,
        n_bins=params["n_bins_rate"],
        max_per_bin=params["max_per_bin_rate"],
        seed=1,
    )

    idx_far = farthest_point_subset(
        X_full,
        M=params["M"],
        seed=2,
    )

    idx_stride = np.arange(0, N_full, params["stride"], dtype=int)

    idx_all = np.unique(np.concatenate([idx_angle, idx_rate, idx_far, idx_stride]))
    return idx_all


# --------------------------------------------------------------
# 3) High-level training function
# --------------------------------------------------------------

def _load_seed_npz(seed_npz_path: str, dt: float, seed_episode_id: int):
    D = np.load(seed_npz_path)

    for k in ("flip", "rate", "u", "dt"):
        if k not in D.files:
            raise KeyError(f"Seed NPZ missing key '{k}'. Found keys: {list(D.files)}")

    dt_seed = float(np.asarray(D["dt"]).reshape(()))
    if not np.isclose(dt_seed, float(dt), rtol=1e-3, atol=1e-6):
        raise ValueError(f"Seed dt mismatch: seed dt={dt_seed} vs requested dt={dt}")

    flip_seed = np.asarray(D["flip"], dtype=np.float32).reshape(-1)
    rate_seed = np.asarray(D["rate"], dtype=np.float32).reshape(-1)
    u_seed    = np.asarray(D["u"],    dtype=np.float32).reshape(-1)

    ep_seed = np.full(len(flip_seed), int(seed_episode_id), dtype=np.int64)
    return flip_seed, rate_seed, u_seed, ep_seed




def train_dynamics_gp_from_arrays(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None,
    N_target: int = 1000,
    kernel: str = "RQ",
    iters: int = 300,
    seed_npz_path: Optional[str] = None,
    seed_episode_id: int = -1,
    keep_seed: bool = True,
) -> tuple[list[GPManager], np.ndarray, np.ndarray]:
    """
    Trains GP dynamics using:
      - Seed NPZ transitions (always kept if keep_seed=True)
      - PLUS best-selected NEW transitions using your stratified selector

    Key behavior:
      - Selection (stratified+farthest+stride) is applied ONLY to NEW data
        for N_extra = N_target - N_seed points.
      - Seed is never “wasted” inside the selector.
    """

    rng = np.random.default_rng(123)

    # -------------------------------
    # 0) Normalize + trim new arrays
    # -------------------------------
    flip_arr = np.asarray(flip_arr, dtype=np.float32).reshape(-1)
    rate_arr = np.asarray(rate_arr, dtype=np.float32).reshape(-1)
    u_arr    = np.asarray(u_arr,    dtype=np.float32).reshape(-1)

    if episode_id is not None:
        ep_new = np.asarray(episode_id, dtype=np.int64).reshape(-1)
        Nn = min(len(flip_arr), len(rate_arr), len(u_arr), len(ep_new))
        flip_arr = flip_arr[:Nn]
        rate_arr = rate_arr[:Nn]
        u_arr    = u_arr[:Nn]
        ep_new   = ep_new[:Nn]
    else:
        Nn = min(len(flip_arr), len(rate_arr), len(u_arr))
        flip_arr = flip_arr[:Nn]
        rate_arr = rate_arr[:Nn]
        u_arr    = u_arr[:Nn]
        ep_new   = None

    # -------------------------------
    # 1) Build SEED transitions (kept)
    # -------------------------------
    X_seed = None
    Y_seed = None
    n_seed = 0

    if seed_npz_path is not None and keep_seed:
        flip_seed, rate_seed, u_seed, ep_seed = _load_seed_npz(
            seed_npz_path, dt=dt, seed_episode_id=seed_episode_id
        )
        # build transitions using existing episode-safe builder
        X_seed, Y_seed = build_full_dataset(
            flip_seed, rate_seed, u_seed, dt, episode_id=ep_seed
        )
        n_seed = int(X_seed.shape[0])

    # -------------------------------
    # 2) Build NEW transitions
    # -------------------------------
    X_new, Y_new = build_full_dataset(
        flip_arr, rate_arr, u_arr, dt, episode_id=ep_new
    )

    # -------------------------------
    # 3) Select best NEW points only
    # -------------------------------
    N_extra = max(0, int(N_target) - int(n_seed))

    if N_extra > 0:
        flip_new = X_new[:, 0]
        rate_new = X_new[:, 1]
        idx_new = select_indices(X_new, flip_new, rate_new, N_target=N_extra)
        rng.shuffle(idx_new)
        X_sel_new = X_new[idx_new].astype(np.float32)
        Y_sel_new = Y_new[idx_new].astype(np.float32)
    else:
        X_sel_new = np.empty((0, 3), dtype=np.float32)
        Y_sel_new = np.empty((0, 2), dtype=np.float32)

    # -------------------------------
    # 4) Final training set = seed + selected_new
    # -------------------------------
    if X_seed is not None:
        X = np.vstack([X_seed.astype(np.float32), X_sel_new])
        Y = np.vstack([Y_seed.astype(np.float32), Y_sel_new])
        print(f"Seed kept: {n_seed} | New selected: {len(X_sel_new)} | Final: {len(X)} (target≈{N_target})")

        if n_seed > N_target:
            print(f"WARNING: seed transitions ({n_seed}) exceed N_target ({N_target}). Final uses ALL seed points.")
    else:
        X = X_sel_new
        Y = Y_sel_new
        print(f"New selected: {len(X)} (target≈{N_target})")

    print("Final X shape:", X.shape)
    print("Final Y shape:", Y.shape)

    # -------------------------------
    # 5) Train one GP per output dim
    # -------------------------------
    n_output = Y.shape[1]
    gps = [GPManager(kernel=kernel, iters=iters) for _ in range(n_output)]

    for d in range(n_output):
        gps[d].fit(X, Y[:, d])
        print(f"Trained GP for d_state[{d}] with {len(X)} samples.")

    return gps, X, Y



# --------------------------------------------------------------
# 4) Optional: training directly from an NPZ data file
# --------------------------------------------------------------

def train_dynamics_gp_from_npz(
    npz_path: str,
    dt: Optional[float] = None,
    N_target: int = 1000,
    kernel: str = "RQ",
    iters: int = 300,
) -> tuple[list[GPManager], np.ndarray, np.ndarray]:
    """
    Train GPs from an NPZ file.

    Expected keys:
      - flip, rate, u
    Optional keys:
      - episode_id
      - dt          (recommended: saved by your fixed-rate logger)

    Behavior:
      - If dt is provided explicitly, it is used.
      - Else if NPZ contains 'dt', that value is used.
      - Else raises an error (to avoid silent mistakes).
    """
    D = np.load(npz_path)

    # ---- required arrays ----
    for k in ("flip", "rate", "u"):
        if k not in D.files:
            raise KeyError(f"NPZ missing key '{k}'. Found keys: {list(D.files)}")

    flip_arr = np.asarray(D["flip"], dtype=np.float32).reshape(-1)
    rate_arr = np.asarray(D["rate"], dtype=np.float32).reshape(-1)
    u_arr    = np.asarray(D["u"],    dtype=np.float32).reshape(-1)

    # ---- episode filtering optional ----
    episode_id = None
    if "episode_id" in D.files:
        episode_id = np.asarray(D["episode_id"], dtype=np.int64).reshape(-1)

    # ---- choose dt safely ----
    if dt is None:
        if "dt" in D.files:
            dt = float(np.asarray(D["dt"]).reshape(()))
        else:
            raise ValueError(
                "dt was not provided and NPZ does not contain key 'dt'. "
                "Pass dt=... or save 'dt' in the NPZ."
            )

    if not (dt > 0.0):
        raise ValueError(f"dt must be > 0. Got dt={dt}")

    # ---- basic length check ----
    N = min(len(flip_arr), len(rate_arr), len(u_arr))
    if N < 2:
        raise ValueError(f"Not enough samples to build transitions. N={N}")

    flip_arr = flip_arr[:N]
    rate_arr = rate_arr[:N]
    u_arr    = u_arr[:N]
    if episode_id is not None:
        if len(episode_id) < N:
            raise ValueError(f"episode_id shorter than data: {len(episode_id)} < {N}")
        episode_id = episode_id[:N]

    return train_dynamics_gp_from_arrays(
        flip_arr=flip_arr,
        rate_arr=rate_arr,
        u_arr=u_arr,
        dt=dt,
        episode_id=episode_id,
        N_target=N_target,
        kernel=kernel,
        iters=iters,
    )



if __name__ == "__main__":
    # Example usage (adjust npz_path and dt to your experiment):
    # npz_path = "mujoco_random_run.npz"
    npz_path = "mujoco_random_run_dt0p2.npz"
    dt = 0.1  # your ctrl_dt or average sample time

    gps, X_sel, Y_sel = train_dynamics_gp_from_npz(
        npz_path=npz_path,
        dt=dt,
        N_target=1000,
        kernel="RQ",
        iters=300,
    )
    
    # Save each output GP separately, e.g. to a models/ folder
    import os
    os.makedirs("models", exist_ok=True)

    for d, gp in enumerate(gps):
        out_path = f"models/gp_dynamics_{d}.pt"
        gp.save(out_path)
        print(f"Saved GP[{d}] to {out_path}")
