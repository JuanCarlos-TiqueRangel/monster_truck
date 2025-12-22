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
                       dt: float) -> tuple[np.ndarray, np.ndarray]:
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

    # One-step-ahead pairs: (t -> t+1)
    X_full = np.stack(
        [flip_arr[:-1], rate_arr[:-1], u_arr[:-1]],
        axis=1
    )  # (N-1, 3)

    Y_full = np.stack(
        [
            (flip_arr[1:] - flip_arr[:-1]) / dt,   # d(flip)/dt
            (rate_arr[1:] - rate_arr[:-1]) / dt,   # d(rate)/dt
        ],
        axis=1,
    )  # (N-1, 2)

    return X_full, Y_full


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
        mask = (values >= bins[i]) & (values < bins[i + 1])
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
        mask = (abs_v >= bins[i]) & (abs_v < bins[i + 1])
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

def train_dynamics_gp_from_arrays(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    N_target: int = 1000,
    kernel: str = "RQ",
    iters: int = 300,
) -> tuple[list[GPManager], np.ndarray, np.ndarray]:
    """
    High-level training function when you already have arrays.

    Inputs:
        flip_arr, rate_arr, u_arr: 1D arrays of length N
        dt:        time step between samples
        N_target:  approximate number of training points to select
        kernel:    kernel type for each GPManager ('RBF', 'Matern', 'RQ')
        iters:     training iterations for each GP

    Returns:
        gps: list of GPManager (one per output dimension)
        X:   selected input array, shape (N_sel, 3)
        Y:   selected target array, shape (N_sel, 2)
    """
    # 1) Full dataset
    X_full, Y_full = build_full_dataset(flip_arr, rate_arr, u_arr, dt)
    flip_all = X_full[:, 0]
    rate_all = X_full[:, 1]

    # 2) Index selection
    idx_all = select_indices(X_full, flip_all, rate_all, N_target=N_target)

    rng = np.random.default_rng(123)
    rng.shuffle(idx_all)

    X = X_full[idx_all]
    Y = Y_full[idx_all]

    print("Selected:", len(idx_all), " (target ≈", N_target, ")")
    print("Final X shape:", X.shape)
    print("Final Y shape:", Y.shape)

    # 3) Train GP for each output dimension
    n_output = Y.shape[1]  # e.g. 2: [d(flip)/dt, d(rate)/dt]
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
    dt: float,
    N_target: int = 1000,
    kernel: str = "RQ",
    iters: int = 300,
) -> tuple[list[GPManager], np.ndarray, np.ndarray]:
    """
    Convenience wrapper:

    Assumes NPZ file has keys:
        - 'flip'  : flip_rel over time
        - 'rate'  : pitch rate over time
        - 'u'     : control inputs

    Compatible with the logger that saved:
        t, pitch, flip, u, rate, acc, vz, vx
    """
    D = np.load(npz_path)
    flip_arr = D["flip"]
    rate_arr = D["rate"]
    u_arr    = D["u"]

    return train_dynamics_gp_from_arrays(
        flip_arr, rate_arr, u_arr,
        dt=dt,
        N_target=N_target,
        kernel=kernel,
        iters=iters,
    )


if __name__ == "__main__":
    # Example usage (adjust npz_path and dt to your experiment):
    npz_path = "mujoco_random_run.npz"
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
