#!/usr/bin/env python3
"""
re_train_dynamics_gp.py

Episode-safe GP dynamics training (NO subsampling / NO stratified selection).

Trains 2 scalar-output GPs:
  - GP 0: d(flip)/dt
  - GP 1: d(rate)/dt

Inputs X(t) = [flip_t, rate_t, u_t]
Targets Y(t) = [(flip_{t+1}-flip_t)/dt, (rate_{t+1}-rate_t)/dt]

Key feature:
- If episode_id is provided, we only form transitions where episode_id[t+1] == episode_id[t]
  (prevents fake derivatives across resets)

Optional:
- Stack a seed NPZ dataset (kept always) + new online data.
"""

from __future__ import annotations
from typing import Optional, Tuple, List

import os
import numpy as np

from gp.gp_dynamics import GPManager


# --------------------------------------------------------------
# 1) Build full state–action → derivative dataset (episode-safe)
# --------------------------------------------------------------
def build_full_dataset(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds:
      X_full: (M, 3) where rows are [flip_t, rate_t, u_t]
      Y_full: (M, 2) where rows are [d(flip)/dt, d(rate)/dt]
    using only valid within-episode transitions (t->t+1).

    dt must be > 0.
    """
    if not (float(dt) > 0.0):
        raise ValueError(f"dt must be > 0. Got dt={dt}")

    flip_arr = np.asarray(flip_arr, dtype=np.float32).reshape(-1)
    rate_arr = np.asarray(rate_arr, dtype=np.float32).reshape(-1)
    u_arr    = np.asarray(u_arr,    dtype=np.float32).reshape(-1)

    N = min(len(flip_arr), len(rate_arr), len(u_arr))
    if N < 2:
        raise ValueError(f"Need at least 2 samples to build transitions. N={N}")

    flip_arr = flip_arr[:N]
    rate_arr = rate_arr[:N]
    u_arr    = u_arr[:N]

    # valid transitions mask: keep only same-episode t->t+1
    if episode_id is None:
        valid = np.ones(N - 1, dtype=bool)
    else:
        episode_id = np.asarray(episode_id, dtype=np.int64).reshape(-1)
        if len(episode_id) < N:
            raise ValueError(f"episode_id shorter than data: {len(episode_id)} < {N}")
        episode_id = episode_id[:N]
        valid = (episode_id[1:] == episode_id[:-1])

    # one-step pairs, filtered
    flip_t   = flip_arr[:-1][valid]
    rate_t   = rate_arr[:-1][valid]
    u_t      = u_arr[:-1][valid]
    flip_tp1 = flip_arr[1:][valid]
    rate_tp1 = rate_arr[1:][valid]

    X_full = np.stack([flip_t, rate_t, u_t], axis=1).astype(np.float32)
    Y_full = np.stack([(flip_tp1 - flip_t) / float(dt),
                       (rate_tp1 - rate_t) / float(dt)], axis=1).astype(np.float32)

    if X_full.shape[0] == 0:
        raise ValueError("No valid (t->t+1) transitions after episode_id filtering.")

    return X_full, Y_full


# --------------------------------------------------------------
# 2) Seed NPZ loader (optional)
# --------------------------------------------------------------
def _load_seed_npz(seed_npz_path: str, dt: float, seed_episode_id: int):
    """
    Expected keys in seed NPZ:
      - flip, rate, u
    Optional key:
      - dt (if present, we check it matches)
    Returns (flip_seed, rate_seed, u_seed, ep_seed)
    """
    D = np.load(seed_npz_path)

    for k in ("flip", "rate", "u"):
        if k not in D.files:
            raise KeyError(f"Seed NPZ missing key '{k}'. Found keys: {list(D.files)}")

    # Optional dt check
    if "dt" in D.files:
        dt_seed = float(np.asarray(D["dt"]).reshape(()))
        if not np.isclose(dt_seed, float(dt), rtol=1e-3, atol=1e-6):
            raise ValueError(f"Seed dt mismatch: seed dt={dt_seed} vs requested dt={dt}")

    flip_seed = np.asarray(D["flip"], dtype=np.float32).reshape(-1)
    rate_seed = np.asarray(D["rate"], dtype=np.float32).reshape(-1)
    u_seed    = np.asarray(D["u"],    dtype=np.float32).reshape(-1)

    # Mark seed as its own episode (prevents mixing boundary transitions if concatenated)
    ep_seed = np.full(len(flip_seed), int(seed_episode_id), dtype=np.int64)

    return flip_seed, rate_seed, u_seed, ep_seed


# --------------------------------------------------------------
# 3) High-level training function (NO subsampling)
# --------------------------------------------------------------
def train_dynamics_gp_from_arrays(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None,
    N_target: int = 1000,          # kept for API compatibility (ignored)
    kernel: str = "RQ",
    iters: int = 300,
    seed_npz_path: Optional[str] = None,
    seed_episode_id: int = -1,
    keep_seed: bool = True,
) -> Tuple[List[GPManager], np.ndarray, np.ndarray]:
    """
    Trains on ALL valid transitions from:
      - (optional) seed NPZ
      - (new) arrays from online logging

    NOTE: N_target is ignored on purpose (no subsampling).
    Returns: (gps, X, Y)
    """
    # Normalize + trim new arrays
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

    # Build seed transitions (optional)
    X_seed = Y_seed = None
    n_seed = 0
    if keep_seed and (seed_npz_path is not None) and os.path.exists(seed_npz_path):
        flip_seed, rate_seed, u_seed, ep_seed = _load_seed_npz(seed_npz_path, dt=dt, seed_episode_id=seed_episode_id)
        X_seed, Y_seed = build_full_dataset(flip_seed, rate_seed, u_seed, dt, episode_id=ep_seed)
        n_seed = int(X_seed.shape[0])

    # Build new transitions
    X_new, Y_new = build_full_dataset(flip_arr, rate_arr, u_arr, dt, episode_id=ep_new)
    n_new = int(X_new.shape[0])

    # Stack seed + new
    if X_seed is not None:
        X = np.vstack([X_seed, X_new]).astype(np.float32)
        Y = np.vstack([Y_seed, Y_new]).astype(np.float32)
    else:
        X = X_new.astype(np.float32)
        Y = Y_new.astype(np.float32)

    print(f"[TRAIN DATA] seed_transitions={n_seed} | new_transitions={n_new} | total={len(X)}")
    print(f"[TRAIN DATA] X shape={X.shape} | Y shape={Y.shape} | kernel={kernel} | iters={iters}")

    # Train 1 GP per output dimension
    n_output = Y.shape[1]  # should be 2
    gps = [GPManager(kernel=kernel, iters=iters) for _ in range(n_output)]

    for d in range(n_output):
        gps[d].fit(X, Y[:, d])
        print(f"[TRAIN] GP[{d}] trained with {len(X)} samples.")

    return gps, X, Y


# --------------------------------------------------------------
# 4) Optional: training directly from an NPZ
# --------------------------------------------------------------
def train_dynamics_gp_from_npz(
    npz_path: str,
    dt: Optional[float] = None,
    kernel: str = "RQ",
    iters: int = 300,
) -> Tuple[List[GPManager], np.ndarray, np.ndarray]:
    """
    Expected NPZ keys:
      - flip, rate, u
    Optional:
      - episode_id
      - dt

    If dt is None, we require NPZ to contain 'dt'.
    """
    D = np.load(npz_path)
    for k in ("flip", "rate", "u"):
        if k not in D.files:
            raise KeyError(f"NPZ missing key '{k}'. Found keys: {list(D.files)}")

    flip_arr = np.asarray(D["flip"], dtype=np.float32).reshape(-1)
    rate_arr = np.asarray(D["rate"], dtype=np.float32).reshape(-1)
    u_arr    = np.asarray(D["u"],    dtype=np.float32).reshape(-1)

    episode_id = None
    if "episode_id" in D.files:
        episode_id = np.asarray(D["episode_id"], dtype=np.int64).reshape(-1)

    if dt is None:
        if "dt" in D.files:
            dt = float(np.asarray(D["dt"]).reshape(()))
        else:
            raise ValueError("dt not provided and NPZ has no 'dt' key. Pass dt=... or store dt in NPZ.")

    return train_dynamics_gp_from_arrays(
        flip_arr=flip_arr,
        rate_arr=rate_arr,
        u_arr=u_arr,
        dt=float(dt),
        episode_id=episode_id,
        kernel=kernel,
        iters=iters,
        # no seed by default here
        seed_npz_path=None,
        keep_seed=False,
    )
