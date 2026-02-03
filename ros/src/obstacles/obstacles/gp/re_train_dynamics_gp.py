#!/usr/bin/env python3
"""
re_train_dynamics_gp.py

Modular, episode-safe GP retraining (NO subsampling / NO stratified).
- Works with ANY input/output size (1D or 2D signals)
- Episode-safe transitions: only uses t->t+1 where episode_id[t+1] == episode_id[t]
- target_mode: "derivative" | "delta" | "next"
- Optional seed stacking: seed NPZ + new online signals

Main API (MODULAR):
    train_dynamics_gp_from_arrays(
        signals_new, dt, input_keys, output_keys,
        episode_id=..., target_mode="derivative",
        kernel="RQ", iters=300,
        seed_npz_path=..., keep_seed=True
    ) -> (gps, X, Y, y_names)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import os
import numpy as np

from gp.gp_dynamics import GPManager


# -----------------------------
# Utilities
# -----------------------------
def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is (N, D). Accepts (N,) or (N, D)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"Signal array must be 1D or 2D. Got shape {a.shape}.")


def _align_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Chop all signals to same length N = min length across keys."""
    N = min(v.shape[0] for v in signals.values())
    if N < 2:
        raise ValueError(f"Need at least 2 samples to form transitions. N={N}")
    return {k: v[:N] for k, v in signals.items()}


def _valid_mask(N: int, episode_id: Optional[np.ndarray]) -> np.ndarray:
    """valid[t] True if transition (t->t+1) stays in same episode."""
    if episode_id is None:
        return np.ones(N - 1, dtype=bool)

    ep = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    if len(ep) < N:
        raise ValueError(f"episode_id shorter than data: {len(ep)} < {N}")
    ep = ep[:N]
    return (ep[1:] == ep[:-1])


def _build_X(signals: Dict[str, np.ndarray], input_keys: List[str], valid: np.ndarray) -> np.ndarray:
    parts = []
    for k in input_keys:
        if k not in signals:
            raise KeyError(f"Missing input key '{k}'. Available keys: {list(signals.keys())}")
        parts.append(signals[k][:-1][valid])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_Y(
    signals: Dict[str, np.ndarray],
    output_keys: List[str],
    dt: float,
    mode: str,
    valid: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    mode = mode.lower().strip()
    if mode not in ("derivative", "delta", "next"):
        raise ValueError(f"Unsupported target_mode='{mode}'. Use derivative|delta|next.")

    Y_parts: List[np.ndarray] = []
    y_names: List[str] = []

    for k in output_keys:
        if k not in signals:
            raise KeyError(f"Missing output key '{k}'. Available keys: {list(signals.keys())}")

        s = signals[k]  # (N, Dk)

        if mode == "derivative":
            yk = (s[1:] - s[:-1]) / float(dt)
            suffix = "d_dt"
        elif mode == "delta":
            yk = (s[1:] - s[:-1])
            suffix = "delta"
        else:  # next
            yk = s[1:]
            suffix = "next"

        yk = yk[valid].astype(np.float32)
        Y_parts.append(yk)

        Dk = yk.shape[1]
        if Dk == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(Dk):
                y_names.append(f"{k}{j}_{suffix}")

    Y = np.concatenate(Y_parts, axis=1).astype(np.float32)
    return Y, y_names


def _maybe_check_dt(D: np.lib.npyio.NpzFile, dt: float, context: str):
    if "dt" in D.files:
        dt_file = float(np.asarray(D["dt"]).reshape(()))
        if not np.isclose(dt_file, float(dt), rtol=1e-3, atol=1e-6):
            raise ValueError(f"{context} dt mismatch: file dt={dt_file} vs requested dt={dt}")


def _load_seed_npz(
    seed_npz_path: str,
    keys_needed: List[str],
    dt: float,
    seed_episode_id: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Loads a seed NPZ with arbitrary keys_needed.
    Returns: (signals_seed_dict, episode_id_seed_array)
    """
    D = np.load(seed_npz_path)
    _maybe_check_dt(D, dt, context="Seed NPZ")

    missing = [k for k in keys_needed if k not in D.files]
    if missing:
        raise KeyError(
            f"Seed NPZ missing keys {missing}. Needed={keys_needed}. Available={list(D.files)}"
        )

    signals_seed = {k: _as_2d(D[k]) for k in keys_needed}
    signals_seed = _align_signals(signals_seed)
    Ns = next(iter(signals_seed.values())).shape[0]

    if "episode_id" in D.files:
        ep_seed = np.asarray(D["episode_id"], dtype=np.int64).reshape(-1)[:Ns]
    else:
        ep_seed = np.full(Ns, int(seed_episode_id), dtype=np.int64)

    return signals_seed, ep_seed


# --------------------------------------------------------------
# MAIN MODULAR API (this is what you wanted)
# --------------------------------------------------------------
def train_dynamics_gp_from_arrays(
    signals_new: Dict[str, np.ndarray],
    dt: float,
    input_keys: List[str],
    output_keys: List[str],
    episode_id: Optional[np.ndarray] = None,
    target_mode: str = "derivative",
    kernel: str = "RQ",
    iters: int = 300,
    seed_npz_path: Optional[str] = None,
    seed_episode_id: int = -1,
    keep_seed: bool = True,
) -> Tuple[List[GPManager], np.ndarray, np.ndarray, List[str]]:
    """
    Trains on ALL valid transitions from:
      - optional seed NPZ
      - new online signals (signals_new)

    Returns:
      gps (1 per output dim), X, Y, y_names
    """
    if not (float(dt) > 0.0):
        raise ValueError(f"dt must be > 0. Got dt={dt}")

    # Normalize new signals
    signals_new = {k: _as_2d(v) for k, v in signals_new.items()}
    signals_new = _align_signals(signals_new)
    Nn = next(iter(signals_new.values())).shape[0]

    valid_new = _valid_mask(Nn, episode_id)
    X_new = _build_X(signals_new, input_keys, valid_new)
    Y_new, y_names = _build_Y(signals_new, output_keys, dt, target_mode, valid_new)

    if X_new.shape[0] == 0:
        raise ValueError("No valid NEW transitions after episode filtering.")

    # Seed (optional)
    X_seed = Y_seed = None
    n_seed = 0
    if keep_seed and seed_npz_path is not None and os.path.exists(seed_npz_path):
        keys_needed = sorted(set(input_keys + output_keys))
        signals_seed, ep_seed = _load_seed_npz(seed_npz_path, keys_needed, dt, seed_episode_id)
        Ns = next(iter(signals_seed.values())).shape[0]
        valid_seed = _valid_mask(Ns, ep_seed)

        X_seed = _build_X(signals_seed, input_keys, valid_seed)
        Y_seed, _ = _build_Y(signals_seed, output_keys, dt, target_mode, valid_seed)
        n_seed = int(X_seed.shape[0])

        if X_seed.shape[0] == 0:
            raise ValueError("Seed NPZ produced 0 valid transitions (episode filtering removed all).")

    # Stack seed + new
    if X_seed is not None:
        X = np.vstack([X_seed, X_new]).astype(np.float32)
        Y = np.vstack([Y_seed, Y_new]).astype(np.float32)
    else:
        X = X_new.astype(np.float32)
        Y = Y_new.astype(np.float32)

    print(f"[TRAIN DATA] seed_transitions={n_seed} | new_transitions={len(X_new)} | total={len(X)}")
    print(f"[TRAIN DATA] X={X.shape} | Y={Y.shape} | mode={target_mode} | kernel={kernel} | iters={iters}")
    print(f"[TRAIN DATA] inputs={input_keys}")
    print(f"[TRAIN DATA] outputs={output_keys}")

    # Train 1 GP per output dimension
    Dy = Y.shape[1]
    gps = [GPManager(kernel=kernel, iters=iters) for _ in range(Dy)]
    for j in range(Dy):
        gps[j].fit(X, Y[:, j])
        print(f"[TRAIN] GP[{j}] trained for '{y_names[j]}' with {len(X)} samples.")

    return gps, X, Y, y_names


# --------------------------------------------------------------
# Optional convenience wrapper (NOT the main API)
# --------------------------------------------------------------
def train_pitch_rate_u_derivative_gps(
    pitch_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    episode_id: Optional[np.ndarray] = None,
    kernel: str = "RQ",
    iters: int = 300,
    seed_npz_path: Optional[str] = None,
    keep_seed: bool = True,
) -> Tuple[List[GPManager], np.ndarray, np.ndarray, List[str]]:
    """
    Convenience for your current controller-style use.
    Equivalent to:
      input_keys=["pitch","rate","u"], output_keys=["pitch","rate"], target_mode="derivative"
    """
    signals = {"pitch": pitch_arr, "rate": rate_arr, "u": u_arr}
    return train_dynamics_gp_from_arrays(
        signals_new=signals,
        dt=dt,
        input_keys=["pitch", "rate", "u"],
        output_keys=["pitch", "rate"],
        episode_id=episode_id,
        target_mode="derivative",
        kernel=kernel,
        iters=iters,
        seed_npz_path=seed_npz_path,
        keep_seed=keep_seed,
    )
