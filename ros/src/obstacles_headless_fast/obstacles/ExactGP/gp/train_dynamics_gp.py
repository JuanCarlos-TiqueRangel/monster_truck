#!/usr/bin/env python3
"""
train_gp_modular.py

Modular Exact-GP training driven by configuration:
- Choose input signal keys -> build X(t)
- Choose output signal keys -> build Y(t) in modes: derivative / delta / next
- Train 1 GP per output dimension (scalar output per GP)

Expected NPZ: arrays keyed by your chosen signal names.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gp_dynamics import GPManager

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    npz_path: str
    dt: float

    input_keys: List[str]
    output_keys: List[str]

    target_mode: str = cfg_params.gp.type_of_data   # "derivative" | "delta" | "next"
    N_target: int | None = None       # None -> use all
    seed: int = 123

    kernel: str = cfg_params.gp.kernel
    iters: int = 300

    out_dir: str = "models"
    prefix: str = "gp_dynamics"            # saved as f"{prefix}_{name}.pt"


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


def load_signals_npz(npz_path: str, keys: List[str]) -> Dict[str, np.ndarray]:
    D = np.load(npz_path)
    missing = [k for k in keys if k not in D]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {missing}. Available: {list(D.keys())}")
    return {k: _as_2d(D[k]) for k in keys}


def align_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Chop all signals to same length N = min length across keys."""
    N = min(v.shape[0] for v in signals.values())
    if N < 2:
        raise ValueError("Need at least 2 samples to form (t -> t+1) pairs.")
    return {k: v[:N] for k, v in signals.items()}


def build_X(signals: Dict[str, np.ndarray], input_keys: List[str]) -> np.ndarray:
    """
    X(t) uses values at time t, so we use [:N-1] to match targets based on (t->t+1).
    If a signal is (N, Dk), it contributes Dk columns.
    """
    X_parts = [signals[k][:-1] for k in input_keys]
    X = np.concatenate(X_parts, axis=1)
    return X  # (N-1, Dx)


def build_Y(signals: Dict[str, np.ndarray], output_keys: List[str], dt: float, mode: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build Y in one of these modes:
      - derivative: (s[t+1] - s[t]) / dt
      - delta:      (s[t+1] - s[t])
      - next:       s[t+1]

    Returns:
      Y: (N-1, Dy)
      y_names: list of names per output dimension (for saving)
    """
    mode = mode.lower().strip()
    if mode not in ("derivative", "delta", "next"):
        raise ValueError(f"Unsupported target_mode='{mode}'. Use derivative|delta|next.")

    Y_parts = []
    y_names: List[str] = []

    for k in output_keys:
        s = signals[k]  # (N, Dk)
        if mode == "derivative":
            yk = (s[1:] - s[:-1]) / float(dt)
            suffix = "d_dt"
        elif mode == "delta":
            yk = (s[1:] - s[:-1])
            suffix = "delta"
        else:  # "next"
            yk = s[1:]
            suffix = "next"

        Y_parts.append(yk)

        # name each column
        Dk = yk.shape[1]
        if Dk == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(Dk):
                y_names.append(f"{k}{j}_{suffix}")

    Y = np.concatenate(Y_parts, axis=1)  # (N-1, Dy)
    return Y, y_names


def maybe_subsample(X: np.ndarray, Y: np.ndarray, N_target: int | None, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform random subsample (NOT stratified)."""
    N = X.shape[0]
    if N_target is None or N_target >= N:
        return X, Y

    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=int(N_target), replace=False)
    rng.shuffle(idx)
    return X[idx], Y[idx]


# -----------------------------
# Training
# -----------------------------
def train_gps_from_npz(cfg: TrainConfig) -> Tuple[List[GPManager], np.ndarray, np.ndarray, List[str]]:
    """
    Loads signals from NPZ, builds X/Y per cfg, trains one GP per output dim,
    returns (gps, X_used, Y_used, y_names).
    """
    all_keys = sorted(set(cfg.input_keys + cfg.output_keys))
    signals = load_signals_npz(cfg.npz_path, all_keys)
    signals = align_signals(signals)

    X_full = build_X(signals, cfg.input_keys)
    Y_full, y_names = build_Y(signals, cfg.output_keys, cfg.dt, cfg.target_mode)

    if X_full.shape[0] != Y_full.shape[0]:
        raise RuntimeError(f"Shape mismatch: X has {X_full.shape[0]} rows, Y has {Y_full.shape[0]} rows.")

    X, Y = maybe_subsample(X_full, Y_full, cfg.N_target, cfg.seed)

    print(f"[DATA] npz={cfg.npz_path}")
    print(f"[DATA] X_full={X_full.shape}, Y_full={Y_full.shape}, using X={X.shape}, Y={Y.shape}")
    print(f"[DATA] inputs={cfg.input_keys}")
    print(f"[DATA] outputs={cfg.output_keys} (mode={cfg.target_mode})")

    Dy = Y.shape[1]
    gps = [GPManager(kernel=cfg.kernel, iters=cfg.iters) for _ in range(Dy)]

    for j in range(Dy):
        gps[j].fit(X, Y[:, j])
        print(f"[TRAIN] GP[{j}] trained for '{y_names[j]}' with {X.shape[0]} samples.")

    return gps, X, Y, y_names


def save_gps(gps: List[GPManager], y_names: List[str], out_dir: str, prefix: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for j, gp in enumerate(gps):
        name = y_names[j] if j < len(y_names) else f"y{j}"
        # safe filename
        safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
        p = out_path / f"{prefix}_{safe}.pt"
        gp.save(str(p))
        print(f"[SAVE] {p}")


# -----------------------------
# Example main (edit here)
# -----------------------------
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent        # obstacles/gp
    obstacles_dir = script_dir                   # obstacles

    npz_file = cfg_params.files.ini_data_file
    npz = obstacles_dir / "data" / npz_file

    input_  = ["xpos", "xpos_dot", "pitch", "pitch_dot", "u"]
    output_ = ["xpos", "xpos_dot", "pitch", "pitch_dot"]

    # input_=["pitch", "rate", "u"]
    # output_=["pitch", "rate"]

    # modes: derivative, delta, next
    mode = cfg_params.gp.type_of_data

    cfg = TrainConfig(
        npz_path=str(npz),
        dt=cfg_params.gp.sample_time_dt,
        input_keys=input_,
        output_keys=output_,
        target_mode=mode,
        N_target=1000,
        kernel=cfg_params.gp.kernel,
        iters=300,
        out_dir="models",
        prefix="gp_dynamics",
    )

    print(f"[PATH] cwd={Path.cwd()}")
    print(f"[PATH] npz_path={cfg.npz_path} exists={Path(cfg.npz_path).exists()}")

    gps, X_used, Y_used, y_names = train_gps_from_npz(cfg)
    save_gps(gps, y_names, cfg.out_dir, cfg.prefix)

