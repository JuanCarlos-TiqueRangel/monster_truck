#!/usr/bin/env python3
"""
train_svgp_modular.py

Modular SVGP training driven by configuration:
- Choose input signal keys -> build X(t)
- Choose output signal keys -> build Y(t) in modes: derivative / delta / next
- Train 1 SVGP per output dimension (scalar output per GP)

Expected NPZ: arrays keyed by your chosen signal names.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

from svgp_dynamics import SVGPManager


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainSVGPConfig:
    npz_path: str
    dt: float

    input_keys: List[str]
    output_keys: List[str]

    target_mode: str = "delta"   # "derivative" | "delta" | "next"
    N_target: int | None = None       # None -> use all
    seed: int = 123

    kernel: str = "RBF"
    iters: int = 300
    lr: float = 0.01
    batch_size: int = 256
    num_inducing: int = 128
    learn_inducing_locations: bool = True
    freeze_norm: bool = True
    device: str = "cuda"

    out_dir: str = "models"
    prefix: str = "svgp_dynamics"     # saved as f"{prefix}_{name}.pt"
    store_train_data_in_ckpt: bool = True  # enables warm-start after load()


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
    X = np.concatenate(X_parts, axis=1).astype(np.float32)
    return X  # (N-1, Dx)


def build_Y(
    signals: Dict[str, np.ndarray],
    output_keys: List[str],
    dt: float,
    mode: str,
) -> Tuple[np.ndarray, List[str]]:
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

    Y_parts: List[np.ndarray] = []
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

        yk = yk.astype(np.float32)
        Y_parts.append(yk)

        # name each column
        Dk = yk.shape[1]
        if Dk == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(Dk):
                y_names.append(f"{k}{j}_{suffix}")

    Y = np.concatenate(Y_parts, axis=1).astype(np.float32)  # (N-1, Dy)
    return Y, y_names


def maybe_subsample(
    X: np.ndarray, Y: np.ndarray, N_target: int | None, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
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
def train_svgps_from_npz(
    cfg: TrainSVGPConfig,
) -> Tuple[List[SVGPManager], np.ndarray, np.ndarray, List[str]]:
    """
    Loads signals from NPZ, builds X/Y per cfg, trains one SVGP per output dim,
    returns (gps, X_used, Y_used, y_names).
    """
    # Torch seeding for inducing-point sampling reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    all_keys = sorted(set(cfg.input_keys + cfg.output_keys))
    signals = load_signals_npz(cfg.npz_path, all_keys)
    signals = align_signals(signals)

    X_full = build_X(signals, cfg.input_keys)
    Y_full, y_names = build_Y(signals, cfg.output_keys, cfg.dt, cfg.target_mode)

    if X_full.shape[0] != Y_full.shape[0]:
        raise RuntimeError(
            f"Shape mismatch: X has {X_full.shape[0]} rows, Y has {Y_full.shape[0]} rows."
        )

    X, Y = maybe_subsample(X_full, Y_full, cfg.N_target, cfg.seed)

    N = X.shape[0]
    if N < 5:
        raise RuntimeError(f"Not enough transitions to train: N_transitions={N}")

    num_inducing_eff = int(min(cfg.num_inducing, N))
    batch_size_eff = int(min(cfg.batch_size, N))

    dev = torch.device(cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print(f"[DATA] npz={cfg.npz_path}")
    print(f"[DATA] X_full={X_full.shape}, Y_full={Y_full.shape}, using X={X.shape}, Y={Y.shape}")
    print(f"[DATA] inputs={cfg.input_keys}")
    print(f"[DATA] outputs={cfg.output_keys} (mode={cfg.target_mode})")
    print(f"[SVGP] device={dev}, iters={cfg.iters}, lr={cfg.lr}, B={batch_size_eff}, M={num_inducing_eff}")

    Dy = Y.shape[1]
    gps: List[SVGPManager] = []

    for j in range(Dy):
        gp = SVGPManager(
            kernel=cfg.kernel,
            lr=cfg.lr,
            iters=cfg.iters,
            batch_size=batch_size_eff,
            num_inducing=num_inducing_eff,
            learn_inducing_locations=cfg.learn_inducing_locations,
            device=dev,
            store_train_data_in_ckpt=cfg.store_train_data_in_ckpt,
        )
        gp.fit(X, Y[:, j], freeze_norm=cfg.freeze_norm)
        gps.append(gp)
        print(f"[TRAIN] SVGP[{j}] trained for '{y_names[j]}' with N={N}, M={num_inducing_eff}, B={batch_size_eff}")

    return gps, X, Y, y_names


def save_svgps(gps: List[SVGPManager], y_names: List[str], out_dir: str, prefix: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for j, gp in enumerate(gps):
        name = y_names[j] if j < len(y_names) else f"y{j}"
        safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
        p = out_path / f"{prefix}_{safe}.pt"
        gp.save(str(p))
        print(f"[SAVE] {p}")


# -----------------------------
# Example main (edit here)
# -----------------------------
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent        # obstacles/gp
    obstacles_dir = script_dir.parent                   # obstacles

    # file names mujoco_manual_run_flip, mujoco_manual_wheelie, mujoco_manual_run_svgp, mujoco_manual_run_obs
    #npz = obstacles_dir / "data" / "mujoco_manual_run_flip.npz"
    npz_file = cfg_params.files.ini_data_file
    npz = obstacles_dir / "data" / npz_file
    # npz = obstacles_dir / "data" / "mujoco_random_run_flip.npz"

    # input_  = ["x_pose", "linear_speed_x", "flip", "rate", "u"]
    # output_ = ["x_pose", "linear_speed_x", "flip", "rate"]

    input_=["up_z", "up_z_dot", "u"]
    output_=["up_z", "up_z_dot"]

    mode = "delta"  # derivative | delta | next

    cfg = TrainSVGPConfig(
        npz_path=str(npz),
        dt=0.02,
        input_keys=input_,
        output_keys=output_,
        target_mode=mode,
        N_target=1000,
        seed=123,

        kernel="RBF",
        iters=300,
        lr=0.01,
        batch_size=256,
        num_inducing=128,
        learn_inducing_locations=True,
        freeze_norm=True, # If is true, it does not Recompute x_mean and x_Std on the whole dataset.
        device="cuda",

        out_dir="models",
        prefix="svgp_dynamics",
        store_train_data_in_ckpt=True,
    )

    print(f"[PATH] cwd={Path.cwd()}")
    print(f"[PATH] npz_path={cfg.npz_path} exists={Path(cfg.npz_path).exists()}")

    gps, X_used, Y_used, y_names = train_svgps_from_npz(cfg)
    save_svgps(gps, y_names, cfg.out_dir, cfg.prefix)
