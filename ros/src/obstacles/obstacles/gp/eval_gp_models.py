#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from gp_dynamics import GPManager


# -----------------------------
# Same utilities as training (must match!)
# -----------------------------
def _as_2d(a: np.ndarray) -> np.ndarray:
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
    N = min(v.shape[0] for v in signals.values())
    if N < 2:
        raise ValueError("Need at least 2 samples to form (t -> t+1) pairs.")
    return {k: v[:N] for k, v in signals.items()}


def build_X(signals: Dict[str, np.ndarray], input_keys: List[str]) -> np.ndarray:
    X_parts = [signals[k][:-1] for k in input_keys]
    return np.concatenate(X_parts, axis=1).astype(np.float32)


def build_Y(
    signals: Dict[str, np.ndarray],
    output_keys: List[str],
    dt: float,
    mode: str,
) -> Tuple[np.ndarray, List[str]]:
    mode = mode.lower().strip()
    if mode not in ("derivative", "delta", "next"):
        raise ValueError("mode must be derivative|delta|next")

    Y_parts = []
    y_names: List[str] = []

    for k in output_keys:
        s = signals[k].astype(np.float32)
        if mode == "derivative":
            yk = (s[1:] - s[:-1]) / float(dt)
            suffix = "d_dt"
        elif mode == "delta":
            yk = (s[1:] - s[:-1])
            suffix = "delta"
        else:
            yk = s[1:]
            suffix = "next"

        Y_parts.append(yk)

        Dk = yk.shape[1]
        if Dk == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(Dk):
                y_names.append(f"{k}{j}_{suffix}")

    return np.concatenate(Y_parts, axis=1).astype(np.float32), y_names


def safe_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)


# -----------------------------
# GPManager-adapted predict
# -----------------------------
def gp_predict_mean_std(gp: GPManager, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Your GPManager API:
      mean, var = gp.predict_torch(X)
    Returns numpy mean and std (sqrt(var)).
    """
    with torch.no_grad():
        mean_t, var_t = gp.predict_torch(X)  # torch tensors on gp.device
        mean = mean_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        var = var_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


# -----------------------------
# Config
# -----------------------------
@dataclass
class EvalConfig:
    dt: float = 0.1
    target_mode: str = "derivative"

    input_keys: List[str] = None
    output_keys: List[str] = None

    prefix: str = "gp_dynamics"   # must match training
    test_frac: float = 0.2
    seed: int = 42


def find_models_dir(script_dir: Path) -> Path:
    """
    Try common locations depending on where you ran training from.
    """
    obstacles_dir = script_dir.parent  # obstacles/

    candidates = [
        script_dir / "models",           # obstacles/gp/models
        obstacles_dir / "models",        # obstacles/models
        obstacles_dir / "gp" / "models", # obstacles/gp/models (duplicate, safe)
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    # fallback to first (and create eval_plots there)
    return candidates[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent     # obstacles/gp
    obstacles_dir = script_dir.parent                # obstacles
    npz_path = obstacles_dir / "data" / "mujoco_random_wheelie.npz"

    # input_=["x_pose", "linear_speed_x", "pitch", "rate", "u"],
    # output_=["x_pose", "linear_speed_x", "pitch", "rate"],

    input_=["pitch", "rate", "u"]
    output_=["pitch", "rate"]

    cfg = EvalConfig(
        dt=0.1,
        target_mode="derivative",
        input_keys=input_,
        output_keys=output_,
        prefix="gp_dynamics",
        test_frac=0.2,
        seed=42,
    )

    models_dir = find_models_dir(script_dir)
    plots_dir = models_dir / "eval_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] cwd={Path.cwd()}")
    print(f"[PATH] npz={npz_path} exists={npz_path.exists()}")
    print(f"[PATH] models_dir={models_dir} exists={models_dir.exists()}")
    print(f"[DEV ] device={device}")

    all_keys = sorted(set(cfg.input_keys + cfg.output_keys))
    signals = align_signals(load_signals_npz(str(npz_path), all_keys))
    X = build_X(signals, cfg.input_keys)
    Y, y_names = build_Y(signals, cfg.output_keys, cfg.dt, cfg.target_mode)

    N = X.shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(N)

    n_test = int(round(cfg.test_frac * N))
    test_idx = idx[:n_test]

    X_test = X[test_idx]
    Y_test = Y[test_idx]

    # contiguous slice for time-series overlay (more meaningful visually)
    slice_len = min(1000, N)
    start = max(0, N - slice_len)
    X_slice = X[start:start + slice_len]
    Y_slice = Y[start:start + slice_len]

    for j, name in enumerate(y_names):
        model_path = models_dir / f"{cfg.prefix}_{safe_name(name)}.pt"
        if not model_path.exists():
            print(f"[WARN] Missing model file for '{name}': {model_path}")
            continue

        # IMPORTANT: your load is a classmethod
        gp = GPManager.load(str(model_path), device=device)

        mu, std = gp_predict_mean_std(gp, X_test)
        y = Y_test[:, j].reshape(-1)

        rmse = float(np.sqrt(np.mean((mu - y) ** 2)))
        mae = float(np.mean(np.abs(mu - y)))

        z = (y - mu) / (std + 1e-9)
        coverage_2 = float(np.mean(np.abs(z) <= 2.0))  # ~0.95 if uncertainty is calibrated

        print(f"[{j:02d}] {name:20s}  RMSE={rmse:.5f}  MAE={mae:.5f}  cov(|z|<=2)={coverage_2:.3f}")

        # ---- Plot 1: Predicted vs True scatter ----
        plt.figure()
        plt.scatter(y, mu, s=8)
        lo = float(min(np.min(y), np.min(mu)))
        hi = float(max(np.max(y), np.max(mu)))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{name} | RMSE={rmse:.5f} MAE={mae:.5f}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{j:02d}_{safe_name(name)}__scatter.png", dpi=170)
        plt.close()

        # ---- Plot 2: Time-series overlay + ±2σ band ----
        mu_s, std_s = gp_predict_mean_std(gp, X_slice)
        y_s = Y_slice[:, j].reshape(-1)

        plt.figure()
        plt.plot(y_s, label="true")
        plt.plot(mu_s, label="pred")
        plt.fill_between(
            np.arange(len(mu_s)),
            mu_s - 2.0 * std_s,
            mu_s + 2.0 * std_s,
            alpha=0.2,
            label="±2σ",
        )
        plt.title(f"{name} time slice")
        plt.xlabel("sample")
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"{j:02d}_{safe_name(name)}__timeseries.png", dpi=170)
        plt.close()

        # ---- Plot 3: z-score histogram ----
        plt.figure()
        plt.hist(z, bins=60)
        plt.title(f"{name} z-scores | cov(|z|<=2)={coverage_2:.3f}")
        plt.xlabel("z = (y - mu) / std")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{j:02d}_{safe_name(name)}__z_hist.png", dpi=170)
        plt.close()

    print(f"[DONE] Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
