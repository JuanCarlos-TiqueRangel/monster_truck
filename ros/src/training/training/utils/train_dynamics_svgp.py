#!/usr/bin/env python3
"""
train_dynamics_svgp.py

Train 2 single-output SVGP dynamics models:
  - model 0: d(flip)/dt
  - model 1: d(rate)/dt

Saves (MPPI-compatible):
  models/gp_dynamics_0.pt
  models/gp_dynamics_1.pt
"""

from __future__ import annotations
import os
import numpy as np
import torch

from svgp_dynamics import SVGPManager


def build_full_dataset(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    flip_arr = np.asarray(flip_arr, dtype=np.float32)
    rate_arr = np.asarray(rate_arr, dtype=np.float32)
    u_arr    = np.asarray(u_arr,    dtype=np.float32)

    N = min(len(flip_arr), len(rate_arr), len(u_arr))
    flip_arr = flip_arr[:N]
    rate_arr = rate_arr[:N]
    u_arr    = u_arr[:N]

    X_full = np.stack([flip_arr[:-1], rate_arr[:-1], u_arr[:-1]], axis=1).astype(np.float32)

    Y_full = np.stack(
        [
            (flip_arr[1:] - flip_arr[:-1]) / dt,
            (rate_arr[1:] - rate_arr[:-1]) / dt,
        ],
        axis=1,
    ).astype(np.float32)

    return X_full, Y_full


def train_dynamics_svgp_from_arrays(
    flip_arr: np.ndarray,
    rate_arr: np.ndarray,
    u_arr: np.ndarray,
    dt: float,
    kernel: str = "RQ",
    iters: int = 3000,
    lr: float = 0.01,
    batch_size: int = 256,
    num_inducing: int = 128,
    device: str = "cuda",
) -> tuple[list[SVGPManager], np.ndarray, np.ndarray]:
    X_full, Y_full = build_full_dataset(flip_arr, rate_arr, u_arr, dt)
    N = X_full.shape[0]
    if N < 10:
        raise RuntimeError(f"Not enough transitions to train: N_transitions={N}")

    num_inducing_eff = int(min(num_inducing, N))
    batch_size_eff   = int(min(batch_size, N))

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    gps: list[SVGPManager] = []
    for d in range(Y_full.shape[1]):
        gp = SVGPManager(
            kernel=kernel,
            lr=lr,
            iters=iters,
            batch_size=batch_size_eff,
            num_inducing=num_inducing_eff,
            learn_inducing_locations=True,
            device=dev,
            store_train_data_in_ckpt=True,  # ✅ keep checkpoints lightweight
        )
        gp.fit(X_full, Y_full[:, d], freeze_norm=True)
        gps.append(gp)
        print(f"[OK] Trained SVGP output {d} with N={N}, M={num_inducing_eff}, B={batch_size_eff}")

    return gps, X_full, Y_full


def train_dynamics_svgp_from_npz(npz_path: str, dt: float, **kwargs):
    D = np.load(npz_path)
    flip_arr = D["flip"]
    rate_arr = D["rate"]
    u_arr    = D["u"]
    return train_dynamics_svgp_from_arrays(flip_arr, rate_arr, u_arr, dt=dt, **kwargs)


if __name__ == "__main__":
    npz_path = "mujoco_random_run_dt0p1.npz"
    dt = 0.1

    gps, X, Y = train_dynamics_svgp_from_npz(
        npz_path=npz_path,
        dt=dt,
        kernel="RQ",
        iters=3000,
        lr=0.01,
        batch_size=256,
        num_inducing=128,
        device="cuda",
    )

    os.makedirs("models", exist_ok=True)
    for d, gp in enumerate(gps):
        out_path = f"models/svgp_dynamics_{d}.pt"  # ✅ MPPI-compatible name
        gp.save(out_path)
        print(f"[OK] Saved SVGP[{d}] -> {out_path}")
