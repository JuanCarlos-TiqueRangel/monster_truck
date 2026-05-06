#!/usr/bin/env python3
"""
Quick diagnostic: load each saved SVGP model and report fit quality
(RMSE and R^2) on its own training data, plus a held-out 20% split.

Run from inside the SVGP/ directory:
    python3 check_gp_fit.py
"""
import numpy as np
import torch
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gp.svgp_dynamics import SVGPManager

MODEL_DIR = HERE / "gp" / "models"
OUTPUTS = ["xpos", "xpos_dot", "pitch", "pitch_dot"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_fit(gp: SVGPManager, X: torch.Tensor, Y: np.ndarray) -> tuple[float, float]:
    with torch.no_grad():
        Y_pred = gp.predict_torch(X).cpu().numpy()
    err = Y_pred - Y
    rmse = float(np.sqrt((err ** 2).mean()))
    var = float(Y.var())
    r2 = 1.0 - (err ** 2).mean() / var if var > 0 else float("nan")
    return rmse, r2


def main():
    print(f"{'output':10s} | {'N':>5s} | {'Y_std':>7s} | "
          f"{'train RMSE':>10s} {'train R²':>8s} | "
          f"{'test RMSE':>10s} {'test R²':>8s}")
    print("-" * 82)

    for name in OUTPUTS:
        path = MODEL_DIR / f"svgp_dynamics_{name}_delta.pt"
        if not path.exists():
            print(f"{name:10s} | MISSING: {path}")
            continue

        gp = SVGPManager.load(str(path))
        gp.device = DEVICE

        X_all = gp.X_train
        Y_all = gp.Y_train.cpu().numpy()
        N = X_all.shape[0]
        y_std = float(Y_all.std())

        # Train-set fit (optimistic)
        rmse_tr, r2_tr = eval_fit(gp, X_all, Y_all)

        # Held-out split: last 20% by index (no shuffle so result is reproducible)
        cut = int(0.8 * N)
        X_te = X_all[cut:]
        Y_te = Y_all[cut:]
        if X_te.shape[0] >= 5:
            rmse_te, r2_te = eval_fit(gp, X_te, Y_te)
            test_cols = f"{rmse_te:>10.4f} {r2_te:>8.3f}"
        else:
            test_cols = f"{'-':>10s} {'-':>8s}"

        print(f"{name:10s} | {N:5d} | {y_std:7.4f} | "
              f"{rmse_tr:>10.4f} {r2_tr:>8.3f} | {test_cols}")

    print()
    print("Target: R² >= 0.95 on all outputs. R² < 0.9 means under-converged.")


if __name__ == "__main__":
    main()