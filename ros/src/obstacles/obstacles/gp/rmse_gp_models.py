#!/usr/bin/env python3
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from svgp_dynamics import SVGPManager
from config.config_loader import cfg_params

title_label = 25
axes_label = 20
tick_params_label = 15
legend_label = 16


# ============================================================
# Config
# ============================================================
device = torch.device("cuda")

script_dir = Path(__file__).resolve().parent
obstacles_dir = script_dir.parent

# ------------------------------------------------------------
# Put your files here
# ------------------------------------------------------------
train_npz_path = obstacles_dir / "logs" / "train_data.npz"
eval_npz_path  = obstacles_dir / "logs" / "eval_data2.npz"

# IMPORTANT:
# This should match how the models were trained
dt = 0.1

input_keys  = ["xpos", "xpos_dot", "pitch", "pitch_dot", "u"]
output_keys = ["xpos", "xpos_dot", "pitch", "pitch_dot"]
target_mode = cfg_params.gp.type_of_data

batch_size = 4096
n_show = 100


# ============================================================
# Helpers
# ============================================================
def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"Bad shape: {a.shape}")


def load_signals_npz(npz_path: str, keys):
    D = np.load(npz_path)
    missing = [k for k in keys if k not in D]
    if missing:
        raise KeyError(f"Missing keys in NPZ: {missing}. Available keys: {list(D.keys())}")
    return {k: _as_2d(D[k]) for k in keys}


def align_signals(signals):
    N = min(v.shape[0] for v in signals.values())
    if N < 2:
        raise ValueError("Need at least 2 samples.")
    return {k: v[:N] for k, v in signals.items()}


def build_X(signals, input_keys):
    X_parts = [signals[k][:-1] for k in input_keys]
    return np.concatenate(X_parts, axis=1).astype(np.float32)


def build_Y(signals, output_keys, dt, mode="delta"):
    mode = mode.lower().strip()
    Y_parts = []
    y_names = []

    for k in output_keys:
        s = signals[k]

        if mode == "delta":
            yk = (s[1:] - s[:-1]).astype(np.float32)
            suffix = "delta"
        elif mode == "next":
            yk = s[1:].astype(np.float32)
            suffix = "next"
        elif mode == "derivative":
            yk = ((s[1:] - s[:-1]) / float(dt)).astype(np.float32)
            suffix = "d_dt"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        Y_parts.append(yk)

        if yk.shape[1] == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(yk.shape[1]):
                y_names.append(f"{k}{j}_{suffix}")

    Y = np.concatenate(Y_parts, axis=1).astype(np.float32)
    return Y, y_names


@torch.inference_mode()
def predict_all_models(gps, X_np, device, batch_size=4096):
    X_cpu = torch.tensor(X_np, dtype=torch.float32)
    pred_cols = []

    for gp in gps:
        chunks = []
        for i in range(0, X_cpu.shape[0], batch_size):
            xb = X_cpu[i:i+batch_size].to(device, non_blocking=True)
            yb = gp.predict_mean_torch(xb).reshape(-1, 1).cpu()
            chunks.append(yb)
        pred_cols.append(torch.cat(chunks, dim=0))

    Y_pred = torch.cat(pred_cols, dim=1).numpy()
    return Y_pred


def compute_metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    bias = np.mean(err)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    nrmse_std = rmse / (np.std(y_true) + 1e-12)
    nrmse_range = rmse / (np.ptp(y_true) + 1e-12)

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
        "nrmse_std": nrmse_std,
        "nrmse_range": nrmse_range,
    }


def print_metrics_block(title, y_names, Y_true, Y_pred):
    print(f"\n================ {title} ================\n")
    metrics_list = []

    for j, name in enumerate(y_names):
        m = compute_metrics(Y_true[:, j], Y_pred[:, j])
        metrics_list.append(m)

        print(f"{name}")
        print(f"  RMSE        = {m['rmse']:.6f}")
        print(f"  MAE         = {m['mae']:.6f}")
        print(f"  Bias        = {m['bias']:.6f}")
        print(f"  R2          = {m['r2']:.6f}")
        print(f"  NRMSE(std)  = {m['nrmse_std']:.6f}")
        print(f"  NRMSE(range)= {m['nrmse_range']:.6f}")
        print()

    return metrics_list


def plot_metrics_block(title, y_names, Y_true, Y_pred, n_show=500):
    show_n = min(n_show, Y_true.shape[0])

    fig, axes = plt.subplots(4, 3, figsize=(18, 18))
    fig.suptitle(title, fontsize=title_label)

    for j, name in enumerate(y_names):
        # 1) time/index plot
        ax = axes[j, 0]
        ax.plot(Y_true[:show_n, j], label="true")
        ax.plot(Y_pred[:show_n, j], label="pred")
        ax.set_title(f"{name} - first {show_n} samples", fontsize=title_label)
        ax.set_xlabel("sample", fontsize=axes_label)
        ax.set_ylabel(name, fontsize=axes_label)
        ax.tick_params(axis='both', labelsize=tick_params_label)
        ax.grid()
        if j == 0:
            ax.legend(fontsize=legend_label)

        # 2) error plot
        ax = axes[j, 1]
        err = Y_pred[:show_n, j] - Y_true[:show_n, j]
        ax.plot(err)
        ax.set_title(f"{name} error", fontsize=title_label)
        ax.set_xlabel("sample", fontsize=axes_label)
        ax.set_ylabel("pred - true", fontsize=axes_label)
        ax.tick_params(axis='both', labelsize=tick_params_label)
        ax.grid()

        # 3) scatter true vs pred
        ax = axes[j, 2]
        ax.scatter(Y_true[:, j], Y_pred[:, j], s=20, alpha=0.9)
        lo = min(Y_true[:, j].min(), Y_pred[:, j].min())
        hi = max(Y_true[:, j].max(), Y_pred[:, j].max())
        ax.plot([lo, hi], [lo, hi], "--")
        ax.set_title(f"{name}: true vs pred", fontsize=title_label)
        ax.set_xlabel("true", fontsize=axes_label)
        ax.set_ylabel("pred", fontsize=axes_label)
        ax.tick_params(axis='both', labelsize=tick_params_label)
        ax.grid()

    plt.tight_layout()
    plt.show()


def build_XY_without_cross_episode(signals, input_keys, output_keys, dt, mode="derivative"):
    ep = signals["episode_id"].reshape(-1)

    X_full = np.concatenate([signals[k][:-1] for k in input_keys], axis=1).astype(np.float32)

    Y_parts = []
    y_names = []

    for k in output_keys:
        s = signals[k]

        if mode == "delta":
            yk = (s[1:] - s[:-1]).astype(np.float32)
            suffix = "delta"
        elif mode == "next":
            yk = s[1:].astype(np.float32)
            suffix = "next"
        elif mode == "derivative":
            yk = ((s[1:] - s[:-1]) / float(dt)).astype(np.float32)
            suffix = "d_dt"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        Y_parts.append(yk)

        if yk.shape[1] == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(yk.shape[1]):
                y_names.append(f"{k}{j}_{suffix}")

    Y_full = np.concatenate(Y_parts, axis=1).astype(np.float32)

    valid = ep[1:] == ep[:-1]

    X = X_full[valid]
    Y = Y_full[valid]

    return X, Y, y_names


def evaluate_dataset(npz_path, gps, dataset_name):
    all_keys = sorted(set(input_keys + output_keys))
    # signals = load_signals_npz(str(npz_path), all_keys)
    # signals = align_signals(signals)

    # X = build_X(signals, input_keys)
    # Y_true, y_names = build_Y(signals, output_keys, dt, target_mode)

    signals = load_signals_npz(str(npz_path), all_keys + ["episode_id"])
    signals = align_signals(signals)

    X, Y_true, y_names = build_XY_without_cross_episode(
        signals, input_keys, output_keys, dt, target_mode
    )

    Y_pred = predict_all_models(gps, X, device=device, batch_size=batch_size)

    print(f"\n[DATASET] {dataset_name}")
    print(f"[DATA] npz_path = {npz_path}")
    print(f"[DATA] X shape  = {X.shape}")
    print(f"[DATA] Y shape  = {Y_true.shape}")
    print(f"[PRED] Y_pred shape = {Y_pred.shape}")


    metrics_target = print_metrics_block(
        f"{dataset_name} {target_mode.upper()} METRICS", y_names, Y_true, Y_pred
    )
    # Reconstruct next-state predictions because mode=delta
    X_state = X[:, :4]

    if target_mode == "delta":
        Y_true_next = X_state + Y_true
        Y_pred_next = X_state + Y_pred
    elif target_mode == "derivative":
        Y_true_next = X_state + dt * Y_true
        Y_pred_next = X_state + dt * Y_pred
    elif target_mode == "next":
        Y_true_next = Y_true
        Y_pred_next = Y_pred
    else:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    next_names = ["xpos_next", "xpos_dot_next", "pitch_next", "pitch_dot_next"]

    print(f"\n=========== {dataset_name} NEXT-STATE METRICS ===========\n")
    metrics_next = []

    for j, name in enumerate(next_names):
        m = compute_metrics(Y_true_next[:, j], Y_pred_next[:, j])
        metrics_next.append(m)

        print(f"{name}")
        print(f"  RMSE        = {m['rmse']:.6f}")
        print(f"  MAE         = {m['mae']:.6f}")
        print(f"  Bias        = {m['bias']:.6f}")
        print(f"  R2          = {m['r2']:.6f}")
        print(f"  NRMSE(std)  = {m['nrmse_std']:.6f}")
        print(f"  NRMSE(range)= {m['nrmse_range']:.6f}")
        print()

    plot_metrics_block(
        f"{dataset_name} - One-Step {target_mode.capitalize()} Evaluation",
        y_names, Y_true, Y_pred, n_show=n_show
    )

    print(f"\n============= {dataset_name} QUICK SUMMARY =============\n")
    for j, name in enumerate(y_names):
        m = metrics_target[j]
        print(
            f"{name:18s} | "
            f"R2={m['r2']:.4f} | "
            f"RMSE={m['rmse']:.6f} | "
            f"NRMSE(std)={m['nrmse_std']:.4f}"
        )

    return {
        "X": X,
        "Y_true": Y_true,
        "Y_pred": Y_pred,
        "Y_true_next": Y_true_next,
        "Y_pred_next": Y_pred_next,
        "y_names": y_names,
        "next_names": next_names,
        "metrics_target": metrics_target,
        "metrics_next": metrics_next,
    }


# ============================================================
# Load GP models once
# ============================================================
models_dir = script_dir / "models"

model_paths = [
    models_dir / cfg_params.models.xpos,
    models_dir / cfg_params.models.xpos_dot,
    models_dir / cfg_params.models.pitch,
    models_dir / cfg_params.models.pitch_dot,
]

gps = [SVGPManager.load(str(p), device=device) for p in model_paths]


# ============================================================
# Evaluate train and eval datasets
# ============================================================
train_results = evaluate_dataset(train_npz_path, gps, dataset_name="TRAIN")
eval_results  = evaluate_dataset(eval_npz_path, gps, dataset_name="EVAL")