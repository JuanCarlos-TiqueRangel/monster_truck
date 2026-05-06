#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

#from svgp_dynamics import SVGPManager
from gp_dynamics import GPManager


# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
GP_PATH = f"models/{cfg_params.models.xpos_dot}"
FEATURE_NAMES = ["xpos", "xpos_dot", "pitch", "pitch_dot", "u"]

# Change this label to match what your .pt predicts.
# Examples:
TARGET_LABEL = "delta_pitch_dot"
# TARGET_LABEL = "pitch_dot_next"
# TARGET_LABEL = "delta_pitch"
# TARGET_LABEL = "delta_xpos_dot"

# For the surface plot, choose which two inputs vary.
# Example for xpos vs u:
# SURFACE_X_IDX = 0   # xpos
# SURFACE_Y_IDX = 4   # u
SURFACE_X_IDX = 3   # u
SURFACE_Y_IDX = 0   # xpos

u_fixed = 0.5

# Figure sizes
FIGSIZE_1D = (12, 10)
FIGSIZE_3D = (10, 7)

# Font sizes
scale_size = 1.1
TITLE_FONTSIZE = 18 * scale_size
LABEL_FONTSIZE = 16 * scale_size
TICK_FONTSIZE = 14 * scale_size
LEGEND_FONTSIZE = 13 * scale_size
SUPTITLE_FONTSIZE = 18 * scale_size
COLORBAR_LABEL_FONTSIZE = 14 * scale_size
COLORBAR_TICK_FONTSIZE = 12 * scale_size

# Line/scatter sizes
LINEWIDTH_MEAN = 2.0
SCATTER_SIZE = 18




# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_1d(y, name="array"):
    y = _to_numpy(y)
    if y.ndim == 1:
        return y
    if y.ndim == 2 and y.shape[1] == 1:
        return y[:, 0]
    raise ValueError(f"{name} must have shape (N,) or (N, 1). Got {y.shape}.")


def _load_training_data(gp, feature_names):
    X_train, Y_train = gp.dataset()
    X_train = _to_numpy(X_train)
    Y_train = _to_1d(Y_train, name="Y_train")

    if X_train.ndim != 2:
        raise ValueError(f"X_train must be 2D. Got shape {X_train.shape}.")

    if X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} inputs {feature_names}, "
            f"but X_train has shape {X_train.shape}."
        )

    return X_train, Y_train


def _predict_gp(gp, X_query):
    mean_t, var_t = gp.predict_torch(X_query)
    mean = _to_1d(mean_t, name="predicted mean")
    var = _to_1d(var_t, name="predicted variance")
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, std


def _get_fixed_values(X_train, fixed_values=None, u_fixed=0.0):
    if fixed_values is None:
        vals = np.median(X_train, axis=0).astype(float)
        vals[-1] = u_fixed
        return vals

    vals = np.asarray(fixed_values, dtype=float)
    if vals.shape != (X_train.shape[1],):
        raise ValueError(
            f"fixed_values must have shape ({X_train.shape[1]},). Got {vals.shape}."
        )
    return vals


def _build_nearby_mask(X_train, fixed_values, varying_indices, tolerance_fraction):
    mask = np.ones(X_train.shape[0], dtype=bool)
    varying_indices = set(varying_indices)

    for j in range(X_train.shape[1]):
        if j in varying_indices:
            continue

        data_min = float(np.min(X_train[:, j]))
        data_max = float(np.max(X_train[:, j]))
        width = tolerance_fraction * max(data_max - data_min, 1e-8)
        mask &= np.abs(X_train[:, j] - fixed_values[j]) <= width

    return mask


def print_input_ranges(X_train, feature_names):
    print("\nInput ranges from training data:")
    for i, name in enumerate(feature_names):
        x_min = float(np.min(X_train[:, i]))
        x_max = float(np.max(X_train[:, i]))
        x_med = float(np.median(X_train[:, i]))
        print(f"  {name:9s} -> min={x_min: .6f}, max={x_max: .6f}, median={x_med: .6f}")
    print()


# ---------------------------------------------------------
# PLOTS
# ---------------------------------------------------------
def plot_gp_1d_slices(
    gp,
    feature_names,
    fixed_values=None,
    u_fixed=0.0,
    n_points=200,
    tolerance_fraction=0.10,
    target_label="GP output",
):
    """
    One figure with 5 subplots:
      - xpos       -> GP output
      - xpos_dot   -> GP output
      - pitch      -> GP output
      - pitch_dot  -> GP output
      - u          -> GP output

    The other inputs are held fixed.
    """
    X_train, Y_train = _load_training_data(gp, feature_names)
    fixed_values = _get_fixed_values(X_train, fixed_values=fixed_values, u_fixed=u_fixed)

    n_features = len(feature_names)
    ncols = 2
    nrows = math.ceil(n_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=FIGSIZE_1D)
    axes = np.atleast_1d(axes).ravel()

    for i, name in enumerate(feature_names):
        ax = axes[i]
        x_min = float(np.min(X_train[:, i]))
        x_max = float(np.max(X_train[:, i]))
        x_grid = np.linspace(x_min, x_max, n_points)

        X_query = np.tile(fixed_values, (n_points, 1))
        X_query[:, i] = x_grid

        mean, std = _predict_gp(gp, X_query)
        mask = _build_nearby_mask(
            X_train=X_train,
            fixed_values=fixed_values,
            varying_indices=[i],
            tolerance_fraction=tolerance_fraction,
        )

        ax.plot(x_grid, mean, lw=LINEWIDTH_MEAN, label="GP mean")
        ax.fill_between(x_grid, mean - 2.0 * std, mean + 2.0 * std, alpha=0.25, label="±2σ")
        ax.scatter(X_train[mask, i], Y_train[mask], s=SCATTER_SIZE, alpha=0.7, color="k", label="nearby data")
        ax.set_xlabel(name, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(target_label, fontsize=LABEL_FONTSIZE)
        ax.set_title(f"{target_label} vs {name}", fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(True)
        ax.legend(fontsize=LEGEND_FONTSIZE)

    for j in range(n_features, len(axes)):
        fig.delaxes(axes[j])

    fixed_text = ", ".join(f"{n}={v:.3f}" for n, v in zip(feature_names, fixed_values))
    fig.suptitle(
        f"1D GP slices\nFixed values: {fixed_text}",
        y=1.02,
        fontsize=SUPTITLE_FONTSIZE,
    )
    plt.tight_layout()
    plt.show()



def plot_gp_2d_surface(
    gp,
    feature_names,
    x_idx=0,
    y_idx=4,
    fixed_values=None,
    u_fixed=0.0,
    n_grid=60,
    tolerance_fraction=0.10,
    target_label="GP output",
):
    """
    One 3D mean surface using two selected inputs.
    The remaining inputs are held fixed.
    """
    X_train, Y_train = _load_training_data(gp, feature_names)
    fixed_values = _get_fixed_values(X_train, fixed_values=fixed_values, u_fixed=u_fixed)

    x_min = float(np.min(X_train[:, x_idx]))
    x_max = float(np.max(X_train[:, x_idx]))
    y_min = float(np.min(X_train[:, y_idx]))
    y_max = float(np.max(X_train[:, y_idx]))

    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    Xg, Yg = np.meshgrid(x_grid, y_grid)

    X_query = np.tile(fixed_values, (n_grid * n_grid, 1))
    X_query[:, x_idx] = Xg.ravel()
    X_query[:, y_idx] = Yg.ravel()

    mean, _ = _predict_gp(gp, X_query)
    Z = mean.reshape(Xg.shape)

    mask = _build_nearby_mask(
        X_train=X_train,
        fixed_values=fixed_values,
        varying_indices=[x_idx, y_idx],
        tolerance_fraction=tolerance_fraction,
    )

    fig = plt.figure(figsize=FIGSIZE_3D)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(Xg, Yg, Z, cmap="viridis", alpha=0.88, linewidth=0)
    ax.scatter(
        X_train[mask, x_idx],
        X_train[mask, y_idx],
        Y_train[mask],
        color="k",
        s=SCATTER_SIZE,
        alpha=0.8,
        label="nearby data",
    )

    fixed_text = ", ".join(
        f"{feature_names[i]}={fixed_values[i]:.3f}"
        for i in range(len(feature_names))
        if i not in [x_idx, y_idx]
    )

    ax.set_xlabel(feature_names[x_idx], fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.set_ylabel(feature_names[y_idx], fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.set_zlabel(target_label, fontsize=LABEL_FONTSIZE, labelpad=12)
    ax.set_title(
        f"{target_label} surface: {feature_names[x_idx]} vs {feature_names[y_idx]}\n"
        f"Fixed: {fixed_text}",
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="z", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.75, pad=0.10)
    cbar.set_label("GP mean", fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    gp = GPManager.load(GP_PATH)

    # If you already know a specific operating point, put it here.
    # Order: [xpos, xpos_dot, pitch, pitch_dot, u]
    # Example:
    # fixed_values = [0.0, 0.0, 0.0, 0.0, 0.0]
    fixed_values = None

    # If fixed_values=None, the code uses the median of the dataset for the 4 states,
    # and forces u = u_fixed.

    X_train, _ = _load_training_data(gp, FEATURE_NAMES)
    print_input_ranges(X_train, FEATURE_NAMES)
    print(f"Surface x-axis ({FEATURE_NAMES[SURFACE_X_IDX]}) range comes from training data min/max.")
    print(f"Surface y-axis ({FEATURE_NAMES[SURFACE_Y_IDX]}) range comes from training data min/max.\n")
    print(f"If fixed_values=None, xpos fixed value is median(X_train[:, 0]) = {np.median(X_train[:, 0]):.6f}")
    print(f"If fixed_values=None, u fixed value is u_fixed = {u_fixed:.6f}\n")

    plot_gp_1d_slices(
        gp,
        feature_names=FEATURE_NAMES,
        fixed_values=fixed_values,
        u_fixed=u_fixed,
        n_points=500,
        tolerance_fraction=0.50,
        target_label=TARGET_LABEL,
    )

    plot_gp_2d_surface(
        gp,
        feature_names=FEATURE_NAMES,
        x_idx=SURFACE_X_IDX,
        y_idx=SURFACE_Y_IDX,
        fixed_values=fixed_values,
        u_fixed=u_fixed,
        n_grid=60,
        tolerance_fraction=0.5,
        target_label=TARGET_LABEL,
    )


if __name__ == "__main__":
    main()