#!/usr/bin/env python3
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

from svgp_dynamics import SVGPManager


# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------
# Kept exactly in the same style you had
GP_PATH = f"models/{cfg_params.models.xpos_dot}"

FEATURE_NAMES = ["xpos", "xpos_dot", "pitch", "pitch_dot", "u"]

# IMPORTANT:
# The z-axis is always the output of the GP loaded from GP_PATH.
# Right now, because GP_PATH points to cfg_params.models.xpos_dot,
# the plotted output is assumed to be xpos_dot-related.
# If you want delta_pitch on z, then point GP_PATH to your delta-pitch model
# and change TARGET_LABEL accordingly.
TARGET_LABEL = "delta_xpos_dot"

# You asked for:
# x = u
# y = xpos
# z = GP output
SURFACE_X_IDX = 4   # u
SURFACE_Y_IDX = 0   # xpos

# Figure sizes
FIGSIZE_1D = (12, 10)
FIGSIZE_3D = (11, 8)

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

# Plot toggles
PLOT_1D_SLICES = True
PLOT_2D_SURFACES = True

# ---------------------------------------------------------
# RANGES YOU ASKED FOR
# ---------------------------------------------------------
# These ranges are used for plotting axes when the variable is on an axis.
# If a variable is not on an axis, these ranges are used to build sweep values.
INPUT_RANGES = {
    "xpos": (0.0, 5.0),
    "xpos_dot": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "pitch_dot": (0.2, 0.8),
    # If you want to force a custom u range too, uncomment this:
    # "u": (-1.0, 1.0),
}

# Number of sweep points for fixed variables
# 3 means low / middle / high -> total surfaces = 3*3*3 = 27
N_SWEEP_XPOS_DOT = 3
N_SWEEP_PITCH = 3
N_SWEEP_PITCH_DOT = 3

# Used only when fixed_values is None or as the base fixed value for u
U_FIXED = 1.0

# Tolerance for selecting nearby training points to overlay as black scatter
TOLERANCE_FRACTION_1D = 0.50
TOLERANCE_FRACTION_2D = 0.50

# Sampling density
N_POINTS_1D = 500
N_GRID_2D = 60


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


def _get_plot_range(X_train, feature_names, idx, input_ranges=None):
    name = feature_names[idx]
    if input_ranges is not None and name in input_ranges:
        return input_ranges[name]
    return float(np.min(X_train[:, idx])), float(np.max(X_train[:, idx]))


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
    input_ranges=None,
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
        x_min, x_max = _get_plot_range(X_train, feature_names, i, input_ranges=input_ranges)
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
        ax.scatter(
            X_train[mask, i],
            Y_train[mask],
            s=SCATTER_SIZE,
            alpha=0.7,
            color="k",
            label="nearby data",
        )
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
    input_ranges=None,
):
    """
    One 3D mean surface using two selected inputs.
    The remaining inputs are held fixed.
    """
    X_train, Y_train = _load_training_data(gp, feature_names)
    fixed_values = _get_fixed_values(X_train, fixed_values=fixed_values, u_fixed=u_fixed)

    x_min, x_max = _get_plot_range(X_train, feature_names, x_idx, input_ranges=input_ranges)
    y_min, y_max = _get_plot_range(X_train, feature_names, y_idx, input_ranges=input_ranges)

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
    gp = SVGPManager.load(GP_PATH)

    X_train, _ = _load_training_data(gp, FEATURE_NAMES)
    print_input_ranges(X_train, FEATURE_NAMES)

    print("Custom plotting ranges:")
    for name, (vmin, vmax) in INPUT_RANGES.items():
        print(f"  {name:9s} -> [{vmin}, {vmax}]")
    print()

    print(f"Surface x-axis = {FEATURE_NAMES[SURFACE_X_IDX]}")
    print(f"Surface y-axis = {FEATURE_NAMES[SURFACE_Y_IDX]}")
    print(f"Surface z-axis = {TARGET_LABEL} (the output of the loaded GP)\n")

    # Base operating point: median of the training data
    # Order: [xpos, xpos_dot, pitch, pitch_dot, u]
    base_fixed_values = np.median(X_train, axis=0).astype(float)
    base_fixed_values[4] = U_FIXED

    # Clamp base fixed values to your requested ranges where applicable
    for i, name in enumerate(FEATURE_NAMES):
        if name in INPUT_RANGES:
            vmin, vmax = INPUT_RANGES[name]
            base_fixed_values[i] = np.clip(base_fixed_values[i], vmin, vmax)

    print("Base fixed values used before sweeping:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:9s} = {base_fixed_values[i]: .6f}")
    print()

    # Optional 1D slices
    if PLOT_1D_SLICES:
        plot_gp_1d_slices(
            gp,
            feature_names=FEATURE_NAMES,
            fixed_values=base_fixed_values,
            u_fixed=U_FIXED,
            n_points=N_POINTS_1D,
            tolerance_fraction=TOLERANCE_FRACTION_1D,
            target_label=TARGET_LABEL,
            input_ranges=INPUT_RANGES,
        )

    # Build sweep values for the variables that are FIXED in the 2D surface.
    # Since you requested:
    #   x = u
    #   y = xpos
    # the fixed variables are:
    #   xpos_dot, pitch, pitch_dot
    xpos_dot_values = np.linspace(INPUT_RANGES["xpos_dot"][0], INPUT_RANGES["xpos_dot"][1], N_SWEEP_XPOS_DOT)
    pitch_values = np.linspace(INPUT_RANGES["pitch"][0], INPUT_RANGES["pitch"][1], N_SWEEP_PITCH)
    pitch_dot_values = np.linspace(INPUT_RANGES["pitch_dot"][0], INPUT_RANGES["pitch_dot"][1], N_SWEEP_PITCH_DOT)

    total_surfaces = len(xpos_dot_values) * len(pitch_values) * len(pitch_dot_values)
    print(f"Generating {total_surfaces} surface plot(s)...\n")

    if PLOT_2D_SURFACES:
        for xpos_dot_val, pitch_val, pitch_dot_val in itertools.product(
            xpos_dot_values, pitch_values, pitch_dot_values
        ):
            fixed_values = base_fixed_values.copy()
            fixed_values[1] = xpos_dot_val   # xpos_dot
            fixed_values[2] = pitch_val      # pitch
            fixed_values[3] = pitch_dot_val  # pitch_dot

            print(
                "Plotting surface with fixed values -> "
                f"xpos_dot={xpos_dot_val:.3f}, "
                f"pitch={pitch_val:.3f}, "
                f"pitch_dot={pitch_dot_val:.3f}, "
                f"u(base)={fixed_values[4]:.3f}"
            )

            plot_gp_2d_surface(
                gp,
                feature_names=FEATURE_NAMES,
                x_idx=SURFACE_X_IDX,
                y_idx=SURFACE_Y_IDX,
                fixed_values=fixed_values,
                u_fixed=U_FIXED,
                n_grid=N_GRID_2D,
                tolerance_fraction=TOLERANCE_FRACTION_2D,
                target_label=TARGET_LABEL,
                input_ranges=INPUT_RANGES,
            )


if __name__ == "__main__":
    main()
