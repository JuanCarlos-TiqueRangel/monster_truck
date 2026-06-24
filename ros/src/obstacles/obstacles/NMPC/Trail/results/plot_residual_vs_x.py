#!/usr/bin/env python3
"""
plot_residual_vs_x.py
---------------------
Spatial view of the learned GP residual and the system state, ALL plotted against
the x position -- so you can read straight off the plot what happens AS the truck
approaches and climbs the obstacle. The four quantities are:

  1. GP v_dot residual     (measured r_v_dot + GP prediction gp_v_dot_pred)
  2. v = x_dot             (whole-system forward speed estimate)
  3. GP omega_dot residual (measured r_omega_dot + GP prediction gp_omega_dot_pred)
  4. omega = pitch_dot     (pitch rate)

The dashed vertical lines mark the obstacle boxes (parsed from the XML); the green
line marks the goal.

Three views:
  * default      -> the LAST (most-learned) episode, four stacked line panels vs x.
  * --episode N  -> the same line view for a chosen episode.
  * --all        -> overlay every episode, colour-graded by episode index.
  * --heatmap    -> PHASE-PORTRAIT heatmaps: x position (x) x pitch angle (y), colour =
                    the quantity averaged per cell, pooled over all episodes. Panels are
                    angular accel, linear accel, linear speed, angular speed -- so you see
                    'at this position + pitch, what accel/speed the truck shows'.

Usage:
    python plot_residual_vs_x.py [trajectory.csv] [--episode N | --all | --heatmap]
                                 [--nx N] [--ny N]
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


# (column, title, y/cbar label, is_signed)  -- LINE view (quantity vs x position).
# The residual channels use the GP PREDICTION (the field the controller actually uses).
QUANTITIES = [
    ("gp_v_dot_pred",     "GP v_dot residual",     "v_dot residual [m/s^2]",   True),
    ("x_dot",             "forward speed v",       "v = x_dot [m/s]",          False),
    ("gp_omega_dot_pred", "GP omega_dot residual", "omega_dot res [rad/s^2]",  True),
    ("pitch_dot",         "omega = pitch_dot",     "omega [rad/s]",            True),
]
# measured-residual companion column for the line view's faint scatter.
MEASURED = {"gp_v_dot_pred": "r_v_dot", "gp_omega_dot_pred": "r_omega_dot"}

# (column, title, colour-bar label, is_signed)  -- HEATMAP view. Phase-portrait cells of
# x position (x) x pitch angle (y); colour = the MEASURED quantity averaged per cell.
# Accelerations use the FILTERED measurement (cleaner than *_raw; swap to *_raw if wanted).
HEAT_PANELS = [
    ("omega_dot_filtered", "angular acceleration", "omega_dot [rad/s^2]",      True),
    ("v_dot_filtered",     "linear acceleration",  "v_dot [m/s^2]",            True),
    ("x_dot",              "linear speed",         "v = x_dot [m/s]",          False),
    ("pitch_dot",          "angular speed",        "omega = pitch_dot [rad/s]", True),
]
HEAT_YCOL = "pitch_deg"   # y-axis of every heatmap panel: pitch angle [deg]
HEAT_YLIM = (-180.0, 180.0)  # hardcoded heatmap pitch-angle band [deg] (lo, hi)


def obstacle_xs(xml_path):
    """Read the obstacle geom x-positions STRAIGHT from the MuJoCo XML so the markers
    always match the actual scene -- commented-out geoms are ignored by the parser, so
    this stays correct when obstacles are toggled in the XML. Picks geoms whose name
    starts with 'obs_box' or 'pole_box'."""
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as e:
        print(f"[warn] could not read obstacles from {xml_path}: {e}")
        return []
    xs = []
    for geom in root.iter("geom"):
        if geom.get("name", "").startswith(("obs_box", "pole_box")):
            try:
                xs.append(float(geom.get("pos", "").split()[0]))
            except (IndexError, ValueError):
                pass
    return sorted(set(xs))


def parse_args(argv):
    """Returns (csv_path_or_None, episode_or_None, show_all, heatmap, nx, ny)."""
    csv_path, episode, show_all, heatmap, nx, ny = None, None, False, False, 100, 80
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--all":
            show_all = True
        elif a == "--heatmap":
            heatmap = True
        elif a == "--episode":
            i += 1
            episode = int(argv[i])
        elif a.startswith("--episode="):
            episode = int(a.split("=", 1)[1])
        elif a == "--nx":
            i += 1
            nx = int(argv[i])
        elif a.startswith("--nx="):
            nx = int(a.split("=", 1)[1])
        elif a == "--ny":
            i += 1
            ny = int(argv[i])
        elif a.startswith("--ny="):
            ny = int(a.split("=", 1)[1])
        else:
            csv_path = Path(a)
        i += 1
    return csv_path, episode, show_all, heatmap, nx, ny


def mark_obstacles(ax, goal, obstacles, color="0.75"):
    for xo in obstacles:
        ax.axvline(xo, color=color, ls=":", lw=0.8, alpha=0.7)
    if goal is not None:
        ax.axvline(goal, color="lime" if color != "0.75" else "green", ls="--", lw=1.0)


def save_fig(fig, here, fname):
    img_dir = here / "images"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / fname
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp") / fname
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")


# ============================================================
# Line view (per-episode or overlay)
# ============================================================

def plot_lines(df, obstacles, goal, here, episode, show_all):
    if show_all:
        eps_all = df["episode"].to_numpy()
        cmap = plt.get_cmap("viridis")
        norm = plt.Normalize(eps_all.min(), eps_all.max())
        title_tag = f"all {df['episode'].nunique()} episodes"
    else:
        ep_sel = int(df["episode"].max()) if episode is None else int(episode)
        g = df[df["episode"] == ep_sel]
        if g.empty:
            raise SystemExit(f"episode {ep_sel} not in CSV "
                             f"(have {int(df['episode'].min())}..{int(df['episode'].max())})")
        title_tag = f"episode {ep_sel}"

    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, (col, title, ylabel, signed) in zip(axs, QUANTITIES):
        meas = MEASURED.get(col)
        if show_all:
            for ep, gg in df.groupby("episode"):
                if col in gg:
                    gs = gg.sort_values("x")
                    ax.plot(gs["x"], gs[col], color=cmap(norm(ep)), lw=0.8, alpha=0.7)
        else:
            if meas and meas in g:   # faint measured-residual scatter behind the GP line
                ax.scatter(g["x"], g[meas], s=8, c="0.6", alpha=0.5, label="measured residual")
            if col in g:
                gs = g.sort_values("x") if meas else g
                ax.plot(gs["x"], gs[col], color="C3" if meas else "C0", lw=1.5,
                        label="GP prediction" if meas else None)
            if meas:
                ax.legend(fontsize=8, loc="best")
        if signed:
            ax.axhline(0.0, color="k", lw=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        mark_obstacles(ax, goal, obstacles)
        ax.grid(True, alpha=0.4)
    axs[-1].set_xlabel("x position [m]")

    fig.suptitle(f"Residual + state vs x position ({title_tag})")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    if show_all:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axs, label="episode", fraction=0.025, pad=0.01)

    save_fig(fig, here, "residual_vs_x_all.png" if show_all else "residual_vs_x.png")


# ============================================================
# Heatmap view (phase portrait: x position x pitch angle, colour = the quantity)
# ============================================================

def robust_lims(grid, signed):
    """2-98 percentile colour limits so contact-impulse outliers don't wash out the
    colours; symmetric about 0 for signed quantities."""
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return None, None
    lo, hi = np.percentile(finite, [2, 98])
    if signed:
        m = float(max(abs(lo), abs(hi))) or 1.0
        return -m, m
    return float(lo), float(hi)


def cell_mean(x, y, c, xr, yr, nx, ny):
    """Mean of c over an (x, y) grid via sum/count histograms. Empty cells -> NaN.
    Returns mean[ny, nx] (already oriented for imshow), x_edges, y_edges."""
    s, xe, ye = np.histogram2d(x, y, bins=[nx, ny], range=[xr, yr], weights=c)
    n, _, _ = np.histogram2d(x, y, bins=[nx, ny], range=[xr, yr])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
    mean[n == 0] = np.nan
    return mean.T, xe, ye          # .T -> rows=y, cols=x for imshow(origin="lower")


def plot_heatmaps(df, obstacles, goal, here, nx, ny):
    """Phase-portrait heatmaps: every sample (all episodes) binned into an
    x position (x) x pitch angle (y) grid, colour = the quantity averaged in each cell.
    So you read 'at this position + pitch, what acceleration/speed does the truck show'.
    Panels: angular accel, linear accel, linear speed, angular speed."""
    if HEAT_YCOL not in df:
        raise SystemExit(f"'{HEAT_YCOL}' column missing from CSV; cannot build phase heatmap.")
    n_ep = df["episode"].nunique()
    x = df["x"].to_numpy()
    pitch = df[HEAT_YCOL].to_numpy()
    xr = [float(np.nanmin(x)), float(np.nanmax(x))]
    yr = list(HEAT_YLIM)   # hardcoded pitch-angle band (edit HEAT_YLIM at top of file)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, title, cbar_label, signed) in zip(axs.ravel(), HEAT_PANELS):
        if col not in df:
            ax.set_visible(False)
            continue
        c = df[col].to_numpy()
        m = np.isfinite(x) & np.isfinite(pitch) & np.isfinite(c)
        mean, xe, ye = cell_mean(x[m], pitch[m], c[m], xr, yr, nx, ny)
        cmap = plt.get_cmap("coolwarm" if signed else "viridis").copy()
        cmap.set_bad("0.92")                        # unvisited (x, pitch) cells -> light grey
        vmin, vmax = robust_lims(mean, signed)
        im = ax.imshow(np.ma.masked_invalid(mean), origin="lower", aspect="auto",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.axhline(0.0, color="0.4", lw=0.8)        # flat-pitch reference
        mark_obstacles(ax, goal, obstacles, color="k")
        ax.set_title(title)
        ax.set_xlabel("x position [m]")
        ax.set_ylabel("pitch angle [deg]")
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.02)

    fig.suptitle(f"Phase-portrait heatmaps  (x position x pitch angle, all {n_ep} episodes)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save_fig(fig, here, "residual_heatmap.png")


def main():
    here = Path(__file__).parent
    csv_arg, episode, show_all, heatmap, nx, ny = parse_args(sys.argv[1:])
    path = csv_arg if csv_arg is not None else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    # Obstacle markers come from the XML one level up (Trail/monster_truck_flip_2d.xml).
    obstacles = obstacle_xs(here.parent / "monster_truck_flip_2d.xml")

    # The CSV can be hundreds of MB (100 episodes), so read ONLY the columns we plot
    # (line view: residuals + speeds; heatmap view: pitch angle + measured accelerations).
    want = ["episode", "x", "goal_x", "x_dot", "pitch_dot", "pitch_deg",
            "v_dot_filtered", "omega_dot_filtered",
            "r_v_dot", "gp_v_dot_pred", "r_omega_dot", "gp_omega_dot_pred"]
    avail = set(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in want if c not in avail]
    if missing:
        print(f"[warn] columns not in CSV (skipped): {missing}")
    df = pd.read_csv(path, usecols=[c for c in want if c in avail])

    goal = float(df["goal_x"].iloc[0]) if "goal_x" in df else None

    if heatmap:
        plot_heatmaps(df, obstacles, goal, here, nx, ny)
    else:
        plot_lines(df, obstacles, goal, here, episode, show_all)
    plt.show()


if __name__ == "__main__":
    main()
