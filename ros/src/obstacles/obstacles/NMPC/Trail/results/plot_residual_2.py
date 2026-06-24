#!/usr/bin/env python3
"""
plot_residual_vs_x.py
---------------------
Spatial view of the learned GP residual and the system state, all plotted against
x position so you can read straight off the figure what happens AS the truck
approaches and climbs the obstacle. Four stacked panels share the x axis:

  1. GP v_dot residual      vs x   (measured r_v_dot + GP prediction)
  2. v = x_dot              vs x   (forward speed)
  3. GP omega_dot residual  vs x   (measured r_omega_dot + GP prediction)
  4. omega = pitch_dot      vs x   (pitch rate)

Obstacle boxes are shaded bands; the goal is the green dashed line.

By default the LAST episode is shown (the most-learned trajectory). Pass an
episode index to pick another, or --all to overlay every episode. In --all mode
episodes are graded by recency: early/exploratory runs recede into a faint cloud,
the most-learned runs are drawn crisp and on top, so convergence is visible.

Usage:
    python plot_residual_vs_x.py [trajectory.csv] [--episode N | --all]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------- config
OBSTACLE_XS = [2.0]      # box positions in monster_truck_flip_2d.xml
OBSTACLE_HALF = 0.18               # shaded half-width drawn around each box [m]
CMAP = "viridis"                   # episode colour map
INK = "#1b1b1b"
MUTE = "#6b7280"
GOAL_C = "#1b7a3a"
ZERO_C = "#9aa0a6"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.edgecolor":    "#cbd0d6",
        "axes.linewidth":    1.0,
        "axes.grid":         False,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    12.5,
        "axes.titleweight":  "semibold",
        "axes.titlepad":     8,
        "axes.titlelocation": "left",
        "axes.labelsize":    10.5,
        "axes.labelcolor":   INK,
        "axes.titlecolor":   INK,
        "xtick.labelsize":   9.5,
        "ytick.labelsize":   9.5,
        "xtick.color":       MUTE,
        "ytick.color":       MUTE,
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "legend.frameon":    False,
        "legend.fontsize":   9,
    })


# --------------------------------------------------------------------------- helpers
def parse_args(argv):
    """Returns (csv_path_or_None, episode_or_None, show_all)."""
    csv_path, episode, show_all = None, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--all":
            show_all = True
        elif a == "--episode":
            i += 1
            episode = int(argv[i])
        elif a.startswith("--episode="):
            episode = int(a.split("=", 1)[1])
        else:
            csv_path = Path(a)
        i += 1
    return csv_path, episode, show_all


def robust_ylim(*arrays, lo=1.5, hi=98.5, pad=0.10, symmetric=False):
    """Limits from robust percentiles so a few contact spikes don't crush the scale."""
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays if a is not None])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    a, b = np.percentile(vals, [lo, hi])
    if a == b:
        a, b = float(vals.min()), float(vals.max())
    if symmetric:
        m = max(abs(a), abs(b))
        a, b = -m, m
    span = (b - a) or 1.0
    return a - span * pad, b + span * pad


def ep_grade(ep, emin, emax):
    """Recency -> (alpha, linewidth). Late episodes pop; early ones form a faint cloud."""
    t = 0.0 if emax == emin else (ep - emin) / (emax - emin)
    alpha = 0.10 + 0.72 * (t ** 1.6)
    lw = 0.55 + 1.05 * t
    return alpha, lw


def style_panel(ax, ylabel, goal, zero=False):
    for xo in OBSTACLE_XS:
        ax.axvspan(xo - OBSTACLE_HALF, xo + OBSTACLE_HALF,
                   color="#aeb4bd", alpha=0.16, lw=0, zorder=0)
    if goal is not None:
        ax.axvline(goal, color=GOAL_C, ls=(0, (5, 3)), lw=1.4, zorder=1.5)
    if zero:
        ax.axhline(0.0, color=ZERO_C, lw=1.0, zorder=1)
    ax.grid(axis="y", color="#eceef1", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)


# ----------------------------------------------------------------------------- main
def main():
    setup_style()
    here = Path(__file__).parent
    csv_arg, episode, show_all = parse_args(sys.argv[1:])
    path = csv_arg if csv_arg is not None else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    want = ["episode", "x", "goal_x", "x_dot", "pitch_dot",
            "r_v_dot", "gp_v_dot_pred", "r_omega_dot", "gp_omega_dot_pred"]
    avail = set(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in want if c not in avail]
    if missing:
        print(f"[warn] columns not in CSV (skipped): {missing}")
    df = pd.read_csv(path, usecols=[c for c in want if c in avail])
    goal = float(df["goal_x"].iloc[0]) if "goal_x" in df else None

    fig, axs = plt.subplots(4, 1, figsize=(13, 11.5), sharex=True,
                            gridspec_kw=dict(hspace=0.32))
    a1, a2, a3, a4 = axs

    # --- residual panel: spatial field f(x) the controller actually queries -------
    def panel_residual(ax, r_col, gp_col):
        clip = None
        if not show_all:
            meas = g[r_col].to_numpy() if r_col in g else None
            pred = None
            if r_col in g:
                ax.scatter(g["x"], g[r_col], s=9, c=MUTE, alpha=0.30,
                           linewidths=0, zorder=2, label="measured residual")
            if gp_col in g:
                gs = g.sort_values("x")
                pred = gs[gp_col].to_numpy()
                ax.plot(gs["x"], gs[gp_col], color="#c1121f", lw=2.0,
                        zorder=3, label="GP prediction")
            clip = robust_ylim(meas, pred, symmetric=True)
        else:
            allvals = []
            for ep, gg in groups:                       # ascending episode -> recent on top
                if gp_col in gg:
                    gs = gg.sort_values("x")             # residual read as a spatial field
                    al, lw = ep_grade(ep, emin, emax)
                    ax.plot(gs["x"], gs[gp_col], color=cmap(norm(ep)),
                            lw=lw, alpha=al, solid_capstyle="round", zorder=2)
                    allvals.append(gs[gp_col].to_numpy())
            clip = robust_ylim(*allvals, symmetric=True) if allvals else None
        if clip:
            ax.set_ylim(*clip)

    # --- state panel: trajectory in time order (x is not monotonic near the box) ---
    def panel_state(ax, col, color):
        if not show_all:
            ax.plot(g["x"], g[col], color=color, lw=2.0, zorder=3)
            return robust_ylim(g[col].to_numpy())
        allvals = []
        for ep, gg in groups:
            al, lw = ep_grade(ep, emin, emax)
            ax.plot(gg["x"], gg[col], color=cmap(norm(ep)),     # time order, not sorted
                    lw=lw, alpha=al, solid_capstyle="round", zorder=2)
            allvals.append(gg[col].to_numpy())
        return robust_ylim(*allvals)

    if show_all:
        # Group once (sorted by episode) so each of the 4 panels reuses the same slices.
        groups = sorted(df.groupby("episode", sort=False), key=lambda kv: kv[0])
        eps_sorted = np.array([ep for ep, _ in groups])
        emin, emax = int(eps_sorted.min()), int(eps_sorted.max())
        cmap = plt.get_cmap(CMAP)
        norm = plt.Normalize(emin, emax)
        title_tag = f"all {df['episode'].nunique()} episodes"
    else:
        ep_sel = int(df["episode"].max()) if episode is None else int(episode)
        g = df[df["episode"] == ep_sel]
        if g.empty:
            raise SystemExit(f"episode {ep_sel} not in CSV "
                             f"(have {int(df['episode'].min())}..{int(df['episode'].max())})")
        title_tag = f"episode {ep_sel} (most-learned)" if episode is None else f"episode {ep_sel}"

    style_panel(a1, r"$\dot{v}$ residual  [m s$^{-2}$]", goal, zero=True)
    panel_residual(a1, "r_v_dot", "gp_v_dot_pred")
    a1.set_title(r"Learned $\dot{v}$ residual vs. position")

    style_panel(a2, r"$v$  [m s$^{-1}$]", goal)
    yl = panel_state(a2, "x_dot", "#0b6fa4")
    if yl: a2.set_ylim(*yl)
    a2.set_title("Forward speed vs. position")

    style_panel(a3, r"$\dot{\omega}$ residual  [rad s$^{-2}$]", goal, zero=True)
    panel_residual(a3, "r_omega_dot", "gp_omega_dot_pred")
    a3.set_title(r"Learned $\dot{\omega}$ residual vs. position  (excluded from rollouts)")

    style_panel(a4, r"$\omega$  [rad s$^{-1}$]", goal)
    yl = panel_state(a4, "pitch_dot", "#b5651d")
    if yl: a4.set_ylim(*yl)
    a4.set_title("Pitch rate vs. position")
    a4.set_xlabel("x position  [m]")

    # --- one shared legend strip for the scene cues ------------------------------
    cues = [Patch(facecolor="#aeb4bd", alpha=0.16, label="obstacle"),
            Line2D([0], [0], color=GOAL_C, ls=(0, (5, 3)), lw=1.4, label="goal")]
    if show_all:
        cues.append(Line2D([0], [0], color=cmap(0.85), lw=1.6, label="trajectory (by episode)"))
    else:
        cues += [Line2D([0], [0], marker="o", color=MUTE, ls="", ms=5, alpha=0.6,
                        label="measured residual"),
                 Line2D([0], [0], color="#c1121f", lw=2.0, label="GP prediction")]
    a1.legend(handles=cues, loc="upper right", ncol=len(cues),
              bbox_to_anchor=(1.0, 1.32), borderaxespad=0, columnspacing=1.4,
              handlelength=1.6)

    fig.suptitle("Residual and state vs. x position", x=0.012, ha="left",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.012, 0.967, title_tag, ha="left", fontsize=10.5, color=MUTE)

    fig.subplots_adjust(left=0.085, right=0.90, top=0.92, bottom=0.06)

    if show_all:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.915, 0.06, 0.015, 0.86])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("episode", fontsize=10)
        cb.outline.set_edgecolor("#cbd0d6")
        cb.ax.tick_params(labelsize=9, color="#cbd0d6")

    img_dir = here / "images"
    fname = "residual_vs_x_all.png" if show_all else "residual_vs_x.png"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / fname
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp") / fname
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()