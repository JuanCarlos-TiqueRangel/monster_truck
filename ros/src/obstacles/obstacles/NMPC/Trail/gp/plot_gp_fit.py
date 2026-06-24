#!/usr/bin/env python3
"""
plot_gp_fit.py
--------------
The canonical GP picture for one channel of the learned residual: measured points
(x markers), the GP posterior mean (line), and the predictive uncertainty as a
shaded +/-2 sigma band -- read against x position so it doubles as a spatial map
of where the model is confident.

The band is the GP's PREDICTIVE STANDARD DEVIATION. To draw it honestly you must
log sigma from the GP at each query point, alongside the mean:

    gp_mean, gp_var = gp.predict(X, return_var=True)     # your SVGP already returns this
    row["gp_v_dot_std"]     = float(np.sqrt(gp_var_vdot))
    row["gp_omega_dot_std"] = float(np.sqrt(gp_var_wdot))

If a *_std column is present this script uses it (true posterior band). If it is
absent it falls back to an EMPIRICAL band -- the binned spread of (measured - mean)
-- and labels it as such, because you cannot recover GP uncertainty from the mean
alone. The empirical band is fine for a quick look, not for a posterior claim.

Usage:
    python plot_gp_fit.py [trajectory.csv] [--episode N] [--channel v|omega] [--bare]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OBSTACLE_XS = [1.0, 3.0, 5.0]
OBSTACLE_HALF = 0.18
INK = "#1b1b1b"
MUTE = "#6b7280"
GOAL_C = "#1b7a3a"
MEAN_C = "#2741d6"
BAND_C = "#6f7ee8"

CHANNELS = {
    "v":     dict(r="r_v_dot", mean="gp_v_dot_pred", std="gp_v_dot_std",
                  ylabel=r"$\dot{v}$ residual  [m s$^{-2}$]",
                  title=r"GP $\dot{v}$ residual fit with predictive uncertainty"),
    "omega": dict(r="r_omega_dot", mean="gp_omega_dot_pred", std="gp_omega_dot_std",
                  ylabel=r"$\dot{\omega}$ residual  [rad s$^{-2}$]",
                  title=r"GP $\dot{\omega}$ residual fit with predictive uncertainty"
                        + "  (excluded from rollouts)"),
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#cbd0d6",
        "axes.linewidth": 1.0, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 12, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "legend.frameon": False, "legend.fontsize": 11,
    })


def parse_args(argv):
    csv_path, episode, channel, bare = None, None, "v", False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--bare":
            bare = True
        elif a == "--episode":
            i += 1; episode = int(argv[i])
        elif a.startswith("--episode="):
            episode = int(a.split("=", 1)[1])
        elif a == "--channel":
            i += 1; channel = argv[i]
        elif a.startswith("--channel="):
            channel = a.split("=", 1)[1]
        else:
            csv_path = Path(a)
        i += 1
    if channel not in CHANNELS:
        raise SystemExit(f"--channel must be one of {list(CHANNELS)}")
    return csv_path, episode, channel, bare


def empirical_band(x, err, nbins=45):
    """Binned std of (measured - mean) along x, interpolated back to x. Fallback only."""
    x = np.asarray(x, float); err = np.asarray(err, float)
    ok = np.isfinite(x) & np.isfinite(err)
    x, err = x[ok], err[ok]
    if x.size < 5:
        return np.full_like(x, np.nan)
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sig = np.full(nbins, np.nan)
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            sig[b] = err[m].std()
    good = np.isfinite(sig)
    if good.sum() < 2:
        return np.full_like(x, np.nanstd(err))
    return np.interp(x, centers[good], sig[good])


def main():
    setup_style()
    here = Path(__file__).parent
    csv_arg, episode, channel, bare = parse_args(sys.argv[1:])
    path = csv_arg if csv_arg is not None else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    spec = CHANNELS[channel]
    want = ["episode", "x", "goal_x", spec["r"], spec["mean"], spec["std"]]
    avail = set(pd.read_csv(path, nrows=0).columns)
    use = [c for c in want if c in avail]
    df = pd.read_csv(path, usecols=use)

    ep = int(df["episode"].max()) if episode is None else int(episode)
    g = df[df["episode"] == ep]
    if g.empty:
        raise SystemExit(f"episode {ep} not in CSV "
                         f"(have {int(df['episode'].min())}..{int(df['episode'].max())})")
    g = g.sort_values("x")
    x = g["x"].to_numpy()
    goal = float(g["goal_x"].iloc[0]) if "goal_x" in g else None

    meas = g[spec["r"]].to_numpy() if spec["r"] in g else None
    mean = g[spec["mean"]].to_numpy() if spec["mean"] in g else None

    have_std = spec["std"] in g.columns
    if have_std:
        sig = g[spec["std"]].to_numpy()
        band_label = r"GP $\pm 2\sigma$"
    else:
        err = (meas - mean) if (meas is not None and mean is not None) else None
        sig = empirical_band(x, err) if err is not None else None
        band_label = r"empirical $\pm 2\sigma$ (no GP $\sigma$ logged)"
        print("[note] no predictive-std column found -> drawing an EMPIRICAL band.\n"
              "       Log gp.predict(..., return_var=True) as e.g. "
              f"'{spec['std']}' for the true posterior band.")

    fig, ax = plt.subplots(figsize=(12, 6.2))

    if not bare:
        for xo in OBSTACLE_XS:
            ax.axvspan(xo - OBSTACLE_HALF, xo + OBSTACLE_HALF,
                       color="#aeb4bd", alpha=0.13, lw=0, zorder=0)
        if goal is not None:
            ax.axvline(goal, color=GOAL_C, ls=(0, (5, 3)), lw=1.4,
                       zorder=1.5, label="goal")

    if mean is not None and sig is not None:
        ax.fill_between(x, mean - 2 * sig, mean + 2 * sig,
                        color=BAND_C, alpha=0.28, lw=0, zorder=2, label=band_label)
    if meas is not None:
        ax.scatter(x, meas, s=26, c=INK, marker="x", linewidths=1.1,
                   alpha=0.85, zorder=4, label="measured residual")
    if mean is not None:
        ax.plot(x, mean, color=MEAN_C, lw=2.4, zorder=5, label="GP mean")

    ax.axhline(0.0, color="#9aa0a6", lw=1.0, zorder=1)
    ax.grid(axis="y", color="#eceef1", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("x position  [m]")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    ax.margins(x=0.01)

    # robust y-limits so the band, not a stray contact spike, sets the frame
    stack = np.concatenate([a[np.isfinite(a)] for a in
                            [meas, mean - 2 * sig if sig is not None else None,
                             mean + 2 * sig if sig is not None else None]
                            if a is not None])
    lo, hi = np.percentile(stack, [1, 99])
    pad = 0.10 * ((hi - lo) or 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(l) for l in
             ["measured residual", "GP mean", band_label, "goal"] if l in labels]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.14),
              borderaxespad=0, columnspacing=1.6, handlelength=1.8)

    fig.text(0.01, 0.005, f"episode {ep}" + ("" if episode is not None else " (most-learned)"),
             ha="left", fontsize=10, color=MUTE)
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    img_dir = here / "images"
    fname = f"gp_fit_{channel}.png"
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
