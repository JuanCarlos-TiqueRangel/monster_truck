#!/usr/bin/env python3
"""
plot_gp_fit.py
--------------
The canonical GP picture for one channel of the learned residual: the measured
observations (x markers), the GP posterior mean (line), and the predictive
uncertainty as a +/-2 sigma band, read against x position.

Three things about THIS log that the plot has to respect (discovered from the data):

  * r_v_dot / r_omega_dot are ZERO-ORDER HELD: a new residual observation is logged
    only every ~50 control steps. Plotting every step draws each observation 50x and
    smears the markers into bars. This script de-duplicates to the real observations
    (pass --raw to see every sample).

  * gp_*_pred is evaluated live at every step along the trajectory and rings/overshoots
    through the contact (the GP input is 2-D, (position, pitch), and pitch swings hard
    during the wheelie). gp_*_pred_pre -- the held-out prediction at each observation --
    is what actually tracks the residual, so it is the default 'GP mean'. Use --post to
    force the live column.

  * No predictive sigma is logged, so the band is an empirical held-out noise level
    (robust MAD of measured-minus-predicted). Log gp.predict(X, return_var=True) as
    gp_v_dot_std / gp_omega_dot_std for a true posterior band that widens where the
    truck has not interacted.

By default the figure auto-zooms to the active (contact) region; pass --full for the
whole run.

Usage:
    python plot_gp_fit.py [trajectory.csv] [--episode N] [--channel v|omega]
                          [--post] [--raw] [--full] [--xlim a,b]
                          [--obstacles x1,x2] [--bare]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INK = "#1b1b1b"; MUTE = "#6b7280"; GOAL_C = "#1b7a3a"
MEAN_C = "#2741d6"; BAND_C = "#6f7ee8"

CHANNELS = {
    "v": dict(r="r_v_dot", pre="gp_v_dot_pred_pre", post="gp_v_dot_pred",
              std="gp_v_dot_std", ylabel=r"$\dot{v}$ residual  [m s$^{-2}$]",
              title=r"GP $\dot{v}$ residual fit"),
    "omega": dict(r="r_omega_dot", pre="gp_omega_dot_pred_pre", post="gp_omega_dot_pred",
                  std="gp_omega_dot_std", ylabel=r"$\dot{\omega}$ residual  [rad s$^{-2}$]",
                  title=r"GP $\dot{\omega}$ residual fit  (excluded from rollouts)"),
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#cbd0d6",
        "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 12,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelsize": 12, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "xtick.color": MUTE, "ytick.color": MUTE,
        "legend.frameon": False, "legend.fontsize": 11,
    })


def parse_args(argv):
    csv, ep, ch = None, None, "v"
    post, raw, full, xlim, obs, bare = False, False, False, None, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--post": post = True
        elif a == "--raw": raw = True
        elif a == "--full": full = True
        elif a == "--bare": bare = True
        elif a == "--episode": i += 1; ep = int(argv[i])
        elif a.startswith("--episode="): ep = int(a.split("=", 1)[1])
        elif a == "--channel": i += 1; ch = argv[i]
        elif a.startswith("--channel="): ch = a.split("=", 1)[1]
        elif a == "--xlim": i += 1; xlim = tuple(float(v) for v in argv[i].split(","))
        elif a.startswith("--xlim="): xlim = tuple(float(v) for v in a.split("=", 1)[1].split(","))
        elif a == "--obstacles": i += 1; obs = [float(v) for v in argv[i].split(",") if v.strip()]
        elif a.startswith("--obstacles="): obs = [float(v) for v in a.split("=", 1)[1].split(",") if v.strip()]
        else: csv = Path(a)
        i += 1
    if ch not in CHANNELS:
        raise SystemExit(f"--channel must be one of {list(CHANNELS)}")
    return csv, ep, ch, post, raw, full, xlim, obs, bare


def mad(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if v.size else 0.0


def detect_focus(x, r, min_width=2.0):
    x = np.asarray(x, float); r = np.asarray(r, float)
    thr = max(6.0 * mad(r), 0.08)
    active = np.abs(r - np.median(r)) > thr
    if active.sum() < 3:
        return None
    lo, hi = x[active].min(), x[active].max()
    pad = 0.55 * max(hi - lo, 0.3)
    xlo, xhi = lo - pad, hi + pad
    if (xhi - xlo) < min_width:
        c = 0.5 * (xlo + xhi); xlo, xhi = c - min_width / 2, c + min_width / 2
    return max(x.min(), xlo), min(x.max(), xhi)


def main():
    setup_style()
    here = Path(__file__).parent
    csv_arg, episode, channel, post, raw, full, xlim, obstacles, bare = parse_args(sys.argv[1:])
    path = csv_arg if csv_arg is not None else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    spec = CHANNELS[channel]
    want = ["episode", "x", "goal_x", spec["r"], spec["pre"], spec["post"], spec["std"]]
    avail = set(pd.read_csv(path, nrows=0).columns)
    df = pd.read_csv(path, usecols=[c for c in want if c in avail])

    ep = int(df["episode"].max()) if episode is None else int(episode)
    g = df[df["episode"] == ep].sort_values("x").reset_index(drop=True)
    if g.empty:
        raise SystemExit(f"episode {ep} not in CSV "
                         f"(have {int(df['episode'].min())}..{int(df['episode'].max())})")

    if spec["r"] not in g:
        raise SystemExit(f"missing residual column {spec['r']}")
    r_all = g[spec["r"]].to_numpy()

    # ---- pick the GP-mean column: held-out *_pre tracks; live column rings ----------
    if post or spec["pre"] not in g.columns:
        mean_col = spec["post"]
        if post and spec["pre"] in g.columns:
            print("[info] --post: using live GP column", spec["post"])
        elif spec["pre"] not in g.columns:
            print(f"[info] {spec['pre']} not in CSV; using {spec['post']}")
    else:
        mean_col = spec["pre"]
    m_all = g[mean_col].to_numpy() if mean_col in g else None
    if m_all is not None and np.isfinite(r_all).all():
        c = np.corrcoef(r_all, m_all)[0, 1]
        print(f"[fit] GP mean = {mean_col}   corr(residual, mean) = {c:+.3f}")

    # ---- de-duplicate the zero-order hold to real observations ----------------------
    if raw:
        idx = np.arange(len(g))
        obs_note = f"{len(idx)} samples (raw, held)"
    else:
        idx = np.r_[0, np.where(np.diff(r_all) != 0)[0] + 1]
        obs_note = f"{len(idx)} observations"
        print(f"[hold] r de-held: {len(idx)} observations from {len(g)} steps "
              f"(~{len(g)//max(len(idx),1)}x hold)")

    x = g["x"].to_numpy()[idx]
    r = r_all[idx]
    m = m_all[idx] if m_all is not None else None
    goal = float(g["goal_x"].iloc[0]) if "goal_x" in g else None

    # ---- band: true GP sigma if logged, else robust held-out noise level -------------
    if spec["std"] in g.columns:
        sig = g[spec["std"]].to_numpy()[idx]
        band_lo, band_hi = m - 2 * sig, m + 2 * sig
        band_label = r"GP $\pm 2\sigma$"
    elif m is not None:
        err = r - m
        s = mad(err)
        band_lo, band_hi = m - 2 * s, m + 2 * s
        band_label = r"$\pm2\sigma$ (held-out)"
        rmse = float(np.sqrt(np.nanmean(err ** 2)))
        print(f"[band] no GP sigma logged -> held-out noise band, robust sigma={s:.4f}, "
              f"RMSE={rmse:.4f}")
    else:
        band_lo = band_hi = None

    # ---- x window -------------------------------------------------------------------
    if xlim is not None:
        xlo, xhi = xlim
    elif not full and detect_focus(x, r):
        xlo, xhi = detect_focus(x, r)
        print(f"[focus] auto-zoomed to x in [{xlo:.2f}, {xhi:.2f}] (--full for whole run)")
    else:
        xlo, xhi = float(x.min()), float(x.max())

    fig, ax = plt.subplots(figsize=(12, 5.8))
    if not bare:
        for xo in (obstacles or []):
            if xlo <= xo <= xhi:
                ax.axvline(xo, color="#9aa0a6", lw=1.0, ls=":", zorder=1)
        if goal is not None and xlo <= goal <= xhi:
            ax.axvline(goal, color=GOAL_C, ls=(0, (5, 3)), lw=1.4, zorder=1.5, label="goal")

    if band_lo is not None:
        ax.fill_between(x, band_lo, band_hi, color=BAND_C, alpha=0.25, lw=0,
                        zorder=2, label=band_label)
    if m is not None:
        ax.plot(x, m, color=MEAN_C, lw=2.2, zorder=4, label="GP mean")
    obs_short = f"{len(idx)} obs" if not raw else f"{len(idx)} samples"
    ax.scatter(x, r, s=42, c=INK, marker="x", linewidths=1.4, zorder=5,
               label=f"measured ({obs_short})")

    ax.axhline(0.0, color="#9aa0a6", lw=0.9, zorder=1)
    ax.grid(axis="y", color="#eceef1", lw=0.9, zorder=0); ax.set_axisbelow(True)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("x position  [m]")
    ax.set_ylabel(spec["ylabel"])

    inwin = (x >= xlo) & (x <= xhi)
    yref = r[inwin][np.isfinite(r[inwin])]
    if yref.size:
        lo, hi = yref.min(), yref.max()
        pad = 0.12 * ((hi - lo) or 1.0)
        ax.set_ylim(lo - pad, hi + pad)

    # title + compact one-row legend, stacked above the axes (plot keeps full width)
    ax.set_title(spec["title"], loc="left", pad=30)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x_) for x_ in
             [f"measured ({obs_short})", "GP mean", band_label, "goal"] if x_ in l]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(order),
              frameon=False, handlelength=1.4, columnspacing=1.3,
              handletextpad=0.5, fontsize=10.5, borderaxespad=0.0)

    tag = f"episode {ep}" + ("" if episode is not None else " (most-learned)")
    win = ("full run" if (full or (xlim is None and not detect_focus(x, r)))
           else f"x in [{xlo:.1f}, {xhi:.1f}] m")
    fig.text(0.085, 0.015, f"{tag}   |   {win}", ha="left", fontsize=9.5, color=MUTE)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.84, bottom=0.13)

    img_dir = here / "images"; fname = f"gp_fit_{channel}.png"
    try:
        img_dir.mkdir(exist_ok=True); out = img_dir / fname
        fig.savefig(out, dpi=200, bbox_inches="tight"); print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp") / fname
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
