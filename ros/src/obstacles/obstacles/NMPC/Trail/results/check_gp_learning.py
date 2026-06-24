#!/usr/bin/env python3
"""
check_gp_learning.py
--------------------
Is the GPyTorch residual GP actually LEARNING across episodes? This reads the sim's
CSV (results/obstacle_mujoco.csv) and answers it with NUMBERS + a clear verdict,
headless (saves a PNG, no display needed -- safe in docker).

THE HONEST METRIC -- one-episode-ahead generalization
-----------------------------------------------------
The GP is FROZEN during an episode: while episode N runs, the controller uses the
model fit at the END of episode N-1. The sim logs, per step:
    r_v_dot, r_omega_dot              = MEASURED residual (the GP's target)
    gp_v_dot_pred_pre, ..._pred_pre   = that FROZEN model's prediction at the SAME
                                        point, BEFORE this episode's data is absorbed
So (r - pred_pre) over episode N = how well the model trained through episode N-1
predicts NEW data in episode N. If the GP is learning, this error goes DOWN as more
episodes accumulate. (gp_*_pred is the in-sample prediction -- shown for contrast.)

WHAT IT REPORTS  (per channel: v_dot "blockage", omega_dot "pitch")
  * per-episode one-step RMSE, robust median|err|, and R^2 (variance explained)
  * a LINEAR TREND of median|err| vs episode with a 95% CI on the slope:
        slope CI entirely < 0  -> IMPROVING (learning)
        slope CI spans 0       -> FLAT      (not learning over episodes)
        slope CI entirely > 0  -> WORSENING
  * pooled correlation(pred_pre, measured) -- does the prediction track reality?
  * the same trend restricted to the auto-detected OBSTACLE region (where |residual|
    is largest), which is the most sensitive view (open ground has ~0 residual and
    only dilutes the signal).

Usage:
    python3 check_gp_learning.py [obstacle_mujoco.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")                 # headless: no display, just save a PNG
import matplotlib.pyplot as plt

CHANNELS = [("v_dot", "r_v_dot", "gp_v_dot_pred_pre", "gp_v_dot_pred", "blockage (forward decel)"),
            ("omega_dot", "r_omega_dot", "gp_omega_dot_pred_pre", "gp_omega_dot_pred", "pitch accel")]


def per_episode_metrics(df, r_col, pre_col):
    """One row per episode: n, RMSE, median|err|, R^2 -- of (measured - pred_pre)."""
    rows = []
    for ep, g in df.groupby("episode"):
        if len(g) < 5:
            continue
        err = (g[r_col] - g[pre_col]).to_numpy()
        r = g[r_col].to_numpy()
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 1e-12 else np.nan
        rows.append((int(ep), len(g),
                     float(np.sqrt(np.mean(err ** 2))),
                     float(np.median(np.abs(err))), r2))
    return pd.DataFrame(rows, columns=["episode", "n", "rmse", "med_abs_err", "r2"])


def trend(x, y):
    """Linear fit y~x with a 95% CI on the slope. Returns (slope, lo, hi, verdict)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, "too few episodes"
    lr = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(x) - 2)
    lo, hi = lr.slope - tcrit * lr.stderr, lr.slope + tcrit * lr.stderr
    if hi < 0:
        v = "IMPROVING"            # error falling, CI entirely below 0
    elif lo > 0:
        v = "WORSENING"           # error rising
    else:
        v = "FLAT"                # CI spans 0 -> no significant trend
    return lr.slope, lo, hi, v


def obstacle_window(df, r_col, bin_w=0.2, half_width=0.75):
    """Auto-find the x-region of the obstacle = bins where mean|residual| peaks."""
    x = df["x"].to_numpy()
    if x.size == 0:
        return None
    edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + bin_w, bin_w)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    mag = np.abs(df[r_col].to_numpy())
    means = np.array([mag[idx == b].mean() if np.any(idx == b) else 0.0
                      for b in range(len(edges) - 1)])
    if not means.any():
        return None
    center = 0.5 * (edges[means.argmax()] + edges[means.argmax() + 1])
    return center - half_width, center + half_width


def report_channel(name, df, r_col, pre_col, pred_col, desc):
    print(f"\n================  {name}  ({desc})  ================")
    ready = df[df["gp_ready"] > 0.5]
    if ready.empty:
        print("  GP never became READY in this run -> nothing learned.")
        print("  (needs >= max_points samples before the first fit; raise SIM_TIME / "
              "N_EPISODES or lower GPConfig.max_points.)")
        return None

    glob = per_episode_metrics(ready, r_col, pre_col)
    win = obstacle_window(ready, r_col)
    obs = ready[ready["x"].between(*win)] if win else ready
    obsm = per_episode_metrics(obs, r_col, pre_col)

    corr = float(np.corrcoef(ready[pre_col], ready[r_col])[0, 1]) if len(ready) > 2 else np.nan
    s_g, lo_g, hi_g, v_g = trend(glob["episode"], glob["med_abs_err"])
    s_o, lo_o, hi_o, v_o = trend(obsm["episode"], obsm["med_abs_err"])

    def fl(df_):
        if df_.empty:
            return None, None
        return df_.iloc[0], df_.iloc[-1]

    f, l = fl(glob)
    print(f"  ready episodes: {int(glob['episode'].min())}..{int(glob['episode'].max())}  "
          f"({len(glob)} episodes, ~{int(glob['n'].mean())} pts/ep)")
    if win:
        print(f"  obstacle region auto-detected: x in [{win[0]:.2f}, {win[1]:.2f}] m")
    if f is not None:
        print(f"  median|err| (global):   {f['med_abs_err']:.4f}  ->  {l['med_abs_err']:.4f}   "
              f"(ep {int(f['episode'])} -> {int(l['episode'])})")
        print(f"  R^2 variance explained: {f['r2']:+.3f}  ->  {l['r2']:+.3f}")
    print(f"  pred-vs-measured corr (pooled): {corr:+.3f}   (-> +1 = tracks reality)")
    print(f"  TREND median|err| vs episode:")
    print(f"      global:   slope={s_g:+.2e} /ep  95%CI[{lo_g:+.2e}, {hi_g:+.2e}]  -> {v_g}")
    print(f"      obstacle: slope={s_o:+.2e} /ep  95%CI[{lo_o:+.2e}, {hi_o:+.2e}]  -> {v_o}")
    # omega's residual is not an obstacle-of-x like the blockage, so its auto-window
    # is unreliable -- fall back to the global trend for the headline when that happens.
    verdict = v_o if v_o not in ("too few episodes",) else v_g
    return {"name": name, "glob": glob, "obs": obsm, "win": win, "corr": corr,
            "verdict": verdict, "ready": ready,
            "r_col": r_col, "pre_col": pre_col, "pred_col": pred_col}


def make_figure(results, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    colors = {"v_dot": "C0", "omega_dot": "C3"}

    # (0,0) median|err| vs episode (obstacle region) -- the learning curve
    ax = axes[0, 0]
    for R in results:
        ax.plot(R["obs"]["episode"], R["obs"]["med_abs_err"], "o-", ms=3,
                color=colors[R["name"]], label=f"{R['name']} ({R['verdict']})")
    ax.set_title("Learning curve: one-step median|err| vs episode  (DOWN = learning)")
    ax.set_xlabel("episode"); ax.set_ylabel("median |measured - pred_pre|  @obstacle")
    ax.grid(True); ax.legend()

    # (0,1) R^2 vs episode (global)
    ax = axes[0, 1]
    for R in results:
        ax.plot(R["glob"]["episode"], R["glob"]["r2"], "o-", ms=3,
                color=colors[R["name"]], label=R["name"])
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_title("R^2 (variance of the residual explained)  -> 1 = perfect, <0 = useless")
    ax.set_xlabel("episode"); ax.set_ylabel("R^2 (global, one-step)")
    ax.set_ylim(-1.0, 1.0); ax.grid(True); ax.legend()

    # (1,0) predicted vs measured (v_dot), colored by episode
    R = results[0]
    ready = R["ready"]
    sub = ready.iloc[:: max(1, len(ready) // 6000)]
    pred, meas = sub[R["pre_col"]].to_numpy(), sub[R["r_col"]].to_numpy()
    lim = float(np.nanmax(np.abs(np.concatenate([pred, meas])))) or 1.0
    ax = axes[1, 0]
    sc = ax.scatter(pred, meas, s=5, c=sub["episode"], cmap="viridis", alpha=0.4)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="perfect (y=x)")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(f"{R['name']}: predicted vs measured  (corr={R['corr']:+.3f})")
    ax.set_xlabel("GP predicted (pred_pre)"); ax.set_ylabel("measured residual")
    ax.grid(True); ax.legend(); fig.colorbar(sc, ax=ax, label="episode")

    # (1,1) residual vs x: first ready episode vs last (obstacle map emerging)
    ax = axes[1, 1]
    eps = sorted(ready["episode"].unique())
    first, last = ready[ready["episode"] == eps[0]], ready[ready["episode"] == eps[-1]]
    ax.scatter(first["x"], first[R["r_col"]], s=6, alpha=0.4, color="0.6", label=f"ep {int(eps[0])} (measured)")
    ax.scatter(last["x"], last[R["r_col"]], s=6, alpha=0.5, color="C0", label=f"ep {int(eps[-1])} (measured)")
    ls = last.sort_values("x")
    ax.plot(ls["x"], ls[R["pred_col"]], "-", color="C1", lw=1.5,
            label=f"ep {int(eps[-1])} GP fit")
    if R["win"]:
        ax.axvspan(*R["win"], color="orange", alpha=0.12, label="obstacle region")
    ax.set_title(f"{R['name']}: residual vs x  (the obstacle the GP localises)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("residual"); ax.grid(True); ax.legend(fontsize=8)

    fig.suptitle("Is the GP learning?", fontsize=14)
    fig.tight_layout()
    try:
        out_path.parent.mkdir(exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"\nSaved figure: {out_path}")
    except PermissionError:
        out_path = Path("/tmp/gp_is_learning.png")
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"\n[warn] images/ not writable; saved to {out_path}")


def main():
    here = Path(__file__).parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    avail = set(pd.read_csv(path, nrows=0).columns)
    need = {"episode", "x", "gp_ready", "r_v_dot", "gp_v_dot_pred_pre", "gp_v_dot_pred"}
    if not need.issubset(avail):
        raise SystemExit(f"CSV missing columns {need - avail}. Re-run the sim to log them.")
    cols = sorted(need | {c for _, a, b, d, _ in CHANNELS for c in (a, b, d) if c in avail})
    print(f"Reading {path.name} (columns: {len(cols)}) ...")
    df = pd.read_csv(path, usecols=cols)
    print(f"  {len(df):,} rows | episodes {int(df['episode'].min())}..{int(df['episode'].max())} "
          f"| GP ready in {int((df['gp_ready'] > 0.5).sum()):,} rows")

    results = []
    for name, r_col, pre_col, pred_col, desc in CHANNELS:
        if r_col in df and pre_col in df:
            R = report_channel(name, df, r_col, pre_col, pred_col, desc)
            if R is not None:
                results.append(R)

    if not results:
        raise SystemExit("\nNo usable channel data (GP never ready). Nothing to plot.")

    make_figure(results, here / "images" / "gp_is_learning.png")

    # ---- one-line headline verdict (v_dot blockage = what drives climbing) ----
    head = results[0]
    print("\n" + "=" * 64)
    print(f"VERDICT: GP {head['name']} is {head['verdict']} over episodes "
          f"(obstacle-region trend), pred-vs-measured corr {head['corr']:+.2f}.")
    print("  IMPROVING = learning;  FLAT = not improving across episodes;  "
          "WORSENING = drifting.")
    print("=" * 64)


if __name__ == "__main__":
    main()
