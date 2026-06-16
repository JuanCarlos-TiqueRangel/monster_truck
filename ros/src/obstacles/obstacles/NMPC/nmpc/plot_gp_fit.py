#!/usr/bin/env python3
"""
plot_gp_fit.py
--------------
Standard GP-regression diagnostics for the blockage channel (the GP learning the
v_dot residual r_v_dot as a function of (x, theta)). Two textbook panels:

  1. PARITY plot  -- GP one-step prediction vs the measured residual, with the y=x
     line and R^2 / RMSE. Points on the diagonal = accurate predictions. (The standard
     "predicted vs observed" regression plot.)
  2. GP POSTERIOR plot -- the GP predictive mean +/- 2 sigma as a function of position
     x (at a representative pitch slice), with the raw residual data overlaid. (The
     canonical GP plot, Rasmussen & Williams style.) A dip with a tight band at each
     obstacle = it learned where they are, confidently.

Usage: python plot_gp_fit.py [trajectory.csv] [model.npz]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OBSTACLE_XS = [1.0, 3.0, 6.8, 7.2]


def main():
    here = Path(__file__).parent
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "obstacle_mujoco.csv"
    if len(sys.argv) > 2:
        model = Path(sys.argv[2])
    else:  # derive the model from the CSV (obstacle_test.csv -> obstacle_test_model.npz)
        cand = csv.with_name(csv.stem + "_model.npz")
        model = cand if cand.exists() else here / "obstacle_model.npz"
    if not csv.exists():
        raise SystemExit(f"CSV not found: {csv}\nRun obstacle_mujoco_simulation.py first.")

    df = pd.read_csv(csv, usecols=["episode", "x", "pitch_rad", "r_v_dot",
                                   "gp_v_dot_pred_pre", "gp_ready"])
    d = df[df["gp_ready"] > 0.5]
    pred = d["gp_v_dot_pred_pre"].to_numpy()
    meas = d["r_v_dot"].to_numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1) PARITY: predicted vs measured -----------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pred), size=min(6000, len(pred)), replace=False)
    lim = float(np.nanpercentile(np.abs(np.concatenate([pred, meas])), 99)) or 1.0
    a1.scatter(pred[idx], meas[idx], s=6, alpha=0.3, color="C0")
    a1.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="y = x (perfect)")
    sst = np.sum((meas - meas.mean()) ** 2)
    r2 = 1.0 - np.sum((meas - pred) ** 2) / sst if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean((meas - pred) ** 2)))
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim); a1.set_aspect("equal", "box")
    a1.set_title(f"1) predicted vs measured blockage   (R$^2$={r2:.2f}, RMSE={rmse:.2f})")
    a1.set_xlabel("GP predicted  r_v_dot  [m/s$^2$]")
    a1.set_ylabel("measured  r_v_dot  [m/s$^2$]")
    a1.grid(True); a1.legend(fontsize=9)

    # 2) GP POSTERIOR: mean +/- 2 sigma vs x -----------------------------------
    # Reconstruct the v_dot-channel GP DIRECTLY from the saved arrays (Z, l, sf2, sn2,
    # posterior m & S) -- robust to the current config's max_points and to old/new key
    # names (m_v_dot vs the pre-rename m_v).
    try:
        if not model.exists():
            raise FileNotFoundError(model)
        from SSGP import StreamingSparseVGP
        md = np.load(model, allow_pickle=False)
        g = {k[len("gp_"):]: md[k] for k in md.files if k.startswith("gp_")}
        mkey = "m_v_dot" if "m_v_dot" in g else "m_v"      # new vs pre-rename keys
        skey = "S_v_dot" if "S_v_dot" in g else "S_v"
        vgp = StreamingSparseVGP(np.asarray(g["Z"], float), np.asarray(g["l"], float),
                                 float(g["sf2"]), float(g["sn2"]))
        vgp.m = np.asarray(g[mkey], float); vgp.S = np.asarray(g[skey], float)
        th0 = float(np.median(d["pitch_rad"]))
        xs = np.linspace(0.0, float(d["x"].max()), 200)
        mu = np.empty_like(xs); sd = np.empty_like(xs)
        for i, xx in enumerate(xs):
            mean, var = vgp.predict(np.array([xx, th0]))
            mu[i], sd[i] = mean, np.sqrt(var)
        a2.scatter(d["x"].iloc[::20], d["r_v_dot"].iloc[::20], s=5, alpha=0.15,
                   color="0.6", label="data (measured)")
        a2.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color="C3", alpha=0.2, label="GP $\\pm2\\sigma$")
        a2.plot(xs, mu, color="C3", lw=2, label="GP mean")
        a2.set_title(f"2) GP posterior: blockage vs x   (pitch slice $\\theta$={np.degrees(th0):.0f}$\\degree$)")
    except Exception as e:
        print(f"[panel 2] GP posterior could not load {model.name}: {type(e).__name__}: {e}")
        a2.text(0.5, 0.5, f"GP posterior needs the model file:\n{model.name}\n\n{type(e).__name__}: {e}",
                ha="center", va="center", transform=a2.transAxes, fontsize=9)
        a2.set_title("2) GP posterior vs x  (model not loaded)")
    a2.axhline(0, color="0.8", lw=0.8)
    for xo in OBSTACLE_XS:
        a2.axvline(xo, color="0.7", ls=":", lw=0.8)
    a2.set_xlabel("x [m]"); a2.set_ylabel("v_dot residual  [m/s$^2$]")
    a2.grid(True); a2.legend(fontsize=8)

    fig.suptitle("GP blockage model -- standard regression diagnostics")
    fig.tight_layout()

    img_dir = here / "images"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / "gp_fit.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp/gp_fit.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")
    print(f"parity: R^2={r2:.3f}, RMSE={rmse:.3f}")
    plt.show()


if __name__ == "__main__":
    main()
