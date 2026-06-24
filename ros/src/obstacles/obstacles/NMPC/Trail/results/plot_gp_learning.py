#!/usr/bin/env python3
"""
plot_gp_learning.py
-------------------
Is the GP learning the obstacle? It learns the VELOCITY residual r_v_dot -- the
"blockage" (extra deceleration the dynamics model can't explain) -- as a function of
(x, theta). Two panels, from the predict-BEFORE-update logs (gp_v_dot_pred_pre = the GP's
one-step prediction on a point it had not yet seen, paired with the measured r_v_dot):

  1. GP one-step RMSE vs episode    -> DOWN over episodes = learning.
  2. predicted vs measured r_v_dot      -> points on the y=x line (correlation -> 1) =
                                       the GP's blockage prediction matches reality.
                                       A blob with corr ~ 0 = not learning.

Only CLEAN steps are used (GP ready, not flip-gated).

Usage:
    python plot_gp_learning.py [trajectory.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    avail = set(pd.read_csv(path, nrows=0).columns)
    if "gp_v_dot_pred_pre" not in avail or "gp_ready" not in avail:
        raise SystemExit("CSV lacks gp_v_dot_pred_pre/gp_ready -- re-run the sim (these are "
                         "the predict-before-update logs needed to judge GP learning).")
    cols = ["episode", "x", "r_v_dot", "gp_v_dot_pred_pre", "gp_ready"] + (["gp_gated"] if "gp_gated" in avail else [])
    df = pd.read_csv(path, usecols=cols)

    m = df["gp_ready"] > 0.5
    if "gp_gated" in df:
        m &= df["gp_gated"] < 0.5
    df = df[m]
    # Focus on the OBSTACLE region (x ~ box), where the learnable blockage lives -- on open
    # ground the residual is ~0 and would dilute the signal. Use the MEDIAN |error| (robust):
    # the RMSE is swamped by the rare contact/flip spikes the smooth model can never predict,
    # so it looks like random noise even while the GP is clearly learning.
    obs = df[df["x"].between(1.0, 3.5)]

    eps, mederr = [], []
    for ep, g in obs.groupby("episode"):
        if len(g) < 5:
            continue
        eps.append(ep)
        mederr.append(float((g["r_v_dot"] - g["gp_v_dot_pred_pre"]).abs().median()))
    eps = np.array(eps)

    sub = obs.iloc[::max(1, len(obs) // 8000)]
    pred, meas = sub["gp_v_dot_pred_pre"].to_numpy(), sub["r_v_dot"].to_numpy()
    corr = np.corrcoef(obs["gp_v_dot_pred_pre"], obs["r_v_dot"])[0, 1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    a1.plot(eps, mederr, "o-", ms=4, color="C0")
    a1.set_title("1) GP one-step error vs episode  (DOWN then flat = learning + converged)")
    a1.set_xlabel("episode"); a1.set_ylabel("median |err|  r_v_dot @obstacle")
    a1.grid(True)

    lim = float(np.nanmax(np.abs(np.concatenate([pred, meas])))) or 1.0
    a2.scatter(pred, meas, s=5, c=sub["episode"], cmap="viridis", alpha=0.4)
    a2.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="perfect (y=x)")
    a2.set_xlim(-lim, lim); a2.set_ylim(-lim, lim)
    a2.set_title(f"2) predicted vs measured blockage  (corr = {corr:+.3f}; "
                 f"-> 1 = learning)")
    a2.set_xlabel("GP predicted r_v_dot"); a2.set_ylabel("measured r_v_dot")
    a2.grid(True); a2.legend(fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(cmap="viridis",
                 norm=plt.Normalize(eps.min(), eps.max())), ax=a2, label="episode")

    verdict = "LEARNING" if (corr > 0.3 and mederr[-1] < 0.85 * mederr[0]) else "NOT learning"
    fig.suptitle(f"Is the GP learning the obstacle (blockage)?  ->  {verdict}   "
                 f"(median err {mederr[0]:.1f} -> {mederr[-1]:.1f}, corr {corr:+.2f})")
    fig.tight_layout()

    img_dir = here / "images"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / "gp_learning.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp/gp_learning.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")
    print(f"verdict: GP is {verdict}  (corr={corr:+.3f}, median err {mederr[0]:.1f} -> {mederr[-1]:.1f})")
    plt.show()


if __name__ == "__main__":
    main()
