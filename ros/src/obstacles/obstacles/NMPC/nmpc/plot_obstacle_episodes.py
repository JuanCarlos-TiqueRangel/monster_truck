#!/usr/bin/env python3
"""
plot_obstacle_episodes.py
-------------------------
Per-EPISODE diagnostics for obstacle_mujoco_simulation.py. Four clean panels, all
plotted against the episode index (EVERY episode is shown -- none dropped):

  1. episode duration       -> time the episode took: reached episodes finish fast,
                               stuck episodes run to the SIM_TIME cap.
  2. how far it got (x_max) -> where it stalls (obstacle lines marked) + reached.
  3. RLS a_g vs episode     -> the learned gravity term (nominal marked).
  4. GP one-step RMSE       -> over CLEAN steps (ready, not gated); DOWN = GP learning.

green = reached the goal, red = stuck.

Usage:
    python plot_obstacle_episodes.py [trajectory.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OBSTACLE_XS = [1.0, 3.0, 5.0]    # box positions in monster_truck_flip_2d.xml
REACH_TOL = 0.15                 # x_max within this of goal counts as "reached"


def nominal_a_g():
    try:
        from params_mujoco import WheelieParams
        from rls import nominal_rls_parameters
        return float(nominal_rls_parameters(WheelieParams())[5])
    except Exception:
        return np.nan


def main():
    here = Path(__file__).parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    avail = set(pd.read_csv(path, nrows=0).columns)
    want = ["episode", "time", "x", "goal_x", "a_g", "r_omega_dot",
            "gp_omega_dot_pred_pre", "gp_ready", "gp_gated"]
    df = pd.read_csv(path, usecols=[c for c in want if c in avail])
    goal = float(df["goal_x"].iloc[0])
    g_a_g = nominal_a_g()

    eps, dur, x_max, a_g_end, rmse_w = [], [], [], [], []
    for ep, g in df.groupby("episode"):
        eps.append(ep)
        dur.append(g["time"].iloc[-1])
        x_max.append(g["x"].max())
        a_g_end.append(g["a_g"].iloc[-1] if "a_g" in g else np.nan)
        if {"gp_ready", "gp_omega_dot_pred_pre", "r_omega_dot"} <= set(g.columns):
            m = g["gp_ready"] > 0.5
            if "gp_gated" in g:
                m &= g["gp_gated"] < 0.5
            e = (g["r_omega_dot"][m] - g["gp_omega_dot_pred_pre"][m]).to_numpy()
            rmse_w.append(float(np.sqrt(np.mean(e * e))) if e.size else np.nan)
        else:
            rmse_w.append(np.nan)

    eps = np.array(eps); dur = np.array(dur); x_max = np.array(x_max)
    reached = np.abs(x_max - goal) < REACH_TOL
    third = max(1, len(eps) // 3)
    print(f"{len(eps)} episodes | reached goal: {100*reached.mean():.0f}%  "
          f"(first {third}: {100*reached[:third].mean():.0f}% -> "
          f"last {third}: {100*reached[-third:].mean():.0f}%) | "
          f"median duration {np.median(dur):.2f}s")

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    a1, a2, a3, a4 = axs.ravel()

    # 1. episode duration (EVERY episode; reached fast, stuck = cap)
    a1.scatter(eps[reached], dur[reached], s=20, c="C2", label="reached goal")
    a1.scatter(eps[~reached], dur[~reached], s=20, c="C3", label="stuck")
    a1.set_title("1) episode duration per episode")
    a1.set_xlabel("episode"); a1.set_ylabel("duration [s]"); a1.grid(True); a1.legend(fontsize=8)

    # 2. how far it got
    a2.scatter(eps[reached], x_max[reached], s=20, c="C2", label="reached goal")
    a2.scatter(eps[~reached], x_max[~reached], s=20, c="C3", label="stuck")
    for xo in OBSTACLE_XS:
        a2.axhline(xo, color="0.7", ls=":", lw=0.8)
    a2.axhline(goal, color="green", ls="--", lw=1.0)
    a2.set_title("2) how far it got (x_max) per episode")
    a2.set_xlabel("episode"); a2.set_ylabel("x_max [m]"); a2.grid(True); a2.legend(fontsize=8)

    # 3. RLS a_g
    a3.plot(eps, a_g_end, "o-", ms=4, color="C4")
    if np.isfinite(g_a_g):
        a3.axhline(g_a_g, color="k", ls="--", lw=1.0, label=f"nominal {g_a_g:.2f}")
    a3.axhline(0.0, color="0.7", lw=0.8)
    a3.set_title("3) RLS a_g vs episode (learned gravity term)")
    a3.set_xlabel("episode"); a3.set_ylabel("a_g"); a3.grid(True); a3.legend(fontsize=8)

    # 4. GP one-step RMSE
    a4.plot(eps, rmse_w, "o-", ms=4, color="C0")
    a4.set_title("4) GP one-step RMSE, clean steps  (DOWN = GP learning)")
    a4.set_xlabel("episode"); a4.set_ylabel("RMSE r_omega_dot [rad/s^2]"); a4.grid(True)

    fig.suptitle("Obstacle learning diagnostics (per episode)")
    fig.tight_layout()

    img_dir = here / "images"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / "obstacle_diagnostics.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path("/tmp/obstacle_diagnostics.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
