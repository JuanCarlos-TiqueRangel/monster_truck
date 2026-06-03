#!/usr/bin/env python3
"""
compare_nmpc_vs_mppi.py
-----------------------
Compares four controllers on ONE fixed, reproducible climbable obstacle
(scenario_clean.xml), same online RLS+GP model and cost weights:

    plain NMPC     CasADi NMPC alone           -> climbs, then RUNS AWAY past goal
    plain MPPI     GPU sampling alone          -> climbs, settles near goal
    Park + NMPC    NMPC for the wheelie + PD-park guarantee   -> climbs, STOPS at goal
    Park + MPPI    MPPI for the wheelie + PD-park guarantee   -> climbs, STOPS at goal

The Park wrapper (goal_park.py) adds a model-free PD law that provably brings
the truck to rest at the goal, so even the NMPC -- which alone overshoots --
now stops exactly at B. Prints a table and saves plots (incl. Time vs x).

    python3 compare_nmpc_vs_mppi.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nmpc import WheelieParams, MPCConfig, WheelieNMPC          # noqa: E402
from rls import RLSConfig                                       # noqa: E402
from gp_residual import GPConfig                                # noqa: E402

from mppi import WheelieMPPI, MPPIConfig                        # noqa: E402
from goal_park import ParkController, ParkConfig                # noqa: E402
from sim_harness import run_episode                             # noqa: E402

HERE = Path(__file__).resolve().parent
SCENARIO = HERE / "scenario_clean.xml"

P = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = GPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)
SHARED = dict(N=20, q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
              r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0)

METRIC_ROWS = ["climbed", "settle_err", "final_x", "final_v", "max_x",
               "max_abs_pitch", "flipped", "mean_abs_tau", "solve_ms_mean"]
LABELS = {"climbed": "climbed", "settle_err": "settle err @goal[m]",
          "final_x": "final x [m]", "final_v": "final v [m/s]", "max_x": "max x [m]",
          "max_abs_pitch": "max |pitch| [deg]", "flipped": "flipped",
          "mean_abs_tau": "mean |tau| [N.m]", "solve_ms_mean": "solve mean [ms]"}


def nmpc():
    return WheelieNMPC(P, MPCConfig(dt=0.05, ipopt_max_iter=80, **SHARED), GP_CFG)


def mppi():
    return WheelieMPPI(P, MPPIConfig(dt=0.05, num_samples=2048, temperature=10.0,
                                     noise_sigma=4.0, seed=0, **SHARED), GP_CFG)


def go(ctrl):
    return run_episode(ctrl, P, GP_CFG, RLS_CFG, xml_path=SCENARIO)


def cell(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main():
    runs = [
        ("plain NMPC", "C0", go(nmpc())),
        ("Park+NMPC",  "C3", go(ParkController(nmpc(), P, ParkConfig()))),
        ("plain MPPI", "C1", go(mppi())),
        ("Park+MPPI",  "C2", go(ParkController(mppi(), P, ParkConfig()))),
    ]
    names = [r[0] for r in runs]
    M = {r[0]: r[2]["metrics"] for r in runs}

    print("\n" + "=" * 92)
    hdr = f"{'metric':18s} | " + " | ".join(f"{n:>14s}" for n in names)
    print(hdr); print("-" * 92)
    for k in METRIC_ROWS:
        print(f"{LABELS[k]:18s} | " + " | ".join(f"{cell(M[n][k]):>14s}" for n in names))
    print("=" * 92)
    print("obstacle far edge x =", round(M[names[0]]["obstacle_far_x"], 3), " goal x = 5.0")

    # ---- Time vs x ----
    goal_x, far_x = 5.0, M[names[0]]["obstacle_far_x"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color, out in runs:
        h = out["history"]
        ax.plot(h["t"], h["x"], color=color, lw=2, label=name)
    ax.axhline(goal_x, ls="--", c="k", lw=1.2, label="goal B")
    ax.axhline(far_x, ls=":", c="gray", lw=1.0, label="obstacle edge")
    ax.set_xlabel("time [s]"); ax.set_ylabel("x position [m]")
    ax.set_title("Time vs position x  —  PD-park makes NMPC (and MPPI) stop at the goal")
    ax.grid(True, alpha=0.4); ax.legend()
    fig.tight_layout()
    p1 = HERE / "compare_x_vs_t.png"
    fig.savefig(p1, dpi=150); plt.close(fig)
    print("saved plot:", p1)


if __name__ == "__main__":
    main()
