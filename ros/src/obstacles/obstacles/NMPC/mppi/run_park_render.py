#!/usr/bin/env python3
"""
run_park_render.py
------------------
Watch the Park-wrapped controller CLIMB the obstacle and STOP at the goal in the
MuJoCo viewer. The PD-park (goal_park.py) guarantees the stop; the inner
controller (MPPI or NMPC) does the wheelie.

    python3 run_park_render.py        # opens the MuJoCo passive viewer

Pick the inner controller (INNER) and the obstacle (SCENARIO) below.
Set RENDER = False for a headless run that just prints the trajectory.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nmpc import WheelieParams, MPCConfig, WheelieNMPC      # noqa: E402
from rls import RLSConfig                                   # noqa: E402
from gp_residual import GPConfig                            # noqa: E402

from mppi import WheelieMPPI, MPPIConfig                    # noqa: E402
from goal_park import ParkController, ParkConfig            # noqa: E402
from sim_harness import run_episode                         # noqa: E402

HERE = Path(__file__).resolve().parent

# ============================================================
# Settings
# ============================================================
RENDER = True
SIM_TIME = 16.0
INNER = "mppi"            # "mppi" (fast, recommended for the car) or "nmpc"

# Obstacle to watch on. Default: the fixed clean climbable scenario.
SCENARIO = HERE / "scenario_clean.xml"
# To watch on your own current obstacle instead, use:
#   SCENARIO = HERE.parent / "monster_truck_flip_2d.xml"

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = GPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)
PARK_CFG = ParkConfig(kp=4.0, kd=6.0, pitch_on_deg=14.0, pitch_off_deg=8.0)

SHARED = dict(N=20, q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
              r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0)


def make_inner():
    if INNER == "nmpc":
        return WheelieNMPC(PARAMS, MPCConfig(dt=0.05, ipopt_max_iter=80, **SHARED), GP_CFG)
    return WheelieMPPI(PARAMS, MPPIConfig(dt=0.05, num_samples=2048, temperature=10.0,
                                          noise_sigma=4.0, seed=0, **SHARED), GP_CFG)


def main():
    ctrl = ParkController(make_inner(), PARAMS, PARK_CFG)
    print(f"Park + {INNER.upper()}  (PD-park guarantees the stop at the goal)")
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, xml_path=SCENARIO,
                      sim_time=SIM_TIME, render=RENDER, verbose=True)
    print("\nMetrics:")
    for k, val in out["metrics"].items():
        print(f"  {k:16s}: {val}")


if __name__ == "__main__":
    main()
