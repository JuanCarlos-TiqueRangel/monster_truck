#!/usr/bin/env python3
"""
run_mppi_render.py
------------------
Watch the MPPI controller drive the truck in the MuJoCo viewer -- the MPPI
equivalent of wheelie_gp_climb.py (which runs the CasADi NMPC).

    python3 run_mppi_render.py        # opens the MuJoCo passive viewer

Set RENDER = False for a headless run that just prints the trajectory.
All knobs live in the block below.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nmpc import WheelieParams                  # noqa: E402
from rls import RLSConfig                       # noqa: E402
from gp_residual import GPConfig                # noqa: E402

from mppi import WheelieMPPI, MPPIConfig        # noqa: E402
from sim_harness import run_episode             # noqa: E402


# ============================================================
# Settings
# ============================================================
RENDER = True
SIM_TIME = 20.0

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = GPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

MPPI_CFG = MPPIConfig(
    dt=0.05,
    N=20,                 # horizon steps
    num_samples=2048,     # K rollouts (drop to ~512 for a Jetson)
    temperature=10.0,
    noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
    r_tau=0.05, r_dtau=1.0,
    q_terminal_theta=6.0, q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
)


def main():
    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG,
                      sim_time=SIM_TIME, render=RENDER, verbose=True)
    print("\nMetrics:")
    for k, val in out["metrics"].items():
        print(f"  {k:16s}: {val}")


if __name__ == "__main__":
    main()
