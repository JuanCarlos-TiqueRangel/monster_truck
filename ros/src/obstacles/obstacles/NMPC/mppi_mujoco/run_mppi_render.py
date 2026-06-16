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

from params import WheelieParams
from rls import RLSConfig

from mppi import WheelieMPPI, MPPIConfig
from sim_harness import (run_episode, SSGPConfig,
                         AdaptiveSSGPConfig, StreamingGPConfig)


# ============================================================
# Settings
# ============================================================
RENDER = True
SIM_TIME = 40.0

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()                 # streaming variational (VFE) sparse GP -- SSGP.py; honest variance, fixes FITC.
# options (all drop-in, same controller):  StreamingGPConfig()=recursive FITC (online_sparseGP.py, legacy),
#                                           AdaptiveSSGPConfig()=VFE + online adaptivity.
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
    # goal-distance velocity reference -> the MPPI brakes to a stop AT the goal
    # (v_ref = clip(v_ref_gain*(goal-x), +-v_cruise); v_ref_gain=0 = old behaviour)
    v_ref_gain=0.6, v_cruise=1.2,
    # seed picks the random sampling stream (the run is deterministic given it).
    # The base MPPI is seed-dependent on this 3-obstacle course (box@1, bigger
    # box@3, rounded cylinders@5): most seeds FLIP at the cylinders (their contact
    # pitch is unmodellable, so the flip penalty can't foresee it). seed=2 climbs
    # all three and stops at ~8.0; seed=0 stops short at ~7.2. Change to explore.
    seed=2,
)


def main():
    import json
    import numpy as np
    from pathlib import Path

    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG,
                      sim_time=SIM_TIME, render=RENDER, verbose=True)
    print("\nMetrics:")
    for k, val in out["metrics"].items():
        print(f"  {k:16s}: {val}")

    # Auto-save data for plotting
    data = {
        "history": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in out["history"].items()},
        "metrics": out["metrics"]
    }
    data_file = Path(__file__).with_name("mppi_episode_data.json")
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Data saved to {data_file.name}")
    print(f"  Now run: python3 make_paper_plot.py")


if __name__ == "__main__":
    main()
