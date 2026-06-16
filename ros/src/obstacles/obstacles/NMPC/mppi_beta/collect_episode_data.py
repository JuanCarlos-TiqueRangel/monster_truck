#!/usr/bin/env python3
"""
collect_episode_data.py
-----------------------
Collect MPPI Beta episode data ONCE and save to file.
Then make_paper_plot.py just loads and plots (no re-simulation).

    python3 collect_episode_data.py
"""
import json
import numpy as np
from pathlib import Path

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

MPPI_CFG = MPPIConfig(
    dt=0.05,
    N=40,  # Beta: doubled horizon for flip prevention
    num_samples=2048,
    temperature=10.0,
    noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
    r_tau=0.05, r_dtau=1.0,
    q_terminal_theta=6.0, q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
    v_ref_gain=0.6, v_cruise=1.2,
    seed=2,
)

OUTPUT_FILE = Path(__file__).with_name("mppi_beta_episode_data.json")

print("Running MPPI Beta episode (40s)...")
ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG,
                  sim_time=40.0, render=False, verbose=True)

# Convert numpy arrays to lists for JSON serialization
data = {
    "history": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in out["history"].items()},
    "metrics": out["metrics"]
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n✓ Saved episode data to {OUTPUT_FILE.name}")
print(f"  Now run: python3 make_paper_plot.py")
