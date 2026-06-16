#!/usr/bin/env python3
"""
watch_learned.py
----------------
Watch ONE lap in the MuJoCo viewer using the weights learned by learn_cost.py.

    python3 watch_learned.py            # watch the LEARNED cost (learned_cost.json)
    python3 watch_learned.py baseline   # watch the original baseline, to compare

Real-time, so a ~5 s lap takes ~5 s. The lap ends when the truck crosses the goal.
"""
import sys
import json
from pathlib import Path

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

HERE = Path(__file__).resolve().parent
LEARNED = HERE / "learned_cost.json"

# match the training setup: v_max lifted so the speed barrier doesn't cap us
PARAMS = WheelieParams(v_max=4.0, v_min=-4.0)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

# pick weights: learned (default) or baseline
want_baseline = len(sys.argv) > 1 and sys.argv[1].lower().startswith("base")
if want_baseline:
    v_cruise, v_ref_gain, q_theta = 1.2, 0.6, 6.0
    label = "BASELINE"
elif LEARNED.exists():
    d = json.load(open(LEARNED))
    v_cruise, v_ref_gain, q_theta = d["v_cruise"], d["v_ref_gain"], d["q_theta"]
    label = "LEARNED"
else:
    v_cruise, v_ref_gain, q_theta = 1.2, 0.6, 6.0
    label = "BASELINE (no learned_cost.json yet -- run learn_cost.py first)"

print("\n" + "=" * 60)
print(f"WATCHING: {label}")
print(f"  v_cruise={v_cruise:.2f}  v_ref_gain={v_ref_gain:.2f}  q_theta={q_theta:.1f}")
print("=" * 60)

cfg = MPPIConfig(
    dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=float(q_theta), q_omega=60.0,
    r_tau=0.05, r_dtau=1.0, q_terminal_theta=float(q_theta), q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
    v_ref_gain=float(v_ref_gain), v_cruise=float(v_cruise), seed=2)

ctrl = WheelieMPPI(PARAMS, cfg, GP_CFG)
out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                  goal_x=8.0, render=True, verbose=False, gp_checkpoint=None)

m = out["metrics"]
print(f"\nreached goal: {m['reached_goal']}   max pitch: {m['max_abs_pitch']:.0f}°   "
      f"final x: {m['final_x']:.2f} m\n")
