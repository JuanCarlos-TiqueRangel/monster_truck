#!/usr/bin/env python3
"""
run_ilc.py
----------
Iterative Learning Control on top of MPPI + SSGP: learn to go A->B FASTER, lap
over lap, on the repeated track.

    python3 run_ilc.py [num_episodes]

Each lap:
  * MPPI (feedback) + ILC feedforward drive the truck, using the persistent
    SSGP world-model (RLS linear + GP residual).
  * After the lap, ILC updates its path-indexed feedforward from the speed error,
    scaled by the learned model gain b0; the GP posterior is checkpointed.
Result: time-to-goal drops over episodes (GP-only was proven flat at ~5.55 s, so
any drop here is the ILC feedforward).

Set RENDER=True to watch a lap in MuJoCo (slow, real time).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save PNG, no display dependency
import matplotlib.pyplot as plt

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig
from ilc import ILC

NUM_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RENDER = True
SIM_TIME = 40.0
GOAL_X = 8.0

HERE = Path(__file__).resolve().parent
GP_CKPT = HERE / "gp_learned_checkpoint.pkl"
ILC_FILE = HERE / "ilc_profile.json"
PLOT = HERE / "ilc_learning.png"

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)
MPPI_CFG = MPPIConfig(
    dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
    r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
    v_ref_gain=0.6, v_cruise=1.2, seed=2)

# start each run clean so the curve is reproducible
GP_CKPT.unlink(missing_ok=True)
ILC_FILE.unlink(missing_ok=True)

ilc = ILC(goal_x=GOAL_X, v_des=PARAMS.v_max, lr=0.6, ff_max=4.0,
          pitch_mask_deg=70.0, pitch_gate_deg=100.0)

print("\n" + "=" * 84)
print("ITERATIVE LEARNING CONTROL  (MPPI feedback + ILC feedforward + persistent SSGP)")
print("=" * 84)
print(f"{'ep':>3} {'t2goal':>8} {'reached':>8} {'maxpitch':>9} {'mean_ff':>8} {'gate':>9}  note")
print("-" * 84)

ep_list, t2g_list, pitch_list = [], [], []

for ep in range(1, NUM_EPISODES + 1):
    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=SIM_TIME,
                      goal_x=GOAL_X, render=RENDER, verbose=False,
                      gp_checkpoint=str(GP_CKPT), ilc=ilc)

    h, m = out["history"], out["metrics"]
    x = np.array(h["x"]); t = np.array(h["t"])
    gi = np.where(x >= GOAL_X)[0]
    t2g = float(t[gi[0]]) if len(gi) else SIM_TIME
    reached = len(gi) > 0

    # ILC learns from this lap, scaled by the learned model gain b0
    diag = ilc.update(h, out["rls_b0"], t2g)

    # persist both memories
    out["gp"].save_checkpoint(str(GP_CKPT))
    ilc.save(str(ILC_FILE))

    ep_list.append(ep); t2g_list.append(t2g); pitch_list.append(m["max_abs_pitch"])
    note = "" if ep == 1 else (f"{(t2g_list[0]-t2g)/t2g_list[0]*100:+.0f}% vs ep1")
    print(f"{ep:>3} {t2g:>8.2f} {str(reached):>8} {m['max_abs_pitch']:>9.1f} "
          f"{diag['mean_ff']:>8.3f} {diag['gate']:>9}  {note}")

# ---- summary ----
t = np.array(t2g_list)
print("-" * 84)
print(f"first lap : {t[0]:.2f} s")
print(f"best lap  : {t.min():.2f} s  (episode {int(np.argmin(t))+1})")
print(f"last lap  : {t[-1]:.2f} s")
print(f"speedup   : {t[0]/t.min():.2f}x   ({(t[0]-t.min())/t[0]*100:+.1f}% faster)")
print("=" * 84 + "\n")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ILC on MPPI+SSGP: learning to reach the goal faster", fontweight="bold")

ax[0].plot(ep_list, t2g_list, "o-", lw=2.5, ms=8, color="#0072B2")
z = np.polyfit(ep_list, t2g_list, 1); p = np.poly1d(z)
ax[0].plot(ep_list, p(ep_list), "--", color="red", alpha=0.6, label=f"trend {z[0]:+.3f}s/ep")
ax[0].set_xlabel("episode", fontweight="bold")
ax[0].set_ylabel("time to goal (s)", fontweight="bold")
ax[0].set_title("Time A→B (down = learning)")
ax[0].grid(alpha=0.3); ax[0].legend()

ax[1].plot(ep_list, pitch_list, "^-", lw=2, ms=7, color="#CC79A7", label="max pitch")
ax[1].axhline(100, color="orange", ls=":", label="ILC gate 100°")
ax[1].axhline(110, color="red", ls="--", label="flip 110°")
ax[1].set_xlabel("episode", fontweight="bold")
ax[1].set_ylabel("max pitch (°)", fontweight="bold")
ax[1].set_title("Safety stays bounded")
ax[1].grid(alpha=0.3); ax[1].legend()

fig.tight_layout()
fig.savefig(PLOT, dpi=150, bbox_inches="tight")
print("\n" + "=" * 84)
print(f"✓ PLOT SAVED:  {PLOT}")
print(f"✓ ILC profile: {ILC_FILE}")
print("=" * 84)

# try to pop the image open with whatever viewer the system has (non-blocking)
import subprocess, shutil
for opener in ("xdg-open", "eog", "feh", "display"):
    if shutil.which(opener):
        try:
            subprocess.Popen([opener, str(PLOT)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  (opening with {opener}...)")
            break
        except Exception:
            pass
else:
    print(f"  Open it yourself:  xdg-open {PLOT}")
print()
