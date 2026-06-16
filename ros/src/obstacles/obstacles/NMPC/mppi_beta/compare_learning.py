#!/usr/bin/env python3
"""
compare_learning.py
-------------------
Compare learning WITH and WITHOUT persistent GP to diagnose the benefit.
Run 5 episodes fresh, then 5 episodes with persistent GP.

    python3 compare_learning.py

Shows whether persistent GP actually helps or if system is already optimal.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

MPPI_CFG = MPPIConfig(
    dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
    r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
    v_ref_gain=0.6, v_cruise=1.2, seed=2
)

CHECKPOINT = Path(__file__).with_name("gp_learned_checkpoint.pkl")

print("\n" + "="*80)
print("LEARNING DIAGNOSTIC: WITH vs WITHOUT Persistent GP")
print("="*80 + "\n")

# Phase 1: Fresh GP (no checkpoint)
print("PHASE 1: Fresh GP (reset checkpoint each time)")
print("-" * 80)

CHECKPOINT.unlink(missing_ok=True)  # Delete checkpoint

fresh_results = {
    'episode': [],
    'time_to_goal': [],
    'max_x': [],
    'max_pitch': []
}

for ep in range(1, 4):
    print(f"Episode {ep}/3 (fresh)...", end=" ", flush=True)

    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                     render=False, verbose=False, gp_checkpoint=None)

    m = out["metrics"]
    h = out["history"]
    x_arr = np.array(h["x"])
    t_arr = np.array(h["t"])

    goal_idx = np.where(x_arr >= 8.0)[0]
    time_to_goal = float(t_arr[goal_idx[0]]) if len(goal_idx) > 0 else 40.0

    fresh_results['episode'].append(ep)
    fresh_results['time_to_goal'].append(time_to_goal)
    fresh_results['max_x'].append(m['max_x'])
    fresh_results['max_pitch'].append(m['max_abs_pitch'])

    print(f"t2g={time_to_goal:.1f}s max_x={m['max_x']:.1f}m pitch={m['max_abs_pitch']:.1f}°")

# Phase 2: Persistent GP
print("\nPHASE 2: Persistent GP (checkpoint loaded)")
print("-" * 80)

CHECKPOINT.unlink(missing_ok=True)  # Delete checkpoint to start fresh

persistent_results = {
    'episode': [],
    'time_to_goal': [],
    'max_x': [],
    'max_pitch': []
}

for ep in range(1, 4):
    print(f"Episode {ep}/3 (persistent)...", end=" ", flush=True)

    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                     render=False, verbose=False,
                     gp_checkpoint=str(CHECKPOINT))

    m = out["metrics"]
    h = out["history"]
    x_arr = np.array(h["x"])
    t_arr = np.array(h["t"])

    goal_idx = np.where(x_arr >= 8.0)[0]
    time_to_goal = float(t_arr[goal_idx[0]]) if len(goal_idx) > 0 else 40.0

    persistent_results['episode'].append(ep)
    persistent_results['time_to_goal'].append(time_to_goal)
    persistent_results['max_x'].append(m['max_x'])
    persistent_results['max_pitch'].append(m['max_abs_pitch'])

    # Save checkpoint
    gp = out["gp"]
    gp.save_checkpoint(str(CHECKPOINT))

    print(f"t2g={time_to_goal:.1f}s max_x={m['max_x']:.1f}m pitch={m['max_abs_pitch']:.1f}°")

# Analysis
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

fresh_times = np.array(fresh_results['time_to_goal'])
persistent_times = np.array(persistent_results['time_to_goal'])

print(f"\nTime to Goal (seconds):")
print(f"  Fresh GP:")
for ep, t in zip(fresh_results['episode'], fresh_results['time_to_goal']):
    print(f"    Episode {ep}: {t:.2f}s")
print(f"  Persistent GP:")
for ep, t in zip(persistent_results['episode'], persistent_results['time_to_goal']):
    print(f"    Episode {ep}: {t:.2f}s")

print(f"\nAverage performance:")
print(f"  Fresh GP: {fresh_times.mean():.2f}s ± {fresh_times.std():.2f}s")
print(f"  Persistent GP: {persistent_times.mean():.2f}s ± {persistent_times.std():.2f}s")

improvement = (fresh_times.mean() - persistent_times.mean()) / fresh_times.mean() * 100
print(f"\nImprovement with persistent GP: {improvement:+.1f}%")

if improvement > 5:
    print("✓ PERSISTENT GP IS HELPING (significant improvement)")
elif improvement > 0:
    print("~ PERSISTENT GP HELPS SLIGHTLY")
else:
    print("✗ NO IMPROVEMENT - System may already be optimal")
    print("  Possible reasons:")
    print("  1. The controller tuning is already near-optimal for this task")
    print("  2. The residual learning is limited (pitch is hard to learn)")
    print("  3. Temperature/sampling already finds good solutions")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Persistent GP Learning: Diagnostic Comparison', fontsize=14, fontweight='bold')

# Plot 1: Time to goal
ax = axes[0]
x_pos = np.arange(3)
width = 0.35
ax.bar(x_pos - width/2, fresh_times, width, label='Fresh GP', color='#FF6B6B', alpha=0.8)
ax.bar(x_pos + width/2, persistent_times, width, label='Persistent GP', color='#4ECDC4', alpha=0.8)
ax.set_xlabel('Episode', fontweight='bold')
ax.set_ylabel('Time to Goal (s)', fontweight='bold')
ax.set_title('Time to Reach Goal (Lower = Better)')
ax.set_xticks(x_pos)
ax.set_xticklabels([1, 2, 3])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Distance traveled
ax = axes[1]
ax.bar(x_pos - width/2, fresh_results['max_x'], width, label='Fresh GP', color='#FF6B6B', alpha=0.8)
ax.bar(x_pos + width/2, persistent_results['max_x'], width, label='Persistent GP', color='#4ECDC4', alpha=0.8)
ax.axhline(8.0, color='green', linestyle='--', linewidth=2, label='Goal (8.0m)')
ax.set_xlabel('Episode', fontweight='bold')
ax.set_ylabel('Max Distance (m)', fontweight='bold')
ax.set_title('Traversal Distance')
ax.set_xticks(x_pos)
ax.set_xticklabels([1, 2, 3])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(Path(__file__).with_name('learning_diagnostic.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved diagnostic plot to: learning_diagnostic.png\n")

plt.show()
