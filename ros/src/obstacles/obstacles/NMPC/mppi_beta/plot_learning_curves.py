#!/usr/bin/env python3
"""
plot_learning_curves.py
-----------------------
Run multiple episodes and plot learning curves showing how performance improves
with persistent GP learning across episodes.

    python3 plot_learning_curves.py [num_episodes]
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

# Configuration
NUM_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CHECKPOINT_FILE = Path(__file__).with_name("gp_learned_checkpoint.pkl")

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

# Run multiple episodes and collect metrics
print("\n" + "="*70)
print(f"LEARNING CURVE: Running {NUM_EPISODES} episodes with persistent GP learning")
print("="*70 + "\n")

results = {
    'episode': [],
    'max_x': [],
    'final_x': [],
    'settle_err': [],
    'max_pitch': [],
    'solve_time_mean': [],
    'solve_time_p95': [],
    'reached_goal': [],
    'flipped': [],
}

for ep in range(1, NUM_EPISODES + 1):
    print(f"Episode {ep}/{NUM_EPISODES}...", end=" ", flush=True)

    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                     render=False, verbose=False,
                     gp_checkpoint=str(CHECKPOINT_FILE))

    m = out["metrics"]
    results['episode'].append(ep)
    results['max_x'].append(m['max_x'])
    results['final_x'].append(m['final_x'])
    results['settle_err'].append(m['settle_err'])
    results['max_pitch'].append(m['max_abs_pitch'])
    results['solve_time_mean'].append(m['solve_ms_mean'])
    results['solve_time_p95'].append(m['solve_ms_p95'])
    results['reached_goal'].append(1 if m['reached_goal'] else 0)
    results['flipped'].append(1 if m['flipped'] else 0)

    # Save checkpoint after each episode
    gp = out["gp"]
    gp.save_checkpoint(str(CHECKPOINT_FILE))

    status = "✓ GOAL" if m['reached_goal'] else "✗ SHORT"
    flip = "FLIP!" if m['flipped'] else "OK"
    print(f"max_x={m['max_x']:.1f}m {status} {flip}")

print("\n" + "="*70)
print("CREATING LEARNING CURVES...")
print("="*70 + "\n")

# Create comprehensive learning curves
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('MPPI Learning Curves: Persistent GP Learning Across Episodes', fontsize=16, fontweight='bold')

episodes = np.array(results['episode'])

# Plot 1: Distance Traveled (Main Metric)
ax = axes[0, 0]
ax.plot(episodes, results['max_x'], 'o-', linewidth=2, markersize=8, label='Max Traversal', color='#0072B2')
ax.axhline(8.0, color='green', linestyle='--', linewidth=2, label='Goal (8.0m)')
ax.fill_between(episodes, 0, results['max_x'], alpha=0.2, color='#0072B2')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Max Distance (m)', fontsize=11, fontweight='bold')
ax.set_title('Traversal Distance (Learning = Going Further)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim([0, max(results['max_x']) * 1.1])

# Plot 2: Final Position (Settling)
ax = axes[0, 1]
ax.plot(episodes, results['final_x'], 's-', linewidth=2, markersize=8, label='Final Position', color='#E69F00')
ax.axhline(8.0, color='green', linestyle='--', linewidth=2, label='Goal (8.0m)')
ax.fill_between(episodes, 0, results['final_x'], alpha=0.2, color='#E69F00')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Final Position (m)', fontsize=11, fontweight='bold')
ax.set_title('Final Settling Position', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.set_ylim([0, max(results['final_x']) * 1.1])

# Plot 3: Settling Error
ax = axes[0, 2]
ax.plot(episodes, results['settle_err'], '^-', linewidth=2, markersize=8, color='#56B4E9')
ax.fill_between(episodes, results['settle_err'], alpha=0.2, color='#56B4E9')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Settling Error (m)', fontsize=11, fontweight='bold')
ax.set_title('Settling Accuracy (Lower = Better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, max(results['settle_err']) * 1.2])

# Plot 4: Max Pitch (Safety)
ax = axes[1, 0]
ax.plot(episodes, results['max_pitch'], 'v-', linewidth=2, markersize=8, label='Max Pitch', color='#CC79A7')
ax.axhline(110.0, color='red', linestyle='--', linewidth=2, label='Flip Threshold (110°)')
ax.axhline(85.0, color='orange', linestyle=':', linewidth=2, label='Penalty Threshold (85°)')
ax.fill_between(episodes, results['max_pitch'], alpha=0.2, color='#CC79A7')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Max Pitch (°)', fontsize=11, fontweight='bold')
ax.set_title('Safety Margin (Lower = Safer)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 5: Solve Time (Efficiency)
ax = axes[1, 1]
ax.plot(episodes, results['solve_time_mean'], 'D-', linewidth=2, markersize=8, label='Mean', color='#009E73')
ax.plot(episodes, results['solve_time_p95'], 's--', linewidth=2, markersize=8, label='P95', color='#D55E00', alpha=0.7)
ax.fill_between(episodes, results['solve_time_mean'], results['solve_time_p95'], alpha=0.2, color='#009E73')
ax.axhline(50.0, color='red', linestyle='--', linewidth=1, label='Real-time budget (50ms)')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Solve Time (ms)', fontsize=11, fontweight='bold')
ax.set_title('Computation Speed (Lower = Faster)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 6: Success Rate
ax = axes[1, 2]
successes = np.cumsum(results['reached_goal'])
success_rate = successes / episodes * 100
ax.plot(episodes, success_rate, 'o-', linewidth=3, markersize=10, color='#F0E442')
ax.fill_between(episodes, 0, success_rate, alpha=0.2, color='#F0E442')
ax.set_xlabel('Episode', fontsize=11, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Success Rate', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 105])
ax.set_yticks([0, 25, 50, 75, 100])

# Add success markers on main plots
for ep, goal in zip(results['episode'], results['reached_goal']):
    if goal:
        axes[0, 0].axvline(ep, color='green', alpha=0.1, linewidth=2)

plt.tight_layout()
plt.savefig(Path(__file__).with_name('learning_curves.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved learning curves to: learning_curves.png\n")
plt.show()

# Print summary statistics
print("="*70)
print("LEARNING SUMMARY")
print("="*70)
print(f"Episodes run: {NUM_EPISODES}")
print(f"\nDistance traveled:")
print(f"  Episode 1: {results['max_x'][0]:.2f}m")
print(f"  Episode {NUM_EPISODES}: {results['max_x'][-1]:.2f}m")
if NUM_EPISODES > 1:
    improvement = (results['max_x'][-1] - results['max_x'][0]) / results['max_x'][0] * 100
    print(f"  Improvement: {improvement:+.1f}%")

print(f"\nGoal reaching:")
print(f"  Success rate: {sum(results['reached_goal'])}/{NUM_EPISODES} ({sum(results['reached_goal'])/NUM_EPISODES*100:.0f}%)")

print(f"\nSafety (max pitch):")
print(f"  Episode 1: {results['max_pitch'][0]:.1f}°")
print(f"  Episode {NUM_EPISODES}: {results['max_pitch'][-1]:.1f}°")
print(f"  Flips: {sum(results['flipped'])} episodes")

print(f"\nSolve time (mean):")
print(f"  Episode 1: {results['solve_time_mean'][0]:.1f}ms")
print(f"  Episode {NUM_EPISODES}: {results['solve_time_mean'][-1]:.1f}ms")

print("="*70)
