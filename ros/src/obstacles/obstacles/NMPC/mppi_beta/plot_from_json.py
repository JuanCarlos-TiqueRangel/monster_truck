#!/usr/bin/env python3
"""
plot_from_json.py
-----------------
Plot learning curves IMMEDIATELY from saved JSON data WITHOUT re-running episodes.
Fast, instant visualization of collected data.

    python3 plot_from_json.py

Loads: mppi_beta_episode_data.json (saved by run_mppi_render.py)
Outputs: learning_curves_from_json.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

json_file = Path(__file__).with_name("mppi_beta_episode_data.json")

# Check if file exists
if not json_file.exists():
    print(f"\n❌ ERROR: {json_file} not found!")
    print("   Run run_mppi_render.py first to collect episode data.\n")
    exit(1)

# Load data
print(f"\n✓ Loading data from {json_file.name}...", end=" ", flush=True)
with open(json_file, 'r') as f:
    data = json.load(f)

history = data['history']
metrics = data['metrics']
print("Done!\n")

# Extract time to reach goal
x_arr = np.array(history['x'])
t_arr = np.array(history['t'])

goal_idx = np.where(x_arr >= 8.0)[0]
if len(goal_idx) > 0:
    time_to_goal = float(t_arr[goal_idx[0]])
    reached_goal = True
else:
    time_to_goal = 40.0
    reached_goal = False

# Create quick plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('MPPI Episode Data: Instant Plot from JSON', fontsize=14, fontweight='bold')

# Plot 1: Distance over time
ax = axes[0, 0]
ax.plot(t_arr, x_arr, 'b-', linewidth=2, label='Position')
ax.axhline(8.0, color='green', linestyle='--', linewidth=2, label='Goal (8.0m)')
if reached_goal:
    ax.axvline(time_to_goal, color='red', linestyle=':', linewidth=2, label=f'Goal reached at {time_to_goal:.1f}s')
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Position (m)', fontweight='bold')
ax.set_title('Trajectory: Distance vs Time', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Velocity
ax = axes[0, 1]
v_arr = np.array(history['v'])
ax.plot(t_arr, v_arr, 'g-', linewidth=2)
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Velocity (m/s)', fontweight='bold')
ax.set_title('Velocity Profile', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Pitch angle
ax = axes[1, 0]
pitch_arr = np.array(history['pitch_wrapped'])
ax.plot(t_arr, pitch_arr, 'r-', linewidth=2)
ax.axhline(85, color='orange', linestyle=':', linewidth=2, label='Penalty (85°)')
ax.axhline(-85, color='orange', linestyle=':', linewidth=2)
ax.axhline(110, color='red', linestyle='--', linewidth=2, label='Flip (110°)')
ax.axhline(-110, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Pitch (°)', fontweight='bold')
ax.set_title('Pitch Angle (Safety)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Control input
ax = axes[1, 1]
tau_arr = np.array(history['tau'])
ax.plot(t_arr, tau_arr, 'purple', linewidth=2)
ax.axhline(0, color='black', linestyle='-', alpha=0.3)
ax.set_xlabel('Time (s)', fontweight='bold')
ax.set_ylabel('Torque (N·m)', fontweight='bold')
ax.set_title('Control Input', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save and show
output_file = Path(__file__).with_name("learning_curves_from_json.png")
fig.savefig(str(output_file), dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_file.name}\n")

# Print metrics
print("="*70)
print("EPISODE METRICS")
print("="*70)
print(f"\nGoal reaching:")
print(f"  Time to reach goal: {time_to_goal:.1f}s {'✓ REACHED' if reached_goal else '✗ SHORT'}")
print(f"  Max distance: {metrics['max_x']:.2f}m")
print(f"  Final position: {metrics['final_x']:.2f}m")

print(f"\nSafety:")
print(f"  Max pitch: {metrics['max_abs_pitch']:.1f}°")
print(f"  Flipped: {'Yes ✗' if metrics['flipped'] else 'No ✓'}")

print(f"\nPerformance:")
print(f"  Solve time (mean): {metrics['solve_ms_mean']:.1f}ms")
print(f"  Solve time (p95): {metrics['solve_ms_p95']:.1f}ms")
print(f"  Mean abs torque: {metrics['mean_abs_tau']:.2f}N·m")

print("="*70)
print("✓ Plot displayed!\n")

plt.show()
