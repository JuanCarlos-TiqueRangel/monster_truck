#!/usr/bin/env python3
"""
diagnose_learning.py
--------------------
Diagnose why learning isn't improving by checking:
1. Is persistent GP actually helping?
2. What metrics are changing across episodes?
3. Is the system already optimal?

    python3 diagnose_learning.py
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
print("LEARNING DIAGNOSTIC: Why Isn't Performance Improving?")
print("="*80 + "\n")

# Run 10 episodes and track detailed metrics
results = {
    'episode': [],
    'time_to_goal': [],
    'max_x': [],
    'final_x': [],
    'max_pitch': [],
    'settle_err': [],
    'mean_tau': [],
    'gp_n_seen': [],
}

# Start fresh
CHECKPOINT.unlink(missing_ok=True)

for ep in range(1, 11):
    print(f"Episode {ep}/10...", end=" ", flush=True)

    ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                     render=False, verbose=False,
                     gp_checkpoint=str(CHECKPOINT))

    m = out["metrics"]
    h = out["history"]
    gp = out["gp"]

    x_arr = np.array(h["x"])
    t_arr = np.array(h["t"])
    tau_arr = np.array(h["tau"])

    goal_idx = np.where(x_arr >= 8.0)[0]
    time_to_goal = float(t_arr[goal_idx[0]]) if len(goal_idx) > 0 else 40.0

    results['episode'].append(ep)
    results['time_to_goal'].append(time_to_goal)
    results['max_x'].append(m['max_x'])
    results['final_x'].append(m['final_x'])
    results['max_pitch'].append(m['max_abs_pitch'])
    results['settle_err'].append(m['settle_err'])
    results['mean_tau'].append(m['mean_abs_tau'])
    results['gp_n_seen'].append(gp.n_seen)

    # Save checkpoint
    gp.save_checkpoint(str(CHECKPOINT))

    print(f"t2g={time_to_goal:.2f}s GP(n_seen={gp.n_seen})")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80 + "\n")

# Statistical analysis
times = np.array(results['time_to_goal'])
max_xs = np.array(results['max_x'])
pitches = np.array(results['max_pitch'])
settle_errs = np.array(results['settle_err'])

print("TIME TO GOAL:")
print(f"  Episode 1-3: {times[:3].mean():.2f}s ± {times[:3].std():.2f}s")
print(f"  Episode 8-10: {times[-3:].mean():.2f}s ± {times[-3:].std():.2f}s")
change = (times[-3:].mean() - times[:3].mean()) / times[:3].mean() * 100
print(f"  Change: {change:+.1f}% (negative = faster)\n")

print("MAX DISTANCE (traversal):")
print(f"  Episode 1-3: {max_xs[:3].mean():.2f}m ± {max_xs[:3].std():.2f}m")
print(f"  Episode 8-10: {max_xs[-3:].mean():.2f}m ± {max_xs[-3:].std():.2f}m")
change = (max_xs[-3:].mean() - max_xs[:3].mean()) / max_xs[:3].mean() * 100
print(f"  Change: {change:+.1f}%\n")

print("MAX PITCH (safety):")
print(f"  Episode 1-3: {pitches[:3].mean():.2f}° ± {pitches[:3].std():.2f}°")
print(f"  Episode 8-10: {pitches[-3:].mean():.2f}° ± {pitches[-3:].std():.2f}°")
change = (pitches[-3:].mean() - pitches[:3].mean()) / pitches[:3].mean() * 100
print(f"  Change: {change:+.1f}% (negative = safer)\n")

print("SETTLING ERROR (accuracy):")
print(f"  Episode 1-3: {settle_errs[:3].mean():.3f}m ± {settle_errs[:3].std():.3f}m")
print(f"  Episode 8-10: {settle_errs[-3:].mean():.3f}m ± {settle_errs[-3:].std():.3f}m")
change = (settle_errs[-3:].mean() - settle_errs[:3].mean()) / settle_errs[:3].mean() * 100
print(f"  Change: {change:+.1f}%\n")

print("GP LEARNING:")
print(f"  Episode 1 GP n_seen: {results['gp_n_seen'][0]}")
print(f"  Episode 10 GP n_seen: {results['gp_n_seen'][-1]}")
total_learned = results['gp_n_seen'][-1]
print(f"  Total observations learned: {total_learned}\n")

# Diagnosis
print("="*80)
print("DIAGNOSIS")
print("="*80 + "\n")

time_variation = times.std() / times.mean()
if time_variation < 0.05:
    print("✓ TIME TO GOAL: Extremely consistent (std < 5%)")
    print("  → System is STABLE and REPEATABLE\n")
else:
    print(f"⚠ TIME TO GOAL: High variation ({time_variation*100:.1f}%)\n")

time_improvement = (times[0] - times[-1]) / times[0] * 100
if abs(time_improvement) < 1:
    print(f"✗ NO IMPROVEMENT: Time to goal {time_improvement:+.2f}%")
    print("  → System may already be OPTIMAL")
    print("  → Or learning is LIMITED by controller tuning\n")
elif time_improvement > 10:
    print(f"✓ SIGNIFICANT IMPROVEMENT: {time_improvement:+.1f}%")
    print("  → Learning IS working!\n")
else:
    print(f"~ SLIGHT IMPROVEMENT: {time_improvement:+.1f}%")
    print("  → Learning is happening but subtle\n")

if times.mean() < 5.0:
    print("✓ FAST GOAL-REACHING: < 5 seconds")
    print("  → Controller is EXCELLENT at this task\n")

print("="*80)
print("POSSIBLE REASONS FOR NO IMPROVEMENT:")
print("="*80)
print("""
1. SYSTEM IS ALREADY OPTIMAL
   - The MPPI controller + RLS model is already excellent
   - The persistent GP adds minimal benefit
   - Further improvement needs different approach (hardware, perception)

2. GP LEARNING IS LIMITED
   - Pitch residual is irreducible (needs external sensors)
   - Only velocity/position residuals are learnable
   - These are already small, so limited improvement space

3. MPPI EXPLORATION IS GOOD ENOUGH
   - The sampling-based approach already finds good policies
   - Adding learned residuals provides < 1% improvement

4. TEMPERATURE/TUNING IS OPTIMAL
   - Current temperature=10.0 already balances exploration/exploitation
   - No gain from learning more

WHAT TO TRY INSTEAD:
→ Use compare_learning.py to verify persistent GP helps at all
→ Try different obstacles (faster/harder climbs)
→ Adjust controller parameters (N, num_samples, temperature)
→ Use model-predictive approach (NMPC vs sampling)
→ Add external perception (camera) for pitch estimation
""")

# Create plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Learning Diagnostic: 10 Episodes', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(results['episode'], results['time_to_goal'], 'o-', linewidth=2, markersize=8, color='#0072B2')
ax.set_ylabel('Time to Goal (s)', fontweight='bold')
ax.set_title(f'Time to Goal (Variation: {time_variation*100:.1f}%)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.axhline(times.mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean: {times.mean():.2f}s')
ax.legend()

ax = axes[0, 1]
ax.plot(results['episode'], results['max_x'], 's-', linewidth=2, markersize=8, color='#E69F00')
ax.axhline(8.0, color='green', linestyle='--', linewidth=2, label='Goal (8.0m)')
ax.set_ylabel('Max Distance (m)', fontweight='bold')
ax.set_title('Traversal Distance', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 0]
ax.plot(results['episode'], results['max_pitch'], '^-', linewidth=2, markersize=8, color='#CC79A7')
ax.axhline(110.0, color='red', linestyle='--', alpha=0.5, label='Flip (110°)')
ax.set_ylabel('Max Pitch (°)', fontweight='bold')
ax.set_title('Safety Margin', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1, 1]
ax.plot(results['episode'], results['gp_n_seen'], 'd-', linewidth=2, markersize=8, color='#009E73')
ax.set_xlabel('Episode', fontweight='bold')
ax.set_ylabel('GP Observations', fontweight='bold')
ax.set_title('GP Learning Progress', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(Path(__file__).with_name('diagnostic.png'), dpi=150, bbox_inches='tight')
print(f"✓ Saved diagnostic plot to: diagnostic.png\n")
plt.show()
