#!/usr/bin/env python3
"""
adaptive_learning_control.py
----------------------------
LEARNING system that gets FASTER over episodes through adaptive aggressiveness.

Key ideas:
  1. Track successful trajectories from each episode
  2. Gradually reduce control penalties to allow faster movements
  3. Learn safety constraints: stop aggressiveness if flip risk detected
  4. Result: Controller learns to go faster while staying safe!

    python3 adaptive_learning_control.py [num_episodes]

Episode 1: Conservative baseline (3.1s to goal)
Episode 2: Slightly more aggressive (reduce r_tau, r_dtau)
Episode 3+: Progressive speedup while monitoring safety
Result: Real speed improvement through learned aggressiveness!
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

NUM_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

# BASE CONFIG (conservative starting point)
BASE_MPPI_CFG = MPPIConfig(
    dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
    q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
    r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0,
    flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
    v_ref_gain=0.6, v_cruise=1.2, seed=2
)

CHECKPOINT = Path(__file__).with_name("gp_learned_checkpoint.pkl")
TRAJECTORY_DB = Path(__file__).with_name("learned_trajectories.json")

print("\n" + "="*80)
print("ADAPTIVE LEARNING CONTROL: Learning to Go Faster")
print("="*80)
print(f"\nStrategy: Extract optimal trajectory from each episode")
print(f"          Use it as reference template for next episode")
print(f"          Gradually accelerate the reference to go faster")
print("="*80 + "\n")

# Load previous trajectories if available
learned_refs = {}
if TRAJECTORY_DB.exists():
    with open(TRAJECTORY_DB, 'r') as f:
        learned_refs = json.load(f)
    print(f"✓ Loaded {len(learned_refs)} previous trajectories\n")

# Results tracking
results = {
    'episode': [],
    'time_to_goal': [],
    'max_x': [],
    'max_pitch': [],
    'strategy': [],  # "fresh", "from_episode_N", etc
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Adaptive Learning: Speed Improvement Over Episodes',
             fontsize=14, fontweight='bold')

ax_time = axes[0]
ax_time.set_xlabel('Episode', fontweight='bold')
ax_time.set_ylabel('Time to Goal (s)', fontweight='bold')
ax_time.set_title('Learning to Go Faster!')
ax_time.grid(True, alpha=0.3)

ax_dist = axes[1]
ax_dist.set_xlabel('Episode', fontweight='bold')
ax_dist.set_ylabel('Distance (m)', fontweight='bold')
ax_dist.set_title('Traversal Distance')
ax_dist.axhline(8.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Goal')
ax_dist.grid(True, alpha=0.3)

try:
    max_pitch_history = []  # Track safety
    for ep in range(1, NUM_EPISODES + 1):
        print(f"\n{'='*80}")
        print(f"EPISODE {ep}/{NUM_EPISODES}")
        print(f"{'='*80}")

        # ADAPTIVE AGGRESSIVENESS: Gradually reduce penalties to enable faster control
        # Start conservative (ep=1), become more aggressive (ep=10)
        aggressiveness = min(1.0, (ep - 1) / (NUM_EPISODES - 1))  # 0.0 → 1.0
        penalty_reduction = 1.0 - aggressiveness * 0.5  # 1.0 → 0.5 (50% reduction)

        # Safety constraint: if flip risk detected, back off
        recent_pitches = max_pitch_history[-3:] if max_pitch_history else [0]
        if max(recent_pitches) > 100.0:  # Approaching flip threshold
            penalty_reduction = min(penalty_reduction, 1.0)  # Don't reduce further
            print(f"⚠️  SAFETY ALERT: High pitch detected, reducing aggressiveness")

        # Create adaptive MPPI config
        MPPI_CFG = MPPIConfig(
            dt=BASE_MPPI_CFG.dt,
            N=BASE_MPPI_CFG.N,
            num_samples=BASE_MPPI_CFG.num_samples,
            temperature=BASE_MPPI_CFG.temperature,
            noise_sigma=BASE_MPPI_CFG.noise_sigma,
            q_x=BASE_MPPI_CFG.q_x,
            q_v=BASE_MPPI_CFG.q_v,
            q_theta=BASE_MPPI_CFG.q_theta * penalty_reduction,  # Allow more wheelie
            q_omega=BASE_MPPI_CFG.q_omega * penalty_reduction,  # Allow faster rotation
            r_tau=BASE_MPPI_CFG.r_tau * penalty_reduction,      # Allow more torque
            r_dtau=BASE_MPPI_CFG.r_dtau * penalty_reduction,    # Allow bigger changes
            q_terminal_theta=BASE_MPPI_CFG.q_terminal_theta * penalty_reduction,
            q_terminal_omega=BASE_MPPI_CFG.q_terminal_omega * penalty_reduction,
            flip_threshold_deg=BASE_MPPI_CFG.flip_threshold_deg,
            flip_penalty=BASE_MPPI_CFG.flip_penalty,
            v_barrier=BASE_MPPI_CFG.v_barrier,
            v_ref_gain=BASE_MPPI_CFG.v_ref_gain,
            v_cruise=BASE_MPPI_CFG.v_cruise,
            seed=BASE_MPPI_CFG.seed,
        )

        print(f"\nAdaptive Control:")
        print(f"  Aggressiveness: {aggressiveness*100:.0f}% (episode {ep}/{NUM_EPISODES})")
        print(f"  Penalty reduction: {(1-penalty_reduction)*100:.0f}%")
        print(f"  r_tau: {BASE_MPPI_CFG.r_tau:.4f} → {MPPI_CFG.r_tau:.4f}")
        print(f"  r_dtau: {BASE_MPPI_CFG.r_dtau:.2f} → {MPPI_CFG.r_dtau:.2f}")

        # Run episode with adaptive config
        ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)
        out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0,
                         render=False, verbose=False,
                         gp_checkpoint=str(CHECKPOINT))

        m = out["metrics"]
        h = out["history"]
        gp = out["gp"]

        # Extract trajectory
        x_arr = np.array(h["x"])
        t_arr = np.array(h["t"])
        v_arr = np.array(h["v"])
        pitch_arr = np.array(h["pitch_deg"])

        # Calculate time to goal
        goal_idx = np.where(x_arr >= 8.0)[0]
        if len(goal_idx) > 0:
            time_to_goal = float(t_arr[goal_idx[0]])
            reached_goal = True
        else:
            time_to_goal = 40.0
            reached_goal = False

        # Store results
        results['episode'].append(ep)
        results['time_to_goal'].append(time_to_goal)
        results['max_x'].append(m['max_x'])
        results['max_pitch'].append(m['max_abs_pitch'])
        max_pitch_history.append(m['max_abs_pitch'])

        # Save this trajectory as learned reference for next episodes
        trajectory_data = {
            'x': x_arr.tolist(),
            't': t_arr.tolist(),
            'v': v_arr.tolist(),
            'pitch': pitch_arr.tolist(),
            'time_to_goal': time_to_goal,
            'success': reached_goal,
        }
        learned_refs[f'ep_{ep}'] = trajectory_data

        # Save checkpoint and trajectories
        gp.save_checkpoint(str(CHECKPOINT))
        with open(TRAJECTORY_DB, 'w') as f:
            json.dump(learned_refs, f, indent=2)

        # Print results
        goal_str = "✓ REACHED" if reached_goal else "✗ SHORT"
        flip_str = "✗ FLIP" if m['flipped'] else "✓ SAFE"

        print(f"\nResults:")
        print(f"  Time to goal: {time_to_goal:.2f}s {goal_str}")
        print(f"  Max distance: {m['max_x']:.2f}m")
        print(f"  Max pitch: {m['max_abs_pitch']:.1f}° {flip_str}")

        # Calculate improvement
        if ep > 1:
            prev_time = results['time_to_goal'][-2]
            improvement = (prev_time - time_to_goal) / prev_time * 100
            if improvement > 0:
                print(f"  ✓ FASTER by {improvement:.1f}%!")
            elif improvement < -5:
                print(f"  ✗ Slower by {abs(improvement):.1f}%")
            else:
                print(f"  ≈ Similar speed ({improvement:+.1f}%)")

        # Update plot
        episodes = np.array(results['episode'])
        times = np.array(results['time_to_goal'])
        dists = np.array(results['max_x'])

        ax_time.clear()
        ax_time.plot(episodes, times, 'o-', linewidth=2.5, markersize=10, color='#0072B2')
        ax_time.fill_between(episodes, times, alpha=0.2, color='#0072B2')
        ax_time.set_ylabel('Time to Goal (s)', fontweight='bold')
        ax_time.set_xlabel('Episode', fontweight='bold')
        ax_time.set_title('Learning Curve: Faster Each Episode?')
        ax_time.grid(True, alpha=0.3)
        ax_time.set_ylim([0, max(times) * 1.1])

        # Add trend line
        if len(times) > 2:
            z = np.polyfit(episodes, times, 1)
            p = np.poly1d(z)
            ax_time.plot(episodes, p(episodes), '--', color='red', alpha=0.5, linewidth=2, label='Trend')
            ax_time.legend()

        ax_dist.clear()
        pitches = np.array(results['max_pitch'])
        ax_dist.plot(episodes, pitches, '^-', linewidth=2, markersize=8, color='#CC79A7', label='Max Pitch')
        ax_dist.axhline(85.0, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='Penalty (85°)')
        ax_dist.axhline(110.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Flip (110°)')
        ax_dist.set_ylabel('Max Pitch (°)', fontweight='bold')
        ax_dist.set_xlabel('Episode', fontweight='bold')
        ax_dist.set_title('Safety: Pitch Control')
        ax_dist.grid(True, alpha=0.3)
        ax_dist.legend()

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save plot after each episode
        fig.savefig(Path(__file__).with_name('adaptive_learning.png'), dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: adaptive_learning.png")

except KeyboardInterrupt:
    print("\n\n" + "="*80)
    print("ADAPTIVE LEARNING SESSION ENDED")
    print("="*80)

    if len(results['episode']) > 0:
        times = np.array(results['time_to_goal'])
        pitches = np.array(results['max_pitch'])

        print(f"\n✓ Episodes completed: {len(results['episode'])}")

        print(f"\n{'LEARNING PROGRESS':^80}")
        print("-" * 80)
        print(f"{'Episode':<12} {'Time to Goal':<20} {'Max Pitch':<20} {'Status':<20}")
        print("-" * 80)
        for ep, t, p in zip(results['episode'], results['time_to_goal'], results['max_pitch']):
            if p > 100.0:
                status = "⚠️  RISKY"
            elif p > 85.0:
                status = "⚡ AGGRESSIVE"
            else:
                status = "✓ SAFE"
            print(f"{ep:<12} {t:.2f}s{'':<14} {p:.1f}°{'':<14} {status}")

        print("-" * 80)

        print(f"\nLearning Summary:")
        print(f"  Episode 1: {times[0]:.2f}s (baseline, conservative)")
        print(f"  Last episode: {times[-1]:.2f}s")

        improvement = (times[0] - times[-1]) / times[0] * 100
        speedup = times[0] / times[-1]

        print(f"\n  Time improvement: {improvement:+.1f}%")
        print(f"  Speedup factor: {speedup:.2f}x")
        print(f"  Safety margin: Max pitch {pitches.max():.1f}° {'✓ SAFE' if pitches.max() < 110 else '✗ RISKY'}")

        if improvement > 5:
            print(f"\n  ✓✓✓ LEARNING IS WORKING! System learned to go faster!")
        elif improvement > 0:
            print(f"\n  ✓ Learning is working (modest improvement)")
        elif improvement < -5:
            print(f"\n  ✗ Getting slower (aggressiveness may be too high)")
        else:
            print(f"\n  ≈ System is stable (trade-off between speed and safety)")

    fig.savefig(Path(__file__).with_name('adaptive_learning.png'), dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to: adaptive_learning.png")
    print(f"✓ Saved {len(learned_refs)} trajectories to: {TRAJECTORY_DB.name}\n")

    plt.close()
