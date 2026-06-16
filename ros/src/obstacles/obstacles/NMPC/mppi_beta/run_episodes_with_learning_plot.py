#!/usr/bin/env python3
"""
run_episodes_with_learning_plot.py
-----------------------------------
Run MPPI episodes continuously with learning visualization.

Two modes:
  RENDER=True:  Show MuJoCo viewer (real-time), update plot after each episode
  RENDER=False: Headless (fast), live plot updates during episodes

    python3 run_episodes_with_learning_plot.py

Press Ctrl+C to stop.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

# Configuration
CHECKPOINT_FILE = Path(__file__).with_name("gp_learned_checkpoint.pkl")
RENDER = True  # Set to False for headless (faster); True to see MuJoCo viewer
SIM_TIME = 40.0

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

# Store results
results = {
    'episode': [],
    'time_to_goal': [],
    'goal_reached': [],
    'max_distance': [],
    'max_pitch': [],
}

print("\n" + "="*80)
print("CONTINUOUS LEARNING WITH PERSISTENT GP")
print("="*80)
print(f"\nRENDER MODE: {RENDER}")
if RENDER:
    print("⚠️  SLOW MODE: Simulation runs in real-time (40s episode = 40s wall time)")
    print("📺 MuJoCo viewer window opens for each episode")
    print("📊 Plot updates after episode completes")
else:
    print("⚡ FAST MODE: Headless (no viewer, ~1-2s per episode)")
    print("📊 Live plot updates during episodes")
print("\nCar restarts immediately when reaching goal")
print("Press Ctrl+C to stop")
print("="*80 + "\n")

# Only create plot if RENDER=False (live updates)
if not RENDER:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MPPI Learning: Time to Reach Goal (A→B) - Continuous Episodes',
                 fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time to Reach Goal (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Learning = Going From A→B Faster!', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    line1, = ax1.plot([], [], 'o-', linewidth=3, markersize=10, color='#0072B2', label='Time to Goal')
    ax1.set_ylim([0, SIM_TIME])

    ax2 = axes[1]
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Traversal Distance & Safety', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    line2, = ax2.plot([], [], 's-', linewidth=2, markersize=8, color='#E69F00', label='Max Distance')
    line3, = ax2.plot([], [], '^-', linewidth=2, markersize=8, color='#CC79A7', label='Max Pitch (°)', alpha=0.7)
    ax2.axhline(8.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Goal (8.0m)')
    ax2.axhline(110.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Flip (110°)')
    ax2.set_ylim([0, max(50, SIM_TIME)])
    ax2.legend(loc='upper left', fontsize=10)

    fig.tight_layout()
    plt.ion()  # Interactive mode
else:
    fig, axes = None, None
    ax1, ax2 = None, None
    line1, line2, line3 = None, None, None

# Run episodes continuously
ep = 0
try:
    while True:
        ep += 1

        print(f"\n{'='*80}")
        print(f"EPISODE {ep}")
        print(f"{'='*80}")

        ctrl = WheelieMPPI(PARAMS, MPPI_CFG, GP_CFG)

        # Display checkpoint status
        if CHECKPOINT_FILE.exists() and ep > 1:
            print("✓ Loading learned GP from previous episode (skipping 60-step warmup!)")
        else:
            print("Training fresh GP (60-step warmup)")

        # Run episode
        out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=SIM_TIME,
                         render=RENDER, verbose=False,
                         gp_checkpoint=str(CHECKPOINT_FILE))

        m = out["metrics"]
        h = out["history"]

        # Calculate time to reach goal (8.0m)
        x_arr = np.array(h["x"])
        t_arr = np.array(h["t"])

        goal_idx = np.where(x_arr >= 8.0)[0]
        if len(goal_idx) > 0:
            time_to_goal = float(t_arr[goal_idx[0]])
            reached_goal = True
        else:
            time_to_goal = SIM_TIME
            reached_goal = False

        # Store results
        results['episode'].append(ep)
        results['time_to_goal'].append(time_to_goal)
        results['goal_reached'].append(reached_goal)
        results['max_distance'].append(m['max_x'])
        results['max_pitch'].append(m['max_abs_pitch'])

        # Save checkpoint
        gp = out["gp"]
        gp.save_checkpoint(str(CHECKPOINT_FILE))

        # Print episode summary
        goal_str = "✓ REACHED" if reached_goal else "✗ SHORT"
        flip_str = "✗ FLIPPED" if m['flipped'] else "✓ SAFE"

        print(f"\nResults:")
        print(f"  Time to reach goal: {time_to_goal:.1f}s {goal_str}")
        print(f"  Max distance: {m['max_x']:.1f}m")
        print(f"  Max pitch: {m['max_abs_pitch']:.1f}° {flip_str}")
        print(f"  Settling error: {m['settle_err']:.2f}m")

        # Update plot
        if RENDER:
            # RENDER=True: Create/update plot after episode
            if fig is None:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                fig.suptitle('MPPI Learning: Time to Reach Goal (A→B) - With Viewer',
                             fontsize=14, fontweight='bold')

                ax1 = axes[0]
                ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Time to Reach Goal (seconds)', fontsize=12, fontweight='bold')
                ax1.set_title('Learning = Going From A→B Faster!', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                line1, = ax1.plot([], [], 'o-', linewidth=3, markersize=10, color='#0072B2', label='Time to Goal')
                ax1.set_ylim([0, SIM_TIME])

                ax2 = axes[1]
                ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Distance (m)', fontsize=12, fontweight='bold')
                ax2.set_title('Traversal Distance & Safety', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                line2, = ax2.plot([], [], 's-', linewidth=2, markersize=8, color='#E69F00', label='Max Distance')
                line3, = ax2.plot([], [], '^-', linewidth=2, markersize=8, color='#CC79A7', label='Max Pitch (°)', alpha=0.7)
                ax2.axhline(8.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Goal (8.0m)')
                ax2.axhline(110.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Flip (110°)')
                ax2.set_ylim([0, max(50, SIM_TIME)])
                ax2.legend(loc='upper left', fontsize=10)

                fig.tight_layout()

            # Update plots
            episodes = np.array(results['episode'])
            line1.set_data(episodes, results['time_to_goal'])
            line2.set_data(episodes, results['max_distance'])
            line3.set_data(episodes, results['max_pitch'])

            ax1.set_xlim([episodes[0] - 0.5, episodes[-1] + 0.5])
            ax1.set_xticks(episodes)
            ax2.set_xlim([episodes[0] - 0.5, episodes[-1] + 0.5])
            ax2.set_xticks(episodes)
            ax2.set_ylim([0, max(max(results['max_distance']) * 1.1, 20)])

            # Add annotations
            for ep_num, t2g, reached in zip(results['episode'], results['time_to_goal'], results['goal_reached']):
                color = 'green' if reached else 'red'
                ax1.annotate(f'{t2g:.1f}s', (ep_num, t2g), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9, color=color, fontweight='bold')

            fig.canvas.draw()
            fig.canvas.flush_events()
            print("📊 Plot updated")

        else:
            # RENDER=False: Update plot live
            episodes = np.array(results['episode'])
            window_size = 20
            plot_episodes = episodes[-window_size:] if len(episodes) > window_size else episodes
            plot_times = results['time_to_goal'][-window_size:] if len(results['time_to_goal']) > window_size else results['time_to_goal']
            plot_dist = results['max_distance'][-window_size:] if len(results['max_distance']) > window_size else results['max_distance']
            plot_pitch = results['max_pitch'][-window_size:] if len(results['max_pitch']) > window_size else results['max_pitch']

            line1.set_data(plot_episodes, plot_times)
            line2.set_data(plot_episodes, plot_dist)
            line3.set_data(plot_episodes, plot_pitch)

            ax1.set_xlim([plot_episodes[0] - 0.5, plot_episodes[-1] + 0.5])
            ax1.set_xticks(plot_episodes)
            ax2.set_xlim([plot_episodes[0] - 0.5, plot_episodes[-1] + 0.5])
            ax2.set_xticks(plot_episodes)
            ax2.set_ylim([0, max(max(plot_dist) * 1.1, 20)])

            for ep_num, t2g, reached in zip(plot_episodes[-5:], plot_times[-5:], results['goal_reached'][-5:]):
                color = 'green' if reached else 'red'
                ax1.annotate(f'{t2g:.1f}s', (ep_num, t2g), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=9, color=color, fontweight='bold')

            fig.canvas.draw()
            fig.canvas.flush_events()

except KeyboardInterrupt:
    # Graceful shutdown on Ctrl+C
    print("\n\n" + "="*80)
    print("LEARNING SUMMARY: TIME TO REACH GOAL (A→B)")
    print("="*80)

    if len(results['episode']) > 0:
        episodes = np.array(results['episode'])
        times = np.array(results['time_to_goal'])

        print(f"\nTotal episodes run: {len(results['episode'])}")
        print(f"\nLast 5 episodes (seconds to reach goal):")
        for i in range(max(0, len(results['episode'])-5), len(results['episode'])):
            ep = results['episode'][i]
            t = results['time_to_goal'][i]
            reached = results['goal_reached'][i]
            status = "✓ GOAL" if reached else "✗ FAIL"
            print(f"  Episode {ep}: {t:5.1f}s {status}")

        if len(episodes) > 1:
            speedup = times[0] / times[-1] if times[-1] > 0 else float('inf')
            improvement = (times[0] - times[-1]) / times[0] * 100
            print(f"\nLearning improvement (Episode 1 → Last):")
            print(f"  {times[0]:.1f}s → {times[-1]:.1f}s")
            print(f"  Speedup: {speedup:.1f}x faster")
            print(f"  Time reduction: {improvement:+.1f}%")

        success_rate = sum(results['goal_reached']) / len(results['goal_reached']) * 100
        print(f"\nSuccess rate: {sum(results['goal_reached'])}/{len(results['goal_reached'])} ({success_rate:.0f}%)")

    print("\n" + "="*80)
    print("✓ Learning session ended")
    print("="*80 + "\n")

    # Save plot
    if fig is not None:
        plot_file = Path(__file__).with_name("learning_plot_final.png")
        fig.savefig(str(plot_file), dpi=150, bbox_inches='tight')
        print(f"✓ Saved final plot to: {plot_file}\n")

    plt.show()
