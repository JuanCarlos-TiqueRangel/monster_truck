import sys
sys.path.insert(0, '/home/juan/lehigh_PhD/monster_truck/ros/src/obstacles/obstacles/NMPC/mppi_beta')

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

print("\n" + "="*80)
print("MULTI-SEED ROBUSTNESS TEST (Adaptive Temperature Enabled)")
print("="*80)
print(f"{'Seed':<6} {'Goal':<8} {'Max X':<8} {'Pitch':<8} {'Flipped':<10} {'Solve':<8}")
print("-"*80)

seeds_to_test = [0, 1, 2, 3, 4]
results = []

for seed in seeds_to_test:
    cfg_test = MPPIConfig(
        dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
        q_x=15.0, q_v=8.0, q_theta=6.0, q_omega=60.0,
        r_tau=0.05, r_dtau=1.0, q_terminal_theta=6.0, q_terminal_omega=60.0,
        flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
        adaptive_temperature=True, v_ref_gain=0.6, v_cruise=1.2, seed=seed
    )
    ctrl = WheelieMPPI(PARAMS, cfg_test, GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=40.0, render=False, verbose=False)
    m = out['metrics']
    
    goal_str = "✓" if m['reached_goal'] else "✗"
    flip_str = "✗ FLIP" if m['flipped'] else "✓ OK"
    print(f"{seed:<6} {goal_str:<8} {m['max_x']:<8.2f} {m['max_abs_pitch']:<8.1f}° {flip_str:<10} {m['solve_ms_mean']:<8.1f}ms")
    results.append(m)

print("="*80)
goals = sum(1 for m in results if m['reached_goal'])
flips = sum(1 for m in results if m['flipped'])
print(f"✓ Success Rate: {goals}/{len(seeds_to_test)} seeds reached goal")
print(f"✓ Safety: {len(seeds_to_test)-flips}/{len(seeds_to_test)} seeds did NOT flip")
print("="*80 + "\n")
