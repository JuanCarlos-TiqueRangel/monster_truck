"""Measure the cost-parameter landscape on a FIXED held-out seed set, with paired
bootstrap CIs vs the naive controller. Tells us whether there is a large, robust
(chaos-surviving) improvement for CEM to find -- before paying for a full run.
"""
import os
import numpy as np
import cem_learn as C

SEEDS = [100 + 600 + i for i in range(int(os.environ.get("PB_SEEDS", "16")))]  # held-out

POINTS = {
    # name:          theta_obs, w_progress, q_flip, theta_soft
    "naive(ram)":    [0.0,  15.0, 2000.0, 80.0],
    "midpoint":      [50.0, 17.5, 2250.0, 71.5],
    "angle55":       [55.0, 15.0, 2000.0, 80.0],
    "highbar50":     [50.0, 15.0, 3800.0, 80.0],
    "combo55":       [55.0, 15.0, 3800.0, 80.0],
}


def main():
    print(f"probe on {len(SEEDS)} held-out seeds: {SEEDS}")
    Jper = {}
    for name, theta in POINTS.items():
        Jp = C.eval_candidate(np.array(theta), SEEDS, per_seed=True)
        J, t, f, r = C.eval_candidate(np.array(theta), SEEDS)
        Jper[name] = Jp
        print(f"  {name:12s} J={J:6.2f}  t_goal={t:5.2f}s  flip={f:5.1f}deg  reach={r:.2f}")
    print("\nPaired improvement vs naive(ram)  [mean (95% CI)]  (positive = better than naive):")
    base = Jper["naive(ram)"]
    for name, Jp in Jper.items():
        if name == "naive(ram)":
            continue
        m, lo, hi = C.paired_ci(base - Jp)
        sig = "SIGNIFICANT" if lo > 0 else ("WORSE" if hi < 0 else "ns")
        print(f"  {name:12s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {sig}")


if __name__ == "__main__":
    main()
