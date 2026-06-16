#!/usr/bin/env python3
"""
learn_cost.py
-------------
Episode-level COST LEARNING to go from A to B (the goal) FASTER.

Outer loop = policy search over the MPPI cost/speed weights; inner loop = the
MPPI+SSGP closed-loop lap. After each lap we score it by TIME-TO-GOAL and adapt
the weights, so the controller DISCOVERS over episodes how to reach the goal
sooner. Safety is a constraint, not the objective: a lap that doesn't reach or
that flips is penalised, everything else is ranked purely by lap time.

This is the piece SSGP/ILC could not provide: SSGP learns the dynamics model and
ILC a feedforward, but neither changes the controller's STRATEGY. Learning the
cost weights changes the strategy itself, so the lap time can actually drop.

Learned weights (the speed-relevant ones):
    v_cruise    -- the cruise-speed cap in the velocity reference   (higher = faster)
    v_ref_gain  -- how hard it chases the goal-distance speed ref
    q_theta     -- pitch penalty (buys safety margin so it can push harder)

Learner: Cross-Entropy Method (CEM), robust to the noisy/bimodal plant.

    python3 learn_cost.py [generations]
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from params import WheelieParams
from rls import RLSConfig
from mppi import WheelieMPPI, MPPIConfig
from sim_harness import run_episode, SSGPConfig

GENERATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
POP = 6            # candidates per generation
ELITE = 2         # elites kept to refit the sampling Gaussian
SEEDS = (0, 1, 2)  # each candidate is scored over several seeds -> ROBUST speed,
                   # not a lucky single rollout on a knife-edge (flip-prone) plant
SIM_TIME = 40.0
GOAL_X = 8.0

HERE = Path(__file__).resolve().parent
PLOT = HERE / "cost_learning.png"
BEST_FILE = HERE / "learned_cost.json"

# v_max lifted so the controller's speed barrier does not cap us below the target
# we are trying to learn; the flip penalty + q_theta keep it safe instead.
PARAMS = WheelieParams(v_max=4.0, v_min=-4.0)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

# search vector theta = [v_cruise, v_ref_gain, q_theta]
BASELINE = np.array([1.2, 0.6, 6.0])      # the original settings (the "before")
MU0 = np.array([1.5, 0.8, 8.0])
SIG0 = np.array([1.0, 0.6, 15.0])
LO = np.array([1.2, 0.4, 3.0])
HI = np.array([4.0, 2.5, 60.0])


def make_cfg(theta, seed):
    v_cruise, v_ref_gain, q_theta = [float(t) for t in theta]
    return MPPIConfig(
        dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
        q_x=15.0, q_v=8.0, q_theta=q_theta, q_omega=60.0,
        r_tau=0.05, r_dtau=1.0, q_terminal_theta=q_theta, q_terminal_omega=60.0,
        flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
        v_ref_gain=v_ref_gain, v_cruise=v_cruise, seed=int(seed))


def run_one(theta, seed):
    ctrl = WheelieMPPI(PARAMS, make_cfg(theta, seed), GP_CFG)
    out = run_episode(ctrl, PARAMS, GP_CFG, RLS_CFG, sim_time=SIM_TIME,
                      goal_x=GOAL_X, render=False, verbose=False,
                      gp_checkpoint=None)         # fresh GP -> fair across candidates
    h, m = out["history"], out["metrics"]
    x = np.asarray(h["x"], float); t = np.asarray(h["t"], float)
    gi = np.where(x >= GOAL_X)[0]
    t2g = float(t[gi[0]]) if len(gi) else SIM_TIME
    return dict(t2g=t2g, reached=len(gi) > 0, flipped=bool(m["flipped"]),
                max_pitch=m["max_abs_pitch"], hist=h)


def evaluate(theta, seeds=SEEDS):
    """Score a cost vector over several seeds -> ROBUST expected time.
    J = mean time-to-goal + penalties for any non-reach / flip across seeds."""
    rs = [run_one(theta, s) for s in seeds]
    t2gs = np.array([r["t2g"] for r in rs])
    frac_fail = np.mean([not r["reached"] for r in rs])
    frac_flip = np.mean([r["flipped"] for r in rs])
    J = float(t2gs.mean() + 100.0 * frac_fail + 150.0 * frac_flip + 5.0 * t2gs.std())
    # representative lap = the seed closest to the mean time (for the velocity plot)
    repr_r = rs[int(np.argmin(np.abs(t2gs - t2gs.mean())))]
    return dict(J=J, mean_t2g=float(t2gs.mean()), std_t2g=float(t2gs.std()),
                worst=float(t2gs.max()), frac_fail=float(frac_fail),
                frac_flip=float(frac_flip),
                max_pitch=max(r["max_pitch"] for r in rs), hist=repr_r["hist"])


# ── CEM over the speed weights ───────────────────────────────────────────────
mu, sigma = MU0.copy(), SIG0.copy()

print("\n" + "=" * 96)
print("COST LEARNING (CEM, robust over seeds)  --  objective: MINIMUM TIME A->B")
print("=" * 96)
print(f"{'gen':>3} | {'v_cruise':>8} {'v_ref_g':>8} {'q_theta':>8} | "
      f"{'mean t2g':>9} {'std':>6} {'worst':>6} {'flip%':>6}")
print("-" * 96)

gen_best_t2g = []
best_theta, best_eval = None, dict(J=float("inf"))
for g in range(1, GENERATIONS + 1):
    samples = np.clip(mu + sigma * np.random.randn(POP, 3), LO, HI)
    results = [evaluate(s) for s in samples]
    Js = np.array([r["J"] for r in results])
    order = np.argsort(Js)
    elites = samples[order[:ELITE]]
    mu = elites.mean(0)
    sigma = np.maximum(elites.std(0), [0.15, 0.1, 2.0])

    b = results[order[0]]; bt = samples[order[0]]
    if b["J"] < best_eval["J"]:                # keep the best config ever seen
        best_theta, best_eval = bt.copy(), b
    gen_best_t2g.append(b["mean_t2g"])
    print(f"{g:>3} | {bt[0]:>8.2f} {bt[1]:>8.2f} {bt[2]:>8.1f} | "
          f"{b['mean_t2g']:>9.2f} {b['std_t2g']:>6.2f} {b['worst']:>6.2f} "
          f"{b['frac_flip']*100:>5.0f}%")

# ── before (baseline) vs after (BEST config found), both scored over seeds ────
print("-" * 96)
print("comparing baseline vs best-found ...")
before = evaluate(BASELINE)
after = best_eval                              # report the BEST config, not the mean
mu = best_theta

print("-" * 96)
print(f"learned weights:  v_cruise={mu[0]:.2f}  v_ref_gain={mu[1]:.2f}  q_theta={mu[2]:.1f}")
print(f"TIME A->B (mean±std over {len(SEEDS)} seeds):")
print(f"   baseline {before['mean_t2g']:.2f}±{before['std_t2g']:.2f}s  (worst {before['worst']:.2f}, flips {before['frac_flip']*100:.0f}%)")
print(f"   learned  {after['mean_t2g']:.2f}±{after['std_t2g']:.2f}s  (worst {after['worst']:.2f}, flips {after['frac_flip']*100:.0f}%)")
gain = (before['mean_t2g'] - after['mean_t2g']) / before['mean_t2g'] * 100
print(f"   -> {gain:+.0f}% faster (robust)")
print("=" * 96 + "\n")

with open(BEST_FILE, "w") as f:
    json.dump({"v_cruise": float(mu[0]), "v_ref_gain": float(mu[1]),
               "q_theta": float(mu[2]),
               "t2g_before_mean": before["mean_t2g"], "t2g_before_std": before["std_t2g"],
               "t2g_after_mean": after["mean_t2g"], "t2g_after_std": after["std_t2g"]},
              f, indent=2)

# ── plots ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Cost learning: reaching the goal (A→B) faster over generations",
             fontweight="bold")

gens = list(range(1, GENERATIONS + 1))
ax[0].plot(gens, gen_best_t2g, "o-", lw=2.5, ms=9, color="#0072B2")
ax[0].set_xlabel("generation", fontweight="bold"); ax[0].set_ylabel("best mean time to goal (s)", fontweight="bold")
ax[0].set_title("Lap time ↓ = learning to go faster"); ax[0].grid(alpha=0.3)

# velocity vs position: representative baseline vs learned lap
for h, lab, c in ((before["hist"], f"baseline ({before['mean_t2g']:.1f}s)", "#999999"),
                  (after["hist"], f"learned ({after['mean_t2g']:.1f}s)", "#009E73")):
    x = np.asarray(h["x"], float); v = np.asarray(h["v"], float)
    k = np.concatenate(([True], np.diff(x) > 1e-9))
    ax[1].plot(x[k], v[k], lw=2, color=c, label=lab)
ax[1].axvline(5.0, color="red", ls="--", alpha=0.6, label="obstacle ~5.0 m")
ax[1].set_xlabel("position x (m)", fontweight="bold"); ax[1].set_ylabel("speed (m/s)", fontweight="bold")
ax[1].set_title("Speed along the track: before vs learned"); ax[1].grid(alpha=0.3); ax[1].legend()

ax[2].bar(["baseline", "learned"], [before["mean_t2g"], after["mean_t2g"]],
          yerr=[before["std_t2g"], after["std_t2g"]], capsize=8,
          color=["#999999", "#009E73"])
ax[2].set_ylabel("time to goal (s, mean±std)", fontweight="bold")
ax[2].set_title("A→B time: before vs learned"); ax[2].grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(PLOT, dpi=150, bbox_inches="tight")
print("=" * 92)
print(f"✓ PLOT SAVED:  {PLOT}")
print(f"✓ learned cost: {BEST_FILE}")
print("=" * 92)

import subprocess, shutil
for opener in ("xdg-open", "eog", "feh", "display"):
    if shutil.which(opener):
        try:
            subprocess.Popen([opener, str(PLOT)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  (opening with {opener}...)"); break
        except Exception:
            pass
else:
    print(f"  open it yourself:  xdg-open {PLOT}")
print()
