"""CEM cost-learner for the obstacle-crossing controller (denoised MBRL).

The per-episode signal is chaos-dominated (see nmpc-ssgp-no-episodic-learning):
a single rollout's crossing time varies +/-4 s, so single-sample learning just
mines noise. The cure is to DENOISE the reward BEFORE each update: evaluate every
candidate over a FIXED set of K seeds (Common Random Numbers) and rank on the
seed-MEAN. Same recipe that gave 20% faster / 10x more consistent in the MPPI
cost-learner.

We optimise 4 interpretable MPCConfig knobs:
    theta_obs_deg   : GP-gated pre-rear climb posture
    w_progress      : forward-progress reward (speed vs flip-risk trade-off)
    q_flip          : flip-barrier strength
    theta_soft_deg  : pitch at which the barrier engages

Objective (minimised), per rollout:
    J = t_goal + W_FLIP * relu(peak_pitch - 90) + W_FAIL * (not reached)
t_goal = time to reach GOAL_X (SIM_TIME if never). Candidate score = seed-mean of
the LAST (warmed) episode's J.

Two pieces of honest evidence:
  (1) the TRAIN learning curve (CEM mean's J on the fixed train seeds) falls, and
  (2) a HELD-OUT validation: the learned params vs the untuned midpoint baseline on
      FRESH seeds, with a PAIRED bootstrap CI on J_baseline - J_learned. CI > 0 means
      the improvement is statistically real and generalises (not overfit to the
      training seeds). Saves cem_learning.png + cem_best.json.
"""

import io
import os
import json
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import obstacle_mujoco_simulation as O

# ---- CEM configuration (env-overridable) ----
POP = int(os.environ.get("CEM_POP", "8"))
ELITE = int(os.environ.get("CEM_ELITE", "3"))
ITERS = int(os.environ.get("CEM_ITERS", "5"))
K_SEEDS = int(os.environ.get("CEM_SEEDS", "6"))     # fixed train seeds (CRN)
V_SEEDS = int(os.environ.get("CEM_VAL", "16"))      # held-out validation seeds
EP_PER = int(os.environ.get("CEM_EP", "2"))         # episodes/rollout; score the last (warm) one
SIM_TIME = float(os.environ.get("CEM_SIMTIME", "14.0"))  # cap eval rollout length (bound wall time)
W_FLIP = float(os.environ.get("CEM_WFLIP", "0.1"))  # s per deg of pitch above 90
W_FAIL = float(os.environ.get("CEM_WFAIL", "30.0")) # s penalty for not reaching the goal
SEED0 = int(os.environ.get("CEM_SEED0", "100"))

TRAIN_SEEDS = [SEED0 + i for i in range(K_SEEDS)]
VAL_SEEDS = [SEED0 + 500 + i for i in range(V_SEEDS)]   # disjoint from train

SPEC = [
    ("theta_obs_deg", 30.0, 70.0),
    ("w_progress",     5.0, 30.0),
    ("q_flip",       500.0, 4000.0),
    ("theta_soft_deg", 55.0, 88.0),
]
NAMES = [s[0] for s in SPEC]
LO = np.array([s[1] for s in SPEC])
HI = np.array([s[2] for s in SPEC])


def _rollout_J(seed, overrides):
    """One seeded rollout (EP_PER episodes); returns J, t_goal, flip_deg, reached."""
    O.CSV_PATH = Path(f"/tmp/cem_{seed}.csv")
    O.MODEL_PATH = Path(f"/tmp/cem_{seed}_m.npz")
    with contextlib.redirect_stdout(io.StringIO()):
        hist = O.main(seed=seed, overrides=overrides)
    df = pd.DataFrame(hist)
    g = df[df.episode == df.episode.max()]               # the warmed episode
    at_goal = np.abs(g.x - O.GOAL_X) < O.GOAL_TOL
    reached = bool(at_goal.any())
    t_goal = float(g.time[at_goal].iloc[0]) if reached else O.SIM_TIME
    peak = float(g.pitch_deg.abs().max())
    flip = max(0.0, peak - 90.0)
    J = t_goal + W_FLIP * flip + W_FAIL * (0.0 if reached else 1.0)
    return J, t_goal, flip, float(reached)


def eval_candidate(theta, seeds, per_seed=False):
    """Denoised score over a FIXED seed list (CRN). Returns means, or per-seed J."""
    overrides = {n: float(v) for n, v in zip(NAMES, theta)}
    O.RENDER = False; O.LOAD_MODEL = False; O.PREWHEELIE_LEARN = False
    O.RLS_FREEZE = True; O.GP_ENABLED = True; O.N_EPISODES = EP_PER; O.SIM_TIME = SIM_TIME
    rows = [_rollout_J(s, overrides) for s in seeds]
    A = np.array(rows)                                    # (seeds, 4)
    if per_seed:
        return A[:, 0]                                    # per-seed J (for paired test)
    return tuple(A.mean(axis=0))                          # J,t_goal,flip,reach means


def slope_ci(y):
    y = np.asarray(y, float); n = len(y); x = np.arange(n)
    rng = np.random.default_rng(0); sl = []
    for _ in range(4000):
        idx = np.sort(rng.integers(0, n, n))
        if len(np.unique(idx)) >= 2:
            sl.append(np.polyfit(x[idx], y[idx], 1)[0])
    sl = np.array(sl)
    return float(np.median(sl)), float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))


def paired_ci(diff):
    """Bootstrap 95% CI on the mean of paired differences (baseline - learned)."""
    diff = np.asarray(diff, float); n = len(diff)
    rng = np.random.default_rng(1)
    m = [diff[rng.integers(0, n, n)].mean() for _ in range(5000)]
    return float(np.mean(diff)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    mu = 0.5 * (LO + HI)
    mu0 = mu.copy()                                      # untuned midpoint = baseline
    sigma = 0.5 * (HI - LO)
    rng = np.random.default_rng(0)

    print(f"CEM pop={POP} elite={ELITE} iters={ITERS} train-seeds={K_SEEDS} "
          f"val-seeds={V_SEEDS} ep/rollout={EP_PER}")
    print(f"params={NAMES}\ntrain CRN seeds={TRAIN_SEEDS}")
    hist_mu_J, hist_best_J, hist_mu_T, hist_mu_F, hist_mu_R = [], [], [], [], []

    for it in range(ITERS):
        cands = np.clip(rng.normal(mu, sigma, size=(POP, len(mu))), LO, HI)
        cands[0] = mu                                    # eval the current mean as cand 0
        scores = np.array([eval_candidate(c, TRAIN_SEEDS) for c in cands])
        order = np.argsort(scores[:, 0])
        elite = cands[order[:ELITE]]
        # record the CURRENT mean's quality (cand 0) BEFORE updating it -> clean curve
        muJ, muT, muF, muR = scores[0]
        hist_mu_J.append(muJ); hist_mu_T.append(muT)
        hist_mu_F.append(muF); hist_mu_R.append(muR)
        hist_best_J.append(float(scores[order[0], 0]))
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.04 * (HI - LO))

        ps = "  ".join(f"{n}={v:7.1f}" for n, v in zip(NAMES, mu))
        print(f"it{it}: muJ={muJ:6.2f}(t={muT:5.2f}s flip={muF:5.1f} reach={muR:.2f}) "
              f"bestJ={hist_best_J[-1]:6.2f} | mu-> {ps}", flush=True)

    # final tuned mean's quality on TRAIN seeds (after the last update)
    fJ, fT, fF, fR = eval_candidate(mu, TRAIN_SEEDS)
    hist_mu_J.append(fJ); hist_mu_T.append(fT); hist_mu_F.append(fF); hist_mu_R.append(fR)
    hist_best_J.append(fJ)

    mJ, loJ, hiJ = slope_ci(hist_mu_J)
    verdict = "REAL DOWN (learning)" if hiJ < 0 else (
        "REAL UP" if loJ > 0 else "NOISE (CI spans 0)")
    print(f"\nTRAIN mean-J slope/iter {mJ:+.3f} [{loJ:+.3f},{hiJ:+.3f}]  {verdict}")

    # ---- held-out validation: learned vs baseline, PAIRED on fresh seeds ----
    print(f"\nHELD-OUT validation on {V_SEEDS} fresh seeds (paired):")
    Jb = eval_candidate(mu0, VAL_SEEDS, per_seed=True)
    Jl = eval_candidate(mu, VAL_SEEDS, per_seed=True)
    bb = eval_candidate(mu0, VAL_SEEDS)
    bl = eval_candidate(mu, VAL_SEEDS)
    md, lod, hid = paired_ci(Jb - Jl)
    sig = "SIGNIFICANT" if lod > 0 else ("WORSE" if hid < 0 else "not significant")
    print(f"  baseline(mid): J={bb[0]:.2f} t_goal={bb[1]:.2f}s flip={bb[2]:.1f}deg reach={bb[3]:.2f}")
    print(f"  learned      : J={bl[0]:.2f} t_goal={bl[1]:.2f}s flip={bl[2]:.1f}deg reach={bl[3]:.2f}")
    print(f"  mean J improvement {md:+.2f} [{lod:+.2f},{hid:+.2f}]  -> {sig}")

    best = {"params": dict(zip(NAMES, mu.tolist())),
            "val_J": bl[0], "val_t_goal": bl[1], "val_flip_deg": bl[2], "val_reach": bl[3],
            "baseline_val_J": bb[0], "improvement_J": md, "improvement_ci": [lod, hid]}
    Path(__file__).with_name("cem_best.json").write_text(json.dumps(best, indent=2))

    its = np.arange(len(hist_mu_J))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(its, hist_mu_J, "o-", label="CEM mean (train)")
    ax[0].plot(np.arange(len(hist_best_J)), hist_best_J, "s--", color="0.6", label="iter best")
    ax[0].axhline(bb[0], color="C3", ls=":", label=f"baseline (held-out {bb[0]:.1f})")
    ax[0].axhline(bl[0], color="C2", ls=":", label=f"learned (held-out {bl[0]:.1f})")
    ax[0].set_title(f"(A) objective J\ntrain slope {mJ:+.2f}/it -> {verdict}")
    ax[0].set_xlabel("CEM iteration"); ax[0].set_ylabel("J (lower=better)")
    ax[1].plot(its, hist_mu_T, "o-", color="C1")
    ax[1].set_title("(B) time to goal (train mean)"); ax[1].set_xlabel("CEM iteration")
    ax[1].set_ylabel("t_goal [s]")
    ax[2].plot(its, hist_mu_F, "o-", color="C3")
    ax[2].set_title("(C) flip overshoot (train mean)"); ax[2].set_xlabel("CEM iteration")
    ax[2].set_ylabel("peak pitch above 90 [deg]")
    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"CEM cost-learning ({K_SEEDS}-seed CRN, RLS_FREEZE=True) | "
                 f"held-out J {bb[0]:.1f}->{bl[0]:.1f} ({sig})")
    fig.tight_layout()
    out = Path(__file__).with_name("cem_learning.png")
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}\nbest params: {json.dumps(best['params'], indent=2)}")


if __name__ == "__main__":
    main()
