"""Multi-seed learning curve for the obstacle-crossing MBRL system.

Runs N_SEEDS independent runs (different chaotic contact realisations) of
N_EPISODES each, with the GP residual + RLS persisting across episodes inside a
run. It then SEED-AVERAGES the per-episode metrics so the genuine learning trend
is visible above the per-contact deterministic chaos (which on a single run swamps
the ~1 s learning effect with a +/-2-3 s spread).

Three signals, all vs episode index:
  (A) GP one-step omega error over the obstacle region  -> the MODEL learning
  (B) obstacle-crossing time (time to reach x=3)         -> PERFORMANCE
  (C) peak |pitch|                                        -> SAFETY (flip margin)

Reports EARLY (ep 0-2) vs LATE (last 3) for each, and saves a figure.
"""

import io
import os
import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import obstacle_mujoco_simulation as O

# ---- experiment configuration (env-overridable so several configs run isolated) ----
N_SEEDS = int(os.environ.get("LC_SEEDS", "6"))
N_EPISODES = int(os.environ.get("LC_EPISODES", "15"))
THETA_OBS_DEG = float(os.environ.get("LC_THETA_OBS", "0.0"))  # GP-gated pre-rear posture
RLS_FREEZE = os.environ.get("LC_RLS_FREEZE", "1") == "1"
TAG = os.environ.get("LC_TAG", "lc")
OBST_X_LO, OBST_X_HI = 1.0, 3.5   # region where the contact residual lives
CROSS_X = 3.0                     # obstacle (x=2) considered cleared past here


def run_one_seed(seed: int):
    """Run a single seeded experiment; return a per-episode metrics DataFrame."""
    O.RENDER = False
    O.LOAD_MODEL = False
    O.PREWHEELIE_LEARN = False
    O.THETA_OBS_DEG = THETA_OBS_DEG
    O.RLS_FREEZE = RLS_FREEZE
    O.N_EPISODES = N_EPISODES
    O.CSV_PATH = Path(f"/tmp/{TAG}_{seed}.csv")
    O.MODEL_PATH = Path(f"/tmp/{TAG}_{seed}_m.npz")

    with contextlib.redirect_stdout(io.StringIO()):
        hist = O.main(seed=seed)
    df = pd.DataFrame(hist)

    rows = []
    for ep, g in df.groupby("episode"):
        cleared = g.x >= CROSS_X
        t_cross = float(g.time[cleared].iloc[0]) if cleared.any() else O.SIM_TIME
        reached = bool((np.abs(g.x - O.GOAL_X) < O.GOAL_TOL).any())
        max_pitch = float(g.pitch_deg.abs().max())

        # GP one-step omega error over the obstacle region, where the GP is ready.
        m = (g.gp_ready == 1) & (g.x >= OBST_X_LO) & (g.x <= OBST_X_HI)
        if m.any():
            err = (g.r_omega_dot[m] - g.gp_omega_dot_pred_pre[m]).abs()
            gp_err = float(err.median())
        else:
            gp_err = np.nan
        rows.append(dict(episode=int(ep), t_cross=t_cross, reached=reached,
                         max_pitch=max_pitch, gp_err=gp_err))
    return pd.DataFrame(rows)


def main():
    print(f"Running {N_SEEDS} seeds x {N_EPISODES} episodes "
          f"(RLS_FREEZE={O.RLS_FREEZE}, w_progress in cfg) ...")
    per = [run_one_seed(s) for s in range(N_SEEDS)]

    eps = np.arange(N_EPISODES)
    T = np.array([p.t_cross.values for p in per])          # (seeds, eps)
    P = np.array([p.max_pitch.values for p in per])
    G = np.array([p.gp_err.values for p in per])
    R = np.array([p.reached.values for p in per]).astype(float)

    def band(ax, A, color, label):
        mu = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0)
        ax.plot(eps, mu, "o-", color=color, ms=4, label=label)
        ax.fill_between(eps, mu - sd, mu + sd, color=color, alpha=0.15)
        return mu

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    g_mu = band(ax[0], G, "C0", "median |omega err|")
    ax[0].set_title("(A) GP model error\n(obstacle region)")
    ax[0].set_xlabel("episode"); ax[0].set_ylabel(r"$|\dot\omega - \hat{\dot\omega}|$ [rad/s$^2$]")

    t_mu = band(ax[1], T, "C1", "crossing time")
    ax[1].set_title("(B) obstacle-crossing time")
    ax[1].set_xlabel("episode"); ax[1].set_ylabel("time to clear x=3 [s]")

    p_mu = band(ax[2], P, "C3", "peak |pitch|")
    ax[2].axhline(90, color="0.6", ls="--", lw=1, label="flip (90$\\degree$)")
    ax[2].set_title("(C) peak pitch (flip margin)")
    ax[2].set_xlabel("episode"); ax[2].set_ylabel("max |pitch| [deg]")

    for a in ax:
        a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle(f"Episodic learning, seed-averaged over {N_SEEDS} runs "
                 f"(RLS_FREEZE={O.RLS_FREEZE})")
    fig.tight_layout()
    out = Path(__file__).with_name(f"learning_curve_{TAG}.png")
    fig.savefig(out, dpi=130)
    print(f"Saved: {out}")

    def slope_ci(A, name):
        """Bootstrap CI on the per-episode slope (averaging seeds each resample).

        A is (seeds, eps). Resample seeds with replacement, take the seed-mean
        curve, OLS-fit vs episode, collect the slope. A slope whose 95% CI is
        entirely < 0 is a statistically real downward trend (genuine learning);
        a CI straddling 0 means the early->late change is just chaos.
        """
        ns, ne = A.shape
        x = np.arange(ne)
        rng = np.random.default_rng(0)
        slopes = []
        for _ in range(2000):
            idx = rng.integers(0, ns, ns)
            mu = np.nanmean(A[idx], axis=0)
            ok = np.isfinite(mu)
            slopes.append(np.polyfit(x[ok], mu[ok], 1)[0])
        slopes = np.array(slopes)
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        med = np.median(slopes)
        verdict = "REAL DOWN" if hi < 0 else ("REAL UP" if lo > 0 else "NOISE (CI spans 0)")
        e, l = float(np.nanmean(np.nanmean(A, 0)[:3])), float(np.nanmean(np.nanmean(A, 0)[-3:]))
        print(f"  {name:14s} early {e:7.3f} -> late {l:7.3f} | "
              f"slope/ep {med:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")

    print(f"\nReach rate: early {R[:, :3].mean():.2f}  late {R[:, -3:].mean():.2f}")
    slope_ci(G, "GP model err")
    slope_ci(T, "crossing time")
    slope_ci(P, "peak pitch")


if __name__ == "__main__":
    main()
