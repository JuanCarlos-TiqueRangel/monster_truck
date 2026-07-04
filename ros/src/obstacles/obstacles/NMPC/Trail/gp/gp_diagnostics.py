#!/usr/bin/env python3
"""GP health check against a recorded run.

Answers, in order:
  (1) is the GP learning anything from THIS data?   -> skill vs zero predictor
  (2) is the frozen standardization still valid?    -> regime / span check
  (3) is the GP actually reaching the rollout?      -> contribution magnitude

Run from the Trail/ root after an episode set:
    python3 gp_diagnostics.py [results/obstacle_mujoco.csv] [results/obstacle_model.npz]

For the implementation-correctness question (export math, kernel sync,
alpha recovery) run the built-in self-test instead:  python3 GP.py
It must print export-vs-torch max diff ~1e-6 and small held-out RMSE.
"""

import csv
import sys
from pathlib import Path

import numpy as np

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/obstacle_mujoco.csv")
NPZ_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/obstacle_model.npz")

rows = list(csv.DictReader(CSV_PATH.open()))
def col(name):
    return np.array([float(r[name]) for r in rows])

ready = col("gp_ready") > 0.5
episode = col("episode").astype(int)

# ---------------------------------------------------------------------------
# (1) learning: one-step skill vs the zero predictor, per channel, per episode
#     skill = 1 - rmse(r - pred_pre) / std(r)
#     > 0.3 learning usefully | ~0 predicting the mean | < 0 worse than nothing
# ---------------------------------------------------------------------------
print("=" * 78)
print("(1) LEARNING: one-step skill vs zero predictor (needs gp_ready steps)")
print("=" * 78)
channels = {"v_dot": ("r_v_dot", "gp_v_dot_pred_pre"),
            "omega_dot": ("r_omega_dot", "gp_omega_dot_pred_pre")}
for ch, (r_name, p_name) in channels.items():
    r_all, p_all = col(r_name), col(p_name)
    for e in sorted(set(episode)):
        m = ready & (episode == e)
        if m.sum() < 50:
            continue
        r, p = r_all[m], p_all[m]
        base = r.std()
        err = float(np.sqrt(np.mean((r - p) ** 2)))
        skill = 1.0 - err / max(base, 1e-9)
        spikes = float(np.mean(np.abs(r - r.mean()) > 3.0 * base))
        corr = float(np.corrcoef(r, p)[0, 1]) if p.std() > 1e-12 else 0.0
        print(f"  [{ch:9s}] ep{e}: std(r)={base:8.3f}  rmse(r-pred)={err:8.3f}  "
              f"skill={skill:+.2f}  corr={corr:+.2f}  spikes>3sig={100*spikes:4.1f}%")
print("  NOTE: heavy spike fraction on omega_dot = contact impulses in the target;")
print("  a stationary GP averages them into mush. Consider clipping the residual")
print("  targets before observe(), e.g. np.clip(r_w, -40, 40).")

# ---------------------------------------------------------------------------
# (2) regime: the standardization and inducing points freeze at the FIRST fit.
#     If later episodes visit states that standardize far outside the inducing
#     span, every kernel column is ~0 there and the GP returns y_mean.
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(2) REGIME: standardized data span vs inducing span (|z| >> Z span = off-map)")
print("=" * 78)
if NPZ_PATH.exists():
    d = np.load(NPZ_PATH)
    xm, xs = d["gp_x_mean"], d["gp_x_std"]
    Z = d["gp_Z"].reshape(-1, xm.size)
    feats = np.stack([col("x"), col("x_dot"), col("pitch_rad"),
                      col("pitch_dot"), col("tau_cmd")], axis=1)
    zs = (feats - xm) / xs
    names = ["x", "v", "theta", "omega", "tau"]
    off_map = 0
    for i, n in enumerate(names):
        lo, hi = zs[:, i].min(), zs[:, i].max()
        zlo, zhi = Z[:, i].min(), Z[:, i].max()
        flag = ""
        if hi > zhi + 3.0 or lo < zlo - 3.0:
            flag = "  <-- OFF-MAP (GP ~ y_mean here)"
            off_map += 1
        print(f"  {n:6s} data z-span [{lo:+7.1f}, {hi:+7.1f}]   "
              f"inducing span [{zlo:+6.1f}, {zhi:+6.1f}]   x_std={xs[i]:.4f}{flag}")
    if off_map:
        print(f"  {off_map} feature(s) leave the trained region. Fix: physical")
        print("  standardization instead of data-driven (set x_std from known ranges,")
        print("  e.g. theta 0.6, omega 4, tau 5, v 4, x 3) or refit structure when")
        print("  the operating regime changes.")
else:
    print(f"  {NPZ_PATH} not found - run at least one episode set with saving on.")

# ---------------------------------------------------------------------------
# (3) plumbing: is the GP contribution nonzero where it should act?
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(3) PLUMBING: is the prediction actually nonzero in the rollout regime?")
print("=" * 78)
for ch, p_name in (("v_dot", "gp_v_dot_pred"), ("omega_dot", "gp_omega_dot_pred")):
    p = col(p_name)
    for e in sorted(set(episode)):
        m = ready & (episode == e)
        if m.sum() < 50:
            continue
        frac = float(np.mean(np.abs(p[m]) > 0.05))
        print(f"  [{ch:9s}] ep{e}: mean|pred|={np.mean(np.abs(p[m])):7.3f}   "
              f"frac |pred|>0.05 = {100*frac:5.1f}%")
print("  If skill in (1) is good but mean|pred| here is ~0, the controller is not")
print("  receiving the model. Check, in this order:")
print("    - CONTROLLER imports mppi_gp (mppi_dynamics never calls the GP)")
print("    - GP_ENABLED = True in the node")
print("    - after LOAD_MODEL=True, predict_torch() returns ZEROS until the next")
print("      end_episode(); a torch rollout must call predict_torch_fast() instead")
print("    - episode 1 is always GP-free by design; judge from episode 2 onward")
