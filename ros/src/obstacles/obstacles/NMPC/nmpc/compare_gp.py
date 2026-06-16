#!/usr/bin/env python3
"""
compare_gp.py
-------------
VFE (SSGP.py) vs FITC (SGP.py), head-to-head, FAIR and FAST.

Instead of re-running the whole MuJoCo sim twice (slow, and the controller would take
DIFFERENT trajectories under each GP -- an unfair, confounded comparison), this REPLAYS
the SAME recorded residual stream (z = [x, theta], targets r_v_dot, r_omega_dot) from an
existing run through a FRESH VFE learner and a FRESH FITC learner. Same data, same
inducing grid, same lengthscales -> the ONLY difference is VFE's R = sn2 vs FITC's
R = sn2 + gamma. Each learner predicts BEFORE it updates (one-step generalisation), so
the score is honest. We report, per channel:

  RMSE   one-step predictive RMSE on clean steps        (LOWER  = better fit)
  corr   corr(prediction, measurement)                  (HIGHER = tracks the signal)
  NLL    mean Gaussian negative log-likelihood          (LOWER  = better-calibrated
         using each model's OWN predictive mean+var            uncertainty; this is
                                                               where VFE/FITC differ most)

The v_dot channel is the learnable obstacle "blockage"; the omega_dot channel is
contact-impulse dominated (mostly irreducible) -- so v_dot is the channel that matters.

Usage:
    python compare_gp.py [trajectory.csv]      # default: obstacle_mujoco.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from SSGP import SSGPConfig
from SGP import SGPConfig


def _replay(build_cfg, Z_feat, r_v, r_w, warmup):
    """Stream (z, r_v, r_w) through a fresh learner, predicting BEFORE each update.
    Returns per-step (pred_v, pred_sd_v, pred_w, pred_sd_w), aligned to inputs; the
    warmup steps (before the GP is ready) are returned as NaN so they are excluded."""
    gp = build_cfg.build(n_features=Z_feat.shape[1])
    n = len(r_v)
    pv = np.full(n, np.nan); sv = np.full(n, np.nan)
    pw = np.full(n, np.nan); sw = np.full(n, np.nan)
    for i in range(n):
        z = Z_feat[i]
        if gp.ready:                       # predict the point it has NOT yet seen
            mv, s_v, mw, s_w = gp.predict_channels(z)
            pv[i], sv[i], pw[i], sw[i] = mv, s_v, mw, s_w
        gp.observe(z, r_v[i], r_w[i])      # then fold it in
    return pv, sv, pw, sw, gp


def _metrics(pred, sd, meas):
    ok = np.isfinite(pred) & np.isfinite(meas) & np.isfinite(sd)
    pred, sd, meas = pred[ok], sd[ok], meas[ok]
    if pred.size < 10:
        return dict(rmse=np.nan, corr=np.nan, nll=np.nan, n=int(pred.size))
    err = pred - meas
    rmse = float(np.sqrt(np.mean(err**2)))
    corr = float(np.corrcoef(pred, meas)[0, 1]) if np.std(pred) > 1e-9 else 0.0
    var = np.maximum(sd**2, 1e-9)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * err**2 / var))
    return dict(rmse=rmse, corr=corr, nll=nll, n=int(pred.size))


def main():
    here = Path(__file__).parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "obstacle_mujoco.csv"
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}\nRun obstacle_mujoco_simulation.py first.")

    cols = ["x", "pitch_rad", "r_v_dot", "r_omega_dot", "gp_gated"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    # Drop steps taken in a flipped state (gp_gated) -- the same samples the live GP
    # refuses to learn from; replaying them would just inject garbage into both models.
    if "gp_gated" in df:
        df = df[df["gp_gated"] < 0.5]
    df = df.dropna(subset=["x", "pitch_rad", "r_v_dot", "r_omega_dot"])

    Z_feat = df[["x", "pitch_rad"]].to_numpy(float)        # GP input z = [x, theta]
    r_v = df["r_v_dot"].to_numpy(float)
    r_w = df["r_omega_dot"].to_numpy(float)
    print(f"Replaying {len(r_v)} clean steps from {path.name} through VFE and FITC "
          f"(identical inducing grid)\n")

    out = {}
    for name, cfg in (("VFE (SSGP)", SSGPConfig()), ("FITC (SGP)", SGPConfig())):
        pv, sv, pw, sw, gp = _replay(cfg, Z_feat, r_v, r_w, cfg.warmup)
        out[name] = dict(v=_metrics(pv, sv, r_v), w=_metrics(pw, sw, r_w))

    hdr = f"{'model':<12}{'channel':<10}{'RMSE':>10}{'corr':>9}{'NLL':>10}{'n':>9}"
    print(hdr); print("-" * len(hdr))
    for name in out:
        for ch, lab in (("v", "v_dot"), ("w", "omega_dot")):
            m = out[name][ch]
            print(f"{name:<12}{lab:<10}{m['rmse']:>10.4f}{m['corr']:>9.3f}"
                  f"{m['nll']:>10.3f}{m['n']:>9d}")

    # verdict on the v_dot channel (the learnable blockage)
    v, f = out["VFE (SSGP)"]["v"], out["FITC (SGP)"]["v"]
    print("\n--- v_dot (the learnable obstacle blockage) ---")
    better_rmse = "FITC" if f["rmse"] < v["rmse"] else "VFE"
    better_corr = "FITC" if f["corr"] > v["corr"] else "VFE"
    better_nll = "FITC" if f["nll"] < v["nll"] else "VFE"
    print(f"  lower RMSE : {better_rmse}   ({v['rmse']:.4f} VFE vs {f['rmse']:.4f} FITC)")
    print(f"  higher corr: {better_corr}   ({v['corr']:+.3f} VFE vs {f['corr']:+.3f} FITC)")
    print(f"  lower  NLL : {better_nll}   ({v['nll']:.3f} VFE vs {f['nll']:.3f} FITC)")
    votes = [better_rmse, better_corr, better_nll]
    winner = "FITC" if votes.count("FITC") >= 2 else "VFE"
    print(f"\n  => {winner} wins the v_dot channel ({votes.count(winner)}/3 metrics).")


if __name__ == "__main__":
    main()
