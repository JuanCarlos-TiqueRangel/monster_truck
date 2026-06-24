#!/usr/bin/env python3
"""
rls_batch_id.py
---------------
OFFLINE batch identification of the 10 full-dynamics RLS weights -- an alternative to the
streaming rls_training.py (which it leaves untouched). The excitation log is available all
at once offline, so the optimal estimate is a single batch least-squares, NOT a sequential
filter. This file does the two things the streaming method can't:

  #1  ROBUST batch least-squares (Huber IRLS) per channel -> the optimum in one pass, with
      no forgetting-factor windup or sample-ordering effects. Contact/flip outliers are
      SOFT down-weighted (not hard-gated), so no information is thrown away abruptly.

  #2  HONEST uncertainty + generalization:
      * parameter +/- 1 sigma from the LS covariance (and a z = a/sigma "significance"),
        so you see which weights are pinned vs loose -- it lines up with the PE-weak
        directions, and would have settled the sin(theta) question objectively.
      * K-fold CROSS-VALIDATED (out-of-sample) RMSE, so you judge generalization rather
        than the optimistic in-sample error the streaming run reports.

It REUSES rls_training.py's PRBS excitation (PRBSSignal) and MuJoCo wrapper (TruckModel),
and reads MuJoCo's exact acceleration (qacc) as the target. Output: a weights+/-sigma table,
train-vs-CV RMSE, rls_batch.npz, and a parity/residual figure.

Run:  python rls_batch_id.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Import rls_training FIRST: it puts the sibling controller packages (mppi/ gp/ rls/ nmpc/)
# on sys.path, so the imports below resolve. We reuse its config + MuJoCo + signal classes.
from rls_training import TrainingConfig, TruckModel, PRBSSignal
from rls import nominal_rls_parameters
from params_mppi import WheelieParams


NPZ_PATH = Path(__file__).with_name("rls_batch.npz")
IMG_PATH = Path(__file__).with_name("images") / "rls_batch_id.png"
THETA_VALID_DEG = 110.0   # drop clearly post-flip samples (phi is valid but the dynamics are garbage)
HUBER_DELTA = 1.345       # Huber threshold in robust-sigma units (95% Gaussian efficiency)


# ============================================================
# 1) Data collection -- run the PRBS excitation, log matched (phi, qacc) pairs
# ============================================================

def collect_dataset(cfg: TrainingConfig, p: WheelieParams):
    """Drive the truck with PRBS torque and log a matched (phi, qacc) pair every control
    step. qacc is sampled AFTER mj_forward so it is consistent with the SAME (state, tau)
    that build phi -- no finite differencing, no half-step bias."""
    truck = TruckModel(cfg.xml_path, cfg.initial_z)
    signal = PRBSSignal(cfg.prbs_amp, cfg.prbs_dwell, cfg.prbs_seed)
    sim_dt = truck.sim_dt
    ctrl_steps = max(1, int(round(cfg.ctrl_dt / sim_dt)))

    rows = []
    truck.reset_pose()
    signal.reset()
    t0 = truck.time
    for s in range(int(round(cfg.duration / sim_dt))):
        if s % ctrl_steps == 0:
            st = truck.read_state()
            tau = float(np.clip(signal(truck.time - t0, st), p.tau_min, p.tau_max))
            truck.apply_ctrl(tau)
            # qacc consistent with (state, tau) -> matched target for phi (same instant)
            truck.forward()
            v_dot, omega_dot = truck.read_accel("qacc", ctrl_steps * sim_dt)
            x, v, th, om = st
            phi_v = [tau, v, abs(v) * v, tau * (np.cos(th) - 1.0), 1.0]
            phi_w = [np.cos(th), tau, om, v, 1.0]
            rows.append(phi_v + phi_w + [v_dot, omega_dot, th])
        truck.step()

    D = np.array(rows, dtype=float)
    valid = np.abs(np.degrees(D[:, 12])) <= THETA_VALID_DEG
    D = D[valid]
    return D[:, 0:5], D[:, 5:10], D[:, 10], D[:, 11]   # Phi_v, Phi_w, y_v, y_w


# ============================================================
# 2) Robust batch least-squares + uncertainty + cross-validation
# ============================================================

def huber_irls(Phi, y, delta=HUBER_DELTA, iters=15):
    """Robust weighted least-squares via iteratively-reweighted Huber. Returns
    (a, se, sigma, w): coefficients, +/-1 sigma per coeff (WLS covariance), robust
    residual sigma, and the per-sample Huber weights (w<1 == down-weighted outlier)."""
    a, *_ = np.linalg.lstsq(Phi, y, rcond=None)        # OLS start
    w = np.ones(len(y))
    for _ in range(iters):
        r = y - Phi @ a
        mad = np.median(np.abs(r - np.median(r))) + 1e-12
        thr = delta * 1.4826 * mad                     # robust scale -> Huber threshold
        ar = np.abs(r)
        w = np.where(ar <= thr, 1.0, thr / np.maximum(ar, 1e-12))
        WPhi = w[:, None] * Phi
        a = np.linalg.solve(Phi.T @ WPhi, Phi.T @ (w * y))
    r = y - Phi @ a
    dof = max(float(w.sum()) - Phi.shape[1], 1.0)
    sigma2 = float((w * r**2).sum() / dof)
    cov = sigma2 * np.linalg.inv(Phi.T @ (w[:, None] * Phi))
    return a, np.sqrt(np.maximum(np.diag(cov), 0.0)), float(np.sqrt(sigma2)), w


def trimmed_rmse(Phi, y, a, keep=0.9):
    """RMSE over the inlier residuals (drop the worst 1-keep fraction). Comparable to the
    streaming run's gated RMSE -- the contact/flip spikes are outliers, not model error."""
    r = np.abs(y - Phi @ a)
    m = r <= np.quantile(r, keep)
    rr = y[m] - Phi[m] @ a
    return float(np.sqrt(np.mean(rr ** 2)))


def kfold_cv(Phi, y, k=5, seed=0):
    """K-fold OUT-OF-SAMPLE RMSE: fit (robust) on k-1 folds, score (trimmed) on held-out."""
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, k)
    tr, va = [], []
    for i in range(k):
        vi = folds[i]
        ti = np.concatenate([folds[j] for j in range(k) if j != i])
        a, *_ = huber_irls(Phi[ti], y[ti])
        tr.append(trimmed_rmse(Phi[ti], y[ti], a))
        va.append(trimmed_rmse(Phi[vi], y[vi], a))
    return float(np.mean(tr)), float(np.mean(va)), float(np.std(va))


def normalized_cond(Phi, w):
    """Condition number of the RMS-normalized weighted info matrix (units removed) -- the
    interpretable conditioning/identifiability number (cf. pe_report's lambda_min/cond)."""
    rms = np.sqrt(np.mean((w[:, None] * Phi) ** 2, axis=0))
    Ps = Phi / np.where(rms < 1e-12, 1.0, rms)
    G = (Ps.T @ (w[:, None] * Ps)) / w.sum()
    ev = np.linalg.eigvalsh(0.5 * (G + G.T))
    return float(ev[-1] / max(ev[0], 1e-12))


# ============================================================
# Report + plot
# ============================================================

def fit_and_report(label, Phi, y, names, a_nom):
    a, se, sigma, w = huber_irls(Phi, y)
    cond = normalized_cond(Phi, w)
    tr, va, va_sd = kfold_cv(Phi, y)
    out_frac = float(np.mean(w < 0.999))

    print(f"\n========== {label} ==========")
    print(f"robust sigma={sigma:.4f}   normalized info cond={cond:.1f}   "
          f"down-weighted (outlier) fraction={out_frac:.2%}")
    print(f"train RMSE={tr:.4f}    5-fold CV RMSE={va:.4f} +/- {va_sd:.4f}   (out-of-sample)")
    print(f"{'coef':10s} {'learned':>10s} {'+/- 1sigma':>11s} {'z=a/sig':>9s}  {'nominal':>10s}")
    for i, nm in enumerate(names):
        z = a[i] / se[i] if se[i] > 0 else float("inf")
        flag = "" if abs(z) >= 2.0 else "   (~0, not significant)"
        print(f"{nm:10s} {a[i]:10.4f} {se[i]:11.4f} {z:9.1f}  {a_nom[i]:10.4f}{flag}")
    return a, se, w, (tr, va, va_sd)


def parity_panel(ax, Phi, y, a, w, title, unit):
    pred = Phi @ a
    keep = w > 0.999
    ax.scatter(pred[keep], y[keep], s=6, alpha=0.25, color="C0", label="inlier")
    ax.scatter(pred[~keep], y[~keep], s=10, alpha=0.5, color="C3", label="down-weighted")
    lim = float(np.percentile(np.abs(np.concatenate([pred, y])), 99)) or 1.0
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.0)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal", "box")
    r = y - pred
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - np.sum(r[keep] ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2) if keep.any() else np.nan
    ax.set_title(f"{title}   (inlier R$^2$={r2:.2f})")
    ax.set_xlabel(f"batch-fit prediction [{unit}]")
    ax.set_ylabel(f"measured qacc [{unit}]")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)


def main():
    # Headless (no viewer) + a longer PRBS run than the streaming default for richer batch data.
    cfg = TrainingConfig(render=False, duration=30.0)
    p = WheelieParams()
    a_nom = nominal_rls_parameters(p)
    Phi_v, Phi_w, y_v, y_w = collect_dataset(cfg, p)
    print(f"collected {len(y_v)} valid samples (|theta|<= {THETA_VALID_DEG:.0f} deg), "
          f"target = qacc (exact)")

    nv = ["b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0"]
    nw = ["a_g", "a_tau", "a_omega", "a_v", "a_0"]
    av, sev, wv, cvv = fit_and_report("v_dot channel", Phi_v, y_v, nv, a_nom[:5])
    aw, sew, ww, cvw = fit_and_report("omega_dot channel", Phi_w, y_w, nw, a_nom[5:])

    a = np.concatenate([av, aw])
    np.savez(NPZ_PATH, a_rls=a, se=np.concatenate([sev, sew]), a_nom=a_nom)
    print(f"\nSaved batch weights: {NPZ_PATH.name}")
    print("a_rls =", np.array2string(a, precision=6, separator=", "))

    # side-by-side with the streaming result, if present
    stream = NPZ_PATH.with_name("rls_trained.npz")
    if stream.exists():
        a_s = np.load(stream)["a_rls"]
        if a_s.shape == a.shape:
            print("\n  coef        batch      streaming")
            for i, nm in enumerate(nv + nw):
                print(f"  {nm:10s} {a[i]:10.4f}  {a_s[i]:10.4f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    parity_panel(a1, Phi_v, y_v, av, wv, "v_dot", "m/s$^2$")
    parity_panel(a2, Phi_w, y_w, aw, ww, "omega_dot", "rad/s$^2$")
    fig.suptitle("Batch robust RLS identification -- predicted vs measured (qacc)")
    fig.tight_layout()
    try:
        IMG_PATH.parent.mkdir(exist_ok=True)
        fig.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {IMG_PATH}")
    except PermissionError:
        alt = Path("/tmp/rls_batch_id.png")
        fig.savefig(alt, dpi=150, bbox_inches="tight")
        print(f"[warn] saved to {alt}")
    plt.show()


if __name__ == "__main__":
    main()
