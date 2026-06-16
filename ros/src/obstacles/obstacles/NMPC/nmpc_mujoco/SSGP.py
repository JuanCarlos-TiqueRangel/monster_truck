#!/usr/bin/env python3
"""
SSGP.py
-------
The streaming sparse VARIATIONAL Gaussian process (VFE) used for online residual
learning -- the modern, principled learner the controller calls by default.

SELF-CONTAINED: this file shares NO code with online_sparseGP.py (the recursive-FITC
legacy). It carries its own kernel, inducing-point selection, single-output GP and
two-output residual wrapper, so SSGP.py and online_sparseGP.py are fully independent
and can be read / changed in isolation.

This implements the streaming sparse variational GP of Bui, Nguyen & Turner
("Streaming Sparse Gaussian Process Approximations", NeurIPS 2017), specialised to
FIXED inducing inputs and FIXED hyperparameters (appropriate for a real-time
controller with a fixed inducing set). In that case their online variational update
reduces to the EXACT recursive VFE posterior implemented here -- the per-datum noise
is R = sn2 (the Titsias 2009 bound), NOT FITC's R = sn2 + gamma_n. Unlike FITC, VFE
does not under-estimate the predictive variance (Bauer, van der Wilk & Rasmussen,
2016), so its uncertainty is honest -- which is the whole point of keeping a GP here,
since the residual MEAN is irreducible (see the project notes).

Method: with inducing inputs Z fixed, the inducing values u = f(Z) are a
linear-Gaussian state and each datum y_n = p_n^T u + e_n is a linear observation, so
the online posterior over u is the EXACT recursive-Bayes (Kalman/information) update:

        mean(x*) = p*^T m,    var(x*) = p*^T S p* + (k(x*,x*) - k(Z,x*)^T Kzz^{-1} k(Z,x*))
        p = Kzz^{-1} k(Z, x)

The residual wrapper emits the same mpc_params() values (Z, alpha_v, alpha_omega) as
the legacy learner, so it is a DROP-IN for the controller (rollout/cost/barriers are
untouched). Pick it via SSGPConfig().build() / AdaptiveSSGPConfig().build().
"""

from dataclasses import dataclass

import numpy as np


# ── kernel ───────────────────────────────────────────────────────────────────

def rbf(A, B, ell, sf2):
    """ARD RBF kernel between rows of A (n,d) and B (m,d) -> (n,m)."""
    Al = A / ell
    Bl = B / ell
    sq = (np.sum(Al**2, 1)[:, None] + np.sum(Bl**2, 1)[None, :] - 2.0 * Al @ Bl.T)
    return sf2 * np.exp(-0.5 * np.maximum(sq, 0.0))


# ── greedy k-centre inducing-point selection ─────────────────────────────────

def _kcenter(X, M, rng):
    n = X.shape[0]
    if n <= M:
        return np.arange(n)
    Xw = (X - X.mean(0)) / (X.std(0) + 1e-9)         # whiten
    idx = [int(rng.integers(n))]
    d2 = np.sum((Xw - Xw[idx[0]])**2, 1)
    for _ in range(M - 1):
        j = int(np.argmax(d2))
        idx.append(j)
        d2 = np.minimum(d2, np.sum((Xw - Xw[j])**2, 1))
    return np.array(idx)


# ── single-output streaming sparse VARIATIONAL (VFE) GP ──────────────────────

class StreamingSparseVGP:
    """Single-output streaming sparse VARIATIONAL (VFE / Titsias) GP with FIXED
    inducing inputs -- the BNT 2017 streaming update for fixed Z + hyperparameters.
    The inducing values u have an online posterior N(m, S), updated by exact
    recursive Bayes with the VFE HOMOSCEDASTIC noise R = sn2 (FITC would use
    R = sn2 + gamma). Optional bounded process noise q keeps the posterior ADAPTIVE
    (online) instead of freezing after warmup; it is capped at the prior Kzz so S
    cannot diverge."""

    def __init__(self, Z, ell, sf2, sn2, q=0.0, jitter=1e-6):
        self.Z = np.asarray(Z, float)
        self.ell = np.asarray(ell, float).reshape(-1)
        self.sf2 = float(sf2)
        self.sn2 = float(sn2)
        self.M = self.Z.shape[0]
        Kzz = rbf(self.Z, self.Z, self.ell, self.sf2) + jitter * np.eye(self.M)
        self.Qzz = np.linalg.inv(Kzz)
        self.m = np.zeros(self.M)        # posterior mean over u  (prior: 0)
        self.S = Kzz.copy()              # posterior cov  over u  (prior: Kzz)
        self.Kzz0 = np.linalg.inv(self.Qzz)               # prior covariance (= Kzz)
        self.q = float(q)
        self.Q = self.q * self.Kzz0 if self.q > 0.0 else None

    def update(self, x, y):
        """One exact recursive-Bayes (VFE) step for datum (x, y)."""
        if self.Q is not None:                            # bounded fading -> stays adaptive
            self.S = self.S + self.Q
            np.fill_diagonal(self.S, np.minimum(np.diag(self.S), np.diag(self.Kzz0)))
        kx = rbf(np.asarray(x, float)[None, :], self.Z, self.ell, self.sf2)[0]
        p = self.Qzz @ kx
        R = self.sn2                                      # VFE noise (FITC would add +gamma)
        Sp = self.S @ p
        denom = R + p @ Sp
        if denom <= 1e-12:
            return
        Kg = Sp / denom
        self.m = self.m + Kg * (float(y) - p @ self.m)
        self.S = self.S - np.outer(Kg, Sp)
        self.S = 0.5 * (self.S + self.S.T)

    def predict(self, x):
        kx = rbf(np.asarray(x, float)[None, :], self.Z, self.ell, self.sf2)[0]
        p = self.Qzz @ kx
        mean = float(p @ self.m)
        var = float(p @ (self.S @ p) + max(self.sf2 - kx @ p, 0.0))
        return mean, max(var, 0.0)

    @property
    def alpha(self):                      # mean(x*) = k(x*,Z) . alpha
        return self.Qzz @ self.m


# ── two-output residual wrapper (drop-in for the controller) ─────────────────

class SSGPResidual:
    """The streaming variational (VFE) residual learner. Uses SHARED inducing inputs
    Z + one kernel for both channels (the MPPI/NMPC rollout evaluates a single
    kernel), with a separate streaming posterior per channel. Exposes
    Z, alpha_v, alpha_omega, active, l, sf2, sn2, mpc_params() so it is a drop-in
    for the controller. Sets per-episode hyperparameters from a short warmup
    (empirical target variance for sf2 + a median-heuristic lengthscale), picks
    inducing inputs by greedy k-centre, then streams. q>0 -> bounded online
    adaptivity."""

    def __init__(self, n_features=4, max_points=30, warmup=60, sn2_frac=0.1,
                 ell_scale=1.0, seed=0, q=0.0):
        self.d = n_features
        self.M = max_points
        self.warmup = warmup
        self.sn2_frac = sn2_frac
        self.ell_scale = ell_scale
        self.q = float(q)
        self.rng = np.random.default_rng(seed)
        self._buf = []
        self.ready = False
        self.n_seen = 0
        # controller-facing attributes (placeholders until warmup completes)
        self.Z = np.zeros((self.M, self.d))
        self.alpha_v = np.zeros(self.M)
        self.alpha_omega = np.zeros(self.M)
        self.active = np.zeros(self.M, dtype=bool)
        self.l = np.ones(self.d)
        self.sf2 = 1.0
        self.sn2 = 0.25
        self.gp_v = self.gp_w = None

    def _init(self):
        X = np.array([b[0] for b in self._buf])
        Yv = np.array([b[1] for b in self._buf]); Yw = np.array([b[2] for b in self._buf])
        ell = np.empty(self.d)                       # median-heuristic ARD lengthscales
        for k in range(self.d):
            dif = np.abs(X[:, k][:, None] - X[:, k][None, :])
            med = np.median(dif[dif > 0]) if np.any(dif > 0) else 1.0
            ell[k] = self.ell_scale * max(med, 1e-2)
        sf2 = max(float(np.var(Yv)), float(np.var(Yw)), 1e-3)
        sn2 = max(self.sn2_frac * sf2, 1e-3)
        Z = X[_kcenter(X, self.M, self.rng)]
        self.gp_v = StreamingSparseVGP(Z, ell, sf2, sn2, q=self.q)
        self.gp_w = StreamingSparseVGP(Z, ell, sf2, sn2, q=self.q)
        for z, rv, rw in self._buf:
            self.gp_v.update(z, rv); self.gp_w.update(z, rw)
        self.Z = Z; self.l = ell; self.sf2 = sf2; self.sn2 = sn2
        self.active = np.ones(Z.shape[0], dtype=bool)
        self._refresh_alpha()
        self.ready = True

    def _refresh_alpha(self):
        self.alpha_v = self.gp_v.alpha
        self.alpha_omega = self.gp_w.alpha

    def observe(self, z, r_v, r_omega):
        self.n_seen += 1
        z = np.asarray(z, float)
        if not self.ready:
            self._buf.append((z, float(r_v), float(r_omega)))
            if len(self._buf) >= self.warmup:
                self._init()
            return
        self.gp_v.update(z, r_v); self.gp_w.update(z, r_omega)
        self._refresh_alpha()

    def predict(self, z):
        if not self.ready:
            return 0.0, 0.0, 1.0
        mv, vv = self.gp_v.predict(z)
        mw, vw = self.gp_w.predict(z)
        return mv, mw, float(np.sqrt(0.5 * (vv + self.sn2 + vw + self.sn2)))

    def predict_channels(self, z):
        """Per-channel (mean, std) at z -> (mv, sv, mw, sw). For plotting/diagnostics
        of the residual model (the controller uses predict()/mpc_params())."""
        if not self.ready:
            s = float(np.sqrt(self.sf2))
            return 0.0, s, 0.0, s
        mv, vv = self.gp_v.predict(z)
        mw, vw = self.gp_w.predict(z)
        return mv, float(np.sqrt(vv + self.sn2)), mw, float(np.sqrt(vw + self.sn2))

    def refit(self):
        pass

    def mpc_params(self):
        return np.concatenate([self.Z.reshape(-1), self.alpha_v, self.alpha_omega])

    @property
    def n_active(self):
        return int(self.active.sum())


# ── configs (each builds its own learner -- no external factory) ─────────────

@dataclass
class SSGPConfig:
    """Streaming sparse VARIATIONAL GP (VFE, BNT 2017) -- the default learner.
    sn2_frac=4.0 (vs FITC's 2.0): VFE's noise is R=sn2 without FITC's +gamma, so it
    needs the higher frac to regularise the irreducible residual toward ~0 to the
    same degree (else it injects residual noise into the seed-brittle rollout).
    Verified to clear all 3 obstacles and stop at the goal."""
    max_points: int = 50        # inducing-set size M (also the controller's rollout M)
    warmup: int = 60            # control steps buffered before the SGP fits its kernel
    sn2_frac: float = 4.0       # noise/signal -> regularises the irreducible residual ~0
    refit_every: int = 1        # harness-loop cadence (the SGP updates online; ~no-op)
    lengthscales: tuple = (0.30, 2.0, 1.0, 3.0)   # controller's INITIAL rollout kernel (ARD)
    sf2: float = 4.0            # initial rollout kernel scale; the SGP fits its own online

    def build(self, n_features=4):
        return SSGPResidual(n_features=n_features, max_points=self.max_points,
                            warmup=self.warmup, sn2_frac=self.sn2_frac, q=0.0)


@dataclass
class AdaptiveSSGPConfig(SSGPConfig):
    """Streaming VFE sparse GP that stays ADAPTIVE online (bounded process noise q):
    it keeps tracking instead of freezing after warmup. q is kept tiny (<=1e-3)
    because the contact residual is irreducible -- any real adaptivity just tracks
    noise and destabilises the brittle controller (q>=3e-3 flips). Use it only on a
    system whose LEARNABLE dynamics actually drift (e.g. real hardware)."""
    q: float = 0.001

    def build(self, n_features=4):
        return SSGPResidual(n_features=n_features, max_points=self.max_points,
                            warmup=self.warmup, sn2_frac=self.sn2_frac, q=self.q)


if __name__ == "__main__":
    # quick check: the VFE update runs and produces finite inducing weights
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 4)); ell = np.ones(4)
    gp = StreamingSparseVGP(Z, ell, 1.0, 0.5)
    for _ in range(50):
        gp.update(rng.normal(size=4), float(rng.normal()))
    m, v = gp.predict(rng.normal(size=4))
    print(f"StreamingSparseVGP ok: mean={m:+.3f} var={v:.3f} alpha_finite={np.all(np.isfinite(gp.alpha))}")
    res = SSGPConfig().build()
    for _ in range(80):
        res.observe(rng.normal(size=4), float(rng.normal()), float(rng.normal()))
    print(f"SSGPResidual ok: ready={res.ready} n_active={res.n_active} "
          f"mpc_params={res.mpc_params().shape}")
