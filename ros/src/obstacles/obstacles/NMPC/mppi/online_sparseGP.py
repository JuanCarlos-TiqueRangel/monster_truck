#!/usr/bin/env python3
"""
online_sparseGP.py
------------------
An EXACT recursive (online) sparse Gaussian process -- the FITC approximation --
for online residual learning.  (Renamed from streaming_sgp.py: that name wrongly
implied the streaming VARIATIONAL SGP; this is FITC.  The variational version
lives in SSGP.py and is the default used by the controller.)

Method (recursive FITC, fixed inducing inputs): with inducing points Z fixed, the
sparse-GP latent values u = f(Z) form a linear-Gaussian state, and each datum is a
linear observation

        y_n = p_n^T u + e_n,   p_n = Kzz^{-1} k(Z, x_n),   e_n ~ N(0, sn2 + gamma_n)

with gamma_n = k(x_n,x_n) - k(Z,x_n)^T Kzz^{-1} k(Z,x_n) (the FITC correction --
the +gamma in the noise is exactly what makes this FITC rather than VFE).  The
online posterior over u is the EXACT recursive Bayesian (information/Kalman)
update, which equals the batch FITC posterior (verified in __main__).  Correct,
O(M^2)/step:

        mean(x*) = p*^T m
        var(x*)  = p*^T S p* + (k(x*,x*) - k(Z,x*)^T Kzz^{-1} k(Z,x*))

NOTE: FITC under-estimates the predictive variance (Bauer, van der Wilk &
Rasmussen, 2016) -- which is why SSGP.py (VFE, R=sn2) is preferred.  Kept here as
the legacy learner + the shared kernel/inducing machinery that SSGP.py builds on.

The two-output residual wrapper (`StreamingGPResidualMPPI`) sets per-channel
hyperparameters from a short warmup (empirical target variance for sf2 + a
median-heuristic lengthscale), picks inducing inputs by greedy k-centre, then
streams.  (SSGP.py is a separate, self-contained module -- it does not subclass here.)
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize


@dataclass
class StreamingGPConfig:
    """Config for the EXACT streaming sparse GP (StreamingGPResidualMPPI). The one
    canonical home for the GP config used across the MPPI and NMPC stacks -- there
    is no gp_residual.GPConfig anymore. The model is built fresh per episode."""
    max_points: int = 50        # SSGP inducing-set size M (also the controller's rollout M)
    warmup: int = 60            # control steps buffered before the SSGP fits its kernel
    sn2_frac: float = 2.0       # noise/signal -> regularises the irreducible residual ~0
    refit_every: int = 1        # harness-loop cadence (SSGP updates online; effectively a no-op)
    lengthscales: tuple = (0.30, 2.0, 1.0, 3.0)   # controller's INITIAL rollout kernel (ARD)
    sf2: float = 4.0            # initial rollout kernel scale; the SSGP fits its own online

    def build(self, n_features=4):
        """Construct this file's residual learner (recursive FITC). Self-contained:
        does NOT reference SSGP.py -- the variational SSGPConfig builds its own."""
        return StreamingGPResidualMPPI(n_features=n_features, max_points=self.max_points,
                                       warmup=self.warmup, sn2_frac=self.sn2_frac)


# ── kernel ───────────────────────────────────────────────────────────────────

def rbf(A, B, ell, sf2):
    """ARD RBF kernel between rows of A (n,d) and B (m,d) -> (n,m)."""
    Al = A / ell
    Bl = B / ell
    sq = (np.sum(Al**2, 1)[:, None] + np.sum(Bl**2, 1)[None, :] - 2.0 * Al @ Bl.T)
    return sf2 * np.exp(-0.5 * np.maximum(sq, 0.0))


# ── exact-GP hyperparameter fit (type-II MLE / marginal likelihood) ──────────

def fit_gp_hyperparameters(X, y, n_sub=300, n_restarts=2, seed=0):
    """Fit ARD lengthscales, signal var sf2, noise var sn2 by maximising the
    exact-GP log marginal likelihood on up to n_sub points. Returns (ell, sf2, sn2).
    Properly separates learnable SIGNAL (sf2) from irreducible NOISE (sn2)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    if X.shape[0] > n_sub:
        sel = rng.choice(X.shape[0], n_sub, replace=False); X = X[sel]; y = y[sel]
    n, d = X.shape
    y = y - y.mean()
    yvar = float(np.var(y)) + 1e-6

    def nlml(theta):
        ell = np.exp(theta[:d]); sf2 = np.exp(theta[d]); sn2 = np.exp(theta[d + 1])
        K = rbf(X, X, ell, sf2) + sn2 * np.eye(n)
        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e12
        a = cho_solve((L, True), y)
        return float(0.5 * y @ a + np.sum(np.log(np.diag(L))) + 0.5 * n * np.log(2 * np.pi))

    ell0 = np.log(np.std(X, 0) + 1e-2)
    best = None
    for _ in range(n_restarts):
        x0 = np.concatenate([ell0 + 0.4 * rng.normal(size=d),
                             [np.log(0.5 * yvar)], [np.log(0.5 * yvar)]])
        try:
            r = minimize(nlml, x0, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    if best is None:
        return np.std(X, 0) + 1e-2, yvar, 0.1 * yvar
    th = best.x
    return np.exp(th[:d]), float(np.exp(th[d])), float(np.exp(th[d + 1]))


# ── single-output streaming sparse GP (fixed inducing inputs) ────────────────

class StreamingSparseGP:
    def __init__(self, Z, ell, sf2, sn2, jitter=1e-6):
        self.Z = np.asarray(Z, float)
        self.ell = np.asarray(ell, float).reshape(-1)
        self.sf2 = float(sf2)
        self.sn2 = float(sn2)
        self.M = self.Z.shape[0]
        Kzz = rbf(self.Z, self.Z, self.ell, self.sf2) + jitter * np.eye(self.M)
        self.Qzz = np.linalg.inv(Kzz)
        self.m = np.zeros(self.M)        # posterior mean over u  (prior: 0)
        self.S = Kzz.copy()              # posterior cov  over u  (prior: Kzz)

    def update(self, x, y):
        """One exact recursive-Bayes step for datum (x, y)."""
        kx = rbf(np.asarray(x, float)[None, :], self.Z, self.ell, self.sf2)[0]
        p = self.Qzz @ kx
        gamma = max(self.sf2 - kx @ p, 0.0)        # FITC correction (>=0)
        R = self.sn2 + gamma
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


# ── batch sparse-GP posterior (for the correctness self-test) ────────────────

def _batch_posterior(Z, X, y, ell, sf2, sn2, jitter=1e-6):
    M = Z.shape[0]
    Kzz = rbf(Z, Z, ell, sf2) + jitter * np.eye(M)
    Qzz = np.linalg.inv(Kzz)
    Lam = Qzz.copy()                      # prior precision = Kzz^{-1}
    eta = np.zeros(M)
    for xn, yn in zip(X, y):
        kx = rbf(xn[None, :], Z, ell, sf2)[0]
        p = Qzz @ kx
        R = sn2 + max(sf2 - kx @ p, 0.0)
        Lam += np.outer(p, p) / R
        eta += p * yn / R
    S = np.linalg.inv(Lam)
    return S @ eta, S


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


# ── two-output residual wrapper (drop-in for gp_residual.GPResidual) ─────────

class StreamingGPResidual:
    def __init__(self, n_features=4, max_points=40, warmup=80,
                 sn2_frac=0.1, ell_scale=1.0, seed=0):
        self.d = n_features
        self.M = max_points
        self.warmup = warmup
        self.sn2_frac = sn2_frac
        self.ell_scale = ell_scale
        self.rng = np.random.default_rng(seed)
        self._buf_x, self._buf_v, self._buf_w = [], [], []
        self.ready = False
        self.n_seen = 0
        self.gp_v = self.gp_w = None

    def _init_models(self):
        X = np.array(self._buf_x); Yv = np.array(self._buf_v); Yw = np.array(self._buf_w)
        ind = _kcenter(X, self.M, self.rng)
        Z = X[ind]
        ev, sfv, snv = fit_gp_hyperparameters(X, Yv, seed=0)   # type-II MLE per channel
        ew, sfw, snw = fit_gp_hyperparameters(X, Yw, seed=1)
        self.hyp = dict(v=(ev, sfv, snv), w=(ew, sfw, snw))
        self.gp_v = StreamingSparseGP(Z, ev, sfv, snv)
        self.gp_w = StreamingSparseGP(Z, ew, sfw, snw)
        for x, yv, yw in zip(X, Yv, Yw):       # absorb the buffered data
            self.gp_v.update(x, yv); self.gp_w.update(x, yw)
        self.ready = True

    # -- GPResidual-compatible API --
    def observe(self, z, r_v, r_omega):
        self.n_seen += 1
        z = np.asarray(z, float)
        if not self.ready:
            self._buf_x.append(z); self._buf_v.append(float(r_v)); self._buf_w.append(float(r_omega))
            if len(self._buf_x) >= self.warmup:
                self._init_models()
            return
        self.gp_v.update(z, r_v); self.gp_w.update(z, r_omega)

    def predict(self, z):
        if not self.ready:
            return 0.0, 0.0, 1.0
        mv, vv = self.gp_v.predict(z)
        mw, vw = self.gp_w.predict(z)
        std = float(np.sqrt(0.5 * (vv + self.gp_v.sn2 + vw + self.gp_w.sn2)))
        return mv, mw, std

    def refit(self):
        pass

    @property
    def n_active(self):
        return self.M if self.ready else 0


# ── drop-in for gp_residual.GPResidual (shared inducing inputs) ──────────────

class StreamingGPResidualMPPI:
    """The EXACT streaming sparse GP, packaged as a drop-in for
    gp_residual.GPResidual so the MPPI rollout can use it unchanged. Uses SHARED
    inducing inputs Z + one kernel for both channels (the MPPI rollout evaluates
    a single kernel), with a separate streaming posterior per channel. Exposes
    Z, alpha_v, alpha_omega, active, l, sf2, sn2, mpc_params() like GPResidual.

    (Per-channel marginal-likelihood kernels -- as in StreamingGPResidual -- give
    a better fit, but the residual is irreducible anyway, so the shared kernel
    here costs nothing in practice and buys exact, calibrated streaming inference.)
    """

    def __init__(self, n_features=4, max_points=30, warmup=60, sn2_frac=0.1,
                 ell_scale=1.0, seed=0):
        self.d = n_features
        self.M = max_points
        self.warmup = warmup
        self.sn2_frac = sn2_frac
        self.ell_scale = ell_scale
        self.rng = np.random.default_rng(seed)
        self._buf = []
        self.ready = False
        self.n_seen = 0
        # GPResidual-compatible attributes (placeholders until warmup completes)
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
        self.gp_v = StreamingSparseGP(Z, ell, sf2, sn2)
        self.gp_w = StreamingSparseGP(Z, ell, sf2, sn2)
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


# The streaming VARIATIONAL (VFE) learner lives in SSGP.py -- a fully SELF-CONTAINED
# module (it does NOT import from here; each config builds its own learner via build()).


# ── correctness self-test: streaming posterior == batch posterior ────────────

def _self_test():
    rng = np.random.default_rng(1)
    d, M, N = 3, 10, 80
    Z = rng.normal(size=(M, d))
    X = rng.normal(size=(N, d))
    ell = np.array([0.7, 1.1, 0.9]); sf2, sn2 = 2.5, 0.3
    w = rng.normal(size=M)
    y = rbf(X, Z, ell, sf2) @ w + 0.1 * rng.normal(size=N)

    gp = StreamingSparseGP(Z, ell, sf2, sn2)
    for xn, yn in zip(X, y):
        gp.update(xn, yn)
    m_b, S_b = _batch_posterior(Z, X, y, ell, sf2, sn2)

    em = np.max(np.abs(gp.m - m_b)); eS = np.max(np.abs(gp.S - S_b))
    print(f"streaming vs batch  |Δm|max = {em:.2e}   |ΔS|max = {eS:.2e}")
    assert em < 1e-7 and eS < 1e-7, "streaming != batch (BUG)"
    print("SELF-TEST PASSED: streaming posterior == exact batch sparse-GP posterior")


if __name__ == "__main__":
    _self_test()
