#!/usr/bin/env python3
"""
SGP.py
------
The streaming sparse FITC Gaussian process (the recursive-FITC learner, the legacy
online_sparseGP.py method) used for online residual learning -- provided as a DROP-IN
ALTERNATIVE to SSGP.py so the two sparse-GP approximations can be compared head-to-head
on the same task.

SELF-CONTAINED: this file shares NO code with SSGP.py. It carries its own kernel,
inducing-point selection, single-output GP and two-output residual wrapper, with the
EXACT same public interface (observe / predict / predict_channels / mpc_params /
state_dict / load_state_dict, the SGPConfig fields) and the SAME state_dict keys as
SSGP.py -- so checkpoints are interchangeable and swapping
`from SSGP import SSGPConfig` -> `from SGP import SGPConfig as SSGPConfig`
in the driver changes ONLY VFE->FITC, nothing else.

FITC vs VFE (the one line that differs): both keep the inducing values u = f(Z) as a
linear-Gaussian state and fold each datum y_n = p_n^T u + e_n in by exact recursive
Bayes. They differ ONLY in the per-datum observation noise:

    VFE (SSGP.py):   R = sn2                              (Titsias 2009 bound)
    FITC (this):     R = sn2 + gamma_n,
                     gamma_n = k(x,x) - k(Z,x)^T Kzz^{-1} k(Z,x)   ( >= 0 )

gamma_n is the prior variance of the datum NOT explained by the inducing set: FITC
inflates the noise where x is far from Z (a poorly-covered point is trusted less),
the Snelson & Ghahramani (2006) / Quinonero-Candela & Rasmussen (2005) heteroscedastic
correction. Trade-off (Bauer, van der Wilk & Rasmussen 2016): FITC can UNDER-estimate
predictive variance and over-fit the noise, whereas VFE is more conservative. Which one
is better on THIS residual is empirical -- that is why this file exists.

    mean(x*) = p*^T m,   var(x*) = p*^T S p* + (k(x*,x*) - k(Z,x*)^T Kzz^{-1} k(Z,x*))
    p = Kzz^{-1} k(Z, x)

The residual wrapper emits the same mpc_params() values (Z, alpha_v_dot,
alpha_omega_dot) as SSGP, so it is a DROP-IN for the controller (rollout/cost/barriers
untouched). Pick it via SGPConfig().build() / AdaptiveSGPConfig().build().
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


def _grid_inducing(bounds, M, d):
    """Place ~M inducing points on a GRID covering `bounds` ([(lo,hi) per dim]) so the
    GP has support over the WHOLE domain -- not just where the warmup happened (the
    k-centre default confines Z to x~0, blinding the GP to the rest of the track).
    For d=2 it uses 2 theta levels (flat / reared) with the rest spread over x."""
    if d == 2:
        n1 = 2
        n0 = max(1, M // n1)
        xs = np.linspace(bounds[0][0], bounds[0][1], n0)
        ts = np.linspace(bounds[1][0], bounds[1][1], n1)
        Z = np.array([[x, t] for x in xs for t in ts], dtype=float)
    else:
        n = max(2, int(round(M ** (1.0 / d))))
        axes = [np.linspace(lo, hi, n) for (lo, hi) in bounds]
        grids = np.meshgrid(*axes, indexing="ij")
        Z = np.stack([g.ravel() for g in grids], axis=1).astype(float)
    return Z[:M]


# ── single-output streaming sparse FITC GP ───────────────────────────────────

class StreamingSparseFITC:
    """Single-output streaming sparse FITC GP with FIXED inducing inputs -- the
    recursive-Bayes update for fixed Z + hyperparameters. The inducing values u have an
    online posterior N(m, S); each datum is folded in with the FITC HETEROSCEDASTIC
    noise R = sn2 + gamma_n, gamma_n = k(x,x) - k(Z,x)^T Kzz^{-1} k(Z,x) (>=0) -- the
    one term that distinguishes it from the VFE update in SSGP.py (which uses R = sn2).
    Optional bounded process noise q keeps the posterior ADAPTIVE (online) instead of
    freezing after warmup; it is capped at the prior Kzz so S cannot diverge."""

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
        """One exact recursive-Bayes (FITC) step for datum (x, y)."""
        if self.Q is not None:                            # bounded fading -> stays adaptive
            self.S = self.S + self.Q
            np.fill_diagonal(self.S, np.minimum(np.diag(self.S), np.diag(self.Kzz0)))
        kx = rbf(np.asarray(x, float)[None, :], self.Z, self.ell, self.sf2)[0]
        p = self.Qzz @ kx
        gamma = self.sf2 - kx @ p                          # FITC variance not in Z (>=0)
        R = self.sn2 + max(gamma, 0.0)                     # FITC noise (VFE uses just sn2)
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

class SGPResidual:
    """The streaming FITC residual learner. Uses SHARED inducing inputs Z + one kernel
    for both channels (the MPPI/NMPC rollout evaluates a single kernel), with a separate
    streaming posterior per channel. Exposes Z, alpha_v_dot, alpha_omega_dot, active, l,
    sf2, sn2, mpc_params() so it is a drop-in for the controller, IDENTICAL in interface
    to SSGPResidual -- only the per-datum noise (FITC's sn2 + gamma) differs. Sets
    per-episode hyperparameters from a short warmup (empirical target variance for sf2 +
    a median-heuristic lengthscale), picks inducing inputs by greedy k-centre (or a grid
    over the domain), then streams. q>0 -> bounded online adaptivity."""

    def __init__(self, n_features=2, max_points=30, warmup=60, sn2_frac=0.1,
                 ell_scale=1.0, seed=0, q=0.0, lengthscales=None, inducing_bounds=None):
        self.d = n_features
        self.M = max_points
        self.warmup = warmup
        self.sn2_frac = sn2_frac
        self.ell_scale = ell_scale
        self.q = float(q)
        self.lengthscales = lengthscales          # if set: FIXED kernel (not warmup-fit)
        self.inducing_bounds = inducing_bounds     # if set: GRID inducing pts over domain
        self.rng = np.random.default_rng(seed)
        self._buf = []
        self.ready = False
        self.n_seen = 0
        # controller-facing attributes (placeholders until warmup completes)
        self.Z = np.zeros((self.M, self.d))
        self.alpha_v_dot = np.zeros(self.M)
        self.alpha_omega_dot = np.zeros(self.M)
        self.active = np.zeros(self.M, dtype=bool)
        self.l = np.ones(self.d)
        self.sf2 = 1.0
        self.sn2 = 0.25
        self.gp_v_dot = self.gp_omega_dot = None

    def _init(self):
        X = np.array([b[0] for b in self._buf])
        Yv = np.array([b[1] for b in self._buf]); Yw = np.array([b[2] for b in self._buf])
        if self.inducing_bounds is not None:
            # FIXED grid over the whole domain + FIXED lengthscales. The warmup buffer
            # only covers x~0, so data-driven Z (k-centre) and ell (median) would blind
            # the GP to the obstacles further along -- this is the coverage fix.
            Z = _grid_inducing(self.inducing_bounds, self.M, self.d)
            ell = np.asarray(self.lengthscales, float).reshape(-1)
        else:
            ell = np.empty(self.d)                   # median-heuristic ARD lengthscales
            for k in range(self.d):
                dif = np.abs(X[:, k][:, None] - X[:, k][None, :])
                med = np.median(dif[dif > 0]) if np.any(dif > 0) else 1.0
                ell[k] = self.ell_scale * max(med, 1e-2)
            Z = X[_kcenter(X, self.M, self.rng)]
        sf2 = max(float(np.var(Yv)), float(np.var(Yw)), 1e-3)
        sn2 = max(self.sn2_frac * sf2, 1e-3)
        self.gp_v_dot = StreamingSparseFITC(Z, ell, sf2, sn2, q=self.q)
        self.gp_omega_dot = StreamingSparseFITC(Z, ell, sf2, sn2, q=self.q)
        for z, rv, rw in self._buf:
            self.gp_v_dot.update(z, rv); self.gp_omega_dot.update(z, rw)
        self.Z = Z; self.l = ell; self.sf2 = sf2; self.sn2 = sn2
        self.active = np.ones(Z.shape[0], dtype=bool)
        self._refresh_alpha()
        self.ready = True

    def _refresh_alpha(self):
        self.alpha_v_dot = self.gp_v_dot.alpha
        self.alpha_omega_dot = self.gp_omega_dot.alpha

    def observe(self, z, r_v_dot, r_omega_dot):
        self.n_seen += 1
        z = np.asarray(z, float)
        if not self.ready:
            self._buf.append((z, float(r_v_dot), float(r_omega_dot)))
            if len(self._buf) >= self.warmup:
                self._init()
            return
        self.gp_v_dot.update(z, r_v_dot); self.gp_omega_dot.update(z, r_omega_dot)
        self._refresh_alpha()

    def predict(self, z):
        if not self.ready:
            return 0.0, 0.0, 1.0
        mv, vv = self.gp_v_dot.predict(z)
        mw, vw = self.gp_omega_dot.predict(z)
        return mv, mw, float(np.sqrt(0.5 * (vv + self.sn2 + vw + self.sn2)))

    def predict_channels(self, z):
        """Per-channel (mean, std) at z -> (mv, sv, mw, sw). For plotting/diagnostics
        of the residual model (the controller uses predict()/mpc_params())."""
        if not self.ready:
            s = float(np.sqrt(self.sf2))
            return 0.0, s, 0.0, s
        mv, vv = self.gp_v_dot.predict(z)
        mw, vw = self.gp_omega_dot.predict(z)
        return mv, float(np.sqrt(vv + self.sn2)), mw, float(np.sqrt(vw + self.sn2))

    def refit(self):
        pass

    def mpc_params(self):
        return np.concatenate([self.Z.reshape(-1), self.alpha_v_dot, self.alpha_omega_dot])

    def state_dict(self) -> dict:
        """Serialise the LEARNED state (fitted kernel + per-channel posterior) so the
        GP can be restored in a later run via load_state_dict(). Same keys as
        SSGPResidual, so SGP/SSGP checkpoints are interchangeable. Returns {} if the GP
        has not warmed up yet (nothing learned to save)."""
        if not self.ready:
            return {}
        return {
            "n_seen": self.n_seen,
            "q": self.q,
            "Z": self.Z,
            "l": self.l,
            "sf2": self.sf2,
            "sn2": self.sn2,
            "active": self.active,
            "m_v_dot": self.gp_v_dot.m,
            "S_v_dot": self.gp_v_dot.S,
            "m_omega_dot": self.gp_omega_dot.m,
            "S_omega_dot": self.gp_omega_dot.S,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore the learned state produced by state_dict() (e.g. from a previous
        run) so this GP continues with that obstacle knowledge instead of warming up
        from scratch. The two single-output GPs are rebuilt from (Z, l, sf2, sn2) --
        which regenerates Kzz/Qzz -- then their posteriors (m, S) are overwritten."""
        Z = np.asarray(sd["Z"], float)
        if Z.shape != (self.M, self.d):
            raise ValueError(
                f"checkpoint Z shape {Z.shape} != expected ({self.M}, {self.d}); "
                f"the saved model used a different max_points/n_features."
            )
        self.Z = Z
        self.l = np.asarray(sd["l"], float)
        self.sf2 = float(sd["sf2"])
        self.sn2 = float(sd["sn2"])
        self.q = float(sd["q"])
        self.gp_v_dot = StreamingSparseFITC(self.Z, self.l, self.sf2, self.sn2, q=self.q)
        self.gp_omega_dot = StreamingSparseFITC(self.Z, self.l, self.sf2, self.sn2, q=self.q)
        self.gp_v_dot.m = np.asarray(sd["m_v_dot"], float)
        self.gp_v_dot.S = np.asarray(sd["S_v_dot"], float)
        self.gp_omega_dot.m = np.asarray(sd["m_omega_dot"], float)
        self.gp_omega_dot.S = np.asarray(sd["S_omega_dot"], float)
        self.active = np.asarray(sd["active"], bool)
        self.n_seen = int(sd["n_seen"])
        self.ready = True
        self._refresh_alpha()

    @property
    def n_active(self):
        return int(self.active.sum())


# ── configs (each builds its own learner -- no external factory) ─────────────

@dataclass
class SGPConfig:
    """Streaming sparse FITC GP (recursive FITC) -- the DROP-IN ALTERNATIVE to
    SSGPConfig. Fields are IDENTICAL to SSGPConfig (same inducing grid, lengthscales,
    warmup, sf2, sn2_frac) so a run differs from the VFE baseline ONLY by the FITC
    noise R = sn2 + gamma. Note: FITC's +gamma already adds noise where Z is sparse, so
    if it over-regularises you can LOWER sn2_frac relative to the VFE setting; if it
    over-fits the contact spikes, raise it."""
    max_points: int = 50        # inducing-set size M (also the controller's rollout M)
    warmup: int = 60            # control steps buffered before the SGP fits its kernel
    sn2_frac: float = 1.0       # base noise/signal (FITC adds +gamma on top per point)
    refit_every: int = 1        # harness-loop cadence (the SGP updates online; ~no-op)
    # ARD lengthscales for feature z = [x, theta]. The obstacle is a function of POSITION
    # (and pitch), so the GP lives in (x, theta) only. x sharp (~0.4 m, resolve the box);
    # theta broad (~0.7 rad, interpolate flat<->reared). FIXED (used by both GP and NMPC).
    lengthscales: tuple = (0.40, 0.70)
    # Inducing points on a GRID over (x, theta) so the GP has support over the WHOLE
    # track -- not just the warmup region near x=0. Without this the GP is blind past
    # the start and predicts 0 at every obstacle.
    inducing_bounds: tuple = ((0.0, 11.0), (0.0, 1.20))
    sf2: float = 4.0            # initial rollout kernel scale; the SGP fits its own online

    def build(self, n_features=2):
        return SGPResidual(n_features=n_features, max_points=self.max_points,
                           warmup=self.warmup, sn2_frac=self.sn2_frac, q=0.0,
                           lengthscales=self.lengthscales,
                           inducing_bounds=self.inducing_bounds)


@dataclass
class AdaptiveSGPConfig(SGPConfig):
    """Streaming FITC sparse GP that stays ADAPTIVE online (bounded process noise q):
    it keeps tracking instead of freezing after warmup. q is kept tiny (<=1e-3) because
    the contact residual is irreducible -- any real adaptivity just tracks noise and
    destabilises the brittle controller (q>=3e-3 flips). Use it only on a system whose
    LEARNABLE dynamics actually drift (e.g. real hardware)."""
    q: float = 0.001

    def build(self, n_features=2):
        return SGPResidual(n_features=n_features, max_points=self.max_points,
                           warmup=self.warmup, sn2_frac=self.sn2_frac, q=self.q,
                           lengthscales=self.lengthscales,
                           inducing_bounds=self.inducing_bounds)


if __name__ == "__main__":
    # quick check: the FITC update runs and produces finite inducing weights
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(8, 4)); ell = np.ones(4)
    gp = StreamingSparseFITC(Z, ell, 1.0, 0.5)
    for _ in range(50):
        gp.update(rng.normal(size=4), float(rng.normal()))
    m, v = gp.predict(rng.normal(size=4))
    print(f"StreamingSparseFITC ok: mean={m:+.3f} var={v:.3f} alpha_finite={np.all(np.isfinite(gp.alpha))}")
    res = SGPConfig().build()           # 2-D feature z = [x, theta]
    for _ in range(80):
        res.observe(rng.normal(size=2), float(rng.normal()), float(rng.normal()))
    print(f"SGPResidual ok: ready={res.ready} n_active={res.n_active} "
          f"mpc_params={res.mpc_params().shape}")
