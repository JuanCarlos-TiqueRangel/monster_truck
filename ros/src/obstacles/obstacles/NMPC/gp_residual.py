#!/usr/bin/env python3
"""
gp_residual.py
---------------
Sparse Gaussian-Process residual model designed to plug into a CasADi NMPC.

WHY THIS FORM
=============
A full GP mean is  mu(z*) = k(z*, X) (K + sn2 I)^-1 y, which depends on ALL
training points. That is not tractable inside a real-time IPOPT solve.

Instead we use a SPARSE GP with a small dictionary of M inducing points Z
(M ~ 15-25). The mean becomes

        mu(z*) = sum_i alpha_i * k(z*, Z_i)

which is a fixed sum of M kernel terms. We pass Z and alpha to the solver as
PARAMETERS, so the CasADi computation graph is built once and only the numeric
values change as the GP learns online. This is the standard way GPs are put
inside MPC (subset-of-regressors / fixed-inducing sparse GP).

WHAT IT MODELS
==============
The residual between measured dynamics and the nominal (physics + RLS) model:

    r_omega = omega_dot_measured - omega_dot_nominal     (pitch residual)
    r_v     = v_dot_measured     - v_dot_nominal         (longitudinal residual)

During free rolling this residual is ~0. When a wheel hits an obstacle, the
contact force injects a sharp, nonlinear disturbance into both channels — that
is exactly what the GP captures. Two output channels share one set of inducing
inputs Z (efficient) with separate weight vectors alpha_v, alpha_omega.

FEATURE VECTOR
==============
    z = [theta, omega, v, tau]      (proprioceptive only — no obstacle position)
"""

from dataclasses import dataclass

import numpy as np
import casadi as ca


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GPConfig:
    max_points: int = 20
    sf2: float = 4.0
    sn2: float = 0.25
    novelty_thresh: float = 0.85
    activation_thresh: float = 0.6           # |residual| to count as contact
    # ARD lengthscales for z = [theta, omega, v, tau]
    lengthscales: tuple = (0.30, 2.0, 1.0, 3.0)
    refit_every: int = 1                     # recompute alpha every k observations


# ── Kernel (RBF / squared-exponential, ARD lengthscales) ─────────────────────

def _rbf_np(A: np.ndarray, B: np.ndarray,
            lengthscales: np.ndarray, sf2: float) -> np.ndarray:
    """
    RBF kernel matrix between rows of A (n x d) and B (m x d).
    Returns (n x m).
    """
    Al = A / lengthscales
    Bl = B / lengthscales
    # squared euclidean distances in scaled space
    sq = (np.sum(Al**2, axis=1)[:, None]
          + np.sum(Bl**2, axis=1)[None, :]
          - 2.0 * Al @ Bl.T)
    sq = np.maximum(sq, 0.0)
    return sf2 * np.exp(-0.5 * sq)


class GPResidual:
    """
    Online sparse GP residual model with two output channels (v_dot, omega_dot).

    Parameters
    ----------
    n_features : int
        Dimension of feature vector z (default 4: theta, omega, v, tau).
    max_points : int
        Dictionary size M (number of inducing points). Keep small (15-25)
        so the MPC stays real-time.
    lengthscales : array (n_features,)
        Per-dimension RBF lengthscales (ARD). Tune to feature scales.
    sf2 : float
        Signal variance (kernel output scale).
    sn2 : float
        Observation noise variance (also acts as ridge regularizer).
    novelty_thresh : float
        A new point is added only if its max kernel similarity to existing
        dictionary points is below this (i.e. it is "novel"). Range (0,1).
    activation_thresh : float
        Minimum |residual| for a sample to be considered a contact event
        worth learning. Filters out free-rolling noise.
    """

    def __init__(self,
                 n_features: int = 4,
                 max_points: int = 20,
                 lengthscales=None,
                 sf2: float = 4.0,
                 sn2: float = 0.25,
                 novelty_thresh: float = 0.85,
                 activation_thresh: float = 0.5):
        self.d   = n_features
        self.M   = max_points
        self.sf2 = float(sf2)
        self.sn2 = float(sn2)
        self.novelty_thresh    = float(novelty_thresh)
        self.activation_thresh = float(activation_thresh)

        if lengthscales is None:
            # defaults for [theta(rad), omega(rad/s), v(m/s), tau(N·m)]
            lengthscales = np.array([0.30, 2.0, 1.0, 3.0], dtype=float)
        self.l = np.asarray(lengthscales, dtype=float).reshape(-1)
        assert self.l.size == self.d

        # Dictionary storage (pre-allocated, fixed size for MPC param shape)
        self.Z       = np.zeros((self.M, self.d), dtype=float)
        self.y_v     = np.zeros(self.M, dtype=float)
        self.y_omega = np.zeros(self.M, dtype=float)
        self.active  = np.zeros(self.M, dtype=bool)   # which slots are filled
        self.age     = np.zeros(self.M, dtype=float)  # for eviction

        # Weights (recomputed when dictionary changes)
        self.alpha_v     = np.zeros(self.M, dtype=float)
        self.alpha_omega = np.zeros(self.M, dtype=float)

        self._dirty = False   # set True when dictionary changed, alpha stale
        self.n_seen = 0

    # ── Online ingestion ─────────────────────────────────────────────────────

    def observe(self, z: np.ndarray, r_v: float, r_omega: float) -> bool:
        """
        Consider a new (feature, residual) sample for the dictionary.

        Returns True if the sample was added/updated, False if rejected
        (not novel enough or below activation threshold).
        """
        z = np.asarray(z, dtype=float).reshape(-1)
        self.n_seen += 1
        self.age[self.active] += 1.0

        # Only learn meaningful (contact) residuals, not free-rolling noise
        if max(abs(r_v), abs(r_omega)) < self.activation_thresh:
            return False

        if not self.active.any():
            self._insert(0, z, r_v, r_omega)
            return True

        # Novelty test: similarity to existing dictionary points
        idx_active = np.where(self.active)[0]
        Zc  = self.Z[idx_active]
        sim = _rbf_np(z[None, :], Zc, self.l, self.sf2)[0] / self.sf2  # in [0,1]
        max_sim = float(np.max(sim))

        if max_sim >= self.novelty_thresh:
            # Not novel: refresh the nearest point's target (running average)
            j = idx_active[int(np.argmax(sim))]
            self.y_v[j]     = 0.5 * self.y_v[j]     + 0.5 * r_v
            self.y_omega[j] = 0.5 * self.y_omega[j] + 0.5 * r_omega
            self.age[j]     = 0.0
            self._dirty = True
            return True

        # Novel: add to a free slot, or evict the oldest if full
        free = np.where(~self.active)[0]
        if free.size > 0:
            self._insert(int(free[0]), z, r_v, r_omega)
        else:
            j = int(np.argmax(self.age))   # oldest
            self._insert(j, z, r_v, r_omega)
        return True

    def _insert(self, slot: int, z: np.ndarray,
                r_v: float, r_omega: float) -> None:
        self.Z[slot]       = z
        self.y_v[slot]     = r_v
        self.y_omega[slot] = r_omega
        self.active[slot]  = True
        self.age[slot]     = 0.0
        self._dirty = True

    # ── Weight computation (kernel ridge / sparse-GP mean) ───────────────────

    def refit(self) -> None:
        """
        Recompute alpha = (K_MM + sn2 I)^-1 y over the active dictionary.
        Inactive slots get alpha = 0 (so they contribute nothing in the MPC).
        Cheap: M is small (<=25), so this is a tiny linear solve.
        """
        self.alpha_v[:]     = 0.0
        self.alpha_omega[:] = 0.0
        if not self.active.any():
            self._dirty = False
            return

        idx = np.where(self.active)[0]
        Zc  = self.Z[idx]
        K   = _rbf_np(Zc, Zc, self.l, self.sf2)
        K  += self.sn2 * np.eye(len(idx))

        # Solve once, reuse factorization for both channels
        try:
            L = np.linalg.cholesky(K)
            av = np.linalg.solve(L.T, np.linalg.solve(L, self.y_v[idx]))
            ao = np.linalg.solve(L.T, np.linalg.solve(L, self.y_omega[idx]))
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(K)
            av = Kinv @ self.y_v[idx]
            ao = Kinv @ self.y_omega[idx]

        self.alpha_v[idx]     = av
        self.alpha_omega[idx] = ao
        self._dirty = False

    # ── Numpy prediction (for the supervisor / logging) ──────────────────────

    def predict(self, z: np.ndarray) -> tuple:
        """
        Returns (mean_v, mean_omega, std) at feature z.
        std is the predictive standard deviation (shared across channels;
        it reflects how far z is from the dictionary — high std = novel terrain).
        """
        if self._dirty:
            self.refit()
        z = np.asarray(z, dtype=float).reshape(1, -1)

        if not self.active.any():
            return 0.0, 0.0, float(np.sqrt(self.sf2))

        idx = np.where(self.active)[0]
        Zc  = self.Z[idx]
        kss = self.sf2
        ks  = _rbf_np(z, Zc, self.l, self.sf2)[0]    # (m,)

        mean_v     = float(ks @ self.alpha_v[idx])
        mean_omega = float(ks @ self.alpha_omega[idx])

        # predictive variance
        K = _rbf_np(Zc, Zc, self.l, self.sf2) + self.sn2 * np.eye(len(idx))
        try:
            vsol = np.linalg.solve(K, ks)
            var  = max(kss - float(ks @ vsol), 1e-9)
        except np.linalg.LinAlgError:
            var = kss
        return mean_v, mean_omega, float(np.sqrt(var))

    # ── Parameter export for the MPC ─────────────────────────────────────────

    def mpc_params(self) -> np.ndarray:
        """
        Flattened parameter vector handed to the CasADi solver each step:
            [ Z (M*d) , alpha_v (M) , alpha_omega (M) ]
        Inactive slots contribute alpha = 0, so their Z values are irrelevant
        (we still pass them; they are multiplied by zero weight).
        """
        if self._dirty:
            self.refit()
        return np.concatenate([
            self.Z.reshape(-1),
            self.alpha_v,
            self.alpha_omega,
        ])

    @property
    def n_active(self) -> int:
        return int(self.active.sum())


# ── CasADi symbolic GP mean (matches predict() exactly) ──────────────────────

def casadi_gp_mean(z_sym, Z_sym, alpha_sym,
                   lengthscales: np.ndarray, sf2: float):
    """
    Symbolic GP mean  mu(z) = sum_i alpha_i * sf2 * exp(-0.5 ||(z - Z_i)/l||^2)

    Parameters
    ----------
    z_sym     : ca.SX (d,)   feature point (symbolic state/control)
    Z_sym     : ca.SX (M, d) inducing locations (solver parameter)
    alpha_sym : ca.SX (M,)   weights (solver parameter)
    lengthscales : np.ndarray (d,)  fixed ARD lengthscales (compile-time)
    sf2 : float                     fixed signal variance (compile-time)

    Returns
    -------
    ca.SX scalar GP mean.
    """
    M = Z_sym.shape[0]
    l = ca.DM(lengthscales.reshape(-1))
    out = 0
    for i in range(M):
        diff = (z_sym - Z_sym[i, :].T) / l
        k_i  = sf2 * ca.exp(-0.5 * ca.dot(diff, diff))
        out += alpha_sym[i] * k_i
    return out
