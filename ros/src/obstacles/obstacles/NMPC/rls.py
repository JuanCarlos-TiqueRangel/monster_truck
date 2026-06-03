#!/usr/bin/env python3
"""
rls.py
------
Recursive Least Squares (RLS) online identification of the truck's
acceleration dynamics, with exponential forgetting.

WHY
===
The NMPC needs an accurate model of how torque maps to linear acceleration
(v_dot) and angular acceleration (omega_dot). Those coefficients drift with
battery level, tyre wear, payload, terrain, etc. RLS estimates them online
from measured data, **all the time** (there is no freeze / contact gate).

The model is affine-in-parameters for each channel:

    v_dot      ~=  b . phi_v(theta, v, tau)        (linear  channel)
    omega_dot  ~=  a . phi_w(theta, omega, v, tau) (angular channel)

The GP residual model (see gp_residual.py) then captures only the sharp,
nonlinear part that this smooth linear model cannot explain (contact spikes).

SINGLE SOURCE OF TRUTH
======================
The regressor (feature) functions below are used in TWO places:
  1. here, numerically (np.cos / np.sin), to update the estimates, and
  2. in nmpc.py, symbolically (ca.cos / ca.sin), inside the predictor.
Keeping one definition guarantees the estimator and the MPC never disagree.
"""

from dataclasses import dataclass

import numpy as np


# ── Regressor feature vectors (shared numeric / symbolic) ────────────────────
#
# Each returns a plain Python list. Callers assemble it as they need:
#   numpy   :  np.array(phi)
#   casadi  :  ca.vertcat(*phi)
# Pass cos=np.cos / sin=np.sin for numeric use, or cos=ca.cos / sin=ca.sin
# for symbolic use.

def omega_regressor(theta, omega, v, tau, *, cos):
    """Features for the angular-acceleration model.

        omega_dot ~= a . [cos(theta), tau, omega, v, 1]

    Physical reading of the seed parameters:
        a[0] ~ m g l / I_eff  (gravity torque about the contact)
        a[1] ~ -1 / I_eff     (drive-torque reaction)
        a[2], a[3]            (pitch / longitudinal coupling, learned)
        a[4]                  (constant offset, learned)
    """
    return [cos(theta), tau, omega, v, 1.0]


def v_regressor(theta, v, tau, *, sin):
    """Features for the linear-acceleration model.

        v_dot ~= b . [tau, v, sin(theta), 1]

    Physical reading of the seed parameters:
        b[0] ~ 1 / (m r)      (drive force -> acceleration)
        b[1] ~ -c_v           (linear drag)
        b[2]                  (gravity projection / pitch coupling, learned)
        b[3]                  (constant offset, learned)
    """
    return [tau, v, sin(theta), 1.0]


N_OMEGA_FEATURES = 5
N_V_FEATURES = 4


@dataclass
class RLSConfig:
    """Tuning shared by both RLS channels."""
    forgetting: float = 0.999   # lambda in (0, 1]; lower = faster, noisier
    p0_scale: float = 3.0       # initial covariance P0 = p0_scale * I


class RLS:
    """
    Recursive least squares with exponential forgetting for one scalar output

        y ~= phi . theta

    The estimate `theta` and covariance `P` are updated every step. This
    implementation never freezes: it always adapts toward the latest data.
    """

    def __init__(self, theta0, *, forgetting: float = 0.999,
                 p0_scale: float = 3.0, P0=None):
        self.theta = np.asarray(theta0, dtype=float).copy()
        n = self.theta.size
        self.P = (np.eye(n) * float(p0_scale) if P0 is None
                  else np.asarray(P0, dtype=float).copy())
        self.lam = float(forgetting)

    def predict(self, phi) -> float:
        """Model output for feature vector `phi` at the current estimate."""
        return float(np.asarray(phi, dtype=float) @ self.theta)

    def update(self, phi, y: float) -> dict:
        """
        One RLS step against target `y` with features `phi`.

        Returns a small info dict (prediction before/after, error) for logging.
        """
        phi = np.asarray(phi, dtype=float)
        y_hat_before = float(phi @ self.theta)
        error = float(y) - y_hat_before

        Pphi = self.P @ phi
        denom = self.lam + float(phi @ Pphi)
        if abs(denom) < 1e-12:
            # Degenerate (should not happen with p0_scale > 0); skip safely.
            return {"y": float(y), "y_hat": y_hat_before,
                    "error": error, "updated": False}

        K = Pphi / denom
        self.theta = self.theta + K * error
        # P <- (I - K phi^T) P / lam, written without forming I - K phi^T:
        #   K phi^T P = K (P phi)^T = outer(K, Pphi)   (P symmetric)
        self.P = (self.P - np.outer(K, Pphi)) / self.lam
        self.P = 0.5 * (self.P + self.P.T)             # keep symmetric

        y_hat_after = float(phi @ self.theta)
        return {"y": float(y), "y_hat": y_hat_after,
                "error": float(y) - y_hat_after, "updated": True}
