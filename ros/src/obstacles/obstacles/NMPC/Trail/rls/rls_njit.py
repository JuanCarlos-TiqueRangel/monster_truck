#!/usr/bin/env python3

import numpy as np
from numba import njit

from rls import nominal_rls_parameters   # single source of truth (re-exported below)

__all__ = ["rls_update", "nominal_rls_parameters"]


@njit(cache=True, fastmath=True)
def _rls_core(v_prev, theta_prev, omega_prev, tau, a, P, y, lam, sv2, sw2):
    """Numeric RLS measurement update on the FULL-dynamics regressor. Pure arrays/scalars
    so it can be JIT-compiled. Plain weighted RLS: every measurement updates the weights
    (no outlier gate, no clipping). Returns (a_out, P_out, yhat0, yhat1) where yhat is the
    POSTERIOR prediction."""
    c = np.cos(theta_prev)

    # Block regressor H (2x10): a = [a_v(5), a_w(5)].
    H = np.zeros((2, 10))
    H[0, 0] = tau                 # drive torque -> forward traction force
    H[0, 1] = v_prev              # linear (viscous/rolling) drag
    H[0, 2] = abs(v_prev) * v_prev    # quadratic aero drag (signed)
    H[0, 3] = tau * (c - 1.0)     # traction roll-off as it rears (0 at flat)
    H[0, 4] = 1.0                 # bias
    H[1, 5] = c                   # gravity restoring torque (pendulum)
    H[1, 6] = tau                 # wheel reaction torque that pops the wheelie
    H[1, 7] = omega_prev          # pitch-rate damping
    H[1, 8] = v_prev              # weight-transfer/speed coupling
    H[1, 9] = 1.0                 # bias

    yhat_before = H @ a
    e0 = y[0] - yhat_before[0]
    e1 = y[1] - yhat_before[1]

    Pp = P / lam
    HP = H @ Pp                   # 2x10
    S = HP @ H.T                  # 2x2
    S00 = S[0, 0] + sv2           # R = diag(sv2, sw2) > 0 -> S always invertible
    S11 = S[1, 1] + sw2
    S01 = S[0, 1]
    S10 = S[1, 0]

    det = S00 * S11 - S01 * S10
    iS00 = S11 / det
    iS11 = S00 / det
    iS01 = -S01 / det
    iS10 = -S10 / det

    # --- gain K = Pp H' S^-1  (10x2) ---
    PHt = Pp @ H.T                # 10x2
    K = np.empty((10, 2))
    for i in range(10):
        K[i, 0] = PHt[i, 0] * iS00 + PHt[i, 1] * iS10
        K[i, 1] = PHt[i, 0] * iS01 + PHt[i, 1] * iS11

    a_out = a.copy()
    for i in range(10):
        a_out[i] = a[i] + K[i, 0] * e0 + K[i, 1] * e1

    # Joseph-form covariance update (numerically stable, stays symmetric PSD).
    IKH = np.eye(10) - K @ H
    R = np.zeros((2, 2))
    R[0, 0] = sv2
    R[1, 1] = sw2
    P_out = IKH @ Pp @ IKH.T + K @ R @ K.T
    P_out = 0.5 * (P_out + P_out.T)

    yhat_after = H @ a_out
    return a_out, P_out, yhat_after[0], yhat_after[1]


def rls_update(
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    a: np.ndarray,
    P: np.ndarray,
    forgetting_factor: float = 0.999,
    sigma_v_dot: float = 2.0,
    sigma_omega_dot: float = 5.0,
    y_dot_meas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """numba-accelerated drop-in for rls.rls_update (plain weighted RLS).

    RAW data in, RAW weights out: no derivative filtering, no outlier gating, no clipping.
    """
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    # ---- target [v_dot, omega_dot]: measured accel, or raw finite difference ----
    if y_dot_meas is not None:
        y = np.asarray(y_dot_meas, dtype=float).reshape(2).copy()
    else:
        y = np.array([(float(v_next) - float(v_prev)) / dt,
                      (float(omega_next) - float(omega_prev)) / dt], dtype=float)

    a_out, P_out, yhat0, yhat1 = _rls_core(
        float(v_prev), float(theta_prev), float(omega_prev), float(tau),
        np.asarray(a, dtype=float), np.asarray(P, dtype=float), y,
        float(forgetting_factor), sigma_v_dot ** 2, sigma_omega_dot ** 2,
    )

    info = {
        "v_dot_raw": float(y[0]),
        "omega_dot_raw": float(y[1]),
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(yhat0),
        "omega_dot_hat": float(yhat1),
        "v_dot_error": float(y[0] - yhat0),
        "omega_dot_error": float(y[1] - yhat1),
    }
    return a_out, P_out, info
