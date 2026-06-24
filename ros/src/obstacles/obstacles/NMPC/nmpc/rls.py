import numpy as np

from params_mujoco import WheelieParams

# ============================================================
# Full-dynamics two-output RLS
# ============================================================

# Model:
#   v_dot     = phi_v^T a_v
#   omega_dot = phi_w^T a_w
#
# Feature vectors:
#   phi_v = [tau, v, |v|v, tau*(cos(theta)-1), 1]
#   phi_w = [cos(theta), tau, omega, v, 1]
#
#
# ===> HARDWARE NOTE on the "-1.0" in tau*(cos(theta)-1) <===
# The -1.0 is NOT plant physics -- it is a pure RE-PARAMETERISATION (a basis shift) chosen
# so the term vanishes at theta=0. It transfers to real hardware UNCHANGED, with ONE
# proviso: it assumes theta=0 is the FLAT resting attitude. If on hardware your pitch
# sensor has an offset (theta = theta0 != 0 when flat, e.g. IMU mounting tilt), then either
# (a) calibrate the pitch zero so theta=0 at rest (preferred), or (b) replace the "1.0"
# with cos(theta0). Do NOT flip its sign for hardware -- the plant-dependent things to
# revisit instead are PITCH_SIGN / ACTUATOR_SIGN and the learned gains (m, r), not this.
#
# Stacked parameter vector:
#   a = [a_v(5), a_w(5)]
#
# Block regression:
#   y = [v_dot, omega_dot]
#   y = H a


def nominal_rls_parameters(p: WheelieParams) -> np.ndarray:
    # HARDCODED IDENTIFIED WEIGHTS (from rls_batch_id.py -- robust batch LS on qacc targets).
    # These replace the analytical model; with RLS_FREEZE=True the controller holds them
    # fixed and the GP learns the obstacle residual on top. Keep clip_parameters=False
    # (the clip box ratios around these signed values would be nonsensical).
    #   analytical fallbacks: b_tau=1/(m*r), b_v=-c_v, a_g=m*g*l/I_eff, a_tau=-1/I_eff.
    return np.array(
        [
            # v_dot = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0
            -1.0418,   # b_tau      (well identified)
            -0.0743,   # b_v
            -0.0018,   # b_abs_v    (~0, not significant)
            -0.3089,   # b_tau_cos
            -0.0250,   # b_0        (~0, not significant)

            # omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
            -0.4424,   # a_g        (omega channel poorly identified -- GP carries the pitch)
            -0.1916,   # a_tau
            -0.5000,   # a_omega    (batch gave +2.34 = DESTABILISING; forced negative for a stable rollout)
            -0.1851,   # a_v
             1.3827,   # a_0
        ],
        dtype=float,
    )


def rls_update(
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    a: np.ndarray,
    P: np.ndarray,
    filtered_y_dot: np.ndarray | None,
    forgetting_factor: float = 0.999,
    derivative_alpha: float = 0.85,
    sigma_v_dot: float = 2.0,
    sigma_omega_dot: float = 5.0,
    nis_gate: float = float("inf"),
    clip_parameters: bool = True,
    y_dot_meas: np.ndarray | None = None,
    p: WheelieParams | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    if y_dot_meas is not None:
        # MEASURED acceleration target (MuJoCo qacc, or an IMU): a clean, instantaneous
        # [v_dot, omega_dot] that REPLACES the noisy finite-difference of the velocities.
        # No numerical differentiation, and -- when sampled at the same instant as phi --
        # no half-step bias. Filtering is skipped (this IS the derivative already).
        y_raw = np.asarray(y_dot_meas, dtype=float).reshape(2)
        v_dot_raw, omega_dot_raw = float(y_raw[0]), float(y_raw[1])
        y = y_raw.copy()
        filtered_y_dot = y_raw.copy()
    else:
        # 1) Measured derivatives (legacy: finite difference of the velocities).
        v_dot_raw = (float(v_next) - float(v_prev)) / dt
        omega_dot_raw = (float(omega_next) - float(omega_prev)) / dt
        y_raw = np.array([v_dot_raw, omega_dot_raw], dtype=float)

        # 2) Filter derivatives.
        if filtered_y_dot is None:
            filtered_y_dot = y_raw.copy()
        else:
            filtered_y_dot = (
                derivative_alpha * filtered_y_dot
                + (1.0 - derivative_alpha) * y_raw
            )

        y = filtered_y_dot.copy()

    # 3) Different feature vectors.
    phi_v = np.array(
        [
            tau,                              # drive torque -> forward traction force
            v_prev,                           # linear (viscous/rolling) drag, ~ -c_v*v
            abs(v_prev) * v_prev,             # quadratic aero drag (signed), dominates at speed
            tau * (np.cos(theta_prev) - 1.0), # traction roll-off as it rears (0 at flat)
            1.0,                              # constant bias (absorbs un-modelled offset)
        ],
        dtype=float,
    )

    phi_w = np.array(
        [
            np.cos(theta_prev),               # gravity restoring torque (pendulum about pivot)
            tau,                              # wheel reaction torque that pops the wheelie
            omega_prev,                       # pitch-rate damping (aero + joint)
            v_prev,                           # weight-transfer/speed coupling into pitch
            1.0,                              # constant bias
        ],
        dtype=float,
    )

    # 4) Block regression matrix H. a = [a_v(5), a_w(5)] -> 10 params.
    n = a.shape[0]
    H = np.zeros((2, n), dtype=float)
    H[0, 0:5] = phi_v
    H[1, 5:n] = phi_w

    # 5) Prediction and error.
    y_hat_before = H @ a
    error = y - y_hat_before

    # 6) Weighted/Joseph-form RLS.
    R = np.diag([sigma_v_dot**2, sigma_omega_dot**2])
    P_pred = P / forgetting_factor
    S = H @ P_pred @ H.T + R

    if np.linalg.cond(S) > 1e12:
        info = {
            "v_dot_raw": float(v_dot_raw),
            "omega_dot_raw": float(omega_dot_raw),
            "v_dot_measured": float(y[0]),
            "omega_dot_measured": float(y[1]),
            "v_dot_hat": float(y_hat_before[0]),
            "omega_dot_hat": float(y_hat_before[1]),
            "v_dot_error": float(y[0] - y_hat_before[0]),
            "omega_dot_error": float(y[1] - y_hat_before[1]),
            "skipped": True,
        }
        return a, P, filtered_y_dot, info

    # Outlier gate (robust RLS): the normalized innovation squared is chi^2(2) under
    # the nominal model. A contact impulse or a flip makes it huge -- the parametric
    # model can't represent it -- so REJECT the parameter update, leaving a/P
    # unchanged. The measurement still defines a residual (returned below) for the GP
    # to learn; we only refuse to corrupt the smooth-dynamics parameters with it.
    nis = float(error @ np.linalg.solve(S, error))
    if nis > nis_gate:
        info = {
            "v_dot_raw": float(v_dot_raw),
            "omega_dot_raw": float(omega_dot_raw),
            "v_dot_measured": float(y[0]),
            "omega_dot_measured": float(y[1]),
            "v_dot_hat": float(y_hat_before[0]),
            "omega_dot_hat": float(y_hat_before[1]),
            "v_dot_error": float(y[0] - y_hat_before[0]),
            "omega_dot_error": float(y[1] - y_hat_before[1]),
            "skipped": True,
        }
        return a, P, filtered_y_dot, info

    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    I = np.eye(n)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    # 7) Optional projection/clipping.
    if clip_parameters and p is not None:
        a_nom = nominal_rls_parameters(p)

        # v_dot coefficients.
        a[0] = np.clip(a[0], 0.25 * a_nom[0], 2.50 * a_nom[0])
        a[1] = np.clip(a[1], 2.50 * a_nom[1], 0.25 * a_nom[1])
        a[2] = np.clip(a[2], -20.0, 20.0)
        a[3] = np.clip(a[3], -20.0, 20.0)
        a[4] = np.clip(a[4], -20.0, 20.0)

        # omega_dot coefficients.
        a[5] = np.clip(a[5], 0.25 * a_nom[5], 2.50 * a_nom[5])
        a[6] = np.clip(a[6], 3.00 * a_nom[6], 0.10 * a_nom[6])
        a[7] = np.clip(a[7], -30.0, 30.0)
        a[8] = np.clip(a[8], -30.0, 30.0)
        a[9] = np.clip(a[9], -80.0, 80.0)

    y_hat_after = H @ a

    info = {
        "v_dot_raw": float(v_dot_raw),
        "omega_dot_raw": float(omega_dot_raw),
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat_after[0]),
        "omega_dot_hat": float(y_hat_after[1]),
        "v_dot_error": float(y[0] - y_hat_after[0]),
        "omega_dot_error": float(y[1] - y_hat_after[1]),
        "skipped": False,
    }

    return a, P, filtered_y_dot, info