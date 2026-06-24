import numpy as np


# def nominal_rls_parameters(p=None) -> np.ndarray:

#     return np.array(
#         [
#             # v_dot = b_tau*tau + b_v*v + b_quad*|v|v + b_tau_theta*tau*cos(theta) + b_0
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,

#             # omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#             0.0,
#         ],
#         dtype=float,
#     )



def nominal_rls_parameters(p=None) -> np.ndarray:

    return np.array(
        [
            # v_dot = b_tau*tau + b_v*v + b_quad*|v|v + b_tau_theta*tau*cos(theta) + b_0
            0.5797325887864817,
            -0.052103921404089176, 
            -0.002319556678023478, 
            0.4888750173164126, 
            0.04229290897515063,

            # omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
            0.8252933830564172, 
            -0.5537942257289579, 
            1.4961818961972944, 
            0.19807363839226516, 
            -0.3025026481143033
        ],
        dtype=float,
    )


# def nominal_rls_parameters(p=None) -> np.ndarray:

#     return np.array(
#         [
#             # v_dot = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0
#             -1.0418,   # b_tau      (well identified)
#             -0.0743,   # b_v
#             -0.0018,   # b_abs_v    (~0, not significant)
#             -0.3089,   # b_tau_cos
#             -0.0250,   # b_0        (~0, not significant)

#             # omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
#             -0.4424,   # a_g        (omega channel poorly identified -- GP carries the pitch)
#             -0.1916,   # a_tau
#             -0.5000,   # a_omega    (batch gave +2.34 = DESTABILISING; forced negative for a stable rollout)
#             -0.1851,   # a_v
#              1.3827,   # a_0
#         ],
#         dtype=float,
#     )


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
    """Plain recursive-least-squares update for the two-output full-dynamics model.

    RAW data in, RAW weights out: standard exponentially-forgetting weighted RLS, with
    NO derivative low-pass filtering, NO outlier (NIS) gating, and NO parameter
    clipping/projection. Every measurement updates the weights.

    The acceleration target is either the MEASURED accel `y_dot_meas` (MuJoCo qacc / IMU,
    sampled at the same instant as phi) or, when that is None, the finite difference of
    the velocities.
    """
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    # 1) Acceleration target (raw -- no filtering).
    if y_dot_meas is not None:
        y = np.asarray(y_dot_meas, dtype=float).reshape(2).copy()
    else:
        y = np.array(
            [(float(v_next) - float(v_prev)) / dt,
             (float(omega_next) - float(omega_prev)) / dt],
            dtype=float,
        )

    # 2) Feature vectors.
    phi_v = np.array(
        [
            tau,                              # drive torque -> forward traction force
            v_prev,                           # linear (viscous/rolling) drag,  -c_v*v
            abs(v_prev) * v_prev,             # quadratic aero drag (signed), dominates at speed
            tau * (np.cos(theta_prev)), # traction roll-off as it rears (0 at flat)
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

    # 3) Block regression matrix H. a = [a_v(5), a_w(5)] -> 10 params.
    n = a.shape[0]
    H = np.zeros((2, n), dtype=float)
    H[0, 0:5] = phi_v
    H[1, 5:n] = phi_w

    # 4) Prediction and error.
    y_hat_before = H @ a
    error = y - y_hat_before

    # 5) Weighted/Joseph-form RLS update.
    R = np.diag([sigma_v_dot**2, sigma_omega_dot**2])
    P_pred = P / forgetting_factor
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    I = np.eye(n)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    y_hat_after = H @ a

    info = {
        "v_dot_raw": float(y[0]),
        "omega_dot_raw": float(y[1]),
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat_after[0]),
        "omega_dot_hat": float(y_hat_after[1]),
        "v_dot_error": float(y[0] - y_hat_after[0]),
        "omega_dot_error": float(y[1] - y_hat_after[1]),
        # Predictive 1-sigma per channel = sqrt(diag of the innovation covariance S).
        # Includes both the parameter uncertainty (H P H^T) and the measurement noise R.
        "v_dot_std": float(np.sqrt(S[0, 0])),
        "omega_dot_std": float(np.sqrt(S[1, 1])),
    }

    return a, P, info