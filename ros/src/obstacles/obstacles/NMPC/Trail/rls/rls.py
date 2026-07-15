#!/usr/bin/env python3

import numpy as np


def nominal_rls_parameters(p):
    return np.array(
        [
            1.0 / (p.m * p.r),
            -p.c_v,
            0.0,
            0.0,
            0.0,
            p.m * p.g * p.l / p.I_eff,
            -1.0 / p.I_eff,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )


def rls_update(
    state_prev,
    tau,
    state_next,
    dt,
    a,
    P,
    forgetting_factor=0.999,
    sigma_v_dot=2.0,
    sigma_omega_dot=5.0,
    y_dot_meas=None,
):
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    if y_dot_meas is None:
        y = np.array(
            [
                (float(v_next) - float(v_prev)) / dt,
                (float(omega_next) - float(omega_prev)) / dt,
            ],
            dtype=float,
        )
    else:
        y = np.asarray(y_dot_meas, dtype=float).reshape(2)

    phi_v = np.array(
        [
            tau,
            v_prev,
            abs(v_prev) * v_prev,
            tau * np.cos(theta_prev),
            1.0,
        ],
        dtype=float,
    )

    phi_w = np.array(
        [
            np.cos(theta_prev),
            tau,
            omega_prev,
            v_prev,
            1.0,
        ],
        dtype=float,
    )

    H = np.zeros((2, 10), dtype=float)
    H[0, 0:5] = phi_v
    H[1, 5:10] = phi_w

    y_hat_before = H @ a
    error = y - y_hat_before

    R = np.diag([sigma_v_dot**2, sigma_omega_dot**2])
    P_pred = P / forgetting_factor
    S = H @ P_pred @ H.T + R

    if np.linalg.cond(S) > 1e12:
        info = {
            "v_dot_measured": float(y[0]),
            "omega_dot_measured": float(y[1]),
            "v_dot_hat": float(y_hat_before[0]),
            "omega_dot_hat": float(y_hat_before[1]),
            "v_dot_error": float(error[0]),
            "omega_dot_error": float(error[1]),
            "skipped": True,
        }
        return a, P, info

    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    I = np.eye(a.size)
    P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    y_hat_after = H @ a
    error_after = y - y_hat_after

    info = {
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat_after[0]),
        "omega_dot_hat": float(y_hat_after[1]),
        "v_dot_error": float(error_after[0]),
        "omega_dot_error": float(error_after[1]),
        "skipped": False,
    }
    return a, P, info