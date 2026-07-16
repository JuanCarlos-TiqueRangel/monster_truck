import numpy as np
import torch


def nominal_rls_parameters(p) -> np.ndarray:
    """Return the RLS coefficients corresponding to the nominal plant model."""
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
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    a: np.ndarray,
    P: np.ndarray,
    forgetting_factor: float = 0.999,
    sigma_v_dot: float = 2.0,
    sigma_omega_dot: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Update the two-output recursive least-squares model."""
    _, v_prev, theta_prev, omega_prev = state_prev
    _, v_next, _, omega_next = state_next

    y = np.array(
        [
            (float(v_next) - float(v_prev)) / dt,
            (float(omega_next) - float(omega_prev)) / dt,
        ],
        dtype=float,
    )

    phi_v = np.array(
        [tau, v_prev, abs(v_prev) * v_prev, tau * np.cos(theta_prev), 1.0],
        dtype=float,
    )
    phi_w = np.array(
        [np.cos(theta_prev), tau, omega_prev, v_prev, 1.0],
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
        info = _rls_info(y, y_hat_before, skipped=True)
        return a, P, info

    K = P_pred @ H.T @ np.linalg.inv(S)
    a = a + K @ error

    identity = np.eye(10)
    P = (identity - K @ H) @ P_pred @ (identity - K @ H).T + K @ R @ K.T
    P = 0.5 * (P + P.T)

    return a, P, _rls_info(y, H @ a, skipped=False)


def _rls_info(y: np.ndarray, y_hat: np.ndarray, skipped: bool) -> dict:
    return {
        "v_dot_measured": float(y[0]),
        "omega_dot_measured": float(y[1]),
        "v_dot_hat": float(y_hat[0]),
        "omega_dot_hat": float(y_hat[1]),
        "v_dot_error": float(y[0] - y_hat[0]),
        "omega_dot_error": float(y[1] - y_hat[1]),
        "skipped": skipped,
    }


def rls_dynamics_torch(
    state: torch.Tensor, tau: torch.Tensor, a: torch.Tensor
) -> torch.Tensor:
    """Evaluate the RLS dynamics model for a batch of states and controls."""
    x = state[..., 0]
    v = state[..., 1]
    theta = state[..., 2]
    omega = state[..., 3]

    x_dot = v
    v_dot = (
        a[0] * tau
        + a[1] * v
        + a[2] * torch.abs(v) * v
        + a[3] * tau * torch.cos(theta)
        + a[4]
    )
    theta_dot = omega
    omega_dot = (
        a[5] * torch.cos(theta)
        + a[6] * tau
        + a[7] * omega
        + a[8] * v
        + a[9]
    )

    # Unilateral ground contact: at theta == 0 the floor prevents the
    # predicted chassis from rotating nose-down into positive pitch.
    on_ground = (theta >= 0.0) & (omega_dot > 0.0)
    omega_dot = torch.where(on_ground, torch.zeros_like(omega_dot), omega_dot)

    return torch.stack([x_dot, v_dot, theta_dot, omega_dot], dim=-1)


def rk4_step_torch(
    state: torch.Tensor, tau: torch.Tensor, dt: float, a: torch.Tensor
) -> torch.Tensor:
    """Advance the batched RLS model by one RK4 step."""
    k1 = rls_dynamics_torch(state, tau, a)
    k2 = rls_dynamics_torch(state + 0.5 * dt * k1, tau, a)
    k3 = rls_dynamics_torch(state + 0.5 * dt * k2, tau, a)
    k4 = rls_dynamics_torch(state + dt * k3, tau, a)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Project a predicted landing back onto the ground-contact manifold.
    landing = (next_state[..., 2] >= 0.0) & (next_state[..., 3] > 0.0)
    return torch.stack(
        [
            next_state[..., 0],
            next_state[..., 1].clamp(-10.0, 10.0),
            next_state[..., 2].clamp(max=0.0),
            torch.where(
                landing,
                torch.zeros_like(next_state[..., 3]),
                next_state[..., 3].clamp(-30.0, 30.0),
            ),
        ],
        dim=-1,
    )
