# import numpy as np
# import torch


# # def nominal_rls_parameters(p):

# #     a_v = np.array([
# #             5.55367477, 
# #             -0.74380849, 
# #             0.38885884, 
# #             -2.39405055
# #     ])

# #     a_w = np.array([
# #             -37.63521888, 
# #             0.0, 
# #             0.24423920, 
# #             -7.11714250, 
# #             -18.98413245, 
# #             4.44897735
# #     ])


# def nominal_rls_parameters(p):
#     a_v = np.zeros(13)
#     a_w = np.zeros(16)

#     return np.concatenate([a_v, a_w,]).astype(float)


# def rls_update(
#     state: np.ndarray,
#     tau: float,
#     vx_dot: float,
#     omega_pitch_dot: float,
#     a: np.ndarray,
#     P: np.ndarray,
#     forgetting_factor: float = 0.999,
#     sigma_v_dot: float = 2.0,
#     sigma_omega_dot: float = 5.0,
#     update_angular: bool = True,
#     features=None,
# ) -> tuple[np.ndarray, np.ndarray, dict]:

#     xpos, v, theta, omega = state

#     y = np.array([ vx_dot, omega_pitch_dot, ], dtype=float)

#     airborne = 1.0 if theta < -0.1 else 0.0
#     ground = 1.0 - airborne

#     omega_pos = max(omega, 0.0)
#     omega_neg = min(omega, 0.0)

#     phi_v = np.array([
#         # tau,
#         # v,
#         # abs(v) * v,
#         # tau * np.cos(theta),
#         # 1.0,

#         tau * ground,
#         v * ground,
#         abs(v) * v * ground,
#         omega_pos * ground,
#         omega_neg * ground,
#         abs(omega) * omega * ground,
#         np.sin(theta) * ground,

#         tau * airborne,
#         np.sin(theta) * airborne,
#         v * airborne,
#         abs(v) * v * airborne,
#         omega * airborne,
#         abs(omega) * omega * airborne,

#     ], dtype=float)


#     air_w = [
#         tau,
#         omega,
#         np.cos(theta),
#         tau * np.cos(theta),
#         theta,
#         omega * theta,
#         abs(v) * omega,
#     ]

#     ground_w = [
#         tau,
#         omega_pos,
#         omega_neg,
#         np.cos(theta),
#         tau * np.cos(theta),
#         theta,
#         omega * theta,
#         abs(v) * omega,
#         1.0,
#     ]

#     phi_w = np.array(
#         [x * airborne for x in air_w]
#         + [x * ground for x in ground_w],
#         dtype=float,
#     )

#     # phi_w = np.array([
#     #     # np.cos(theta),
#     #     # tau,
#     #     # omega,
#     #     # v,
#     #     # 1.0,

#     #     tau,
#     #     omega,
#     #     np.cos(theta),
#     #     tau * np.cos(theta),
#     #     theta,
#     #     omega * theta,
#     #     abs(v) * omega,        # <-- Captures gyroscopic/stabilization tracking effects from speed

#     # ], dtype=float)

#     if features is not None:
#         phi_v, phi_w = features

#     # Automatically determine dimensions
#     n_v = phi_v.size
#     n_w = phi_w.size
#     n_params = n_v + n_w

#     H_full = np.zeros((2, n_params), dtype=float)

#     H_full[0, :n_v] = phi_v
#     H_full[1, n_v:n_v + n_w] = phi_w

#     #print(f"H_full: {H_full.shape}, a: {a.shape}, P: {P.shape}")

#     y_hat_before = H_full @ a

#     # --------------------------------------------------------
#     # Ground contact
#     # --------------------------------------------------------

#     if update_angular:
#         H = H_full
#         y_update = y
#         R = np.diag([ sigma_v_dot**2, sigma_omega_dot**2])

#     else:
#         # Only identify longitudinal dynamics
#         H = H_full[0:1, :]
#         y_update = y[0:1]

#         R = np.array([ [sigma_v_dot**2] ], dtype=float)

#     # --------------------------------------------------------
#     # RLS
#     # --------------------------------------------------------

#     error = y_update - H @ a

#     P_pred = P / forgetting_factor
#     S = H @ P_pred @ H.T + R

#     if not np.all(np.isfinite(S)) or np.linalg.cond(S) > 1e12:
#         return a, P, _rls_info(
#             y,
#             y_hat_before,
#             skipped=True,
#         )

#     PHt = P_pred @ H.T
#     K = np.linalg.solve(S, PHt.T).T
#     a = a + K @ error
#     P = P_pred - K @ PHt.T
#     P = 0.5 * (P + P.T)

#     # K = P_pred @ H.T @ np.linalg.inv(S)

#     # a = a + K @ error

#     # # Physical projection for angular dynamics
#     # n_v = phi_v.size

#     # I = np.eye(n_params)
#     # P = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
#     # P = 0.5 * (P + P.T)

#     return a, P, _rls_info(
#         y,
#         y_hat_before,
#         skipped=False,
#     )



# def _rls_info(y: np.ndarray, y_hat: np.ndarray, skipped: bool) -> dict:
#     return {
#         "v_dot_measured": float(y[0]),
#         "omega_dot_measured": float(y[1]),
#         "v_dot_hat": float(y_hat[0]),
#         "omega_dot_hat": float(y_hat[1]),
#         "v_dot_error": float(y[0] - y_hat[0]),
#         "omega_dot_error": float(y[1] - y_hat[1]),
#         "skipped": skipped,
#     }


# # def rls_dynamics_torch( state: torch.Tensor, tau: torch.Tensor, a: torch.Tensor ) -> torch.Tensor:
# #     """Evaluate the RLS dynamics model for a batch of states and controls."""
# #     x = state[..., 0]
# #     v = state[..., 1]
# #     theta = state[..., 2]
# #     omega = state[..., 3]

# #     x_dot = v
# #     v_dot = (
# #         a[0] * tau
# #         + a[1] * v
# #         + a[2] * torch.abs(v) * v
# #         + a[3] * tau * torch.cos(theta)
# #         + a[4]

# #         # + a[0] * tau
# #         # + a[1] * v
# #         # + a[2] * torch.abs(omega) * omega
# #         # + a[3] * omega * torch.cos(theta)

# #     )
# #     theta_dot = omega
# #     omega_dot = (
# #         a[5] * torch.cos(theta)
# #         + a[6] * tau
# #         + a[7] * omega
# #         + a[8] * v
# #         + a[9]

# #         # a[4] * tau
# #         # + a[5] * omega
# #         # + a[6] * torch.cos(theta)
# #         # + a[7] * tau * torch.cos(theta)
# #         # + a[8] * theta
# #         # + a[9] * omega * theta
# #     )

# #     # Unilateral ground contact: at theta == 0 the floor prevents the
# #     # predicted chassis from rotating nose-down into positive pitch.
# #     on_ground = (theta >= 0.0) & (omega_dot > 0.0)
# #     omega_dot = torch.where(on_ground, torch.zeros_like(omega_dot), omega_dot)

# #     return torch.stack([x_dot, v_dot, theta_dot, omega_dot], dim=-1)


# # def rk4_step_torch( state: torch.Tensor, tau: torch.Tensor, dt: float, a: torch.Tensor ) -> torch.Tensor:

# #     """Advance the batched RLS model by one RK4 step."""
# #     k1 = rls_dynamics_torch(state, tau, a)
# #     k2 = rls_dynamics_torch(state + 0.5 * dt * k1, tau, a)
# #     k3 = rls_dynamics_torch(state + 0.5 * dt * k2, tau, a)
# #     k4 = rls_dynamics_torch(state + dt * k3, tau, a)
# #     next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

# #     # Project a predicted landing back onto the ground-contact manifold.
# #     landing = (next_state[..., 2] >= 0.0) & (next_state[..., 3] > 0.0)
# #     return torch.stack(
# #         [
# #             next_state[..., 0],
# #             next_state[..., 1].clamp(-10.0, 10.0),
# #             next_state[..., 2].clamp(max=0.0),
# #             torch.where(
# #                 landing,
# #                 torch.zeros_like(next_state[..., 3]),
# #                 next_state[..., 3].clamp(-30.0, 30.0),
# #             ),
# #         ],
# #         dim=-1,
# #     )




# def history_features(z, t):
#     z, t = np.asarray(z, float), np.asarray(t, float)
#     idx = np.maximum(np.arange(len(z))[:, None] - np.arange(5), 0)
#     h = z[idx]
#     xw = h.reshape(len(z), 20)
#     dt = np.maximum(t[:, None] - t[idx[:, 1:]], 0.04)
#     rates = (h[:, :1, 1:] - h[:, 1:, 1:]) / dt[..., None]
#     return np.column_stack((xw, rates.reshape(len(z), 12))), xw




# def history_predict(models, z, t, latest=False):
#     xv, xw = history_features(z, t)
#     if latest:
#         xv, xw = xv[-1:], xw[-1:]
#     return np.column_stack((models[0].predict(xv)[:, 0],
#                             models[1].predict(xw)[:, 1]))
































import numpy as np
import torch

def nominal_rls_parameters(p=None):
    """Initializes a flexible, oversized parameter placeholder vector."""
    return np.zeros(18, dtype=float)

def rls_update(
    state: np.ndarray,
    tau: float,
    vx_dot: float,
    omega_pitch_dot: float,
    a: np.ndarray,
    P: np.ndarray,
    forgetting_factor: float = 0.999,  # Lower memory depth for instant tracking with zero lag
    sigma_v_dot: float = 5.0,         # Low values force immediate adjustment
    sigma_omega_dot: float = 5.0,     # Low values force immediate adjustment
    update_angular: bool = True,
    features=None,
) -> tuple[np.ndarray, np.ndarray, dict]:

    xpos, v, theta, omega = state

    # =========================================================================
    # EDIT ONLY THESE TWO LINES TO ALTER THE FEATURE VECTOR ATTRIBUTES
    # =========================================================================
    # phi_v = np.array([tau, v, abs(v)*v, -np.abs(np.sin(theta)), tau*v, 1.0], dtype=float)
    # phi_w = np.array([tau, omega, theta, np.cos(theta), 1.0], dtype=float)
    
    phi_v = np.array([
            tau, 
            v, 
            abs(v) * v, 
            np.sin(theta), 
            (omega**2) * np.sin(theta),  # Ensures the linear model catches the sharpest braking
            tau * v, 
            1.0
        ], dtype=float)


    phi_w = np.array([
            tau, 
            omega, 
            abs(omega) * omega,    
            np.sin(theta),         
            np.cos(theta), 
            np.sin(theta) * np.cos(theta), 
            tau * np.cos(theta),   
            (omega**2) * np.sin(theta) * np.cos(theta), # Centrifugal torque coupling
            v * np.cos(theta),                          # Velocity torque coupling
            abs(v) * omega,        
            1.0
    ], dtype=float)

    # =========================================================================

    # 1. Dynamically compute feature lengths at runtime
    n_v = phi_v.size
    n_w = phi_w.size

    # 2. Extract adaptive parameter tracks dynamically based on sizing
    a_v = a[0:n_v]
    a_w = a[n_v:n_v + n_w]
    
    # 3. Extract adaptive covariance sub-matrices dynamically
    P_v = P[0:n_v, 0:n_v]
    P_w = P[n_v:n_v + n_w, n_v:n_v + n_w]

    # Calculate individual baseline estimates
    v_dot_hat = float(phi_v @ a_v)
    omega_dot_hat = float(phi_w @ a_w)

    # -------------------------------------------------------------------------
    # FILTER 1: Isolated Adaptive RLS Channel for Linear Acceleration (v_dot)
    # -------------------------------------------------------------------------
    error_v = vx_dot - v_dot_hat
    P_v_pred = P_v / forgetting_factor
    S_v = float(phi_v @ P_v_pred @ phi_v.T) + sigma_v_dot**2
    
    if np.isfinite(S_v) and S_v > 1e-6:
        K_v = (P_v_pred @ phi_v.T) / S_v
        a_v = a_v + K_v * error_v
        P_v = P_v_pred - np.outer(K_v, phi_v) @ P_v_pred
        P_v = 0.5 * (P_v + P_v.T)

    # -------------------------------------------------------------------------
    # FILTER 2: Isolated Adaptive RLS Channel for Angular Acceleration (omega_dot)
    # -------------------------------------------------------------------------
    if update_angular:
        error_w = omega_pitch_dot - omega_dot_hat
        P_w_pred = P_w / forgetting_factor
        S_w = float(phi_w @ P_w_pred @ phi_w.T) + sigma_omega_dot**2
        
        if np.isfinite(S_w) and S_w > 1e-6:
            K_w = (P_w_pred @ phi_w.T) / S_w
            a_w = a_w + K_w * error_w
            P_w = P_w_pred - np.outer(K_w, phi_w) @ P_w_pred
            P_w = 0.5 * (P_w + P_w.T)

# -------------------------------------------------------------------------
    # 4. Safely reassemble updating tracks back into global static arrays
    # -------------------------------------------------------------------------
    a_new = a.copy()
    a_new[0:n_v] = a_v
    a_new[n_v:n_v + n_w] = a_w
    
    P_new = P.copy()
    P_new[0:n_v, 0:n_v] = P_v
    P_new[n_v:n_v + n_w, n_v:n_v + n_w] = P_w

    # Calculate predictions using the newly updated weights (A Posteriori)
    v_dot_hat_post = float(phi_v @ a_new[0:n_v])
    omega_dot_hat_post = float(phi_w @ a_new[n_v:n_v+n_w])

    return a_new, P_new, {
        "v_dot_measured": float(vx_dot),
        "omega_dot_measured": float(omega_pitch_dot),
        "v_dot_hat_post": v_dot_hat_post,            # Updated 
        "omega_dot_hat__post": omega_dot_hat_post,    # Updated
        "v_dot_hat": v_dot_hat_post,            # Updated 
        "omega_dot_hat": omega_dot_hat_post,    # Updated
        "v_dot_error": float(vx_dot - v_dot_hat), # Keep a priori error for true residual
        "omega_dot_error": float(omega_pitch_dot - omega_dot_hat),
        "n_v": n_v,
        "n_w": n_w,
        "skipped": False,
    }








def rls_dynamics_torch( state: torch.Tensor, tau: torch.Tensor, a: torch.Tensor ) -> torch.Tensor:
    """Evaluate the RLS dynamics model for a batch of states and controls."""
    x = state[..., 0]
    v = state[..., 1]
    theta = state[..., 2]
    omega = state[..., 3]

    x_dot = v
    v_dot = (
        a[0] * tau
        + a[1] * v 
        + a[2] * abs(v) * v
        + a[3] * torch.sin(theta)
        + a[4] * (omega**2) * torch.sin(theta)  # Ensures the linear model catches the sharpest braking
        + a[5] * tau * v
        + a[6] * 1.0
        )

    theta_dot = omega
    omega_dot = (
       a[7] * tau
       + a[8] * omega
       + a[9] * abs(omega) * omega
       + a[10] * torch.sin(theta)      
       + a[11] * torch.cos(theta) 
       + a[12] * torch.sin(theta) * torch.cos(theta)
       + a[13] * tau * torch.cos(theta)
       + a[14] * (omega**2) * torch.sin(theta) * torch.cos(theta) # Centrifugal torque coupling
       + a[15] * v * torch.cos(theta)                          # Velocity torque coupling
       + a[16] * abs(v) * omega        
       + a[17] * 1.0
    )

    # Unilateral ground contact: at theta == 0 the floor prevents the
    # predicted chassis from rotating nose-down into positive pitch.
    on_ground = (theta >= 0.0) & (omega_dot > 0.0)
    omega_dot = torch.where(on_ground, torch.zeros_like(omega_dot), omega_dot)

    return torch.stack([x_dot, v_dot, theta_dot, omega_dot], dim=-1)