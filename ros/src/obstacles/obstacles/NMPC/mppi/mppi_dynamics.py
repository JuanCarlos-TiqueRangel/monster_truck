#!/usr/bin/env python3
"""
mppi_dynamics.py
----------------
Batched (GPU) rollout model for MPPI. This is the SAME gray-box model the
CasADi NMPC uses, so the two controllers can be compared fairly:

    x_dot     = v
    v_dot     = b . [tau, v, sin(theta), 1]          + gp_v(z)
    theta_dot = omega
    omega_dot = a . [cos(theta), tau, omega, v, 1]   + gp_omega(z)
    z = [theta, omega, v, tau]

  * a (5,), b (4,)  : RLS-estimated coefficients (online, from rls.py)
  * gp_v, gp_omega  : sparse-GP residual means (from gp_residual.py), evaluated
                      here as a batched RBF sum over the M inducing points.

Everything is vectorised over K rollouts with PyTorch, so a whole MPPI step is
one big GPU kernel. This is deployable on the car: it needs NO simulator, just
the learned coefficients + GP dictionary (a few hundred floats).

The regressor ORDER here must match rls.py exactly (single source of truth is
rls.py: omega_regressor = [cos(theta), tau, omega, v, 1],
        v_regressor     = [tau, v, sin(theta), 1]).
"""

import torch

# Physical clamps on the GP residual contribution (identical to nmpc.py), so a
# mis-learned/extrapolating GP cannot inject unphysical accelerations.
GP_V_CLAMP = 1.5 * 9.81      # max |longitudinal residual accel|  [m/s^2]
GP_OMEGA_CLAMP = 8.0         # max |pitch residual accel|         [rad/s^2]


def _soft_clip(val, lim):
    """Smooth, differentiable clamp to [-lim, lim] (matches nmpc.py)."""
    return lim * torch.tanh(val / lim)


def gp_mean_batch(z, Z, alpha_v, alpha_w, inv_l2, sf2):
    """
    Batched sparse-GP mean for both channels.

    z        : (K, 4) query features [theta, omega, v, tau]
    Z        : (M, 4) inducing points
    alpha_v  : (M,)   weights for the v_dot residual
    alpha_w  : (M,)   weights for the omega_dot residual
    inv_l2   : (4,)   1 / lengthscale^2  (ARD)
    sf2      : float  signal variance

    Returns (gp_v, gp_w), each (K,). Inactive dictionary slots carry alpha=0,
    so they contribute nothing (their Z value is irrelevant).
    """
    diff = z[:, None, :] - Z[None, :, :]                 # (K, M, 4)
    d2 = (diff * diff * inv_l2[None, None, :]).sum(-1)   # (K, M)
    k = sf2 * torch.exp(-0.5 * d2)                       # (K, M)
    gp_v = _soft_clip(k @ alpha_v, GP_V_CLAMP)
    gp_w = _soft_clip(k @ alpha_w, GP_OMEGA_CLAMP)
    return gp_v, gp_w


def f_batch(state, tau, a, b, Z, alpha_v, alpha_w, inv_l2, sf2):
    """
    Continuous-time dynamics x_dot = f(x, u), batched over K.

    state : (K, 4) = [x, v, theta, omega]
    tau   : (K,)   torque
    a     : (5,)   angular RLS coefficients
    b     : (4,)   linear  RLS coefficients
    Returns x_dot : (K, 4).
    """
    v = state[:, 1]
    theta = state[:, 2]
    omega = state[:, 3]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    z = torch.stack([theta, omega, v, tau], dim=-1)      # (K, 4)
    gp_v, gp_w = gp_mean_batch(z, Z, alpha_v, alpha_w, inv_l2, sf2)

    # v_dot = b . [tau, v, sin(theta), 1] + gp_v
    v_dot = b[0] * tau + b[1] * v + b[2] * sin_t + b[3] + gp_v
    # omega_dot = a . [cos(theta), tau, omega, v, 1] + gp_w
    omega_dot = a[0] * cos_t + a[1] * tau + a[2] * omega + a[3] * v + a[4] + gp_w

    return torch.stack([v, v_dot, omega, omega_dot], dim=-1)


def rk4_batch(state, tau, dt, a, b, Z, alpha_v, alpha_w, inv_l2, sf2):
    """One RK4 integration step, batched over K."""
    args = (a, b, Z, alpha_v, alpha_w, inv_l2, sf2)
    k1 = f_batch(state, tau, *args)
    k2 = f_batch(state + 0.5 * dt * k1, tau, *args)
    k3 = f_batch(state + 0.5 * dt * k2, tau, *args)
    k4 = f_batch(state + dt * k3, tau, *args)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
