#!/usr/bin/env python3
"""Whole-dynamics RLS basis. u is normalized /drive, NOT torque in N m.

The v2 model predicts interval-average velocity changes from causal inputs.
Use rls_step_torch, not RK4, to advance this discrete identified model.
"""
import numpy as np
import torch

V_FEATURES = ('u', 'v', 'abs(v)*v', 'sin(theta)', 'omega**2*sin(theta)', 'u*v', '1')
W_FEATURES = ('u', 'omega', 'abs(omega)*omega', 'sin(theta)', 'cos(theta)', 'sin(theta)*cos(theta)', 'u*cos(theta)', 'omega**2*sin(theta)*cos(theta)', 'v*cos(theta)', 'abs(v)*omega', '1')
MODEL_TYPE = 'whole_dynamics_rls_forward_v2'


def nominal_rls_parameters(p=None):
    return np.zeros(len(V_FEATURES) + len(W_FEATURES), dtype=float)


def rls_features_numpy(state, u):
    state = np.asarray(state, dtype=float)
    v, theta, omega = state[..., 1], state[..., 2], state[..., 3]
    u = np.broadcast_to(np.asarray(u, dtype=float), v.shape)
    sn, cs, one = np.sin(theta), np.cos(theta), np.ones_like(v)
    fv = np.stack((u, v, abs(v)*v, sn, omega**2*sn, u*v, one), axis=-1)
    fw = np.stack((u, omega, abs(omega)*omega, sn, cs, sn*cs, u*cs, omega**2*sn*cs, v*cs, abs(v)*omega, one), axis=-1)
    return fv, fw


def rls_update(state, tau, vx_dot, omega_pitch_dot, a, P, forgetting_factor=1.0, sigma_v_dot=1.0, sigma_omega_dot=1.0, update_angular=True, features=None):
    """RLS update; predictions named *_hat are PRIOR to seeing this target.

    features may supply globally scaled regressors during offline fitting.
    sigma values are output weighting scales, not estimated sensor accuracy.
    """
    a, P = np.asarray(a, float), np.asarray(P, float)
    if a.shape != (18,) or P.shape != (18, 18):
        raise ValueError('Expected a.shape=(18,), P.shape=(18,18).')
    if not 0.0 < forgetting_factor <= 1.0 or min(sigma_v_dot, sigma_omega_dot) <= 0.0:
        raise ValueError('Forgetting factor must be in (0,1]; weighting scales must be positive.')
    fv, fw = rls_features_numpy(state, tau) if features is None else features
    fv, fw = np.asarray(fv, float), np.asarray(fw, float)
    if fv.shape != (7,) or fw.shape != (11,):
        raise ValueError('Expected 7 linear and 11 angular features.')
    y = np.array((vx_dot, omega_pitch_dot), dtype=float)
    prior = np.array((fv @ a[:7], fw @ a[7:]))
    if not all(np.isfinite(z).all() for z in (a, P, fv, fw, y)):
        return a.copy(), P.copy(), {'skipped': True, 'v_dot_hat': float('nan'), 'omega_dot_hat': float('nan')}
    new_a, new_P = a.copy(), P.copy()
    for phi, target, sl, scale, enabled in ((fv, y[0], slice(0, 7), sigma_v_dot, True), (fw, y[1], slice(7, 18), sigma_omega_dot, update_angular)):
        if not enabled:
            continue
        covariance = P[sl, sl] / forgetting_factor
        p_phi = covariance @ phi
        denominator = scale**2 + phi @ p_phi
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise FloatingPointError('RLS covariance lost positive definiteness.')
        new_a[sl] += p_phi * (target - phi @ a[sl]) / denominator
        covariance -= np.outer(p_phi, p_phi) / denominator
        new_P[sl, sl] = 0.5 * (covariance + covariance.T)
    posterior = np.array((fv @ new_a[:7], fw @ new_a[7:]))
    return new_a, new_P, {'skipped': False, 'v_dot_hat': float(prior[0]), 'omega_dot_hat': float(prior[1]), 'v_dot_hat_post': float(posterior[0]), 'omega_dot_hat_post': float(posterior[1]), 'v_dot_error': float(y[0]-prior[0]), 'omega_dot_error': float(y[1]-prior[1]), 'n_v': 7, 'n_w': 11}


def rls_dynamics_torch(state, tau, a):
    """Batched [v, predicted delta-v/dt, omega, predicted delta-omega/dt]."""
    v, theta, omega = state[..., 1], state[..., 2], state[..., 3]
    tau = torch.as_tensor(tau, dtype=state.dtype, device=state.device)
    if tau.ndim == state.ndim and tau.shape[-1] == 1:
        tau = tau.squeeze(-1)
    sn, cs = torch.sin(theta), torch.cos(theta)
    dv = a[0]*tau + a[1]*v + a[2]*v.abs()*v + a[3]*sn + a[4]*omega.square()*sn + a[5]*tau*v + a[6]
    dw = a[7]*tau + a[8]*omega + a[9]*omega.abs()*omega + a[10]*sn + a[11]*cs + a[12]*sn*cs + a[13]*tau*cs + a[14]*omega.square()*sn*cs + a[15]*v*cs + a[16]*v.abs()*omega + a[17]
    return torch.stack((v, dv, omega, dw), dim=-1)


def rls_step_torch(state, u, dt, a):
    """One identified forward step, with trapezoidal position/pitch kinematics.

    Ground contact is the only state projection. Divergent velocities/rates
    are NOT clipped or converted to valid states; MPPI rejects those rollouts.
    """
    derivative = rls_dynamics_torch(state, u, a)
    dw = derivative[..., 3]
    dw = torch.where((state[..., 2] >= 0.0) & (state[..., 3] >= 0.0) & (dw > 0.0), 0.0, dw)
    next_v = state[..., 1] + dt * derivative[..., 1]
    next_w = state[..., 3] + dt * dw
    next_x = state[..., 0] + 0.5 * dt * (state[..., 1] + next_v)
    next_theta = state[..., 2] + 0.5 * dt * (state[..., 3] + next_w)
    landing = (next_theta >= 0.0) & (next_w > 0.0)
    next_w = torch.where(landing, 0.0, next_w)
    return torch.stack((next_x, next_v, next_theta.clamp(max=0.0), next_w), dim=-1)
