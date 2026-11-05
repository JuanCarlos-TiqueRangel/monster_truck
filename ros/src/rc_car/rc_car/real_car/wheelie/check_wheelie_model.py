"""Preflight rejection checks for the current 10-parameter wheelie RLS.

Assumptions: state=[x,v,theta,omega], radians, nose-up negative;
positive numerical u must generate more nose-up acceleration when lifted.
This is NOT a calibration, model repair, or closed-loop safety guarantee.
It must not be reused with a different feature ordering.
"""
import numpy as np


def check_wheelie_model(a, theta_ref, u_min, u_max):
    a = np.asarray(a, dtype=float)
    if a.shape != (10,) or not np.isfinite(a).all():
        raise ValueError('Expected 10 finite weights in the current feature order.')
    if not np.isfinite([theta_ref, u_min, u_max]).all() or u_min >= u_max:
        raise ValueError('Invalid reference or input bounds.')
    if not -np.pi / 2 <= theta_ref < 0:
        raise ValueError('This preflight expects a lifted reference in [-pi/2, 0).')

    # b(theta)=a4+a7*cos(theta). Endpoints suffice on [-pi/2,0].
    gain_flat = float(a[4] + a[7])
    gain_vertical = float(a[4])
    c = float(np.cos(theta_ref))
    gain_ref = float(a[4] + a[7] * c)
    drift_ref = float(a[6] * c + a[8] * theta_ref)
    accel_at_bounds = drift_ref + gain_ref * np.array([u_min, u_max])
    u_eq = float(-drift_ref / gain_ref) if abs(gain_ref) > 1e-12 else np.nan

    faults = []
    if max(gain_flat, gain_vertical) >= 0:
        faults.append('Input gain is not strictly negative over [-90 deg, 0 deg].')
    if a[6] < 0:
        faults.append('At rest and u=0, the ground-contact model lifts the nose by itself.')
    if not accel_at_bounds.min() < 0 < accel_at_bounds.max():
        faults.append('At the pitch target and omega=0, available inputs cannot give both acceleration signs.')

    report = {'gain_flat': gain_flat, 'gain_vertical': gain_vertical,
              'gain_at_reference': gain_ref, 'drift_at_reference': drift_ref,
              'angular_equilibrium_u': u_eq, 'angular_acceleration_at_limits': accel_at_bounds.tolist()}
    print(report)
    if faults:
        raise ValueError('RLS rejected:\n' + '\n'.join(faults))
    return report


if __name__ == '__main__':
    weights = np.array([3.644834440560566, 0.7106856738065334, 0.326904673774442, -2.3885774966594773,
                        5.09767980749295, 5.7419783992054665, -5.518781738280383, 6.099412134888565,
                        -16.217182226310346, 1.4283706399310123])
    check_wheelie_model(weights, np.deg2rad(-55), -0.35, 0.35)
