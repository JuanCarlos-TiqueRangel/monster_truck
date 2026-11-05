"""Check the wheelie plant. Run beside car_dynamics.py with NumPy installed."""

import numpy as np

from car_dynamics import WheelieParams, contact_forces_np, continuous_dynamics_np, rk4_step_np


def energy(state, p):
    """Calculate kinetic and gravitational potential energy."""
    _, v, theta, omega = state
    com_z = p.x_com * np.sin(theta) + p.h_com * np.cos(theta)

    translation_energy = 0.5 * p.m * v**2
    coupling_energy = -p.m * com_z * v * omega
    rotation_energy = 0.5 * p.I_eff * omega**2
    kinetic_energy = translation_energy + coupling_energy + rotation_energy
    potential_energy = p.m * p.g * (p.r + com_z)

    return kinetic_energy + potential_energy


def main():
    p = WheelieParams()
    zero_state = np.zeros(4)

    # 1. The car must remain at rest when no torque is applied.
    derivative_at_rest = continuous_dynamics_np(zero_state, 0.0, p)
    state_after_one_second = rk4_step_np(zero_state, 0.0, 1.0, p)
    np.testing.assert_allclose(derivative_at_rest, zero_state, atol=1e-12)
    np.testing.assert_allclose(state_after_one_second, zero_state, atol=1e-12)

    # 2. The front and rear loads must match the scale measurements.
    front_normal, rear_normal = contact_forces_np(zero_state, 0.0, p)
    front_mass = front_normal / p.g
    rear_mass = rear_normal / p.g
    np.testing.assert_allclose([front_mass, rear_mass], [0.480, 0.803], atol=1e-10)

    # 3. Pitch acceleration must change at the predicted lift-off threshold.
    tau_lift = p.m * p.r * p.g * p.x_com / (p.h_com + p.r)
    below_lift = continuous_dynamics_np(zero_state, 0.99 * tau_lift, p)
    above_lift = continuous_dynamics_np(zero_state, 1.01 * tau_lift, p)
    at_test_torque = continuous_dynamics_np(zero_state, 0.30, p)
    below_lift_state = rk4_step_np(zero_state, 0.15, 0.2, p)

    assert below_lift[3] == 0.0
    assert above_lift[3] > 0.0
    assert at_test_torque[3] > 0.0
    assert below_lift_state[2] == 0.0

    # 4. With damping removed, stationary balance must be unstable.
    ideal = WheelieParams(c_v=0.0, c_pitch=0.0)
    balance_pitch = np.deg2rad(ideal.balance_pitch_deg)
    balance_state = np.array([0.0, 0.0, balance_pitch, 0.0])
    pitch_perturbation = np.array([0.0, 0.0, 0.01, 0.0])

    balance_derivative = continuous_dynamics_np(balance_state, 0.0, ideal)
    above_balance = continuous_dynamics_np(balance_state + pitch_perturbation, 0.0, ideal)
    below_balance = continuous_dynamics_np(balance_state - pitch_perturbation, 0.0, ideal)

    np.testing.assert_allclose(balance_derivative, zero_state, atol=1e-10)
    assert above_balance[3] > 0.0
    assert below_balance[3] < 0.0

    # 5. A landing must conserve horizontal momentum and dissipate energy.
    impact_state = np.array([0.0, 0.2, 0.0, -1.0])
    unchanged_state = rk4_step_np(impact_state, 0.0, 0.0, ideal)
    landed_state = rk4_step_np(impact_state, 0.0, 0.01, ideal)
    expected_velocity = impact_state[1] - ideal.h_com * impact_state[3]

    np.testing.assert_array_equal(unchanged_state, impact_state)
    np.testing.assert_allclose(landed_state[1], expected_velocity, atol=1e-12)
    np.testing.assert_allclose(landed_state[2:4], [0.0, 0.0], atol=1e-12)
    assert energy(landed_state, ideal) < energy(impact_state, ideal)

    # The landing model cannot require the ground to pull a wheel downward.
    high_inertia = WheelieParams(I_body=0.03, c_v=0.0)
    try:
        rk4_step_np(impact_state, 0.0, 0.01, high_inertia)
    except RuntimeError as error:
        assert "Landing requires rear-contact loss" in str(error)
    else:
        raise AssertionError("An inadmissible rear contact impulse must be rejected.")

    # 6. Landing time and final state must agree with a smaller integration step.
    falling_state = np.array([0.0, 0.2, 0.1, -1.0])
    settled_state = rk4_step_np(falling_state, 0.0, 0.2, ideal)
    refined = WheelieParams(c_v=0.0, integration_dt=0.0005)
    refined_state = rk4_step_np(falling_state, 0.0, 0.2, refined)

    np.testing.assert_allclose(settled_state[2:4], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(settled_state, refined_state, atol=1e-7)
    assert energy(settled_state, ideal) < energy(falling_state, ideal)

    # 7. Lifted motion without input or damping must conserve energy.
    free_state = balance_state + np.array([0.0, 0.2, 0.03, 0.1])
    free_next = rk4_step_np(free_state, 0.0, 0.04, ideal)
    np.testing.assert_allclose(energy(free_next, ideal), energy(free_state, ideal), atol=1e-8)

    # 8. Constructor mass changes must update the default inertia estimate.
    changed = WheelieParams(m=2.0)
    expected_inertia = changed.m * (changed.L**2 + changed.H**2) / 12.0
    np.testing.assert_allclose(changed.I_body, expected_inertia)

    # 9. Check the equations independently using COM force, moment and power.
    rng = np.random.default_rng(4)

    for _ in range(100):
        v = rng.uniform(-1.0, 1.0)
        theta = rng.uniform(0.1, 1.3)
        omega = rng.uniform(-2.0, 2.0)
        tau = rng.uniform(-0.4, 0.4)
        state = np.array([0.0, v, theta, omega])

        derivative = continuous_dynamics_np(state, tau, p)
        acceleration = derivative[1]
        alpha = derivative[3]
        com_x = p.x_com * np.cos(theta) - p.h_com * np.sin(theta)
        com_z = p.x_com * np.sin(theta) + p.h_com * np.cos(theta)

        force = tau / p.r - p.m * p.c_v * v
        normal = p.m * (p.g + com_x * alpha - com_z * omega**2)
        com_acceleration = acceleration - com_z * alpha - com_x * omega**2
        expected_moment = tau - p.c_pitch * omega + com_z * force - com_x * normal

        np.testing.assert_allclose(p.m * com_acceleration, force, atol=1e-11)
        np.testing.assert_allclose(p.I_body * alpha, expected_moment, atol=1e-11)

        energy_dot = (p.m * v - p.m * com_z * omega) * acceleration
        energy_dot += (-p.m * com_x * v * omega + p.m * p.g * com_x) * omega
        energy_dot += (p.I_eff * omega - p.m * com_z * v) * alpha

        drive_power = tau * (v / p.r + omega)
        damping_power = p.m * p.c_v * v**2 + p.c_pitch * omega**2
        net_power = drive_power - damping_power
        np.testing.assert_allclose(energy_dot, net_power, atol=1e-11)

    # 10. Rear-contact loss must be reported because flight is not modeled.
    rear_contact_loss_state = np.array([0.0, 0.0, 1.0, 20.0])
    try:
        rk4_step_np(rear_contact_loss_state, 0.0, 0.01, ideal)
    except RuntimeError as error:
        assert "Rear contact lost" in str(error)
    else:
        raise AssertionError("Rear-contact loss must be reported.")

    print("PASS: rest, axle loads, lift-off, unstable balance, landing, energy, inertia and rear-contact loss.")
    print(f"Measured COM: x={p.x_com:.6f} m, h={p.h_com:.6f} m above axle.")
    print(f"Ideal from-rest lift threshold: {tau_lift:.6f} N m; stationary balance: {p.balance_pitch_deg:.3f} deg.")
    print("This checks the reduced plant only. Original RLS/MPPI rollouts still require a matching update.")


if __name__ == "__main__":
    main()
