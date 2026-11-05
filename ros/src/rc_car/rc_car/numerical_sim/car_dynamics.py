from dataclasses import dataclass
import math

import numpy as np


@dataclass
class WheelieParams:
    """Physical parameters and simulation settings for the wheelie model."""

    # m: float = 5.1
    # l: float = 0.2
    # I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    # r: float = 0.081

    x_com = 0.067342
    h_com = 0.073866

    m: float = 1.283
    l: float = np.sqrt(x_com**2 + h_com**2)
    L: float = 0.18
    H = 0.19
    I_body: float = (1.0 / 12.0) * m * (L**2 + H**2)
    r: float = 0.0375
    
    g: float = 9.81
    c_v: float = 9.0
    tau_min: float = -0.2
    tau_max: float = 0.2
    theta_min: float = np.deg2rad(0.0)
    theta_max: float = np.deg2rad(90.0)
    omega_min: float = -5.0
    omega_max: float = 5.0
    v_min: float = -5.0
    v_max: float = 5.0
    
    pitch_ref: float = 70.0
    sim_time: float = 5.0
    sim_dt: float = 0.01

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


def continuous_dynamics_np(
    state: np.ndarray,
    tau: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    """Evaluate the continuous-time numerical plant model."""
    x, v, theta, omega = state
    x_dot = v
    v_dot = tau / (p.m * p.r) - p.c_v * v
    theta_dot = omega
    omega_dot = (-tau + p.m * p.g * p.l * np.cos(theta)) / p.I_eff

    if plant_has_mismatch:
        v_dot += 0.10 * np.sin(theta) + 0.15 * tau * np.cos(theta)
        omega_dot += 0.5 * omega + 1.0 * v + 3.0 * np.sin(theta)

    return np.array([x_dot, v_dot, theta_dot, omega_dot], dtype=float)


def rk4_step_np(
    state: np.ndarray,
    tau: float,
    dt: float,
    p: WheelieParams,
    plant_has_mismatch: bool = False,
) -> np.ndarray:
    """Advance the numerical plant by one fourth-order Runge-Kutta step."""
    k1 = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
    k2 = continuous_dynamics_np(state + 0.5 * dt * k1, tau, p, plant_has_mismatch)
    k3 = continuous_dynamics_np(state + 0.5 * dt * k2, tau, p, plant_has_mismatch)
    k4 = continuous_dynamics_np(state + dt * k3, tau, p, plant_has_mismatch)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)





















# """Wheelie dynamics with front-wheel support and landing.

# State: [rear axle position, rear axle velocity, nose-up pitch, pitch rate].
# Input: actual rear-wheel torque in N m, summed across the rear axle.

# The model assumes rear-wheel rolling without slip. It does not model wheel
# spin inertia, suspension movement, body collisions or flight. COM and mass
# come from measurements; inertia, damping and available torque remain estimates.
# """

# from dataclasses import dataclass
# import math

# import numpy as np


# @dataclass
# class WheelieParams:
#     # Measured geometry. COM height is measured above the rear axle center.
#     m: float = 1.283
#     x_com: float = 0.06734216679657054
#     h_com: float = 0.07386563942621582
#     wheelbase: float = 0.18
#     r: float = 0.0375
#     g: float = 9.81

#     # Retained box approximation, not measured inertia or verified body length.
#     L: float = 0.18
#     H: float = 0.19
#     I_body: float | None = None

#     # Uncalibrated damping and controller torque bounds.
#     # The resisting force acts at the rear axle: F_loss = m * c_v * v.
#     c_v: float = 9.0
#     c_pitch: float = 0.0
#     tau_min: float = -0.4
#     tau_max: float = 0.4

#     theta_min: float = 0.0
#     theta_max: float = math.pi / 2
#     omega_min: float = -5.0
#     omega_max: float = 5.0
#     v_min: float = -5.0
#     v_max: float = 5.0
#     pitch_ref: float = 60.0  # Retained task reference; not stationary balance.
#     sim_time: float = 5.0
#     sim_dt: float = 0.01
#     integration_dt: float = 0.002

#     def __post_init__(self):
#         if self.I_body is None:
#             self.I_body = self.m * (self.L**2 + self.H**2) / 12.0

#         positive_parameters = [self.m, self.r, self.g, self.wheelbase, self.I_body, self.integration_dt]
#         for value in positive_parameters:
#             if not math.isfinite(value) or value <= 0.0:
#                 raise ValueError("Mass, radius, gravity, wheelbase, inertia and integration_dt must be positive.")

#         if not 0.0 < self.x_com < self.wheelbase:
#             raise ValueError("The COM must lie between the rear and front axle centers.")

#         if not math.isfinite(self.h_com) or self.h_com <= 0.0:
#             raise ValueError("COM height above the rear axle must be finite and positive.")

#         for damping in [self.c_v, self.c_pitch]:
#             if not math.isfinite(damping) or damping < 0.0:
#                 raise ValueError("Damping coefficients must be finite and nonnegative.")

#         if not math.isfinite(self.tau_min) or not math.isfinite(self.tau_max):
#             raise ValueError("Controller torque bounds must be finite.")

#         if self.tau_min >= self.tau_max:
#             raise ValueError("Controller torque bounds must be ordered.")

#     @property
#     def l(self):
#         return math.hypot(self.x_com, self.h_com)

#     @property
#     def I_eff(self):
#         return self.I_body + self.m * self.l**2

#     @property
#     def balance_pitch_deg(self):
#         return math.degrees(math.atan2(self.x_com, self.h_com))


# def _free_dynamics_np(state, tau, p, plant_has_mismatch=False):
#     """Calculate coupled accelerations while the front wheels are lifted."""
#     _, v, theta, omega = state

#     drive_tau = float(tau)
#     c_v = p.c_v
#     c_pitch = p.c_pitch

#     if plant_has_mismatch:
#         drive_tau *= 0.9
#         c_v *= 1.1
#         c_pitch *= 1.2

#     # COM position relative to the rear axle at the current pitch.
#     cos_theta = np.cos(theta)
#     sin_theta = np.sin(theta)
#     com_x = p.x_com * cos_theta - p.h_com * sin_theta
#     com_z = p.x_com * sin_theta + p.h_com * cos_theta

#     drive_force = drive_tau / p.r
#     drag_force = p.m * c_v * v
#     centrifugal_force = p.m * com_x * omega**2
#     force = drive_force - drag_force + centrifugal_force

#     gravity_moment = p.m * p.g * com_x
#     damping_moment = c_pitch * omega
#     moment = drive_tau - gravity_moment - damping_moment

#     # Solve the two coupled equations for forward and pitch acceleration.
#     inertia = p.I_eff
#     coupling = p.m * com_z
#     determinant = p.m * inertia - coupling**2
#     v_dot = (inertia * force + coupling * moment) / determinant
#     omega_dot = (coupling * force + p.m * moment) / determinant

#     return np.array([v, v_dot, omega, omega_dot], dtype=float)


# def continuous_dynamics_np(state, tau, p, plant_has_mismatch=False):
#     """Calculate accelerations with front-wheel support when needed."""
#     _, v, theta, omega = state

#     is_level = abs(theta) <= 1e-12
#     is_not_rotating = abs(omega) <= 1e-12

#     if is_level and is_not_rotating:
#         drive_tau = float(tau)
#         c_v = p.c_v

#         if plant_has_mismatch:
#             drive_tau *= 0.9
#             c_v *= 1.1

#         v_dot = drive_tau / (p.m * p.r) - c_v * v
#         lifting_moment = drive_tau - p.m * p.g * p.x_com + p.m * p.h_com * v_dot

#         # Front support is maintained until the lifting moment is positive.
#         if lifting_moment <= 0.0:
#             return np.array([v, v_dot, 0.0, 0.0], dtype=float)

#     return _free_dynamics_np(state, tau, p, plant_has_mismatch)


# def contact_forces_np(state, tau, p, plant_has_mismatch=False):
#     """Return the front and rear normal forces in newtons."""
#     _, _, theta, omega = state
#     derivative = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
#     v_dot = derivative[1]
#     omega_dot = derivative[3]

#     drive_tau = float(tau)
#     if plant_has_mismatch:
#         drive_tau *= 0.9

#     is_level = abs(theta) <= 1e-12
#     is_not_rotating = abs(omega) <= 1e-12
#     has_zero_pitch_acceleration = omega_dot == 0.0

#     if is_level and is_not_rotating and has_zero_pitch_acceleration:
#         front_normal = (p.m * p.g * p.x_com - drive_tau - p.m * p.h_com * v_dot) / p.wheelbase
#         rear_normal = p.m * p.g - front_normal
#         return float(front_normal), float(rear_normal)

#     com_x = p.x_com * np.cos(theta) - p.h_com * np.sin(theta)
#     com_z = p.x_com * np.sin(theta) + p.h_com * np.cos(theta)
#     rear_normal = p.m * (p.g + com_x * omega_dot - com_z * omega**2)

#     return 0.0, float(rear_normal)


# def _rk4_free_np(state, tau, dt, p, plant_has_mismatch):
#     """Advance lifted motion by one RK4 step."""
#     k1 = _free_dynamics_np(state, tau, p, plant_has_mismatch)
#     k2 = _free_dynamics_np(state + 0.5 * dt * k1, tau, p, plant_has_mismatch)
#     k3 = _free_dynamics_np(state + 0.5 * dt * k2, tau, p, plant_has_mismatch)
#     k4 = _free_dynamics_np(state + dt * k3, tau, p, plant_has_mismatch)
#     return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


# def _landing_state_np(state, p):
#     """Apply the landing impulse when both wheel contacts can support it."""
#     delta_omega = -state[3]
#     front_impulse = (p.I_eff - p.m * p.h_com**2) * delta_omega / p.wheelbase
#     rear_impulse = p.m * p.x_com * delta_omega - front_impulse

#     if front_impulse < -1e-10 or rear_impulse < -1e-10:
#         raise RuntimeError("Landing requires rear-contact loss; a model with flight/front-axle pivoting is needed.")

#     # Preserve horizontal momentum and stop the downward pitch motion.
#     state[1] += p.h_com * delta_omega
#     state[2:4] = 0.0
#     return state


# def rk4_step_np(state, tau, dt, p, plant_has_mismatch=False):
#     """Advance the plant through ground motion, lifted motion and landing.

#     Ground motion uses its exact solution. Lifted motion uses RK4. When the
#     front wheels reach the ground, locate the landing time and apply the
#     contact impulse before integrating the rest of the requested time step.
#     """
#     state = np.array(state, dtype=float, copy=True)

#     if state.shape != (4,):
#         raise ValueError("The state must contain position, velocity, pitch and pitch rate.")

#     if not np.isfinite(state).all():
#         raise ValueError("Every state value must be finite.")

#     if not math.isfinite(tau):
#         raise ValueError("Torque must be finite.")

#     if not math.isfinite(dt) or dt < 0.0:
#         raise ValueError("dt must be finite and nonnegative.")

#     if state[2] < -1e-12:
#         raise ValueError("Initial pitch penetrates the front-wheel ground constraint.")

#     if dt == 0.0:
#         return state

#     # Handle a state that starts exactly at a landing event.
#     if state[2] <= 1e-12 and state[3] < 0.0:
#         state = _landing_state_np(state, p)

#     drive_tau = float(tau)
#     c_v = p.c_v

#     if plant_has_mismatch:
#         drive_tau *= 0.9
#         c_v *= 1.1

#     remaining_time = float(dt)

#     while remaining_time > 1e-14:
#         _, rear_normal = contact_forces_np(state, tau, p, plant_has_mismatch)
#         if rear_normal <= 0.0:
#             raise RuntimeError("Rear contact lost: this model does not represent flight or front-axle pivoting.")

#         step_dt = min(p.integration_dt, remaining_time)
#         derivative = continuous_dynamics_np(state, tau, p, plant_has_mismatch)
#         is_level = abs(state[2]) <= 1e-12
#         is_not_rotating = abs(state[3]) <= 1e-12
#         has_zero_pitch_acceleration = derivative[3] == 0.0

#         # Ground motion: keep pitch fixed while the front wheels provide support.
#         if is_level and is_not_rotating and has_zero_pitch_acceleration:
#             v = state[1]

#             if c_v > 0.0:
#                 terminal_v = drive_tau / (p.m * p.r * c_v)
#                 velocity_offset = v - terminal_v
#                 decay = math.exp(-c_v * step_dt)
#                 decay_integral = -math.expm1(-c_v * step_dt) / c_v

#                 state[0] += terminal_v * step_dt + velocity_offset * decay_integral
#                 state[1] = terminal_v + velocity_offset * decay
#             else:
#                 acceleration = drive_tau / (p.m * p.r)
#                 state[0] += v * step_dt + 0.5 * acceleration * step_dt**2
#                 state[1] += acceleration * step_dt

#             state[2:4] = 0.0
#             remaining_time -= step_dt
#             continue

#         # Lifted motion: integrate both forward and pitch acceleration.
#         next_state = _rk4_free_np(state, tau, step_dt, p, plant_has_mismatch)

#         if not np.isfinite(next_state).all():
#             raise FloatingPointError("Nonfinite lifted-state integration.")

#         if next_state[2] >= 0.0:
#             state = next_state
#             remaining_time -= step_dt
#             continue

#         # Landing: find when pitch first reaches zero within this substep.
#         lower_time = 0.0
#         upper_time = step_dt

#         for _ in range(32):
#             middle_time = 0.5 * (lower_time + upper_time)
#             middle_state = _rk4_free_np(state, tau, middle_time, p, plant_has_mismatch)

#             if middle_state[2] > 0.0:
#                 lower_time = middle_time
#             else:
#                 upper_time = middle_time

#         state = _rk4_free_np(state, tau, upper_time, p, plant_has_mismatch)
#         state = _landing_state_np(state, p)
#         remaining_time -= upper_time

#     _, rear_normal = contact_forces_np(state, tau, p, plant_has_mismatch)
#     if rear_normal <= 0.0:
#         raise RuntimeError("Rear contact lost: this model does not represent flight or front-axle pivoting.")

#     return state
