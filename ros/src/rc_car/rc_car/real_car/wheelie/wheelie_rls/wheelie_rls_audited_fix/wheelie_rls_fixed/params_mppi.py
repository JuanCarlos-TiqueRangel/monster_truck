import math
from dataclasses import dataclass


@dataclass
class WheelieParams:
    # Identification data support forward commands only. This is not a hardware safety certification.
    u_min: float = 0.0
    u_max: float = 0.30
    theta_min: float = math.radians(-90.0)
    theta_max: float = 0.0
    omega_min: float = -12.0
    omega_max: float = 12.0
    v_min: float = -5.0
    v_max: float = 5.0


@dataclass
class MPPIConfig:
    dt: float = 0.04  # 25 Hz, NOT 50 Hz.
    N: int = 20
    q_x: float = 0.0
    q_v: float = 0.0  # Do not ask the car to balance at nonzero pitch AND stop.
    q_theta: float = 100.5
    q_omega: float = 1.0
    r_u: float = 0.01
    r_du: float = 0.2
    q_terminal_x: float = 0.0
    q_terminal_v: float = 0.0
    q_terminal_theta: float = 500.5
    q_terminal_omega: float = 1.0
    K: int = 5000
    SIGMA: float = 0.10
    LAM: float = 100.0
    noise_hold_steps: int = 2  # Explore sustained actions, not just sample-to-sample jitter.
    input_delay_steps: int = 3  # Loaded from the trained model by the ROS node.
    cpu_threads: int = 1
    compile_rollouts: bool = False  # Optional; benchmark on your CUDA installation first.


@dataclass
class RLSConfig:
    forgetting_factor: float = 1.0
    initial_covariance: float = 0.1
    sigma_v_dot: float = 1.0
    sigma_omega_dot: float = 1.0
