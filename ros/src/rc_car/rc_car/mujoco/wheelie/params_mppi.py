import math
from dataclasses import dataclass


# ============================================================
# Parameters
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1
    l: float = 0.2
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.081
    g: float = 9.81
    c_v: float = 9.0

    tau_min: float = -8.0
    tau_max: float = 8.0

    theta_min: float = math.radians(-90.0)
    theta_max: float = math.radians(0.0)

    omega_min: float = -5.0
    omega_max: float = 5.0

    v_min: float = -5.0
    v_max: float = 5.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPPIConfig:
    dt: float = 0.02
    N: int = 15

    # Stage/Running cost weights 
    q_x: float = 1.5
    q_v: float = 1.0
    q_theta: float = 15.0
    q_omega: float = 1.5
    r_tau: float = 0.01
    r_dtau: float = 1.1

    # Terminal cost weights
    q_progress: float = 30.0 # was 30 
    q_terminal_x: float = 2.1
    q_terminal_theta: float = 1.0
    q_terminal_omega: float = 0.0
    ipopt_max_iter: int = 50

    q_gp_var: float = 0.0
    K: int = 5000            # number of sampled rollouts
    SIGMA: float = 10.0        # exploration std on tau
    LAM: float = 1.6           # temperature (softmin sharpness)

    goal_tol_x: float = 0.10
    goal_tol_v: float = 0.10
    goal_tol_theta: float = math.radians(2.0)
    goal_tol_omega: float = 0.10


@dataclass
class RLSConfig:
    forgetting_factor: float = 0.9995
    initial_covariance: float = 3.0
    derivative_alpha: float = 0.0
    clip_parameters: bool = False
    sigma_v_dot: float = 2.0
    sigma_omega_dot: float = 5.0
