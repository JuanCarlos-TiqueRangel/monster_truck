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

    theta_min: float = math.radians(-65.0)
    theta_max: float = math.radians(0.0)

    omega_min: float = -5.0
    omega_max: float = 5.0

    v_min: float = -5.0
    v_max: float = 5.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.05
    N: int = 20

    # GOAL-REACHING + emergent-wheelie cost. Strong goal attraction (q_x), NO slow-down penalty
    # (q_v=0), and only a MILD flat preference (small q_theta) so a wheelie can EMERGE at the
    # obstacle instead of being forbidden. (The old stay-flat regulator used q_x=5, q_v=55,
    # q_theta=540, w_progress=0 -- those cannot climb; the truck stalls before the obstacle.)
    # q_x: float = 85.0
    # q_v: float = 0.0
    # q_theta: float = 0.0
    # q_omega: float = 80.0
    # r_tau: float = 0.5
    # r_dtau: float = 0.5
    # q_terminal_theta: float = 0.0
    # q_terminal_omega: float = 0.0
    ipopt_max_iter: int = 150

    q_x: float = 0.1
    q_v: float = 0.1
    q_theta: float = 1.1
    q_omega: float = 1.1
    r_tau: float = 0.1
    r_dtau: float = 1.1
    q_terminal_theta: float = 0.1
    q_terminal_omega: float = 1.1


@dataclass
class RLSConfig:
    forgetting_factor: float = 0.9995
    initial_covariance: float = 3.0
    derivative_alpha: float = 0.0
    clip_parameters: bool = False
    sigma_v_dot: float = 2.0
    sigma_omega_dot: float = 5.0