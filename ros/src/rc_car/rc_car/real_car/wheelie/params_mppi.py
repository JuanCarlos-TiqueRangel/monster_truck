import math
from dataclasses import dataclass
import numpy as np


# ============================================================
# Parameters
# ============================================================

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
    # tau_min: float = -0.12
    # tau_max: float = 0.12
    u_min: float = -0.40
    u_max: float = 0.40
    theta_min: float = np.deg2rad(-90.0)
    theta_max: float = np.deg2rad(0.0)
    omega_min: float = -10.0
    omega_max: float = 10.0
    v_min: float = -10.0
    v_max: float = 10.0
    

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2

@dataclass
class MPPIConfig:
    dt: float = 0.04
    N: int = 20
    
    # Pitch acquisition has priority; state bounds also constrain speed and rate.
    q_x: float = 0.0  # Wheelie balance has no position target.
    q_v: float = 1.5
    q_theta: float = 1.5
    q_omega: float = 1.0
    # r_tau: float = 0.01
    # r_dtau: float = 0.2
    r_u: float = 0.01
    r_du: float = 0.2
    # Terminal cost weights
    q_terminal_x: float = 0.0
    q_terminal_v: float = 0.5
    q_terminal_theta: float = 1.5
    q_terminal_omega: float = 1.0

    K: int = 4000            # number of sampled rollouts
    SIGMA: float = 0.1        # exploration std in the same units as /drive
    LAM: float = 10.0           # temperature (softmin sharpness)
    # cpu_threads: int = 1       # Avoid overhead from many threads on small rollouts.

@dataclass
class RLSConfig:
    forgetting_factor: float = 0.95
    initial_covariance: float = 1000.0
    sigma_v_dot: float = 5.0
    sigma_omega_dot: float = 5.0
