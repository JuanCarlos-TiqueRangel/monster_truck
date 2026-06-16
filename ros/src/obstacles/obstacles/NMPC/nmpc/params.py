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
    r: float = 0.085
    g: float = 9.81
    c_v: float = 9.0

    tau_min: float = -6.0
    tau_max: float = 8.0

    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(90.0)

    omega_min: float = -8.0
    omega_max: float = 8.0

    v_min: float = -5.0
    v_max: float = 5.0

    pitch_ref: float = 80.0
    sim_time: float = 5.0
    sim_dt: float = 0.1

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.1
    N: int = 30

    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 300.0
    q_omega: float = 15.0

    r_tau: float = 0.1
    r_dtau: float = 0.01

    q_terminal_theta: float = 100.0
    q_terminal_omega: float = 100.0

    ipopt_max_iter: int = 50
