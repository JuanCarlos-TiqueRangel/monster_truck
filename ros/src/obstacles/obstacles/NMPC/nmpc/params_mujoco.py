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

    tau_min: float = -8.0
    tau_max: float = 12.0

    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(100.0)

    omega_min: float = -8.0
    omega_max: float = 8.0

    v_min: float = -5.0
    v_max: float = 5.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.1
    N: int = 10
    q_x: float = 5.0
    q_v: float = 55.0
    q_theta: float = 540.0
    q_omega: float = 0.1
    r_tau: float = 0.5
    r_dtau: float = 2.5
    q_terminal_theta: float = 250.0
    q_terminal_omega: float = 0.1
    ipopt_max_iter: int = 50

    # Smooth flip penalty (off by default; enable in the goal-reaching driver).
    # q_flip  : barrier on pitch beyond theta_soft_deg   (prevents tip-over)
    # q_flipw : penalises up-rotation while reared past theta_climb_deg
    q_flip: float = 0.0
    theta_soft_deg: float = 80.0
    q_flipw: float = 0.0
    theta_climb_deg: float = 55.0
