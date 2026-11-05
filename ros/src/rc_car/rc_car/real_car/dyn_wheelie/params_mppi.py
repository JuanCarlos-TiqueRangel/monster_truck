import math
from dataclasses import dataclass
import numpy as np

# ============================================================
# Parameters
# ============================================================

@dataclass
class WheelieParams:
    # ========================================================
    # Losi LMT physical model
    # ========================================================

    # m: float = 5.1
    # l: float = 0.20
    # I_body: float = ((1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2))
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

    # ========================================================
    # Physical torque used INSIDE the dynamics model [N m]
    # ========================================================

    tau_min: float = -0.5
    tau_max: float = 0.5
    tau_acquire: float = 0.2


    drive_min: float = -0.32
    drive_max: float = 0.32

    # ========================================================
    # Real-car state convention
    #
    # Negative pitch = nose up.
    # ========================================================

    theta_min: float = math.radians(-90.0)
    theta_max: float = math.radians(0.0)

    omega_min: float = -10.0
    omega_max: float = 10.0

    v_min: float = -10.0
    v_max: float = 10.0

    @property
    def I_eff(self) -> float:
        return (
            self.I_body
            + self.m * self.l**2
        )

    @property
    def gravity_torque(self) -> float:
        return (
            self.m
            * self.g
            * self.l
        )

    def tau_to_drive(self, tau: float) -> float:
        """
        Convert the physical model input tau [N m] to the
        normalized command expected on /drive.
        """
        tau = max(
            self.tau_min,
            min(self.tau_max, float(tau)),
        )

        if self.tau_max == self.tau_min:
            return self.drive_min

        ratio = (
            (tau - self.tau_min)
            / (self.tau_max - self.tau_min)
        )

        drive = (
            self.drive_min
            + ratio
            * (self.drive_max - self.drive_min)
        )

        return max(
            self.drive_min,
            min(self.drive_max, drive),
        )

    def drive_to_tau(self, drive: float) -> float:
        """
        Inverse of tau_to_drive().
        """
        drive = max(
            self.drive_min,
            min(self.drive_max, float(drive)),
        )

        if self.drive_max == self.drive_min:
            return self.tau_min

        ratio = (
            (drive - self.drive_min)
            / (self.drive_max - self.drive_min)
        )

        return (
            self.tau_min
            + ratio
            * (self.tau_max - self.tau_min)
        )


@dataclass
class MPPIConfig:
    dt: float = 0.03
    N: int = 25

    # ========================================================
    # Stage / running cost
    # ========================================================

    q_x: float = 0.0
    q_v: float = 1.0
    q_theta: float = 100.5
    q_omega: float = 5.5

    r_tau: float = 0.1
    r_dtau: float = 1.0

    # ========================================================
    # Terminal cost
    # ========================================================

    q_terminal_x: float = 0.0
    q_terminal_v: float = 1.0
    q_terminal_theta: float = 200.0
    q_terminal_omega: float = 10.0

    # ========================================================
    # MPPI
    # ========================================================

    K: int = 5000
    SIGMA: float = 0.15
    LAM: float = 0.6

    # ========================================================
    # Goal tolerances
    # ========================================================

    goal_tol_x: float = 0.10
    goal_tol_v: float = 0.10
    goal_tol_theta: float = math.radians(2.0)
    goal_tol_omega: float = 0.10
