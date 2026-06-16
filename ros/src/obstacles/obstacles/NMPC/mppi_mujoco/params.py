#!/usr/bin/env python3
"""
params.py
---------
Physical model parameters (WheelieParams) and the physics-based RLS seeds
(nominal_rls_seeds) for the planar monster-truck. Self-contained so the MPPI
folder depends on nothing outside it.
"""

import math
from dataclasses import dataclass

import numpy as np


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


def nominal_rls_seeds(p: WheelieParams):
    """Physics-based initial guesses for the two RLS estimators.

    Returns (a0, b0):
        a0 : angular model seed for phi_w = [cos(theta), tau, omega, v, 1]
        b0 : linear  model seed for phi_v = [tau, v, sin(theta), 1]
    """
    a0 = np.array([p.m * p.g * p.l / p.I_eff,   # cos(theta) -> gravity torque
                   -1.0 / p.I_eff,              # tau        -> drive reaction
                   0.0,                         # omega
                   0.0,                         # v
                   0.0])                        # offset
    b0 = np.array([1.0 / (p.m * p.r),           # tau        -> drive accel
                   -p.c_v,                       # v          -> drag
                   0.0,                          # sin(theta) -> gravity proj.
                   0.0])                         # offset
    return a0, b0
