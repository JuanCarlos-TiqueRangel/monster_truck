#!/usr/bin/env python3
"""
goal_park.py
------------
A wrapper that GUARANTEES the truck stops at the goal, around any climbing
controller (NMPC or MPPI).

Why this exists
===============
We proved (hard bound, soft penalty, both optimisers) that NOTHING built on the
online RLS+GP model can be relied on to stop at the goal: after a violent climb
the learned model mispredicts, and a model-based controller that trusts it runs
away. The only thing that can GUARANTEE a stop is a control law whose stability
does NOT depend on the learned model.

A PD law on (position, velocity) is exactly that:

    tau = kp * (x - x_goal) + kd * v          # sign: drive is tau<0 -> +x

On flat ground this is a stable spring-damper (mass + drag + this feedback),
so it provably converges to x = x_goal, v = 0 for sane gains -- regardless of
what the GP/RLS believe. It cannot be fooled by a corrupted model.

The learned controller is only needed for the hard part: the wheelie over the
obstacle. So:

    * |pitch| above a threshold  -> the obstacle is being climbed: use the inner
      controller (NMPC or MPPI), which is good at the contact maneuver;
    * otherwise (flat ground)    -> use the PD park law, which is guaranteed to
      reach and hold the goal.

Three details make it robust (learned the hard way):
    * HARD switch, not a blend -- mixing PD torque into the wheelie corrupts the
      climb. The active controller gets full authority.
    * keep the inner controller WARM -- always call its solve() so its warm
      start never goes stale; just ignore the output while on flat ground. (A
      cold NMPC mis-fires the instant it re-engages and flips.)
    * HYSTERESIS on the pitch threshold -- avoids chattering at the boundary.

Implements the same solve(state, ref, tau_prev, a, b, gp_params) interface as
WheelieNMPC / WheelieMPPI, so the sim harness drives it unchanged.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ParkConfig:
    kp: float = 4.0              # position gain  [N.m / m]
    kd: float = 6.0             # velocity gain  [N.m / (m/s)]  (damping)
    pitch_on_deg: float = 14.0   # |pitch| above this -> hand to inner (climb)
    pitch_off_deg: float = 8.0   # |pitch| below this -> back to PD (hysteresis)


class ParkController:
    def __init__(self, inner, p, cfg: ParkConfig = ParkConfig()):
        self.inner = inner
        self.p = p
        self.cfg = cfg
        self._on = math.radians(cfg.pitch_on_deg)
        self._off = math.radians(cfg.pitch_off_deg)
        self.climbing = False

    def reset(self):
        self.climbing = False
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def solve(self, state, ref, tau_prev, a, b, gp_params):
        x, v, theta, _omega = state

        # Always solve the inner controller so its warm start stays fresh,
        # even while we are parking (output ignored on flat ground).
        tau_inner, _info = self.inner.solve(state, ref, tau_prev, a, b, gp_params)

        # Pitch-based mode with hysteresis: climbing vs flat.
        ap = abs(theta)
        if not self.climbing and ap > self._on:
            self.climbing = True
        elif self.climbing and ap < self._off:
            self.climbing = False

        if self.climbing:
            tau = tau_inner                                   # let it do the wheelie
            mode = "climb"
        else:
            tau = self.cfg.kp * (x - ref[0]) + self.cfg.kd * v   # guaranteed PD park
            mode = "park"

        tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))
        return tau, {"success": True, "mode": mode, "tau_inner": float(tau_inner)}
