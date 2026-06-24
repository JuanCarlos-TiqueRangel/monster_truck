#!/usr/bin/env python3
"""
prewheelie_learner.py
---------------------
Episodic 1-D optimiser for the GP-discovered pre-wheelie angle (NMPC.theta_obs).

Division of labour:
  * the GP discovers WHERE the obstacle is (by interaction, online);
  * this learns HOW MUCH to rear there -- the pre-wheelie angle that crosses the obstacle
    FASTEST while staying SAFE (no flip) -- improving every episode.

It is a trust-region pattern search on the (noisy) crossing-time objective with a safety gate:
  * keep the best SAFE-and-FAST angle seen so far (best_angle / best_time);
  * a candidate that beats it (faster AND safe) -> ACCEPT, keep probing in that direction;
  * otherwise                                  -> shrink the step, reverse, probe from best.
So an unsafe/slow episode never strands the search; best_time is monotone non-increasing.
"""

import json
import numpy as np


class PreWheelieLearner:
    def __init__(self, angle0=0.0, step0=15.0, ang_min=0.0, ang_max=70.0,
                 flip_deg=88.0, anneal=0.6, step_min=2.0, margin=0.05):
        self.angle = float(angle0)              # current candidate angle [deg]
        self.step = float(step0)                # probe step [deg]
        self.dir = 1                            # probe direction (+1 / -1)
        self.ang_min, self.ang_max = float(ang_min), float(ang_max)
        self.flip_deg = float(flip_deg)         # |pitch| above this = unsafe (flip)
        self.anneal = float(anneal)             # step shrink factor on a reject
        self.step_min = float(step_min)
        self.margin = float(margin)             # must beat best by this [s] (noise guard)
        self.best_angle = float(angle0)
        self.best_time = float("inf")
        self.history = []

    def update(self, t_cross, max_pitch, reached):
        """Record this episode's outcome; set & return the NEXT angle to try."""
        safe = bool(reached and max_pitch <= self.flip_deg)
        improved = safe and (t_cross < self.best_time - self.margin)
        if improved:
            self.best_angle, self.best_time = self.angle, float(t_cross)
            nxt = self.angle + self.dir * self.step                 # keep probing this way
            if nxt < self.ang_min or nxt > self.ang_max:            # bound -> reverse, shrink
                self.dir = -self.dir
                self.step = max(self.step * self.anneal, self.step_min)
                nxt = self.best_angle + self.dir * self.step
        else:
            self.step = max(self.step * self.anneal, self.step_min)  # shrink
            self.dir = -self.dir                                     # reverse
            nxt = self.best_angle + self.dir * self.step
        self.history.append(dict(angle=self.angle, t_cross=float(t_cross),
                                 max_pitch=float(max_pitch), safe=safe,
                                 best_angle=self.best_angle, best_time=self.best_time))
        self.angle = float(np.clip(nxt, self.ang_min, self.ang_max))
        return self.angle

    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(best_angle=self.best_angle, best_time=self.best_time,
                           history=self.history), f, indent=2)
