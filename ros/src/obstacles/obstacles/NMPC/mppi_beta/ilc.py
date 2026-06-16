#!/usr/bin/env python3
"""
ilc.py
------
Iterative Learning Control (ILC) on top of the MPPI + SSGP stack.

Role split (see also SSGP.py / mppi.py):
  * MPPI is the fast per-step FEEDBACK solver (horizon ~1 s, no memory of the lap).
  * SSGP is the persistent one-step world-MODEL (RLS linear + GP residual).
  * ILC is the per-EPISODE FEEDFORWARD learner: it remembers the whole previous
    lap and pre-compensates next time. This is the only component with memory
    ACROSS episodes, so it is what actually makes the truck go A->B faster.

How the SSGP/RLS model is used here
-----------------------------------
The feedforward update is velocity-tracking:
    e_j(x)        = v_des(x) - v_j(x)                 (full-lap speed error)
    u_ff_{j+1}(x) = u_ff_j(x) + (lr / b0) * e_j(x)    (only where pitch is safe)
The step is divided by b0 = dv_dot/dtau, the torque->acceleration gain of the
LEARNED model (RLS linear part, sharpened by the GP residual). So the model sets
the correct sign and physical scale of every ILC step; a better model -> better-
aimed, larger, safe steps. A safety gate rolls the feedforward back if a lap gets
too close to flipping, so ILC self-discovers the fastest profile that stays safe.

The feedforward is indexed by PATH POSITION x (not time), so it transfers even
though each lap's timing differs.
"""
import json
import numpy as np


class ILC:
    def __init__(self, goal_x=8.0, grid_n=81, v_des=1.5, lr=0.35,
                 ff_max=3.0, pitch_mask_deg=70.0, pitch_gate_deg=100.0,
                 rollback=0.5, b0_floor=0.5, leak=0.08, smooth=2):
        """
        goal_x         : finish line; feedforward is defined on x in [0, goal_x].
        grid_n         : number of path-grid knots (81 -> 0.1 m spacing for 8 m).
        v_des          : target speed to track [m/s] (the controller's own ceiling).
        lr             : ILC learning rate (dimensionless; scaled by 1/b0).
        ff_max         : clamp on |feedforward torque| [N.m].
        pitch_mask_deg : only push harder where |pitch| < this (leave the delicate
                         wheelie/climb sections alone -- that is where flips happen).
        pitch_gate_deg : if a lap's max |pitch| exceeds this, roll the feedforward
                         back instead of increasing (safety gate).
        rollback       : multiplicative roll-back factor when the gate trips.
        b0_floor       : floor on the model torque-gain so the step never blows up.
        leak           : Q-filter LEAKAGE (forgetting). The leaky update
                         u_ff <- (1-leak)*u_ff + (lr/b0)*err has a STABLE fixed
                         point u_ff* = (lr/(b0*leak))*err*, so the feedforward
                         stops winding up and the lap time converges instead of
                         oscillating into flips.
        smooth         : spatial Q-filter half-width (knots) -- moving-average the
                         feedforward each update so it stays low-frequency and
                         doesn't inject flip-inducing torque spikes.
        """
        self.goal_x = float(goal_x)
        self.grid = np.linspace(0.0, goal_x, grid_n)
        self.u_ff = np.zeros(grid_n)
        self.v_des = float(v_des)
        self.lr = float(lr)
        self.ff_max = float(ff_max)
        self.pitch_mask = float(pitch_mask_deg)
        self.pitch_gate = float(pitch_gate_deg)
        self.rollback = float(rollback)
        self.b0_floor = float(b0_floor)
        self.leak = float(leak)
        self.smooth = int(smooth)
        self.episode = 0
        self.history = []          # per-episode diagnostics
        # trust-region state: never lose the fastest SAFE profile found so far
        self.best_u_ff = np.zeros(grid_n)
        self.best_time = float("inf")
        self.last_inc = np.zeros(grid_n)   # last step taken away from best
        self.n_accept = 0                  # accepted improvements so far
        self.anneal = 0.7                  # shrink exploration per accept -> settles on best

    def _qfilter(self, u):
        """Spatial low-pass (moving average) -- the ILC Q-filter for robustness."""
        if self.smooth <= 0:
            return u
        w = 2 * self.smooth + 1
        ker = np.ones(w) / w
        return np.convolve(u, ker, mode="same")

    # ---- used live, inside the control loop ----
    def feedforward(self, x):
        """Interpolated feedforward torque at path position x (0 outside the track)."""
        if x <= 0.0:
            return float(self.u_ff[0])
        if x >= self.goal_x:
            return float(self.u_ff[-1])
        return float(np.interp(x, self.grid, self.u_ff))

    # ---- used once per episode, to learn ----
    def update(self, hist, b0, t2g):
        """Update the feedforward from one finished lap (trust-region ILC).

        hist : the run_episode history dict (needs 'x', 'v', 'pitch_deg').
        b0   : torque->accel gain dv_dot/dtau from the learned RLS+GP model
               (= rls_v.theta[0]); sets the physical scale of the ILC step.
        t2g  : measured time-to-goal for the lap just run (the objective).

        The profile we just ran is `self.u_ff`. We keep `best_u_ff`, the fastest
        SAFE profile ever seen:
          * lap SAFE and FASTER  -> accept it as the new best, then probe further
            in the velocity-error direction (leaky + Q-filtered);
          * otherwise            -> reject: BACKTRACK halfway from the run profile
            toward best (and halve the step), so a bad lap can never strand us in
            an unsafe profile. This makes improvement monotone, not oscillatory.
        Returns a diagnostics dict.
        """
        self.episode += 1
        x = np.asarray(hist["x"], float)
        v = np.asarray(hist["v"], float)
        pitch = np.abs(np.asarray(hist["pitch_deg"], float))
        max_pitch = float(pitch.max()) if pitch.size else 0.0
        reached = bool(x.size and x.max() >= self.goal_x - 1e-6)
        safe = reached and (max_pitch <= self.pitch_gate)

        # velocity-error direction on the path grid (from this lap)
        keep = np.concatenate(([True], np.diff(x) > 1e-9))
        xs, vs, ps = x[keep], v[keep], pitch[keep]
        if xs.size >= 2:
            v_on_grid = np.interp(self.grid, xs, vs, left=vs[0], right=vs[-1])
            p_on_grid = np.interp(self.grid, xs, ps, left=ps[0], right=ps[-1])
            b0 = max(abs(float(b0)), self.b0_floor)
            err = self.v_des - v_on_grid                 # >0 where too slow
            mask = (p_on_grid < self.pitch_mask).astype(float)
            # anneal exploration as good profiles are found -> converge ONTO best
            step = (self.lr * self.anneal ** self.n_accept / b0) * err * mask
            step[self.grid < 0.2] = 0.0
        else:
            step = np.zeros_like(self.u_ff)

        if safe and t2g < self.best_time - 1e-3:
            # ACCEPT: the lap we ran is the new fastest-safe profile; probe onward
            self.best_u_ff = self.u_ff.copy()
            self.best_time = t2g
            self.n_accept += 1
            proposal = self._qfilter((1.0 - self.leak) * self.best_u_ff + step)
            gate = "ACCEPT"
        else:
            # REJECT: fall back toward the best safe profile, shrink the step
            self.last_inc = 0.5 * self.last_inc
            proposal = self._qfilter(self.best_u_ff + self.last_inc)
            gate = "BACKTRACK"

        proposal = np.clip(proposal, -self.ff_max, self.ff_max)
        self.last_inc = proposal - self.best_u_ff
        self.u_ff = proposal

        diag = dict(episode=self.episode, gate=gate, max_pitch=max_pitch,
                    reached=reached, t2g=float(t2g), best_time=float(self.best_time),
                    mean_ff=float(np.abs(self.u_ff).mean()))
        self.history.append(diag)
        return diag

    # ---- persistence ----
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"grid": self.grid.tolist(), "u_ff": self.u_ff.tolist(),
                       "best_u_ff": self.best_u_ff.tolist(),
                       "best_time": self.best_time, "episode": self.episode,
                       "v_des": self.v_des, "history": self.history}, f, indent=2)

    def load(self, path):
        from pathlib import Path
        if not Path(path).exists():
            return False
        with open(path) as f:
            d = json.load(f)
        self.grid = np.asarray(d["grid"], float)
        self.u_ff = np.asarray(d["u_ff"], float)
        self.best_u_ff = np.asarray(d.get("best_u_ff", d["u_ff"]), float)
        self.best_time = float(d.get("best_time", float("inf")))
        self.episode = int(d.get("episode", 0))
        self.history = d.get("history", [])
        return True
