#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

from svgp_dynamics import SVGPManager   # wherever you put the class


from collections import deque
import os
import threading
import csv

import time
import traceback


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = 0.1
    horizon: int = 20
    num_rollouts: int = 200

    # MPPI hyper-parameters
    lambda_: float = 7.0
    sigma: float = 0.7

    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    pitch_target: float = math.pi/2.0
    flip_stop_abs: float = 3.1

    # Paths to trained GP models
    gp_flip_path: str = "models/svgp_dynamics_0.pt"
    gp_rate_path: str = "models/svgp_dynamics_1.pt"

    # ---- logging ----
    log_dir: str = "logs"
    max_log_points: int = 200_000

    # ---- retrain ----
    min_points_to_train: int = 2_000
    N_target_train: int = 1_000
    train_kernel: str = "RQ"
    train_iters: int = 300

    # training window cap (keeps retrain time stable)
    max_points_for_train: int = 50_000

    # retrain trigger
    min_new_points_between_trains: int = 500

    # ---- live learning curve plot ----
    live_plot: bool = True
    live_plot_save_png: bool = True
    
    episode_timeout_sec: float = 60.0   # hard timeout for an episode (s)
    
    entropy_beta: float = 0.0      # set e.g. 0.05–0.5 to encourage exploration
    entropy_use_log: bool = True   # log-variance entropy (stable)
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True  # scale var by dt^2 (recommended)
    
    # ---- seed dataset (initial offline run) ----
    retrain_every_episodes: int = 20   # or 20
    retrain_force_every_N_episodes: bool = True


    # ---- SVGP warm update ----
    svgp_warm_steps: int = 300         # gradient steps per retrain event
    svgp_max_buffer: int = 50_000      # keep last N transitions for training
    svgp_min_new_points: int = 2_000   # need at least this many transitions before first update
    svgp_batch_size: int = 2048


    # ---- balance termination (for 90-degree task) ----
    balance_success_err_rad: float = 0.12    # ~7 deg
    balance_success_rate_rad_s: float = 0.8  # tune
    balance_hold_sec: float = 0.4           # must satisfy success band for this long

    balance_fail_err_rad: float = 2.8        # safety: too far from target (~160 deg)
    balance_fail_rate_rad_s: float = 25.0    # safety: too fast => reset

    arm_settle_sec: float = 0.15
    arm_up_z_min: float = 0.7          # upright-ish (R[2,2] close to +1)
    arm_max_rate_rad_s: float = 8.0    # avoid arming during violent motion

    swingup_switch_err_rad: float = 0.35   # ~20 deg from target

    # w_theta_swing: float = 8.0
    # w_rate_swing: float = 0.3
    w_u_swing: float = 0.01
    w_du_swing: float = 0.02
    w_du_bal: float   = 2.0

    w_theta_bal: float = 60.0
    w_rate_bal: float = 6.0
    w_u_bal: float = 0.05

    # ---- energy shaping swing-up ----
    energy_k: float = 1.0          # scales potential term (unitless here)
    w_E_swing: float = 35.0        # weight on (E - E*)^2 during swing-up
    w_theta_swing: float = 1.0     # IMPORTANT: keep this SMALL during swing-up
    w_rate_swing: float = 0.2      # small damping during swing-up

    target_abs_rad: float = math.pi / 2.0     # stabilize at ±pi/2
    target_hyst_rad: float = 0.20            # hysteresis to avoid target flipping
    down_unlock_rad: float = 0.30            # if near down again, allow re-choosing target

    # local balance controller (very important)
    use_pd_balance: bool = True
    pd_kp: float = 10.0
    pd_kd: float = 1.6

    # MPPI exploration by mode
    sigma_swing: float = 0.35
    sigma_bal: float   = 0.08               # small noise near balance

    # strong terminal cost (important for “stay there”)
    w_term_theta: float = 250.0
    w_term_rate: float  = 25.0

    # PD balance improvements
    pd_ki: float = 2.0          # start 0.5–5.0 depending on scaling
    pd_i_limit: float = 1.0     # limits integral contribution (in "u" units)
    pd_i_leak: float = 0.0      # 0–0.2 optional leakage to prevent windup

    down_unlock_rate_rad_s: float = 0.5   # only (re)choose target when near down AND slow
    default_target_sign: int = +1         # fallback if perfectly still at down







# ============================================================
# MPPI Controller Node
# ============================================================

class MPPICarControllerNode(Node):
    def __init__(self, cfg: MPPIConfig):
        super().__init__("mppi_car_controller")
        self.cfg = cfg

        # ----- Device -----
        self.device = torch.device("cuda")
        self.get_logger().info(f"Using torch device: {self.device}")

        # ----- Load GP models -----
        self.gp_flip: SVGPManager = SVGPManager.load(self.cfg.gp_flip_path, device=self.device)
        self.gp_rate: SVGPManager = SVGPManager.load(self.cfg.gp_rate_path, device=self.device)

        self.gp_flip.device = self.device
        self.gp_rate.device = self.device

        self.pitch_target_t = torch.tensor(
            self.cfg.pitch_target, dtype=torch.float32, device=self.device
        )

        # Cache cost weights as tensors (avoid per-call allocations)
        self.w_theta_swing_t = torch.tensor(self.cfg.w_theta_swing, device=self.device)
        self.w_rate_swing_t  = torch.tensor(self.cfg.w_rate_swing,  device=self.device)
        self.w_u_swing_t     = torch.tensor(self.cfg.w_u_swing,     device=self.device)

        self.w_theta_bal_t   = torch.tensor(self.cfg.w_theta_bal,   device=self.device)
        self.w_rate_bal_t    = torch.tensor(self.cfg.w_rate_bal,    device=self.device)
        self.w_u_bal_t       = torch.tensor(self.cfg.w_u_bal,       device=self.device)
        self.w_du_swing_t = torch.tensor(self.cfg.w_du_swing, device=self.device)
        self.w_du_bal_t   = torch.tensor(self.cfg.w_du_bal,   device=self.device)


        # Energy shaping tensors
        self.energy_k_t   = torch.tensor(self.cfg.energy_k,   device=self.device)
        self.w_E_swing_t  = torch.tensor(self.cfg.w_E_swing,  device=self.device)

        # Recommended: reduce swing weights vs what you currently had
        self.w_theta_swing_t = torch.tensor(self.cfg.w_theta_swing, device=self.device)
        self.w_rate_swing_t  = torch.tensor(self.cfg.w_rate_swing,  device=self.device)

        # Precompute target energy (constant)
        # Potential: V(theta)=k*(1 - cos(theta)), with theta=0 at start pose
        self.V_target_t = self.energy_k_t * (1.0 - torch.cos(self.pitch_target_t))
        self.E_target_t = self.V_target_t  # desired rate at target is ~0, so kinetic term is 0


        self.swing_switch_t  = torch.tensor(self.cfg.swingup_switch_err_rad, device=self.device)


        # ----- ROS interfaces -----
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "car_imu", self.imu_cb, 10)

        self.reset_client = self.create_client(Trigger, "reset_car")
        self.resetting = False
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_car service...")

        # Latest state from IMU
        self.last_flip_rel: float = 0.0
        self.last_rate: float = 0.0
        self.last_state_valid: bool = False

        # For computing flip_rel from quaternion
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        # MPPI warm start
        self.plan: Optional[torch.Tensor] = None
        self.last_u: float = 0.0

        # Reset / arming logic
        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        # Timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

        # ==========================
        # Online dataset buffer
        # ==========================
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.episode_id: int = 0
        self.log_flip = deque(maxlen=self.cfg.max_log_points)
        self.log_rate = deque(maxlen=self.cfg.max_log_points)
        self.log_u    = deque(maxlen=self.cfg.max_log_points)
        self.log_ep   = deque(maxlen=self.cfg.max_log_points)

        # ==========================
        # Retraining state
        # ==========================
        self.training: bool = False
        self.model_lock = threading.Lock()   # protects GP hot-swap vs predict
        self.log_lock = threading.Lock()     # protects deque access/snapshots
        self.train_thread: Optional[threading.Thread] = None

        # ==========================
        # Episode timing metrics
        # ==========================
        # Episode starts when FIRST MPPI action is sent (not at IMU arming)
        self.episode_start_time = None   # rclpy time
        self.episode_started = False
        
        self.last_svgp_update_size = 0  # number of logged points when we last updated

        self.metrics_path = os.path.join(self.cfg.log_dir, "episode_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "flip_time_sec", "retrain_started"])

        # ==========================
        # Live learning curve plot
        # ==========================
        self.live_plot_ok = False
        self.ep_hist = []
        self.t_hist = []
        if self.cfg.live_plot:
            self._init_live_plot()

        # Balance success counter (consecutive samples in success band)
        self.balance_hold_steps = max(1, int(self.cfg.balance_hold_sec / self.cfg.ctrl_dt))
        self.balance_ok_count = 0

        self.target_sign = -1   # or +1 default; will be chosen automatically
        self.target_locked = False

        self.target_pos_t = torch.tensor(+self.cfg.target_abs_rad, device=self.device)
        self.target_neg_t = torch.tensor(-self.cfg.target_abs_rad, device=self.device)

        self.w_term_theta_t = torch.tensor(self.cfg.w_term_theta, device=self.device)
        self.w_term_rate_t  = torch.tensor(self.cfg.w_term_rate,  device=self.device)

        self.err_int = 0.0
        self.prev_target_sign = self.target_sign

        self.prev_imu_time = None
        self.theta_dot_filt = 0.0
        self.theta_dot_alpha = 0.25  # 0.1–0.4 typical


        self.theta_dot_tau = 0.08  # seconds (tune 0.05–0.15)
        self._rate_prev_theta = None
        self._rate_prev_theta_unwrapped = 0.0
        self._rate_prev_time = None
        self._theta_dot_filt = 0.0




    def _update_target_sign(self, pitch: float, rate: float):
        # Choose ONCE when you're near "down" AND moving slowly.
        # After that, keep target fixed for the episode.
        if self.target_locked:
            return

        near_down = abs(pitch) < float(self.cfg.down_unlock_rad)
        slow = abs(rate) < float(self.cfg.down_unlock_rate_rad_s)

        if not (near_down and slow):
            return

        # Break symmetry at down:
        # Prefer using rate sign if available (gives consistent swing direction),
        # otherwise use pitch sign, otherwise fallback.
        if abs(rate) > 1e-3:
            self.target_sign = +1 if rate > 0.0 else -1
        elif abs(pitch) > 1e-3:
            self.target_sign = +1 if pitch > 0.0 else -1
        else:
            self.target_sign = int(self.cfg.default_target_sign)

        self.target_locked = True


    def _target_tensor(self):
        return self.target_pos_t if self.target_sign > 0 else self.target_neg_t

    def _target_float(self):
        return +float(self.cfg.target_abs_rad) if self.target_sign > 0 else -float(self.cfg.target_abs_rad)


    def _pd_balance_u(self, pitch: float, rate: float) -> float:
        target = self._target_float()
        err = self._wrap_angle_float(pitch - target)

        dt = float(self.cfg.ctrl_dt)

        # Integral update (leaky integrator optional)
        if float(self.cfg.pd_i_leak) > 0.0:
            self.err_int *= (1.0 - float(self.cfg.pd_i_leak))

        self.err_int += err * dt

        # Clamp integral contribution in action units
        ki = float(self.cfg.pd_ki)
        if ki > 0.0:
            # clamp integrator state so ki*err_int is bounded by pd_i_limit
            err_int_max = float(self.cfg.pd_i_limit) / ki
            self.err_int = float(np.clip(self.err_int, -err_int_max, +err_int_max))
            i_term = ki * self.err_int
        else:
            i_term = 0.0

        u = -float(self.cfg.pd_kp) * err - float(self.cfg.pd_kd) * rate - float(i_term)

        # Hard saturation
        u_sat = float(np.clip(u, self.cfg.u_min, self.cfg.u_max))

        # Anti-windup: if saturated, back-calculate a bit by freezing integrator growth
        # (simple version: undo this step’s integration if we hit saturation and err pushes further)
        if abs(u - u_sat) > 1e-9:
            self.err_int -= err * dt


        return u_sat



    def _build_training_batch_from_logs(self, start_point_idx: int = 0):
        flip, rate, u, ep = self._snapshot_dataset()
        if flip.size < 2:
            return None, None, None

        dt = float(self.cfg.ctrl_dt)

        # transitions i -> i+1, but only within same episode
        same_ep = (ep[1:] == ep[:-1])

        # choose which transitions to use (based on new points)
        # if start_point_idx = P, new points begin at P, so new transitions begin at P-1
        start_i = max(0, int(start_point_idx) - 1)
        mask = np.zeros_like(same_ep, dtype=bool)
        mask[start_i:] = True
        mask &= same_ep

        idx = np.where(mask)[0]  # transition indices i

        if idx.size == 0:
            return None, None, None

        X = np.stack([flip[idx], rate[idx], u[idx]], axis=1).astype(np.float32)
        y_flip = ((flip[idx + 1] - flip[idx]) / dt).astype(np.float32)
        y_rate = ((rate[idx + 1] - rate[idx]) / dt).astype(np.float32)
        return X, y_flip, y_rate



    # ========================================================
    # Helpers: episode timing + live plot
    # ========================================================
    def _mark_episode_started(self):
        """Call right before publishing the first MPPI action of an episode."""
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True

    def _init_live_plot(self):
        try:
            import matplotlib.pyplot as plt
            self._plt = plt

            plt.ion()
            self.fig, self.ax = plt.subplots()
            (self.line,) = self.ax.plot([], [], marker="o")
            self.ax.set_xlabel("Episode")
            self.ax.set_ylabel("Time to flip [s]")
            self.ax.grid(True)
            try:
                self.fig.canvas.manager.set_window_title("Learning Curve: Episode vs Time-to-Flip")
            except Exception:
                pass

            self.live_plot_ok = True
            self.get_logger().info("Live plot enabled (matplotlib).")
        except Exception as e:
            self.live_plot_ok = False
            self.get_logger().warn(f"Live plot disabled (matplotlib init failed): {e}")

    def _update_live_plot(self, ep: int, flip_time: float):
        if not self.live_plot_ok:
            return

        self.ep_hist.append(ep)
        self.t_hist.append(flip_time)

        self.line.set_data(self.ep_hist, self.t_hist)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

        if self.cfg.live_plot_save_png:
            out = os.path.join(self.cfg.log_dir, "learning_curve.png")
            self.fig.savefig(out, dpi=150)


    # ========================================================
    # Helpers: quaternion -> R + pitch, angle unwrap
    # ========================================================
    @staticmethod
    def quat_to_R_and_pitch(qw, qx, qy, qz):
        R00 = 1 - 2 * (qy * qy + qz * qz)
        R01 = 2 * (qx * qy - qw * qz)
        R02 = 2 * (qx * qz + qw * qy)

        R10 = 2 * (qx * qy + qw * qz)
        R11 = 1 - 2 * (qx * qx + qz * qz)
        R12 = 2 * (qy * qz - qw * qx)

        R20 = 2 * (qx * qz - qw * qy)
        R21 = 2 * (qy * qz + qw * qx)
        R22 = 1 - 2 * (qx * qx + qy * qy)

        pitch = -math.asin(max(-1.0, min(1.0, R20)))
        R = np.array([[R00, R01, R02],
                      [R10, R11, R12],
                      [R20, R21, R22]], dtype=float)
        return R, pitch


    @staticmethod
    def unwrap_angle(prev_angle, prev_unwrapped, angle):
        if prev_angle is None:
            return angle, angle
        d = angle - prev_angle
        if d > math.pi:
            angle_unwrapped = prev_unwrapped + (d - 2 * math.pi)
        elif d < -math.pi:
            angle_unwrapped = prev_unwrapped + (d + 2 * math.pi)
        else:
            angle_unwrapped = prev_unwrapped + d
        return angle, angle_unwrapped


    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        R, _ = self.quat_to_R_and_pitch(qw, qx, qy, qz)
        up_z = float(R[2, 2])

        # Your chosen angle coordinate
        theta = math.atan2(float(R[0, 2]), float(R[2, 2]))

        now = self.get_clock().now()

        # =========================================================
        # 1) CONSISTENT RATE: d(theta_unwrapped)/dt every IMU sample
        #    (independent of "waiting_post_reset" early returns)
        # =========================================================
        if not hasattr(self, "_rate_prev_theta"):
            self._rate_prev_theta = None
            self._rate_prev_theta_unwrapped = 0.0
            self._rate_prev_time = None
            self._theta_dot_filt = 0.0

        if self._rate_prev_time is None or self._rate_prev_theta is None:
            # First sample after startup/reset
            self._rate_prev_time = now
            self._rate_prev_theta = theta
            self._rate_prev_theta_unwrapped = theta
            theta_unwrapped_rate = theta
            theta_dot = 0.0
            pitch_rate = 0.0
        else:
            dt = (now - self._rate_prev_time).nanoseconds * 1e-9
            self._rate_prev_time = now

            if dt <= 1e-4:
                # Fallback: if timing is broken, do not blow up
                theta_unwrapped_rate = self._rate_prev_theta_unwrapped
                theta_dot = 0.0
            else:
                # Unwrap theta relative to LAST sample for the RATE estimator
                prev_theta = self._rate_prev_theta
                prev_unwrapped = self._rate_prev_theta_unwrapped

                _, theta_unwrapped_rate = self.unwrap_angle(prev_theta, prev_unwrapped, theta)
                theta_dot = (theta_unwrapped_rate - prev_unwrapped) / dt

                # Advance the rate estimator state (CRITICAL)
                self._rate_prev_theta = theta
                self._rate_prev_theta_unwrapped = theta_unwrapped_rate

            # First-order low-pass with time constant (more stable than fixed alpha)
            tau = float(getattr(self, "theta_dot_tau", 0.08))  # 0.05–0.15 typical
            alpha = dt / (tau + dt) if dt > 0.0 else 1.0
            self._theta_dot_filt = (1.0 - alpha) * float(self._theta_dot_filt) + alpha * float(theta_dot)

            pitch_rate = float(self._theta_dot_filt)

        # =========================================================
        # Debug compare (now they should match closely)
        # =========================================================
        if not hasattr(self, "_dbg_prev_theta"):
            self._dbg_prev_theta = theta
            self._dbg_prev_t = now
            self._dbg_k = 0
        else:
            dt_dbg = (now - self._dbg_prev_t).nanoseconds * 1e-9
            if dt_dbg > 1e-4:
                dtheta = self._wrap_angle_float(theta - self._dbg_prev_theta)
                dtheta_dt = dtheta / dt_dbg
                self._dbg_k += 1
                if self._dbg_k % 50 == 0:
                    self.get_logger().info(f"dtheta/dt={dtheta_dt:+.2f}  rate(theta_dot)={pitch_rate:+.2f}")
            self._dbg_prev_theta = theta
            self._dbg_prev_t = now

        # =========================================================
        # 2) POST-RESET ARMING GATE (FIXED up_z COMPARISON)
        # =========================================================
        if self.waiting_post_reset:
            if self.resetting:
                self.last_state_valid = False
                return

            if self.post_reset_start_time is None:
                self.last_state_valid = False
                return

            elapsed = (now - self.post_reset_start_time).nanoseconds * 1e-9
            if elapsed < float(self.cfg.arm_settle_sec):
                self.last_state_valid = False
                return

            # ---- FIX: sign-agnostic up vector gate ----
            # Treat cfg.arm_up_z_min as an ABS threshold (e.g., 0.7).
            # This allows either +1 or -1 depending on your body frame convention.
            upz_abs_min = float(abs(self.cfg.arm_up_z_min))
            if abs(up_z) < upz_abs_min:
                self.last_state_valid = False

                # Throttled reason log (once per ~1s) to verify what's failing
                if not hasattr(self, "_arm_dbg_t"):
                    self._arm_dbg_t = now
                dt_arm_dbg = (now - self._arm_dbg_t).nanoseconds * 1e-9
                if dt_arm_dbg > 1.0:
                    self._arm_dbg_t = now
                    self.get_logger().warn(
                        f"Arming blocked (up_z abs too small): up_z={up_z:+.3f} |abs|={abs(up_z):.3f} "
                        f"min={upz_abs_min:.3f}  rate={pitch_rate:+.3f}  elapsed={elapsed:.2f}s"
                    )
                return

            if abs(pitch_rate) > float(self.cfg.arm_max_rate_rad_s):
                self.last_state_valid = False

                if not hasattr(self, "_arm_dbg_t"):
                    self._arm_dbg_t = now
                dt_arm_dbg = (now - self._arm_dbg_t).nanoseconds * 1e-9
                if dt_arm_dbg > 1.0:
                    self._arm_dbg_t = now
                    self.get_logger().warn(
                        f"Arming blocked (rate too high): up_z={up_z:+.3f}  rate={pitch_rate:+.3f} "
                        f"max={float(self.cfg.arm_max_rate_rad_s):.3f}  elapsed={elapsed:.2f}s"
                    )
                return

            # Arm: initialize reference
            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            self.theta0 = theta

            self.last_flip_rel = 0.0
            self.last_rate = pitch_rate
            self.last_state_valid = True

            self.waiting_post_reset = False
            self.watchdog_fired = False
            return


        # =========================================================
        # 3) NORMAL CONTROL LOGIC (unwrap + relative angle)
        # =========================================================
        self.prev_theta, theta_unwrapped_ctrl = self.unwrap_angle(
            self.prev_theta, self.prev_theta_unwrapped, theta
        )
        self.prev_theta_unwrapped = theta_unwrapped_ctrl

        if self.theta0 is None:
            self.theta0 = theta_unwrapped_ctrl

        flip_rel = theta_unwrapped_ctrl - self.theta0
        flip_rel = (flip_rel + math.pi) % (2 * math.pi) - math.pi

        self.last_flip_rel = float(flip_rel)
        self.last_rate = float(pitch_rate)
        self.last_state_valid = True







    # ========================================================
    # Torch helpers: angdiff, stage cost, GP step
    # ========================================================
    @staticmethod
    def angdiff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi


    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor, target_t: torch.Tensor):
        pitch = states[:, 0]
        rate  = states[:, 1]
        u     = actions

        # angle error to target
        err = self.angdiff_torch(pitch, target_t)
        abs_err = torch.abs(err)

        # regime switch (swing-up far from target)
        swing = abs_err > self.swing_switch_t  # shape (K,)

        # --------------------------
        # Swing-up energy shaping
        # --------------------------
        # Potential V = k*(1 - cos(pitch)), with pitch=0 at "down/start"
        V = self.energy_k_t * (1.0 - torch.cos(pitch))
        # Total "energy" proxy (unitless)
        E = 0.5 * (rate ** 2) + V

        # Target energy corresponds to reaching pitch_target with ~0 rate
        E_err = E - self.E_target_t

        # Swing cost:
        # - match energy to target (pump energy)
        # - SMALL angle term to choose the correct side
        # - small damping and control penalty
        cost_swing = (
            self.w_E_swing_t * (E_err ** 2)
            + self.w_theta_swing_t * (err ** 2)
            + self.w_rate_swing_t * (rate ** 2)
            + self.w_u_swing_t * (u ** 2)
        )

        # --------------------------
        # Balance near target
        # --------------------------
        cost_bal = (
            self.w_theta_bal_t * (err ** 2)
            + self.w_rate_bal_t  * (rate ** 2)
            + self.w_u_bal_t     * (u ** 2)
        )

        # Select per-sample cost
        swing_f = swing.to(cost_bal.dtype)
        return cost_bal + swing_f * (cost_swing - cost_bal)








    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        dt = float(self.cfg.ctrl_dt)
        X = torch.stack([states[:, 0], states[:, 1], actions], dim=-1)

        with self.model_lock:
            d_flip_mean, _ = self.gp_flip.predict_torch(X)  # (K,)
            d_rate_mean, _ = self.gp_rate.predict_torch(X)  # (K,)

        next_states = torch.empty_like(states)
        next_states[:, 0] = states[:, 0] + d_flip_mean * dt
        next_states[:, 1] = states[:, 1] + d_rate_mean * dt

        next_states[:, 0] = torch.remainder(next_states[:, 0] + torch.pi, 2 * torch.pi) - torch.pi
        next_states[:, 1].clamp_(-20.0, 20.0)

        return next_states, None





    # ========================================================
    # MPPI core
    # ========================================================
    @torch.no_grad()
    def mppi_action(self, x0_np, sigma: float):
        cfg = self.cfg
        H = cfg.horizon
        K = cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        assert x0.shape == (2,)

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        #eps = torch.randn(K, H, device=self.device) * cfg.sigma
        eps = torch.randn(K, H, device=self.device) * float(sigma)
        U = torch.clamp(u_init.unsqueeze(0) + eps, cfg.u_min, cfg.u_max)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        beta = float(self.cfg.entropy_beta)

        last_u = torch.tensor(self.last_u, device=self.device, dtype=torch.float32)

        target_t = self._target_tensor()

        for t in range(H):
            u_t = U[:, t]

            if t == 0:
                du_t = u_t - last_u
            else:
                du_t = u_t - U[:, t-1]

            # err = self.angdiff_torch(states[:, 0], self.pitch_target_t)
            err = self.angdiff_torch(states[:, 0], target_t)
            swing = torch.abs(err) > self.swing_switch_t
            w_du_t = torch.where(swing, self.w_du_swing_t, self.w_du_bal_t)

            stage = self.stage_cost_torch(states, u_t, target_t) + w_du_t * du_t**2

            states, ent = self.gp_step_batch_torch(states, u_t)
            
            if ent is not None:
                stage = stage - beta * ent  # maximize entropy

            costs = costs + stage

        # terminal cost
        # terminal cost: forces MPPI to care about end-of-horizon being stable
        errT = self.angdiff_torch(states[:, 0], target_t)
        costs = costs + self.w_term_theta_t * (errT ** 2) + self.w_term_rate_t * (states[:, 1] ** 2)

        J_min = costs.min()
        weights = torch.exp(-(costs - J_min) / cfg.lambda_)
        weights_sum = weights.sum() + 1e-8

        du = (weights.unsqueeze(1) * eps).sum(dim=0) / weights_sum
        u_new = torch.clamp(u_init + du, cfg.u_min, cfg.u_max)

        self.plan = u_new.detach()
        return float(u_new[0].detach().cpu())


    # ========================================================
    # Control timer callback
    # ========================================================
    def control_timer_cb(self):

        t0 = time.perf_counter()

        # Pause MPPI while training/reloading
        if self.training:
            self.plan = None
            self.publish_u(0.0)
            return

        # Watchdog for arming
        if self.waiting_post_reset and (not self.resetting) and (self.post_reset_start_time is not None):
            elapsed = (self.get_clock().now() - self.post_reset_start_time).nanoseconds * 1e-9
            if (elapsed > 3.0) and (not self.watchdog_fired):
                self.watchdog_fired = True
                self.get_logger().warn("Stuck in waiting_post_reset. Forcing reset retry.")
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        if self.resetting or self.waiting_post_reset:
            self.publish_u(0.0)
            return

        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first IMU message...")
                self.warned_no_imu = True
            self.publish_u(0.0)
            return
        self.warned_no_imu = False

        flip_rel = self.last_flip_rel
        rate = self.last_rate

        self._update_target_sign(flip_rel, rate)
        target = self._target_float()
        err_target = self._wrap_angle_float(flip_rel - target)

        if self.target_sign != self.prev_target_sign:
            self.err_int = 0.0
            self.plan = None
            self.prev_target_sign = self.target_sign

        # -----------------------------
        # Episode timeout (always ends episode)
        # -----------------------------
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(self.cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={self.cfg.episode_timeout_sec:.2f}s)."
                )
                # On timeout, do NOT force retrain; let gating decide (unless periodic_force triggers)
                self._end_episode(reason="timeout", force_retrain=False)
                return

        # FAIL (use selected ± target)
        if (abs(err_target) >= float(self.cfg.balance_fail_err_rad)) or (abs(rate) >= float(self.cfg.balance_fail_rate_rad_s)):
            self.get_logger().warn(
                f"Episode {int(self.episode_id)} FAIL: |err|={abs(err_target):.3f}, |rate|={abs(rate):.3f}."
            )
            self._end_episode(reason="fail", force_retrain=False)
            return

        # SUCCESS (pass target)
        if self._update_balance_success_counter(flip_rel, rate, target):
            self.get_logger().info(
                f"Episode {int(self.episode_id)} SUCCESS: held for {self.balance_hold_steps} steps."
            )
            self._end_episode(reason="success", force_retrain=False)
            return


        # -----------------------------
        # Optional: keep flip_stop_abs as a hard safety (if it ever full-flips)
        # -----------------------------
        if abs(flip_rel) >= float(self.cfg.flip_stop_abs):
            self.get_logger().warn(
                f"Episode {int(self.episode_id)} reached flip_stop_abs={self.cfg.flip_stop_abs:.2f}."
            )
            self._end_episode(reason="flip_stop_abs", force_retrain=True)
            return

        # use err_target everywhere:
        in_balance_zone = abs(err_target) <= float(self.cfg.swingup_switch_err_rad)


        if in_balance_zone and bool(self.cfg.use_pd_balance):
            # Deterministic stabilizer (recommended)
            self.plan = None
            u_cmd = self._pd_balance_u(flip_rel, rate)
        else:
            # MPPI swing-up / recovery
            sigma = float(self.cfg.sigma_bal) if in_balance_zone else float(self.cfg.sigma_swing)
            x0 = np.array([flip_rel, rate], dtype=np.float32)
            try:
                u_cmd = self.mppi_action(x0, sigma=sigma)
            except Exception as e:
                self.get_logger().error(f"MPPI error: {e}")
                u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, self.cfg.u_min, self.cfg.u_max))

        print("")
        print("[flip_rel]: ", flip_rel)
        print("[target]: ", target)
        print("[err_target]: ", err_target)
        print("[u_cmd]: ", u_cmd)
        dt = time.perf_counter() - t0
        if dt > 0.02:
            self.get_logger().warn(f"control_timer_cb took {dt*1000:.1f} ms")

        print("")



        self._mark_episode_started()
        self.publish_u(u_cmd)
        self._shift_plan()
        self._log_step(flip_rel, rate, u_cmd)




    def _shift_plan(self):
        if self.plan is None:
            return
        with torch.no_grad():
            self.plan = torch.cat([self.plan[1:], self.plan[-1:].clone()], dim=0)
            self.plan[-1] = 0.0  # or keep last value


    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)


    # ==========================
    # Logging + retraining
    # ==========================
    def _log_step(self, flip_rel: float, rate: float, u: float):
        with self.log_lock:
            self.log_flip.append(float(flip_rel))
            self.log_rate.append(float(rate))
            self.log_u.append(float(u))
            self.log_ep.append(int(self.episode_id))

    def _snapshot_dataset(self):
        with self.log_lock:
            flip = np.asarray(list(self.log_flip), dtype=np.float32)
            rate = np.asarray(list(self.log_rate), dtype=np.float32)
            u    = np.asarray(list(self.log_u),    dtype=np.float32)
            ep   = np.asarray(list(self.log_ep),   dtype=np.int64)
        return flip, rate, u, ep

    def _save_dataset_npz(self, flip, rate, u, ep):
        out = os.path.join(self.cfg.log_dir, f"dataset_ep{self.episode_id:04d}.npz")
        np.savez_compressed(out, flip=flip, rate=rate, u=u, episode_id=ep, dt=np.array(self.cfg.ctrl_dt, dtype=np.float32))
        self.get_logger().info(f"Saved dataset snapshot: {out}")
        

    def _start_retrain_async(self, force: bool = False) -> bool:
        if self.training:
            return False

        with self.log_lock:
            n_points = len(self.log_flip)

        n_trans_total = max(0, n_points - 1)

        if not force:
            if n_trans_total < int(self.cfg.svgp_min_new_points):
                return False

            start_point_idx = int(self.last_svgp_update_size)
            X_new, _, _ = self._build_training_batch_from_logs(start_point_idx=start_point_idx)
            n_new_trans = 0 if X_new is None else int(X_new.shape[0])
            if n_new_trans < int(self.cfg.min_new_points_between_trains):
                return False

        # (optional) if force=True and there are 0 transitions total, we can still warm_update
        # only if the SVGP checkpoint already has a buffer stored.
        if force and n_trans_total == 0:
            if self.gp_flip.Xn_train is None or self.gp_flip.Xn_train.size(0) == 0:
                self.get_logger().info("Force retrain requested but no transitions and no stored SVGP buffer.")
                return False

        self.training = True
        n_at_start = int(n_points)

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(n_at_start, force),
            daemon=True,
        )
        self.train_thread.start()
        return True





    def _train_worker(self, n_at_start: int, force: bool = False):
        """
        Train/update SVGP models asynchronously.

        Behavior:
        - Normal (force=False): train ONLY on NEW valid transitions since last update.
        - Forced  (force=True): if no new transitions, fall back to ALL valid transitions.
        - If still no transitions and SVGP has a stored replay buffer (from ckpt),
            do a pure warm_update() with no new data.
        """
        t0 = time.perf_counter()

        # "new points begin here" (points index, not transitions)
        start_point_idx = int(self.last_svgp_update_size)

        try:
            # 1) Preferred: only NEW transitions since last update
            X, y_flip, y_rate = self._build_training_batch_from_logs(start_point_idx=start_point_idx)

            # 2) If forced and there were no new valid transitions, fall back to ALL transitions
            if X is None and force:
                self.get_logger().warn(
                    "Force retrain: no NEW valid transitions. Falling back to ALL valid transitions."
                )
                X, y_flip, y_rate = self._build_training_batch_from_logs(start_point_idx=0)

            # 3) If still nothing, attempt pure warm_update using existing SVGP buffers (if present)
            if X is None:
                with self.model_lock:
                    has_buf = (
                        getattr(self.gp_flip, "Xn_train", None) is not None
                        and self.gp_flip.Xn_train.numel() > 0
                        and getattr(self.gp_rate, "Xn_train", None) is not None
                        and self.gp_rate.Xn_train.numel() > 0
                    )

                    if not has_buf:
                        self.get_logger().info(
                            "SVGP update skipped: no valid transitions available and no stored SVGP replay buffer."
                        )
                        return

                    self.get_logger().warn(
                        "Force retrain: no transitions available. Running warm_update() on stored SVGP buffer only."
                    )

                    self.gp_flip.warm_update(
                        steps=int(self.cfg.svgp_warm_steps),
                        batch_size=int(self.cfg.svgp_batch_size) if hasattr(self.cfg, "svgp_batch_size") else None,
                    )
                    self.gp_rate.warm_update(
                        steps=int(self.cfg.svgp_warm_steps),
                        batch_size=int(self.cfg.svgp_batch_size) if hasattr(self.cfg, "svgp_batch_size") else None,
                    )

                    # Atomic save
                    os.makedirs(os.path.dirname(self.cfg.gp_flip_path) or ".", exist_ok=True)
                    tmp_flip = self.cfg.gp_flip_path + ".tmp"
                    tmp_rate = self.cfg.gp_rate_path + ".tmp"
                    self.gp_flip.save(tmp_flip)
                    self.gp_rate.save(tmp_rate)
                    os.replace(tmp_flip, self.cfg.gp_flip_path)
                    os.replace(tmp_rate, self.cfg.gp_rate_path)

                # Commit last update size after success
                self.last_svgp_update_size = int(n_at_start)

                elapsed = time.perf_counter() - t0
                self.get_logger().info(
                    f"SVGP warm-update (buffer-only) done in {elapsed:.2f}s | "
                    f"warm_steps={self.cfg.svgp_warm_steps} | "
                    f"start_point_idx={start_point_idx} -> n_at_start={n_at_start}"
                )
                return

            # Optional cap on how many transitions to train on (keeps time stable)
            K = int(self.cfg.max_points_for_train)
            if X.shape[0] > K:
                X = X[-K:]
                y_flip = y_flip[-K:]
                y_rate = y_rate[-K:]

            # 4) Warm-start update in-place (add new data then train)
            with self.model_lock:
                self.gp_flip.add_data(
                    X_new=X,
                    Y_new=y_flip,
                    retrain=True,
                    warm_steps=int(self.cfg.svgp_warm_steps),
                    max_points=int(self.cfg.svgp_max_buffer),
                    keep_raw=False,
                )
                self.gp_rate.add_data(
                    X_new=X,
                    Y_new=y_rate,
                    retrain=True,
                    warm_steps=int(self.cfg.svgp_warm_steps),
                    max_points=int(self.cfg.svgp_max_buffer),
                    keep_raw=False,
                )

                # Atomic checkpoint save
                os.makedirs(os.path.dirname(self.cfg.gp_flip_path) or ".", exist_ok=True)
                tmp_flip = self.cfg.gp_flip_path + ".tmp"
                tmp_rate = self.cfg.gp_rate_path + ".tmp"
                self.gp_flip.save(tmp_flip)
                self.gp_rate.save(tmp_rate)
                os.replace(tmp_flip, self.cfg.gp_flip_path)
                os.replace(tmp_rate, self.cfg.gp_rate_path)

            # Commit "last update size" ONLY after success
            self.last_svgp_update_size = int(n_at_start)

            elapsed = time.perf_counter() - t0
            self.get_logger().info(
                f"SVGP warm-update done in {elapsed:.2f}s | "
                f"transitions_used={X.shape[0]} | warm_steps={self.cfg.svgp_warm_steps} | "
                f"start_point_idx={start_point_idx} -> n_at_start={n_at_start} | force={int(force)}"
            )
            self.get_logger().info("Saved SVGP checkpoints after warm-update.")

        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.get_logger().error(f"SVGP update failed after {elapsed:.2f}s: {e}")
            self.get_logger().error(traceback.format_exc())

        finally:
            self.training = False

            

    @staticmethod
    def _wrap_angle_float(x: float) -> float:
        return (x + math.pi) % (2 * math.pi) - math.pi

    def _end_episode(self, reason: str, force_retrain: bool = False):
        """
        Ends the current episode, optionally starts retraining, records metrics, then resets.
        Retrain is only attempted at episode boundaries (safe).
        """
        self.get_logger().info(f"Ending episode {int(self.episode_id)} | reason={reason}")

        # stop motor immediately
        self.publish_u(0.0)

        # Decide whether to force retrain on this boundary
        ep_num_1based = int(self.episode_id) + 1
        periodic_force = (
            bool(self.cfg.retrain_force_every_N_episodes)
            and int(self.cfg.retrain_every_episodes) > 0
            and (ep_num_1based % int(self.cfg.retrain_every_episodes) == 0)
        )
        force = bool(force_retrain or periodic_force)

        # Attempt retrain; _start_retrain_async already gates on "enough new data" if not forced
        started = self._start_retrain_async(force=force)

        # Record episode metric + whether retrain started
        self._record_episode_metric(retrain_started=started)

        # Reset (force so we don't get stuck)
        self.request_reset(force=True)

    def _update_balance_success_counter(self, flip_rel: float, rate: float, target: float) -> bool:
        """
        Returns True if balance success achieved (held for required duration).
        """
        err = self._wrap_angle_float(flip_rel - target)

        if (abs(err) <= float(self.cfg.balance_success_err_rad)) and (abs(rate) <= float(self.cfg.balance_success_rate_rad_s)):
            self.balance_ok_count += 1
        else:
            self.balance_ok_count = 0

        return self.balance_ok_count >= int(self.balance_hold_steps)



    # ==========================
    # Episode metric recording
    # ==========================
    def _record_episode_metric(self, retrain_started: bool):
        if self.episode_start_time is None:
            # episode never started (no MPPI action sent), nothing to record
            self.episode_started = False
            return

        dt = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
        ep = int(self.episode_id)

        self.get_logger().info(
            f"Episode {ep} flip time: {dt:.3f} s | retrain_started={int(retrain_started)}"
        )

        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, float(dt), int(retrain_started)])

        # live plot update
        self._update_live_plot(ep, float(dt))

        # reset to avoid double logging
        self.episode_start_time = None
        self.episode_started = False


    # ==========================
    # Reset logic
    # ==========================
    def _local_reset_state(self):
        self.prev_theta = None
        self.prev_theta_unwrapped = 0.0
        self.theta0 = None
        self.last_state_valid = False
        self.plan = None
        self.last_u = 0.0

        # reset episode timing
        self.episode_start_time = None
        self.episode_started = False

        self.balance_ok_count = 0
        self.err_int = 0.0

        self._rate_prev_theta = None
        self._rate_prev_theta_unwrapped = 0.0
        self._rate_prev_time = None
        self._theta_dot_filt = 0.0

        self.target_locked = False
        self.target_sign = int(self.cfg.default_target_sign)


        # Optional: reset debug
        if hasattr(self, "_dbg_prev_theta"):
            del self._dbg_prev_theta
            del self._dbg_prev_t
            del self._dbg_k




    def request_reset(self, force: bool = False):
        if self.resetting:
            return
        if self.waiting_post_reset and not force:
            return

        self.resetting = True
        self.waiting_post_reset = True
        self.post_reset_start_time = None
        self.watchdog_fired = False

        # next samples belong to next episode
        self.episode_id += 1

        self._local_reset_state()

        req = Trigger.Request()
        future = self.reset_client.call_async(req)

        def done_callback(f):
            try:
                resp = f.result()
                self.get_logger().info(f"Reset response: {resp.message}")
            except Exception as e:
                self.get_logger().warn(f"Reset service call failed: {e}")

            self.resetting = False
            self.post_reset_start_time = self.get_clock().now()
            self.watchdog_fired = False

        future.add_done_callback(done_callback)


# ============================================================
# main()
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    cfg = MPPIConfig()
    node = MPPICarControllerNode(cfg)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down MPPI controller, sending u=0.0")
        node.publish_u(0.0)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
