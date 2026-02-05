#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import time
import traceback
from collections import deque
import os
import threading
import csv

from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


BASE_DIR = Path(__file__).resolve().parent   # .../obstacles
DATA_DIR = BASE_DIR / "data"
GP_DIR   = BASE_DIR / "gp"

#from gp_dynamics import GPManager  # <-- your GPManager with .load()
from gp.gp_dynamics import GPManager

from utils import geometry

# from re_train_dynamics_gp import train_dynamics_gp_from_arrays
from gp.re_train_dynamics_gp import train_dynamics_gp_from_arrays


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = 0.1
    horizon: int = 20
    num_rollouts: int = 2000

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    pitch_target: float = 1.2 #math.pi/2.0
    flip_stop_abs: float = 2.2

    # Paths to trained GP models
    # gp_flip_path: str = "gp/models/gp_dynamics_pitch_d_dt.pt"
    # gp_rate_path: str = "gp/models/gp_dynamics_rate_d_dt.pt"

    gp_flip_path: str = str(GP_DIR / "models" / "gp_dynamics_flip_d_dt.pt")
    gp_rate_path: str = str(GP_DIR / "models" / "gp_dynamics_rate_d_dt.pt")
    gp_pos_x_path: str = str(GP_DIR / "models" / "gp_dynamics_x_pose_d_dt.pt")
    gp_vx_path: str = str(GP_DIR / "models" / "gp_dynamics_linear_speed_x_d_dt.pt")

    # ---- logging ----
    log_dir: str = str(BASE_DIR / "logs")
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
    
    episode_timeout_sec: float = 20.0   # hard timeout for an episode (s)
    
    entropy_beta: float = 0.1      # set e.g. 0.05–0.5 to encourage exploration
    entropy_use_log: bool = True   # log-variance entropy (stable)
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True  # scale var by dt^2 (recommended)
    
    # ---- seed dataset (initial offline run) ----
    seed_npz_path: str = str(DATA_DIR / "mujoco_manual_wheelie.npz")

    seed_episode_id: int = -1                           # fixed “seed episode”
    keep_seed: bool = True                              # always include seed points
    retrain_every_episodes: int = 20   # or 20

    # ---- obstacle trigger (wheelie only when close) ----
    obs_trigger_dist: float = 1.0       # [m] start wheelie around here
    obs_trigger_smooth: float = 0.20    # [m] smoothness of sigmoid gate

    # ---- weights ----
    w_flat: float = 6.0
    w_wheelie: float = 1.5
    w_u: float = 0.01
    w_rate: float = 2.0

    # safety: prevent over-rotation (wheelie not full flip)
    pitch_limit: float = 1.45           # ~83 deg
    w_pitch_limit: float = 80.0

    wheelie_hold_time_sec: float = 0.5
    wheelie_pitch_tol: float = 0.15
    wheelie_rate_tol: float = 1.0

    goal_x: float = 5.0      # example: set this to “past the obstacle”
    w_goal: float = 3.0
    w_vx: float = 0.5
    v_des: float = 1.0       # desired forward speed

    wheelie_hold_sigma_scale: float = 0.25   # reduce sigma during hold
    wheelie_hold_extra_wheelie: float = 2.0  # increase wheelie weight during hold



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
        self.gp_flip: GPManager = GPManager.load(self.cfg.gp_flip_path)
        self.gp_rate: GPManager = GPManager.load(self.cfg.gp_rate_path)
        self.gp_pose_x: GPManager = GPManager.load(self.cfg.gp_pos_x_path)
        self.gp_vx: GPManager = GPManager.load(self.cfg.gp_vx_path)
        self.gp_flip.device = self.device
        self.gp_rate.device = self.device
        self.gp_pose_x.device = self.device
        self.gp_vx.device = self.device

        self.pitch_target_t = torch.tensor(
            self.cfg.pitch_target, dtype=torch.float32, device=self.device
        )

        self.obs_pos_x: Optional[float] = None
        self.car_pos_x: Optional[float] = None
        self.car_vx: float = 0.0
        self.last_odom_valid: bool = False
        self.wheelie_hold_steps = 0

        # ----- ROS interfaces -----
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)

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
        self.log_x  = deque(maxlen=self.cfg.max_log_points)
        self.log_vx = deque(maxlen=self.cfg.max_log_points)
        
        self.log_cost_j = 0.0


        # ==========================
        # Retraining state
        # ==========================
        self.training: bool = False
        self.reload_pending: bool = False
        self.model_lock = threading.Lock()   # protects GP hot-swap vs predict
        self.log_lock = threading.Lock()     # protects deque access/snapshots
        self.train_thread: Optional[threading.Thread] = None
        self.reset_after_retrain = False

        # last_train_size is only "committed" AFTER a successful train
        self.last_train_size = 0

        # ==========================
        # Episode timing metrics
        # ==========================
        # Episode starts when FIRST MPPI action is sent (not at IMU arming)
        self.episode_start_time = None   # rclpy time
        self.episode_started = False

        self.metrics_path = os.path.join(self.cfg.log_dir, "episode_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "time_to_goal_sec", "retrain_started"])

        # ==========================
        # Live learning curve plot
        # ==========================
        self.live_plot_ok = False
        self.ep_hist = []
        self.t_hist = []
        self.c_hist = []
        if self.cfg.live_plot:
            self._init_live_plot()


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
            self.ax.set_ylabel("cost")
            self.ax.grid(True)
            try:
                self.fig.canvas.manager.set_window_title("Learning Curve: Episode vs Time_to_goal_sec")
            except Exception:
                pass

            self.live_plot_ok = True
            self.get_logger().info("Live plot enabled (matplotlib).")
        except Exception as e:
            self.live_plot_ok = False
            self.get_logger().warn(f"Live plot disabled (matplotlib init failed): {e}")

    def _update_live_plot(self, ep: int, flip_time: float, cost: float):
        if not self.live_plot_ok:
            return

        self.ep_hist.append(ep)
        self.t_hist.append(flip_time)
        self.c_hist.append(cost)

        # self.line.set_data(self.ep_hist, self.t_hist)
        self.line.set_data(self.ep_hist, self.c_hist)
        self.ax.relim()
        self.ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._plt.pause(0.001)

        if self.cfg.live_plot_save_png:
            out = os.path.join(self.cfg.log_dir, "learning_curve.png")
            self.fig.savefig(out, dpi=150)


    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        R, _ = geometry.quat_to_R_and_pitch(qw, qx, qy, qz)
        up_z = R[2, 2]
        theta = math.atan2(R[0, 2], R[2, 2])
        pitch_rate = float(msg.angular_velocity.y)

        # ---------------------------
        # POST-RESET ARMING LOGIC
        # ---------------------------
        if self.waiting_post_reset:
            if self.resetting:
                self.last_state_valid = False
                return

            # only arm when reasonably "ready"
            if up_z < 0.8:
                self.last_state_valid = False
                return

            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            self.theta0 = theta

            self.last_flip_rel = 0.0
            self.last_rate = pitch_rate
            self.last_state_valid = True

            # NOTE: episode timing DOES NOT start here anymore
            self.waiting_post_reset = False
            self.watchdog_fired = False
            return


        # --- guard for first-ever IMU / startup race ---
        if self.prev_theta is None:
            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            if self.theta0 is None:
                self.theta0 = theta


        # ---------------------------
        # Normal episode logic
        # ---------------------------
        self.prev_theta, theta_unwrapped = geometry.unwrap_angle(
            self.prev_theta, self.prev_theta_unwrapped, theta
        )
        self.prev_theta_unwrapped = theta_unwrapped

        if self.theta0 is None:
            self.theta0 = theta_unwrapped

        flip_rel = theta_unwrapped - self.theta0
        flip_rel = max(-math.pi, min(math.pi, flip_rel))

        self.last_flip_rel = flip_rel
        self.last_rate = pitch_rate
        self.last_state_valid = True


    def obs_callback(self, msg : PoseStamped):
        self.obs_pos_x = float(msg.pose.position.x)

    def car_callback(self, msg: Odometry):
        self.car_pos_x = float(msg.pose.pose.position.x)
        self.car_vx = float(msg.twist.twist.linear.x)
        self.last_odom_valid = True



    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor, obs_x: torch.Tensor) -> torch.Tensor:
        # states: (K,4) = [x, vx, pitch, rate]
        x     = states[:, 0]
        vx    = states[:, 1]
        pitch = states[:, 2]
        rate  = states[:, 3]
        u     = actions

        # distance to obstacle in front (if behind -> treat as far)
        d_obs = obs_x - x
        d_obs = torch.where(d_obs > 0.0, d_obs, torch.full_like(d_obs, 1e6))

        # wheelie gate near obstacle
        d_trig  = float(self.cfg.obs_trigger_dist)
        d_sigma = float(self.cfg.obs_trigger_smooth)
        g = torch.sigmoid((d_trig - d_obs) / (d_sigma + 1e-6))

        # pitch tracking (flat far, wheelie near)
        err_wheelie = geometry.angdiff_torch(pitch, self.pitch_target_t)
        err_flat    = pitch

        cost_pitch = (1.0 - g) * self.cfg.w_flat    * (err_flat ** 2) \
                + (g)       * self.cfg.w_wheelie * (err_wheelie ** 2)

        # goal + speed
        goal_x = float(self.cfg.goal_x)
        goal_scale = (1.0 - 0.7 * g)  # still cares about goal near obstacle
        cost_goal  = goal_scale * self.cfg.w_goal * (x - goal_x)**2
        cost_vx    = goal_scale * self.cfg.w_vx   * (vx - float(self.cfg.v_des))**2


        # regularization
        cost_u    = self.cfg.w_u * (u ** 2)
        cost_rate = self.cfg.w_rate * (rate ** 2)

        # pitch limit safety
        pitch_lim = float(self.cfg.pitch_limit)
        cost_lim  = self.cfg.w_pitch_limit * torch.relu(torch.abs(pitch) - pitch_lim) ** 2

        return cost_pitch + cost_goal + cost_vx + cost_u + cost_rate + cost_lim




    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        # states: (K,4) = [x, vx, pitch, rate]
        x    = states[:, 0]
        vx   = states[:, 1]
        pitch= states[:, 2]
        rate = states[:, 3]

        # X: (K,5) = [x, vx, pitch, rate, u]
        X = torch.stack([x, vx, pitch, rate, actions], dim=-1)

        with self.model_lock:
            if self.cfg.entropy_beta <= 0.0:
                dx_mean, _    = self.gp_pose_x.predict_torch(X)
                dvx_mean, _   = self.gp_vx.predict_torch(X)
                dp_mean, _    = self.gp_flip.predict_torch(X)
                dr_mean, _    = self.gp_rate.predict_torch(X)
                dx_var = dvx_var = dp_var = dr_var = None
            else:
                dx_mean,  dx_var  = self.gp_pose_x.predict_torch(X)
                dvx_mean, dvx_var = self.gp_vx.predict_torch(X)
                dp_mean,  dp_var  = self.gp_flip.predict_torch(X)
                dr_mean,  dr_var  = self.gp_rate.predict_torch(X)

        dt = float(self.cfg.ctrl_dt)

        next_states = torch.empty_like(states)
        next_states[:, 0] = x     + dx_mean  * dt
        next_states[:, 1] = vx    + dvx_mean * dt
        next_states[:, 2] = pitch + dp_mean  * dt
        next_states[:, 3] = rate  + dr_mean  * dt

        # reasonable clamps (tune)
        next_states[:, 2].clamp_(-math.pi, math.pi)
        next_states[:, 3].clamp_(-20.0, 20.0)

        if self.cfg.entropy_beta <= 0.0:
            return next_states, None

        # entropy (optional)
        dx_var  = torch.clamp(dx_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dvx_var = torch.clamp(dvx_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dp_var  = torch.clamp(dp_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        dr_var  = torch.clamp(dr_var,  min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)

        if self.cfg.entropy_dt_scale:
            var_next = torch.stack([dx_var, dvx_var, dp_var, dr_var], dim=-1) * (dt * dt)
        else:
            var_next = torch.stack([dx_var, dvx_var, dp_var, dr_var], dim=-1)

        if self.cfg.entropy_use_log:
            entropy = 0.5 * torch.log(var_next).sum(dim=-1)
        else:
            entropy = var_next.sum(dim=-1)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return next_states, entropy


    # ========================================================
    # MPPI core
    # ========================================================
    @torch.no_grad()
    def mppi_action(self, x0_np):
        cfg = self.cfg
        H, K = cfg.horizon, cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        assert x0.shape == (4,)

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan
        eps = torch.randn(K, H, device=self.device) * cfg.sigma
        U = torch.clamp(u_init.unsqueeze(0) + eps, cfg.u_min, cfg.u_max)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        beta = float(cfg.entropy_beta)

        # obs_x for rollout gating
        if self.obs_pos_x is None:
            obs_x0 = torch.full((K,), 1e6, dtype=torch.float32, device=self.device)
        else:
            obs_x0 = torch.full((K,), float(self.obs_pos_x), dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t, obs_x0)
            states, ent = self.gp_step_batch_torch(states, u_t)
            if not torch.isfinite(states).all():
                self.get_logger().error("Rollout produced non-finite states (GP output likely NaN/Inf).")
                break

            if ent is not None:
                stage = stage - beta * ent
            costs = costs + stage

        self.log_cost_j = costs.mean().item()

        J_min = costs.min()
        w = torch.exp(-(costs - J_min) / cfg.lambda_)
        wsum = w.sum() + 1e-8

        du = (w.unsqueeze(1) * eps).sum(dim=0) / wsum
        u_new = torch.clamp(u_init + du, cfg.u_min, cfg.u_max)

        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        
        return float(u_new[0].detach().cpu())



    def control_timer_cb(self):
        cfg = self.cfg

        # Pause MPPI while training/reloading
        if self.training or self.reload_pending:
            self.plan = None
            self.wheelie_hold_steps = 0
            self.publish_u(0.0)
            self._reload_models_if_ready()
            return

        # Watchdog for arming
        if self.waiting_post_reset and (not self.resetting) and (self.post_reset_start_time is not None):
            elapsed = (self.get_clock().now() - self.post_reset_start_time).nanoseconds * 1e-9
            if (elapsed > 3.0) and (not self.watchdog_fired):
                self.watchdog_fired = True
                self.get_logger().warn("Stuck in waiting_post_reset. Forcing reset retry.")
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        if self.resetting or self.waiting_post_reset:
            self.wheelie_hold_steps = 0
            self.publish_u(0.0)
            return

        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first IMU message...")
                self.warned_no_imu = True
            self.wheelie_hold_steps = 0
            self.publish_u(0.0)
            return
        self.warned_no_imu = False

        flip_rel = float(self.last_flip_rel)  # pitch (your "flip_rel")
        rate = float(self.last_rate)

        if not self.last_odom_valid:
            self.publish_u(0.0)
            return

        # -------------------------------------------------
        # Obstacle distance (1D along +x). If unknown or behind -> FAR.
        # -------------------------------------------------
        if (self.obs_pos_x is None) or (self.car_pos_x is None):
            d_obs = 1e6
        elif self.obs_pos_x <= self.car_pos_x:
            d_obs = 1e6  # obstacle behind -> FAR (important!)
        else:
            d_obs = float(self.obs_pos_x - self.car_pos_x)

        # -------------------------------------------------
        # Episode timeout: if we started sending actions but haven't succeeded, force reset
        # -------------------------------------------------
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={cfg.episode_timeout_sec:.2f}s). Forcing reset."
                )

                self._record_episode_metric(retrain_started=False)
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        # -------------------------------------------------
        # Emergency stop: flipped way too far
        # (Keep this as a safety even for wheelies.)
        # -------------------------------------------------
        if abs(flip_rel) >= float(cfg.flip_stop_abs):
            self.publish_u(0.0)

            ep_num = self.episode_id + 1  # 1-based
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = False
            if do_retrain_now:
                started = self._start_retrain_async(force=True)
            else:
                self.get_logger().info(
                    f"Skipping retrain this episode (ep_num={ep_num}). "
                    f"retrain_every_episodes={cfg.retrain_every_episodes}"
                )

            self._record_episode_metric(retrain_started=started)

            self.wheelie_hold_steps = 0
            if started:
                self.reset_after_retrain = True
                return
            else:
                self.reset_after_retrain = False
                self.request_reset()
                return

        # -------------------------------------------------
        # Wheelie hold logic (stabilize wheelie near obstacle)
        # -------------------------------------------------
        wheelie_hold_time_sec = float(cfg.wheelie_hold_time_sec)
        wheelie_pitch_tol     = float(cfg.wheelie_pitch_tol)
        wheelie_rate_tol      = float(cfg.wheelie_rate_tol)

        hold_steps_needed = max(1, int(round(wheelie_hold_time_sec / float(cfg.ctrl_dt))))
        close = (d_obs < float(cfg.obs_trigger_dist))

        # angle-wrapped pitch error to target
        err = (flip_rel - float(cfg.pitch_target) + math.pi) % (2.0 * math.pi) - math.pi

        # Update hold counter FIRST
        if close and (abs(err) < wheelie_pitch_tol) and (abs(rate) < wheelie_rate_tol):
            self.wheelie_hold_steps += 1
        else:
            self.wheelie_hold_steps = 0

        holding = close and (self.wheelie_hold_steps >= hold_steps_needed)

        if holding and self.wheelie_hold_steps == hold_steps_needed:
            self.plan = None

        # Need x to proceed
        if self.car_pos_x is None:
            self.publish_u(0.0)
            return

        # Goal check (success)
        if self.car_pos_x >= float(cfg.goal_x):
            self.publish_u(0.0)

            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = self._start_retrain_async(force=False) if do_retrain_now else False
            self._record_episode_metric(retrain_started=started)

            if started:
                self.reset_after_retrain = True
                return
            else:
                self.request_reset()
                return


        # -------------------------------------------------
        # MPPI (normal control) with temporary "hold" shaping
        # -------------------------------------------------
        x0 = np.array([self.car_pos_x, self.car_vx, flip_rel, rate], dtype=np.float32)

        # Save originals (always restore)
        old_sigma     = cfg.sigma
        old_w_wheelie = cfg.w_wheelie
        old_w_flat    = cfg.w_flat

        try:
            if holding:
                cfg.sigma     = old_sigma * float(cfg.wheelie_hold_sigma_scale)
                cfg.w_wheelie = old_w_wheelie * float(cfg.wheelie_hold_extra_wheelie)
                cfg.w_flat    = old_w_flat * 0.5  # IMPORTANT: do NOT *= (no accumulation)

            try:
                u_cmd = self.mppi_action(x0)
                if not math.isfinite(u_cmd):
                    self.get_logger().error("u_cmd is NaN/Inf coming out of MPPI. Forcing 0.")
                    u_cmd = 0.0

            except Exception as e:
                self.get_logger().error(f"MPPI error: {e}")
                self.get_logger().error(traceback.format_exc())
                u_cmd = 0.0

        finally:
            # ALWAYS restore, even if returns/errors happen
            cfg.sigma     = old_sigma
            cfg.w_wheelie = old_w_wheelie
            cfg.w_flat    = old_w_flat

        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self.publish_u(u_cmd)
        self._log_step(flip_rel, rate, u_cmd)



    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)


    # ==========================
    # Logging + retraining
    # ==========================
    def _log_step(self, flip_rel: float, rate: float, u: float):
        if self.car_pos_x is None:
            return
        with self.log_lock:
            self.log_flip.append(float(flip_rel))
            self.log_rate.append(float(rate))
            self.log_x.append(float(self.car_pos_x))
            self.log_vx.append(float(self.car_vx))
            self.log_u.append(float(u))
            self.log_ep.append(int(self.episode_id))


    def _snapshot_dataset(self):
        with self.log_lock:
            flip = np.asarray(list(self.log_flip), dtype=np.float32)
            rate = np.asarray(list(self.log_rate), dtype=np.float32)
            x  = np.asarray(list(self.log_x), dtype=np.float32)
            vx = np.asarray(list(self.log_vx), dtype=np.float32)
            u    = np.asarray(list(self.log_u),    dtype=np.float32)
            ep   = np.asarray(list(self.log_ep),   dtype=np.int64)
        return flip, rate, x, vx, u, ep

    def _save_dataset_npz(self, flip, rate, x, vx, u, ep):
        out = os.path.join(self.cfg.log_dir, f"dataset_ep{self.episode_id:04d}.npz")
        np.savez_compressed(
            out, 
            flip=flip, 
            rate=rate, 
            x_pose=x, 
            linear_speed_x=vx, 
            u=u, 
            episode_id=ep, 
            dt=np.array(self.cfg.ctrl_dt, dtype=np.float32)
            )
        
        self.get_logger().info(f"Saved dataset snapshot: {out}")

    def _start_retrain_async(self, force: bool = False) -> bool:
        if self.training:
            self.get_logger().info("Retrain requested but training is already running; skipping.")
            return False

        with self.log_lock:
            n = len(self.log_flip)

        # Only enforce these checks if NOT forced
        if not force:
            if n < self.cfg.min_points_to_train:
                self.get_logger().info(f"Not enough data to retrain yet: {n} < {self.cfg.min_points_to_train}")
                return False

            if (n - self.last_train_size) < self.cfg.min_new_points_between_trains:
                self.get_logger().info("Not enough new data since last train; skipping.")
                return False

        flip, rate, x, vx, u, ep = self._snapshot_dataset()
        self._save_dataset_npz(flip, rate, x, vx, u, ep)

        # cap training window
        M = self.cfg.max_points_for_train
        if len(flip) > M:
            flip = flip[-M:]
            rate = rate[-M:]
            x    = x[-M:]
            vx   = vx[-M:]
            u    = u[-M:]
            ep   = ep[-M:]


        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(flip, rate, x, vx, u, ep, n_at_start),
            daemon=True,
        )
        self.train_thread.start()
        self.get_logger().info("Started GP retraining thread.")
        return True



    def _train_worker(self, flip, rate, x, vx, u, ep, n_at_start: int):
        t0 = time.perf_counter()

        signals = {
            "x_pose": x,
            "linear_speed_x": vx,
            "flip": flip,
            "rate": rate,
            "u": u,
        }

        input_  = ["x_pose", "linear_speed_x", "flip", "rate", "u"]
        output_ = ["x_pose", "linear_speed_x", "flip", "rate"]

        try: 
            gps, X, Y, y_names = train_dynamics_gp_from_arrays(
                signals_new=signals,
                dt=self.cfg.ctrl_dt,
                input_keys=input_,
                output_keys=output_,
                episode_id=ep,
                target_mode="derivative",
                kernel=self.cfg.train_kernel,
                iters=self.cfg.train_iters,
                seed_npz_path=self.cfg.seed_npz_path,
                keep_seed=self.cfg.keep_seed,
            )


            paths = [self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]
            assert len(gps) == len(paths)

            # Save both models atomically
            for gp, out_path in zip(gps, paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            # Commit train size only after successful full save
            self.last_train_size = n_at_start

            elapsed = time.perf_counter() - t0
            self.get_logger().info(
                f"GP retraining finished in {elapsed:.2f}s | "
                f"N_used={len(flip)} | kernel={self.cfg.train_kernel} | iters={self.cfg.train_iters}"
            )

            # Log fingerprints once
            try:
                for d, out_path in enumerate(paths):
                    mtime = os.path.getmtime(out_path)
                    fsize = os.path.getsize(out_path)
                    self.get_logger().info(
                        f"Model[{d}] written: {out_path} | mtime={mtime:.0f} | size={fsize} bytes"
                    )
            except Exception as e:
                self.get_logger().warn(f"Could not stat model files after save: {e}")

            self.reload_pending = True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.get_logger().error(f"Retraining failed after {elapsed:.2f}s: {e}")
            self.get_logger().error(traceback.format_exc())

        finally:
            self.training = False

            

    def _reload_models_if_ready(self):
        if not self.reload_pending or self.training:
            return

        with self.model_lock:
            self.gp_flip: GPManager = GPManager.load(self.cfg.gp_flip_path)
            self.gp_rate: GPManager = GPManager.load(self.cfg.gp_rate_path)
            self.gp_pose_x: GPManager = GPManager.load(self.cfg.gp_pos_x_path)
            self.gp_vx: GPManager = GPManager.load(self.cfg.gp_vx_path)
            self.gp_flip.device = self.device
            self.gp_rate.device = self.device
            self.gp_pose_x.device = self.device
            self.gp_vx.device = self.device

        self.plan = None  # clear warm-start after model swap
        self.reload_pending = False
        self.get_logger().info("Reloaded GP models (hot swap).")

        try:
            for d, p in enumerate([self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]):
                self.get_logger().info(
                    f"Reloaded file[{d}]: {p} | mtime={os.path.getmtime(p):.0f} | size={os.path.getsize(p)} bytes"
                )
        except Exception as e:
            self.get_logger().warn(f"Could not stat model files after reload: {e}")

        if self.reset_after_retrain:
            self.reset_after_retrain = False
            self.request_reset()


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
            f"Episode {ep} time_to_goal: {dt:.3f} s | retrain_started={int(retrain_started)}"
                               )

        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, float(dt), int(retrain_started)])

        # live plot update
        self._update_live_plot(ep, float(dt), self.log_cost_j)

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
        self.wheelie_hold_steps = 0


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
