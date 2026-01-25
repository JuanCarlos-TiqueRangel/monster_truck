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
    num_rollouts: int = 2000

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    pitch_target: float = -1.5 #math.pi
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
    
    episode_timeout_sec: float = 20.0   # hard timeout for an episode (s)
    
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
            if up_z > -0.8:
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

        # ---------------------------
        # Normal episode logic
        # ---------------------------
        self.prev_theta, theta_unwrapped = self.unwrap_angle(
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


    # ========================================================
    # Torch helpers: angdiff, stage cost, GP step
    # ========================================================
    @staticmethod
    def angdiff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pitch = states[:, 0]
        u = actions
        err = self.angdiff_torch(pitch, self.pitch_target_t)
        cost_pitch = 100.0 * err ** 2
        cost_u = 0.01 * u ** 2
        return cost_u + cost_pitch


    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        X = torch.stack([states[:, 0], states[:, 1], actions], dim=-1)

        # lock so reload can't happen mid-predict
        with self.model_lock:
            # If entropy is OFF, avoid variance work entirely
            if self.cfg.entropy_beta <= 0.0:
                d_flip_mean, _ = self.gp_flip.predict_torch(X)
                d_rate_mean, _ = self.gp_rate.predict_torch(X)
                d_flip_var = None
                d_rate_var = None
            else:
                d_flip_mean, d_flip_var = self.gp_flip.predict_torch(X)  # var (NOT std)
                d_rate_mean, d_rate_var = self.gp_rate.predict_torch(X)  # var (NOT std)

        dt = float(self.cfg.ctrl_dt)

        # propagate mean dynamics
        next_states = torch.empty_like(states)
        next_states[:, 0] = states[:, 0] + d_flip_mean * dt
        next_states[:, 1] = states[:, 1] + d_rate_mean * dt
        next_states[:, 0].clamp_(-math.pi, math.pi)
        next_states[:, 1].clamp_(-20.0, 20.0)

        # If entropy OFF -> return immediately (no extra compute)
        if self.cfg.entropy_beta <= 0.0:
            return next_states, None

        # --------------------------
        # Entropy compute (only if ON)
        # --------------------------
        d_flip_var = torch.clamp(d_flip_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        d_rate_var = torch.clamp(d_rate_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)

        # variance of next state after integration: Var[x + dt*Δ] = dt^2 Var[Δ]
        if self.cfg.entropy_dt_scale:
            var_next = torch.stack([d_flip_var, d_rate_var], dim=-1) * (dt * dt)
        else:
            var_next = torch.stack([d_flip_var, d_rate_var], dim=-1)

        # entropy proxy (drop constants): 0.5 * sum log(var)
        if self.cfg.entropy_use_log:
            entropy = 0.5 * torch.log(var_next).sum(dim=-1)    # shape (K,)
        else:
            entropy = var_next.sum(dim=-1)                     # shape (K,)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return next_states, entropy




    # ========================================================
    # MPPI core
    # ========================================================
    @torch.no_grad()
    def mppi_action(self, x0_np):
        cfg = self.cfg
        H = cfg.horizon
        K = cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        assert x0.shape == (2,)

        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        eps = torch.randn(K, H, device=self.device) * cfg.sigma
        U = torch.clamp(u_init.unsqueeze(0) + eps, cfg.u_min, cfg.u_max)

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        beta = float(self.cfg.entropy_beta)

        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t)
            states, ent = self.gp_step_batch_torch(states, u_t)
            
            if ent is not None:
                stage = stage - beta * ent  # maximize entropy

            costs = costs + stage

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

        # Episode timeout: if we started sending actions but haven't flipped, force reset
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(self.cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={self.cfg.episode_timeout_sec:.2f}s). Forcing reset."
                )

                # record metric at timeout (no retrain triggered by timeout)
                self._record_episode_metric(retrain_started=False)

                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        # -----------------------------
        # Stop condition: flip done -> retrain, then reset AFTER reload
        # -----------------------------
        if abs(flip_rel) >= self.cfg.flip_stop_abs:
            self.publish_u(0.0)

            ep_num = self.episode_id + 1  # 1-based
            do_retrain_now = (self.cfg.retrain_every_episodes > 0) and \
                            (ep_num % self.cfg.retrain_every_episodes == 0)

            started = False
            if do_retrain_now:
                force = bool(self.cfg.retrain_force_every_N_episodes)
                started = self._start_retrain_async(force=force)  # <-- important
            else:
                self.get_logger().info(
                    f"Skipping retrain this episode (ep_num={ep_num}). "
                    f"retrain_every_episodes={self.cfg.retrain_every_episodes}"
                )

            self._record_episode_metric(retrain_started=started)

            if started:
                self.request_reset()
                return
            else:
                self.request_reset()
                return

        # -----------------------------
        # MPPI
        # -----------------------------
        x0 = np.array([flip_rel, rate], dtype=np.float32)
        try:
            u_cmd = self.mppi_action(x0)
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, self.cfg.u_min, self.cfg.u_max))

        # ✅ episode starts exactly when first MPPI action is sent
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
