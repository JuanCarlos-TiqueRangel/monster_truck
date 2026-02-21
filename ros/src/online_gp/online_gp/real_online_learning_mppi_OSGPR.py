#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import traceback
import threading
import sys
import select
import time

from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from nav_msgs.msg import Odometry

from queue import SimpleQueue, Empty


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GP_DIR   = BASE_DIR / "gp"

from utils import geometry

# SVGP manager (your class)
from gp.svgp_dynamics import SVGPManager

# utils modules
from utils.mppi_core import MPPICore
from utils.dataset_buffer import DatasetBuffer
from utils.osgpr_retrain_manager import OSGPRTrainableZRetrainManager
from utils.live_plot import LivePlotter
from utils.episode_metrics import EpisodeMetricsWriter
from utils.key_listener import KeyListener

# ============================================================
# Config
# ============================================================
@dataclass
class MPPIConfig:
    # ---- RUN MODE ----
    # "sim": uses reset_car service, automatic resets like before
    # "real": no reset service; you manually start episodes with keyboard
    #run_mode: str = "sim"   # "sim" | "real"
    run_mode: str = "real"

    # Timing
    ctrl_dt: float = 0.1
    horizon: int = 30
    num_rollouts: int = 2000

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    # u_min: float = -1.0
    # u_max: float = 1.0

    u_min: float = -0.2
    u_max: float = 0.2

    # Target / stop conditions
    pitch_target: float = 1.3
    flip_stop_abs: float = 1.6

    # Goal definition:
    # - sim: absolute x >= goal_x (like before)
    # - real: relative distance since episode start >= goal_x (prevents instant re-goal)
    goal_x: float = 0.0  # placeholder, set in __post_init__

    def __post_init__(self):
        rm = str(self.run_mode).strip().lower()
        if rm == "sim":
            self.goal_x = 5.0
        elif rm == "real":
            self.goal_x = 2.5
        else:
            raise ValueError(f"Unknown run_mode: {self.run_mode}")

    # Paths to trained SVGP models
    gp_flip_path: str = str(GP_DIR / "models" / "svgp_dynamics_flip_d_dt.pt")
    gp_rate_path: str = str(GP_DIR / "models" / "svgp_dynamics_rate_d_dt.pt")
    gp_pos_x_path: str = str(GP_DIR / "models" / "svgp_dynamics_x_pose_d_dt.pt")
    gp_vx_path: str = str(GP_DIR / "models" / "svgp_dynamics_linear_speed_x_d_dt.pt")

    # ---- logging ----
    log_dir: str = str(BASE_DIR / "logs")
    max_log_points: int = 200_000

    # ---- retrain/update ----
    min_points_to_train: int = 5
    max_points_for_train: int = 50_000
    min_new_points_between_trains: int = 5
    retrain_every_episodes: int = 1
    svgp_warm_steps: int = 200

    # ---- live learning curve plot ----
    live_plot: bool = True
    live_plot_save_png: bool = True
    live_plot_mode: str = "both"

    episode_timeout_sec: float = 20.0

    entropy_beta: float = 0.0
    entropy_use_log: bool = True
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True

    # ---- seed dataset ----
    seed_npz_path: str = str(DATA_DIR / "mujoco_manual_wheelie.npz")
    seed_episode_id: int = -1
    keep_seed: bool = True

    # ---- obstacle trigger ----
    obs_trigger_dist: float = 1.0
    obs_trigger_smooth: float = 0.20

    # ---- weights ----
    w_flat: float = 6.0
    w_wheelie: float = 1.5
    w_u: float = 100.1
    w_rate: float = 2.0

    pitch_limit: float = 1.45
    w_pitch_limit: float = 80.0

    wheelie_hold_time_sec: float = 0.5
    wheelie_pitch_tol: float = 0.15
    wheelie_rate_tol: float = 1.0

    w_goal: float = 3.0
    w_vx: float = 0.5
    v_des: float = 0.0

    wheelie_hold_sigma_scale: float = 0.25
    wheelie_hold_extra_wheelie: float = 2.0

    # --- OSGPR-style streaming update (trainable Z) ---
    osgpr_steps: int = 60
    osgpr_batch_size: int = 2048
    osgpr_lr_theta: float = 1e-2
    osgpr_lr_z: float = 2e-4
    osgpr_anchor_beta: float = 0.05
    osgpr_z_reg: float = 1e-3
    osgpr_freeze_hypers: bool = True


# ============================================================
# MPPI Controller Node
# ============================================================
class MPPICarControllerNode(Node):
    def __init__(self, cfg: MPPIConfig):
        super().__init__("mppi_car_controller")
        self.cfg = cfg

        self.run_mode = str(cfg.run_mode).strip().lower()
        self.is_sim = (self.run_mode == "sim")
        self.is_real = (self.run_mode == "real")

        # ----- Device -----
        self.device = torch.device("cuda")
        self.get_logger().info(f"Using torch device: {self.device}")
        self.get_logger().info(f"RUN MODE: {'SIMULATION' if self.is_sim else 'REAL ROBOT'}")

        # ----- Load SVGP models -----
        self.gp_pose_x = SVGPManager.load(self.cfg.gp_pos_x_path, device=self.device)
        self.gp_vx     = SVGPManager.load(self.cfg.gp_vx_path, device=self.device)
        self.gp_flip   = SVGPManager.load(self.cfg.gp_flip_path, device=self.device)
        self.gp_rate   = SVGPManager.load(self.cfg.gp_rate_path, device=self.device)

        # ----- State -----
        self.obs_pos_x: Optional[float] = None
        self.car_pos_x: Optional[float] = None
        self.car_vx: float = 0.0
        self.last_odom_valid: bool = False
        self.wheelie_hold_steps = 0

        # Episode gating (REAL mode)
        self.episode_active: bool = self.is_sim  # sim behaves like before; real waits for manual start
        self.pending_start: bool = False          # set True by keyboard 's' in real mode
        self.episode_x0: Optional[float] = None   # real-mode goal relative reference
        self._shutdown_requested: bool = False

        # ----- ROS interfaces -----

        if self.is_real:
            self.car_sub = self.create_subscription(Odometry, "/optitrack/odom", self.car_callback, 10)
            self.imu_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 10)
            self.imu_rpy_sub = self.create_subscription(Vector3, "/imu/rpy", self.imu_rpy_cb, 10)
            self.real_cmd_pub = self.create_publisher(Twist, "/cmd_real", 10)
        if self.is_sim:
            self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)
            self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
            self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)

        self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)

        # Reset service (SIM only)
        self.reset_client = None
        self.resetting = False

        if self.is_sim:
            self.reset_client = self.create_client(Trigger, "reset_car")
            while not self.reset_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for reset_car service...")
        else:
            self.get_logger().info("Real mode: reset_car service is disabled (manual episodes).")

        # Latest state from IMU
        self.last_flip_rel: float = 0.0
        self.last_rate: float = 0.0
        self.last_state_valid: bool = False
        self.rpy_valid: bool = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.car_pitch_aspeed: float = 0.0


        # For computing flip_rel from quaternion
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        # Reset / arming logic (reused for "episode alignment" in real mode)
        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        # Episode metrics
        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0
        self.episode_start_time = None
        self.episode_started = False

        self.last_rpy_time = 0.0

        # Timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

        # Locks + managers
        self.model_lock = threading.Lock()

        self.mppi = MPPICore(
            cfg=self.cfg,
            device=self.device,
            gp_pose_x=self.gp_pose_x,
            gp_vx=self.gp_vx,
            gp_flip=self.gp_flip,
            gp_rate=self.gp_rate,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )

        self.episode_id: int = 0
        self.dataset = DatasetBuffer(
            maxlen=self.cfg.max_log_points,
            log_dir=self.cfg.log_dir,
            ctrl_dt=self.cfg.ctrl_dt,
            logger=self.get_logger(),
        )

        self.retrain = OSGPRTrainableZRetrainManager(
            cfg=self.cfg,
            device=self.device,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )
        self.reset_after_retrain = False  # sim-only behavior

        self.plotter = LivePlotter(
            enabled=self.cfg.live_plot,
            save_png=self.cfg.live_plot_save_png,
            out_dir=self.cfg.log_dir,
            mode=self.cfg.live_plot_mode,
            logger=self.get_logger(),
        )
        self.metrics = EpisodeMetricsWriter(
            log_dir=self.cfg.log_dir,
            plotter=self.plotter,
            logger=self.get_logger(),
        )

        self.log_cost_j = 0.0

        # Keyboard help + listener (THREAD-SAFE)
        self._print_key_help()

        self._key_q = SimpleQueue()

        # Keyboard thread: ONLY enqueue keys (never call _on_key here)
        self.key_listener = KeyListener(lambda ch: self._key_q.put(ch), logger=self.get_logger())
        self.key_listener.start()

        # Main thread: process keys safely (Tk/matplotlib is OK here)
        self.key_timer = self.create_timer(0.05, self._process_keys_mainthread)


        # If real: keep stopped until manual start
        if self.is_real:
            self.publish_u(0.0)

        self.retrain_started_count = 0
        self.model_reload_count = 0
        self.last_reload_stats = None  # optional: file mtimes/sizes



    # ========================================================
    # Keyboard handling
    # ========================================================
    def _print_key_help(self):
        if self.is_real:
            self.get_logger().info("Keys: [s]=start episode | [x]=stop episode | [q]=quit (always sends u=0)")
        else:
            self.get_logger().info("Keys: [r]=reset sim | [q]=quit")

    def _on_key(self, ch: str):
        ch = (ch or "").strip().lower()
        if not ch:
            return

        if ch == "q":
            self.get_logger().warn("Quit requested. Sending u=0 and shutting down.")
            self._shutdown_requested = True
            return

        if self.is_real:
            if ch == "s":
                self.request_manual_start()
            elif ch == "x":
                self.end_episode(reason="MANUAL_STOP", allow_retrain=True)
        else:
            if ch == "r":
                self.get_logger().info("Manual reset requested (sim).")
                self.request_reset(force=True)

    def request_manual_start(self):
        """REAL mode: arm a new episode start (no reset service)."""
        if self.is_sim:
            self.get_logger().warn("request_manual_start() is intended for real mode.")
            return

        if self.episode_active or self.pending_start:
            self.get_logger().info("Episode already active (or pending start).")
            return

        self.episode_id += 1
        self.pending_start = True
        self._local_reset_state()

        # reuse the "post reset alignment" to re-zero theta0, etc.
        self.waiting_post_reset = True
        self.post_reset_start_time = self.get_clock().now()
        self.watchdog_fired = False

        self.get_logger().info(f"[REAL] Episode {self.episode_id} pending start. Waiting for valid IMU+odom...")


    def _process_keys_mainthread(self):
        while True:
            try:
                ch = self._key_q.get_nowait()
            except Empty:
                break
            self._on_key(ch)   # runs on ROS/main thread now


    # ========================================================
    # Helpers
    # ========================================================
    def _mark_episode_started(self):
        """Called once at the beginning of an active episode."""
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True
            self.ep_cost_sum = 0.0
            self.ep_cost_steps = 0

            # Real mode: store x0 for relative goal
            if self.is_real and self.car_pos_x is not None:
                self.episode_x0 = float(self.car_pos_x)

    def _accumulate_executed_cost(self, x0_np, u_cmd, obs_pos_x):
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        u  = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)

        if obs_pos_x is None:
            obs = torch.as_tensor([1e6], dtype=torch.float32, device=self.device)
        else:
            obs = torch.as_tensor([float(obs_pos_x)], dtype=torch.float32, device=self.device)

        c = self.mppi.stage_cost_torch(x0, u, obs)
        self.ep_cost_sum += float(c.item())
        self.ep_cost_steps += 1

    def publish_u(self, u: float):
        msg = Float32()
        real_msg = Twist()

        if self.is_sim:
            msg.data = float(u)
            self.cmd_pub.publish(msg)
        if self.is_real:
            real_msg.linear.x = float(u)
            real_msg.angular.z = 0.1
            self.real_cmd_pub.publish(real_msg)


    def _log_step(self, flip_rel: float, rate: float, u: float):
        if self.car_pos_x is None:
            return
        # Only log while an episode is actually active (important for real mode)
        if self.is_real and not self.episode_active:
            return

        self.dataset.append_step(
            flip_rel=float(flip_rel),
            rate=float(rate),
            x=float(self.car_pos_x),
            vx=float(self.car_vx),
            u=float(u),
            episode_id=int(self.episode_id),
        )

    def _record_episode_metric(self, retrain_started: bool):
        if self.episode_start_time is None:
            self.episode_started = False
            return

        dt = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
        ep = int(self.episode_id)
        avg_cost = self.ep_cost_sum / max(1, self.ep_cost_steps)

        self.metrics.write(
            episode=ep,
            time_to_goal_sec=float(dt),
            retrain_started=bool(retrain_started),
            cost=float(avg_cost),
        )

        self.episode_start_time = None
        self.episode_started = False

    def _local_reset_state(self):
        # Reset IMU unwrapping + references
        self.prev_theta = None
        self.prev_theta_unwrapped = 0.0
        self.theta0 = None
        self.last_state_valid = False

        self.mppi.reset_plan()

        # Episode metrics
        self.episode_start_time = None
        self.episode_started = False
        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0

        self.wheelie_hold_steps = 0
        self.episode_x0 = None

        self.rpy_valid = False
        self.last_rpy_time = 0.0


    # ========================================================
    # Episode end behavior (SIM vs REAL)
    # ========================================================
    def end_episode(self, reason: str, allow_retrain: bool):
        """
        Ends current episode safely:
          - publish u=0
          - optionally trigger retrain
          - SIM: can reset car (like before)
          - REAL: stays stopped until manual 's'
        """
        self.get_logger().warn(f"Episode end: {reason} (mode={'sim' if self.is_sim else 'real'})")
        self.publish_u(0.0)

        # Decide if we retrain now (same logic as before)
        ep_num = self.episode_id  # already the current episode id
        do_retrain_now = (
            allow_retrain
            and (self.cfg.retrain_every_episodes > 0)
            and (ep_num % self.cfg.retrain_every_episodes == 0)
        )

        started = False
        if do_retrain_now:
            started = self.retrain.maybe_start_retrain_async(self.dataset, episode_id=self.episode_id, force=True)

        self._record_episode_metric(retrain_started=started)

        if started:
            self.retrain_started_count += 1
            self.get_logger().info(f"Retrain started count: {self.retrain_started_count}")

        # Stop episode in real mode (stay idle)
        if self.is_real:
            self.episode_active = False
            self.pending_start = False
            self.waiting_post_reset = False
            self.reset_after_retrain = False
            return

        # Sim behavior: if retrain started, reset after reload; else reset now
        self.episode_active = True  # sim stays active generally
        if started:
            self.reset_after_retrain = True
            return

        self.reset_after_retrain = False
        self.request_reset()

    # ========================================================
    # ROS callbacks
    # ========================================================
    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        R, _ = geometry.quat_to_R_and_pitch(qw, qx, qy, qz)
        up_z = R[2, 2]
        theta = math.atan2(R[0, 2], R[2, 2])
        pitch_rate = float(msg.angular_velocity.y)

        if self.waiting_post_reset:
            # For SIM: ignore until reset done; For REAL: just "alignment gate"
            if self.is_sim and self.resetting:
                self.last_state_valid = False
                return

            # Require upright-ish before we "arm" (same behavior as your sim gate)
            if up_z < 0.8:
                self.last_state_valid = False
                return

            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            self.theta0 = theta

            self.last_flip_rel = 0.0
            self.last_rate = pitch_rate
            self.last_state_valid = True

            self.waiting_post_reset = False
            self.watchdog_fired = False
            return

        if self.prev_theta is None:
            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            if self.theta0 is None:
                self.theta0 = theta

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

    def imu_rpy_cb(self, msg: Vector3):
        self.roll = msg.x
        self.pitch = msg.y
        self.yaw = msg.z
        self.rpy_valid = True
        self.last_rpy_time = time.time()
        if self.is_real and self.waiting_post_reset:
            self.waiting_post_reset = False



    def obs_callback(self, msg: PoseStamped):
        self.obs_pos_x = float(msg.pose.position.x)

    def car_callback(self, msg: Odometry):
        self.car_pos_x = float(msg.pose.pose.position.x)
        self.car_vx = float(msg.twist.twist.linear.x)
        self.car_pitch_aspeed = float(msg.twist.twist.angular.y)
        self.last_odom_valid = True

    # ========================================================
    # SIM reset
    # ========================================================
    def request_reset(self, force: bool = False):
        if not self.is_sim:
            self.get_logger().warn("request_reset() called in real mode; ignoring.")
            return
        if self.reset_client is None:
            self.get_logger().error("reset_client is None in sim mode (unexpected).")
            return

        if self.resetting:
            return
        if self.waiting_post_reset and not force:
            return

        self.resetting = True
        self.waiting_post_reset = True
        self.post_reset_start_time = None
        self.watchdog_fired = False

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

    # ========================================================
    # Control loop
    # ========================================================
    def control_timer_cb(self):
        cfg = self.cfg

        # Shutdown request
        if self._shutdown_requested:
            self.publish_u(0.0)
            rclpy.shutdown()
            return

        # Pause MPPI while training/reloading
        if self.retrain.training or self.retrain.reload_pending:
            self.mppi.reset_plan()
            self.wheelie_hold_steps = 0
            self.publish_u(0.0)

            loaded = self.retrain.reload_models_if_ready()
            if loaded is not None:
                self.model_reload_count += 1
                self.get_logger().info(f"Model reload count: {self.model_reload_count} (now using updated SVGP models)")

                gp_pose_x, gp_vx, gp_flip, gp_rate = loaded
                self.gp_pose_x, self.gp_vx, self.gp_flip, self.gp_rate = gp_pose_x, gp_vx, gp_flip, gp_rate
                self.mppi.set_models(gp_pose_x, gp_vx, gp_flip, gp_rate)

                # SIM only: after retrain, we reset the sim to start clean
                if self.is_sim and self.reset_after_retrain:
                    self.reset_after_retrain = False
                    self.request_reset()
            return

        # Watchdog only makes sense in SIM mode (because it can force reset)
        if self.is_sim and self.waiting_post_reset and (not self.resetting) and (self.post_reset_start_time is not None):
            elapsed = (self.get_clock().now() - self.post_reset_start_time).nanoseconds * 1e-9
            if (elapsed > 3.0) and (not self.watchdog_fired):
                self.watchdog_fired = True
                self.get_logger().warn("Stuck in waiting_post_reset. Forcing reset retry.")
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        # REAL mode: handle manual start gating (always hold u=0 until started)
        if self.is_real:
            # If user asked to start, wait until sensors are valid, then activate.
            if self.pending_start:
                rpy_fresh = self.rpy_valid and ((time.time() - self.last_rpy_time) <= 0.30)

                # Need fresh RPY + valid odom
                if (not rpy_fresh) or (not self.last_odom_valid):
                    self.publish_u(0.0)
                    return

                # Start episode now
                self.pending_start = False
                self.episode_active = True
                self.waiting_post_reset = False   # <-- REAL: do not gate on /imu/data alignment
                self._mark_episode_started()
                self.get_logger().info(f"[REAL] Episode {self.episode_id} STARTED. x0={self.episode_x0}")
                self.mppi.reset_plan()

            # If not active -> keep stopped
            if not self.episode_active:
                self.publish_u(0.0)
                return


        # SIM mode: keep previous behavior gates
        if self.is_sim:
            if self.resetting or self.waiting_post_reset:
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                return

        # Need valid attitude source
        if self.is_sim:
            if not self.last_state_valid:
                if not self.warned_no_imu:
                    self.get_logger().warn("Waiting for first IMU message...")
                    self.warned_no_imu = True
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                return
        else:
            # REAL: require fresh /imu/rpy
            rpy_fresh = self.rpy_valid and ((time.time() - self.last_rpy_time) <= 0.30)
            if not rpy_fresh:
                if not self.warned_no_imu:
                    self.get_logger().warn("Waiting for fresh /imu/rpy ...")
                    self.warned_no_imu = True
                self.wheelie_hold_steps = 0
                self.publish_u(0.0)
                return
        
        self.warned_no_imu = False

        # Need valid odom
        if not self.last_odom_valid:
            self.publish_u(0.0)
            return

        if self.is_sim:
            flip_rel = float(self.last_flip_rel)
            rate = float(self.last_rate)
        if self.is_real:
            flip_rel = float(self.pitch)
            rate = float(self.car_pitch_aspeed)

        # Obstacle distance
        if (self.obs_pos_x is None) or (self.car_pos_x is None):
            d_obs = 1e6
        elif self.obs_pos_x <= self.car_pos_x:
            d_obs = 1e6
        else:
            d_obs = float(self.obs_pos_x - self.car_pos_x)

        # Episode timeout
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(cfg.episode_timeout_sec):
                self.end_episode(reason=f"TIMEOUT ({elapsed_ep:.2f}s)", allow_retrain=True)
                return

        # Emergency stop (flip too large)
        if abs(flip_rel) >= float(cfg.flip_stop_abs):
            self.end_episode(reason="EMERGENCY_STOP (flip too large)", allow_retrain=True)
            return

        # Wheelie hold logic
        hold_steps_needed = max(1, int(round(float(cfg.wheelie_hold_time_sec) / float(cfg.ctrl_dt))))
        close = (d_obs < float(cfg.obs_trigger_dist))
        err = (flip_rel - float(cfg.pitch_target) + math.pi) % (2.0 * math.pi) - math.pi

        if close and (abs(err) < float(cfg.wheelie_pitch_tol)) and (abs(rate) < float(cfg.wheelie_rate_tol)):
            self.wheelie_hold_steps += 1
        else:
            self.wheelie_hold_steps = 0

        holding = close and (self.wheelie_hold_steps >= hold_steps_needed)
        if holding and self.wheelie_hold_steps == hold_steps_needed:
            self.mppi.reset_plan()

        if self.car_pos_x is None:
            self.publish_u(0.0)
            return

        # Goal check
        if self.is_real:
            # relative goal to avoid instant success next episode
            if self.episode_x0 is None:
                self.episode_x0 = float(self.car_pos_x)
            x_rel = float(self.car_pos_x - self.episode_x0)
            if x_rel >= float(cfg.goal_x):
                # IMPORTANT: real car must stop throttle at goal
                self.end_episode(reason=f"GOAL_REACHED (x_rel={x_rel:.2f}m)", allow_retrain=True)
                return
        else:
            if self.car_pos_x >= float(cfg.goal_x):
                self.end_episode(reason=f"GOAL_REACHED (x={self.car_pos_x:.2f})", allow_retrain=True)
                return

        # MPPI action
        x0 = np.array([self.car_pos_x, self.car_vx, flip_rel, rate], dtype=np.float32)

        old_sigma     = cfg.sigma
        old_w_wheelie = cfg.w_wheelie
        old_w_flat    = cfg.w_flat

        try:
            if holding:
                cfg.sigma     = old_sigma * float(cfg.wheelie_hold_sigma_scale)
                cfg.w_wheelie = old_w_wheelie * float(cfg.wheelie_hold_extra_wheelie)
                cfg.w_flat    = old_w_flat * 0.5

            try:
                u_cmd = self.mppi.action(x0, self.obs_pos_x)
                self.log_cost_j = float(self.mppi.last_mean_cost)

                if not math.isfinite(u_cmd):
                    self.get_logger().error("u_cmd is NaN/Inf from MPPI. Forcing 0.")
                    u_cmd = 0.0
            except Exception as e:
                self.get_logger().error(f"MPPI error: {e}")
                self.get_logger().error(traceback.format_exc())
                u_cmd = 0.0

        finally:
            cfg.sigma     = old_sigma
            cfg.w_wheelie = old_w_wheelie
            cfg.w_flat    = old_w_flat

        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self._accumulate_executed_cost(x0, u_cmd, self.obs_pos_x)
        self.publish_u(u_cmd)
        self._log_step(flip_rel, rate, u_cmd)


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
        # stop key listener if present
        try:
            node.key_listener.stop()
            node.key_listener.join(timeout=1.0)  # IMPORTANT: wait for cleanup
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
