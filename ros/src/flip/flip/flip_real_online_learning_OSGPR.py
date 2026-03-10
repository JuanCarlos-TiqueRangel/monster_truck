#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import traceback
import threading
import time

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


# File configuration
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GP_DIR   = BASE_DIR / "gp"

# SVGP manager (your class)
from gp.svgp_dynamics import SVGPManager

# utils modules
from utils.mppi_core import MPPICore
from utils.dataset_buffer import DatasetBuffer
from utils.osgpr_retrain_manager import OSGPRTrainableZRetrainManager
from utils.live_plot import LivePlotter
from utils.episode_metrics import EpisodeMetricsWriter
from utils.key_listener import KeyListener
from utils.geometry import quat_to_R, up_and_updot_from_quat_gyro

# ============================================================
# Config
# ============================================================
@dataclass
class MPPIConfig:
    # ---- RUN MODE ----
    # "sim": uses reset_car service, automatic resets like before
    # "real": no reset service; you manually start episodes with keyboard
    run_mode: str = "sim"   # "sim" | "real"
    #run_mode: str = "real"

    # Timing
    ctrl_dt: float = 0.02
    horizon: int = 20
    num_rollouts: int = 256

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    up_goal: float = 0.99
    updot_goal: float = 2.0

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
    gp_up_z_path: str = str(GP_DIR / "models" / cfg_params.models.up_z)
    gp_up_z_dot_path: str = str(GP_DIR / "models" / cfg_params.models.up_z_dot)

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

    episode_timeout_sec: float = 60.0

    # ---- seed dataset ----
    # seed_npz_path: str = str(DATA_DIR / "mujoco_manual_flip.npz")
    seed_npz_path: str = str(DATA_DIR / cfg_params.files.ini_data_file)
    seed_episode_id: int = -1
    keep_seed: bool = True

    # ---- weights ----
    w_up_z: float = 30.1
    w_u: float = 100.1
    w_up_z_dot: float = 1.0

    # --- OSGPR-style streaming update (trainable Z) ---
    osgpr_steps: int = 60
    osgpr_batch_size: int = 256
    osgpr_lr_theta: float = 1e-3
    osgpr_lr_z: float = 2e-4
    osgpr_anchor_beta: float = 0.1
    osgpr_z_reg: float = 1e-3
    osgpr_freeze_hypers: bool = True # freeze hypers if desired (recommended for stability)


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
        self.gp_up_z   = SVGPManager.load(self.cfg.gp_up_z_path, device=self.device)
        self.gp_up_z_dot   = SVGPManager.load(self.cfg.gp_up_z_dot_path, device=self.device)

        # ----- State -----
        self.last_odom_valid: bool = False

        # Episode gating (REAL mode)
        self.episode_active: bool = self.is_sim  # sim behaves like before; real waits for manual start
        self.pending_start: bool = False          # set True by keyboard 's' in real mode
        self.episode_x0: Optional[float] = None   # real-mode goal relative reference
        self._shutdown_requested: bool = False

        # ----- ROS interfaces -----

        if self.is_real:
            self.car_sub = self.create_subscription(Odometry, "/optitrack/odom", self.car_callback, 10)
            self.imu_sub = self.create_subscription(Imu, "/imu/data", self.imu_cb, 10)
            self.real_cmd_pub = self.create_publisher(Twist, "/cmd_real", 10)
        if self.is_sim:
            self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
            self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)

        #self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)

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
        self.last_up_z: float = 0.0
        self.last_up_z_dot: float = 0.0
        self.last_state_valid: bool = False

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

        # Timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

        # Locks + managers lock to avoid crashes and keep synchronization 
        self.model_lock = threading.Lock()

        self.mppi = MPPICore(
            cfg=self.cfg,
            device=self.device,
            gp_up_z=self.gp_up_z,
            gp_up_z_dot=self.gp_up_z_dot,
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

        self.goal_hold_steps = 0



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

    def _accumulate_executed_cost(self, x0_np, u_cmd):
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 2)
        u  = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)

        c = self.mppi.stage_cost_torch(x0, u)
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


    def _log_step(self, up_z: float, up_z_dot: float, u: float):
        # Only log while an episode is actually active (important for real mode)
        if self.is_real and not self.episode_active:
            return

        self.dataset.append_step(
            up_z=float(up_z),
            up_z_dot=float(up_z_dot),
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
        self.last_state_valid = False

        self.mppi.reset_plan()

        # Episode metrics
        self.episode_start_time = None
        self.episode_started = False
        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0
        self.episode_x0 = None


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
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        self.up_z, self.up_z_dot, _   = up_and_updot_from_quat_gyro(qw, qx, qy, qz, wx, wy, wz)

        if self.waiting_post_reset:
            # For SIM: ignore until reset done; For REAL: just "alignment gate"
            if self.is_sim and self.resetting:
                self.last_state_valid = False
                return

            # Require upright-ish before we "arm" (same behavior as your sim gate)
            if self.up_z > 0.8:
                self.last_state_valid = False
                return

            self.last_up_z = 0.0
            self.last_up_z_dot = self.up_z_dot
            self.last_state_valid = True
            self.waiting_post_reset = False
            self.watchdog_fired = False
            return

        self.last_up_z = self.up_z
        self.last_up_z_dot = self.up_z_dot
        self.last_state_valid = True


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
            self.goal_hold_steps = 0
            self.publish_u(0.0)

            loaded = self.retrain.reload_models_if_ready()
            if loaded is not None:
                self.model_reload_count += 1
                self.get_logger().info(f"Model reload count: {self.model_reload_count} (now using updated SVGP models)")

                gp_up_z, gp_up_z_dot = loaded
                self.gp_up_z, self.gp_up_z_dot = gp_up_z, gp_up_z_dot
                self.mppi.set_models(gp_up_z, gp_up_z_dot)

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
                self.goal_hold_steps = 0
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        # REAL mode: handle manual start gating (always hold u=0 until started)
        if self.is_real:
            # If user asked to start, wait until sensors are valid, then activate.
            if self.pending_start:

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
                self.goal_hold_steps = 0
                self.publish_u(0.0)
                return

        # Need valid attitude source
        if self.is_sim:
            if not self.last_state_valid:
                if not self.warned_no_imu:
                    self.get_logger().warn("Waiting for first IMU message...")
                    self.warned_no_imu = True
                self.goal_hold_steps = 0
                self.publish_u(0.0)
                return
        else:
            # REAL: require fresh /imu/rpy
            rpy_fresh = self.rpy_valid and ((time.time() - self.last_rpy_time) <= 0.30)
            if not rpy_fresh:
                if not self.warned_no_imu:
                    self.get_logger().warn("Waiting for fresh /imu/rpy ...")
                    self.warned_no_imu = True
                self.goal_hold_steps = 0
                self.publish_u(0.0)
                return
        
        self.warned_no_imu = False

        if self.is_sim:
            up_z = float(self.last_up_z)
            up_z_dot = float(self.last_up_z_dot)
        if self.is_real:
            up_z = float(self.pitch)
            up_z_dot = float(self.car_pitch_aspeed)

        # Episode timeout
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(cfg.episode_timeout_sec):
                self.end_episode(reason=f"TIMEOUT ({elapsed_ep:.2f}s)", allow_retrain=True)
                return

        #if (up_z >= cfg.up_goal) and (abs(up_z_dot) <= cfg.updot_goal):
        if up_z >= cfg.up_goal:
            self.end_episode(
                reason=f"GOAL_REACHED (up_z={up_z:.3f}, up_z_dot={up_z_dot:.3f})",
                allow_retrain=True,
            )
            return

        # MPPI action
        x0 = np.array([up_z, up_z_dot], dtype=np.float32)


        try:
            t0 = time.perf_counter()
            u_cmd = self.mppi.action(x0)
            # torch.cuda.synchronize()
            # dt_ms = 1e3 * (time.perf_counter() - t0)
            # #self.get_logger().info(f"MPPI action time: {dt_ms:.3f} ms")
            if not math.isfinite(u_cmd):
                self.get_logger().error("u_cmd is NaN/Inf from MPPI. Forcing 0.")
                u_cmd = 0.0
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            self.get_logger().error(traceback.format_exc())
            u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self._accumulate_executed_cost(x0, u_cmd)
        self.publish_u(u_cmd)
        #self.publish_u(0.0)
        self._log_step(up_z, up_z_dot, u_cmd)


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
