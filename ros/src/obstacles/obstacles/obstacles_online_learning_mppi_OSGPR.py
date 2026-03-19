#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

import traceback
import threading

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

# File configuration
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

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


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = cfg_params.gp.sample_time_dt
    horizon: int = 20
    num_rollouts: int = 256

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    u_min: float = -0.5
    u_max: float = 0.5

    # Target / stop conditions
    pitch_target: float = 1.45
    pitch_stop_abs: float = 2.2

    # Paths to trained SVGP models
    gp_xpos_path: str = str(GP_DIR / "models" / cfg_params.models.xpos)
    gp_xpos_dot_path: str = str(GP_DIR / "models" / cfg_params.models.xpos_dot)
    gp_pitch_path: str = str(GP_DIR / "models" / cfg_params.models.pitch)
    gp_pitch_dot_path: str = str(GP_DIR / "models" / cfg_params.models.pitch_dot)

    # ---- logging ----
    log_dir: str = str(BASE_DIR / "logs")
    max_log_points: int = 200_000

    # ---- retrain/update ----
    min_points_to_train: int = 20
    train_kernel: str = "RBF"
    train_iters: int = 300

    # training window cap
    max_points_for_train: int = 50_000
    min_new_points_between_trains: int = 5

    # SVGP warm-update steps per retrain event
    svgp_warm_steps: int = 200

    # retrain trigger
    retrain_every_episodes: int = 1

    # ---- live learning curve plot ----
    live_plot: bool = True
    live_plot_save_png: bool = True

    episode_timeout_sec: float = 20.0

    # ---- seed dataset ----
    seed_npz_path: str = str(DATA_DIR / cfg_params.files.ini_data_file)
    seed_episode_id: int = -1
    keep_seed: bool = True

    # ---- weights ----
    w_u: float = 100.1
    w_pitch = 100.1
    w_pitch_dot: float = 30.0
    w_goal: float = 400.0

    pitch_limit: float = 1.6
    w_pitch_limit: float = 80.0

    goal_x: float = 5.0
    w_xpos_dot: float = 0.5

    # --- OSGPR-style streaming update (trainable Z) ---
    osgpr_steps: int = 60
    osgpr_batch_size: int = 256
    osgpr_lr_theta: float = 1e-2     # model/likelihood params lr
    osgpr_lr_z: float = 2e-4         # inducing locations lr (SMALL)
    osgpr_anchor_beta: float = 0.05  # strength of "old posterior becomes prior"
    osgpr_z_reg: float = 1e-3        # keeps Z from drifting too much
    osgpr_freeze_hypers: bool = True # start stable; can unfreeze later

    # in case we want to use the entropy
    entropy_beta: float = 0.0
    entropy_use_log: bool = True
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True

    live_plot_mode: str = "both"

    backward_reset_dist: float = 3.0   # meters (if car goes this far backward from start -> reset)



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

        # ----- Load SVGP models -----
        # IMPORTANT: these checkpoints must be saved by SVGPManager.save()
        self.gp_xpos        = SVGPManager.load(self.cfg.gp_xpos_path, device=self.device)
        self.gp_xpos_dot    = SVGPManager.load(self.cfg.gp_xpos_dot_path, device=self.device)
        self.gp_pitch       = SVGPManager.load(self.cfg.gp_pitch_path, device=self.device)
        self.gp_pitch_dot   = SVGPManager.load(self.cfg.gp_pitch_dot_path, device=self.device)

        self.episode_x_start: Optional[float] = None

        # ----- State -----
        self.obs_xpos: Optional[float] = None
        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0
        self.last_odom_valid: bool = False

        # ----- ROS interfaces -----
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)

        self.reset_client = self.create_client(Trigger, "reset_car")
        self.resetting = False
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_car service...")

        # State from IMU
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_dot = 0.0
        self.pitch_dot = 0.0
        self.yaw_dot = 0.0


        self.last_pitch: float = 0.0
        self.last_rate: float = 0.0
        self.last_state_valid: bool = False

        # Reset / arming logic
        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0

        # Timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

        # ==========================
        # Locks + managers
        # ==========================
        self.model_lock = threading.Lock()   # protects hot-swap vs predict

        # MPPI core
        self.mppi = MPPICore(
            cfg=self.cfg,
            device=self.device,
            gp_xpos=self.gp_xpos,
            gp_xpos_dot=self.gp_xpos_dot,
            gp_pitch=self.gp_pitch,
            gp_pitch_dot=self.gp_pitch_dot,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )

        # Dataset buffer
        self.episode_id: int = 0
        self.dataset = DatasetBuffer(
            maxlen=self.cfg.max_log_points,
            log_dir=self.cfg.log_dir,
            ctrl_dt=self.cfg.ctrl_dt,
            logger=self.get_logger(),
        )

        # OSGPR warm-update retrainer
        self.retrain = OSGPRTrainableZRetrainManager(
            cfg=self.cfg,
            device=self.device,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )
        self.reset_after_retrain = False

        # Episode timing metrics
        self.episode_start_time = None
        self.episode_started = False

        # Live plot + metrics
        self.plotter = LivePlotter(
            enabled=self.cfg.live_plot,
            save_png=self.cfg.live_plot_save_png,
            out_dir=self.cfg.log_dir,
            mode = self.cfg.live_plot_mode,
            logger=self.get_logger(),
        )
        self.metrics = EpisodeMetricsWriter(
            log_dir=self.cfg.log_dir,
            plotter=self.plotter,
            logger=self.get_logger(),
        )

        self.log_cost_j = 0.0
        self._prev_x = 0.0

    # ========================================================
    # Helpers
    # ========================================================
    def _mark_episode_started(self):
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True
            self.ep_cost_sum = 0.0
            self.ep_cost_steps = 0

            # store x reference for backward reset
            if self.xpos is not None:
                self.episode_x_start = float(self.xpos)



    def imu_cb(self, msg: Imu):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        if self.t0_sim is None:
            self.t0_sim = t
        self.latest_t_sim = float(t - self.t0_sim)

        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        # fixed bug: use x, y, z correctly
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        (self.roll, 
        self.pitch, 
        self.yaw, 
        self.roll_dot, 
        self.pitch_dot, 
        self.yaw_dot) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        self.have_imu = True

    # ========================================================
    # ROS callbacks
    # ========================================================
    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        # fixed bug: use x, y, z correctly
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        (self.roll, 
        self.pitch, 
        self.yaw, 
        self.roll_dot, 
        self.pitch_dot, 
        self.yaw_dot) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        if self.waiting_post_reset:
            if self.resetting:
                self.last_state_valid = False
                return

        #     self.prev_theta = theta
        #     self.prev_theta_unwrapped = theta
        #     self.theta0 = theta

        #     self.last_flip_rel = 0.0
        #     self.last_rate = pitch_rate
        #     self.last_state_valid = True

            self.waiting_post_reset = False
            self.watchdog_fired = False
            return

        # if self.prev_theta is None:
        #     self.prev_theta = theta
        #     self.prev_theta_unwrapped = theta
        #     if self.theta0 is None:
        #         self.theta0 = theta

        # self.prev_theta, theta_unwrapped = geometry.unwrap_angle(
        #     self.prev_theta, self.prev_theta_unwrapped, theta
        # )
        # self.prev_theta_unwrapped = theta_unwrapped

        # if self.theta0 is None:
        #     self.theta0 = theta_unwrapped

        # flip_rel = theta_unwrapped - self.theta0
        # flip_rel = max(-math.pi, min(math.pi, flip_rel))

        # self.last_flip_rel = flip_rel
        # self.last_rate = pitch_rate
        self.last_state_valid = True

    def obs_callback(self, msg: PoseStamped):
        self.obs_xpos = float(msg.pose.position.x)

    def car_callback(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        
        self.last_odom_valid = True


    def _accumulate_executed_cost(self, x0_np, u_cmd, obs_xpos):
        # x0_np = [x, xdot, pitch, pitch_dot]
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        u  = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)
        if obs_xpos is None:
            obs = torch.as_tensor([1e6], dtype=torch.float32, device=self.device)
        else:
            obs = torch.as_tensor([float(obs_xpos)], dtype=torch.float32, device=self.device)

        # same cost function as planner (but evaluated at REAL state/action)
        # c = self.mppi.stage_cost_torch(x0, u, obs)  
        c = self.mppi.stage_cost_torch(x0, u)
        c = float(c.item())

        # optional: scale by dt to approximate integral
        # c *= float(self.cfg.ctrl_dt)

        self.ep_cost_sum += c
        self.ep_cost_steps += 1


    # ========================================================
    # Control loop
    # ========================================================
    def control_timer_cb(self):
        cfg = self.cfg

        # Pause MPPI while training/reloading
        if self.retrain.training or self.retrain.reload_pending:
            self.mppi.reset_plan()
            self.publish_u(0.0)

            loaded = self.retrain.reload_models_if_ready()
            if loaded is not None:
                gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot = loaded
                self.gp_xpos, self.gp_xpos_dot, self.gp_pitch, self.gp_pitch_dot = gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot
                self.mppi.set_models(gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot)

                if self.reset_after_retrain:
                    self.reset_after_retrain = False
                    self.request_reset()
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

        pitch = self.pitch #float(self.last_pitch)
        pitch_dot = self.pitch_dot #float(self.last_pitch_dot)

        if not self.last_odom_valid:
            self.publish_u(0.0)
            return

        # Backward reset condition (episode-relative)
        if self.episode_x_start is not None and self.xpos is not None:
            x_rel = float(self.xpos - self.episode_x_start)
            if x_rel <= -float(cfg.backward_reset_dist):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} BACKWARD_RESET: x_rel={x_rel:.2f}m <= -{cfg.backward_reset_dist:.2f}m"
                )

                ep_num = self.episode_id + 1
                do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

                started = False
                if do_retrain_now:
                    started = self.retrain.maybe_start_retrain_async(self.dataset, episode_id=self.episode_id, force=True)

                self._record_episode_metric(retrain_started=started, success=0)
                self.publish_u(0.0)

                if started:
                    self.reset_after_retrain = True
                    return

                self.request_reset(force=True)
                return

        # Episode timeout
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={cfg.episode_timeout_sec:.2f}s). Forcing reset."
                )

                ep_num = self.episode_id + 1
                do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

                started = False
                if do_retrain_now:
                    started = self.retrain.maybe_start_retrain_async(
                        self.dataset, episode_id=self.episode_id, force=True
                    )

                self._record_episode_metric(retrain_started=started, success=0)
                self.publish_u(0.0)

                if started:
                    self.reset_after_retrain = True
                    return

                self.request_reset(force=True)
                return

        # Emergency stop
        if abs(pitch) >= float(cfg.pitch_stop_abs):
            self.publish_u(0.0)

            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = False
            if do_retrain_now:
                started = self.retrain.maybe_start_retrain_async(self.dataset, episode_id=self.episode_id, force=True)
            else:
                self.get_logger().info(
                    f"Skipping retrain this episode (ep_num={ep_num}). retrain_every_episodes={cfg.retrain_every_episodes}"
                )

            self._record_episode_metric(retrain_started=started, success=0)

            if started:
                self.reset_after_retrain = True
                return
            self.reset_after_retrain = False
            self.request_reset()
            return

        if self.xpos is None:
            self.publish_u(0.0)
            return

        # Goal check
        if self.xpos >= float(cfg.goal_x):
            self.publish_u(0.0)

            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = self.retrain.maybe_start_retrain_async(self.dataset, episode_id=self.episode_id, force=True) if do_retrain_now else False
            self._record_episode_metric(retrain_started=started, success=1)

            if started:
                self.reset_after_retrain = True
                return
            self.request_reset()
            return

        # MPPI action
        x0 = np.array([self.xpos, self.xpos_dot, pitch, pitch_dot], dtype=np.float32)

        try:
            u_cmd = self.mppi.action(x0, self.obs_xpos)
            self.log_cost_j = float(self.mppi.last_mean_cost)

            if not math.isfinite(u_cmd):
                self.get_logger().error("u_cmd is NaN/Inf from MPPI. Forcing 0.")
                u_cmd = 0.0
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            self.get_logger().error(traceback.format_exc())
            u_cmd = 0.0


        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self._accumulate_executed_cost(x0, u_cmd, self.obs_xpos)
        self.publish_u(u_cmd)
        #self.publish_u(0.0)
        self._log_step(self.xpos, self.xpos_dot, pitch, pitch_dot, u_cmd)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)

    def _log_step(self, xpos: float, xpos_dot: float, pitch: float, pitch_dot: float, u: float):
        if self.xpos is None:
            return
        self.dataset.append_step(
            pitch=float(pitch),
            pitch_dot=float(pitch_dot),
            xpos=float(self.xpos),
            xpos_dot=float(self.xpos_dot),
            u=float(u),
            episode_id=int(self.episode_id),
        )

    def _record_episode_metric(self, retrain_started: bool, success: int):
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
            success=success
        )

        self.episode_start_time = None
        self.episode_started = False

    def _local_reset_state(self):
        # self.prev_theta = None
        # self.prev_theta_unwrapped = 0.0
        # self.theta0 = None
        self.last_state_valid = False
        self.episode_x_start = None

        self.mppi.reset_plan()

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
