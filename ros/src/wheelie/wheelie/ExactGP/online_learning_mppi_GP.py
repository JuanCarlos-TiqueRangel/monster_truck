#!/usr/bin/env python3
import math
from dataclasses import dataclass, field
from typing import Optional

import traceback
import threading

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

BASE_DIR = Path(__file__).resolve().parent   # .../ExactGP
GP_DIR   = BASE_DIR / "gp"
DATA_DIR = GP_DIR / "data"

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

from gp.gp_dynamics import GPManager
from gp.gp_retrain_manager import GPRetrainManager
from utils import geometry

from utils.mppi_core import MPPICore
from utils.dataset_buffer import DatasetBuffer
from utils.live_plot import LivePlotter
from utils.episode_metrics import EpisodeMetricsWriter


def _cfg_tuple(section, name):
    value = getattr(section, name, None)
    if value is None:
        raise ValueError(f"Missing '{name}' in ExactGP config.yaml gp section.")
    return tuple(value)


def _default_gp_model_paths():
    return {
        name: str(GP_DIR / "models" / filename)
        for name, filename in vars(cfg_params.models).items()
    }


def _unique_keys(*groups):
    return tuple(dict.fromkeys(key for group in groups for key in group))


def _default_dataset_signal_keys():
    return _unique_keys(
        _cfg_tuple(cfg_params.gp, "input_keys"),
        _cfg_tuple(cfg_params.gp, "output_keys"),
    )


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = cfg_params.gp.sample_time_dt
    horizon: int = 30
    num_rollouts: int = 2000

    # MPPI hyper-parameters
    lambda_: float = 5.0
    sigma: float = 1.6
    #sigma: float = 0.3


    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # target / stop conditions
    goal_x = 100.0
    x_min_terminate: float = -100.0
    pitch_target = math.radians(85.0)
    roll_stop_abs = math.radians(70.0)

    # Paths to trained GP velocity models. xpos and pitch are integrated in MPPI.
    gp_xpos_dot_path: str = str(GP_DIR / "models" / cfg_params.models.xpos_dot)
    gp_pitch_dot_path: str = str(GP_DIR / "models" / cfg_params.models.pitch_dot)
    gp_target_mode: str = cfg_params.gp.type_of_data
    gp_input_keys: tuple = _cfg_tuple(cfg_params.gp, "input_keys")
    gp_output_keys: tuple = _cfg_tuple(cfg_params.gp, "output_keys")
    gp_model_paths: dict = field(default_factory=_default_gp_model_paths)
    dataset_signal_keys: tuple = field(default_factory=_default_dataset_signal_keys)

    # ---- logging ----
    log_dir: str = str(BASE_DIR / "logs")
    max_log_points: int = 200_000

    # ---- retrain ----
    min_points_to_train: int = 20
    N_target_train: int = 10000
    train_kernel: str = cfg_params.gp.kernel
    train_iters: int = 300
    max_points_for_train: int = 50_000
    min_new_points_between_trains: int = 20

    # ---- live learning curve plot ----
    live_plot: bool = True
    live_plot_save_png: bool = True

    episode_timeout_sec: float = 60.0   # hard timeout for an episode (s)

    # ---- seed dataset (initial offline run) ----
    # Keep this path only if the referenced file exists in your project.
    seed_npz_path: str = str(DATA_DIR / cfg_params.files.ini_data_file)
    seed_episode_id: int = -1
    keep_seed: bool = True

    retrain_every_episodes: int = 1   # or 20

    # ---- weights for wheelie ----
    # Stationary wheelie costs
    w_x_hold: float = 20.0
    w_xdot: float = 80.0
    w_pitch: float = 200.0
    w_pitch_dot: float = 40.0
    w_pitch_region: float = 100.0
    w_u: float = 7.1
    w_du: float = 15.0
    w_pitch_hard: float = 1000.0

    # Wheelie region
    pitch_min: float = math.radians(60.0)
    pitch_max: float = math.radians(100.0)
    pitch_hard_max: float = math.radians(120.0)



    live_plot_mode: str = "both"
    just_gp_model: bool = False
    stop_re_training_mode: bool = False


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
        self.gp_pitch_dot: GPManager = GPManager.load(self.cfg.gp_pitch_dot_path, device=self.device)
        self.gp_xpos_dot: GPManager = GPManager.load(self.cfg.gp_xpos_dot_path, device=self.device)

        # ----- State -----
        self.obs_xpos: Optional[float] = None
        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0
        self.last_odom_valid: bool = False

        self.pitch_unwrapped = None

        # ----- ROS interfaces -----
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)

        self.reset_client = self.create_client(Trigger, "reset_car")
        self.resetting = False
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_car service...")

        self.flip = 0.0

        # Latest state from IMU
        self.last_state_valid: bool = False

        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0

        # Reset / arming logic
        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        # Timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

        # ==========================
        # Locks + managers
        # ==========================
        self.model_lock = threading.Lock()   # protects GP hot-swap vs predict

        self.mppi = MPPICore(
            cfg=self.cfg,
            device=self.device,
            gp_xpos_dot=self.gp_xpos_dot,
            gp_pitch_dot=self.gp_pitch_dot,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )

        # Online dataset buffer
        self.episode_id: int = 0
        self.dataset = DatasetBuffer(
            maxlen=self.cfg.max_log_points,
            log_dir=self.cfg.log_dir,
            ctrl_dt=self.cfg.ctrl_dt,
            signal_keys=self.cfg.dataset_signal_keys,
            logger=self.get_logger(),
        )

        # Retraining manager
        self.retrain = GPRetrainManager(
            cfg=self.cfg,
            device=self.device,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )
        self.reset_after_retrain = False

        # Episode timing metrics
        self.episode_start_time = None   # rclpy time
        self.episode_started = False

        # Live plot + metrics writer
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

    # ========================================================
    # Helpers: episode timing
    # ========================================================
    def _mark_episode_started(self):
        """Call right before publishing the first MPPI action of an episode."""
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True
            self.ep_cost_sum = 0.0
            self.ep_cost_steps = 0

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

        # Keep Euler roll/yaw only for diagnostics / emergency roll check.
        (
            self.roll,
            _bad_euler_pitch,
            self.yaw,
            self.roll_dot,
            _bad_euler_pitch_dot,
            self.yaw_dot,
        ) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        self.flip, _, _ = geometry.up_and_updot_from_quat_gyro(qw, qx, qy, qz, wx, wy, wz)

        # Use singularity-free wheelie pitch for the controller and GP data.
        self.pitch, self.pitch_dot = geometry.quat_to_wheelie_state(
            qw, qx, qy, qz,
            wx, wy, wz,
            prev_pitch_unwrapped=self.pitch_unwrapped,
            pitch_rate_sign=1.0,
        )

        self.pitch_unwrapped = self.pitch
        #print("[PITCH RAD]: ", self.pitch, "[flip]: ", self.flip)
        # print(" ")

        if self.waiting_post_reset:
            if self.resetting:
                self.last_state_valid = False
                return

            self.waiting_post_reset = False
            self.watchdog_fired = False
            return

        self.last_state_valid = True


    def obs_callback(self, msg: PoseStamped):
        self.obs_xpos = float(msg.pose.position.x)

    def car_callback(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        self.last_odom_valid = True

    def _accumulate_executed_cost(self, x0_np, u_cmd):
        # x0_np = [x, xpos_dot, pitch, pitch_dot]
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        u = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)

        # same cost function as planner (but evaluated at REAL state/action)
        c = self.mppi.stage_cost_torch(x0, u)  # (1,)
        c = float(c.item())

        print("[DEBUG COST]: ", c)

        # optional: scale by dt to approximate integral
        # c *= float(self.cfg.ctrl_dt)

        self.ep_cost_sum += c
        self.ep_cost_steps += 1


    def _should_reset_for_flip(self, pitch: float) -> bool:
        cfg = self.cfg

        pitch_too_far_forward = pitch <= float(cfg.pitch_fall_forward_limit)
        pitch_too_far_backward = pitch >= float(cfg.pitch_fall_backward_limit)
        roll_too_large = abs(self.roll) >= float(cfg.roll_stop_abs)

        return pitch_too_far_forward or pitch_too_far_backward or roll_too_large

    # ========================================================
    # MPPI + training orchestration
    # ========================================================
    def control_timer_cb(self):
        cfg = self.cfg

        # Pause MPPI while training/reloading
        if self.retrain.training or self.retrain.reload_pending:
            self.mppi.reset_plan()
            self.publish_u(0.0)

            loaded = self.retrain.reload_models_if_ready()
            if loaded is not None:
                gp_xpos_dot = loaded["xpos_dot"]
                gp_pitch_dot = loaded["pitch_dot"]
                self.gp_xpos_dot = gp_xpos_dot
                self.gp_pitch_dot = gp_pitch_dot
                self.mppi.set_models(gp_xpos_dot, gp_pitch_dot)

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

        pitch = float(self.pitch)
        pitch_dot = float(self.pitch_dot)

        if not self.last_odom_valid:
            self.publish_u(0.0)
            return


        if self.dataset.n_points() < cfg.N_target_train and self.cfg.just_gp_model == False:
            self.cfg.re_training_mode = True
        else:
            self.cfg.re_training_mode = False


        # Need x to proceed
        if self.xpos is None:
            self.publish_u(0.0)
            return

        # -------------------------------------------------
        # Left boundary check (failure)
        # -------------------------------------------------
        if self.xpos <= float(cfg.x_min_terminate):
            self.get_logger().warn(
                f"Episode {int(self.episode_id)} terminated: "
                f"xpos={self.xpos:.3f} m <= {cfg.x_min_terminate:.3f} m"
            )
            self.publish_u(0.0)

            # self.dataset.drop_episode(self.episode_id)
            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = False
            if do_retrain_now:
                started = self.retrain.maybe_start_retrain_async(
                    self.dataset, episode_id=self.episode_id, force=False
                )

            self._record_episode_metric(retrain_started=started, success=1)

            if started:
                self.reset_after_retrain = True
                return
            else:
                self.request_reset()
                return


        # -------------------------------------------------
        # Goal check (success)
        # -------------------------------------------------
        if self.cfg.re_training_mode:
            if self.xpos >= float(cfg.goal_x):
                self.publish_u(0.0)

                ep_num = self.episode_id + 1
                do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

                started = False
                if do_retrain_now:
                    started = self.retrain.maybe_start_retrain_async(
                        self.dataset, episode_id=self.episode_id, force=False
                    )

                self._record_episode_metric(retrain_started=started, success=1)

                if started:
                    self.reset_after_retrain = True
                    return
                else:
                    self.request_reset()
                    return
        else:
            if self.xpos >= float(cfg.goal_x):
                print("[SUCCESS] !!!")
                self.publish_u(0.0)

                self.dataset.drop_episode(self.episode_id)
                self._record_episode_metric(retrain_started=True, success=1)
                self.request_reset()
                return


        # -------------------------------------------------
        # Episode timeout
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Emergency stop: car flipped or nearly flipped
        # -------------------------------------------------
        if self.cfg.re_training_mode:
            #if self._should_reset_for_flip(pitch):
            if self.flip < -0.3:
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
        else:
            #if self._should_reset_for_flip(pitch):
            if self.flip < -0.3:
                self.get_logger().warn(
                    f"Flip detected. roll={self.roll:.3f}, pitch_rads={pitch:.3f}, pitch_deg={math.degrees(pitch):.3f}"
                )
                self.publish_u(0.0)
                self.dataset.drop_episode(self.episode_id)
                self._record_episode_metric(retrain_started=False, success=0)
                self.request_reset()
                return


        # -------------------------------------------------
        # MPPI (normal control)
        # -------------------------------------------------
        x0 = np.array([self.xpos, self.xpos_dot, pitch, pitch_dot], dtype=np.float32)

        try:
            u_cmd = self.mppi.action(x0)

            if not math.isfinite(u_cmd):
                self.get_logger().error("u_cmd is NaN/Inf coming out of MPPI. Forcing 0.")
                u_cmd = 0.0

        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            self.get_logger().error(traceback.format_exc())
            u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self._accumulate_executed_cost(x0, u_cmd)
        self.publish_u(u_cmd)
        self._log_step(pitch, pitch_dot, u_cmd)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)

    # ==========================
    # Logging
    # ==========================
    def _log_step(self, pitch: float, pitch_dot: float, u: float):
        if self.xpos is None:
            return
        signals = {
            "xpos": float(self.xpos),
            "xpos_dot": float(self.xpos_dot),
            "pitch": float(pitch),
            "pitch_dot": float(pitch_dot),
            "u": float(u),
        }
        self.dataset.append_step(
            episode_id=int(self.episode_id),
            signals=signals,
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

    # ==========================
    # Reset logic
    # ==========================
    # def _local_reset_state(self):
    #     self.last_state_valid = False

    #     # Important: reset continuous pitch memory
    #     self.pitch_unwrapped = None

    #     self.mppi.reset_plan()

    #     # reset episode timing
    #     self.episode_start_time = None
    #     self.episode_started = False

    def _local_reset_state(self):
        self.last_state_valid = False
        self.last_odom_valid = False

        # Important: reset continuous pitch memory
        self.pitch_unwrapped = None

        # Optional but useful
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.roll = 0.0
        self.roll_dot = 0.0

        self.mppi.reset_plan()

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
