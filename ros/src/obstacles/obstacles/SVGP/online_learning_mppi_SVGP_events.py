#!/usr/bin/env python3
import math
import traceback
import threading
import inspect
from dataclasses import dataclass
from typing import Optional

from pathlib import Path
import sys

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32, Int64
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GP_DIR   = BASE_DIR / "gp"

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

from gp.svgp_dynamics import SVGPManager
from gp.svgp_retrain_manager import GPRetrainManager
from utils import geometry
from utils.mppi_core import MPPICore
from utils.dataset_buffer import DatasetBuffer
from utils.live_plot import LivePlotter
from utils.episode_metrics import EpisodeMetricsWriter


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    ctrl_dt: float = cfg_params.gp.sample_time_dt
    horizon: int = 30
    num_rollouts: int = 2000

    lambda_: float = 500
    sigma: float = 0.3

    u_min: float = 0.0
    u_max: float = 1.0
    u_max_start: float = 0.3
    u_max_ramp_episodes: int = 15

    goal_x: float = 5.0
    pitch_target: float = 1.0
    pitch_stop_abs: float = 1.5
    roll_stop_abs: float = 1.2
    x_min_terminate: float = -3.0
    episode_timeout_sec: float = 200.0

    gp_xpos_path: str = str(GP_DIR / "models" / cfg_params.models.xpos)
    gp_xpos_dot_path: str = str(GP_DIR / "models" / cfg_params.models.xpos_dot)
    gp_pitch_path: str = str(GP_DIR / "models" / cfg_params.models.pitch)
    gp_pitch_dot_path: str = str(GP_DIR / "models" / cfg_params.models.pitch_dot)

    log_dir: str = str(BASE_DIR / "logs")
    max_log_points: int = 200_000

    min_points_to_train: int = 60
    N_target_train: int = 100000
    train_kernel: str = cfg_params.gp.kernel
    train_iters: int = cfg_params.gp.iterations
    train_lr: float = cfg_params.gp.learning_rate
    train_num_inducing: int = cfg_params.gp.num_inducing
    train_batch_size: int | None = cfg_params.gp.batch_size
    gp_target_mode: str = cfg_params.gp.type_of_data
    min_new_points_between_trains: int = 20

    live_plot: bool = True
    live_plot_save_png: bool = True
    live_plot_mode: str = "both"

    seed_npz_path: str = str(DATA_DIR / cfg_params.files.ini_data_file)
    seed_episode_id: int = -1
    keep_seed: bool = True

    w_u: float = 7.1
    w_du: float = 15.0
    w_pitch: float = 10.0
    w_pitch_dot: float = 32.0
    w_goal: float = 10.0
    w_xpos_dot: float = 20.0

    just_gp_model: bool = False
    stop_re_training_mode: bool = False
    re_training_mode: bool = False

    online_update_steps: int = 50
    online_replay_size: int = 1024
    online_max_keep_points: int = 20000
    full_retrain_every_episodes: int = 1
    retrain_every_episodes: int = 1
    max_points_for_train: int = 20000

    # This timer is only for reset watchdog + async GP reload.
    housekeeping_dt: float = 0.10


# ============================================================
# MPPI Controller Node (lock-step event-driven)
# ============================================================

class MPPICarControllerNode(Node):
    def __init__(self, cfg: MPPIConfig):
        super().__init__("mppi_car_controller")
        self.cfg = cfg

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using torch device: {self.device}")

        # ----------------------------------------------------
        # Load GP models
        # ----------------------------------------------------
        self.gp_pitch: SVGPManager = SVGPManager.load(self.cfg.gp_pitch_path)
        self.gp_pitch_dot: SVGPManager = SVGPManager.load(self.cfg.gp_pitch_dot_path)
        self.gp_xpos: SVGPManager = SVGPManager.load(self.cfg.gp_xpos_path)
        self.gp_xpos_dot: SVGPManager = SVGPManager.load(self.cfg.gp_xpos_dot_path)

        self.gp_pitch.device = self.device
        self.gp_pitch_dot.device = self.device
        self.gp_xpos.device = self.device
        self.gp_xpos_dot.device = self.device

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------
        self.obs_xpos: Optional[float] = None

        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0

        self.roll: float = 0.0
        self.pitch: float = 0.0
        self.yaw: float = 0.0
        self.roll_dot: float = 0.0
        self.pitch_dot: float = 0.0
        self.yaw_dot: float = 0.0

        self.last_state_valid: bool = False
        self.last_odom_valid: bool = False

        self.latest_imu_stamp_ns: Optional[int] = None
        self.latest_odom_stamp_ns: Optional[int] = None

        # Lock-step gating:
        # The simulator publishes state with a shared ROS stamp and then publishes
        # /sim_step_stamp_ns carrying that exact stamp. We compute exactly one action
        # when IMU and odom both match that stamp.
        self.pending_step_id: Optional[int] = None
        self.pending_step_stamp_ns: Optional[int] = None
        self.last_acted_step_stamp_ns: Optional[int] = None

        self.control_busy: bool = False

        # ----------------------------------------------------
        # ROS interfaces
        # ----------------------------------------------------
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)

        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.obs_sub = self.create_subscription(PoseStamped, "/obstacle_pose", self.obs_callback, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)
        self.step_id_sub = self.create_subscription(Int64, "/sim_step_id", self.sim_step_id_cb, 10)
        self.step_stamp_sub = self.create_subscription(Int64, "/sim_step_stamp_ns", self.sim_step_stamp_cb, 10)

        self.reset_client = self.create_client(Trigger, "reset_car")
        self.resetting = False

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_car service...")

        # ----------------------------------------------------
        # Episode / reset logic
        # ----------------------------------------------------
        self.ep_cost_sum = 0.0
        self.ep_cost_steps = 0

        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        self.episode_id: int = -1
        self.episode_start_time = None
        self.episode_started = False

        # ----------------------------------------------------
        # Locks + managers
        # ----------------------------------------------------
        self.model_lock = threading.Lock()

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

        # root utils.mppi_core expects action(x0, obs_pos_x), while some other copies
        # use action(x0). Detect once and stay compatible with either version.
        self._mppi_action_nargs = len(inspect.signature(self.mppi.action).parameters)

        self.dataset = DatasetBuffer(
            maxlen=self.cfg.max_log_points,
            log_dir=self.cfg.log_dir,
            ctrl_dt=self.cfg.ctrl_dt,
            logger=self.get_logger(),
        )

        self.retrain = GPRetrainManager(
            cfg=self.cfg,
            device=self.device,
            model_lock=self.model_lock,
            logger=self.get_logger(),
        )
        self.reset_after_retrain = False

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

        self.housekeeping_timer = self.create_timer(self.cfg.housekeeping_dt, self.housekeeping_cb)

        self.get_logger().info(
            "MPPI controller initialized in lock-step EVENT-DRIVEN mode. "
            "It publishes exactly one action per simulator step signal."
        )

        # Ask for a fresh initial state after subscriptions are ready.
        self.request_reset(force=True)

    # ========================================================
    # Helpers
    # ========================================================
    def _mark_episode_started(self):
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True
            self.ep_cost_sum = 0.0
            self.ep_cost_steps = 0

    def _local_reset_state(self):
        self.last_state_valid = False
        self.last_odom_valid = False

        self.xpos = None
        self.xpos_dot = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll_dot = 0.0
        self.pitch_dot = 0.0
        self.yaw_dot = 0.0

        self.latest_imu_stamp_ns = None
        self.latest_odom_stamp_ns = None

        self.pending_step_id = None
        self.pending_step_stamp_ns = None
        self.last_acted_step_stamp_ns = None

        self.mppi.reset_plan()
        self.episode_start_time = None
        self.episode_started = False

    def _swap_reloaded_models_if_ready(self):
        loaded = self.retrain.reload_models_if_ready()
        if loaded is None:
            return False

        gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot = loaded
        self.gp_xpos = gp_xpos
        self.gp_xpos_dot = gp_xpos_dot
        self.gp_pitch = gp_pitch
        self.gp_pitch_dot = gp_pitch_dot

        self.mppi.set_models(gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot)
        self.get_logger().info("Reloaded retrained GP models.")
        return True

    def _call_mppi_action(self, x0_np: np.ndarray) -> float:
        if self._mppi_action_nargs >= 2:
            obs_val = float(self.obs_xpos) if self.obs_xpos is not None else 0.0
            return self.mppi.action(x0_np, obs_val)
        return self.mppi.action(x0_np)

    # ========================================================
    # Housekeeping only
    # ========================================================
    def housekeeping_cb(self):
        if self.retrain.training or self.retrain.reload_pending:
            loaded_now = self._swap_reloaded_models_if_ready()
            if loaded_now and self.reset_after_retrain and not self.resetting:
                self.reset_after_retrain = False
                self.request_reset(force=True)
            return

        if self.waiting_post_reset and (not self.resetting) and (self.post_reset_start_time is not None):
            elapsed = (self.get_clock().now() - self.post_reset_start_time).nanoseconds * 1e-9
            if (elapsed > 3.0) and (not self.watchdog_fired):
                self.watchdog_fired = True
                self.get_logger().warn("Stuck waiting for post-reset state. Retrying reset.")
                self.request_reset(force=True)

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

        (
            self.roll,
            self.pitch,
            self.yaw,
            self.roll_dot,
            self.pitch_dot,
            self.yaw_dot,
        ) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        self.latest_imu_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.last_state_valid = True
        self.maybe_try_control_step()

    def obs_callback(self, msg: PoseStamped):
        self.obs_xpos = float(msg.pose.position.x)

    def car_callback(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        self.latest_odom_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.last_odom_valid = True
        self.maybe_try_control_step()

    def sim_step_id_cb(self, msg: Int64):
        self.pending_step_id = int(msg.data)

    def sim_step_stamp_cb(self, msg: Int64):
        self.pending_step_stamp_ns = int(msg.data)
        self.maybe_try_control_step()

    # ========================================================
    # Cost accumulation
    # ========================================================
    def _accumulate_executed_cost(self, x0_np, u_cmd):
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        u = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)
        c = self.mppi.stage_cost_torch(x0, u)
        c = float(c.item())
        self.ep_cost_sum += c
        self.ep_cost_steps += 1

    # ========================================================
    # Lock-step control
    # ========================================================
    def maybe_try_control_step(self):
        if self.control_busy:
            return
        if self.resetting:
            return
        if self.retrain.training or self.retrain.reload_pending:
            return

        target = self.pending_step_stamp_ns
        if target is None:
            return

        if self.latest_imu_stamp_ns != target:
            return
        if self.latest_odom_stamp_ns != target:
            return

        if self.last_acted_step_stamp_ns == target:
            return

        if self.xpos is None:
            return

        self.control_busy = True
        try:
            if self.waiting_post_reset:
                self.waiting_post_reset = False
                self.post_reset_start_time = None
                self.watchdog_fired = False
                self.get_logger().info("Received synchronized post-reset state. Starting next episode.")

            self._process_current_state_and_maybe_publish(target)
        finally:
            self.control_busy = False

    def _process_current_state_and_maybe_publish(self, matched_stamp_ns: int):
        cfg = self.cfg

        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Still waiting for valid IMU state.")
                self.warned_no_imu = True
            return
        self.warned_no_imu = False

        if not self.last_odom_valid:
            return

        pitch = float(self.pitch)
        pitch_dot = float(self.pitch_dot)

        if self.dataset.n_points() < cfg.N_target_train and (self.cfg.just_gp_model is False):
            self.cfg.re_training_mode = True
        else:
            self.cfg.re_training_mode = False

        # Left boundary failure
        if self.xpos <= float(cfg.x_min_terminate):
            self.get_logger().warn(
                f"Episode {int(self.episode_id)} terminated: xpos={self.xpos:.3f} m <= {cfg.x_min_terminate:.3f} m"
            )
            self.dataset.drop_episode(self.episode_id)
            self._record_episode_metric(retrain_started=False, success=0)
            self.last_acted_step_stamp_ns = matched_stamp_ns
            self.request_reset()
            return

        # Goal reached
        if self.xpos >= float(cfg.goal_x):
            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = False
            if self.cfg.re_training_mode and do_retrain_now:
                started = self.retrain.maybe_start_retrain_async(
                    self.dataset,
                    episode_id=self.episode_id,
                    force=False,
                )

            self.get_logger().info(
                f"[SUCCESS] Episode {int(self.episode_id)} reached goal at x={self.xpos:.3f}"
            )
            self._record_episode_metric(retrain_started=started, success=1)
            self.last_acted_step_stamp_ns = matched_stamp_ns

            if started:
                self.reset_after_retrain = True
                return

            self.request_reset()
            return

        # Episode timeout
        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={cfg.episode_timeout_sec:.2f}s)."
                )

                ep_num = self.episode_id + 1
                do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

                started = False
                if do_retrain_now:
                    started = self.retrain.maybe_start_retrain_async(
                        self.dataset,
                        episode_id=self.episode_id,
                        force=True,
                    )

                self._record_episode_metric(retrain_started=started, success=0)
                self.last_acted_step_stamp_ns = matched_stamp_ns

                if started:
                    self.reset_after_retrain = True
                    return

                self.request_reset(force=True)
                return

        # Unsafe pose / flip
        if (abs(pitch) >= float(cfg.pitch_stop_abs)) or (abs(self.roll) >= float(cfg.roll_stop_abs)):
            self.get_logger().warn(
                f"Flip detected. roll={self.roll:.3f}, pitch={pitch:.3f}"
            )

            ep_num = self.episode_id + 1
            do_retrain_now = (cfg.retrain_every_episodes > 0) and (ep_num % cfg.retrain_every_episodes == 0)

            started = False
            if self.cfg.re_training_mode:
                if do_retrain_now:
                    started = self.retrain.maybe_start_retrain_async(
                        self.dataset,
                        episode_id=self.episode_id,
                        force=True,
                    )
                self._record_episode_metric(retrain_started=started, success=0)
                self.last_acted_step_stamp_ns = matched_stamp_ns

                if started:
                    self.reset_after_retrain = True
                    return
            else:
                self.dataset.drop_episode(self.episode_id)
                self._record_episode_metric(retrain_started=False, success=0)
                self.last_acted_step_stamp_ns = matched_stamp_ns

            self.request_reset()
            return

        # Normal MPPI action
        x0 = np.array([self.xpos, self.xpos_dot, pitch, pitch_dot], dtype=np.float32)

        try:
            u_cmd = self._call_mppi_action(x0)
            if not math.isfinite(u_cmd):
                self.get_logger().error("u_cmd is NaN/Inf. Forcing 0.")
                u_cmd = 0.0
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            self.get_logger().error(traceback.format_exc())
            u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, cfg.u_min, cfg.u_max))

        self._mark_episode_started()
        self._accumulate_executed_cost(x0, u_cmd)
        self._log_step(pitch, pitch_dot, u_cmd)

        self.last_acted_step_stamp_ns = matched_stamp_ns
        self.publish_u(u_cmd)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)

    # ========================================================
    # Logging
    # ========================================================
    def _log_step(self, pitch: float, pitch_dot: float, u: float):
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
            success=success,
        )

        self.episode_start_time = None
        self.episode_started = False

    # ========================================================
    # Reset logic
    # ========================================================
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

            # In case the synchronized post-reset state already arrived, try now.
            self.maybe_try_control_step()

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
        node.get_logger().info("Shutting down MPPI controller.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()