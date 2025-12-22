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

from gp_dynamics import GPManager  # <-- your GPManager with .load()
from flip_dataset_manager import FlipEpisodeDataset


# ============================================================
# Config
# ============================================================

@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = 0.1          # [s] controller period (same dt used in GP training)
    horizon: int = 20             # H
    num_rollouts: int = 2000      # K

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    pitch_target: float = math.pi   # radians
    flip_stop_abs: float = 3.1  # stop MPPI when |flip_rel| >= this

    # Paths to trained GP models
    gp_flip_path: str = "models/gp_dynamics_0.pt"  # Δflip/dt
    gp_rate_path: str = "models/gp_dynamics_1.pt"  # Δrate/dt

    # Global dataset NPZ file used for training
    dataset_path: str = "mujoco_random_run.npz"

# ============================================================
# MPPI Controller Node
# ============================================================

class MPPICarControllerNode(Node):
    """
    - Subscribes to /car_imu (from MuJoCoImuNode)
    - Computes [flip_rel, pitch_rate] state
    - Runs GP-based MPPI at ctrl_dt
    - Publishes cmd_action (Float32) to drive the car
    """

    def __init__(self, cfg: MPPIConfig):
        super().__init__("mppi_car_controller")
        self.cfg = cfg

        # ----- Device / RNG -----
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")
        self.get_logger().info(f"Using torch device: {self.device}")

        # ----- Load GP models -----
        # gp_flip: Δflip/dt, gp_rate: Δrate/dt
        self.gp_flip: GPManager = GPManager.load(self.cfg.gp_flip_path)
        self.gp_rate: GPManager = GPManager.load(self.cfg.gp_rate_path)

        # (Optional) override device inside GPs if your GPManager exposes `.device`
        self.gp_flip.device = self.device
        self.gp_rate.device = self.device

        # Pre-create target tensor on correct device
        self.pitch_target_t = torch.tensor(
            self.cfg.pitch_target, dtype=torch.float32, device=self.device
        )

        # ----- ROS interfaces -----
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "car_imu", self.imu_cb, 10)

        # ------ service for reset ---------------
        self.reset_client = self.create_client(Trigger, 'reset_car')

        self.resetting = False

        # Wait until service is available (non-blocking version possible too)
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for reset_car service...')

        self.last_state = None
        self.episode_step = 0

        # Latest state from IMU
        self.t0: Optional[float] = None
        self.last_flip_rel: float = 0.0
        self.last_rate: float = 0.0
        self.last_state_valid: bool = False

        # For computing flip_rel from quaternion
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        # MPPI warm start
        self.plan: Optional[torch.Tensor] = None  # (H,) on device
        self.last_u: float = 0.0

        # NEW: episode dataset logger (lives in separate module)
        self.episode_dataset = FlipEpisodeDataset()

        # Control timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)

        self.get_logger().info("MPPI Car Controller node initialized.")





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

    # ========================================================
    # IMU callback: builds [flip_rel, pitch_rate]
    # ========================================================
    def imu_cb(self, msg: Imu):
        # time (only needed if you want logs; MPPI only cares about state)
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t

        # orientation
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        R, _ = self.quat_to_R_and_pitch(qw, qx, qy, qz)
        up_x, up_y, up_z = R[0, 2], R[1, 2], R[2, 2]

        # angle in (z,x) plane
        theta = math.atan2(up_x, up_z)

        # unwrap
        self.prev_theta, theta_unwrapped = self.unwrap_angle(
            self.prev_theta,
            self.prev_theta_unwrapped,
            theta,
        )
        self.prev_theta_unwrapped = theta_unwrapped

        # reference so flip_rel ~ 0 at start
        if self.theta0 is None:
            self.theta0 = theta_unwrapped
        flip_rel = theta_unwrapped - self.theta0

        # IMU rates/accels
        pitch_rate = float(msg.angular_velocity.y)  # assuming y = pitch rate

        self.last_flip_rel = flip_rel
        self.last_rate = pitch_rate
        self.last_state_valid = True

        # NEW: log this timestep into the episode dataset
        # use the most recently applied action self.last_u
        self.episode_dataset.log_step(
            flip_rel,
            pitch_rate,
            self.last_u,
        )



    # ========================================================
    # Torch helpers: angdiff, stage cost, GP step
    # ========================================================
    @staticmethod
    def angdiff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi


    def stage_cost_torch(self,
                         states: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
        """
        states : (K, 2) [flip_rel, pitch_rate]
        actions: (K,)
        """
        pitch = states[:, 0]
        rate = states[:, 1]
        u = actions

        err = self.angdiff_torch(pitch, self.pitch_target_t)

        cost_pitch = 100.0 * err ** 2
        orient_cost = 100.0 * (1.0 + torch.cos(pitch)) ** 2
        cost_rate = 0.1 * rate ** 2
        cost_u = 0.01 * u ** 2

        return orient_cost + cost_rate + cost_u + cost_pitch

    def gp_step_batch_torch(self,
                            states: torch.Tensor,
                            actions: torch.Tensor) -> torch.Tensor:
        """
        states : (K, 2)
        actions: (K,)
        returns next_states: (K, 2)
        """
        X = torch.stack([states[:, 0], states[:, 1], actions], dim=-1)

        d_flip_mean, _ = self.gp_flip.predict_torch(X)
        d_rate_mean, _ = self.gp_rate.predict_torch(X)

        dt = self.cfg.ctrl_dt

        next_states = torch.empty_like(states)
        next_states[:, 0] = states[:, 0] + d_flip_mean * dt
        next_states[:, 1] = states[:, 1] + d_rate_mean * dt

        next_states[:, 0].clamp_(-math.pi, math.pi)
        next_states[:, 1].clamp_(-20.0, 20.0)

        return next_states

    # ========================================================
    # MPPI core
    # ========================================================
    @torch.no_grad()
    def mppi_action(self, x0_np):
        """
        x0_np: np.array shape (2,) -> [flip_rel, pitch_rate]
        returns scalar action u0 (float)
        """
        cfg = self.cfg
        H = cfg.horizon
        K = cfg.num_rollouts
        LAMBDA = cfg.lambda_
        SIGMA = cfg.sigma
        act_low, act_high = cfg.u_min, cfg.u_max

        # initial state on device
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        assert x0.shape == (2,)

        # warm-start
        if self.plan is None:
            u_init = torch.zeros(H, dtype=torch.float32, device=self.device)
        else:
            u_init = self.plan

        # exploration noise
        eps = torch.randn(K, H, device=self.device) * SIGMA
        U = torch.clamp(u_init.unsqueeze(0) + eps, act_low, act_high)  # (K, H)

        # rollout
        states = x0.unsqueeze(0).repeat(K, 1)         # (K, 2)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        for t in range(H):
            u_t = U[:, t]
            costs = costs + self.stage_cost_torch(states, u_t)
            states = self.gp_step_batch_torch(states, u_t)

        # MPPI weighting
        J_min = costs.min()
        weights = torch.exp(-(costs - J_min) / LAMBDA)
        weights_sum = weights.sum() + 1e-8

        du = (weights.unsqueeze(1) * eps).sum(dim=0) / weights_sum  # (H,)
        u_new = torch.clamp(u_init + du, act_low, act_high)         # (H,)

        u0 = float(u_new[0].detach().cpu())
        self.plan = u_new.detach()  # warm start for next call
        return u0

    # ========================================================
    # Control timer callback
    # ========================================================
    def control_timer_cb(self):
        # If we are currently resetting, just send 0 and wait
        if self.resetting:
            self.publish_u(0.0)
            return

        # Need a valid state before doing anything
        if not self.last_state_valid:
            self.get_logger().warn_once("Waiting for first IMU message...")
            self.publish_u(0.0)
            return

        flip_rel = self.last_flip_rel
        rate = self.last_rate

        # If we consider flip "done", send 0 & trigger reset ONCE
        if abs(flip_rel) >= self.cfg.flip_stop_abs:
            self.publish_u(0.0)

            # 2) NEW: append this episode's data to mujoco_random_run.npz
            total_N = self.episode_dataset.append_to_npz(self.cfg.dataset_path)
            self.get_logger().info(
                f"Dataset '{self.cfg.dataset_path}' now has {total_N} samples."
            )

            self.request_reset()
            return

        # MPPI
        x0 = np.array([flip_rel, rate], dtype=np.float32)
        try:
            u_cmd = self.mppi_action(x0)
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            u_cmd = 0.0

        u_cmd = float(np.clip(u_cmd, self.cfg.u_min, self.cfg.u_max))
        self.publish_u(u_cmd)



    # --------------------------------------------------------
    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = u



    # --------------------------------------
    # Local reset of controller-side state
    # --------------------------------------
    def _local_reset_state(self):
        # Forget orientation history
        self.t0 = None
        self.prev_theta = None
        self.prev_theta_unwrapped = 0.0
        self.theta0 = None

        # Forget last state and MPPI history
        self.last_state_valid = False
        self.plan = None
        self.last_u = 0.0
        self.episode_step = 0

        # NEW: clear current episode data
        self.episode_dataset.reset()

    def request_reset(self):
        # Avoid multiple concurrent reset calls
        if self.resetting:
            return

        self.resetting = True

        # Clear local controller state so the next IMU after the reset
        # will be treated as a fresh episode.
        self._local_reset_state()

        req = Trigger.Request()
        future = self.reset_client.call_async(req)

        def done_callback(f):
            try:
                resp = f.result()
                self.get_logger().info(f"Reset response: {resp.message}")
            except Exception as e:
                self.get_logger().warn(f"Reset service call failed: {e}")

            # Allow control loop to resume after reset completes
            self.resetting = False

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
