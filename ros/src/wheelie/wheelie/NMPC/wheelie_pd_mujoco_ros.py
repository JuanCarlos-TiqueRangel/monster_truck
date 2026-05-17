#!/usr/bin/env python3
"""
wheelie_pd_mujoco_ros.py

ROS2 + MuJoCo wheelie PD controller.

This is the ROS/MuJoCo version of your standalone Euler-Lagrange / feedback
linearization PD controller. It uses the same basic ROS interfaces as your MPPI
controller:

    Subscribes:
        /car_imu   : sensor_msgs/Imu
        /car_odom  : nav_msgs/Odometry

    Publishes:
        /cmd_action : std_msgs/Float32

Important:
    - No reset service is used.
    - No GP, MPPI, retraining, logging buffer, or episode logic is used.
    - By default, the torque computed by the PD law is mapped to a normalized
      action in [-1, 1], because your MPPI node publishes normalized actions on
      /cmd_action.

Run:
    ros2 run <your_package> wheelie_pd_mujoco_ros.py

or directly, if sourced:
    python3 wheelie_pd_mujoco_ros.py
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# Keep the same import style used by your MPPI controller.
# This assumes this file is placed in the same folder level as your MPPI node.
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import geometry


# ============================================================
# Controller parameters
# ============================================================

@dataclass
class WheeliePDConfig:
    # ROS timing
    ctrl_dt: float = 0.02          # 50 Hz. Use 0.1 if you want same as old script.

    # Physical parameters from your standalone PD script
    m: float = 5.1                 # kg
    l: float = 0.18                # m, rear axle to COM distance
    body_length: float = 0.53      # m
    body_height: float = 0.30      # m
    r: float = 0.085               # m, rear wheel radius
    g: float = 9.81                # m/s^2

    # PD gains
    kp: float = 30.0
    kd: float = 8.0

    # Target wheelie angle
    pitch_ref_deg: float = 85.0    # use 90.0 if you really want vertical

    # Torque limits from your standalone script
    tau_min: float = -8.0          # N*m
    tau_max: float = 12.0          # N*m

    # Action limits used by the MuJoCo/ROS actuator command
    u_min: float = -1.0
    u_max: float = 1.0

    # True  -> publish normalized action in [-1, 1], same style as MPPI.
    # False -> publish raw torque tau directly on /cmd_action.
    publish_normalized_action: bool = True

    # Use this only if the car moves in the opposite direction in MuJoCo.
    reverse_action_sign: bool = False

    # Optional first-order smoothing on the command.
    # 1.0 = no smoothing. Smaller values = smoother but slower response.
    action_filter_alpha: float = 1.0

    # Print controller debug at this period.
    debug_period_sec: float = 0.5

    @property
    def I_body(self) -> float:
        return (1.0 / 12.0) * self.m * (self.body_length**2 + self.body_height**2)

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2

    @property
    def pitch_ref_rad(self) -> float:
        return math.radians(self.pitch_ref_deg)


# ============================================================
# Small utilities
# ============================================================

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def tau_to_normalized_action(tau: float, cfg: WheeliePDConfig) -> float:
    """
    Convert physical torque [N*m] to normalized action [-1, 1].

    Positive torque is scaled by tau_max.
    Negative torque is scaled by abs(tau_min).
    """
    if tau >= 0.0:
        u = tau / max(1e-9, abs(cfg.tau_max))
    else:
        u = tau / max(1e-9, abs(cfg.tau_min))
    return clamp(u, cfg.u_min, cfg.u_max)


def pd_feedback_linearizing_controller(
    theta: float,
    omega: float,
    cfg: WheeliePDConfig,
) -> float:
    """
    Feedback linearization + PD.

    Original simplified pitch dynamics:
        I_eff * theta_ddot = -tau + m*g*l*cos(theta)

    Desired virtual acceleration:
        theta_ddot = -kp*(theta - theta_ref) - kd*omega

    Solving for tau gives:
        tau = m*g*l*cos(theta) + I_eff*(kp*(theta - theta_ref) + kd*omega)

    The returned value is saturated physical torque [N*m].
    """
    error_pitch = theta - cfg.pitch_ref_rad
    error_dot = omega

    tau = (
        cfg.m * cfg.g * cfg.l * math.cos(theta)
        + cfg.I_eff * (cfg.kp * error_pitch + cfg.kd * error_dot)
    )

    return clamp(tau, cfg.tau_min, cfg.tau_max)


# ============================================================
# ROS2 node
# ============================================================

class WheeliePDControllerNode(Node):
    def __init__(self):
        super().__init__("wheelie_pd_controller")

        self.cfg = WheeliePDConfig()
        self._declare_and_load_parameters()

        # Latest state
        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0
        self.pitch: Optional[float] = None
        self.pitch_dot: float = 0.0
        self.pitch_unwrapped: Optional[float] = None
        self.roll: float = 0.0
        self.yaw: float = 0.0

        self.last_state_valid: bool = False
        self.last_odom_valid: bool = False
        self.warned_no_imu: bool = False
        self.warned_no_odom: bool = False

        self.last_u: float = 0.0
        self.last_tau: float = 0.0
        self.last_debug_time = self.get_clock().now()

        # ROS interfaces: same command and state topics as your MPPI node.
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)

        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)

        self.get_logger().info("Wheelie PD ROS/MuJoCo controller initialized.")
        self.get_logger().info(
            f"ctrl_dt={self.cfg.ctrl_dt:.4f}, pitch_ref={self.cfg.pitch_ref_deg:.2f} deg, "
            f"kp={self.cfg.kp:.3f}, kd={self.cfg.kd:.3f}, "
            f"I_body={self.cfg.I_body:.5f}, I_eff={self.cfg.I_eff:.5f}, "
            f"publish_normalized_action={self.cfg.publish_normalized_action}"
        )

    # --------------------------------------------------------
    # ROS params
    # --------------------------------------------------------
    def _declare_and_load_parameters(self):
        # Timing
        self.declare_parameter("ctrl_dt", self.cfg.ctrl_dt)

        # Physical params
        self.declare_parameter("m", self.cfg.m)
        self.declare_parameter("l", self.cfg.l)
        self.declare_parameter("body_length", self.cfg.body_length)
        self.declare_parameter("body_height", self.cfg.body_height)
        self.declare_parameter("r", self.cfg.r)
        self.declare_parameter("g", self.cfg.g)

        # Gains / target
        self.declare_parameter("kp", self.cfg.kp)
        self.declare_parameter("kd", self.cfg.kd)
        self.declare_parameter("pitch_ref_deg", self.cfg.pitch_ref_deg)

        # Saturations / command mapping
        self.declare_parameter("tau_min", self.cfg.tau_min)
        self.declare_parameter("tau_max", self.cfg.tau_max)
        self.declare_parameter("u_min", self.cfg.u_min)
        self.declare_parameter("u_max", self.cfg.u_max)
        self.declare_parameter("publish_normalized_action", self.cfg.publish_normalized_action)
        self.declare_parameter("reverse_action_sign", self.cfg.reverse_action_sign)
        self.declare_parameter("action_filter_alpha", self.cfg.action_filter_alpha)
        self.declare_parameter("debug_period_sec", self.cfg.debug_period_sec)

        self.cfg.ctrl_dt = float(self.get_parameter("ctrl_dt").value)

        self.cfg.m = float(self.get_parameter("m").value)
        self.cfg.l = float(self.get_parameter("l").value)
        self.cfg.body_length = float(self.get_parameter("body_length").value)
        self.cfg.body_height = float(self.get_parameter("body_height").value)
        self.cfg.r = float(self.get_parameter("r").value)
        self.cfg.g = float(self.get_parameter("g").value)

        self.cfg.kp = float(self.get_parameter("kp").value)
        self.cfg.kd = float(self.get_parameter("kd").value)
        self.cfg.pitch_ref_deg = float(self.get_parameter("pitch_ref_deg").value)

        self.cfg.tau_min = float(self.get_parameter("tau_min").value)
        self.cfg.tau_max = float(self.get_parameter("tau_max").value)
        self.cfg.u_min = float(self.get_parameter("u_min").value)
        self.cfg.u_max = float(self.get_parameter("u_max").value)
        self.cfg.publish_normalized_action = bool(self.get_parameter("publish_normalized_action").value)
        self.cfg.reverse_action_sign = bool(self.get_parameter("reverse_action_sign").value)
        self.cfg.action_filter_alpha = float(self.get_parameter("action_filter_alpha").value)
        self.cfg.debug_period_sec = float(self.get_parameter("debug_period_sec").value)

        self.cfg.action_filter_alpha = clamp(self.cfg.action_filter_alpha, 0.0, 1.0)

    # --------------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------------
    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        # Same diagnostic Euler conversion style as the MPPI node.
        (
            self.roll,
            _bad_euler_pitch,
            self.yaw,
            _roll_dot,
            _bad_euler_pitch_dot,
            _yaw_dot,
        ) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        # Same singularity-free wheelie pitch state used by the MPPI controller.
        self.pitch, self.pitch_dot = geometry.quat_to_wheelie_state(
            qw, qx, qy, qz,
            wx, wy, wz,
            prev_pitch_unwrapped=self.pitch_unwrapped,
            pitch_rate_sign=1.0,
        )
        self.pitch_unwrapped = self.pitch
        self.last_state_valid = True

    def car_callback(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        self.last_odom_valid = True

    # --------------------------------------------------------
    # Control
    # --------------------------------------------------------
    def control_timer_cb(self):
        if not self.last_state_valid or self.pitch is None:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first IMU message...")
                self.warned_no_imu = True
            self.publish_u(0.0)
            return
        self.warned_no_imu = False

        # Odom is not required for the PD law, but we subscribe to it because your
        # MPPI/MuJoCo setup publishes it and it is useful for debugging.
        if not self.last_odom_valid:
            if not self.warned_no_odom:
                self.get_logger().warn("Waiting for first odometry message... controlling pitch only for now.")
                self.warned_no_odom = True
        else:
            self.warned_no_odom = False

        theta = float(self.pitch)
        omega = float(self.pitch_dot)

        tau = pd_feedback_linearizing_controller(theta, omega, self.cfg)

        if self.cfg.publish_normalized_action:
            u_cmd = tau_to_normalized_action(tau, self.cfg)
        else:
            u_cmd = tau

        if self.cfg.reverse_action_sign:
            u_cmd = -u_cmd

        u_cmd = clamp(u_cmd, self.cfg.u_min, self.cfg.u_max)

        # Optional smoothing. alpha=1.0 means no smoothing.
        alpha = self.cfg.action_filter_alpha
        u_cmd = alpha * u_cmd + (1.0 - alpha) * self.last_u
        u_cmd = clamp(u_cmd, self.cfg.u_min, self.cfg.u_max)

        if not math.isfinite(u_cmd):
            self.get_logger().error("PD command became NaN/Inf. Publishing 0.0")
            u_cmd = 0.0

        self.last_tau = tau
        self.last_u = u_cmd
        self.publish_u(u_cmd)
        self._maybe_print_debug(theta, omega, tau, u_cmd)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)

    def _maybe_print_debug(self, theta: float, omega: float, tau: float, u_cmd: float):
        now = self.get_clock().now()
        elapsed = (now - self.last_debug_time).nanoseconds * 1e-9
        if elapsed < self.cfg.debug_period_sec:
            return
        self.last_debug_time = now

        err_deg = math.degrees(theta - self.cfg.pitch_ref_rad)
        x_str = "nan" if self.xpos is None else f"{self.xpos:.3f}"
        v_str = "nan" if not self.last_odom_valid else f"{self.xpos_dot:.3f}"
        self.get_logger().info(
            f"pitch={math.degrees(theta):7.2f} deg, "
            f"ref={self.cfg.pitch_ref_deg:6.2f} deg, "
            f"err={err_deg:7.2f} deg, "
            f"pitch_dot={omega: .3f} rad/s, "
            f"tau={tau: .3f} N*m, "
            f"u={u_cmd: .3f}, "
            f"x={x_str}, xdot={v_str}"
        )


# ============================================================
# main()
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = WheeliePDControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down PD controller, sending u=0.0")
        node.publish_u(0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
