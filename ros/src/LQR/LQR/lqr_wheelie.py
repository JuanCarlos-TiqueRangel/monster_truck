#!/usr/bin/env python3
"""ROS 2 LQR controller for the planar monster-truck wheelie task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32

from utils import geometry


@dataclass
class LQRWheelieConfig:
    """Parameters for the reduced Newton-Euler LQR wheelie model."""

    ctrl_dt: float = 0.1

    # Reduced planar Newton-Euler parameters.
    mass: float = 5.08
    gravity: float = 9.81
    pitch_inertia: float = 0.32
    drive_force_per_cmd: float = 100.0
    drive_moment_arm: float = 0.095
    gravity_moment_arm: float = 0.1905
    linear_damping: float = 3.0
    pitch_damping: float = 0.06
    throttle_sign: float = 1.0
    pitch_rate_sign: float = 1.0

    # Targets.
    pitch_target: float = math.radians(85.0)
    x_ref: float = 0.0
    x_velocity_target: float = 0.0
    hold_current_x: bool = True

    # Input bounds and smoothing.
    u_min: float = -1.0
    u_max: float = 1.0
    du_max_per_sec: float = 60.0

    # LQR weights for [x, x_dot, pitch, pitch_dot] and throttle.
    q_x: float = 0.0
    q_x_dot: float = 0.0
    q_pitch: float = 90.0
    q_pitch_dot: float = 10.0
    r_u: float = 1.0


class NewtonEulerWheelieLQR:
    """Discrete LQR around a reduced Newton-Euler wheelie equilibrium.

    The continuous model is:
        x_dot = v
        m v_dot = F(u) - c_v v
        theta_dot = omega
        I theta_ddot = h F(u) - m g l cos(theta) - c_theta omega

    The input is the normalized throttle published on /cmd_action.
    """

    def __init__(self, cfg: LQRWheelieConfig, logger=None):
        """Build the linearized model and solve the LQR gain."""
        self.cfg = cfg
        self.logger = logger
        self.a_d, self.b_d = self._linearized_discrete_model()
        self.q = np.diag([cfg.q_x, cfg.q_x_dot, cfg.q_pitch, cfg.q_pitch_dot])
        self.r = np.array([[float(cfg.r_u)]], dtype=float)
        self.k = self._solve_lqr_gain(self.a_d, self.b_d, self.q, self.r)
        self.u_eq = self._equilibrium_throttle(cfg.pitch_target)

    def _linearized_discrete_model(self) -> tuple[np.ndarray, np.ndarray]:
        """Linearize and discretize the Newton-Euler model with Euler hold."""
        cfg = self.cfg
        mass = max(float(cfg.mass), 1e-9)
        inertia = max(float(cfg.pitch_inertia), 1e-9)
        force_gain = float(cfg.throttle_sign) * float(cfg.drive_force_per_cmd)

        a_c = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -cfg.linear_damping / mass, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [
                    0.0,
                    0.0,
                    (
                        mass
                        * cfg.gravity
                        * cfg.gravity_moment_arm
                        * math.sin(cfg.pitch_target)
                    )
                    / inertia,
                    -cfg.pitch_damping / inertia,
                ],
            ],
            dtype=float,
        )
        b_c = np.array(
            [
                [0.0],
                [force_gain / mass],
                [0.0],
                [cfg.drive_moment_arm * force_gain / inertia],
            ],
            dtype=float,
        )

        dt = max(float(cfg.ctrl_dt), 1e-6)
        return np.eye(4, dtype=float) + dt * a_c, dt * b_c

    def _equilibrium_throttle(self, pitch: float) -> float:
        """Compute throttle that balances gravity at the target pitch."""
        cfg = self.cfg
        denominator = (
            float(cfg.throttle_sign)
            * float(cfg.drive_force_per_cmd)
            * float(cfg.drive_moment_arm)
        )
        if abs(denominator) < 1e-9:
            return 0.0

        gravity_torque = (
            float(cfg.mass)
            * float(cfg.gravity)
            * float(cfg.gravity_moment_arm)
            * math.cos(float(pitch))
        )
        u_eq = gravity_torque / denominator
        return float(np.clip(u_eq, cfg.u_min, cfg.u_max))

    def _solve_lqr_gain(
        self,
        a_d: np.ndarray,
        b_d: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
    ) -> np.ndarray:
        """Solve the discrete algebraic Riccati equation by iteration."""
        p = q.copy()
        max_iterations = 500
        tolerance = 1e-10

        for _ in range(max_iterations):
            btp = b_d.T @ p
            s = r + btp @ b_d
            gain = np.linalg.solve(s, btp @ a_d)
            p_next = a_d.T @ p @ (a_d - b_d @ gain) + q
            p_next = 0.5 * (p_next + p_next.T)

            if np.max(np.abs(p_next - p)) < tolerance:
                return gain
            p = p_next

        if self.logger is not None:
            self.logger.warn(
                "LQR Riccati iteration did not fully converge; "
                "using last gain."
            )

        btp = b_d.T @ p
        return np.linalg.solve(r + btp @ b_d, btp @ a_d)

    def action(self, state: np.ndarray | list[float], x_ref: float) -> float:
        """Compute throttle for [x, x_dot, pitch, pitch_dot]."""
        cfg = self.cfg
        ref = np.array(
            [
                float(x_ref),
                float(cfg.x_velocity_target),
                float(cfg.pitch_target),
                0.0,
            ],
            dtype=float,
        )
        error = np.asarray(state, dtype=float).reshape(4) - ref
        error[2] = geometry.wrap_to_pi(error[2])

        delta_u = -float((self.k @ error.reshape(4, 1))[0, 0])
        u = self.u_eq + delta_u
        return float(np.clip(u, cfg.u_min, cfg.u_max))

    def gain_string(self) -> str:
        """Return the LQR gain as a compact log string."""
        return np.array2string(self.k.reshape(-1), precision=3, separator=", ")


class LQRWheelieNode(Node):
    """ROS 2 node that runs the wheelie LQR at a fixed control rate."""

    def __init__(self):
        """Create ROS interfaces and initialize the wheelie LQR."""
        super().__init__("lqr_wheelie_controller")
        self.cfg = LQRWheelieConfig()
        self.lqr = NewtonEulerWheelieLQR(self.cfg, logger=self.get_logger())

        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.car_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)

        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0
        self.last_odom_valid = False

        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.pitch_unwrapped: Optional[float] = None
        self.last_state_valid = False

        self.warned_no_imu = False
        self.warned_no_odom = False

        self.x_hold_ref: Optional[float] = None
        self.last_u = 0.0
        self.last_log_time = self.get_clock().now()

        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info(
            "LQR wheelie controller initialized. "
            f"pitch_target={math.degrees(self.cfg.pitch_target):.1f} deg, "
            f"K={self.lqr.gain_string()}, "
            f"u_eq={self.lqr.u_eq:.3f}"
        )

    def imu_cb(self, msg: Imu) -> None:
        """Store latest wheelie pitch and pitch-rate state."""
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        self.pitch, self.pitch_dot = geometry.quat_to_wheelie_state(qw, qx, qy, qz, wx, wy, wz,
            prev_pitch_unwrapped=self.pitch_unwrapped,
            pitch_rate_sign=self.cfg.pitch_rate_sign,
        )
        self.pitch_unwrapped = self.pitch

        self.last_state_valid = True

    def car_callback(self, msg: Odometry) -> None:
        """Store latest car odometry."""
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        self.last_odom_valid = True

    def control_timer_cb(self) -> None:
        """Run one LQR control tick."""
        cfg = self.cfg

        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first IMU message...")
                self.warned_no_imu = True
            self.publish_u(0.0)
            return
        self.warned_no_imu = False

        if (not self.last_odom_valid) or self.xpos is None:
            if not self.warned_no_odom:
                self.get_logger().warn("Waiting for first odometry message...")
                self.warned_no_odom = True
            self.publish_u(0.0)
            return
        self.warned_no_odom = False

        if self.x_hold_ref is None:
            self.x_hold_ref = (
                float(self.xpos) if cfg.hold_current_x else cfg.x_ref
            )

        state = [self.xpos, self.xpos_dot, self.pitch, self.pitch_dot]
        u_cmd = self.lqr.action(state, self.x_hold_ref)
        u_cmd = self._rate_limit(u_cmd)

        self.publish_u(u_cmd)
        self._log_periodic(u_cmd)

    def _rate_limit(self, u_cmd: float) -> float:
        """Apply throttle slew-rate limiting."""
        max_step = (
            max(0.0, self.cfg.du_max_per_sec)
            * max(self.cfg.ctrl_dt, 1e-6)
        )
        if max_step > 0.0:
            u_cmd = geometry.clip(
                u_cmd,
                self.last_u - max_step,
                self.last_u + max_step,
            )
        u_cmd = geometry.clip(u_cmd, self.cfg.u_min, self.cfg.u_max)
        self.last_u = u_cmd
        return u_cmd

    def _log_periodic(self, u_cmd: float) -> None:
        """Log compact controller diagnostics at a low rate."""
        now = self.get_clock().now()
        elapsed = (now - self.last_log_time).nanoseconds * 1e-9
        if elapsed < 0.5:
            return

        self.last_log_time = now
        pitch_error = geometry.wrap_to_pi(self.pitch - self.cfg.pitch_target)
        self.get_logger().info(
            f"x={self.xpos:.2f} "
            f"xdot={self.xpos_dot:.2f} "
            f"pitch={math.degrees(self.pitch):.1f} deg "
            f"err={math.degrees(pitch_error):+.1f} deg "
            f"pitch_dot={self.pitch_dot:.2f} u={u_cmd:+.3f}"
        )

    def publish_u(self, u: float) -> None:
        """Publish normalized throttle."""
        msg = Float32()
        msg.data = geometry.clip(u, self.cfg.u_min, self.cfg.u_max)
        self.cmd_pub.publish(msg)


def main(args=None) -> None:
    """Run the LQR wheelie node."""
    rclpy.init(args=args)
    node = LQRWheelieNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down LQR controller, sending u=0.0")
        node.publish_u(0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
