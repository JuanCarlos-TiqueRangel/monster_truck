#!/usr/bin/env python3
"""Frozen history-model MPPI using the supplied OptiTrack topic conventions.

ENABLE_DRIVE=False evaluates MPPI but publishes zero; explicitly enable only
AFTER checking open-loop prediction, timestamp alignment and solve time.
The existing params_mppi.py and the REAL trained .npz are required.
"""

import sys
import math
import time
from pathlib import Path
from collections import deque
import torch


import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

_ROOT = Path(__file__).resolve().parent
for _sub in ("mppi", "rls"):
    _path = str(_ROOT / _sub)
    if _path not in sys.path:
        sys.path.append(_path)

from params_mppi import WheelieParams, MPPIConfig
from mppi_history import MPPITorch

ENABLE_DRIVE = True
LIVE_PLOT = False
PRINT_STATUS = True
REFERENCE_PITCH = np.deg2rad(-55.0)
OBJECT_ID = 810


class MPPIControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_history_controller")
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.ctrl_dt = float(self.cfg.dt)
        self.controller = MPPITorch(self.p, self.cfg, _ROOT / "rls_history_angular_rollout.npz", integrator="euler", live_plot=LIVE_PLOT, dtype=torch.float64)
        self.ref = np.array([0.0, 0.0, REFERENCE_PITCH, 0.0])

        warm_state = np.zeros(4)
        warm_history = np.zeros((4, 4))
        warm_ages = self.ctrl_dt * np.arange(1, 5, dtype=float)

        for _ in range(3):
            self.controller.solve(warm_state, self.ref, 0.0, warm_history, warm_ages)

        self.controller.last_solution = None

        self.x = self.x_dot = self.pitch = self.pitch_dot = 0.0
        self.state_time = self.velocity_time = None
        self.odom_received_at = self.velocity_received_at = None
        self.u_prev = 0.0
        self.history = deque(maxlen=4)  # (state timestamp, [u_applied, v, theta, omega]), newest first.

        # Receipt-time and timestamp-skew checks are guards, not time synchronization.
        self.sensor_timeout = 3.0 * self.ctrl_dt
        self.max_state_skew = 0.5 * self.ctrl_dt
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(Odometry, f"/optitrack/odom_{OBJECT_ID}", self.odom_callback, qos)
        self.vel_sub = self.create_subscription(TwistStamped, f"/optitrack/twist_{OBJECT_ID}", self.velocity_callback, qos)
        self.drive_pub = self.create_publisher(Float32, "/drive", 10)
        self.timer = self.create_timer(self.ctrl_dt, self.control_callback)
        self.get_logger().info(f"Frozen history MPPI: dt={self.ctrl_dt:.3f} s, drive_enabled={ENABLE_DRIVE}")

    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.state_time is not None and t <= self.state_time:
            return
        q = msg.pose.pose.orientation
        # Preserve the pitch convention in the supplied node; training must match it.
        self.pitch = math.atan2(2.0 * (q.x * q.z + q.w * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.x = msg.pose.pose.position.x
        self.state_time = t
        self.odom_received_at = time.monotonic()

    def velocity_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.velocity_time is not None and t <= self.velocity_time:
            return
        self.x_dot = msg.twist.linear.x
        self.pitch_dot = msg.twist.angular.y
        self.velocity_time = t
        self.velocity_received_at = time.monotonic()

    def stop(self, reason=""):
        self.drive_pub.publish(Float32(data=0.0))
        self.u_prev = 0.0
        self.history.clear()
        self.controller.last_solution = None
        self.stop_count = getattr(self, "stop_count", 0) + 1
        if reason and time.monotonic() - getattr(self, "last_stop_log", -float("inf")) >= 1.0:
            self.get_logger().warning(f"STOP #{self.stop_count}: {reason}")
            self.last_stop_log = time.monotonic()

    def control_callback(self):
        if self.state_time is None or self.velocity_time is None:
            self.stop("waiting for pose/velocity messages")
            return
        now = time.monotonic()
        stale = max(now - self.odom_received_at, now - self.velocity_received_at) > self.sensor_timeout
        skewed = abs(self.state_time - self.velocity_time) > self.max_state_skew
        state = np.array([self.x, self.x_dot, self.pitch, self.pitch_dot], dtype=float)
        if stale or skewed or not np.isfinite(state).all():
            self.stop(f"stale={stale}, skew_ms={1000.0 * abs(self.state_time - self.velocity_time):.1f}, finite={np.isfinite(state).all()}")
            return
        t = float(self.state_time)
        if self.history and not 0.5 * self.ctrl_dt <= t - self.history[0][0] <= 1.5 * self.ctrl_dt:
            self.stop(f"history_dt_ms={1000.0 * (t - self.history[0][0]):.1f}, expected_ms={1000.0 * self.ctrl_dt:.1f}")
            return

        # Collect four genuine measured rows; no invented or planned history.
        if len(self.history) < 4:
            self.drive_pub.publish(Float32(data=0.0))
            self.history.appendleft((t, np.r_[0.0, state[1:]]))
            self.u_prev = 0.0
            return

        past_z = np.array([row for _, row in self.history], dtype=float)
        past_ages = np.array([t - stamp for stamp, _ in self.history], dtype=float)
        started = time.monotonic()
        try:
            u_plan, info = self.controller.solve(state, self.ref, self.u_prev, past_z, past_ages)
        except Exception as exc:
            self.stop()
            self.get_logger().error(f"MPPI stopped: {exc}")
            return
        solve_seconds = time.monotonic() - started
        if not info["success"] or not math.isfinite(u_plan) or solve_seconds > self.ctrl_dt:
            self.stop(f"solve_ms={1000.0 * solve_seconds:.1f}, limit_ms={1000.0 * self.ctrl_dt:.1f}, success={info['success']}, u_plan={u_plan:.4f}, u_published=0.0000")
            return

        # Convention: u_k is issued after measuring s_k, and held for the next interval.
        # This pairing MUST match the CSV logger. No extra input-delay shift is applied.
        u = float(np.clip(u_plan, self.p.u_min, self.p.u_max)) if ENABLE_DRIVE else 0.0
        self.drive_pub.publish(Float32(data=u))
        self.history.appendleft((t, np.r_[u, state[1:]]))
        self.u_prev = u

        if PRINT_STATUS and time.monotonic() - getattr(self, "last_status_log", -float("inf")) >= 1.0:
            self.last_status_log = time.monotonic()
            print(f"pitch={np.degrees(state[2]):+.2f} deg | ref={np.degrees(self.ref[2]):+.2f} deg | "
                  f"u_plan={u_plan:.3f} | u_published={u:.3f} | theta_N_best={info['best_theta_N_deg']:+.2f} deg | "
                  f"solve={1000.0 * solve_seconds:.1f} ms | stops={getattr(self, 'stop_count', 0)} | "
                  f"v={state[1]:+.3f} m/s | omega_now={state[3]:+.3f} rad/s | omega_N_best={info['best_omega_N']:+.3f} rad/s | ")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPPIControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.controller.close_plot()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
