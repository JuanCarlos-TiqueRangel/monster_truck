#!/usr/bin/env python3
"""Frozen history MPPI, without automatic timing stops or history resets.

Uses the reviewed publisher's Odometry with BODY-frame twist. Start with the
receiver neutral. No command is sent until four real past rows are available.
Missing/duplicate measurements and failed solves cause no new publication, not
an automatic zero. The receiver may retain the previous command: keep its
independent timeout/emergency stop. No gap compensation or safety certification
is provided. Ctrl+C/normal shutdown still attempts to publish neutral.
"""

import sys
import math
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
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
LIVE_PLOT = True
PRINT_STATUS = True
REFERENCE_PITCH = np.deg2rad(-55.0)
OBJECT_ID = 810
MODEL_FILE = "rls_history_angular_rollout.npz"
EXPECTED_CHILD_FRAME = "base_link"


class MPPIControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_history_simple_controller")
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.ctrl_dt = float(self.cfg.dt)
        self.controller = MPPITorch(self.p, self.cfg, _ROOT / MODEL_FILE, integrator="euler", live_plot=LIVE_PLOT, dtype=torch.float64)
        self.ref = np.array([0.0, 0.0, REFERENCE_PITCH, 0.0])

        warm_state = np.zeros(4)
        warm_history = np.zeros((4, 4))
        warm_ages = self.ctrl_dt * np.arange(1, 5, dtype=float)
        for _ in range(3):
            _, info = self.controller.solve(warm_state, self.ref, 0.0, warm_history, warm_ages)
            if not info.get("success", False):
                raise RuntimeError("Initialization solve failed.")
        if self.controller.device.type == "cuda":
            torch.cuda.synchronize(self.controller.device)
        self.controller.last_solution = None

        self.state = None
        self.state_stamp_ns = None
        self.last_used_stamp_ns = None
        self.u_prev = 0.0
        self.history = deque(maxlen=4)
        self.last_status_log = self.last_error_log = -math.inf
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(Odometry, f"/optitrack/odom_{OBJECT_ID}", self.odom_callback, qos)
        self.drive_pub = self.create_publisher(Float32, "/drive", 10)
        self.timer = self.create_timer(self.ctrl_dt, self.control_callback)
        self.get_logger().info(f"History MPPI: model={MODEL_FILE}, dt={self.ctrl_dt:.3f} s, dtype=float64, drive_enabled={ENABLE_DRIVE}; automatic timing stops removed")

    def odom_callback(self, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self.state_stamp_ns is not None and stamp_ns <= self.state_stamp_ns:
            return  # Do not store duplicate/out-of-order rows; never publish zero here.
        q = msg.pose.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(quat)
        if msg.child_frame_id != EXPECTED_CHILD_FRAME or not np.isfinite(quat).all() or not math.isfinite(norm) or norm < 1e-12:
            now = time.monotonic()
            if now - self.last_error_log >= 1.0:
                self.get_logger().error("Ignoring Odometry: expected BODY-frame twist in base_link and a finite nonzero quaternion; no command sent.")
                self.last_error_log = now
            return
        qx, qy, qz, qw = quat / norm
        pitch = math.atan2(2.0 * (qx * qz + qw * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        state = np.array([msg.pose.pose.position.x, msg.twist.twist.linear.x, pitch, msg.twist.twist.angular.y], dtype=float)
        if not np.isfinite(state).all():
            now = time.monotonic()
            if now - self.last_error_log >= 1.0:
                self.get_logger().error("Ignoring nonfinite Odometry; no command sent.")
                self.last_error_log = now
            return
        self.state = state
        self.state_stamp_ns = stamp_ns

    def control_callback(self):
        if self.state is None or self.state_stamp_ns == self.last_used_stamp_ns:
            return  # No new measurement: no command, no watchdog, no history reset.
        state = self.state.copy()
        stamp_ns = self.state_stamp_ns
        self.last_used_stamp_ns = stamp_ns

        if len(self.history) < 4:
            self.history.appendleft((stamp_ns, np.r_[self.u_prev, state[1:]]))
            return  # Initial history collection only; do not inject zero commands.

        past_z = np.array([row for _, row in self.history], dtype=float)
        past_ages = np.array([(stamp_ns - stamp) * 1e-9 for stamp, _ in self.history], dtype=float)
        started = time.monotonic()
        try:
            u_plan, info = self.controller.solve(state, self.ref, self.u_prev, past_z, past_ages)
            if self.controller.device.type == "cuda":
                torch.cuda.synchronize(self.controller.device)
            if not info.get("success", False) or not math.isfinite(u_plan):
                raise RuntimeError("Solver failed or returned a nonfinite command.")
        except Exception as exc:
            self.history.appendleft((stamp_ns, np.r_[self.u_prev, state[1:]]))
            now = time.monotonic()
            if now - self.last_error_log >= 1.0:
                self.get_logger().error(f"No new command: {exc}")
                self.last_error_log = now
            return  # Do not send the solver's failure fallback as a forced zero.
        solve_seconds = time.monotonic() - started

        # Same input/state pairing as the previous node; u_prev is the published
        # command, not actuator feedback. Long-gap histories are not repaired.
        u = float(np.clip(u_plan, self.p.u_min, self.p.u_max)) if ENABLE_DRIVE else 0.0
        self.drive_pub.publish(Float32(data=u))
        self.u_prev = u
        self.history.appendleft((stamp_ns, np.r_[u, state[1:]]))

        now = time.monotonic()
        if PRINT_STATUS and now - self.last_status_log >= 1.0:
            self.last_status_log = now
            print(f"pitch={np.degrees(state[2]):+.2f} deg | ref={np.degrees(self.ref[2]):+.2f} deg | "
                  f"u_plan={u_plan:.6f} | u_published={u:.6f} | v={state[1]:+.3f} m/s | omega={state[3]:+.3f} rad/s | "
                  f"theta_N_best={info.get('best_theta_N_deg', math.nan):+.2f} deg | solve={1000.0 * solve_seconds:.1f} ms")


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
            try:
                if rclpy.ok():
                    node.drive_pub.publish(Float32(data=0.0))  # Neutral on exit, not a runtime timing stop.
            finally:
                node.controller.close_plot()
                node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
