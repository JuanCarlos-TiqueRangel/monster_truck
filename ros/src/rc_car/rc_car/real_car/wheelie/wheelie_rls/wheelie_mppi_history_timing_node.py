#!/usr/bin/env python3
"""History MPPI node with bounded waiting instead of resetting on early samples.

Uses only the reviewed publisher's Odometry message, with BODY-frame twist.
No changes to MPPI costs, sampling, dynamics, or learned coefficients.
Start drive-disabled. This does not repair OptiTrack/NatNet outages or provide a
hard real-time watchdog. A receiver-side command timeout is still required.
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
        super().__init__("mppi_history_timing_controller")
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.ctrl_dt = float(self.cfg.dt)
        self.controller = MPPITorch(self.p, self.cfg, _ROOT / MODEL_FILE, integrator="euler", live_plot=LIVE_PLOT, dtype=torch.float64)
        self.ref = np.array([0.0, 0.0, REFERENCE_PITCH, 0.0])

        # First-use CPU/CUDA work happens before ROS subscriptions and timers.
        warm_state = np.zeros(4)
        warm_history = np.zeros((4, 4))
        warm_ages = self.ctrl_dt * np.arange(1, 5, dtype=float)
        for _ in range(3):
            _, info = self.controller.solve(warm_state, self.ref, 0.0, warm_history, warm_ages)
            if not info.get("success", False):
                raise RuntimeError("Initialization solve failed; control timer was not started.")
        if self.controller.device.type == "cuda":
            torch.cuda.synchronize(self.controller.device)
        self.controller.last_solution = None

        self.state = None
        self.state_stamp_ns = None
        self.state_received_at = None
        self.state_error = ""
        self.resume_after_stamp_ns = None
        self.u_prev = 0.0
        self.history = deque(maxlen=4)  # (measurement stamp ns, [u_published, v_body_x, pitch, pitch_rate]).
        self.last_history_at = None
        self.last_command_at = None
        self.next_control_at = None
        self.mode = "WAIT_STATE"
        self.stop_count = 0  # Stop episodes, not every repeated zero publication.
        self.wait_count = 0
        self.last_status_log = self.last_stop_log = -math.inf

        self.sensor_timeout = 3.0 * self.ctrl_dt  # Original receipt-age check.
        self.min_history_dt = 0.5 * self.ctrl_dt
        self.max_history_dt = 1.5 * self.ctrl_dt
        self.fresh_for_solve = 0.5 * self.ctrl_dt  # Do not begin a solve on an older locally received snapshot.
        self.poll_dt = min(0.01, self.ctrl_dt / 4.0)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(Odometry, f"/optitrack/odom_{OBJECT_ID}", self.odom_callback, qos)
        self.drive_pub = self.create_publisher(Float32, "/drive", 10)
        self.timer = self.create_timer(self.poll_dt, self.control_callback)
        self.get_logger().info(f"History MPPI: model={MODEL_FILE}, dt={self.ctrl_dt:.3f} s, poll={self.poll_dt:.3f} s, dtype=float64, drive_enabled={ENABLE_DRIVE}")

    def odom_callback(self, msg):
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self.state_stamp_ns is not None and stamp_ns <= self.state_stamp_ns:
            return  # Never refresh the receipt watchdog with an old/duplicate stamp.
        if msg.child_frame_id != EXPECTED_CHILD_FRAME:
            self.state_error = f"child_frame_id={msg.child_frame_id!r}; expected {EXPECTED_CHILD_FRAME!r} with BODY-frame twist"
            return
        q = msg.pose.pose.orientation
        quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
        norm = np.linalg.norm(quat)
        if not np.isfinite(quat).all() or not math.isfinite(norm) or norm < 1e-12:
            self.state_error = "invalid quaternion"
            return
        qx, qy, qz, qw = quat / norm
        pitch = math.atan2(2.0 * (qx * qz + qw * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        state = np.array([msg.pose.pose.position.x, msg.twist.twist.linear.x, pitch, msg.twist.twist.angular.y], dtype=float)
        if not np.isfinite(state).all():
            self.state_error = "nonfinite measured state"
            return
        self.state = state
        self.state_stamp_ns = stamp_ns
        self.state_received_at = time.monotonic()
        self.state_error = ""

    def stop(self, reason=""):
        self.drive_pub.publish(Float32(data=0.0))
        self.last_command_at = time.monotonic()
        self.u_prev = 0.0
        self.history.clear()
        self.last_history_at = None
        self.next_control_at = None
        self.controller.last_solution = None
        self.resume_after_stamp_ns = self.state_stamp_ns
        if self.mode != "STOP":
            self.stop_count += 1
        self.mode = "STOP"
        if reason and self.last_command_at - self.last_stop_log >= 1.0:
            self.get_logger().warning(f"STOP #{self.stop_count}: {reason}; u_published=0.000")
            self.last_stop_log = self.last_command_at

    def control_callback(self):
        now = time.monotonic()
        if self.state_error:
            self.stop(self.state_error)
            return
        if self.state is None:
            self.stop("waiting for Odometry")
            return
        rx_age = now - self.state_received_at
        if rx_age > self.sensor_timeout:
            self.stop(f"stale Odometry: receive_age_ms={1000.0 * rx_age:.1f}")
            return

        # These clocks do NOT reset when a tick is skipped. Waiting is bounded.
        if self.history and now - self.last_history_at > self.max_history_dt + 1e-9:
            self.stop(f"history watchdog: no usable next control sample within {1000.0 * self.max_history_dt:.1f} ms")
            return
        if self.u_prev != 0.0 and self.last_command_at is not None and now - self.last_command_at > self.max_history_dt + 1e-9:
            self.stop("command-refresh deadline expired")
            return
        if self.next_control_at is not None and now + 1e-6 < self.next_control_at:
            return  # Timer polls at 100 Hz; optimization still targets 25 Hz.
        if self.resume_after_stamp_ns is not None and self.state_stamp_ns <= self.resume_after_stamp_ns:
            return  # After a stop, require a genuinely newer snapshot.

        history_dt = None
        if self.history:
            history_dt = (self.state_stamp_ns - self.history[0][0]) * 1e-9
            if history_dt < 0.0 or history_dt > self.max_history_dt + 1e-9:
                self.stop(f"history gap: history_dt_ms={1000.0 * history_dt:.1f}")
                return
        if rx_age > self.fresh_for_solve or (history_dt is not None and history_dt < self.min_history_dt - 1e-9):
            # No new command, no repeated history row, no warm-start reset.
            # The receiver holds its last command only until the bounded watchdog
            # stops us or a usable sample arrives (receiver timeout may act first).
            self.mode = "WAIT_FRESH_STATE"
            self.wait_count += 1
            return

        state = self.state.copy()
        stamp_ns = self.state_stamp_ns
        received_at = self.state_received_at
        u_plan = math.nan
        solve_seconds = 0.0
        info = {}
        if len(self.history) < 4:
            u = 0.0
            mode = "BUILD_HISTORY"
        else:
            past_z = np.array([row for _, row in self.history], dtype=float)
            past_ages = np.array([(stamp_ns - stamp) * 1e-9 for stamp, _ in self.history], dtype=float)
            started = time.monotonic()
            try:
                u_plan, info = self.controller.solve(state, self.ref, self.u_prev, past_z, past_ages)
                if self.controller.device.type == "cuda":
                    torch.cuda.synchronize(self.controller.device)
            except Exception as exc:
                self.stop(f"solver exception: {exc}")
                return
            finished = time.monotonic()
            solve_seconds = finished - started
            if not info.get("success", False) or not math.isfinite(u_plan) or solve_seconds > self.ctrl_dt:
                self.stop(f"solve_ms={1000.0 * solve_seconds:.1f}, limit_ms={1000.0 * self.ctrl_dt:.1f}, success={info.get('success', False)}, u_plan={u_plan}")
                return
            if finished - received_at > self.max_history_dt:
                self.stop("snapshot too old after solve")
                return
            if self.u_prev != 0.0 and self.last_command_at is not None and finished - self.last_command_at > self.max_history_dt:
                self.stop("command-refresh deadline expired during solve")
                return
            u = float(np.clip(u_plan, self.p.u_min, self.p.u_max)) if ENABLE_DRIVE else 0.0
            mode = "RUN" if ENABLE_DRIVE else "DRIVE_DISABLED"

        self.drive_pub.publish(Float32(data=u))
        self.last_command_at = time.monotonic()
        self.history.appendleft((stamp_ns, np.r_[u, state[1:]]))
        self.last_history_at = now
        self.resume_after_stamp_ns = None
        self.u_prev = u
        self.mode = mode
        # Keep a nominal schedule, but do not burst through missed control ticks.
        next_due = now + self.ctrl_dt if self.next_control_at is None else self.next_control_at + self.ctrl_dt
        self.next_control_at = next_due if next_due >= now + self.min_history_dt else now + self.ctrl_dt
        if PRINT_STATUS and now - self.last_status_log >= 1.0:
            self.last_status_log = now
            print(f"mode={mode} | pitch={np.degrees(state[2]):+.2f} deg | ref={np.degrees(self.ref[2]):+.2f} deg | "
                  f"u_plan={u_plan:.6f} | u_published={u:.6f} | history={len(self.history)}/4 | "
                  f"theta_N_best={info.get('best_theta_N_deg', math.nan):+.2f} deg | "
                  f"solve={1000.0 * solve_seconds:.1f} ms | rx_age={1000.0 * rx_age:.1f} ms | stops={self.stop_count} | waits={self.wait_count}")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPPIControllerNode()
        rclpy.spin(node)  # Keep the existing single-threaded execution convention.
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
