#!/usr/bin/env python3
"""Read-only live plots of the six inputs used by wheelie_mppi_rls_node.py."""

import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import AccelStamped, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


class MPPIInputPlotNode(Node):
    def __init__(self):
        super().__init__("mppi_input_plot")
        self.robot_id = int(self.declare_parameter("robot_id", 810).value)
        self.window_s = float(self.declare_parameter("window_s", 10.0).value)
        self.refresh_hz = float(self.declare_parameter("refresh_hz", 10.0).value)
        self.max_samples = int(self.declare_parameter("max_samples", 6000).value)
        self.subscription_depth = int(self.declare_parameter("subscription_depth", 50).value)
        self.gap_s = float(self.declare_parameter("gap_s", 0.05).value)
        # These defaults match the acceleration caps in the current MPPI node.
        self.linear_accel_cap = float(self.declare_parameter("linear_accel_cap", 15.0).value)
        self.angular_accel_cap = float(self.declare_parameter("angular_accel_cap", 200.0).value)
        for name in ("window_s", "refresh_hz", "gap_s", "linear_accel_cap", "angular_accel_cap"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        if self.subscription_depth < 1:
            raise ValueError("subscription_depth must be positive")

        self._lock = threading.Lock()
        self._origin_ns = None
        self._data = {}
        for name in ("odom", "twist", "accel"):
            self._data[name] = deque(maxlen=self.max_samples)
        self._received = {}
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=self.subscription_depth,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(
            Odometry, f"/optitrack/odom_{self.robot_id}", self.odom_callback, qos)
        self.twist_sub = self.create_subscription(
            TwistStamped, f"/optitrack/twist_{self.robot_id}", self.twist_callback, qos)
        self.accel_sub = self.create_subscription(
            AccelStamped, f"/optitrack/accel_{self.robot_id}", self.accel_callback, qos)
        self.get_logger().info(f"Plotting MPPI input topics for rigid body {self.robot_id}; no publishers")

    def _record(self, topic, msg, values):
        stamp = msg.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        received = time.monotonic()
        with self._lock:
            records = self._data[topic]
            if records and stamp_ns <= records[-1][0]:
                return
            if self._origin_ns is None:
                self._origin_ns = stamp_ns
            records.append((stamp_ns, *values))
            cutoff = stamp_ns - round(self.window_s * 1e9)
            while records and records[0][0] < cutoff:
                records.popleft()
            self._received[topic] = received

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        pitch = math.atan2(2.0 * (q.x*q.z + q.w*q.y),
                           1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        self._record("odom", msg, (msg.pose.pose.position.x, math.degrees(pitch)))

    def twist_callback(self, msg):
        self._record("twist", msg, (msg.twist.linear.x, msg.twist.angular.y))

    def accel_callback(self, msg):
        self._record("accel", msg, (msg.accel.linear.x, msg.accel.angular.y))

    def snapshot(self):
        with self._lock:
            origin = self._origin_ns
            records = {}
            for name, values in self._data.items():
                records[name] = list(values)
            received = dict(self._received)
        now = time.monotonic()
        age = {}
        for name in records:
            if name in received:
                age[name] = now - received[name]
            else:
                age[name] = None
        data = {}
        for name, values in records.items():
            if not values:
                data[name] = np.empty((0, 3))
                continue
            rows = []
            timestamps = []
            for stamp, first, second in values:
                elapsed = (stamp - origin) * 1e-9
                rows.append((elapsed, first, second))
                timestamps.append(stamp)
            array = np.asarray(rows, dtype=float)
            # Do not connect the trace across long gaps in message timestamps.
            stamps = np.asarray(timestamps, dtype=np.int64)
            breaks = np.flatnonzero(np.diff(stamps) > round(self.gap_s * 1e9)) + 1
            data[name] = np.insert(array, breaks, np.nan, axis=0)
        return data, age


def create_plot(node):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    panels = (
        ("odom", 1, "Position x [m]", "#2563eb"),
        ("odom", 2, "Pitch [deg]", "#9333ea"),
        ("twist", 1, "Speed vx [m/s]", "#15803d"),
        ("twist", 2, "omega_y [rad/s]", "#b45309"),
        ("accel", 1, "ax [m/s^2]", "#dc2626"),
        ("accel", 2, "alpha_y [rad/s^2]", "#0891b2"),
    )
    lines, clipped_lines = [], {}
    for i, (ax, (topic, column, label, color)) in enumerate(zip(axes.flat, panels)):
        line, = ax.plot([], [], color=color, linewidth=1.3, label="Topic")
        lines.append(line)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, node.window_s)
        ax.margins(y=0.15)
        if topic == "accel":
            clipped_lines[i], = ax.plot([], [], "k--", linewidth=1.0, label="MPPI clipped")
            ax.legend(loc="upper right", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Message time [s]")
    title = fig.suptitle("", fontsize=11)
    artists = tuple(lines) + tuple(clipped_lines.values())

    def update(_frame=None):
        data, age = node.snapshot()
        end = 0.0
        for array in data.values():
            if len(array) > 0:
                end = max(end, array[-1, 0])
        start = max(0.0, end - node.window_s)
        axes[0, 0].set_xlim(start, max(node.window_s, end))
        for i, (ax, line, (topic, column, _label, _color)) in enumerate(zip(axes.flat, lines, panels)):
            array = data[topic]
            visible = array[(array[:, 0] >= start) | np.isnan(array[:, 0])]
            line.set_data(visible[:, 0], visible[:, column])
            if i in clipped_lines:
                if column == 1:
                    cap = node.linear_accel_cap
                else:
                    cap = node.angular_accel_cap
                clipped_lines[i].set_data(visible[:, 0], np.clip(visible[:, column], -cap, cap))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        status_parts = []
        for name, seconds in age.items():
            if seconds is None:
                status_text = f"{name}: waiting"
            else:
                status_text = f"{name}: last received {seconds*1000:.0f} ms ago"
            status_parts.append(status_text)
        status = " | ".join(status_parts)
        title.set_text(f"MPPI inputs | rigid body {node.robot_id}\n{status}")
        return artists

    def resize(_event=None):
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    update()
    resize()
    fig.canvas.mpl_connect("resize_event", resize)
    return fig, update


def main(args=None):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    rclpy.init(args=args)
    node = executor = ros_thread = fig = None
    try:
        node = MPPIInputPlotNode()
        fig, update = create_plot(node)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except ExternalShutdownException:
                pass

        ros_thread = threading.Thread(target=spin, daemon=True)
        ros_thread.start()

        def refresh(frame):
            if not rclpy.ok():
                plt.close(fig)
                return ()
            return update(frame)

        animation = FuncAnimation(fig, refresh, interval=1000.0 / node.refresh_hz,
                                  cache_frame_data=False, blit=False)
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if ros_thread is not None:
            ros_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if fig is not None:
            plt.close(fig)


if __name__ == "__main__":
    main()
