#!/usr/bin/env python3
"""Observe the existing OptiTrack topics without publishing or loading MPPI.

Run once with only the sensor publishers, then with the drive-disabled MPPI
node also running. Each row summarizes one report interval. Gaps at this
subscriber do not, by themselves, locate the cause at the sensor or publisher.
Uses the same BEST_EFFORT / KEEP_LAST(depth=1) QoS as the supplied controller.
"""

import argparse
import math
import statistics
import time
from dataclasses import dataclass, field


@dataclass
class StreamStats:
    name: str
    last_rx: float | None = None
    last_stamp: int | None = None
    highest_stamp: int | None = None
    last_advanced_rx: float | None = None
    messages: int = 0
    nonadvancing: int = 0
    rewinds: int = 0
    receipt_ms: list[float] = field(default_factory=list)
    stamp_ms: list[float] = field(default_factory=list)

    def observe(self, stamp_ns: int, now: float) -> None:
        self.messages += 1
        if self.last_rx is not None:
            self.receipt_ms.append(1000.0 * (now - self.last_rx))
        if self.last_stamp is not None:
            self.stamp_ms.append((stamp_ns - self.last_stamp) / 1e6)
            self.rewinds += int(stamp_ns < self.last_stamp)
        if self.highest_stamp is not None and stamp_ns <= self.highest_stamp:
            self.nonadvancing += 1
        else:
            self.highest_stamp = stamp_ns
            self.last_advanced_rx = now
        self.last_rx, self.last_stamp = now, stamp_ns

    def report(self, now: float) -> str:
        # Include silence up to the report, even when no new message closes it.
        age = math.inf if self.last_rx is None else 1000.0 * (now - self.last_rx)
        advanced_age = math.inf if self.last_advanced_rx is None else 1000.0 * (now - self.last_advanced_rx)
        gap = max(self.receipt_ms + [age])
        median_dt = statistics.median(self.stamp_ms) if self.stamp_ms else math.nan
        max_dt = max(self.stamp_ms) if self.stamp_ms else math.nan
        result = (f"{self.name}: n={self.messages} | rx_gap_max_ms={gap:.1f} | rx_age_ms={age:.1f} | "
                  f"stamp_dt_median_ms={median_dt:.1f} | stamp_dt_max_ms={max_dt:.1f} | "
                  f"nonadvancing={self.nonadvancing} | rewinds={self.rewinds} | advanced_age_ms={advanced_age:.1f}")
        self.messages = self.nonadvancing = self.rewinds = 0
        self.receipt_ms.clear()
        self.stamp_ms.clear()
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", type=int, default=810)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0.0:
        parser.error("--seconds must be finite and positive")

    # Imported only for the live observer; StreamStats can be tested without ROS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import TwistStamped

    class TimingObserver(Node):
        def __init__(self):
            super().__init__("optitrack_timing_observer")
            self.pose_stats = StreamStats("odom")
            self.twist_stats = StreamStats("twist")
            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pose_sub = self.create_subscription(Odometry, f"/optitrack/odom_{args.object_id}", self.pose_callback, qos)
            self.twist_sub = self.create_subscription(TwistStamped, f"/optitrack/twist_{args.object_id}", self.twist_callback, qos)
            self.report_timer = self.create_timer(1.0, self.print_report)

        def pose_callback(self, msg):
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.pose_stats.observe(stamp_ns, time.monotonic())

        def twist_callback(self, msg):
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            self.twist_stats.observe(stamp_ns, time.monotonic())

        def print_report(self):
            now = time.monotonic()
            print(self.pose_stats.report(now), flush=True)
            print(self.twist_stats.report(now), flush=True)

    print("READ-ONLY OBSERVER: no publishers, motor commands, model fitting, or CSV changes.", flush=True)
    print("n/nonadvancing/rewinds and interval statistics reset each report; age uses local monotonic time.", flush=True)
    print("nonadvancing counts stamps not newer than the largest previously seen stamp, like the controller guard.", flush=True)
    rclpy.init(args=[])
    node = None
    try:
        node = TimingObserver()
        finish = time.monotonic() + args.seconds
        while rclpy.ok() and time.monotonic() < finish:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
