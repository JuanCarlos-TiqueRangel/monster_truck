#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


def clamp(x, lo, hi):
    return max(lo, min(hi, float(x)))


class CmdVelThrottleRamp(Node):
    def __init__(self):
        super().__init__("cmd_vel_throttle_ramp")

        self.declare_parameter("topic", "/cmd_vel")
        self.declare_parameter("hz", 20.0)

        # Ramp parameters
        self.declare_parameter("ramp_period_s", 6.0)   # full cycle up+down (triangle wave)
        self.declare_parameter("throttle_min", 0.0)    # 0..1 by default
        self.declare_parameter("throttle_max", 1.0)

        # Steering test (keep constant)
        self.declare_parameter("steer", 0.0)           # [-1..1]

        # Output mode:
        # "01" -> throttle in [0,1]
        # "pm1" -> throttle in [-1,1]
        self.declare_parameter("throttle_mode", "01")

        self.topic = self.get_parameter("topic").value
        self.hz = float(self.get_parameter("hz").value)
        self.ramp_period_s = float(self.get_parameter("ramp_period_s").value)
        self.th_min = float(self.get_parameter("throttle_min").value)
        self.th_max = float(self.get_parameter("throttle_max").value)
        self.steer = float(self.get_parameter("steer").value)
        self.mode = self.get_parameter("throttle_mode").value

        self.pub = self.create_publisher(Twist, self.topic, 10)
        self.t0 = self.get_clock().now()

        self.timer = self.create_timer(1.0 / max(self.hz, 1e-3), self.on_timer)

        self.get_logger().info(
            f"Publishing {self.topic} @ {self.hz:.1f} Hz | "
            f"triangle ramp period={self.ramp_period_s:.1f}s | "
            f"throttle=[{self.th_min},{self.th_max}] mode={self.mode} | steer={self.steer}"
        )

    def triangle01(self, phase):
        """Triangle wave in [0,1] given phase in [0,1)."""
        if phase < 0.5:
            return 2.0 * phase
        return 2.0 * (1.0 - phase)

    def on_timer(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9

        period = max(self.ramp_period_s, 1e-3)
        phase = (t % period) / period  # [0,1)

        tri = self.triangle01(phase)   # [0,1]
        throttle_01 = self.th_min + (self.th_max - self.th_min) * tri
        throttle_01 = clamp(throttle_01, 0.0, 1.0)

        if self.mode == "pm1":
            # map [0,1] -> [-1,1]
            throttle = 2.0 * throttle_01 - 1.0
        else:
            throttle = throttle_01  # [0,1]

        msg = Twist()
        msg.linear.x = float(throttle)
        msg.angular.z = float(clamp(self.steer, -1.0, 1.0))

        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelThrottleRamp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
