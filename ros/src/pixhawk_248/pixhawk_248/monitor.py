#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3


class ImuRpyMonitor(Node):
    def __init__(self):
        super().__init__("imu_rpy_monitor")

        self.declare_parameter("topic", "/imu/rpy")
        self.declare_parameter("print_hz", 5.0)

        self.topic = self.get_parameter("topic").value
        self.print_hz = float(self.get_parameter("print_hz").value)

        self.sub = self.create_subscription(Vector3, self.topic, self.cb, 10)

        self.last_msg = None
        self.msg_count = 0
        self.last_rate_t = time.time()

        self.timer = self.create_timer(1.0 / max(self.print_hz, 1e-3), self.on_timer)

        self.get_logger().info(f"Monitoring {self.topic} | printing ~{self.print_hz:.1f} Hz")

    def cb(self, msg: Vector3):
        self.last_msg = msg
        self.msg_count += 1

    def on_timer(self):
        now = time.time()
        dt = now - self.last_rate_t
        if dt >= 1.0:
            rate = self.msg_count / dt
            self.msg_count = 0
            self.last_rate_t = now
            self.get_logger().info(f"{self.topic} rate ~ {rate:.1f} Hz")

        if self.last_msg is not None:
            self.get_logger().info(
                f"rpy(rad): roll={self.last_msg.x:+.3f} pitch={self.last_msg.y:+.3f} yaw={self.last_msg.z:+.3f}"
            )


def main():
    rclpy.init()
    node = ImuRpyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
