#!/usr/bin/env python3

import math
import time

import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32


SEND_PERIOD = 0.01
COMMAND_TIMEOUT = 0.3


class DriveNode(Node):
    def __init__(self):
        super().__init__("drive_node")

        self.declare_parameter("port", "/dev/ttyACM0")
        port = self.get_parameter("port").value

        self.ser = serial.Serial(
            port,
            115200,
            timeout=0.05,
            write_timeout=0.1,
        )
        time.sleep(0.2)
        self.ser.reset_input_buffer()

        self.throttle = 0.0
        self.last_command = 0.0
        self.armed = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,)

        self.create_subscription(Float32, "/drive", self.on_drive, qos)
        self.create_timer(SEND_PERIOD, self.on_timer)
        self.get_logger().info(f"{port} open, waiting for /drive")


    def arm(self):
        self.ser.write(b"ARM\n")

        deadline = time.monotonic() + 0.3

        while time.monotonic() < deadline:
            if self.ser.readline().startswith(b"OK"):
                self.armed = True
                self.get_logger().info("armed")
                return

        self.get_logger().error("arm failed")

    def on_drive(self, msg):
        if not math.isfinite(msg.data):
            self.get_logger().error(
                f"rejected non-finite command: {msg.data}"
            )
            return

        self.throttle = max(-1.0, min(1.0, msg.data))
        self.last_command = time.monotonic()

        if not self.armed:
            self.arm()

    def on_timer(self):
        age = time.monotonic() - self.last_command

        if age > COMMAND_TIMEOUT:
            value = 0
        else:
            value = int(round(self.throttle * 1000))

        self.ser.write(f"{value} 0\n".encode())

        pending = self.ser.in_waiting

        if pending:
            for raw in self.ser.read(pending).splitlines():
                if not raw:
                    continue

                text = raw.decode(errors="replace")

                if text.startswith("DEADMAN"):
                    self.armed = False

                self.get_logger().warn(text)

    def stop(self):
        self.ser.write(b"0 0\n")
        self.ser.write(b"DISARM\n")
        self.ser.flush()
        time.sleep(0.05)
        self.ser.close()


def main():
    rclpy.init()

    node = None

    try:
        node = DriveNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()