#!/usr/bin/env python3
import time
import math
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3

from pymavlink import mavutil


def set_message_interval(master, msg_id: int, hz: float):
    interval_us = int(1_000_000 / hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0
    )


def quat_from_rpy(roll, pitch, yaw):
    """Return quaternion (x,y,z,w) from roll,pitch,yaw."""
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class MavlinkAttitudeImuNode(Node):
    def __init__(self):
        super().__init__("mavlink_attitude_imu_node")

        # ------------ Params ------------
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 57600)

        # What we ask PX4 to send (best effort over radio)
        self.declare_parameter("request_hz", 50.0)

        # What we publish to ROS (constant)
        self.declare_parameter("pub_hz", 50.0)

        self.declare_parameter("imu_topic", "imu/data")
        self.declare_parameter("rpy_topic", "imu/rpy")
        self.declare_parameter("frame_id", "imu_link")

        # If no new ATTITUDE for this long, warn (seconds)
        self.declare_parameter("stale_s", 0.25)

        self.declare_parameter("print_stats", True)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.request_hz = float(self.get_parameter("request_hz").value)
        self.pub_hz = float(self.get_parameter("pub_hz").value)

        self.imu_topic = self.get_parameter("imu_topic").value
        self.rpy_topic = self.get_parameter("rpy_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.stale_s = float(self.get_parameter("stale_s").value)
        self.print_stats = bool(self.get_parameter("print_stats").value)

        # ------------ Publishers ------------
        self.pub_imu = self.create_publisher(Imu, self.imu_topic, 10)
        self.pub_rpy = self.create_publisher(Vector3, self.rpy_topic, 10)

        # ------------ Shared state (RX thread -> publish timer) ------------
        self._lock = threading.Lock()
        self._have = False
        self._last_rx_wall = 0.0

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.p = 0.0   # roll rate
        self.q = 0.0   # pitch rate
        self.r = 0.0   # yaw rate

        self._rx_count = 0
        self._rx_loss = float("nan")

        self._stop_evt = threading.Event()
        self._rx_thread = threading.Thread(target=self._mavlink_loop, daemon=True)

        # Constant publish timer
        self._pub_timer = self.create_timer(1.0 / max(self.pub_hz, 1.0), self._publish_cb)
        # Stats timer
        self._stats_timer = self.create_timer(1.0, self._stats_cb)

        self.get_logger().info(
            f"MAVLink {self.port}@{self.baud} | request ATTITUDE {self.request_hz:.1f} Hz | "
            f"publish {self.pub_hz:.1f} Hz | topics: {self.imu_topic}, {self.rpy_topic}"
        )
        self._rx_thread.start()

    def destroy_node(self):
        self._stop_evt.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        super().destroy_node()

    def _connect(self):
        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                m = mavutil.mavlink_connection(
                    self.port,
                    baud=self.baud,
                    robust_parsing=True,
                    autoreconnect=True,
                )
                hb = m.wait_heartbeat(timeout=10)
                if hb is None:
                    raise RuntimeError("No heartbeat within 10s")
                self.get_logger().info(f"Heartbeat OK (sys={m.target_system}, comp={m.target_component})")
                return m
            except Exception as e:
                self.get_logger().warn(f"Connect failed: {e}. Retrying in 2s...")
                time.sleep(2.0)
        return None

    def _mavlink_loop(self):
        m = self._connect()
        if m is None:
            return

        # Request ATTITUDE stream
        try:
            set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, self.request_hz)
        except Exception as e:
            self.get_logger().warn(f"SET_MESSAGE_INTERVAL(ATTITUDE) failed: {e}")

        while rclpy.ok() and not self._stop_evt.is_set():
            try:
                msg = m.recv_match(type="ATTITUDE", blocking=True, timeout=1.0)
                if msg is None:
                    continue

                now = time.time()

                # ATTITUDE fields (radians, rad/s)
                roll = float(msg.roll)
                pitch = float(msg.pitch)
                yaw = float(msg.yaw)
                p = float(msg.rollspeed)
                q = float(msg.pitchspeed)
                r = float(msg.yawspeed)

                with self._lock:
                    self.roll = roll
                    self.pitch = pitch
                    self.yaw = yaw
                    self.p = p
                    self.q = q
                    self.r = r

                    self._have = True
                    self._last_rx_wall = now
                    self._rx_count += 1
                    try:
                        self._rx_loss = m.packet_loss()
                    except Exception:
                        pass

            except Exception as e:
                self.get_logger().warn(f"MAVLink error: {e}. Reconnecting...")
                m = self._connect()
                if m is None:
                    return
                try:
                    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, self.request_hz)
                except Exception:
                    pass

    def _publish_cb(self):
        with self._lock:
            if not self._have:
                return
            roll, pitch, yaw = self.roll, self.pitch, self.yaw
            p, q, r = self.p, self.q, self.r
            last_rx = self._last_rx_wall

        age = time.time() - last_rx
        if age > self.stale_s:
            # avoid spamming every tick
            if int(time.time() * 2) % 2 == 0:
                self.get_logger().warn(f"ATTITUDE stale: {age:.2f}s since last sample")

        # Publish RPY
        rpy = Vector3()
        rpy.x = roll
        rpy.y = pitch
        rpy.z = yaw
        self.pub_rpy.publish(rpy)

        # Publish IMU
        qx, qy, qz, qw = quat_from_rpy(roll, pitch, yaw)

        imu = Imu()
        imu.header.stamp = self.get_clock().now().to_msg()
        imu.header.frame_id = self.frame_id

        imu.orientation.x = qx
        imu.orientation.y = qy
        imu.orientation.z = qz
        imu.orientation.w = qw

        imu.angular_velocity.x = p
        imu.angular_velocity.y = q
        imu.angular_velocity.z = r

        # We are NOT providing linear acceleration from ATTITUDE
        imu.linear_acceleration_covariance[0] = -1.0

        self.pub_imu.publish(imu)

    def _stats_cb(self):
        if not self.print_stats:
            return
        with self._lock:
            n = self._rx_count
            self._rx_count = 0
            loss = self._rx_loss
            last_rx = self._last_rx_wall

        # age = (time.time() - last_rx) if last_rx > 0 else float("inf")
        # self.get_logger().info(
        #     f"RX(ATTITUDE) ~ {n:.1f} Hz | pub {self.pub_hz:.1f} Hz | age {age:.2f}s | loss {loss:.1f}%"
        # )


def main():
    rclpy.init()
    node = MavlinkAttitudeImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
