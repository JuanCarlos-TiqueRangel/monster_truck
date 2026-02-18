#!/usr/bin/env python3
import os
import time
import math
import threading

# Must be set before importing pymavlink
os.environ["MAVLINK_DIALECT"] = "common"
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Imu


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
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def clip(v, lo=-1.0, hi=1.0):
    v = float(v)
    return max(lo, min(hi, v))


class MavlinkCarBridge(Node):
    def __init__(self):
        super().__init__("mavlink_car_bridge")

        # ---------- Params ----------
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 57600)

        # MAVLink stream request rate (best-effort over radio)
        self.declare_parameter("att_request_hz", 30.0)

        # ROS publish rates (constant)
        self.declare_parameter("imu_pub_hz", 50.0)
        self.declare_parameter("cmd_send_hz", 20.0)

        # Topics / frame
        self.declare_parameter("imu_topic", "imu/data")
        self.declare_parameter("rpy_topic", "imu/rpy")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("frame_id", "imu_link")

        # cmd_vel -> actuator mapping
        # throttle = throttle_scale * linear.x + throttle_trim
        # steer    = steer_scale    * angular.z + steer_trim
        self.declare_parameter("throttle_scale", 1.0)
        self.declare_parameter("steer_scale", 1.0)
        self.declare_parameter("throttle_trim", 0.0)
        self.declare_parameter("steer_trim", 0.0)

        # Safety: if no cmd_vel received recently -> neutral
        self.declare_parameter("cmd_timeout_s", 0.3)

        # Optional: treat cmd_vel as already normalized [-1,1]
        # If False, the scaling above maps from physical units (m/s, rad/s) to [-1,1]
        self.declare_parameter("cmd_is_normalized", True)

        self.declare_parameter("print_stats", True)
        self.declare_parameter("stale_att_s", 0.25)

        # Read params
        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)

        self.att_request_hz = float(self.get_parameter("att_request_hz").value)
        self.imu_pub_hz = float(self.get_parameter("imu_pub_hz").value)
        self.cmd_send_hz = float(self.get_parameter("cmd_send_hz").value)

        self.imu_topic = self.get_parameter("imu_topic").value
        self.rpy_topic = self.get_parameter("rpy_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.throttle_scale = float(self.get_parameter("throttle_scale").value)
        self.steer_scale = float(self.get_parameter("steer_scale").value)
        self.throttle_trim = float(self.get_parameter("throttle_trim").value)
        self.steer_trim = float(self.get_parameter("steer_trim").value)

        self.cmd_timeout_s = float(self.get_parameter("cmd_timeout_s").value)
        self.cmd_is_normalized = bool(self.get_parameter("cmd_is_normalized").value)

        self.print_stats = bool(self.get_parameter("print_stats").value)
        self.stale_att_s = float(self.get_parameter("stale_att_s").value)

        # ---------- ROS interfaces ----------
        self.pub_imu = self.create_publisher(Imu, self.imu_topic, 10)
        self.pub_rpy = self.create_publisher(Vector3, self.rpy_topic, 10)

        self.sub_cmd = self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_cb, 10)

        # Timers
        self._imu_timer = self.create_timer(1.0 / max(self.imu_pub_hz, 1.0), self._publish_imu_cb)
        self._cmd_timer = self.create_timer(1.0 / max(self.cmd_send_hz, 1.0), self._send_cmd_cb)
        self._stats_timer = self.create_timer(1.0, self._stats_cb)

        # ---------- Shared state ----------
        self._lock = threading.Lock()

        # Latest attitude from MAVLink ATTITUDE
        self._have_att = False
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.p = 0.0
        self.q = 0.0
        self.r = 0.0
        self._last_att_wall = 0.0

        # Latest cmd_vel
        self._last_cmd_wall = 0.0
        self._throttle_cmd = 0.0  # normalized [-1,1]
        self._steer_cmd = 0.0     # normalized [-1,1]

        # Stats
        self._rx_count = 0
        self._tx_count = 0
        self._loss = float("nan")

        # MAVLink connection + RX thread
        self._stop_evt = threading.Event()
        self._mav = None
        self._rx_thread = threading.Thread(target=self._mavlink_rx_loop, daemon=True)

        self.get_logger().info(
            f"Starting bridge: {self.port}@{self.baud} | "
            f"ATT req {self.att_request_hz:.1f}Hz | IMU pub {self.imu_pub_hz:.1f}Hz | CMD send {self.cmd_send_hz:.1f}Hz | "
            f"cmd_vel: {self.cmd_vel_topic}"
        )

        self._connect_mavlink()
        self._rx_thread.start()

        # Safety: send neutral once at startup
        self._send_actuators(set1=0.0, set2=0.0)

    # ---------- MAVLink ----------
    def _connect_mavlink(self):
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

                self.get_logger().info(f"Heartbeat OK (sys={m.target_system} comp={m.target_component})")

                # Request ATTITUDE
                try:
                    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, self.att_request_hz)
                except Exception as e:
                    self.get_logger().warn(f"SET_MESSAGE_INTERVAL(ATTITUDE) failed: {e}")

                self._mav = m
                return
            except Exception as e:
                self.get_logger().warn(f"MAVLink connect failed: {e}. Retrying in 2s...")
                time.sleep(2.0)

    def _mavlink_rx_loop(self):
        # Continuously read messages so buffers don't clog
        while rclpy.ok() and not self._stop_evt.is_set():
            if self._mav is None:
                self._connect_mavlink()
                continue

            try:
                msg = self._mav.recv_match(blocking=True, timeout=1.0)
                if msg is None:
                    continue
                t = msg.get_type()
                now = time.time()

                if t == "ATTITUDE":
                    with self._lock:
                        self.roll = float(msg.roll)
                        self.pitch = float(msg.pitch)
                        self.yaw = float(msg.yaw)
                        self.p = float(msg.rollspeed)
                        self.q = float(msg.pitchspeed)
                        self.r = float(msg.yawspeed)
                        self._have_att = True
                        self._last_att_wall = now
                        self._rx_count += 1
                        try:
                            self._loss = self._mav.packet_loss()
                        except Exception:
                            pass

            except Exception as e:
                self.get_logger().warn(f"MAVLink RX error: {e}. Reconnecting...")
                self._mav = None  # trigger reconnect

    def _send_actuators(self, *, set1=None, set2=None, set3=None, set4=None, set5=None, set6=None):
        """Non-blocking actuator send (no COMMAND_ACK wait)."""
        if self._mav is None:
            return

        CMD_DO_SET_ACTUATOR = getattr(mavutil.mavlink, "MAV_CMD_DO_SET_ACTUATOR", 187)
        vals = [set1, set2, set3, set4, set5, set6]
        params = [float('nan') if v is None else clip(v) for v in vals]

        self._mav.mav.command_long_send(
            self._mav.target_system,
            self._mav.target_component,
            CMD_DO_SET_ACTUATOR,
            0,
            params[0], params[1], params[2],
            params[3], params[4], params[5],
            float('nan')
        )
        with self._lock:
            self._tx_count += 1

    # ---------- ROS callbacks ----------
    def _cmd_cb(self, msg: Twist):
        # Take cmd_vel and map to normalized throttle/steer
        # By convention:
        #   throttle <- linear.x
        #   steer    <- angular.z
        lx = float(msg.linear.x)
        az = float(msg.angular.z)

        if self.cmd_is_normalized:
            th = lx
            st = az
        else:
            th = self.throttle_scale * lx + self.throttle_trim
            st = self.steer_scale * az + self.steer_trim

        th = clip(th + (0.0 if self.cmd_is_normalized else 0.0))  # keep clip only
        st = clip(st + (0.0 if self.cmd_is_normalized else 0.0))

        with self._lock:
            self._throttle_cmd = th
            self._steer_cmd = st
            self._last_cmd_wall = time.time()

    def _send_cmd_cb(self):
        # Send actuator command at fixed rate; failsafe to neutral if cmd_vel stale.
        with self._lock:
            th = self._throttle_cmd
            st = self._steer_cmd
            last_cmd = self._last_cmd_wall

        if (time.time() - last_cmd) > self.cmd_timeout_s:
            th, st = 0.0, 0.0

        self._send_actuators(set1=th, set2=st)

    def _publish_imu_cb(self):
        # Publish at constant rate using latest attitude
        with self._lock:
            if not self._have_att:
                return
            roll, pitch, yaw = self.roll, self.pitch, self.yaw
            p, q, r = self.p, self.q, self.r
            last_att = self._last_att_wall

        age = time.time() - last_att
        if age > self.stale_att_s:
            # don't spam every tick
            if int(time.time() * 2) % 2 == 0:
                self.get_logger().warn(f"ATTITUDE stale: {age:.2f}s")

        # RPY topic
        rpy = Vector3()
        rpy.x, rpy.y, rpy.z = roll, pitch, yaw
        self.pub_rpy.publish(rpy)

        # IMU message
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

        # No accel in ATTITUDE
        imu.linear_acceleration_covariance[0] = -1.0

        self.pub_imu.publish(imu)

    def _stats_cb(self):
        if not self.print_stats:
            return
        with self._lock:
            rx = self._rx_count
            tx = self._tx_count
            self._rx_count = 0
            self._tx_count = 0
            loss = self._loss
            last_att = self._last_att_wall

        age = (time.time() - last_att) if last_att > 0 else float("inf")
        self.get_logger().info(
            f"RX(ATT) {rx:.0f}/s | TX(cmd) {tx:.0f}/s | loss {loss:.1f}% | att_age {age:.2f}s"
        )

    def destroy_node(self):
        # Safety: neutral on shutdown
        try:
            self._send_actuators(set1=0.0, set2=0.0)
        except Exception:
            pass
        self._stop_evt.set()
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=2.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = MavlinkCarBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
