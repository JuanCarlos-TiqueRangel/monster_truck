#!/usr/bin/env python3
"""Raw OptiTrack pose and finite-difference motion, without filtering.

Motion is expressed in the child frame; RPY is in degrees. Timestamps
describe local reception, not camera capture. Acceleration needs three poses.
"""

import math
import threading
import time
from dataclasses import dataclass

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import AccelStamped, Point, Quaternion, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node

from NatNetClient import NatNetClient


@dataclass
class Sample:
    pos: tuple
    quat: tuple
    t: float
    stamp: Time


def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )


def rotate_vec_by_quat(v, q):
    vx, vy, vz = v
    vector_quaternion = (vx, vy, vz, 0.0)
    rotated = quat_mul(q, vector_quaternion)
    rotated = quat_mul(rotated, quat_conj(q))
    return (rotated[0], rotated[1], rotated[2])


def angular_velocity_from_quats(q_prev, q_curr, dt):
    """Shortest-arc angular velocity in world coordinates."""
    x, y, z, w = quat_mul(q_curr, quat_conj(q_prev))
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    norm = math.sqrt(x*x + y*y + z*z)
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    scale = 2.0 * math.atan2(norm, w) / (norm * dt)
    return (x * scale, y * scale, z * scale)


def rpy_degrees(q):
    x, y, z, w = q
    return (
        math.degrees(math.atan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y))),
        math.degrees(math.atan2(2.0*(x*z + w*y), 1.0 - 2.0*(y*y + z*z))),
        math.degrees(math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))),
    )


class OptiTrackOdometryNode(Node):
    def __init__(self):
        super().__init__("optitrack_odometry_node")
        self.client_ip = self.declare_parameter("client_ip", "192.168.0.20").value
        self.server_ip = self.declare_parameter("server_ip", "192.168.0.4").value
        self.robot_id = int(self.declare_parameter("robot_id", 810).value)
        self.use_multicast = bool(self.declare_parameter("use_multicast", True).value)
        self.frame_id = self.declare_parameter("frame_id", "map").value
        self.child_frame_id = self.declare_parameter("child_frame_id", "base_link").value
        self.publish_hz = float(self.declare_parameter("publish_hz", 300.0).value)
        if not math.isfinite(self.publish_hz) or self.publish_hz <= 0.0:
            raise ValueError("publish_hz must be finite and positive")

        self.odom_pub = self.create_publisher(Odometry, f"/optitrack/odom_{self.robot_id}", 10)
        self.rpy_pub = self.create_publisher(Vector3, f"/optitrack/rpy_{self.robot_id}", 10)
        self.twist_pub = self.create_publisher(TwistStamped, f"/optitrack/twist_{self.robot_id}", 10)
        self.accel_pub = self.create_publisher(AccelStamped, f"/optitrack/accel_{self.robot_id}", 10)

        self._lock = threading.Lock()
        self._latest = None
        self._prev_sample = None
        self._prev_motion = None

        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(self.client_ip)
        self.streaming_client.set_server_address(self.server_ip)
        self.streaming_client.set_use_multicast(self.use_multicast)
        self.streaming_client.rigid_body_listener = self._on_rigid_body
        if not self.streaming_client.run():
            raise RuntimeError("NatNetClient failed to start. Check IPs/multicast and Motive streaming.")

        self.timer = self.create_timer(1.0 / self.publish_hz, self._timer_cb)
        self.get_logger().info(f"Publishing raw OptiTrack pose and motion for rigid body {self.robot_id}")

    def _on_rigid_body(self, rigid_id, position, rotation_quaternion):
        if rigid_id != self.robot_id:
            return
        t = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        pos_x = float(position[0])
        pos_y = float(position[1])
        pos_z = float(position[2])
        quat_x = float(rotation_quaternion[0])
        quat_y = float(rotation_quaternion[1])
        quat_z = float(rotation_quaternion[2])
        quat_w = float(rotation_quaternion[3])
        for value in (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w):
            if not math.isfinite(value):
                return
        norm = math.sqrt(quat_x*quat_x + quat_y*quat_y + quat_z*quat_z + quat_w*quat_w)
        if norm < 1e-12 or not math.isfinite(norm):
            return
        pos = (pos_x, pos_y, pos_z)
        quat = (quat_x / norm, quat_y / norm, quat_z / norm, quat_w / norm)
        sample = Sample(pos, quat, t, stamp)
        with self._lock:
            self._latest = sample

    def _timer_cb(self):
        with self._lock:
            s = self._latest
        if s is None or s is self._prev_sample:
            return

        v_world = (0.0, 0.0, 0.0)
        w_world = (0.0, 0.0, 0.0)
        a_world = (0.0, 0.0, 0.0)
        alpha_world = (0.0, 0.0, 0.0)
        if self._prev_sample is not None:
            prev = self._prev_sample
            dt = s.t - prev.t
            if dt <= 1e-6:
                return
            vx_world = (s.pos[0] - prev.pos[0]) / dt
            vy_world = (s.pos[1] - prev.pos[1]) / dt
            vz_world = (s.pos[2] - prev.pos[2]) / dt
            v_world = (vx_world, vy_world, vz_world)
            w_world = angular_velocity_from_quats(prev.quat, s.quat, dt)
            if self._prev_motion is not None:
                prev_v, prev_w, prev_dt = self._prev_motion
                # Each finite-difference velocity belongs to its interval midpoint.
                motion_dt = 0.5 * (dt + prev_dt)
                ax_world = (v_world[0] - prev_v[0]) / motion_dt
                ay_world = (v_world[1] - prev_v[1]) / motion_dt
                az_world = (v_world[2] - prev_v[2]) / motion_dt
                a_world = (ax_world, ay_world, az_world)

                alpha_x_world = (w_world[0] - prev_w[0]) / motion_dt
                alpha_y_world = (w_world[1] - prev_w[1]) / motion_dt
                alpha_z_world = (w_world[2] - prev_w[2]) / motion_dt
                alpha_world = (alpha_x_world, alpha_y_world, alpha_z_world)
            self._prev_motion = (v_world, w_world, dt)
        self._prev_sample = s

        q_inv = quat_conj(s.quat)
        v_body = rotate_vec_by_quat(v_world, q_inv)
        w_body = rotate_vec_by_quat(w_world, q_inv)
        a_body = rotate_vec_by_quat(a_world, q_inv)
        alpha_body = rotate_vec_by_quat(alpha_world, q_inv)

        odom = Odometry()
        odom.header.stamp = s.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position = Point(x=s.pos[0], y=s.pos[1], z=s.pos[2])
        odom.pose.pose.orientation = Quaternion(x=s.quat[0], y=s.quat[1], z=s.quat[2], w=s.quat[3])

        twist = TwistStamped()
        twist.header.stamp = s.stamp
        twist.header.frame_id = self.child_frame_id
        twist.twist.linear = Vector3(x=v_body[0], y=v_body[1], z=v_body[2])
        twist.twist.angular = Vector3(x=w_body[0], y=w_body[1], z=w_body[2])
        odom.twist.twist = twist.twist

        accel = AccelStamped()
        accel.header.stamp = s.stamp
        accel.header.frame_id = self.child_frame_id
        accel.accel.linear = Vector3(x=a_body[0], y=a_body[1], z=a_body[2])
        accel.accel.angular = Vector3(x=alpha_body[0], y=alpha_body[1], z=alpha_body[2])

        roll, pitch, yaw = rpy_degrees(s.quat)
        rpy = Vector3(x=roll, y=pitch, z=yaw)
        self.odom_pub.publish(odom)
        self.rpy_pub.publish(rpy)
        self.twist_pub.publish(twist)
        self.accel_pub.publish(accel)

    def destroy_node(self):
        self.streaming_client.shutdown()
        super().destroy_node()


def main():
    rclpy.init()
    node = OptiTrackOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
