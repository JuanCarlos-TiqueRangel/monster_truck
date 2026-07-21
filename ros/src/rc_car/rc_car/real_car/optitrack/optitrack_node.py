#!/usr/bin/env python3
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3
from util import quaternion_to_euler



# Your existing NatNet client (same one you are already using)
from NatNetClient import NatNetClient


@dataclass
class Sample:
    pos: Tuple[float, float, float]          # (x,y,z)
    quat: Tuple[float, float, float, float]  # (qx,qy,qz,qw)
    rpy: Tuple[float, float, float]
    t: float                                 # monotonic time (s)


# ---------------- Quaternion helpers (x,y,z,w) ----------------
def quat_norm(q):
    x, y, z, w = q
    return math.sqrt(x*x + y*y + z*z + w*w)

def quat_normalize(q):
    x, y, z, w = q
    n = quat_norm(q)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

def quat_mul(q1, q2):
    # Hamilton product, both (x,y,z,w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def rotate_vec_by_quat(v, q):
    # rotate v (world) by q: v' = q * [v,0] * q_conj
    vx, vy, vz = v
    qv = (vx, vy, vz, 0.0)
    qn = quat_normalize(q)
    out = quat_mul(quat_mul(qn, qv), quat_conj(qn))
    return (out[0], out[1], out[2])

def angular_velocity_from_quats(q_prev, q_curr, dt):
    """
    Compute angular velocity (wx, wy, wz) in WORLD frame from quaternion difference.
    Using shortest-arc delta quaternion.
    """
    if dt <= 1e-9:
        return (0.0, 0.0, 0.0)

    q0 = quat_normalize(q_prev)
    q1 = quat_normalize(q_curr)

    # delta q = q0^{-1} * q1 = conj(q0) * q1 (since normalized)
    dq = quat_mul(quat_conj(q0), q1)
    dq = quat_normalize(dq)

    # shortest path: if w < 0, negate quaternion (same rotation, smaller angle)
    x, y, z, w = dq
    if w < 0.0:
        x, y, z, w = (-x, -y, -z, -w)

    # axis-angle
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)  # in [0, pi]
    s = math.sqrt(max(0.0, 1.0 - w*w))

    if s < 1e-8 or angle < 1e-8:
        return (0.0, 0.0, 0.0)

    ax, ay, az = (x / s, y / s, z / s)
    wx, wy, wz = (ax * angle / dt, ay * angle / dt, az * angle / dt)
    return (wx, wy, wz)


class OptiTrackOdometryNode(Node):
    def __init__(self):
        super().__init__("optitrack_odometry_node")

        # -------- Parameters --------
        self.declare_parameter("client_ip", "192.168.0.123")    # your Ubuntu IP
        self.declare_parameter("server_ip", "192.168.0.4")      # Motive PC IP
        self.declare_parameter("robot_id", 600)
        self.declare_parameter("use_multicast", True)

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_hz", 200.0)

        # If True: publish twist in BODY frame (base_link). If False: WORLD frame.
        self.declare_parameter("twist_in_body_frame", True)

        # Optional smoothing (0 = no smoothing, closer to 1 = more smoothing)
        self.declare_parameter("vel_smoothing_alpha", 0.0)

        self.client_ip = self.get_parameter("client_ip").get_parameter_value().string_value
        self.server_ip = self.get_parameter("server_ip").get_parameter_value().string_value
        self.robot_id = int(self.get_parameter("robot_id").value)
        self.use_multicast = bool(self.get_parameter("use_multicast").value)

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.twist_in_body_frame = bool(self.get_parameter("twist_in_body_frame").value)
        self.alpha = float(self.get_parameter("vel_smoothing_alpha").value)
        self.alpha = max(0.0, min(0.99, self.alpha))

        # -------- ROS publisher --------
        self.odom_pub = self.create_publisher(Odometry, "optitrack/odom", 10)
        self.pub_rpy = self.create_publisher(Vector3, "optitrack/rpy", 10)

        # -------- Data shared with NatNet thread --------
        self._lock = threading.Lock()
        self._latest: Optional[Sample] = None
        self._prev_used: Optional[Sample] = None
        self._latest_used_t: Optional[float] = None

        # filtered velocity
        self._v_filt = (0.0, 0.0, 0.0)
        self._w_filt = (0.0, 0.0, 0.0)

        # -------- NatNet setup --------
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(self.client_ip)
        self.streaming_client.set_server_address(self.server_ip)
        self.streaming_client.set_use_multicast(self.use_multicast)

        # attach callback
        self.streaming_client.rigid_body_listener = self._on_rigid_body

        ok = self.streaming_client.run()
        if not ok:
            raise RuntimeError("NatNetClient failed to start. Check IPs/multicast and Motive streaming.")

        self.get_logger().info(
            f"Streaming started. server_ip={self.server_ip} client_ip={self.client_ip} "
            f"robot_id={self.robot_id} multicast={self.use_multicast}"
        )

        # publish loop
        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self._timer_cb)

    def _on_rigid_body(self, rigid_id, position, rotation_quaternion):
        if rigid_id != self.robot_id:
            return

        # Expect: position=(x,y,z), rotation_quaternion=(qx,qy,qz,qw)
        pos = (float(position[0]), float(position[1]), float(position[2]))
        q = (
            float(rotation_quaternion[0]),
            float(rotation_quaternion[1]),
            float(rotation_quaternion[2]),
            float(rotation_quaternion[3]),
        )
        q = quat_normalize(q)

        roll, pitch, yaw = quaternion_to_euler(rotation_quaternion)
        rpy = (float(roll), float(pitch), float(yaw))

        s = Sample(pos=pos, quat=q, rpy=rpy, t=time.monotonic())
        with self._lock:
            self._latest = s

    def _timer_cb(self):
        with self._lock:
            s = self._latest

        if s is None:
            return

        # only compute/publish when we have a new sample
        if self._latest_used_t is not None and abs(s.t - self._latest_used_t) < 1e-12:
            return

        prev = self._prev_used
        self._prev_used = s
        self._latest_used_t = s.t

        if prev is None:
            v = (0.0, 0.0, 0.0)
            w = (0.0, 0.0, 0.0)
        else:
            dt = s.t - prev.t
            if dt <= 1e-6:
                return

            v_world = (
                (s.pos[0] - prev.pos[0]) / dt,
                (s.pos[1] - prev.pos[1]) / dt,
                (s.pos[2] - prev.pos[2]) / dt,
            )
            
            w_body = angular_velocity_from_quats(prev.quat, s.quat, dt)

            if self.twist_in_body_frame:
                q_inv = quat_conj(s.quat)
                v = rotate_vec_by_quat(v_world, q_inv,)

                # Already calculated in body coordinates
                w = w_body

            else:
                v = v_world

                # Convert body angular velocity to world coordinates
                w = rotate_vec_by_quat(w_body, s.quat,)

        # Optional smoothing
        if self.alpha > 0.0:
            self._v_filt = tuple(self.alpha * self._v_filt[i] + (1.0 - self.alpha) * v[i] for i in range(3))
            self._w_filt = tuple(self.alpha * self._w_filt[i] + (1.0 - self.alpha) * w[i] for i in range(3))
            v_out, w_out = self._v_filt, self._w_filt
        else:
            v_out, w_out = v, w

        # Build Odometry
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        msg.pose.pose.position.x = s.pos[0]
        msg.pose.pose.position.y = s.pos[1]
        msg.pose.pose.position.z = s.pos[2]

        msg.pose.pose.orientation = Quaternion(x=s.quat[0], y=s.quat[1], z=s.quat[2], w=s.quat[3])

        msg.twist.twist.linear.x = v_out[0]
        msg.twist.twist.linear.y = v_out[1]
        msg.twist.twist.linear.z = v_out[2]

        msg.twist.twist.angular.x = w_out[0]
        msg.twist.twist.angular.y = w_out[1]
        msg.twist.twist.angular.z = w_out[2]

        # Build Vector Orientation roll, pitch, yaw 
        rpy = Vector3()
        roll, pitch, yaw = s.rpy[0], s.rpy[1], s.rpy[2] 
        rpy.x, rpy.y, rpy.z = roll, pitch, yaw

        # Publish the information
        self.pub_rpy.publish(rpy)
        self.odom_pub.publish(msg)

    def destroy_node(self):
        # Try to stop NatNet client gracefully if it supports it
        try:
            if hasattr(self.streaming_client, "shutdown"):
                self.streaming_client.shutdown()
        except Exception:
            pass
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
