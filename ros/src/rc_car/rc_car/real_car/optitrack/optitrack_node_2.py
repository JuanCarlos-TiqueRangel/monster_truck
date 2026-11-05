#!/usr/bin/env python3

import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import rclpy
from geometry_msgs.msg import Quaternion, Vector3, TwistStamped, AccelStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from NatNetClient import NatNetClient

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ============================================================
# DATA
# ============================================================

@dataclass
class Sample:
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    t: float


# ============================================================
# QUATERNION HELPERS
# ============================================================

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
    qn = quat_normalize(q)
    out = quat_mul(quat_mul(qn, (vx, vy, vz, 0.0)), quat_conj(qn))
    return (out[0], out[1], out[2])


# ============================================================
# ORIENTATION
#
# Negative pitch = nose up.
# ============================================================

def wheelie_pitch_from_quat(q):
    x, y, z, w = quat_normalize(q)

    return math.atan2(
        2.0 * (x*z + w*y),
        1.0 - 2.0 * (y*y + z*z),
    )


def roll_yaw_from_quat(q):
    x, y, z, w = quat_normalize(q)

    roll = math.atan2(
        2.0 * (w*x + y*z),
        1.0 - 2.0 * (x*x + y*y),
    )

    yaw = math.atan2(
        2.0 * (w*z + x*y),
        1.0 - 2.0 * (y*y + z*z),
    )

    return roll, yaw


# ============================================================
# QUATERNION ANGULAR VELOCITY
#
# Used only for roll and yaw.
# Pitch has its own Kalman estimator.
# ============================================================

def angular_velocity_from_quats(q_prev, q_curr, dt):
    if dt <= 1e-9:
        return (0.0, 0.0, 0.0)

    q0 = quat_normalize(q_prev)
    q1 = quat_normalize(q_curr)
    dq = quat_normalize(quat_mul(quat_conj(q0), q1))

    x, y, z, w = dq

    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w

    norm_v = math.sqrt(x*x + y*y + z*z)

    if norm_v < 1e-10:
        return (0.0, 0.0, 0.0)

    angle = 2.0 * math.atan2(norm_v, max(w, 1e-12))
    scale = angle / (norm_v * dt)

    return (x * scale, y * scale, z * scale)


# ============================================================
# RATE + ACCELERATION KALMAN
#
# state = [rate, acceleration]
# ============================================================

class RateAccelerationKalman:

    def __init__(self, measurement_std, jerk_std, rate_init_std, accel_init_std, gate_sigma=4.0):
        self.x = np.zeros(2, dtype=float)
        self.P = np.diag([rate_init_std**2, accel_init_std**2]).astype(float)

        self.r = float(measurement_std)**2
        self.q = float(jerk_std)**2
        self.gate_sigma = float(gate_sigma)
        self.initialized = False

    def update(self, rate_measured, dt):
        if not self.initialized:
            self.x[0] = float(rate_measured)
            self.x[1] = 0.0
            self.initialized = True
            return float(self.x[0]), float(self.x[1])

        F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ], dtype=float)

        G = np.array([
            [0.5 * dt * dt],
            [dt],
        ], dtype=float)

        Q = self.q * (G @ G.T)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        innovation = float(rate_measured - self.x[0])
        S = float(self.P[0, 0] + self.r)

        if abs(innovation) <= self.gate_sigma * math.sqrt(max(S, 1e-12)):
            K = self.P[:, 0] / S
            self.x = self.x + K * innovation
            self.P = self.P - np.outer(K, self.P[0, :])

        self.P = 0.5 * (self.P + self.P.T)

        return float(self.x[0]), float(self.x[1])


class PitchKalman:

    def __init__(
        self,
        angle_measurement_std=math.radians(0.3),
        jerk_std=25.0,
    ):

        # state = [theta, omega, alpha]
        self.x = np.zeros(3, dtype=float)

        self.P = np.diag([
            math.radians(1.0)**2,  # theta
            0.5**2,                # omega
            8.0**2,                # alpha
        ]).astype(float)

        self.r = angle_measurement_std**2
        self.q = jerk_std**2

        self.initialized = False

    def update(self, theta_measured, dt):

        if not self.initialized:

            self.x[0] = theta_measured
            self.x[1] = 0.0
            self.x[2] = 0.0

            self.initialized = True

            return (
                float(self.x[0]),
                float(self.x[1]),
                float(self.x[2]),
            )

        # Constant-angular-acceleration model
        F = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        # Jerk process noise
        G = np.array([
            [dt**3 / 6.0],
            [0.5 * dt**2],
            [dt],
        ], dtype=float)

        Q = self.q * (G @ G.T)

        # Prediction
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        # Angle innovation
        innovation = theta_measured - self.x[0]
        innovation = (
            innovation + math.pi
        ) % (2.0 * math.pi) - math.pi

        H = np.array([
            [1.0, 0.0, 0.0]
        ], dtype=float)

        S = (H @ self.P @ H.T + self.r).item()
        K = (self.P @ H.T / S).reshape(3)
        self.x = self.x + K * innovation

        I = np.eye(3)

        correction = (
            I - np.outer(K, H.reshape(3))
        )

        self.P = (
            correction
            @ self.P
            @ correction.T
            + self.r * np.outer(K, K)
        )

        self.P = 0.5 * (
            self.P + self.P.T
        )

        return (
            float(self.x[0]),  # theta
            float(self.x[1]),  # omega
            float(self.x[2]),  # alpha
        )

# ============================================================
# OPTITRACK NODE
# ============================================================

class OptiTrackOdometryNode(Node):

    def __init__(self):
        super().__init__("optitrack_odometry_node")

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        self.declare_parameter("client_ip", "192.168.0.16")
        self.declare_parameter("server_ip", "192.168.0.4")
        self.declare_parameter("robot_id", 810)
        self.declare_parameter("use_multicast", True)

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")

        self.declare_parameter("publish_hz", 300.0)
        self.declare_parameter("twist_in_body_frame", True)

        # Linear motion
        self.declare_parameter("linear_velocity_measurement_std", 0.08)
        self.declare_parameter("linear_jerk_std", 15.0)

        # Roll / yaw
        self.declare_parameter("angular_velocity_measurement_std", 0.05)
        self.declare_parameter("angular_jerk_std", 40.0)

        # Pitch
        self.declare_parameter("pitch_angle_measurement_std_deg", 0.55)
        self.declare_parameter("pitch_jerk_std", 15.0)

        self.declare_parameter("innovation_gate_sigma", 4.0)

        # ----------------------------------------------------
        # READ PARAMETERS
        # ----------------------------------------------------

        self.client_ip = self.get_parameter("client_ip").value
        self.server_ip = self.get_parameter("server_ip").value
        self.robot_id = int(self.get_parameter("robot_id").value)
        self.use_multicast = bool(self.get_parameter("use_multicast").value)

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value

        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.twist_in_body_frame = bool(self.get_parameter("twist_in_body_frame").value)

        linear_std = float(self.get_parameter("linear_velocity_measurement_std").value)
        linear_jerk_std = float(self.get_parameter("linear_jerk_std").value)

        angular_std = float(self.get_parameter("angular_velocity_measurement_std").value)
        angular_jerk_std = float(self.get_parameter("angular_jerk_std").value)

        pitch_angle_std_deg = float(self.get_parameter("pitch_angle_measurement_std_deg").value)
        pitch_angle_std = math.radians(pitch_angle_std_deg)
        pitch_jerk_std = float(self.get_parameter("pitch_jerk_std").value)

        gate_sigma = float(self.get_parameter("innovation_gate_sigma").value)

        # ----------------------------------------------------
        # PUBLISHERS
        # ----------------------------------------------------
        self.odom_pub = self.create_publisher(Odometry, f"/optitrack/odom_{self.robot_id}", 100)
        self.rpy_pub = self.create_publisher(Vector3, f"/optitrack/rpy_{self.robot_id}", 100)
        self.twist_pub = self.create_publisher(TwistStamped, f"/optitrack/twist_{self.robot_id}", 100)
        self.accel_pub = self.create_publisher(AccelStamped, f"/optitrack/accel_{self.robot_id}", 100)

        # ----------------------------------------------------
        # NATNET STATE
        # ----------------------------------------------------
        self._lock = threading.Lock()
        self._latest: Optional[Sample] = None
        self._prev_used: Optional[Sample] = None
        self._latest_used_t: Optional[float] = None

        # ----------------------------------------------------
        # LINEAR FILTERS
        # ----------------------------------------------------
        self._lin_kf_x = RateAccelerationKalman(linear_std, linear_jerk_std, 0.5, 5.0, gate_sigma)
        self._lin_kf_y = RateAccelerationKalman(linear_std, linear_jerk_std, 0.5, 5.0, gate_sigma)
        self._lin_kf_z = RateAccelerationKalman(linear_std, linear_jerk_std, 0.5, 5.0, gate_sigma)

        # ----------------------------------------------------
        # ROLL / YAW FILTERS
        # ----------------------------------------------------
        self._ang_kf_x = RateAccelerationKalman(angular_std, angular_jerk_std, 0.25, 8.0, gate_sigma)
        self._ang_kf_z = RateAccelerationKalman(angular_std, angular_jerk_std, 0.25, 8.0, gate_sigma)

        # ----------------------------------------------------
        # PITCH FILTER
        # ----------------------------------------------------
        self._pitch_kf = PitchKalman(
            angle_measurement_std=pitch_angle_std,
            jerk_std=pitch_jerk_std,
        )

        # ----------------------------------------------------
        # NATNET
        # ----------------------------------------------------

        self.streaming_client = NatNetClient()

        self.streaming_client.set_client_address(self.client_ip)
        self.streaming_client.set_server_address(self.server_ip)
        self.streaming_client.set_use_multicast(self.use_multicast)

        self.streaming_client.rigid_body_listener = self._on_rigid_body

        if not self.streaming_client.run():
            raise RuntimeError(
                "NatNetClient failed to start. Check IPs/multicast and Motive streaming."
            )

        # ----------------------------------------------------
        # INFO
        # ----------------------------------------------------

        self.get_logger().info(
            f"Publishing /optitrack/odom_{self.robot_id}, "
            f"/optitrack/rpy_{self.robot_id}, "
            f"/optitrack/twist_{self.robot_id}, "
            f"/optitrack/accel_{self.robot_id}"
        )

        self.get_logger().info(
            f"Linear KF: vel_std={linear_std:.3f} m/s | "
            f"jerk_std={linear_jerk_std:.1f} m/s^3"
        )

        self.get_logger().info(
            f"Pitch KF: angle_std={pitch_angle_std_deg:.3f} deg | "
            f"jerk_std={pitch_jerk_std:.1f} rad/s^3"
        )

        self.timer = self.create_timer(1.0 / max(1.0, self.publish_hz), self._timer_cb)

    # ========================================================
    # NATNET CALLBACK
    # ========================================================

    def _on_rigid_body(self, rigid_id, position, rotation_quaternion):
        if rigid_id != self.robot_id:
            return

        pos = (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )

        quat = quat_normalize((
            float(rotation_quaternion[0]),
            float(rotation_quaternion[1]),
            float(rotation_quaternion[2]),
            float(rotation_quaternion[3]),
        ))

        sample = Sample(pos=pos, quat=quat, t=time.monotonic())

        with self._lock:
            self._latest = sample

    # ========================================================
    # PUBLISH CALLBACK
    # ========================================================

    def _timer_cb(self):
        with self._lock:
            s = self._latest

        if s is None:
            return

        if self._latest_used_t is not None and abs(s.t - self._latest_used_t) < 1e-12:
            return

        prev = self._prev_used
        self._prev_used = s
        self._latest_used_t = s.t

        # ----------------------------------------------------
        # RAW MOTION
        # ----------------------------------------------------

        if prev is None:
            dt = 1.0 / max(self.publish_hz, 1.0)
            v_world_raw = (0.0, 0.0, 0.0)
            w_body_raw = (0.0, 0.0, 0.0)

        else:
            dt = s.t - prev.t

            if dt <= 1e-6:
                return

            v_world_raw = (
                (s.pos[0] - prev.pos[0]) / dt,
                (s.pos[1] - prev.pos[1]) / dt,
                (s.pos[2] - prev.pos[2]) / dt,
            )

            w_body_raw = angular_velocity_from_quats(prev.quat, s.quat, dt)

        # ----------------------------------------------------
        # LINEAR KALMAN
        # ----------------------------------------------------

        vx_world, ax_world = self._lin_kf_x.update(v_world_raw[0], dt)
        vy_world, ay_world = self._lin_kf_y.update(v_world_raw[1], dt)
        vz_world, az_world = self._lin_kf_z.update(v_world_raw[2], dt)

        v_world_filt = (vx_world, vy_world, vz_world)
        a_world_filt = (ax_world, ay_world, az_world)

        # ----------------------------------------------------
        # PITCH KALMAN
        # ----------------------------------------------------

        pitch_raw = wheelie_pitch_from_quat(s.quat)
        pitch_filt, wy_body, alphay_body = (self._pitch_kf.update(pitch_raw, dt,))

        # ----------------------------------------------------
        # ROLL / YAW ANGULAR KALMAN
        # ----------------------------------------------------

        wx_body, alphax_body = self._ang_kf_x.update(w_body_raw[0], dt)
        wz_body, alphaz_body = self._ang_kf_z.update(w_body_raw[2], dt)

        w_body_filt = (wx_body, wy_body, wz_body)
        alpha_body_filt = (alphax_body, alphay_body, alphaz_body,)

        # ----------------------------------------------------
        # BODY FRAME LINEAR MOTION
        # ----------------------------------------------------

        q_inv = quat_conj(s.quat)

        v_body_filt = rotate_vec_by_quat(v_world_filt, q_inv)
        a_body_filt = rotate_vec_by_quat(a_world_filt, q_inv)

        # ----------------------------------------------------
        # ODOM TWIST FRAME
        # ----------------------------------------------------

        if self.twist_in_body_frame:
            v_odom = v_body_filt
            w_odom = w_body_filt

        else:
            v_odom = v_world_filt
            w_odom = rotate_vec_by_quat(w_body_filt, s.quat)

        stamp = self.get_clock().now().to_msg()

        # ----------------------------------------------------
        # ODOMETRY
        # ----------------------------------------------------

        odom = Odometry()

        odom.header.stamp = stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = s.pos[0]
        odom.pose.pose.position.y = s.pos[1]
        odom.pose.pose.position.z = s.pos[2]

        odom.pose.pose.orientation = Quaternion(
            x=s.quat[0],
            y=s.quat[1],
            z=s.quat[2],
            w=s.quat[3],
        )

        odom.twist.twist.linear.x = v_odom[0]
        odom.twist.twist.linear.y = v_odom[1]
        odom.twist.twist.linear.z = v_odom[2]

        odom.twist.twist.angular.x = w_odom[0]
        odom.twist.twist.angular.y = w_odom[1]
        odom.twist.twist.angular.z = w_odom[2]

        # ----------------------------------------------------
        # RPY
        #
        # Published in degrees.
        # ----------------------------------------------------

        roll_raw, yaw_raw = roll_yaw_from_quat(s.quat)

        rpy = Vector3()

        rpy.x = math.degrees(roll_raw)
        rpy.y = math.degrees(pitch_filt)
        rpy.z = math.degrees(yaw_raw)

        # ----------------------------------------------------
        # FILTERED TWIST
        # ----------------------------------------------------

        twist = TwistStamped()

        twist.header.stamp = stamp
        twist.header.frame_id = self.child_frame_id

        twist.twist.linear.x = v_body_filt[0]
        twist.twist.linear.y = v_body_filt[1]
        twist.twist.linear.z = v_body_filt[2]

        twist.twist.angular.x = w_body_filt[0]
        twist.twist.angular.y = w_body_filt[1]
        twist.twist.angular.z = w_body_filt[2]

        # ----------------------------------------------------
        # FILTERED ACCELERATION
        # ----------------------------------------------------

        accel = AccelStamped()

        accel.header.stamp = stamp
        accel.header.frame_id = self.child_frame_id

        accel.accel.linear.x = a_body_filt[0]
        accel.accel.linear.y = a_body_filt[1]
        accel.accel.linear.z = a_body_filt[2]

        accel.accel.angular.x = alpha_body_filt[0]
        accel.accel.angular.y = alpha_body_filt[1]
        accel.accel.angular.z = alpha_body_filt[2]

        # ----------------------------------------------------
        # PUBLISH
        # ----------------------------------------------------

        self.odom_pub.publish(odom)
        self.rpy_pub.publish(rpy)
        self.twist_pub.publish(twist)
        self.accel_pub.publish(accel)

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def destroy_node(self):
        try:
            if hasattr(self.streaming_client, "shutdown"):
                self.streaming_client.shutdown()

        except Exception:
            pass

        super().destroy_node()


# ============================================================
# MAIN
# ============================================================

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