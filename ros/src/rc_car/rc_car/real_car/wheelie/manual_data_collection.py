# #!/usr/bin/env python3
# """Manual wheelie data collection: joystick -> /drive_manual, OptiTrack -> CSV."""

# import csv
# import math
# import time
# from datetime import datetime
# from pathlib import Path

# import rclpy
# from geometry_msgs.msg import AccelStamped, Twist, TwistStamped
# from nav_msgs.msg import Odometry
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy
# from sensor_msgs.msg import Joy


# # ============================================================
# # SETTINGS
# # ============================================================

# ROOT = Path(__file__).resolve().parent
# OBJECT_ID = 810
# JOY_TOPIC = "/joy"
# DRIVE_TOPIC = "/drive_manual"
# DRIVE_AXIS = 1

# RT_AXIS = 5
# LT_AXIS = 4

# STEERING_AXIS = 2
# INVERT_DRIVE = False
# INVERT_STEERING = True
# MAX_PWM = 0.35
# MAX_STEERING_RAD = 1.2
# DRIVE_PERIOD_S = 0.04
# DEADMAN_BUTTON = 9              # Hold LB to move; set None to disable


# # ============================================================
# # NODE
# # ============================================================

# class WheelieJoyCollector(Node):
#     def __init__(self):
#         super().__init__("wheelie_joy_collector")

#         self.u = 0.0
#         self.steering = 0.0
#         self.triggers_ready = False
#         self.enabled = True
#         self.x = 0.0
#         self.v = 0.0
#         self.theta = 0.0
#         self.omega = 0.0
#         self.v_dot = 0.0
#         self.omega_dot = 0.0
#         self.have_odom = False
#         self.have_twist = False
#         self.have_accel = False
#         self.started_at = time.monotonic()

#         self.rt_initialized = False
#         self.lt_initialized = False

#         stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.log_path = ROOT / f"rls_identification_{stamp}.csv"
#         self.log_file = self.log_path.open("w", newline="", encoding="utf-8")
#         self.writer = csv.writer(self.log_file)
#         self.writer.writerow([
#             "elapsed_s", "sequence_index", "u", "x", "v", "theta_rad",
#             "theta_deg", "omega", "v_dot", "omega_dot", "steering",
#         ])

#         qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
#         self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, 10)
#         self.create_subscription(Odometry, f"/optitrack/odom_{OBJECT_ID}", self.odom_callback, qos)
#         self.create_subscription(TwistStamped, f"/optitrack/twist_{OBJECT_ID}", self.twist_callback, qos)
#         self.create_subscription(AccelStamped, f"/optitrack/accel_{OBJECT_ID}", self.accel_callback, qos)
#         self.drive_pub = self.create_publisher(Twist, DRIVE_TOPIC, 10)
#         self.timer = self.create_timer(DRIVE_PERIOD_S, self.timer_callback)
#         self.get_logger().info(f"Publishing {DRIVE_TOPIC}; saving {self.log_path}")


#     # def joy_callback(self, msg):
#     #     drive = float(msg.axes[DRIVE_AXIS]) if len(msg.axes) > DRIVE_AXIS else 0.0
#     #     steer = float(msg.axes[STEERING_AXIS]) if len(msg.axes) > STEERING_AXIS else 0.0
#     #     if INVERT_DRIVE:
#     #         drive = -drive
#     #     if INVERT_STEERING:
#     #         steer = -steer
#     #     self.u = MAX_PWM * max(-1.0, min(1.0, drive))
#     #     self.steering = MAX_STEERING_RAD * max(-1.0, min(1.0, steer))
#     #     self.enabled = DEADMAN_BUTTON is None or (len(msg.buttons) > DEADMAN_BUTTON and bool(msg.buttons[DEADMAN_BUTTON]))


#     def joy_callback(self, msg):
#         rt = float(msg.axes[RT_AXIS])
#         lt = float(msg.axes[LT_AXIS])
#         steer = float(msg.axes[STEERING_AXIS])

#         rt = (1.0 - msg.axes[RT_AXIS]) / 2.0
#         lt = (1.0 - msg.axes[LT_AXIS]) / 2.0

#         if INVERT_STEERING:
#             steer = -steer

#         self.u = MAX_PWM * (rt - lt)
#         self.steering = MAX_STEERING_RAD * max(-1.0, min(1.0, steer))

#         print("throttle: ", self.u)
#         print("steering: ", self.steering)
#         print(" ")


#     def odom_callback(self, msg):
#         q = msg.pose.pose.orientation
#         self.theta = math.atan2(2.0 * (q.x * q.z + q.w * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
#         self.x = float(msg.pose.pose.position.x)
#         self.have_odom = True


#     def twist_callback(self, msg):
#         self.v = float(msg.twist.linear.x)
#         self.omega = float(msg.twist.angular.y)
#         self.have_twist = True


#     def accel_callback(self, msg):
#         self.v_dot = float(msg.accel.linear.x)
#         self.omega_dot = float(msg.accel.angular.y)
#         self.have_accel = True


#     def timer_callback(self):
#         u = self.u if self.enabled else 0.0
#         steering = self.steering if self.enabled else 0.0
#         msg = Twist()
#         msg.linear.x = u
#         msg.angular.z = steering
#         self.drive_pub.publish(msg)

#         if self.have_odom and self.have_twist and self.have_accel:
#             self.writer.writerow([
#                 time.monotonic() - self.started_at, -1, u, self.x, self.v,
#                 self.theta, math.degrees(self.theta), self.omega, self.v_dot,
#                 self.omega_dot, steering,
#             ])


#     def close(self):
#         self.timer.cancel()
#         for _ in range(3):
#             self.drive_pub.publish(Twist())
#         self.log_file.close()
#         self.get_logger().info(f"Saved {self.log_path}")


# def main():
#     rclpy.init()
#     node = WheelieJoyCollector()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.close()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()






























#!/usr/bin/env python3
"""Manual wheelie control or pitch PID; OptiTrack measurements -> CSV."""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import AccelStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent
OBJECT_ID = 810
JOY_TOPIC = "/joy"
DRIVE_TOPIC = "/drive_manual"

RT_AXIS = 5
LT_AXIS = 4
STEERING_AXIS = 2
INVERT_STEERING = True

MAX_PWM = 0.35
MAX_STEERING_RAD = 1.2
DRIVE_PERIOD_S = 0.04

PID_BUTTON = 0                 # Press once: PID on; press again: manual
PITCH_REF_DEG = -25.0         # Negative pitch = front wheels up
KP = 0.35                     # Initial guesses; NOT tuned on the real car
KI = 0.0
KD = 0.06
INTEGRAL_LIMIT = 0.5
PID_SENSOR_TIMEOUT_S = 0.20   # In PID mode, don't use stale OptiTrack state


# ============================================================
# NODE
# ============================================================

class WheelieJoyCollector(Node):
    def __init__(self):
        super().__init__("wheelie_joy_collector")

        self.manual_u = 0.0
        self.u = 0.0
        self.steering = 0.0
        self.x = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.v_dot = 0.0
        self.omega_dot = 0.0
        self.have_odom = False
        self.have_twist = False
        self.have_accel = False
        self.last_odom_time = None
        self.last_twist_time = None
        self.started_at = time.monotonic()

        self.pid_enabled = False
        self.previous_pid_button = False
        self.integral_error = 0.0
        self.last_pid_time = None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = ROOT / f"rls_identification_{stamp}.csv"
        self.log_file = self.log_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.log_file)
        self.writer.writerow([
            "elapsed_s", "sequence_index", "u", "x", "v", "theta_rad",
            "theta_deg", "omega", "v_dot", "omega_dot", "steering",
        ])

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Joy, JOY_TOPIC, self.joy_callback, 10)
        self.create_subscription(Odometry, f"/optitrack/odom_{OBJECT_ID}", self.odom_callback, qos)
        self.create_subscription(TwistStamped, f"/optitrack/twist_{OBJECT_ID}", self.twist_callback, qos)
        self.create_subscription(AccelStamped, f"/optitrack/accel_{OBJECT_ID}", self.accel_callback, qos)
        self.drive_pub = self.create_publisher(Twist, DRIVE_TOPIC, 10)
        self.timer = self.create_timer(DRIVE_PERIOD_S, self.timer_callback)
        self.get_logger().info(f"Publishing {DRIVE_TOPIC}; saving {self.log_path}")
        self.get_logger().info(f"Manual by default. Button {PID_BUTTON} toggles pitch PID, target {PITCH_REF_DEG} deg")


    def joy_callback(self, msg):
        if len(msg.axes) <= max(RT_AXIS, LT_AXIS, STEERING_AXIS):
            return

        rt = (1.0 - float(msg.axes[RT_AXIS])) / 2.0
        lt = (1.0 - float(msg.axes[LT_AXIS])) / 2.0
        steer = float(msg.axes[STEERING_AXIS])

        if INVERT_STEERING:
            steer = -steer

        self.manual_u = MAX_PWM * max(-1.0, min(1.0, rt - lt))
        self.steering = MAX_STEERING_RAD * max(-1.0, min(1.0, steer))

        button_pressed = len(msg.buttons) > PID_BUTTON and bool(msg.buttons[PID_BUTTON])
        if button_pressed and not self.previous_pid_button:
            if self.pid_enabled:
                self.pid_enabled = False
                self.get_logger().info("Pitch PID OFF: manual throttle")
            elif self.have_odom and self.have_twist:
                self.pid_enabled = True
                self.integral_error = 0.0
                self.last_pid_time = None
                self.get_logger().info("Pitch PID ON")
            else:
                self.get_logger().warn("Cannot enable PID without OptiTrack odometry and twist")
        self.previous_pid_button = button_pressed


    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.theta = math.atan2(2.0 * (q.x * q.z + q.w * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.x = float(msg.pose.pose.position.x)
        self.have_odom = True
        self.last_odom_time = time.monotonic()


    def twist_callback(self, msg):
        self.v = float(msg.twist.linear.x)
        self.omega = float(msg.twist.angular.y)
        self.have_twist = True
        self.last_twist_time = time.monotonic()


    def accel_callback(self, msg):
        self.v_dot = float(msg.accel.linear.x)
        self.omega_dot = float(msg.accel.angular.y)
        self.have_accel = True


    def pitch_pid(self, now):
        dt = DRIVE_PERIOD_S
        if self.last_pid_time is not None:
            dt = now - self.last_pid_time
        self.last_pid_time = now

        error = self.theta - math.radians(PITCH_REF_DEG)
        next_integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, self.integral_error + error * dt))
        candidate = KP * error + KI * next_integral + KD * self.omega

        if abs(candidate) <= MAX_PWM or (candidate > MAX_PWM and error < 0.0) or (candidate < -MAX_PWM and error > 0.0):
            self.integral_error = next_integral

        command = KP * error + KI * self.integral_error + KD * self.omega
        return max(-MAX_PWM, min(MAX_PWM, command))


    def timer_callback(self):
        now = time.monotonic()
        u = self.manual_u

        if self.pid_enabled:
            odom_stale = self.last_odom_time is None or now - self.last_odom_time > PID_SENSOR_TIMEOUT_S
            twist_stale = self.last_twist_time is None or now - self.last_twist_time > PID_SENSOR_TIMEOUT_S
            if odom_stale or twist_stale:
                self.pid_enabled = False
                self.manual_u = 0.0
                self.integral_error = 0.0
                u = 0.0
                self.get_logger().error("OptiTrack data stale: PID disabled, publishing neutral")
            else:
                u = self.pitch_pid(now)

        self.u = u
        msg = Twist()
        msg.linear.x = u
        msg.angular.z = self.steering
        self.drive_pub.publish(msg)

        if self.have_odom and self.have_twist and self.have_accel:
            self.writer.writerow([
                now - self.started_at, -1, u, self.x, self.v,
                self.theta, math.degrees(self.theta), self.omega, self.v_dot,
                self.omega_dot, self.steering,
            ])


    def close(self):
        self.timer.cancel()
        for _ in range(3):
            self.drive_pub.publish(Twist())
        self.log_file.close()
        self.get_logger().info(f"Saved {self.log_path}")


def main():
    rclpy.init()
    node = WheelieJoyCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
