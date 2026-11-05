#!/usr/bin/env python3

"""
identify_rls.py

Online RLS identification for the RC wheelie car.

What it does
------------
1. Publishes a predefined normalized PWM command u on /drive.
2. Subscribes to OptiTrack odometry, twist, and acceleration.
3. Builds the state:
       state = [x, v, theta, omega]
   with theta in radians and omega in rad/s.
4. Updates the 10 RLS coefficients online using:
       v_dot     = f_v(state, u)
       omega_dot = f_w(state, u)
5. Saves:
       rls_weights.npz
   when the experiment finishes or Ctrl+C is pressed.

Input convention
----------------
u in [-1, 1]
    0.0 -> 0 %
    1.0 -> 100 % forward command

The original sequence below was expressed using MAX_DRIVE_RAW = 0.12.
It is normalized automatically:
    u = raw_drive / MAX_DRIVE_RAW
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, AccelStamped
from std_msgs.msg import Float32

from params_mppi import WheelieParams, RLSConfig
from rls import nominal_rls_parameters, rls_update


# ============================================================
# PATHS / SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

OBJECT_ID = 810

DRIVE_PERIOD_S = 0.04       # 25 Hz excitation command
INITIAL_REST_S = 1.0
FINAL_REST_S = 1.0

# The OLD drive sequence used 0.12 as 100 %.
MAX_DRIVE_RAW = 0.470

# Safety
ARM_PITCH_DEG = -15.0
STOP_PITCH_DEG = -65.0
STOP_CONFIRM_SAMPLES = 2

# Optional simple measurement sanity limits.
ACCEL_CAP_V = 30.0           # m/s^2
ACCEL_CAP_W = 300.0          # rad/s^2

# Save weights beside this script.
WEIGHTS_PATH = ROOT / "rls_weights.npz"


# ============================================================
# ORIGINAL DRIVE SEQUENCE
# ============================================================

RAW_DRIVE_SEQUENCE = [
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.110,
    0.060,
    0.060,
    0.025,
    0.010,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.020,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.100,
    0.100,
    0.100,
    0.090,
    0.090,
    0.060,
    0.060,
    0.060,
    0.060,
    0.060,
    0.070,
    0.048,
    0.060,
    0.072,
    0.084,
    0.096,
    0.084,
    0.072,
    0.060,
    0.048,
    0.036,
    0.000,
    0.000,
    0.048,
    0.060,
    0.072,
    0.084,
    0.096,
    0.084,
    0.072,
    0.060,
    0.048,
    0.036,
    0.000,
    0.000,
    0.048,
    0.060,
    0.072,
    0.084,
    0.096,
    0.084,
    0.072,
    0.060,
    0.048,
    0.036,
    0.000,
    0.000,
    0.048,
    0.060,
    0.072,
    0.084,
    0.096,
    0.084,
    0.072,
    0.060,
    0.048,
    0.036,
    0.000,
    0.000,
    0.048,
    0.060,
    0.072,
    0.084,
    0.096,
    0.084,
]

# Normalize the exact sequence to the MPPI convention u in [-1, 1].
DRIVE_SEQUENCE = [
    float(np.clip(raw / MAX_DRIVE_RAW, -1.0, 1.0))
    for raw in RAW_DRIVE_SEQUENCE
]

SEQUENCE_DURATION_S = len(DRIVE_SEQUENCE) * DRIVE_PERIOD_S
EXPERIMENT_DURATION_S = INITIAL_REST_S + SEQUENCE_DURATION_S + FINAL_REST_S


# ============================================================
# NODE
# ============================================================

class RLSIdentificationNode(Node):

    def __init__(self):
        super().__init__("rls_identification")

        self.p = WheelieParams()
        self.rls_cfg = RLSConfig()

        # ---------------------------------------------------
        # RLS initialization
        # ----------------------------------------------------
        if WEIGHTS_PATH.is_file():

            with np.load(WEIGHTS_PATH) as data:

                self.rls_parameters = (data["learned_parameters"].copy())

                if "covariance" in data:
                    self.rls_covariance = (
                        data["covariance"].copy()
                    )
                else:
                    self.rls_covariance = (
                        self.rls_cfg.initial_covariance
                        * np.eye(
                            self.rls_parameters.size,
                            dtype=float,
                        )
                    )

                if "update_count" in data:
                    self.rls_update_count = int(
                        data["update_count"][0]
                    )
                else:
                    self.rls_update_count = 0

            print()
            print(f"Loaded previous RLS weights from: " f"{WEIGHTS_PATH}")
            print(f"Previous RLS updates: " f"{self.rls_update_count}")
            print()

        else:

            self.rls_parameters = (nominal_rls_parameters(self.p))
            self.rls_covariance = (self.rls_cfg.initial_covariance * np.eye(self.rls_parameters.size, dtype=float,))

            self.rls_update_count = 0

            print()
            print("No previous RLS weights found.")
            print("Starting from nominal parameters.")
            print()

        # ----------------------------------------------------
        # Latest measurements
        # ----------------------------------------------------
        self.x = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.omega = 0.0

        self.v_dot = 0.0
        self.omega_dot = 0.0

        self.odom_received = False
        self.twist_received = False
        self.accel_received = False

        # Previous regressor state/input.
        self.prev_state = None
        self.prev_u = None

        # Current command actually applied to the vehicle.
        self.current_u = 0.0
        self.current_sequence_index = -1

        # Experiment state.
        self.started_at = None
        self.finished = False

        # Safety state.
        self.flip_armed = False
        self.flip_counter = 0

        # ----------------------------------------------------
        # ROS
        # ----------------------------------------------------
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f"/optitrack/odom_{OBJECT_ID}",
            self.odom_callback,
            qos,
        )

        self.twist_sub = self.create_subscription(
            TwistStamped,
            f"/optitrack/twist_{OBJECT_ID}",
            self.twist_callback,
            qos,
        )

        self.accel_sub = self.create_subscription(
            AccelStamped,
            f"/optitrack/accel_{OBJECT_ID}",
            self.accel_callback,
            qos,
        )

        self.drive_pub = self.create_publisher(
            Float32,
            "/drive",
            10,
        )

        self.timer = self.create_timer(
            DRIVE_PERIOD_S,
            self.timer_callback,
        )

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = ROOT / f"rls_identification_{timestamp}.csv"
        self.log_file = self.log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self.writer = csv.writer(self.log_file)

        header = [
            "elapsed_s",
            "sequence_index",
            "u",
            "x",
            "v",
            "theta_rad",
            "theta_deg",
            "omega",
            "v_dot",
            "omega_dot",
            "rls_updates",
        ]
        header += [f"a{i}" for i in range(10)]
        self.writer.writerow(header)

        print()
        print("==============================================")
        print(" ONLINE RLS IDENTIFICATION")
        print("==============================================")
        print(f"Object ID        : {OBJECT_ID}")
        print(f"Command period   : {DRIVE_PERIOD_S:.3f} s")
        print(f"Sequence samples : {len(DRIVE_SEQUENCE)}")
        print(f"Sequence time    : {SEQUENCE_DURATION_S:.2f} s")
        print(f"Weights output   : {WEIGHTS_PATH}")
        print(f"Log output       : {self.log_path}")
        print()
        print("Initial RLS parameters:")
        print(self.rls_parameters)
        print()

    # ========================================================
    # CALLBACKS
    # ========================================================

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation

        # Pitch in radians.
        self.theta = math.atan2(
            2.0 * (q.x * q.z + q.w * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        self.x = float(msg.pose.pose.position.x)
        self.odom_received = True

        self._check_pitch_safety()

    def twist_callback(self, msg):
        self.v = float(msg.twist.linear.x)
        self.omega = float(msg.twist.angular.y)
        self.twist_received = True

    def accel_callback(self, msg):
        self.v_dot = float(msg.accel.linear.x)
        self.omega_dot = float(msg.accel.angular.y)
        self.accel_received = True

    # ========================================================
    # STATE / RLS
    # ========================================================

    def measurements_ready(self):
        return (
            self.odom_received
            and self.twist_received
            and self.accel_received
        )

    def current_state(self):
        return np.array(
            [
                self.x,
                self.v,
                self.theta,
                self.omega,
            ],
            dtype=float,
        )

    def update_rls(self):
        """
        Update using the previous state/input and the latest measured
        accelerations.

        This preserves the causal pairing:
            (x_{k-1}, u_{k-1}) -> measured acceleration at k
        """
        if (
            self.prev_state is None
            or self.prev_u is None
            or not self.measurements_ready()
        ):
            return None

        if (
            not np.isfinite(self.v_dot)
            or not np.isfinite(self.omega_dot)
            or abs(self.v_dot) > ACCEL_CAP_V
            or abs(self.omega_dot) > ACCEL_CAP_W
        ):
            return None

        self.rls_parameters, self.rls_covariance, info = rls_update(
            self.prev_state,
            self.prev_u,
            self.v_dot,
            self.omega_dot,
            self.rls_parameters,
            self.rls_covariance,
            forgetting_factor=self.rls_cfg.forgetting_factor,
            sigma_v_dot=self.rls_cfg.sigma_v_dot,
            sigma_omega_dot=self.rls_cfg.sigma_omega_dot,
            update_angular=True,
        )

        if not info["skipped"]:
            self.rls_update_count += 1

        return info

    # ========================================================
    # INPUT SEQUENCE
    # ========================================================

    def get_command(self, elapsed):
        if elapsed < INITIAL_REST_S:
            return 0.0, -1

        sequence_time = elapsed - INITIAL_REST_S

        if sequence_time >= SEQUENCE_DURATION_S:
            return 0.0, -1

        index = int(sequence_time / DRIVE_PERIOD_S)
        index = min(index, len(DRIVE_SEQUENCE) - 1)

        return DRIVE_SEQUENCE[index], index

    def publish_u(self, u):
        u = float(np.clip(u, -1.0, 1.0))

        msg = Float32()
        msg.data = u
        self.drive_pub.publish(msg)

        self.current_u = u

    # ========================================================
    # TIMER
    # ========================================================

    def timer_callback(self):
        if self.finished:
            return

        now = time.monotonic()

        if self.started_at is None:
            self.started_at = now

        elapsed = now - self.started_at

        if elapsed >= EXPERIMENT_DURATION_S:
            self.finish("experiment complete")
            return

        # Do not drive until the required sensors have reported at least once.
        if not self.measurements_ready():
            self.publish_u(0.0)
            return

        # ----------------------------------------------------
        # 1. Use the transition generated by the PREVIOUS u
        #    to update RLS.
        # ----------------------------------------------------
        info = self.update_rls()

        # ----------------------------------------------------
        # 2. Read the current state.
        # ----------------------------------------------------
        state = self.current_state()

        # ----------------------------------------------------
        # 3. Select and publish the NEW u.
        # ----------------------------------------------------
        u, index = self.get_command(elapsed)
        self.publish_u(u)
        self.current_sequence_index = index

        # ----------------------------------------------------
        # 4. Save current state/input for the NEXT RLS update.
        # ----------------------------------------------------
        self.prev_state = state.copy()
        self.prev_u = u

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        row = [
            elapsed,
            index,
            u,
            state[0],
            state[1],
            state[2],
            math.degrees(state[2]),
            state[3],
            self.v_dot,
            self.omega_dot,
            self.rls_update_count,
        ]
        row += self.rls_parameters.tolist()

        self.writer.writerow(row)
        self.log_file.flush()

        if info is None:
            rv = float("nan")
            rw = float("nan")
        else:
            rv = info["v_dot_error"]
            rw = info["omega_dot_error"]

        print(
            f"t={elapsed:6.3f} | "
            f"idx={index:3d} | "
            f"u={u:+.3f} | "
            f"pitch={math.degrees(state[2]):+7.2f} deg | "
            f"v={state[1]:+6.3f} | "
            f"omega={state[3]:+7.3f} | "
            f"vdot={self.v_dot:+7.3f} | "
            f"wdot={self.omega_dot:+8.3f} | "
            f"ev={rv:+8.3f} | "
            f"ew={rw:+8.3f} | "
            f"updates={self.rls_update_count}"
        )

    # ========================================================
    # SAFETY
    # ========================================================

    def _check_pitch_safety(self):
        pitch_deg = math.degrees(self.theta)

        if not self.flip_armed:
            if pitch_deg <= ARM_PITCH_DEG:
                self.flip_armed = True
                self.flip_counter = 0
                print(
                    f"Safety armed at pitch={pitch_deg:.2f} deg"
                )
            return

        if pitch_deg <= STOP_PITCH_DEG:
            self.flip_counter += 1
        else:
            self.flip_counter = 0

        if self.flip_counter >= STOP_CONFIRM_SAMPLES:
            self.finish(
                f"safety pitch reached: {pitch_deg:.2f} deg"
            )

    # ========================================================
    # SAVE / STOP
    # ========================================================

    def save_weights(self):
        np.savez(
            WEIGHTS_PATH,
            learned_parameters=self.rls_parameters,
            covariance=self.rls_covariance,
            update_count=np.array(
                [self.rls_update_count],
                dtype=np.int64,
            ),
        )

        print()
        print("Saved RLS weights:")
        print(self.rls_parameters)
        print(f"RLS updates : {self.rls_update_count}")
        print(f"File        : {WEIGHTS_PATH}")
        print()

    def stop_car(self):
        msg = Float32()
        msg.data = 0.0

        # Publish multiple zero commands to make shutdown more robust.
        for _ in range(3):
            self.drive_pub.publish(msg)

        self.current_u = 0.0

    def finish(self, reason="finished"):
        if self.finished:
            return

        self.finished = True

        print()
        print(f"Stopping identification: {reason}")

        self.timer.cancel()
        self.stop_car()
        self.save_weights()

        if not self.log_file.closed:
            self.log_file.flush()

    def close(self):
        try:
            self.finish("node closed")
        finally:
            if not self.log_file.closed:
                self.log_file.close()


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = RLSIdentificationNode()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.005,
            )

    except KeyboardInterrupt:
        print("\nCtrl+C received.")

    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
