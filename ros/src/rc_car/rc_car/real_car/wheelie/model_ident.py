#!/usr/bin/env python3

"""
Wheelie data collection with robust flip detection.

Logs:
    results/optitrack_<timestamp>/
        accel.csv
        odom.csv
        rpy.csv
        twist.csv
        drive.csv

Pitch convention:
    negative pitch = nose up

Flip detection:
    1. Disabled during INITIAL_REST_S
    2. Arms after pitch < ARM_PITCH_DEG
    3. Requires FLIP_CONFIRM_SAMPLES consecutive measurements
       below FLIP_THRESHOLD_DEG
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import AccelStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

OBJECT_ID = 810

DRIVE_PERIOD_S = 0.04

INITIAL_REST_S = 1.0
FINAL_REST_S = 1.0

MAX_DRIVE = 0.12


# ============================================================
# FLIP DETECTION
# ============================================================

# Arm only after the car clearly starts lifting
ARM_PITCH_DEG = -15.0

# Flip threshold
FLIP_THRESHOLD_DEG = -85.0

# Require several consecutive measurements below -90 deg
FLIP_CONFIRM_SAMPLES = 2


# ============================================================
# WHEELIE INPUT SEQUENCE
#
# Each command lasts DRIVE_PERIOD_S = 0.04 s
#
# MAX_DRIVE = 0.12 corresponds to 100%
# ============================================================

DRIVE_SEQUENCE = [
    0.120,   #   0 %
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.120,
    0.110,
    0.110,
    0.110,
    0.110,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.020,   #  40 %
    0.030,   #  50 %
    0.040,   #  60 %
    0.050,   #  70 %
    0.060,   #  80 % peak
    0.110,   #  70 %
    0.110,
    0.100,
    0.100,
    0.100,
    0.090,
    0.090,
    0.060,   #  60 %
    0.060,   #  50 %
    0.060,   #  40 %
    0.060,   #  30 %
    0.060,   #   0 %
    0.070,   #   0 %
    0.048,   #  40 %
    0.060,   #  50 %
    0.072,   #  60 %
    0.084,   #  70 %
    0.096,   #  80 % peak
    0.084,   #  70 %
    0.072,   #  60 %
    0.060,   #  50 %
    0.048,   #  40 %
    0.036,   #  30 %
    0.000,   #   0 %
    0.000,   #   0 %
    0.048,   #  40 %
    0.060,   #  50 %
    0.072,   #  60 %
    0.084,   #  70 %
    0.096,   #  80 % peak
    0.084,   #  70 %
    0.072,   #  60 %
    0.060,   #  50 %
    0.048,   #  40 %
    0.036,   #  30 %
    0.000,   #   0 %
    0.000,   #   0 %
    0.048,   #  40 %
    0.060,   #  50 %
    0.072,   #  60 %
    0.084,   #  70 %
    0.096,   #  80 % peak
    0.084,   #  70 %
    0.072,   #  60 %
    0.060,   #  50 %
    0.048,   #  40 %
    0.036,   #  30 %
    0.000,   #   0 %
    0.000,   #   0 %
    0.048,   #  40 %
    0.060,   #  50 %
    0.072,   #  60 %
    0.084,   #  70 %
    0.096,   #  80 % peak
    0.084,   #  70 %
    0.072,   #  60 %
    0.060,   #  50 %
    0.048,   #  40 %
    0.036,   #  30 %
    0.000,   #   0 %
    0.000,   #   0 %
    0.048,   #  40 %
    0.060,   #  50 %
    0.072,   #  60 %
    0.084,   #  70 %
    0.096,   #  80 % peak
    0.084,   #  70 %
    # 0.072,   #  60 %
    # 0.060,   #  50 %
    # 0.048,   #  40 %
    # 0.036,   #  30 %
    # 0.000,   #   0 %
]


SEQUENCE_DURATION_S = (
    len(DRIVE_SEQUENCE)
    * DRIVE_PERIOD_S
)

EXPERIMENT_DURATION_S = (
    INITIAL_REST_S
    + SEQUENCE_DURATION_S
    + FINAL_REST_S
)


# ============================================================
# HELPERS
# ============================================================

def vector_values(vector):

    return [
        vector.x,
        vector.y,
        vector.z,
    ]


# ============================================================
# NODE
# ============================================================

class OptiTrackDataCollectorNode(Node):

    def __init__(self):

        super().__init__(
            "optitrack_wheelie_data_collector"
        )

        object_id = int(
            self.declare_parameter(
                "object_id",
                OBJECT_ID,
            ).value
        )

        # ====================================================
        # EXPERIMENT STATE
        # ====================================================

        self.experiment_started_at = None
        self.finished = False

        # ====================================================
        # DRIVE STATE
        # ====================================================

        self.current_drive = 0.0
        self.current_sequence_index = -1

        # ====================================================
        # FLIP DETECTOR
        # ====================================================

        self.flip_detector_armed = False
        self.flip_counter = 0

        # ====================================================
        # OUTPUT DIRECTORY
        # ====================================================

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        self.output_dir = (
            ROOT
            / "results"
            / f"optitrack_{timestamp}"
        )

        self.csv_files = {}
        self.csv_writers = {}

        try:

            self._open_logs()

            self._subscribe_to_topics(
                object_id
            )

            self.drive_publisher = (
                self.create_publisher(
                    Float32,
                    "/drive",
                    1,
                )
            )

            self.timer = self.create_timer(
                DRIVE_PERIOD_S,
                self.update_drive,
            )

        except Exception:

            self._close_logs()
            raise

        # ====================================================
        # INFO
        # ====================================================

        print()
        print("==============================================")
        print(" WHEELIE DATA COLLECTION")
        print("==============================================")
        print()

        print(
            f"Initial rest      : "
            f"{INITIAL_REST_S:.2f} s"
        )

        print(
            f"Sequence duration : "
            f"{SEQUENCE_DURATION_S:.2f} s"
        )

        print(
            f"Arm pitch         : "
            f"{ARM_PITCH_DEG:.1f} deg"
        )

        print(
            f"Flip threshold    : "
            f"{FLIP_THRESHOLD_DEG:.1f} deg"
        )

        print(
            f"Confirm samples   : "
            f"{FLIP_CONFIRM_SAMPLES}"
        )

        print()
        print("Drive sequence:")

        for i, command in enumerate(
            DRIVE_SEQUENCE
        ):

            percent = (
                command
                / MAX_DRIVE
                * 100.0
            )

            print(
                f"{i:02d}: "
                f"{command:+.4f} "
                f"({percent:+5.1f}%)"
            )

        print()
        print(
            f"Saving data to: "
            f"{self.output_dir}"
        )
        print()

    # ========================================================
    # CSV FILES
    # ========================================================

    def _open_logs(self):

        self.output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        stamped_columns = [
            "received_stamp_ns",
            "stamp_ns",
            "frame_id",
        ]

        motion_columns = [
            "linear_x",
            "linear_y",
            "linear_z",
            "angular_x",
            "angular_y",
            "angular_z",
        ]

        odom_columns = [
            "child_frame_id",
            "position_x",
            "position_y",
            "position_z",
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ]

        covariance_columns = [
            f"pose_covariance_{i}"
            for i in range(36)
        ]

        covariance_columns += [
            f"twist_covariance_{i}"
            for i in range(36)
        ]

        self._open_csv(
            "accel",
            stamped_columns
            + motion_columns,
        )

        self._open_csv(
            "twist",
            stamped_columns
            + motion_columns,
        )

        self._open_csv(
            "odom",
            stamped_columns
            + odom_columns
            + motion_columns
            + covariance_columns,
        )

        self._open_csv(
            "rpy",
            [
                "received_stamp_ns",
                "roll",
                "pitch",
                "yaw",
                "pitch_deg",
            ],
        )

        self._open_csv(
            "drive",
            [
                "published_stamp_ns",
                "elapsed_s",
                "sequence_index",
                "drive",
                "drive_percent",
            ],
        )

    # ========================================================

    def _open_csv(
        self,
        name,
        columns,
    ):

        handle = (
            self.output_dir
            / f"{name}.csv"
        ).open(
            "w",
            newline="",
            encoding="utf-8",
        )

        self.csv_files[name] = handle

        self.csv_writers[name] = (
            csv.writer(handle)
        )

        self.csv_writers[
            name
        ].writerow(
            columns
        )

    # ========================================================

    def _close_logs(self):

        for handle in (
            self.csv_files.values()
        ):

            if not handle.closed:

                handle.flush()
                handle.close()

    # ========================================================
    # ROS SUBSCRIPTIONS
    # ========================================================

    def _subscribe_to_topics(
        self,
        object_id,
    ):

        qos = QoSProfile(
            depth=20,
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
            ),
        )

        self.subscriptions_ = [

            self.create_subscription(
                AccelStamped,
                f"/optitrack/accel_{object_id}",
                self.on_accel,
                qos,
            ),

            self.create_subscription(
                Odometry,
                f"/optitrack/odom_{object_id}",
                self.on_odom,
                qos,
            ),

            self.create_subscription(
                Vector3,
                f"/optitrack/rpy_{object_id}",
                self.on_rpy,
                qos,
            ),

            self.create_subscription(
                TwistStamped,
                f"/optitrack/twist_{object_id}",
                self.on_twist,
                qos,
            ),
        ]

    # ========================================================
    # ACCELERATION
    # ========================================================

    def on_accel(
        self,
        msg,
    ):

        values = (
            vector_values(
                msg.accel.linear
            )
            +
            vector_values(
                msg.accel.angular
            )
        )

        self._record_stamped_message(
            "accel",
            msg,
            values,
        )

    # ========================================================
    # TWIST
    # ========================================================

    def on_twist(
        self,
        msg,
    ):

        values = (
            vector_values(
                msg.twist.linear
            )
            +
            vector_values(
                msg.twist.angular
            )
        )

        self._record_stamped_message(
            "twist",
            msg,
            values,
        )

    # ========================================================
    # ODOMETRY
    # ========================================================

    def on_odom(
        self,
        msg,
    ):

        pose = msg.pose.pose
        twist = msg.twist.twist

        values = [
            msg.child_frame_id
        ]

        values += vector_values(
            pose.position
        )

        values += [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        values += (
            vector_values(
                twist.linear
            )
            +
            vector_values(
                twist.angular
            )
        )

        values += list(
            msg.pose.covariance
        )

        values += list(
            msg.twist.covariance
        )

        self._record_stamped_message(
            "odom",
            msg,
            values,
        )

    # ========================================================
    # RPY + ROBUST FLIP DETECTION
    # ========================================================

    def on_rpy(self, msg):

        received_stamp_ns = self.get_clock().now().nanoseconds

        roll_deg = float(msg.x)
        pitch_deg = float(msg.y)
        yaw_deg = float(msg.z)

        self.csv_writers["rpy"].writerow([
            received_stamp_ns,
            roll_deg,
            pitch_deg,
            yaw_deg,
        ])

        if self.experiment_started_at is None:
            return

        elapsed = time.monotonic() - self.experiment_started_at

        # Ignore initial rest
        if elapsed < INITIAL_REST_S:
            self.flip_detector_armed = False
            self.flip_counter = 0
            return

        # Arm once vehicle starts lifting
        if not self.flip_detector_armed:

            if pitch_deg < ARM_PITCH_DEG:
                self.flip_detector_armed = True
                self.flip_counter = 0

                print(
                    f"Flip detector armed | "
                    f"pitch={pitch_deg:.2f} deg"
                )

            return

        # Check flip
        if pitch_deg < FLIP_THRESHOLD_DEG:

            self.flip_counter += 1

            print(
                f"Flip check "
                f"{self.flip_counter}/{FLIP_CONFIRM_SAMPLES} | "
                f"pitch={pitch_deg:.2f} deg"
            )

        else:
            self.flip_counter = 0

        if self.flip_counter >= FLIP_CONFIRM_SAMPLES:

            print(
                f"FLIP CONFIRMED | "
                f"pitch={pitch_deg:.2f} deg | "
                f"drive={self.current_drive:.6f}"
            )

            self.finish()

    # ========================================================
    # GENERIC LOGGER
    # ========================================================

    def _record_stamped_message(self, topic, msg, values):

        received_stamp_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        source_stamp_ns = (
            msg.header.stamp.sec
            * 1_000_000_000
            + msg.header.stamp.nanosec
        )

        row = [
            received_stamp_ns,
            source_stamp_ns,
            msg.header.frame_id,
        ] + values

        self.csv_writers[
            topic
        ].writerow(
            row
        )

    # ========================================================
    # DRIVE
    # ========================================================

    def publish_drive(self, command, elapsed, sequence_index=-1,):

        self.current_drive = float(command)
        self.current_sequence_index = int(sequence_index)
        stamp_ns = (self.get_clock().now().nanoseconds)
        msg = Float32()
        msg.data = (self.current_drive)
        self.drive_publisher.publish(msg)
        drive_percent = (self.current_drive / MAX_DRIVE * 100.0)

        self.csv_writers[
            "drive"
        ].writerow([
            stamp_ns,
            elapsed,
            sequence_index,
            self.current_drive,
            drive_percent,
        ])

        # ----------------------------------------------------
        # Print EVERY command sent
        # ----------------------------------------------------

        print(
            f"t={elapsed:6.3f} s | "
            f"drive={self.current_drive:+.6f} | "
            f"{drive_percent:+6.1f}%"
        )

    # ========================================================
    # WHEELIE SEQUENCE
    # ========================================================

    def get_drive_command(self, elapsed,):

        # ----------------------------------------------------
        # Initial rest
        # ----------------------------------------------------

        if (elapsed < INITIAL_REST_S):
            return 0.0, -1

        sequence_time = (elapsed - INITIAL_REST_S)
        # ----------------------------------------------------
        # Sequence finished
        # ----------------------------------------------------

        if (sequence_time >= SEQUENCE_DURATION_S):
            return 0.0, -1

        # ----------------------------------------------------
        # Current sequence index
        # ----------------------------------------------------

        index = int(sequence_time / DRIVE_PERIOD_S)
        index = min(index, len(DRIVE_SEQUENCE) - 1,)

        return (DRIVE_SEQUENCE[index], index,)

    # ========================================================
    # TIMER
    # ========================================================

    def update_drive(
        self,
    ):

        if self.finished:

            return

        now = time.monotonic()

        if (
            self.experiment_started_at
            is None
        ):

            self.experiment_started_at = (
                now
            )

        elapsed = (
            now
            - self.experiment_started_at
        )

        # ----------------------------------------------------
        # Normal end
        # ----------------------------------------------------

        if (
            elapsed
            >= EXPERIMENT_DURATION_S
        ):

            self.finish()

            return

        # ----------------------------------------------------
        # Drive command
        # ----------------------------------------------------

        command, index = (
            self.get_drive_command(
                elapsed
            )
        )

        self.publish_drive(
            command,
            elapsed,
            index,
        )

    # ========================================================
    # FINISH / EMERGENCY STOP
    # ========================================================

    def finish(
        self,
    ):

        if self.finished:

            return

        # ----------------------------------------------------
        # Mark finished first so callbacks cannot trigger twice
        # ----------------------------------------------------

        self.finished = True

        # ----------------------------------------------------
        # Immediately stop the timer
        # ----------------------------------------------------

        self.timer.cancel()

        # ----------------------------------------------------
        # Send zero command
        # ----------------------------------------------------

        stamp_ns = (
            self.get_clock()
            .now()
            .nanoseconds
        )

        stop_msg = Float32()

        stop_msg.data = 0.0

        self.drive_publisher.publish(
            stop_msg
        )

        # ----------------------------------------------------
        # Log final zero
        # ----------------------------------------------------

        if (
            self.experiment_started_at
            is not None
        ):

            elapsed = (
                time.monotonic()
                - self.experiment_started_at
            )

        else:

            elapsed = 0.0

        self.csv_writers[
            "drive"
        ].writerow([
            stamp_ns,
            elapsed,
            -1,
            0.0,
            0.0,
        ])

        print(
            f"t={elapsed:6.3f} s | "
            f"drive={0.0:+.6f} | "
            f"{0.0:+6.1f}%"
        )

        self.current_drive = 0.0
        self.current_sequence_index = -1

        # ----------------------------------------------------
        # Flush logs
        # ----------------------------------------------------

        for handle in (
            self.csv_files.values()
        ):

            handle.flush()

        print()
        print(
            "Drive command set to zero."
        )

        print(
            f"Logs saved in: "
            f"{self.output_dir}"
        )

        print()

    # ========================================================
    # CLOSE
    # ========================================================

    def close(
        self,
    ):

        try:

            self.finish()

        finally:

            self._close_logs()


# ============================================================
# MAIN
# ============================================================

def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = (
            OptiTrackDataCollectorNode()
        )

        while (
            rclpy.ok()
            and not node.finished
        ):

            rclpy.spin_once(
                node,
                timeout_sec=0.005,
            )

    except KeyboardInterrupt:

        pass

    finally:

        if node is not None:

            try:

                node.close()

            finally:

                node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


if __name__ == "__main__":

    main()