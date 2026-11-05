#!/usr/bin/env python3

import csv
import math
import os
import time
from datetime import datetime

import numpy as np

import rclpy
from geometry_msgs.msg import AccelStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32
from params_mppi import MPPIConfig, WheelieParams
from mppi import MPPITorch


# ============================================================
# SETTINGS
# ============================================================

LIVE_PLOT = False

ENABLE_DRIVE = False

REFERENCE_PITCH_DEG = -60.0

ODOM_TOPIC = "/optitrack/odom_810"
RPY_TOPIC = "/optitrack/rpy_810"
TWIST_TOPIC = "/optitrack/twist_810"
ACCEL_TOPIC = "/optitrack/accel_810"
DRIVE_TOPIC = "/drive"

MAX_MEASUREMENT_AGE_S = 0.15


# ============================================================
# STARTUP / ROCK-BACK
# ============================================================

ENABLE_REVERSE_START = True

REVERSE_DRIVE = 0.0
REVERSE_STEPS = 1


# ============================================================
# MPPI CONTROLLER
# ============================================================

class MPPIControllerNode(Node):

    def __init__(self):
        super().__init__("mppi_controller")
        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.ctrl_dt = self.cfg.dt

        # ----------------------------------------------------
        # MPPI
        # ----------------------------------------------------

        self.controller = MPPITorch(p=self.p, cfg=self.cfg, integrator="rk4", live_plot=LIVE_PLOT,)

        # Reference:
        # [x, v, theta, omega]
        self.ref = np.array(
            [
                0.0,
                0.0,
                np.deg2rad(REFERENCE_PITCH_DEG),
                0.0,
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # Measurements
        # ----------------------------------------------------

        self.x = 0.0
        self.v = 0.0
        self.pitch = 0.0
        self.omega = 0.0

        # Acceleration is diagnostic only.
        self.v_dot = 0.0
        self.omega_dot = 0.0

        self.odom_received = False
        self.rpy_received = False
        self.twist_received = False
        self.accel_received = False

        self.odom_time = None
        self.rpy_time = None
        self.twist_time = None
        self.accel_time = None

        # Previous physical torque command used by MPPI.
        self.tau_prev = 0.0

        # Last normalized command actually published.
        self.drive_prev = 0.0

        # Startup reverse maneuver.
        self.reverse_counter = 0
        self.reverse_finished = not ENABLE_REVERSE_START

        # ----------------------------------------------------
        # DATA LOGGING
        # ----------------------------------------------------

        results_dir = os.path.join(
            os.path.dirname(__file__),
            "results",
        )

        os.makedirs(
            results_dir,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.csv_path = os.path.join(
            results_dir,
            f"wheelie_{timestamp}.csv",
        )

        self.csv_file = open(
            self.csv_path,
            "w",
            newline="",
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow(
            [
                "time",
                "phase",

                "x",
                "v",
                "pitch_rad",
                "pitch_deg",
                "omega",

                "v_dot",
                "omega_dot",

                "pitch_ref_rad",
                "pitch_ref_deg",

                "tau",
                "drive",

                "pred_x_N",
                "pred_v_N",
                "pred_pitch_N_rad",
                "pred_pitch_N_deg",
                "pred_omega_N",

                "tau_plan_mean",
                "tau_plan_min",
                "tau_plan_max",
                "tau_plan_terminal",

                "cost",
            ]
        )

        self.start_time = time.monotonic()

        self.get_logger().info(
            f"Logging data to: {self.csv_path}"
        )

        # ----------------------------------------------------
        # ROS QoS
        # ----------------------------------------------------

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ----------------------------------------------------
        # ROS subscribers
        # ----------------------------------------------------

        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, qos,)
        self.rpy_sub = self.create_subscription(Vector3, RPY_TOPIC, self.rpy_callback, qos,)
        self.twist_sub = self.create_subscription(TwistStamped, TWIST_TOPIC, self.twist_callback, qos,)
        self.accel_sub = self.create_subscription(AccelStamped, ACCEL_TOPIC, self.accel_callback, qos,)
        self.drive_pub = self.create_publisher(Float32, DRIVE_TOPIC, 10,)

        # ----------------------------------------------------
        # Control timer
        # ----------------------------------------------------

        self.timer = self.create_timer(
            self.ctrl_dt,
            self.control_callback,
        )

        self.get_logger().info(
            f"MPPI running at "
            f"{1.0 / self.ctrl_dt:.1f} Hz"
        )

        self.get_logger().info(
            f"Reference pitch: "
            f"{REFERENCE_PITCH_DEG:.1f} deg"
        )

        self.get_logger().info(
            f"Using {ODOM_TOPIC}, "
            f"{RPY_TOPIC}, "
            f"{TWIST_TOPIC}, "
            f"{ACCEL_TOPIC}"
        )

        self.get_logger().info(
            f"Physical torque range: "
            f"[{self.p.tau_min:.2f}, "
            f"{self.p.tau_max:.2f}] N m"
        )

        self.get_logger().info(
            f"/drive range: "
            f"[{self.p.drive_min:.2f}, "
            f"{self.p.drive_max:.2f}]"
        )

    # ========================================================
    # OPTITRACK ODOMETRY
    # ========================================================

    def odom_callback(self, msg,):

        self.x = float(msg.pose.pose.position.x)
        self.odom_received = True
        self.odom_time = time.monotonic()

    # ========================================================
    # OPTITRACK RPY
    #
    # RPY is in degrees.
    # Negative pitch = nose up.
    # ========================================================

    def rpy_callback(
        self,
        msg,):

        self.pitch = math.radians(
            float(msg.y)
        )

        self.rpy_received = True
        self.rpy_time = time.monotonic()

    # ========================================================
    # OPTITRACK TWIST
    # ========================================================

    def twist_callback(
        self,
        msg,
    ):

        self.v = float(
            msg.twist.linear.x
        )

        self.omega = float(
            msg.twist.angular.y
        )

        self.twist_received = True
        self.twist_time = time.monotonic()

    # ========================================================
    # OPTITRACK ACCELERATION
    # ========================================================

    def accel_callback(
        self,
        msg,
    ):

        self.v_dot = float(
            msg.accel.linear.x
        )

        self.omega_dot = float(
            msg.accel.angular.y
        )

        self.accel_received = True
        self.accel_time = time.monotonic()

    # ========================================================
    # MEASUREMENT CHECK
    # ========================================================

    def measurements_ready(
        self,
    ):

        # Acceleration is intentionally NOT required.
        if not (
            self.odom_received
            and self.rpy_received
            and self.twist_received
        ):
            return False

        now = time.monotonic()

        return (
            self.odom_time is not None
            and self.rpy_time is not None
            and self.twist_time is not None
            and (
                now - self.odom_time
                <= MAX_MEASUREMENT_AGE_S
            )
            and (
                now - self.rpy_time
                <= MAX_MEASUREMENT_AGE_S
            )
            and (
                now - self.twist_time
                <= MAX_MEASUREMENT_AGE_S
            )
        )

    # ========================================================
    # DRIVE
    # ========================================================

    def publish_drive( self, drive, ):

        drive = float( np.clip( drive, self.p.drive_min, self.p.drive_max, ) )

        if not ENABLE_DRIVE:
            drive = 0.0

        msg = Float32()
        msg.data = drive
        self.drive_pub.publish( msg )
        self.drive_prev = drive

        return drive

    # ========================================================
    # CSV LOGGER
    # ========================================================

    def log_row(
        self,
        phase,
        state,
        tau,
        drive,
        prediction=None,
        u_plan=None,
        cost=np.nan,
    ):

        experiment_time = (
            time.monotonic()
            - self.start_time
        )

        if prediction is None:

            pred_x_N = np.nan
            pred_v_N = np.nan
            pred_pitch_N_rad = np.nan
            pred_pitch_N_deg = np.nan
            pred_omega_N = np.nan

        else:

            pred_x_N = float(
                prediction[-1, 0]
            )

            pred_v_N = float(
                prediction[-1, 1]
            )

            pred_pitch_N_rad = float(
                prediction[-1, 2]
            )

            pred_pitch_N_deg = math.degrees(
                pred_pitch_N_rad
            )

            pred_omega_N = float(
                prediction[-1, 3]
            )

        if u_plan is None:

            tau_plan_mean = np.nan
            tau_plan_min = np.nan
            tau_plan_max = np.nan
            tau_plan_terminal = np.nan

        else:

            u_plan = np.asarray(
                u_plan,
                dtype=float,
            )

            tau_plan_mean = float(
                np.mean(u_plan)
            )

            tau_plan_min = float(
                np.min(u_plan)
            )

            tau_plan_max = float(
                np.max(u_plan)
            )

            tau_plan_terminal = float(
                u_plan[-1]
            )

        self.csv_writer.writerow(
            [
                experiment_time,
                phase,

                state[0],
                state[1],
                state[2],
                math.degrees(state[2]),
                state[3],

                self.v_dot,
                self.omega_dot,

                self.ref[2],
                REFERENCE_PITCH_DEG,

                tau,
                drive,

                pred_x_N,
                pred_v_N,
                pred_pitch_N_rad,
                pred_pitch_N_deg,
                pred_omega_N,

                tau_plan_mean,
                tau_plan_min,
                tau_plan_max,
                tau_plan_terminal,

                cost,
            ]
        )

        # Flush every sample so the experiment is preserved
        # even if the program is interrupted.
        self.csv_file.flush()

    # ========================================================
    # CONTROL
    # ========================================================

    def control_callback(
        self,
    ):

        if not self.measurements_ready():

            self.publish_drive(
                0.0
            )

            self.tau_prev = 0.0

            return

        # ----------------------------------------------------
        # Current measured state
        # ----------------------------------------------------

        state = np.array(
            [
                self.x,
                self.v,
                self.pitch,
                self.omega,
            ],
            dtype=float,
        )

        if not np.isfinite(
            state
        ).all():

            self.publish_drive(
                0.0
            )

            self.tau_prev = 0.0

            return

        # ====================================================
        # STARTUP REVERSE MANEUVER
        # ====================================================

        if not self.reverse_finished:

            applied_drive = self.publish_drive(
                REVERSE_DRIVE
            )

            self.reverse_counter += 1

            # Save startup/reverse samples as well.
            self.log_row(
                phase="reverse",
                state=state,
                tau=np.nan,
                drive=applied_drive,
                prediction=None,
                u_plan=None,
                cost=np.nan,
            )

            print(
                f"[REVERSE] "
                f"step={self.reverse_counter:2d}/{REVERSE_STEPS} | "
                f"x={state[0]:7.3f} | "
                f"v={state[1]:+7.3f} | "
                f"pitch={math.degrees(state[2]):+8.2f} deg | "
                f"drive={applied_drive:+6.3f}"
            )

            if (
                self.reverse_counter
                >= REVERSE_STEPS
            ):

                self.reverse_finished = True

                self.tau_prev = 0.0
                self.controller.last_solution = None

                print(
                    "========== REVERSE FINISHED -> MPPI START =========="
                )

            return

        # ----------------------------------------------------
        # MPPI
        # ----------------------------------------------------

        tau, info = self.controller.solve(
            state,
            self.ref,
            self.tau_prev,
        )

        tau = float(
            np.clip(
                tau,
                self.p.tau_min,
                self.p.tau_max,
            )
        )

        # ----------------------------------------------------
        # Convert physical tau -> normalized real /drive.
        # ----------------------------------------------------

        drive = self.p.tau_to_drive(
            tau
        )

        applied_drive = self.publish_drive(
            drive
        )

        if ENABLE_DRIVE:

            self.tau_prev = (
                self.p.drive_to_tau(
                    applied_drive
                )
            )

        else:

            self.tau_prev = 0.0

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        prediction = info[
            "predicted_states"
        ]

        u_plan = info[
            "u_plan"
        ]

        theta_N_deg = math.degrees(
            prediction[-1, 2]
        )

        # ----------------------------------------------------
        # Save experiment data
        # ----------------------------------------------------

        self.log_row(
            phase="mppi",
            state=state,
            tau=tau,
            drive=applied_drive,
            prediction=prediction,
            u_plan=u_plan,
            cost=info["cost"],
        )

        # ----------------------------------------------------
        # Terminal print
        # ----------------------------------------------------

        print(
            f"x={state[0]:7.3f} | "
            f"v={state[1]:+7.3f} | "
            f"pitch={math.degrees(state[2]):+8.2f} deg | "
            f"omega={state[3]:+8.3f} | "
            f"tau={tau:+7.3f} Nm | "
            f"drive={applied_drive:+6.3f} | "
            f"pred_N={theta_N_deg:+8.2f} deg | "
            f"tau_mean={np.mean(u_plan):+7.3f} | "
            f"tau_N={u_plan[-1]:+7.3f} | "
            f"ax={self.v_dot:+7.3f} | "
            f"alpha={self.omega_dot:+8.3f} | "
            f"cost={info['cost']:.3f}"
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop(
        self,
    ):

        msg = Float32()
        msg.data = 0.0

        self.drive_pub.publish(
            msg
        )

        self.tau_prev = 0.0
        self.drive_prev = 0.0

        if (
            hasattr(self, "csv_file")
            and not self.csv_file.closed
        ):

            self.csv_file.flush()
            self.csv_file.close()

            self.get_logger().info(
                f"Saved experiment: "
                f"{self.csv_path}"
            )


# ============================================================
# MAIN
# ============================================================

def main(
    args=None,
):

    rclpy.init(
        args=args
    )

    node = MPPIControllerNode()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    finally:

        node.stop()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
