#!/usr/bin/env python3

"""
validate_rls.py

Frozen-model validation for the RC wheelie RLS model.

This node:
  1. Loads rls_weights.npz.
  2. NEVER updates the RLS parameters.
  3. Replays the same normalized excitation sequence used for identification.
  4. Subscribes to OptiTrack odometry, twist, and acceleration.
  5. Evaluates:
       - v_dot measured vs predicted
       - omega_dot measured vs predicted
       - Pearson correlation
       - RMSE
       - MAE
       - bias
       - R^2
       - 20-step / 0.6 s open-loop rollout accuracy
  6. Saves:
       - validation CSV
       - metrics text file
       - acceleration time-series plot
       - measured-vs-predicted scatter plot
       - 20-step horizon pitch plot

IMPORTANT:
    The RLS weights are frozen throughout this script.

State convention:
    state = [x, v, theta, omega]
    theta: radians internally
    omega: rad/s
    negative theta = nose up

Input convention:
    u in [-1, 1]
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, AccelStamped
from std_msgs.msg import Float32

from params_mppi import WheelieParams, MPPIConfig


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

OBJECT_ID = 810

# Use the same integration/sample time as MPPI.
CFG = MPPIConfig()
VALIDATION_DT = float(CFG.dt)       # normally 0.03 s
HORIZON_N = int(CFG.N)              # normally 20
HORIZON_TIME = HORIZON_N * VALIDATION_DT

INITIAL_REST_S = 1.0
FINAL_REST_S = 1.0

# Each entry of the original excitation was held for 0.04 s.
COMMAND_HOLD_S = 0.04

# Original command convention: 0.12 = 100 %
MAX_DRIVE_RAW = 0.485

# Safety
ARM_PITCH_DEG = -15.0
STOP_PITCH_DEG = -80.0
STOP_CONFIRM_SAMPLES = 2

# Reject obviously corrupted acceleration measurements from metrics.
ACCEL_CAP_V = 30.0
ACCEL_CAP_W = 300.0

WEIGHTS_PATH = ROOT / "rls_weights.npz"


# ============================================================
# EXCITATION SEQUENCE
# Same raw sequence used in the identification experiment.
# Converted internally to u in [-1, 1].
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
    0.020,
    0.030,
    0.040,
    0.050,
    0.060,
    0.110,
    0.110,
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

DRIVE_SEQUENCE = np.asarray(
    [
        np.clip(raw / MAX_DRIVE_RAW, -1.0, 1.0)
        for raw in RAW_DRIVE_SEQUENCE
    ],
    dtype=float,
)

SEQUENCE_DURATION_S = len(DRIVE_SEQUENCE) * COMMAND_HOLD_S
EXPERIMENT_DURATION_S = (
    INITIAL_REST_S
    + SEQUENCE_DURATION_S
    + FINAL_REST_S
)


# ============================================================
# MODEL
# Must match rls.py
# ============================================================
def rls_derivative_numpy(state, u, a):

    x, v, theta, omega = state

    x_dot = v

    v_dot = (
        a[0] * u
        + a[1] * v
        + a[2] * abs(v) * v
        + a[3] * u * math.cos(theta)
    )

    theta_dot = omega

    omega_dot = (
        a[4] * math.cos(theta)
        + a[5] * u
        + a[6] * omega
        #+ a[7] * v
    )

    # Same ground-contact constraint as MPPI/RLS
    if theta >= 0.0 and omega_dot > 0.0:
        omega_dot = 0.0

    return np.array(
        [
            x_dot,
            v_dot,
            theta_dot,
            omega_dot,
        ],
        dtype=float,
    )



def rk4_step_numpy(state, u, dt, a):
    """Match the RK4 propagation used by MPPI."""
    k1 = rls_derivative_numpy(state, u, a)
    k2 = rls_derivative_numpy(state + 0.5 * dt * k1, u, a)
    k3 = rls_derivative_numpy(state + 0.5 * dt * k2, u, a)
    k4 = rls_derivative_numpy(state + dt * k3, u, a)

    next_state = state + (
        dt / 6.0
    ) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    # Same projection used by MPPI.
    if next_state[2] >= 0.0:
        next_state[2] = 0.0

        if next_state[3] > 0.0:
            next_state[3] = 0.0

    next_state[1] = np.clip(next_state[1], -10.0, 10.0)
    next_state[3] = np.clip(next_state[3], -30.0, 30.0)

    return next_state


def acceleration_prediction(state, u, a):
    derivative = rls_derivative_numpy(state, u, a)
    return derivative[1], derivative[3]


# ============================================================
# METRICS
# ============================================================

def safe_corr(y, y_hat):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    if len(y) < 2:
        return float("nan")

    if np.std(y) < 1e-12 or np.std(y_hat) < 1e-12:
        return float("nan")

    return float(np.corrcoef(y, y_hat)[0, 1])


def regression_metrics(y, y_hat):
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    mask = np.isfinite(y) & np.isfinite(y_hat)
    y = y[mask]
    y_hat = y_hat[mask]

    if len(y) == 0:
        return {
            "n": 0,
            "corr": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
            "r2": float("nan"),
        }

    error = y_hat - y

    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    corr = safe_corr(y, y_hat)

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    if ss_tot > 1e-12:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    return {
        "n": int(len(y)),
        "corr": corr,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": float(r2),
    }


# ============================================================
# NODE
# ============================================================

class RLSValidationNode(Node):

    def __init__(self):
        super().__init__("rls_validation")

        self.p = WheelieParams()

        # ----------------------------------------------------
        # Load FROZEN weights
        # ----------------------------------------------------
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"RLS weights not found: {WEIGHTS_PATH}\n"
                "Run identify_rls.py first."
            )

        with np.load(WEIGHTS_PATH) as data:
            self.a = data["learned_parameters"].astype(float).copy()

        if self.a.shape != (7,):
            raise ValueError(
                f"Expected 7 RLS parameters, got shape {self.a.shape}"
            )

        # ----------------------------------------------------
        # Latest measurements
        # ----------------------------------------------------
        self.x = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.omega = 0.0
        self.v_dot_measured = 0.0
        self.omega_dot_measured = 0.0

        self.odom_received = False
        self.twist_received = False
        self.accel_received = False

        self.current_u = 0.0
        self.current_sequence_index = -1

        self.started_at = None
        self.finished = False

        self.flip_armed = False
        self.flip_counter = 0

        # Sampled validation dataset at MPPI dt.
        self.samples = []

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
            VALIDATION_DT,
            self.timer_callback,
        )

        # ----------------------------------------------------
        # Outputs
        # ----------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = ROOT / "validation_results" / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=False)

        print()
        print("==============================================")
        print(" FROZEN RLS VALIDATION")
        print("==============================================")
        print(f"Weights        : {WEIGHTS_PATH}")
        print(f"Output         : {self.output_dir}")
        print(f"Validation dt  : {VALIDATION_DT:.3f} s")
        print(f"Horizon        : {HORIZON_N} steps")
        print(f"Horizon time   : {HORIZON_TIME:.3f} s")
        print(f"Sequence time  : {SEQUENCE_DURATION_S:.3f} s")
        print()
        print("Frozen parameters:")
        for i, value in enumerate(self.a):
            print(f"a{i} = {value:+.8f}")

        if abs(self.a[5]) > 1e-9:
            u_lift = -self.a[4] / self.a[5]
        else:
            u_lift = float("nan")

        print()

    # ========================================================
    # SENSOR CALLBACKS
    # ========================================================

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation

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
        self.v_dot_measured = float(msg.accel.linear.x)
        self.omega_dot_measured = float(msg.accel.angular.y)
        self.accel_received = True

    # ========================================================
    # HELPERS
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

    def get_command(self, elapsed):
        if elapsed < INITIAL_REST_S:
            return 0.0, -1

        sequence_time = elapsed - INITIAL_REST_S

        if sequence_time >= SEQUENCE_DURATION_S:
            return 0.0, -1

        index = int(sequence_time / COMMAND_HOLD_S)
        index = min(index, len(DRIVE_SEQUENCE) - 1)

        return float(DRIVE_SEQUENCE[index]), index

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
            self.finish("validation complete")
            return

        if not self.measurements_ready():
            self.publish_u(0.0)
            return

        # ========================================================
        # CURRENT STATE
        # ========================================================

        state = self.current_state()

        # ========================================================
        # COMMAND THAT PRODUCED CURRENT ACCELERATION
        # ========================================================

        u_applied = self.current_u
        index_applied = self.current_sequence_index

        v_dot_pred, omega_dot_pred = acceleration_prediction(
            state,
            u_applied,
            self.a,
        )

        valid_accel = (
            np.isfinite(self.v_dot_measured)
            and np.isfinite(self.omega_dot_measured)
            and abs(self.v_dot_measured) <= ACCEL_CAP_V
            and abs(self.omega_dot_measured) <= ACCEL_CAP_W
        )

        # ========================================================
        # NEXT COMMAND
        # This command drives state[k] -> state[k+1]
        # ========================================================

        u_next, index_next = self.get_command(elapsed)

        # ========================================================
        # SAVE SAMPLE
        # ========================================================

        self.samples.append(
            {
                "elapsed": elapsed,

                # For one-step acceleration validation
                "u_applied": u_applied,
                "index_applied": index_applied,

                # For forward rollout / horizon validation
                "u": u_next,
                "index": index_next,
                "state": state.copy(),
                "v_dot_meas": self.v_dot_measured,
                "omega_dot_meas": self.omega_dot_measured,
                "v_dot_pred": v_dot_pred,
                "omega_dot_pred": omega_dot_pred,
                "valid_accel": valid_accel,
            }
        )

        # ========================================================
        # APPLY NEW COMMAND
        # ========================================================

        self.publish_u(u_next)
        self.current_sequence_index = index_next

        print(
            f"t={elapsed:6.3f} | "
            f"u_applied={u_applied:+.3f} | "
            f"u_next={u_next:+.3f} | "
            f"pitch={math.degrees(state[2]):+7.2f} deg | "
            f"v_dot m/p="
            f"{self.v_dot_measured:+7.3f}/"
            f"{v_dot_pred:+7.3f} | "
            f"w_dot m/p="
            f"{self.omega_dot_measured:+8.3f}/"
            f"{omega_dot_pred:+8.3f}"
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
    # ANALYSIS
    # ========================================================

    def analyze(self):
        if len(self.samples) < 2:
            print("Not enough samples for validation.")
            return

        # ----------------------------------------------------
        # Save sampled data
        # ----------------------------------------------------
        csv_path = self.output_dir / "validation_samples.csv"

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            writer.writerow(
                [
                    "elapsed_s",

                    "index_applied",
                    "u_applied",

                    "index_next",
                    "u_next",

                    "x",
                    "v",
                    "theta_rad",
                    "theta_deg",
                    "omega",

                    "v_dot_measured",
                    "v_dot_predicted",

                    "omega_dot_measured",
                    "omega_dot_predicted",

                    "valid_accel",
                ]
            )

            for sample in self.samples:
                state = sample["state"]

                writer.writerow(
                    [
                        sample["elapsed"],

                        sample["index_applied"],
                        sample["u_applied"],

                        sample["index"],
                        sample["u"],

                        state[0],
                        state[1],
                        state[2],
                        math.degrees(state[2]),
                        state[3],

                        sample["v_dot_meas"],
                        sample["v_dot_pred"],

                        sample["omega_dot_meas"],
                        sample["omega_dot_pred"],

                        int(sample["valid_accel"]),
                    ]
                )

        # ----------------------------------------------------
        # One-step acceleration metrics
        # ----------------------------------------------------
        valid = [
            sample
            for sample in self.samples
            if sample["valid_accel"]
        ]

        v_meas = np.array(
            [s["v_dot_meas"] for s in valid],
            dtype=float,
        )
        v_pred = np.array(
            [s["v_dot_pred"] for s in valid],
            dtype=float,
        )

        w_meas = np.array(
            [s["omega_dot_meas"] for s in valid],
            dtype=float,
        )
        w_pred = np.array(
            [s["omega_dot_pred"] for s in valid],
            dtype=float,
        )

        v_metrics = regression_metrics(v_meas, v_pred)
        w_metrics = regression_metrics(w_meas, w_pred)

        # ----------------------------------------------------
        # Multi-step rollout validation
        # ----------------------------------------------------
        actual_states = np.stack(
            [s["state"] for s in self.samples],
            axis=0,
        )

        controls = np.array(
            [s["u"] for s in self.samples],
            dtype=float,
        )

        endpoint_theta_pred = []
        endpoint_theta_real = []

        endpoint_v_pred = []
        endpoint_v_real = []

        endpoint_omega_pred = []
        endpoint_omega_real = []

        # Store several representative horizon traces for plotting.
        horizon_examples = []

        max_start = len(self.samples) - HORIZON_N - 1

        for start in range(max(0, max_start)):
            pred = actual_states[start].copy()

            theta_pred_trace = [
                math.degrees(pred[2])
            ]
            theta_real_trace = [
                math.degrees(actual_states[start, 2])
            ]

            for h in range(HORIZON_N):
                u = controls[start + h]

                pred = rk4_step_numpy(
                    pred,
                    u,
                    VALIDATION_DT,
                    self.a,
                )

                real = actual_states[start + h + 1]

                theta_pred_trace.append(
                    math.degrees(pred[2])
                )
                theta_real_trace.append(
                    math.degrees(real[2])
                )

            real_end = actual_states[start + HORIZON_N]

            endpoint_theta_pred.append(
                math.degrees(pred[2])
            )
            endpoint_theta_real.append(
                math.degrees(real_end[2])
            )

            endpoint_v_pred.append(pred[1])
            endpoint_v_real.append(real_end[1])

            endpoint_omega_pred.append(pred[3])
            endpoint_omega_real.append(real_end[3])

            # Save at most 6 distributed examples.
            if len(horizon_examples) < 6:
                stride = max(
                    1,
                    max_start // 6,
                )

                if start % stride == 0:
                    horizon_examples.append(
                        (
                            start,
                            np.array(theta_real_trace),
                            np.array(theta_pred_trace),
                        )
                    )

        theta_h_metrics = regression_metrics(
            endpoint_theta_real,
            endpoint_theta_pred,
        )

        v_h_metrics = regression_metrics(
            endpoint_v_real,
            endpoint_v_pred,
        )

        omega_h_metrics = regression_metrics(
            endpoint_omega_real,
            endpoint_omega_pred,
        )

        # ----------------------------------------------------
        # Learned lift threshold
        # ----------------------------------------------------
        if abs(self.a[5]) > 1e-9:
            u_lift = -self.a[4] / self.a[5]
        else:
            u_lift = float("nan")

        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------
        print()
        print("==============================================")
        print(" VALIDATION RESULTS")
        print("==============================================")

        self._print_metric_block(
            "v_dot one-step",
            v_metrics,
            "m/s^2",
        )

        self._print_metric_block(
            "omega_dot one-step",
            w_metrics,
            "rad/s^2",
        )

        print(
            f"\n{HORIZON_N}-step / "
            f"{HORIZON_TIME:.2f} s horizon endpoint:"
        )

        self._print_metric_block(
            "theta endpoint",
            theta_h_metrics,
            "deg",
        )

        self._print_metric_block(
            "v endpoint",
            v_h_metrics,
            "m/s",
        )

        self._print_metric_block(
            "omega endpoint",
            omega_h_metrics,
            "rad/s",
        )

        print(
            "\nApprox. learned lift threshold: "
            f"u = {u_lift:+.4f}"
        )

        print()

        # ----------------------------------------------------
        # Save metrics text
        # ----------------------------------------------------
        metrics_path = self.output_dir / "metrics.txt"

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            handle.write(
                "Frozen RLS validation\n"
            )
            handle.write(
                "=====================\n\n"
            )

            handle.write(
                f"dt = {VALIDATION_DT:.6f} s\n"
            )
            handle.write(
                f"N = {HORIZON_N}\n"
            )
            handle.write(
                f"horizon = {HORIZON_TIME:.6f} s\n"
            )
            handle.write(
                f"u_lift_approx = {u_lift:.8f}\n\n"
            )

            for name, metrics, units in [
                ("v_dot", v_metrics, "m/s^2"),
                (
                    "omega_dot",
                    w_metrics,
                    "rad/s^2",
                ),
                (
                    "theta_horizon_endpoint",
                    theta_h_metrics,
                    "deg",
                ),
                (
                    "v_horizon_endpoint",
                    v_h_metrics,
                    "m/s",
                ),
                (
                    "omega_horizon_endpoint",
                    omega_h_metrics,
                    "rad/s",
                ),
            ]:
                handle.write(f"[{name}]\n")
                handle.write(
                    f"n = {metrics['n']}\n"
                )
                handle.write(
                    f"corr = {metrics['corr']:.8f}\n"
                )
                handle.write(
                    f"rmse = {metrics['rmse']:.8f} {units}\n"
                )
                handle.write(
                    f"mae = {metrics['mae']:.8f} {units}\n"
                )
                handle.write(
                    f"bias = {metrics['bias']:.8f} {units}\n"
                )
                handle.write(
                    f"r2 = {metrics['r2']:.8f}\n\n"
                )

        # ----------------------------------------------------
        # Plots
        # ----------------------------------------------------
        self._plot_acceleration_timeseries(valid)
        self._plot_acceleration_scatter(
            v_meas,
            v_pred,
            w_meas,
            w_pred,
        )
        self._plot_horizon_examples(horizon_examples)

        print(f"Validation CSV : {csv_path}")
        print(f"Metrics        : {metrics_path}")
        print(
            "Plots          : "
            f"{self.output_dir}"
        )
        print()

    @staticmethod
    def _print_metric_block(name, metrics, units):
        print()
        print(name)
        print("-" * len(name))
        print(f"N     : {metrics['n']}")
        print(f"corr  : {metrics['corr']:.6f}")
        print(f"RMSE  : {metrics['rmse']:.6f} {units}")
        print(f"MAE   : {metrics['mae']:.6f} {units}")
        print(f"bias  : {metrics['bias']:.6f} {units}")
        print(f"R^2   : {metrics['r2']:.6f}")

    def _plot_acceleration_timeseries(self, valid):
        if not valid:
            return

        t = np.array(
            [s["elapsed"] for s in valid],
            dtype=float,
        )

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
        )

        axs[0].plot(
            t,
            [s["v_dot_meas"] for s in valid],
            label="measured",
            linewidth=1.5,
        )
        axs[0].plot(
            t,
            [s["v_dot_pred"] for s in valid],
            label="predicted",
            linewidth=1.5,
        )
        axs[0].set_ylabel("v_dot [m/s^2]")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        axs[1].plot(
            t,
            [s["omega_dot_meas"] for s in valid],
            label="measured",
            linewidth=1.5,
        )
        axs[1].plot(
            t,
            [s["omega_dot_pred"] for s in valid],
            label="predicted",
            linewidth=1.5,
        )
        axs[1].set_xlabel("time [s]")
        axs[1].set_ylabel("omega_dot [rad/s^2]")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        fig.tight_layout()
        fig.savefig(
            self.output_dir
            / "acceleration_timeseries.png",
            dpi=200,
        )
        plt.close(fig)

    def _plot_acceleration_scatter(
        self,
        v_meas,
        v_pred,
        w_meas,
        w_pred,
    ):
        if len(v_meas) == 0:
            return

        fig, axs = plt.subplots(
            1,
            2,
            figsize=(12, 5),
        )

        axs[0].scatter(
            v_meas,
            v_pred,
            s=16,
            alpha=0.7,
        )

        lo = min(
            np.min(v_meas),
            np.min(v_pred),
        )
        hi = max(
            np.max(v_meas),
            np.max(v_pred),
        )

        axs[0].plot(
            [lo, hi],
            [lo, hi],
            "--",
            linewidth=1.2,
        )
        axs[0].set_xlabel("measured v_dot [m/s^2]")
        axs[0].set_ylabel("predicted v_dot [m/s^2]")
        axs[0].grid(True, alpha=0.3)

        axs[1].scatter(
            w_meas,
            w_pred,
            s=16,
            alpha=0.7,
        )

        lo = min(
            np.min(w_meas),
            np.min(w_pred),
        )
        hi = max(
            np.max(w_meas),
            np.max(w_pred),
        )

        axs[1].plot(
            [lo, hi],
            [lo, hi],
            "--",
            linewidth=1.2,
        )
        axs[1].set_xlabel(
            "measured omega_dot [rad/s^2]"
        )
        axs[1].set_ylabel(
            "predicted omega_dot [rad/s^2]"
        )
        axs[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            self.output_dir
            / "acceleration_scatter.png",
            dpi=200,
        )
        plt.close(fig)

    def _plot_horizon_examples(
        self,
        horizon_examples,
    ):
        if not horizon_examples:
            return

        h_time = (
            np.arange(HORIZON_N + 1)
            * VALIDATION_DT
        )

        fig, ax = plt.subplots(
            figsize=(11, 6),
        )

        for i, (
            start,
            theta_real,
            theta_pred,
        ) in enumerate(horizon_examples):

            ax.plot(
                h_time,
                theta_real,
                linewidth=1.6,
                label=(
                    "real"
                    if i == 0
                    else None
                ),
            )

            ax.plot(
                h_time,
                theta_pred,
                "--",
                linewidth=1.6,
                label=(
                    "RLS prediction"
                    if i == 0
                    else None
                ),
            )

        ax.set_xlabel("prediction time [s]")
        ax.set_ylabel("pitch [deg]")
        ax.set_title(
            f"{HORIZON_N}-step "
            f"({HORIZON_TIME:.2f} s) "
            "open-loop pitch validation"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            self.output_dir
            / "pitch_horizon_validation.png",
            dpi=200,
        )
        plt.close(fig)

    # ========================================================
    # STOP / FINISH
    # ========================================================

    def stop_car(self):
        msg = Float32()
        msg.data = 0.0

        for _ in range(3):
            self.drive_pub.publish(msg)

        self.current_u = 0.0

    def finish(self, reason="finished"):
        if self.finished:
            return

        self.finished = True

        print()
        print(f"Stopping validation: {reason}")

        self.timer.cancel()
        self.stop_car()

        # IMPORTANT: no RLS update, no weight saving.
        self.analyze()

    def close(self):
        self.finish("node closed")


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)

    node = RLSValidationNode()

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
