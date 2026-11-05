#!/usr/bin/env python3

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import AccelStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

from params_mppi import MPPIConfig, WheelieParams
from mppi_rls import MPPIRLSTorch
from rls import nominal_rls_parameters, rls_update


# ============================================================
# SETTINGS
# ============================================================

ENABLE_DRIVE = True
REFERENCE_PITCH_DEG = -75.0

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
# RLS SETTINGS
# ============================================================

ENABLE_RLS = True
RESET_RLS = False
ZERO_INITIALIZATION = False

RLS_FORGETTING_FACTOR = 0.9995
RLS_INITIAL_COVARIANCE = 3.0
RLS_SIGMA_V_DOT = 2.0
RLS_SIGMA_OMEGA_DOT = 5.0

RLS_V_DOT_CAP = 30.0
RLS_OMEGA_DOT_CAP = 250.0
RLS_DT_MIN = 0.005
RLS_DT_MAX = 0.15
RLS_SAVE_EVERY_UPDATES = 25


class MPPIRLSControllerNode(Node):
    def __init__(self):
        super().__init__("mppi_rls_controller")

        self.root = Path(__file__).resolve().parent
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.ctrl_dt = float(self.cfg.dt)

        self.rls_weights_path = self.root / "rls_weights_lmt.npz"
        self.rls_parameters, self.rls_covariance, self.previous_trial_count, self.total_rls_updates = self.load_rls_state()
        self.trial_id = self.previous_trial_count + 1

        self.controller = MPPIRLSTorch(
            p=self.p,
            cfg=self.cfg,
            rls_parameters=self.rls_parameters,
            integrator="rk4",
        )

        self.ref = np.array([0.0, 0.0, math.radians(REFERENCE_PITCH_DEG), 0.0], dtype=float)

        self.x = 0.0
        self.v = 0.0
        self.pitch = 0.0
        self.omega = 0.0
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

        self.prev_state = None
        self.prev_tau_applied = None
        self.prev_transition_time = None

        self.tau_prev = 0.0
        self.drive_prev = 0.0

        self.rls_update_count_trial = 0
        self.rls_v_error_sq = []
        self.rls_omega_error_sq = []
        self.latest_rls_info = self.empty_rls_info()

        self.reverse_counter = 0
        self.reverse_finished = not ENABLE_REVERSE_START

        self.setup_logging()

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, qos)
        self.rpy_sub = self.create_subscription(Vector3, RPY_TOPIC, self.rpy_callback, qos)
        self.twist_sub = self.create_subscription(TwistStamped, TWIST_TOPIC, self.twist_callback, qos)
        self.accel_sub = self.create_subscription(AccelStamped, ACCEL_TOPIC, self.accel_callback, qos)

        self.drive_pub = self.create_publisher(Float32, DRIVE_TOPIC, 10)
        self.timer = self.create_timer(self.ctrl_dt, self.control_callback)

        self.get_logger().info(f"MPPI + persistent RLS running at {1.0 / self.ctrl_dt:.1f} Hz")
        self.get_logger().info(f"RLS trial: {self.trial_id}")
        self.get_logger().info(f"RLS model: {self.rls_weights_path}")
        self.get_logger().info(f"Previous completed trials: {self.previous_trial_count}")
        self.get_logger().info(f"Loaded total RLS updates: {self.total_rls_updates}")
        self.get_logger().info(f"Reference pitch: {REFERENCE_PITCH_DEG:.1f} deg")

    # ========================================================
    # RLS PERSISTENCE
    # ========================================================

    def initial_rls_parameters(self):
        nominal = nominal_rls_parameters(self.p)
        return np.zeros_like(nominal) if ZERO_INITIALIZATION else nominal

    def load_rls_state(self):
        a0 = self.initial_rls_parameters()
        P0 = RLS_INITIAL_COVARIANCE * np.eye(a0.size)

        if RESET_RLS:
            self.get_logger().warning("RESET_RLS=True: starting from a fresh RLS model.")
            return a0, P0, 0, 0

        if not self.rls_weights_path.exists():
            self.get_logger().info("No previous RLS model found. Starting a new learning sequence.")
            return a0, P0, 0, 0

        try:
            with np.load(self.rls_weights_path) as data:
                a = np.asarray(data["learned_parameters"], dtype=float)
                P = np.asarray(data["covariance"], dtype=float)
                trial_count = int(data["trial_count"]) if "trial_count" in data.files else 0
                total_updates = int(data["total_updates"]) if "total_updates" in data.files else 0

            if a.shape != a0.shape or P.shape != P0.shape:
                raise ValueError("Saved RLS dimensions do not match.")
            if not np.isfinite(a).all() or not np.isfinite(P).all():
                raise ValueError("Saved RLS model contains nonfinite values.")

            self.get_logger().info("Loaded RLS model from previous trial.")
            return a.copy(), P.copy(), trial_count, total_updates

        except Exception as exc:
            self.get_logger().error(f"Could not load RLS model: {exc}. Starting fresh.")
            return a0, P0, 0, 0

    def save_rls_state(self, final=False):
        trial_count = self.trial_id if final else self.previous_trial_count

        np.savez(
            self.rls_weights_path,
            learned_parameters=self.rls_parameters,
            covariance=self.rls_covariance,
            trial_count=np.array(trial_count, dtype=int),
            total_updates=np.array(self.total_rls_updates, dtype=int),
        )

    # ========================================================
    # RLS PREDICTION / UPDATE
    # ========================================================

    @staticmethod
    def predict_rls_acceleration_np(state, tau, a):
        _, v, theta, omega = state

        phi_v = np.array([tau, v, abs(v) * v, tau * np.cos(theta), 1.0], dtype=float)
        phi_w = np.array([np.cos(theta), tau, omega, v, 1.0], dtype=float)

        v_dot_hat = float(phi_v @ a[0:5])
        omega_dot_hat = float(phi_w @ a[5:10])

        if theta >= 0.0 and omega_dot_hat > 0.0:
            omega_dot_hat = 0.0

        return v_dot_hat, omega_dot_hat

    @staticmethod
    def empty_rls_info():
        return {
            "skipped": True,
            "dt": np.nan,
            "v_dot_measured": np.nan,
            "omega_dot_measured": np.nan,
            "v_dot_hat_before": np.nan,
            "omega_dot_hat_before": np.nan,
            "v_dot_error_before": np.nan,
            "omega_dot_error_before": np.nan,
            "v_dot_hat_after": np.nan,
            "omega_dot_hat_after": np.nan,
        }

    def update_rls(self, state, now):
        info = self.empty_rls_info()

        if not ENABLE_RLS or not ENABLE_DRIVE:
            return info

        if self.prev_state is None or self.prev_tau_applied is None or self.prev_transition_time is None:
            return info

        dt = float(now - self.prev_transition_time)
        if dt < RLS_DT_MIN or dt > RLS_DT_MAX:
            return info

        v_dot_measured = (state[1] - self.prev_state[1]) / dt
        omega_dot_measured = (state[3] - self.prev_state[3]) / dt

        if not np.isfinite([v_dot_measured, omega_dot_measured]).all():
            return info
        if abs(v_dot_measured) > RLS_V_DOT_CAP or abs(omega_dot_measured) > RLS_OMEGA_DOT_CAP:
            return info

        v_hat_before, omega_hat_before = self.predict_rls_acceleration_np(
            self.prev_state,
            self.prev_tau_applied,
            self.rls_parameters,
        )

        v_error_before = v_dot_measured - v_hat_before
        omega_error_before = omega_dot_measured - omega_hat_before

        a_new, P_new, update_info = rls_update(
            self.prev_state,
            self.prev_tau_applied,
            state,
            dt,
            self.rls_parameters,
            self.rls_covariance,
            forgetting_factor=RLS_FORGETTING_FACTOR,
            sigma_v_dot=RLS_SIGMA_V_DOT,
            sigma_omega_dot=RLS_SIGMA_OMEGA_DOT,
        )

        if update_info["skipped"] or not np.isfinite(a_new).all() or not np.isfinite(P_new).all():
            return info

        self.rls_parameters = a_new
        self.rls_covariance = P_new
        self.controller.set_rls_parameters(self.rls_parameters)

        self.rls_update_count_trial += 1
        self.total_rls_updates += 1
        self.rls_v_error_sq.append(v_error_before**2)
        self.rls_omega_error_sq.append(omega_error_before**2)

        info = {
            "skipped": False,
            "dt": dt,
            "v_dot_measured": v_dot_measured,
            "omega_dot_measured": omega_dot_measured,
            "v_dot_hat_before": v_hat_before,
            "omega_dot_hat_before": omega_hat_before,
            "v_dot_error_before": v_error_before,
            "omega_dot_error_before": omega_error_before,
            "v_dot_hat_after": update_info["v_dot_hat"],
            "omega_dot_hat_after": update_info["omega_dot_hat"],
        }

        if self.rls_update_count_trial % RLS_SAVE_EVERY_UPDATES == 0:
            self.save_rls_state(final=False)

        return info

    # ========================================================
    # LOGGING
    # ========================================================

    def setup_logging(self):
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.results_dir / f"wheelie_rls_trial_{self.trial_id:03d}_{timestamp}.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        header = [
            "time", "trial", "phase",
            "x", "v", "pitch_rad", "pitch_deg", "omega",
            "optitrack_v_dot", "optitrack_omega_dot",
            "pitch_ref_rad", "pitch_ref_deg",
            "tau", "drive",
            "pred_x_N", "pred_v_N", "pred_pitch_N_rad", "pred_pitch_N_deg", "pred_omega_N",
            "tau_plan_mean", "tau_plan_min", "tau_plan_max", "tau_plan_terminal",
            "cost", "cost_mean", "ess",
            "rls_skipped", "rls_dt",
            "rls_v_dot_measured", "rls_v_dot_hat_before", "rls_v_dot_error_before", "rls_v_dot_hat_after",
            "rls_omega_dot_measured", "rls_omega_dot_hat_before", "rls_omega_dot_error_before", "rls_omega_dot_hat_after",
            "rls_trial_update_count", "rls_total_update_count",
        ]
        header += [f"a_{i}" for i in range(self.rls_parameters.size)]

        self.csv_writer.writerow(header)
        self.start_time = time.monotonic()
        self.metrics_path = self.results_dir / "rls_episode_metrics.csv"

    def log_row(self, phase, state, tau, drive, prediction=None, u_plan=None, mppi_info=None, rls_info=None):
        rls_info = self.empty_rls_info() if rls_info is None else rls_info
        pred = [np.nan, np.nan, np.nan, np.nan] if prediction is None else prediction[-1]

        if u_plan is None:
            plan_mean = plan_min = plan_max = plan_terminal = np.nan
        else:
            u_plan = np.asarray(u_plan, dtype=float)
            plan_mean = float(np.mean(u_plan))
            plan_min = float(np.min(u_plan))
            plan_max = float(np.max(u_plan))
            plan_terminal = float(u_plan[-1])

        if mppi_info is None:
            cost = cost_mean = ess = np.nan
        else:
            cost = mppi_info["cost"]
            cost_mean = mppi_info["cost_mean"]
            ess = mppi_info["ess"]

        pred_pitch_deg = math.degrees(pred[2]) if np.isfinite(pred[2]) else np.nan

        row = [
            time.monotonic() - self.start_time,
            self.trial_id,
            phase,
            state[0], state[1], state[2], math.degrees(state[2]), state[3],
            self.v_dot, self.omega_dot,
            self.ref[2], REFERENCE_PITCH_DEG,
            tau, drive,
            pred[0], pred[1], pred[2], pred_pitch_deg, pred[3],
            plan_mean, plan_min, plan_max, plan_terminal,
            cost, cost_mean, ess,
            int(rls_info["skipped"]), rls_info["dt"],
            rls_info["v_dot_measured"], rls_info["v_dot_hat_before"], rls_info["v_dot_error_before"], rls_info["v_dot_hat_after"],
            rls_info["omega_dot_measured"], rls_info["omega_dot_hat_before"], rls_info["omega_dot_error_before"], rls_info["omega_dot_hat_after"],
            self.rls_update_count_trial, self.total_rls_updates,
        ]
        row += self.rls_parameters.tolist()

        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def append_trial_metrics(self):
        rmse_v = math.sqrt(float(np.mean(self.rls_v_error_sq))) if self.rls_v_error_sq else np.nan
        rmse_omega = math.sqrt(float(np.mean(self.rls_omega_error_sq))) if self.rls_omega_error_sq else np.nan
        exists = self.metrics_path.exists()

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow([
                    "trial",
                    "rls_updates",
                    "total_rls_updates",
                    "rmse_v_dot_before_update",
                    "rmse_omega_dot_before_update",
                ])
            writer.writerow([self.trial_id, self.rls_update_count_trial, self.total_rls_updates, rmse_v, rmse_omega])

        return rmse_v, rmse_omega

    # ========================================================
    # OPTITRACK
    # ========================================================

    def odom_callback(self, msg):
        self.x = float(msg.pose.pose.position.x)
        self.odom_received = True
        self.odom_time = time.monotonic()

    def rpy_callback(self, msg):
        self.pitch = math.radians(float(msg.y))
        self.rpy_received = True
        self.rpy_time = time.monotonic()

    def twist_callback(self, msg):
        self.v = float(msg.twist.linear.x)
        self.omega = float(msg.twist.angular.y)
        self.twist_received = True
        self.twist_time = time.monotonic()

    def accel_callback(self, msg):
        self.v_dot = float(msg.accel.linear.x)
        self.omega_dot = float(msg.accel.angular.y)
        self.accel_received = True
        self.accel_time = time.monotonic()

    def measurements_ready(self):
        if not (self.odom_received and self.rpy_received and self.twist_received):
            return False

        now = time.monotonic()

        return (
            self.odom_time is not None
            and self.rpy_time is not None
            and self.twist_time is not None
            and now - self.odom_time <= MAX_MEASUREMENT_AGE_S
            and now - self.rpy_time <= MAX_MEASUREMENT_AGE_S
            and now - self.twist_time <= MAX_MEASUREMENT_AGE_S
        )

    # ========================================================
    # DRIVE / TRANSITIONS
    # ========================================================

    def publish_drive(self, drive):
        drive = float(np.clip(drive, self.p.drive_min, self.p.drive_max))
        if not ENABLE_DRIVE:
            drive = 0.0

        msg = Float32()
        msg.data = drive
        self.drive_pub.publish(msg)
        self.drive_prev = drive
        return drive

    def save_current_transition_start(self, state, applied_drive, now):
        self.prev_state = state.copy()
        self.prev_tau_applied = self.p.drive_to_tau(applied_drive) if ENABLE_DRIVE else 0.0
        self.prev_transition_time = now

    def clear_transition(self):
        self.prev_state = None
        self.prev_tau_applied = None
        self.prev_transition_time = None

    # ========================================================
    # CONTROL
    # ========================================================

    def control_callback(self):
        if not self.measurements_ready():
            self.publish_drive(0.0)
            self.tau_prev = 0.0
            self.clear_transition()
            return

        now = time.monotonic()
        state = np.array([self.x, self.v, self.pitch, self.omega], dtype=float)

        if not np.isfinite(state).all():
            self.publish_drive(0.0)
            self.tau_prev = 0.0
            self.clear_transition()
            return

        self.latest_rls_info = self.update_rls(state, now)

        if not self.reverse_finished:
            applied_drive = self.publish_drive(REVERSE_DRIVE)
            applied_tau = self.p.drive_to_tau(applied_drive) if ENABLE_DRIVE else 0.0

            self.reverse_counter += 1
            self.log_row("reverse", state, applied_tau, applied_drive, rls_info=self.latest_rls_info)
            self.save_current_transition_start(state, applied_drive, now)

            if self.reverse_counter >= REVERSE_STEPS:
                self.reverse_finished = True
                self.tau_prev = applied_tau
                self.controller.last_solution = None
                print("========== REVERSE FINISHED -> MPPI + RLS START ==========")

            return

        tau, info = self.controller.solve(state, self.ref, self.tau_prev)
        tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

        drive = self.p.tau_to_drive(tau)
        applied_drive = self.publish_drive(drive)
        applied_tau = self.p.drive_to_tau(applied_drive) if ENABLE_DRIVE else 0.0
        self.tau_prev = applied_tau

        prediction = info["predicted_states"]
        u_plan = info["u_plan"]
        theta_N_deg = math.degrees(prediction[-1, 2])

        self.log_row(
            "mppi_rls",
            state,
            tau,
            applied_drive,
            prediction=prediction,
            u_plan=u_plan,
            mppi_info=info,
            rls_info=self.latest_rls_info,
        )

        self.save_current_transition_start(state, applied_drive, now)

        print(
            f"trial={self.trial_id:03d} | "
            f"x={state[0]:7.3f} | "
            f"v={state[1]:+7.3f} | "
            f"pitch={math.degrees(state[2]):+8.2f} deg | "
            f"omega={state[3]:+8.3f} | "
            f"tau={tau:+7.3f} Nm | "
            f"drive={applied_drive:+6.3f} | "
            f"pred_N={theta_N_deg:+8.2f} deg | "
            f"RLS_ev={self.latest_rls_info['v_dot_error_before']:+8.3f} | "
            f"RLS_ew={self.latest_rls_info['omega_dot_error_before']:+8.3f} | "
            f"updates={self.rls_update_count_trial:4d} | "
            f"ESS={info['ess']:7.1f} | "
            f"cost={info['cost']:.3f}"
        )

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):
        msg = Float32()
        msg.data = 0.0
        self.drive_pub.publish(msg)

        self.save_rls_state(final=True)
        rmse_v, rmse_omega = self.append_trial_metrics()

        if hasattr(self, "csv_file") and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

        self.get_logger().info(f"Trial {self.trial_id} saved.")
        self.get_logger().info(f"RLS updates this trial: {self.rls_update_count_trial}")
        self.get_logger().info(f"Pre-update RMSE: v_dot={rmse_v:.6f}, omega_dot={rmse_omega:.6f}")
        self.get_logger().info(f"RLS model saved to: {self.rls_weights_path}")
        self.get_logger().info(f"Trial log saved to: {self.csv_path}")
        self.get_logger().info(f"Episode metrics: {self.metrics_path}")


def main(args=None):
    rclpy.init(args=args)
    node = MPPIRLSControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
