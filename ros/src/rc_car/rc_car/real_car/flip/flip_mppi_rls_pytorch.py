#!/usr/bin/env python3

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32


_ROOT = Path(__file__).resolve().parent
for _sub in ("mppi", "gp", "rls", "nmpc"):
    _path = str(_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


CONTROLLER = "mppi"

if CONTROLLER == "mppi":
    from params_mppi import MPPIConfig as ControllerConfig
    from params_mppi import RLSConfig, WheelieParams
    from mppi import MPPITorch as Controller
    from rls import nominal_rls_parameters, rls_update
elif CONTROLLER == "nmpc":
    from params_nmpc import MPCConfig as ControllerConfig
    from params_nmpc import WheelieParams
    from nmpc_dynamics import NMPC as Controller
else:
    raise ValueError(f"Unknown controller: {CONTROLLER}")


RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "flip_ros.csv"
MODEL_PATH = RESULTS_DIR / "flip_rls_model.npz"

ODOM_TOPIC = "/optitrack/odom"
DRIVE_TOPIC = "/drive"

CTRL_DT = float(ControllerConfig.dt)
PRINT_EVERY_N_CONTROLS = 1
LIVE_PLOT = True

ACCEL_CAP_V = 15.0
ACCEL_CAP_W = 200.0

# Keep False until the signs, units, topic, and actuator interface are verified.
ENABLE_DRIVE = True
ENABLE_RLS = True



class EpisodeLogger:
    def __init__(self):
        self.history: list[dict] = []

    def record(self, row: dict):
        self.history.append(row)

    def save_csv(self):
        if not self.history:
            return

        with CSV_PATH.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.history[0].keys()))
            writer.writeheader()
            writer.writerows(self.history)


class FlipNode(Node):
    def __init__(self):
        super().__init__("flip_mppi_rls_controller")

        self.p = WheelieParams()
        self.cfg = ControllerConfig()
        self.rls_cfg = RLSConfig()

        self.ctrl_min = float(self.p.tau_min)
        self.ctrl_max = float(self.p.tau_max)

        self.rls_parameters = nominal_rls_parameters(self.p)
        self.rls_covariance = (
            self.rls_cfg.initial_covariance
            * np.eye(self.rls_parameters.size, dtype=float)
        )
        self.rls_info = self._empty_rls_info()
        self.rls_update_count = 0

        if CONTROLLER == "mppi":
            self.controller = Controller(
                p=self.p,
                cfg=self.cfg,
                rls_parameters=self.rls_parameters,
                integrator="rk4",
                live_plot=LIVE_PLOT,
            )
        else:
            self.controller = Controller(
                self.p,
                self.cfg,
                live_plot=LIVE_PLOT,
            )

        # Ensure the MPPI warm start exists even if its constructor does not set it.
        self.controller.last_solution = None

        # Reference = [x, v, theta, omega]. Position is sacrificed.
        self.ref = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

        self.logger = EpisodeLogger()
        self.start_time = time.monotonic()
        self._finished = False
        self._waiting_message_printed = False

        self.tau_prev = 0.0
        self.tau_cmd = 0.0
        self.ctrl_cmd = 0.0
        self.solve_success = False
        self.cost = math.nan
        self.control_count = 0

        # Previous state/action used by RLS at control frequency.
        self._rls_prev_state = None
        self._prev_applied_tau = None
        self._previous_control_odom_time = None

        # Previous OptiTrack sample used by numerical differentiation.
        self._previous_vx = None
        self._previous_rate = None
        self._previous_odom_time = None

        # Continuous pitch reconstruction.
        self._previous_wrapped_pitch = None
        self._unwrapped_pitch = None

        self._last_vdot = 0.0
        self._last_wdot = 0.0

        self.optitrack_x = 0.0
        self.optitrack_vx = 0.0
        self.optitrack_ax = 0.0
        self.optitrack_pitch = 0.0
        self.optitrack_pitch_rate = 0.0
        self.optitrack_pitch_acceleration = 0.0
        self.optitrack_time = None
        self.optitrack_received = False

        self.odom_sub = self.create_subscription(Odometry, ODOM_TOPIC, self.odom_callback, 10,)
        self.cmd_pub = self.create_publisher(Float32, DRIVE_TOPIC, 10)
        self.control_timer = self.create_timer(CTRL_DT, self.control_update)

        self.get_logger().info(
            f"Controller={CONTROLLER}, odom={ODOM_TOPIC}, drive={DRIVE_TOPIC}, "
            f"period={CTRL_DT:.3f}s, drive_enabled={ENABLE_DRIVE}"
        )

    @staticmethod
    def _empty_rls_info():
        return {
            "v_dot_measured": 0.0,
            "omega_dot_measured": 0.0,
            "v_dot_hat": 0.0,
            "omega_dot_hat": 0.0,
            "v_dot_error": 0.0,
            "omega_dot_error": 0.0,
            "skipped": True,
        }

    @property
    def elapsed_time(self):
        return time.monotonic() - self.start_time

    def odom_callback(self, msg: Odometry):
        x_position = float(msg.pose.pose.position.x)
        vx = float(msg.twist.twist.linear.x)

        qx = float(msg.pose.pose.orientation.x)
        qy = float(msg.pose.pose.orientation.y)
        qz = float(msg.pose.pose.orientation.z)
        qw = float(msg.pose.pose.orientation.w)

        # Assumes optitrack/odom publishes body-frame angular velocity.
        # Negate this value if the Motive rigid-body y-axis has the opposite sign.
        rate = float(msg.twist.twist.angular.y)

        current_time = (
            float(msg.header.stamp.sec)
            + 1e-9 * float(msg.header.stamp.nanosec)
        )
        if current_time <= 0.0:
            current_time = self.get_clock().now().nanoseconds * 1e-9

        pitch_wrapped = math.atan2(
            2.0 * (qx * qz + qw * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )

        # ============================================================
        # Unwrapped pitch
        # ============================================================
        if self._previous_wrapped_pitch is None:
            self._unwrapped_pitch = pitch_wrapped
        else:
            delta_pitch = math.atan2(
                math.sin(pitch_wrapped - self._previous_wrapped_pitch),
                math.cos(pitch_wrapped - self._previous_wrapped_pitch),
            )
            self._unwrapped_pitch += delta_pitch

        self._previous_wrapped_pitch = pitch_wrapped
        pitch = float(self._unwrapped_pitch)

        if (
            self._previous_odom_time is None
            or self._previous_vx is None
            or self._previous_rate is None
        ):
            v_dot = 0.0
            omega_dot = 0.0
        else:
            dt = current_time - self._previous_odom_time

            if dt > 1e-6:
                v_dot = (vx - self._previous_vx) / dt
                omega_dot = (rate - self._previous_rate) / dt
            else:
                v_dot = self._last_vdot
                omega_dot = self._last_wdot

        if abs(v_dot) <= ACCEL_CAP_V and abs(omega_dot) <= ACCEL_CAP_W:
            self._last_vdot = float(v_dot)
            self._last_wdot = float(omega_dot)
        else:
            v_dot = self._last_vdot
            omega_dot = self._last_wdot

        self._previous_vx = vx
        self._previous_rate = rate
        self._previous_odom_time = current_time

        self.optitrack_x = x_position
        self.optitrack_vx = vx
        self.optitrack_ax = float(v_dot)
        self.optitrack_pitch = pitch
        self.optitrack_pitch_rate = rate
        self.optitrack_pitch_acceleration = float(omega_dot)
        self.optitrack_time = current_time
        self.optitrack_received = True

    def publish_command(self, tau: float) -> float:
        applied_tau = float(np.clip(tau, self.ctrl_min, self.ctrl_max))
        if not ENABLE_DRIVE:
            applied_tau = 0.0

        msg = Float32()
        msg.data = applied_tau
        self.cmd_pub.publish(msg)
        return applied_tau


    def control_update(self):
        if not self.optitrack_received or self.optitrack_time is None:
            self.publish_command(0.0)
            if not self._waiting_message_printed:
                self.get_logger().warning(f"Waiting for {ODOM_TOPIC}")
                self._waiting_message_printed = True
            return

        # Do not solve repeatedly using the same OptiTrack measurement.
        if (
            self._previous_control_odom_time is not None
            and self.optitrack_time <= self._previous_control_odom_time
        ):
            return

        state_now = np.array(
            [
                self.optitrack_x,
                self.optitrack_vx,
                self.optitrack_pitch,
                self.optitrack_pitch_rate,
            ],
            dtype=float,
        )

        if (
            ENABLE_RLS
            and self._rls_prev_state is not None
            and self._prev_applied_tau is not None
            and self._previous_control_odom_time is not None
        ):
            dt_rls = self.optitrack_time - self._previous_control_odom_time

            if dt_rls > 1e-6:
                dv = (state_now[1] - self._rls_prev_state[1]) / dt_rls
                dw = (state_now[3] - self._rls_prev_state[3]) / dt_rls

                if abs(dv) <= ACCEL_CAP_V and abs(dw) <= ACCEL_CAP_W:
                    (
                        self.rls_parameters,
                        self.rls_covariance,
                        self.rls_info,
                    ) = rls_update(
                        self._rls_prev_state,
                        self._prev_applied_tau,
                        state_now,
                        dt_rls,
                        self.rls_parameters,
                        self.rls_covariance,
                        forgetting_factor=self.rls_cfg.forgetting_factor,
                        sigma_v_dot=self.rls_cfg.sigma_v_dot,
                        sigma_omega_dot=self.rls_cfg.sigma_omega_dot,
                    )
                    self.controller.set_rls_parameters(self.rls_parameters)

                    if not self.rls_info["skipped"]:
                        self.rls_update_count += 1
                else:
                    self.rls_info = self._empty_rls_info()

        tau, info = self.controller.solve(state_now, self.ref, self.tau_prev)
        tau = float(np.clip(tau, self.ctrl_min, self.ctrl_max))
        applied_tau = self.publish_command(0.5)

        self._rls_prev_state = state_now.copy()
        self._previous_control_odom_time = self.optitrack_time
        self._prev_applied_tau = applied_tau

        self.tau_prev = applied_tau
        self.tau_cmd = tau
        self.ctrl_cmd = applied_tau
        self.solve_success = bool(info["success"])
        self.cost = float(info["cost"])

        self.logger.record(self.log_row())

        if self.control_count % PRINT_EVERY_N_CONTROLS == 0:
            self.get_logger().info(
                f"t={self.elapsed_time:7.3f} | "
                f"x={state_now[0]:7.3f} | v={state_now[1]:7.3f} | "
                f"pitch={math.degrees(state_now[2]):8.2f} deg | "
                f"omega={state_now[3]:8.3f} | "
                f"tau={tau:7.3f} | applied={applied_tau:7.3f} | "
                f"cost={self.cost:.3f}"
            )

        self.control_count += 1

    def log_row(self):
        return {
            "time": self.elapsed_time,
            "odom_time": self.optitrack_time,
            "x": self.optitrack_x,
            "x_dot": self.optitrack_vx,
            "x_ddot": self.optitrack_ax,
            "pitch_rad": self.optitrack_pitch,
            "pitch_deg": math.degrees(self.optitrack_pitch),
            "pitch_dot": self.optitrack_pitch_rate,
            "pitch_ddot": self.optitrack_pitch_acceleration,
            "tau_requested": self.tau_cmd,
            "tau_applied": self.ctrl_cmd,
            "solve_success": int(self.solve_success),
            "rls_skipped": int(self.rls_info["skipped"]),
            "rls_update_count": self.rls_update_count,
            "rls_v_dot_measured": self.rls_info["v_dot_measured"],
            "rls_v_dot_hat": self.rls_info["v_dot_hat"],
            "rls_omega_dot_measured": self.rls_info["omega_dot_measured"],
            "rls_omega_dot_hat": self.rls_info["omega_dot_hat"],
            **{
                f"rls_a{i}": float(value)
                for i, value in enumerate(self.rls_parameters)
            },
        }

    def finish(self):
        if self._finished:
            return

        self._finished = True
        self.logger.save_csv()
        np.savez(
            MODEL_PATH,
            parameters=self.rls_parameters,
            covariance=self.rls_covariance,
        )

        self.get_logger().info(f"Saved log: {CSV_PATH}")
        self.get_logger().info(f"Saved RLS model: {MODEL_PATH}")

    def destroy_node(self):
        self.publish_command(0.0)
        self.finish()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FlipNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()