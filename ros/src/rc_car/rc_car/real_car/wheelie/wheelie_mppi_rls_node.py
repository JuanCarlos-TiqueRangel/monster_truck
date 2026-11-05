#!/usr/bin/env python3

import sys
import math
import time
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, AccelStamped
from std_msgs.msg import Float32

import csv

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ============================================================
# PATHS
# ============================================================

_ROOT = Path(__file__).resolve().parent

for _sub in ("mppi", "rls"):
    _path = str(_ROOT / _sub)

    if _path not in sys.path:
        sys.path.insert(0, _path)

from params_mppi import WheelieParams, MPPIConfig, RLSConfig
from mppi import MPPITorch
from rls import nominal_rls_parameters, rls_update


# ============================================================
# SETTINGS
# ============================================================

LIVE_PLOT = True

ACCEL_CAP_V = 15.0
ACCEL_CAP_W = 200.0

REFERENCE_PITCH = np.deg2rad(-45.0)
OBJECT_ID = 810


# ============================================================
# MPPI CONTROLLER NODE WITH LOGGING
# ============================================================

class MPPIControllerNode(Node):

    def __init__(self):
        super().__init__("mppi_controller")

        # Parameters
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.rls_cfg = RLSConfig()

        self.ctrl_dt = self.cfg.dt

        # ========================================================
        # Load frozen RLS model
        # ========================================================
        rls_weights_file = _ROOT / "wheelie_nn" / "rls_learned_weights.npz"

        if not rls_weights_file.is_file():
            raise FileNotFoundError(
                f"RLS weights not found: {rls_weights_file}\n"
                "Run identify_rls.py before running MPPI."
            )

        with np.load(rls_weights_file, allow_pickle=False) as data:
            self.rls_parameters = np.asarray(data["a"], dtype=float).copy()

        if self.rls_parameters.shape != (18,) or not np.isfinite(self.rls_parameters).all():
            raise ValueError("Expected exactly 18 finite RLS parameters.")

        if np.all(self.rls_parameters == 0.0):
            raise ValueError("The loaded RLS model has all-zero parameters.")

        self.rls_covariance = self.rls_cfg.initial_covariance * np.eye(self.rls_parameters.size)
        self.rls_update_count = 0

        print("Loaded initial RLS parameters:")
        print(self.rls_parameters)

        # MPPI Controller
        self.controller = MPPITorch(
            p=self.p,
            cfg=self.cfg,
            rls_parameters=self.rls_parameters,
            integrator="rk4",
            live_plot=LIVE_PLOT,
        )

        self.controller.last_solution = self.controller.rls_parameters.new_zeros(self.cfg.N)

        # Reference = [x, v, theta, omega]
        self.ref = np.array([0.0, 0.0, REFERENCE_PITCH, 0.0], dtype=float)

        # States & Derivatives
        self.x = 0.0
        self.pitch = 0.0
        self.x_dot = 0.0
        self.pitch_dot = 0.0
        self.vx_dot = 0.0
        self.omega_pitch_dot = 0.0

        self.odom_received = False
        self.imu_received = False
        self.velocity_received = False
        self.accel_received = False

        # Transition History
        self.prev_state = None
        self.prev_u = None
        self.u_prev = 0.0

        # ========================================================
        # LOGGING BUFFER
        # ========================================================
        self.log_records = []
        self.start_time = None

        # ROS Setup
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f"/optitrack/odom_{OBJECT_ID}",
            self.odom_callback,
            qos,
        )

        self.vel_sub = self.create_subscription(
            TwistStamped,
            f"/optitrack/twist_{OBJECT_ID}",
            self.velocity_callback,
            qos,
        )

        self.accel_sub = self.create_subscription(
            AccelStamped,
            f"/optitrack/accel_{OBJECT_ID}",
            self.accel_callback,
            qos,
        )

        self.drive_pub = self.create_publisher(Float32, "/drive", 10)

        # with torch.no_grad():
        #     s0 = self.controller.rls_parameters.new_zeros(4)
        #     ref = s0.new_tensor(self.ref)
        #     prev0 = s0.new_zeros(())
        #     for batch in (self.controller.K, 1):
        #         controls = s0.new_zeros((batch, self.cfg.N))
        #         for _ in range(3):
        #             self.controller._rollout_fast(s0, controls, ref, prev0)
        # torch.cuda.synchronize(self.controller.device)

        self.timer = self.create_timer(self.ctrl_dt, self.control_callback)
        self.get_logger().info(f"MPPI initialized at {1.0 / self.ctrl_dt:.1f} Hz")

    # ========================================================
    # CALLBACKS
    # ========================================================

    def odom_callback(self, msg):
        w = msg.pose.pose.orientation.w
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z

        self.pitch = math.atan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (y * y + z * z))
        self.x = msg.pose.pose.position.x
        self.odom_received = True

    def velocity_callback(self, msg):
        self.x_dot = msg.twist.linear.x
        self.pitch_dot = msg.twist.angular.y
        self.velocity_received = True

    def accel_callback(self, msg):
        # Clip raw acceleration topics to shield RLS from outlier spikes
        self.vx_dot = float(np.clip(msg.accel.linear.x, -ACCEL_CAP_V, ACCEL_CAP_V))
        self.omega_pitch_dot = float(np.clip(msg.accel.angular.y, -ACCEL_CAP_W, ACCEL_CAP_W))
        self.accel_received = True

    # ========================================================
    # RLS UPDATE
    # ========================================================

    def update_rls(self, state):
        if self.prev_state is None or self.prev_u is None:
            return {}

        self.rls_parameters, self.rls_covariance, info = rls_update(
            state,
            self.prev_u,
            self.vx_dot,
            self.omega_pitch_dot,
            self.rls_parameters,
            self.rls_covariance,
            forgetting_factor=self.rls_cfg.forgetting_factor,
            sigma_v_dot=self.rls_cfg.sigma_v_dot,
            sigma_omega_dot=self.rls_cfg.sigma_omega_dot,
            update_angular=True,
        )

        self.controller.set_rls_parameters(self.rls_parameters)

        if not info.get("skipped", False):
            self.rls_update_count += 1

        return info

    # ========================================================
    # CONTROL LOOP
    # ========================================================

    def control_callback(self):
        if not (self.odom_received and self.velocity_received and self.accel_received):
            return

        if self.start_time is None:
            self.start_time = time.time()

        t_now = time.time() - self.start_time
        state = np.array([self.x, self.x_dot, self.pitch, self.pitch_dot], dtype=float)

        # 1. Update RLS using state x_{t-1} that produced current accelerations
        rls_info = {}
        # if self.prev_state is not None:
        #     rls_info = self.update_rls(self.prev_state)

        # 2. Solve MPPI for current state x_t
        u, info = self.controller.solve(state, self.ref, self.u_prev)
        u = float(np.clip(u, self.p.u_min, self.p.u_max))

        # 3. Publish Control Input
        msg = Float32()
        msg.data = u

        t_cmd = time.time() - self.start_time

        self.drive_pub.publish(msg)

        # 4. RECORD EXPERIMENTAL DATA RECORD
        record = {
            "time": t_now,
            "time_cmd": t_cmd,
            # Current States
            "x": self.x,
            "v": self.x_dot,
            "pitch": self.pitch,
            "pitch_deg": math.degrees(self.pitch),
            "omega": self.pitch_dot,
            # Reference
            "ref_pitch": self.ref[2],
            "ref_pitch_deg": math.degrees(self.ref[2]),
            # Control Input
            "u": u,
            # Measured Accelerations
            "v_dot_meas": self.vx_dot,
            "omega_dot_meas": self.omega_pitch_dot,
            # RLS Predictions & Residuals
            "v_dot_hat": rls_info.get("v_dot_hat", np.nan),
            "omega_dot_hat": rls_info.get("omega_dot_hat", np.nan),
            "v_dot_error": rls_info.get("v_dot_error", np.nan),
            "omega_dot_error": rls_info.get("omega_dot_error", np.nan),
            "rls_params": self.rls_parameters.copy(),
            # MPPI Controller Metrics
            "mppi_cost": info.get("cost", np.nan),
            "pitch_cost_now": info.get("pitch_cost_now", np.nan),
            "best_theta_N_deg": info.get("best_theta_N_deg", np.nan),
            "best_omega_N": info.get("best_omega_N", np.nan),
            # state predictions
            "pred_xpos": info.get("pred_xpos", np.nan),
            "pred_speed": info.get("pred_speed", np.nan),
            "pred_pitch": info.get("pred_pitch", np.nan),
            "pred_angular_speed": info.get("pred_angular_speed", np.nan),
        }
        self.log_records.append(record)

        # Update transition history
        self.prev_state = state.copy()
        self.prev_u = u
        self.u_prev = u

        print(
            f"t={t_now:5.2f}s | "
            f"pitch={math.degrees(state[2]):+.2f}° | "
            f"ref={math.degrees(self.ref[2]):+.2f}° | "
            f"u={u:+.2f} | "
            f"best_theta_N={info.get('best_theta_N_deg', float('nan')):+.2f}° |"
            f"cost={info.get('cost', float('nan')):+.4f}° |"
        )

    # ========================================================
    # SAVE LOG FILE ON SHUTDOWN
    # ========================================================

    def save_experiment_log(self):
        if not self.log_records:
            self.get_logger().info("No experiment steps recorded to save.")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = _ROOT / f"mppi_rls_log_{timestamp_str}.csv"

        fieldnames = list(self.log_records[0].keys())

        try:
            with open(filename, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.log_records)

            print(f"\n[LOGGER] Successfully exported {len(self.log_records)} steps to CSV:\n {filename}\n")
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV log: {e}")


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = MPPIControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save experiment log before destroying node
        node.save_experiment_log()

        stop_msg = Float32()
        stop_msg.data = 0.0
        node.drive_pub.publish(stop_msg)

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()