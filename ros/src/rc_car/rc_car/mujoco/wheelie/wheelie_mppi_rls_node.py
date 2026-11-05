#!/usr/bin/env python3

import sys
import math
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


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

LIVE_PLOT = False

ACCEL_CAP_V = 15.0
ACCEL_CAP_W = 200.0

REFERENCE_PITCH_DEG = -70.0


# ============================================================
# MPPI CONTROLLER
# ============================================================

class MPPIControllerNode(Node):

    def __init__(self):
        super().__init__("mppi_controller")

        # Parameters
        self.p = WheelieParams()
        self.cfg = MPPIConfig()
        self.rls_cfg = RLSConfig()

        self.ctrl_dt = self.cfg.dt

        # RLS
        self.rls_parameters = nominal_rls_parameters(self.p)
        self.rls_covariance = self.rls_cfg.initial_covariance * np.eye(self.rls_parameters.size)
        self.rls_update_count = 0

        # MPPI
        self.controller = MPPITorch(
            p=self.p,
            cfg=self.cfg,
            rls_parameters=self.rls_parameters,
            integrator="rk4",
            live_plot=LIVE_PLOT,
        )

        self.controller.plot_wheelie_span = (1.7, 2.3)

        # Reference = [x, v, theta, omega]
        self.ref = np.array([0.0, 0.0, np.deg2rad(REFERENCE_PITCH_DEG), 0.0], dtype=float)

        # Measurements
        self.x = 0.0
        self.v = 0.0
        self.pitch = 0.0
        self.omega = 0.0

        self.odom_received = False
        self.imu_received = False

        # Previous transition
        self.prev_state = None
        self.prev_tau = None
        self.tau_prev = 0.0

        # ROS
        self.odom_sub = self.create_subscription(Odometry, "/mujoco/odom", self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/mujoco/imu", self.imu_callback, 10)
        self.drive_pub = self.create_publisher(Float32, "/drive", 10)

        self.timer = self.create_timer(self.ctrl_dt, self.control_callback)

        self.get_logger().info(f"MPPI running at {1.0 / self.ctrl_dt:.1f} Hz")

    # ========================================================
    # ODOMETRY
    # ========================================================

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.v = msg.twist.twist.linear.x
        self.odom_received = True

    # ========================================================
    # IMU
    # ========================================================

    def imu_callback(self, msg):
        w = msg.orientation.w
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z

        self.pitch = math.atan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (y * y + z * z))
        self.omega = msg.angular_velocity.y

        self.imu_received = True

    # ========================================================
    # RLS
    # ========================================================

    def update_rls(self, state):
        if self.prev_state is None or self.prev_tau is None:
            return

        v_dot = (state[1] - self.prev_state[1]) / self.ctrl_dt
        omega_dot = (state[3] - self.prev_state[3]) / self.ctrl_dt

        if abs(v_dot) > ACCEL_CAP_V or abs(omega_dot) > ACCEL_CAP_W:
            return

        self.rls_parameters, self.rls_covariance, info = rls_update(
            self.prev_state,
            self.prev_tau,
            state,
            self.ctrl_dt,
            self.rls_parameters,
            self.rls_covariance,
            forgetting_factor=self.rls_cfg.forgetting_factor,
            sigma_v_dot=self.rls_cfg.sigma_v_dot,
            sigma_omega_dot=self.rls_cfg.sigma_omega_dot,
        )

        self.controller.set_rls_parameters(self.rls_parameters)

        if not info["skipped"]:
            self.rls_update_count += 1

    # ========================================================
    # CONTROL
    # ========================================================

    def control_callback(self):
        if not self.odom_received or not self.imu_received:
            return

        state = np.array([self.x, self.v, self.pitch, self.omega], dtype=float)

        # Update RLS from the previous transition
        self.update_rls(state)

        # MPPI
        tau, info = self.controller.solve(state, self.ref, self.tau_prev)
        tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

        # Publish torque
        msg = Float32()
        msg.data = tau
        self.drive_pub.publish(msg)

        # Save transition
        self.prev_state = state.copy()
        self.prev_tau = tau
        self.tau_prev = tau

        print(
            f"x={state[0]:7.3f} | "
            f"v={state[1]:7.3f} | "
            f"pitch={math.degrees(state[2]):8.2f} deg | "
            f"omega={state[3]:8.3f} | "
            f"tau={tau:8.3f} | "
            f"cost={info['cost']} | "
            f"solve={bool(info['success'])}"
        )


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
        stop_msg = Float32()
        stop_msg.data = 0.0
        node.drive_pub.publish(stop_msg)

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()