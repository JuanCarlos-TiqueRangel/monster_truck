#!/usr/bin/env python3
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import casadi as ca

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# Same project-style import used by the MPPI node.
# This assumes this file lives inside the same package/folder structure as your
# MPPI controller, where ../utils/geometry.py exists.
THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parents[0] if len(THIS_DIR.parents) > 0 else THIS_DIR
sys.path.append(str(PARENT_DIR))
sys.path.append(str(THIS_DIR))
sys.path.append(str(THIS_DIR.parent))

from utils import geometry

# ============================================================
# Parameters / configuration
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1              # kg, total mass of the vehicle
    l: float = 0.18             # m, rear axle to COM distance
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.085            # m, rear wheel radius
    g: float = 9.81             # m/s^2
    c_v: float = 0.0            # simple longitudinal damping

    tau_min: float = -8.0       # N*m
    tau_max: float = 12.0       # N*m

    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(100.0)
    omega_min: float = -8.0
    omega_max: float = 8.0
    v_min: float = -8.0
    v_max: float = 8.0

    pitch_ref_deg: float = -90.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = 0.1
    N: int = 35

    q_x: float = 0.0
    q_v: float = 0.01
    q_theta: float = 1000.0
    q_omega: float = 15.0
    r_tau: float = 0.1
    r_dtau: float = 0.01
    q_terminal_theta: float = 400.0
    q_terminal_omega: float = 100.0

    ipopt_max_iter: int = 50
    ipopt_tol: float = 1e-4


# ============================================================
# NMPC implementation
# ============================================================

class WheelieNMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self.last_solution: Optional[np.ndarray] = None
        self._build_solver()

    def _f_ca(self, x, u):
        p = self.p
        return ca.vertcat(
            x[1],
            u[0] / (p.m * p.r) - p.c_v * x[1],
            x[3],
            (-u[0] + p.m * p.g * p.l * ca.cos(x[2])) / p.I_eff,
        )

    def _rk4_ca(self, x, u):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u)
        k2 = self._f_ca(x + 0.5 * dt * k1, u)
        k3 = self._f_ca(x + 0.5 * dt * k2, u)
        k4 = self._f_ca(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = int(cfg.N)

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # P = [x, v, theta, omega, x_ref, v_ref, theta_ref, omega_ref, tau_prev]
        P = ca.SX.sym("P", 9)
        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]

        obj = 0
        g = []

        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            # Holding torque near the desired pitch.
            tau_eq = p.m * p.g * p.l * ca.cos(ref[2])

            obj += ca.mtimes([e.T, Q, e])
            obj += cfg.r_tau * (uk[0] - tau_eq) ** 2
            obj += cfg.r_dtau * du**2

            x_next = self._rk4_ca(xk, uk)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        nlp = {"f": obj, "x": opt_vars, "g": g, "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(cfg.ipopt_max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "print_time": 0,
            "verbose": False,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        nX = self.nx * (N + 1)
        nU = self.nu * N

        lbx = []
        ubx = []
        for _ in range(N + 1):
            lbx += [-ca.inf, p.v_min, p.theta_min, p.omega_min]
            ubx += [ ca.inf, p.v_max, p.theta_max, p.omega_max]

        for _ in range(N):
            lbx += [p.tau_min]
            ubx += [p.tau_max]

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)
        self.lbg = np.zeros(self.nx * (N + 1), dtype=float)
        self.ubg = np.zeros(self.nx * (N + 1), dtype=float)

        self.nX = nX
        self.nU = nU

    def reset_plan(self):
        self.last_solution = None

    def _initial_guess(self, state: np.ndarray, tau_prev: float) -> np.ndarray:
        N = int(self.cfg.N)

        if self.last_solution is not None:
            sol = self.last_solution.copy()
            X_sol = sol[:self.nX].reshape((self.nx, N + 1), order="F")
            U_sol = sol[self.nX:].reshape((self.nu, N), order="F")

            X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
            X_guess[:, 0] = state
        else:
            X_guess = np.tile(state.reshape(-1, 1), (1, N + 1))
            U_guess = np.full((self.nu, N), tau_prev, dtype=float)

        return np.concatenate([
            X_guess.reshape(-1, order="F"),
            U_guess.reshape(-1, order="F"),
        ])

    def solve(self, state: np.ndarray, ref: np.ndarray, tau_prev: float) -> tuple[float, dict]:
        params = np.concatenate([state, ref, np.array([tau_prev], dtype=float)])
        x_init = self._initial_guess(state, tau_prev)

        try:
            sol = self.solver(
                x0=x_init,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )
            w = np.array(sol["x"]).flatten()
            self.last_solution = w
            U_opt = w[self.nX:].reshape((self.nu, int(self.cfg.N)), order="F")
            tau = float(U_opt[0, 0])
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))
            return tau, {"success": True, "cost": float(sol["f"])}

        except Exception as exc:
            # Safe fallback: keep previous torque clipped.
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "error": str(exc)}


# ============================================================
# ROS2 Node
# ============================================================

class WheelieNMPCControllerNode(Node):
    def __init__(self):
        super().__init__("wheelie_nmpc_controller")

        # --------------------------
        # ROS parameters
        # --------------------------
        self.declare_parameter("ctrl_dt", 0.1)
        self.declare_parameter("pitch_ref_deg", 90.0)

        self.declare_parameter("m", 5.1)
        self.declare_parameter("l", 0.18)
        self.declare_parameter("I_body", (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2))
        self.declare_parameter("r", 0.085)
        self.declare_parameter("g", 9.81)
        self.declare_parameter("c_v", 9.0)

        self.declare_parameter("tau_min", -8.0)
        self.declare_parameter("tau_max", 12.0)
        self.declare_parameter("theta_min_deg", 0.0)
        self.declare_parameter("theta_max_deg", 100.0)
        self.declare_parameter("omega_min", -8.0)
        self.declare_parameter("omega_max", 8.0)
        self.declare_parameter("v_min", -4.0)
        self.declare_parameter("v_max", 4.0)

        self.declare_parameter("horizon", 35)
        self.declare_parameter("q_x", 0.0)
        self.declare_parameter("q_v", 0.01)
        self.declare_parameter("q_theta", 1000.0)
        self.declare_parameter("q_omega", 15.0)
        self.declare_parameter("r_tau", 0.1)
        self.declare_parameter("r_dtau", 0.01)
        self.declare_parameter("q_terminal_theta", 400.0)
        self.declare_parameter("q_terminal_omega", 100.0)
        self.declare_parameter("ipopt_max_iter", 50)
        self.declare_parameter("ipopt_tol", 1e-4)

        # If true, publish u in [-1, 1], like your MPPI controller.
        # If false, publish raw torque tau [N*m].
        self.declare_parameter("publish_normalized_action", True)

        # Use this if the actuator sign is opposite in MuJoCo.
        self.declare_parameter("reverse_action_sign", False)

        # Optional safety command if sensor data is missing.
        self.declare_parameter("publish_zero_until_ready", True)

        self.ctrl_dt = float(self.get_parameter("ctrl_dt").value)
        self.publish_normalized_action = bool(self.get_parameter("publish_normalized_action").value)
        self.reverse_action_sign = bool(self.get_parameter("reverse_action_sign").value)
        self.publish_zero_until_ready = bool(self.get_parameter("publish_zero_until_ready").value)

        self.p = WheelieParams(
            m=float(self.get_parameter("m").value),
            l=float(self.get_parameter("l").value),
            I_body=float(self.get_parameter("I_body").value),
            r=float(self.get_parameter("r").value),
            g=float(self.get_parameter("g").value),
            c_v=float(self.get_parameter("c_v").value),
            tau_min=float(self.get_parameter("tau_min").value),
            tau_max=float(self.get_parameter("tau_max").value),
            theta_min=math.radians(float(self.get_parameter("theta_min_deg").value)),
            theta_max=math.radians(float(self.get_parameter("theta_max_deg").value)),
            omega_min=float(self.get_parameter("omega_min").value),
            omega_max=float(self.get_parameter("omega_max").value),
            v_min=float(self.get_parameter("v_min").value),
            v_max=float(self.get_parameter("v_max").value),
            pitch_ref_deg=float(self.get_parameter("pitch_ref_deg").value),
        )

        self.cfg = MPCConfig(
            dt=self.ctrl_dt,
            N=int(self.get_parameter("horizon").value),
            q_x=float(self.get_parameter("q_x").value),
            q_v=float(self.get_parameter("q_v").value),
            q_theta=float(self.get_parameter("q_theta").value),
            q_omega=float(self.get_parameter("q_omega").value),
            r_tau=float(self.get_parameter("r_tau").value),
            r_dtau=float(self.get_parameter("r_dtau").value),
            q_terminal_theta=float(self.get_parameter("q_terminal_theta").value),
            q_terminal_omega=float(self.get_parameter("q_terminal_omega").value),
            ipopt_max_iter=int(self.get_parameter("ipopt_max_iter").value),
            ipopt_tol=float(self.get_parameter("ipopt_tol").value),
        )

        self.controller = WheelieNMPC(self.p, self.cfg)

        # --------------------------
        # State variables
        # --------------------------
        self.xpos: Optional[float] = None
        self.xpos_dot: float = 0.0
        self.last_odom_valid = False

        self.pitch: float = 0.0
        self.pitch_dot: float = 0.0
        self.roll: float = 0.0
        self.roll_dot: float = 0.0
        self.yaw: float = 0.0
        self.yaw_dot: float = 0.0
        self.pitch_unwrapped: Optional[float] = None
        self.last_state_valid = False

        self.tau_prev: float = 0.0
        self.warned_no_imu = False
        self.warned_no_odom = False

        self.theta_ref = math.radians(self.p.pitch_ref_deg)
        self.ref = np.array([0.0, 0.0, self.theta_ref, 0.0], dtype=float)

        # --------------------------
        # ROS interfaces
        # --------------------------
        self.cmd_pub = self.create_publisher(Float32, "/cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "/car_imu", self.imu_cb, 10)
        self.odom_sub = self.create_subscription(Odometry, "/car_odom", self.car_callback, 10)
        self.timer = self.create_timer(self.ctrl_dt, self.control_timer_cb)

        self.get_logger().info("Wheelie NMPC ROS/MuJoCo controller initialized.")
        self.get_logger().info(
            f"ctrl_dt={self.ctrl_dt:.3f}, horizon={self.cfg.N}, "
            f"pitch_ref={self.p.pitch_ref_deg:.1f} deg, "
            f"publish_normalized_action={self.publish_normalized_action}, "
            f"reverse_action_sign={self.reverse_action_sign}"
        )
        self.get_logger().info(
            f"Vehicle params: m={self.p.m:.3f}, l={self.p.l:.3f}, "
            f"I_body={self.p.I_body:.5f}, I_eff={self.p.I_eff:.5f}, r={self.p.r:.3f}"
        )

    # ========================================================
    # ROS callbacks
    # ========================================================

    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        try:
            (
                self.roll,
                _bad_euler_pitch,
                self.yaw,
                self.roll_dot,
                _bad_euler_pitch_dot,
                self.yaw_dot,
            ) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

            self.pitch, self.pitch_dot = geometry.quat_to_wheelie_state(
                qw, qx, qy, qz,
                wx, wy, wz,
                prev_pitch_unwrapped=self.pitch_unwrapped,
                pitch_rate_sign=1.0,
            )

            self.pitch = self.pitch * -1
            self.pitch_dot = self.pitch_dot * -1

            self.pitch_unwrapped = self.pitch
            self.last_state_valid = True

        except Exception as exc:
            self.last_state_valid = False
            self.get_logger().warn(f"Failed to compute wheelie pitch from IMU: {exc}")

    def car_callback(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)
        self.last_odom_valid = True

    # ========================================================
    # Control
    # ========================================================

    def control_timer_cb(self):
        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first valid IMU message...")
                self.warned_no_imu = True
            if self.publish_zero_until_ready:
                self.publish_u(0.0)
            return
        self.warned_no_imu = False

        if not self.last_odom_valid or self.xpos is None:
            if not self.warned_no_odom:
                self.get_logger().warn("Waiting for first valid odometry message...")
                self.warned_no_odom = True
            if self.publish_zero_until_ready:
                self.publish_u(0.0)
            return
        self.warned_no_odom = False

        state = np.array(
            [
                float(self.xpos),
                float(self.xpos_dot),
                float(self.pitch),
                float(self.pitch_dot),
            ],
            dtype=float,
        )

        # x_ref is not important when q_x=0, but setting it to current x avoids
        # large numerical offsets if q_x is later increased.
        self.ref[0] = float(self.xpos)
        self.ref[1] = 0.0
        self.ref[2] = float(self.theta_ref)
        self.ref[3] = 0.0

        try:
            tau_cmd, info = self.controller.solve(state, self.ref, self.tau_prev)
        except Exception as exc:
            self.get_logger().error(f"NMPC solve crashed: {exc}")
            self.get_logger().error(traceback.format_exc())
            tau_cmd = float(np.clip(self.tau_prev, self.p.tau_min, self.p.tau_max))
            info = {"success": False, "error": str(exc)}

        if not info.get("success", False):
            self.get_logger().warn(
                "NMPC failed. Using fallback previous torque. "
                f"Error: {info.get('error', 'unknown')}"
            )

        if not math.isfinite(tau_cmd):
            self.get_logger().error("tau_cmd is NaN/Inf. Forcing zero.")
            tau_cmd = 0.0
            self.controller.reset_plan()

        tau_cmd = float(np.clip(tau_cmd, self.p.tau_min, self.p.tau_max))
        self.tau_prev = tau_cmd

        if self.publish_normalized_action:
            u_cmd = self.torque_to_normalized_action(tau_cmd)
        else:
            u_cmd = tau_cmd

        if self.reverse_action_sign:
            u_cmd = -u_cmd

        if self.publish_normalized_action:
            u_cmd = float(np.clip(u_cmd, -1.0, 1.0))

        self.publish_u(u_cmd)

        self.get_logger().info(
            f"x={self.xpos:+.3f}, v={self.xpos_dot:+.3f}, "
            f"pitch={math.degrees(self.pitch):+.2f} deg, "
            f"pitch_dot={self.pitch_dot:+.3f}, "
            f"tau={tau_cmd:+.3f}, u={u_cmd:+.3f}",
            throttle_duration_sec=0.5,
        )

    def torque_to_normalized_action(self, tau: float) -> float:
        """
        Convert NMPC torque to normalized action.

        This maps:
            tau_min -> -1
            tau_max -> +1

        If your MuJoCo actuator uses a different mapping, change only this
        function or run with publish_normalized_action:=false.
        """

        tau = tau/self.p.tau_max
        normalized_tau = np.clip(tau, -1.0, 1.0)

        return normalized_tau

        # tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))
        # center = 0.5 * (self.p.tau_max + self.p.tau_min)
        # half_range = 0.5 * (self.p.tau_max - self.p.tau_min)
        # if half_range <= 1e-9:
        #     return 0.0
        # return float((tau - center) / half_range)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)


# ============================================================
# main()
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = WheelieNMPCControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down NMPC controller, sending u=0.0")
        node.publish_u(0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
