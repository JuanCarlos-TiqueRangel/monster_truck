#!/usr/bin/env python3
import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32, Int64
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


# ============================================================
# EASY EXPERIMENT FLAGS / USER OPTIONS
# ============================================================
START_FLIPPED = False
ENABLE_VIEWER = True

# Drive model:
#   "electrical" -> recommended, closer to ESC + motor + battery behavior
#   "simple"     -> older torque-envelope style
DRIVE_MODEL = "electrical"

# ESC mode:
#   "bidirectional"         -> immediate forward/reverse drive
#   "forward_brake_reverse" -> more RC-like approximation
ESC_MODE = "bidirectional"

# Use nominal 4S or fully charged feel
USE_FULLY_CHARGED_PACK = True

# Optional ESC command lag
ENABLE_ESC_FILTER = True
ESC_TAU = 0.035

# These are PLANAR ROOT heights (root_z), not freejoint body heights.
UPRIGHT_Z0 = 0.169
FLIPPED_Z0 = 0.182

# Debug logging
LOG_DRIVE_DEBUG = False

I_DRIVE_MAX = 160.0
MAX_TENDON_TORQUE_CMD = 45.0


class MujocoImuNode(Node):
    def __init__(self):
        super().__init__("mujoco_imu_node")

        # ---------------- Model loading ----------------
        script_dir = Path(__file__).resolve().parent
        xml_path = script_dir / cfg_params.files.mujoco_model

        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # One external action advances exactly one ctrl_dt of simulation.
        self.sim_dt = float(self.model.opt.timestep)
        self.ctrl_dt = 0.01
        self.steps_per_ctrl = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.get_logger().info(
            f"Loaded {xml_path}, sim_dt={self.sim_dt:.6f}, "
            f"ctrl_dt={self.ctrl_dt:.6f}, steps_per_ctrl={self.steps_per_ctrl}"
        )

        # ---------------- Find planar root joints ----------------
        self.root_x_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_x")
        self.root_z_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_z")
        self.root_pitch_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_pitch")

        if self.root_x_jid < 0 or self.root_z_jid < 0 or self.root_pitch_jid < 0:
            raise RuntimeError("Planar joints 'root_x', 'root_z', 'root_pitch' not found in model")

        self.root_x_qpos_adr = int(self.model.jnt_qposadr[self.root_x_jid])
        self.root_z_qpos_adr = int(self.model.jnt_qposadr[self.root_z_jid])
        self.root_pitch_qpos_adr = int(self.model.jnt_qposadr[self.root_pitch_jid])

        self.root_x_dof_adr = int(self.model.jnt_dofadr[self.root_x_jid])
        self.root_z_dof_adr = int(self.model.jnt_dofadr[self.root_z_jid])
        self.root_pitch_dof_adr = int(self.model.jnt_dofadr[self.root_pitch_jid])

        # ---------------- Chassis body id ----------------
        self.chassis_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "chassis")
        if self.chassis_body_id < 0:
            raise RuntimeError("Body 'chassis' not found in XML")

        # ---------------- IMU sensors ----------------
        gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id < 0:
            raise RuntimeError("imu_gyro sensor not found in XML")
        self.gyro_adr = int(self.model.sensor_adr[gyro_id])
        self.gyro_dim = int(self.model.sensor_dim[gyro_id])

        acc_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_acc")
        if acc_id < 0:
            raise RuntimeError("imu_acc sensor not found in XML")
        self.acc_adr = int(self.model.sensor_adr[acc_id])
        self.acc_dim = int(self.model.sensor_dim[acc_id])

        # ---------------- Obstacle geom id ----------------
        self.obs_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "obs_box_1")
        if self.obs_geom_id < 0:
            raise RuntimeError("Geom 'obs_box_1' not found in XML")

        # ---------------- Drive actuator ----------------
        self.drive_act_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
        if self.drive_act_id < 0:
            raise RuntimeError("Actuator 'drive_motor' not found in XML")

        # ---------------- Wheel joint DOF addresses ----------------
        self.wheel_dof = {}
        for jname in ["j_fl", "j_fr", "j_rl", "j_rr"]:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' not found in XML")
            self.wheel_dof[jname] = int(self.model.jnt_dofadr[jid])

        # ---------------- Command ----------------
        self.last_action = 0.0
        self.u_esc = 0.0

        # Step bookkeeping.
        # We publish both a readable step id and the exact stamp ns used by IMU/odom.
        self.step_count = 0

        # ---------------- Drive / ESC / battery model ----------------
        self.drive_model = DRIVE_MODEL
        self.esc_mode = ESC_MODE
        self.enable_esc_filter = ENABLE_ESC_FILTER
        self.esc_tau = float(ESC_TAU)

        self.V_bat_oc = 16.8 if USE_FULLY_CHARGED_PACK else 14.8
        self.R_bat = 0.012
        self.R_esc = 0.004
        self.R_motor = 0.028
        self.R_total = self.R_bat + self.R_esc + self.R_motor

        self.Kv_rpm_per_V = 2800.0
        self.Kt = 60.0 / (2.0 * np.pi * self.Kv_rpm_per_V)
        self.Ke = self.Kt

        self.final_drive = 18.76
        self.eta_drive = 0.85

        self.I_drive_max = I_DRIVE_MAX
        self.I_brake_max = 90.0

        self.neutral_deadband = 0.02
        self.brake_speed_thresh = 8.0

        self.wheel_drag_visc = 0.015
        self.wheel_drag_coulomb = 0.06
        self.drag_tanh_eps = 2.0

        self.max_tendon_torque_cmd = MAX_TENDON_TORQUE_CMD

        self.tau_motor_max = self.Kt * self.I_drive_max
        self.tau_drive_max = self.eta_drive * self.final_drive * self.tau_motor_max
        self.omega_wheel_nl = self.V_bat_oc / max(self.Ke * self.final_drive, 1e-9)

        self.get_logger().info(
            f"Drive model={self.drive_model}, ESC mode={self.esc_mode}, "
            f"V_bat_oc={self.V_bat_oc:.2f} V, "
            f"tau_drive_max≈{self.tau_drive_max:.3f} N*m, "
            f"omega_wheel_nl≈{self.omega_wheel_nl:.2f} rad/s, "
            f"max actuator cmd={self.max_tendon_torque_cmd:.2f}"
        )

        try:
            ctrl_lo = float(self.model.actuator_ctrlrange[self.drive_act_id, 0])
            ctrl_hi = float(self.model.actuator_ctrlrange[self.drive_act_id, 1])
            if abs(ctrl_lo) < self.max_tendon_torque_cmd or abs(ctrl_hi) < self.max_tendon_torque_cmd:
                self.get_logger().warning(
                    f"XML actuator ctrlrange is [{ctrl_lo:.3f}, {ctrl_hi:.3f}] "
                    f"but Python may command ±{self.max_tendon_torque_cmd:.3f}. "
                    "Increase XML ctrlrange if needed."
                )
        except Exception:
            pass

        # ---------------- Reset once + apply start pose ----------------
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._apply_start_pose_flag()

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # ---------------- Viewer (optional) ----------------
        self.viewer = None
        if ENABLE_VIEWER:
            try:
                self.viewer = mjviewer.launch_passive(self.model, self.data)
            except Exception as e:
                self.get_logger().warning(f"Viewer not started (headless?): {e}")

        # ---------------- ROS interfaces ----------------
        self.sub_cmd = self.create_subscription(Float32, "/cmd_action", self.cmd_action_cb, 10)

        self.pub_imu = self.create_publisher(Imu, "/car_imu", 10)
        self.pub_odom = self.create_publisher(Odometry, "/car_odom", 10)
        self.pub_obs_pose = self.create_publisher(PoseStamped, "/obstacle_pose", 10)
        self.pub_sim_step_id = self.create_publisher(Int64, "/sim_step_id", 10)
        self.pub_sim_step_stamp_ns = self.create_publisher(Int64, "/sim_step_stamp_ns", 10)

        self.reset_srv = self.create_service(Trigger, "reset_car", self.reset_callback)

        self.get_logger().info(
            f"MujocoImuNode ready. START_FLIPPED={START_FLIPPED}. "
            "Simulation is EVENT-DRIVEN: it waits for /cmd_action. "
            "Each new /cmd_action advances MuJoCo by one ctrl_dt and then publishes state."
        )

        # Publish one initial state now.
        self._publish_all_current_state(announce_step=True)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _apply_start_pose_flag(self) -> None:
        if START_FLIPPED:
            pitch0 = np.pi
            z0 = FLIPPED_Z0
        else:
            pitch0 = 0.0
            z0 = UPRIGHT_Z0

        self.data.qpos[self.root_x_qpos_adr] = 0.0
        self.data.qpos[self.root_z_qpos_adr] = z0
        self.data.qpos[self.root_pitch_qpos_adr] = pitch0

        self.data.qvel[self.root_x_dof_adr] = 0.0
        self.data.qvel[self.root_z_dof_adr] = 0.0
        self.data.qvel[self.root_pitch_dof_adr] = 0.0

        for jname in ["j_fl", "j_fr", "j_rl", "j_rr"]:
            self.data.qvel[self.wheel_dof[jname]] = 0.0

        self.data.ctrl[:] = 0.0
        self.u_esc = 0.0

        mj.mj_forward(self.model, self.data)

    def _wheel_speeds(self):
        w_fl = float(self.data.qvel[self.wheel_dof["j_fl"]])
        w_fr = float(self.data.qvel[self.wheel_dof["j_fr"]])
        w_rl = float(self.data.qvel[self.wheel_dof["j_rl"]])
        w_rr = float(self.data.qvel[self.wheel_dof["j_rr"]])
        return w_fl, w_fr, w_rl, w_rr

    def _avg_wheel_speed(self) -> float:
        w_fl, w_fr, w_rl, w_rr = self._wheel_speeds()
        return 0.25 * (w_fl + w_fr + w_rl + w_rr)

    def _wheel_side_drag_torque(self, w_avg: float) -> float:
        return (
            self.wheel_drag_visc * w_avg
            + self.wheel_drag_coulomb * np.tanh(w_avg / max(self.drag_tanh_eps, 1e-9))
        )

    def _compute_drive_torque_simple(self, u: float) -> float:
        if abs(u) < self.neutral_deadband:
            return 0.0

        w_avg = self._avg_wheel_speed()
        speed_factor = max(0.0, 1.0 - abs(w_avg) / max(self.omega_wheel_nl, 1e-9))
        tau_drive = u * self.tau_drive_max * speed_factor
        tau_drag = self._wheel_side_drag_torque(w_avg)
        tau_out = tau_drive - tau_drag
        return float(np.clip(tau_out, -self.max_tendon_torque_cmd, self.max_tendon_torque_cmd))

    def _compute_drive_torque_electrical(self, u: float) -> float:
        w_avg = self._avg_wheel_speed()
        omega_motor = self.final_drive * w_avg

        if abs(u) < self.neutral_deadband:
            I_cmd = 0.0
        else:
            if self.esc_mode == "bidirectional":
                V_cmd = u * self.V_bat_oc
                I_cmd = (V_cmd - self.Ke * omega_motor) / max(self.R_total, 1e-9)
                I_cmd = float(np.clip(I_cmd, -self.I_drive_max, self.I_drive_max))
            elif self.esc_mode == "forward_brake_reverse":
                if u < -self.neutral_deadband and w_avg > self.brake_speed_thresh:
                    I_cmd = -abs(u) * self.I_brake_max
                elif u > self.neutral_deadband and w_avg < -self.brake_speed_thresh:
                    I_cmd = abs(u) * self.I_brake_max
                else:
                    V_cmd = u * self.V_bat_oc
                    I_cmd = (V_cmd - self.Ke * omega_motor) / max(self.R_total, 1e-9)
                    I_cmd = float(np.clip(I_cmd, -self.I_drive_max, self.I_drive_max))
            else:
                raise ValueError(f"Unknown esc_mode: {self.esc_mode}")

        tau_motor = self.Kt * I_cmd
        tau_drive = self.eta_drive * self.final_drive * tau_motor
        tau_drag = self._wheel_side_drag_torque(w_avg)
        tau_out = tau_drive - tau_drag
        tau_out = float(np.clip(tau_out, -self.max_tendon_torque_cmd, self.max_tendon_torque_cmd))

        if LOG_DRIVE_DEBUG:
            self.get_logger().info(
                f"u={u:+.3f}, w_avg={w_avg:+.2f}, omega_motor={omega_motor:+.2f}, "
                f"tau_drive={tau_drive:+.3f}, tau_drag={tau_drag:+.3f}, tau_out={tau_out:+.3f}"
            )

        return tau_out

    def _compute_drive_torque_from_throttle(self, u: float) -> float:
        if self.drive_model == "simple":
            return self._compute_drive_torque_simple(u)
        if self.drive_model == "electrical":
            return self._compute_drive_torque_electrical(u)
        raise ValueError(f"Unknown drive_model: {self.drive_model}")

    def _publish_all_current_state(self, announce_step: bool):
        now = self.get_clock().now()
        stamp = now.to_msg()
        stamp_ns = int(now.nanoseconds)

        self._publish_obstacle_pose(stamp)
        self._publish_imu(stamp)
        self._publish_odom(stamp)

        if announce_step:
            msg_id = Int64()
            msg_id.data = int(self.step_count)
            self.pub_sim_step_id.publish(msg_id)

            msg_stamp = Int64()
            msg_stamp.data = stamp_ns
            self.pub_sim_step_stamp_ns.publish(msg_stamp)

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------
    def cmd_action_cb(self, msg: Float32) -> None:
        self.last_action = float(np.clip(msg.data, -1.0, 1.0))
        self._step_once_from_latest_action()

    def reset_callback(self, request, response):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self._apply_start_pose_flag()

        # Announce a fresh state immediately after reset.
        self.step_count += 1
        self._publish_all_current_state(announce_step=True)

        response.success = True
        response.message = (
            f"Car reset in MuJoCo planar mode (START_FLIPPED={START_FLIPPED}). "
            "Initial post-reset observation published."
        )
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------
    # Publishers
    # ------------------------------------------------------------
    def _publish_obstacle_pose(self, stamp):
        p = self.data.geom_xpos[self.obs_geom_id]
        xmat = self.data.geom_xmat[self.obs_geom_id]

        quat = np.zeros(4, dtype=np.float64)
        mj.mju_mat2Quat(quat, xmat)

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "world"
        msg.pose.position.x = float(p[0])
        msg.pose.position.y = float(p[1])
        msg.pose.position.z = float(p[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        self.pub_obs_pose.publish(msg)

    def _publish_imu(self, stamp):
        qw, qx, qy, qz = self.data.xquat[self.chassis_body_id]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + self.gyro_dim]
        acc = self.data.sensordata[self.acc_adr:self.acc_adr + self.acc_dim]

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = "base_link"
        imu_msg.orientation.w = float(qw)
        imu_msg.orientation.x = float(qx)
        imu_msg.orientation.y = float(qy)
        imu_msg.orientation.z = float(qz)

        if self.gyro_dim >= 3:
            imu_msg.angular_velocity.x = float(gyro[0])
            imu_msg.angular_velocity.y = float(gyro[1])
            imu_msg.angular_velocity.z = float(gyro[2])

        if self.acc_dim >= 3:
            imu_msg.linear_acceleration.x = float(acc[0])
            imu_msg.linear_acceleration.y = float(acc[1])
            imu_msg.linear_acceleration.z = float(acc[2])

        imu_msg.orientation_covariance[0] = -1.0
        imu_msg.angular_velocity_covariance[0] = -1.0
        imu_msg.linear_acceleration_covariance[0] = -1.0
        self.pub_imu.publish(imu_msg)

    def _publish_odom(self, stamp):
        x, y, z = self.data.xpos[self.chassis_body_id]
        qw, qx, qy, qz = self.data.xquat[self.chassis_body_id]

        vel6 = np.zeros(6, dtype=np.float64)
        mj.mj_objectVelocity(
            self.model,
            self.data,
            mj.mjtObj.mjOBJ_BODY,
            self.chassis_body_id,
            vel6,
            0,
        )

        wx, wy, wz, vx, vy, vz = vel6

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)
        odom.pose.pose.orientation.w = float(qw)
        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)

        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.linear.z = float(vz)
        odom.twist.twist.angular.x = float(wx)
        odom.twist.twist.angular.y = float(wy)
        odom.twist.twist.angular.z = float(wz)

        odom.pose.covariance[0] = -1.0
        odom.twist.covariance[0] = -1.0
        self.pub_odom.publish(odom)

    # ------------------------------------------------------------
    # Event-driven sim step
    # ------------------------------------------------------------
    def _step_once_from_latest_action(self) -> None:
        u_cmd = float(np.clip(self.last_action, -1.0, 1.0))

        if self.enable_esc_filter:
            alpha = 1.0 - np.exp(-self.ctrl_dt / max(self.esc_tau, 1e-9))
            self.u_esc += alpha * (u_cmd - self.u_esc)
            u_eff = self.u_esc
        else:
            u_eff = u_cmd

        tau_cmd = self._compute_drive_torque_from_throttle(u_eff)

        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.drive_act_id] = tau_cmd

        for _ in range(self.steps_per_ctrl):
            mj.mj_step(self.model, self.data)

        self.step_count += 1

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        self._publish_all_current_state(announce_step=True)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()