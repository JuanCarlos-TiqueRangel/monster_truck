#!/usr/bin/env python3

import math
from pathlib import Path

import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


# ============================================================
# SETTINGS
# ============================================================

_ROOT = Path(__file__).resolve().parent
XML_PATH = _ROOT.parent / "monster_truck_flip_2d.xml"

PUBLISH_HZ = 100.0
RENDER = True

INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_PITCH_DEG = 0.0


# ============================================================
# MUJOCO NODE
# ============================================================

class MujocoNode(Node):

    def __init__(self):
        super().__init__("mujoco_node")

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        # Joints
        self.x_q, self.x_v = self._joint("root_x")
        self.z_q, self.z_v = self._joint("root_z")
        self.pitch_q, self.pitch_v = self._joint("root_pitch")

        # Sensors
        self.quat_adr = self._sensor("imu_quat")
        self.gyro_adr = self._sensor("imu_gyro")
        self.acc_adr = self._sensor("imu_acc")

        # Actuator
        self.drive = self._actuator("drive_motor")
        self.ctrl_min = float(self.model.actuator_ctrlrange[self.drive, 0])
        self.ctrl_max = float(self.model.actuator_ctrlrange[self.drive, 1])
        self.drive_command = 0.0

        # Simulation timing
        self.sim_dt = float(self.model.opt.timestep)
        requested_period = 1.0 / PUBLISH_HZ
        self.steps_per_update = max(1, round(requested_period / self.sim_dt))
        self.update_period = self.steps_per_update * self.sim_dt

        # ROS
        self.odom_pub = self.create_publisher(Odometry, "/mujoco/odom", 10)
        self.imu_pub = self.create_publisher(Imu, "/mujoco/imu", 10)
        self.drive_sub = self.create_subscription(Float32, "/drive", self.drive_callback, 10)

        self.viewer = None

        self.reset_simulation()

        self.timer = self.create_timer(self.update_period, self.simulation_callback)

        self.get_logger().info(f"MuJoCo running at {1.0 / self.update_period:.1f} Hz")

    # ========================================================
    # MUJOCO HELPERS
    # ========================================================

    def _joint(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if jid < 0:
            raise RuntimeError(f"Joint not found: {name}")

        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)

        if sid < 0:
            raise RuntimeError(f"Sensor not found: {name}")

        return int(self.model.sensor_adr[sid])

    def _actuator(self, name):
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

        if aid < 0:
            raise RuntimeError(f"Actuator not found: {name}")

        return int(aid)

    # ========================================================
    # RESET
    # ========================================================

    def reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[self.x_q] = INITIAL_X
        self.data.qpos[self.z_q] = INITIAL_Z
        self.data.qpos[self.pitch_q] = math.radians(INITIAL_PITCH_DEG)
        self.data.qvel[:] = 0.0
        self.data.ctrl[self.drive] = 0.0

        mujoco.mj_forward(self.model, self.data)

    # ========================================================
    # DRIVE
    # ========================================================

    def drive_callback(self, msg):
        self.drive_command = float(msg.data)

    # ========================================================
    # SIMULATION
    # ========================================================

    def simulation_callback(self):
        self.data.ctrl[self.drive] = max(self.ctrl_min, min(self.drive_command, self.ctrl_max))

        for _ in range(self.steps_per_update):
            mujoco.mj_step(self.model, self.data)

        self.publish_odometry()
        self.publish_imu()

        if self.viewer is not None:
            self.viewer.sync()

    # ========================================================
    # ODOMETRY
    # ========================================================

    def publish_odometry(self):
        msg = Odometry()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        # Position
        msg.pose.pose.position.x = float(self.data.qpos[self.x_q])
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = float(self.data.qpos[self.z_q])

        # MuJoCo quaternion = [w, x, y, z]
        qw = float(self.data.sensordata[self.quat_adr])
        qx = float(self.data.sensordata[self.quat_adr + 1])
        qy = float(self.data.sensordata[self.quat_adr + 2])
        qz = float(self.data.sensordata[self.quat_adr + 3])

        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Linear velocity
        msg.twist.twist.linear.x = float(self.data.qvel[self.x_v])
        msg.twist.twist.linear.y = 0.0
        msg.twist.twist.linear.z = float(self.data.qvel[self.z_v])

        # Angular velocity
        msg.twist.twist.angular.x = float(self.data.sensordata[self.gyro_adr])
        msg.twist.twist.angular.y = float(self.data.sensordata[self.gyro_adr + 1])
        msg.twist.twist.angular.z = float(self.data.sensordata[self.gyro_adr + 2])

        self.odom_pub.publish(msg)

    # ========================================================
    # IMU
    # ========================================================

    def publish_imu(self):
        msg = Imu()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        # MuJoCo quaternion = [w, x, y, z]
        qw = float(self.data.sensordata[self.quat_adr])
        qx = float(self.data.sensordata[self.quat_adr + 1])
        qy = float(self.data.sensordata[self.quat_adr + 2])
        qz = float(self.data.sensordata[self.quat_adr + 3])

        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        # Angular velocity
        msg.angular_velocity.x = float(self.data.sensordata[self.gyro_adr])
        msg.angular_velocity.y = float(self.data.sensordata[self.gyro_adr + 1])
        msg.angular_velocity.z = float(self.data.sensordata[self.gyro_adr + 2])

        # Linear acceleration
        msg.linear_acceleration.x = float(self.data.sensordata[self.acc_adr])
        msg.linear_acceleration.y = float(self.data.sensordata[self.acc_adr + 1])
        msg.linear_acceleration.z = float(self.data.sensordata[self.acc_adr + 2])

        self.imu_pub.publish(msg)


# ============================================================
# MAIN
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()

    try:
        if RENDER:
            with mujoco.viewer.launch_passive(node.model, node.data) as viewer:
                node.viewer = viewer

                while rclpy.ok() and viewer.is_running():
                    rclpy.spin_once(node, timeout_sec=0.001)
        else:
            rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()