#!/usr/bin/env python3

import math
import time
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32


ODOM_TOPIC = "/mujoco/odom"
DRIVE_TOPIC = "/drive"
STEERING_TOPIC = "/steering"


class Mujoco4WDOdomNode(Node):
    def __init__(self):
        super().__init__("mujoco_4wd_odom_node")

        default_xml = Path(__file__).resolve().parent / "small_monster_truck.xml"

        self.declare_parameter("xml_path", str(default_xml))
        self.declare_parameter("publish_hz", 200.0)
        self.declare_parameter("render", True)
        self.declare_parameter("viewer_hz", 60.0)
        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_z", 0.20)
        self.declare_parameter("initial_pitch_deg", 0.0)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")

        xml_path = Path(str(self.get_parameter("xml_path").value)).expanduser().resolve()
        publish_hz = float(self.get_parameter("publish_hz").value)
        self.render = bool(self.get_parameter("render").value)
        self.viewer_hz = float(self.get_parameter("viewer_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)

        if not xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")
        if self.viewer_hz <= 0.0:
            raise ValueError("viewer_hz must be positive")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.root_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if self.root_id < 0 or self.model.jnt_type[self.root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("The XML must contain a free joint named 'root'")
        self.root_qpos = int(self.model.jnt_qposadr[self.root_id])
        self.chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if self.chassis_id < 0:
            raise RuntimeError("Body 'chassis' was not found in the XML")

        self.drive_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive")
        self.steering_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steering_servo")
        if self.drive_id < 0 or self.steering_id < 0:
            raise RuntimeError("The XML must define actuators 'drive' and 'steering_servo'")

        self.drive_min, self.drive_max = map(float, self.model.actuator_ctrlrange[self.drive_id])
        self.steering_min, self.steering_max = map(float, self.model.actuator_ctrlrange[self.steering_id])
        self.drive_command = 0.0
        self.steering_command = 0.0

        self.sim_dt = float(self.model.opt.timestep)
        self.publish_period = 1.0 / publish_hz
        if self.publish_period < self.sim_dt:
            raise ValueError(f"publish_hz must not exceed MuJoCo step frequency ({1.0 / self.sim_dt:.1f} Hz)")
        self.step_accumulator = 0.0

        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.drive_sub = self.create_subscription(Float32, DRIVE_TOPIC, self.drive_callback, 10)
        self.steering_sub = self.create_subscription(Float32, STEERING_TOPIC, self.steering_callback, 10)

        self.viewer = None
        self.last_viewer_sync = 0.0
        self.reset_simulation()
        if self.render:
            self.start_viewer()

        self.timer = self.create_timer(self.publish_period, self.simulation_callback)
        self.publish_odometry()
        self.get_logger().info(f"Loaded: {xml_path}")
        self.get_logger().info(f"MuJoCo dt={self.sim_dt:.4f} s | odom requested={publish_hz:.1f} Hz")
        self.get_logger().info(f"Controls: {DRIVE_TOPIC} [{self.drive_min}, {self.drive_max}], "
                               f"{STEERING_TOPIC} [{self.steering_min}, {self.steering_max}] rad")

    def reset_simulation(self):
        mujoco.mj_resetData(self.model, self.data)

        q = self.root_qpos
        pitch = math.radians(float(self.get_parameter("initial_pitch_deg").value))
        self.data.qpos[q:q + 3] = [float(self.get_parameter("initial_x").value),
                                  float(self.get_parameter("initial_y").value),
                                  float(self.get_parameter("initial_z").value)]
        self.data.qpos[q + 3:q + 7] = [math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.drive_command = 0.0
        self.steering_command = 0.0
        self.step_accumulator = 0.0
        mujoco.mj_forward(self.model, self.data)

    def start_viewer(self):
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            position = self.data.qpos[self.root_qpos:self.root_qpos + 3]
            self.viewer.cam.lookat[:] = np.array([position[0], position[1], 0.12])
            self.viewer.cam.distance = 1.2
            self.viewer.cam.azimuth = 120.0
            self.viewer.cam.elevation = -20.0
        except Exception as error:
            self.viewer = None
            self.render = False
            self.get_logger().warning(f"Viewer disabled: {error}")

    def drive_callback(self, msg: Float32):
        self.drive_command = float(np.clip(msg.data, self.drive_min, self.drive_max))

    def steering_callback(self, msg: Float32):
        self.steering_command = float(np.clip(msg.data, self.steering_min, self.steering_max))

    def simulation_callback(self):
        self.data.ctrl[self.drive_id] = self.drive_command
        self.data.ctrl[self.steering_id] = self.steering_command

        # At 200 Hz publishing and a 0.002 s MuJoCo step, alternate 2 and 3 steps.
        self.step_accumulator += self.publish_period
        steps = int((self.step_accumulator + 1e-12) / self.sim_dt)
        self.step_accumulator -= steps * self.sim_dt
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        self.publish_odometry()
        self.update_viewer()

    def publish_odometry(self):
        q = self.root_qpos
        x, y, z = map(float, self.data.qpos[q:q + 3])
        qw, qx, qy, qz = map(float, self.data.qpos[q + 3:q + 7])

        # ROS Odometry twist is expressed in child_frame_id (vehicle frame).
        # MuJoCo returns [angular_xyz, linear_xyz] for mj_objectVelocity.
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.chassis_id, velocity, 1)
        wx, wy, wz = map(float, velocity[:3])
        vx, vy, vz = map(float, velocity[3:])

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = vz
        msg.twist.twist.angular.x = wx
        msg.twist.twist.angular.y = wy
        msg.twist.twist.angular.z = wz
        self.odom_pub.publish(msg)

    def update_viewer(self):
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            self.viewer.close()
            self.viewer = None
            return

        now = time.monotonic()
        if now - self.last_viewer_sync >= 1.0 / self.viewer_hz:
            position = self.data.qpos[self.root_qpos:self.root_qpos + 3]
            self.viewer.cam.lookat[0] = float(position[0])
            self.viewer.cam.lookat[1] = float(position[1])
            self.viewer.sync()
            self.last_viewer_sync = now

    def destroy_node(self):
        self.drive_command = 0.0
        self.steering_command = 0.0
        self.data.ctrl[self.drive_id] = 0.0
        self.data.ctrl[self.steering_id] = 0.0
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Mujoco4WDOdomNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
