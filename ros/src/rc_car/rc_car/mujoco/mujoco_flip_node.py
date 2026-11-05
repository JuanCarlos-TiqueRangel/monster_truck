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


class MujocoFlipOdomNode(Node):
    def __init__(self):
        super().__init__("mujoco_flip_odom_node")

        default_xml = Path(__file__).resolve().parent / "monster_truck_flip_2d.xml"

        self.declare_parameter("xml_path", str(default_xml))
        self.declare_parameter("publish_hz", 200.0)
        self.declare_parameter("render", True)
        self.declare_parameter("viewer_hz", 60.0)

        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_z", 0.202)
        self.declare_parameter("initial_pitch_deg", 180.0)

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("child_frame_id", "base_link")

        xml_path = Path(
            self.get_parameter("xml_path").get_parameter_value().string_value
        ).expanduser().resolve()

        publish_hz = float(self.get_parameter("publish_hz").value)
        self.render = bool(self.get_parameter("render").value)
        self.viewer_hz = float(self.get_parameter("viewer_hz").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)

        if not xml_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        if publish_hz <= 0.0:
            raise ValueError("publish_hz must be positive")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # The XML must contain: <freejoint name="root"/>
        self.root_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "root",
        )
        if self.root_id < 0:
            raise RuntimeError(
                "Joint 'root' was not found. Replace root_x, root_z, and "
                "root_pitch in the XML with <freejoint name=\"root\"/>."
            )
        if self.model.jnt_type[self.root_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Joint 'root' exists but is not a free joint")

        self.root_qpos = int(self.model.jnt_qposadr[self.root_id])
        self.root_qvel = int(self.model.jnt_dofadr[self.root_id])

        self.drive_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "drive_motor",
        )
        if self.drive_id < 0:
            raise RuntimeError("Actuator 'drive_motor' was not found in the XML")

        self.tau_min = float(self.model.actuator_ctrlrange[self.drive_id, 0])
        self.tau_max = float(self.model.actuator_ctrlrange[self.drive_id, 1])
        self.drive_command = 0.0

        self.sim_dt = float(self.model.opt.timestep)
        requested_period = 1.0 / publish_hz
        self.steps_per_update = max(1, round(requested_period / self.sim_dt))
        self.update_period = self.steps_per_update * self.sim_dt
        self.actual_publish_hz = 1.0 / self.update_period

        self.odom_pub = self.create_publisher(Odometry, ODOM_TOPIC, 10)
        self.drive_sub = self.create_subscription(
            Float32,
            DRIVE_TOPIC,
            self.drive_callback,
            10,
        )

        self.viewer = None
        self.last_viewer_sync = 0.0

        self.reset_flipped()

        if self.render:
            self.start_viewer()

        self.timer = self.create_timer(
            self.update_period,
            self.simulation_callback,
        )

        self.publish_odometry()

        position = self.data.qpos[self.root_qpos : self.root_qpos + 3]
        self.get_logger().info(f"Loaded: {xml_path}")
        self.get_logger().info(
            f"MuJoCo dt={self.sim_dt:.6f} s | "
            f"steps/update={self.steps_per_update} | "
            f"odom={self.actual_publish_hz:.1f} Hz"
        )
        self.get_logger().info(
            f"Initial position: x={position[0]:.3f}, "
            f"y={position[1]:.3f}, z={position[2]:.3f}"
        )

    def reset_flipped(self):
        mujoco.mj_resetData(self.model, self.data)

        x = float(self.get_parameter("initial_x").value)
        y = float(self.get_parameter("initial_y").value)
        z = float(self.get_parameter("initial_z").value)
        pitch = math.radians(
            float(self.get_parameter("initial_pitch_deg").value)
        )

        q = self.root_qpos

        # Free-joint position [x, y, z].
        self.data.qpos[q + 0] = x
        self.data.qpos[q + 1] = y
        self.data.qpos[q + 2] = z

        # MuJoCo quaternion order [w, x, y, z].
        self.data.qpos[q + 3] = math.cos(0.5 * pitch)
        self.data.qpos[q + 4] = 0.0
        self.data.qpos[q + 5] = math.sin(0.5 * pitch)
        self.data.qpos[q + 6] = 0.0

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.drive_command = 0.0

        mujoco.mj_forward(self.model, self.data)

    def start_viewer(self):
        try:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
            )

            position = self.data.qpos[self.root_qpos : self.root_qpos + 3]
            self.viewer.cam.lookat[:] = np.array(
                [float(position[0]), float(position[1]), 0.15]
            )
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 90.0
            self.viewer.cam.elevation = -15.0

        except Exception as error:
            self.viewer = None
            self.render = False
            self.get_logger().warning(f"Viewer disabled: {error}")

    def drive_callback(self, msg: Float32):
        self.drive_command = float(
            np.clip(float(msg.data), self.tau_min, self.tau_max)
        )

    def simulation_callback(self):
        self.data.ctrl[self.drive_id] = self.drive_command

        for _ in range(self.steps_per_update):
            mujoco.mj_step(self.model, self.data)

        self.publish_odometry()
        self.update_viewer()

    def publish_odometry(self):
        q = self.root_qpos
        v = self.root_qvel

        x, y, z = [float(value) for value in self.data.qpos[q : q + 3]]
        qw, qx, qy, qz = [float(value) for value in self.data.qpos[q + 3 : q + 7]]
        vx, vy, vz = [float(value) for value in self.data.qvel[v : v + 3]]
        wx, wy, wz = [float(value) for value in self.data.qvel[v + 3 : v + 6]]

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = z

        # ROS quaternion order [x, y, z, w].
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
        viewer_period = 1.0 / max(self.viewer_hz, 1.0)

        if now - self.last_viewer_sync >= viewer_period:
            position = self.data.qpos[self.root_qpos : self.root_qpos + 3]
            self.viewer.cam.lookat[0] = float(position[0])
            self.viewer.cam.lookat[1] = float(position[1])
            self.viewer.sync()
            self.last_viewer_sync = now

    def destroy_node(self):
        self.drive_command = 0.0
        self.data.ctrl[self.drive_id] = 0.0

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoFlipOdomNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()