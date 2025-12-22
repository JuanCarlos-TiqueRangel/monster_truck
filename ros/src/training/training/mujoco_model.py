#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer   # <-- NEW

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu


class MujocoImuNode(Node):
    def __init__(self):
        super().__init__("mujoco_imu_node")

        # ---------------- Model loading ----------------
        script_dir = Path(__file__).resolve().parent
        xml_path = script_dir / "monstertruck.xml"   # adjust if needed

        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)

        self.data.ctrl[:] = 0.0

        # Timing
        self.sim_dt = float(self.model.opt.timestep)
        self.ctrl_dt = 0.01  # 100 Hz
        self.steps_per_ctrl = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.get_logger().info(
            f"Loaded {xml_path}, sim_dt={self.sim_dt:.6f}, "
            f"ctrl_dt={self.ctrl_dt:.6f}, steps_per_ctrl={self.steps_per_ctrl}"
        )

        # Free joint (base orientation)
        free_j = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                free_j = j
                break
        if free_j is None:
            raise RuntimeError("No free joint found in model")
        self.qadr = self.model.jnt_qposadr[free_j] + 3  # qw,qx,qy,qz start

        # IMU sensors
        gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id < 0:
            raise RuntimeError("imu_gyro sensor not found in XML")
        self.gyro_adr = self.model.sensor_adr[gyro_id]
        self.gyro_dim = self.model.sensor_dim[gyro_id]

        acc_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_acc")
        if acc_id < 0:
            raise RuntimeError("imu_acc sensor not found in XML")
        self.acc_adr = self.model.sensor_adr[acc_id]
        self.acc_dim = self.model.sensor_dim[acc_id]

        # ---- MuJoCo viewer (passive, non-blocking) ----
        # This opens a window and renders the same model/data you are stepping.
        # If there is no DISPLAY (headless container), this will raise an error.
        self.viewer = mjviewer.launch_passive(self.model, self.data)

        # ------------- ----- ROS interfaces ----------------------------------
        self.last_action = 0.0
        self.sub_cmd = self.create_subscription(
            Float32,
            "cmd_action",
            self.cmd_action_cb,
            10,
        )
        self.pub_imu = self.create_publisher(Imu, "car_imu", 10)

        self.reset_srv = self.create_service(
            Trigger,
            'reset_car',
            self.reset_callback
        )

        # store initial state for resets
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        self.timer = self.create_timer(self.ctrl_dt, self.timer_cb)

        self.get_logger().info(
            "MujocoImuNode ready. Subscribing /cmd_action, publishing /car_imu, rendering enabled."
        )

    def cmd_action_cb(self, msg: Float32) -> None:
        self.last_action = float(msg.data)

    def reset_callback(self, request, response):
        # Reset MuJoCo state
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mj.mj_forward(self.model, self.data)

        # Optionally zero last action
        #self.last_action[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Optionally publish fresh state after reset
        state = Float32()
        state.data = float(self.data.qpos[0])
        #self.state_pub.publish(state)

        response.success = True
        response.message = "Car reset in MuJoCo"
        self.get_logger().info("Car reset requested and applied.")
        return response

    def timer_cb(self) -> None:
        # Apply action
        self.data.ctrl[:] = self.last_action

        # Step MuJoCo
        for _ in range(self.steps_per_ctrl):
            mj.mj_step(self.model, self.data)

        # Update viewer if window still open
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()   # redraw using current model/data state

        # Orientation [qw, qx, qy, qz]
        qw, qx, qy, qz = self.data.qpos[self.qadr : self.qadr + 4]

        gyro = self.data.sensordata[self.gyro_adr : self.gyro_adr + self.gyro_dim]
        acc  = self.data.sensordata[self.acc_adr  : self.acc_adr  + self.acc_dim]

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.orientation.x = float(qx)
        msg.orientation.y = float(qy)
        msg.orientation.z = float(qz)
        msg.orientation.w = float(qw)

        if self.gyro_dim >= 3:
            msg.angular_velocity.x = float(gyro[0])
            msg.angular_velocity.y = float(gyro[1])
            msg.angular_velocity.z = float(gyro[2])

        if self.acc_dim >= 3:
            msg.linear_acceleration.x = float(acc[0])
            msg.linear_acceleration.y = float(acc[1])
            msg.linear_acceleration.z = float(acc[2])

        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        self.pub_imu.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.viewer is not None:
            node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
