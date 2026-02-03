#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import mujoco as mj
import mujoco.viewer as mjviewer

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

# ============================================================
# SIMPLE EXPERIMENT FLAG (change this line only)
# ============================================================
START_FLIPPED = False   # True: upside-down at start/reset | False: upright


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

        # Timing
        self.sim_dt = float(self.model.opt.timestep)
        self.ctrl_dt = 0.01  # 100 Hz publish/step loop
        self.steps_per_ctrl = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.get_logger().info(
            f"Loaded {xml_path}, sim_dt={self.sim_dt:.6f}, "
            f"ctrl_dt={self.ctrl_dt:.6f}, steps_per_ctrl={self.steps_per_ctrl}"
        )

        # ---------------- Find FREE joint addresses ----------------
        free_j = None
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
                free_j = j
                break
        if free_j is None:
            raise RuntimeError("No free joint found in model (need <freejoint/>)")

        # qpos layout for free joint: [x y z qw qx qy qz]
        self.free_qpos_adr = int(self.model.jnt_qposadr[free_j])
        self.qadr = self.free_qpos_adr + 3  # start of qw,qx,qy,qz

        # qvel layout for free joint: [vx vy vz wx wy wz]
        self.free_qvel_adr = int(self.model.jnt_dofadr[free_j])

        # ---------------- IMU sensors ----------------
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

        # ---------------- Obstacle (pole_1) geom id ----------------
        self.pole_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "obs_box_1")
        if self.pole_geom_id < 0:
            raise RuntimeError("Geom 'obs_box_1' not found in XML (check name='obs_box_1')")


        # ---------------- Reset once + apply START_FLIPPED ----------------
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._apply_start_pose_flag()

        # Save initial state used by reset service
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()

        # ---------------- Viewer (optional) ----------------
        self.viewer = None
        try:
            self.viewer = mjviewer.launch_passive(self.model, self.data)
        except Exception as e:
            self.get_logger().warning(f"Viewer not started (headless?): {e}")

        # ---------------- ROS interfaces ----------------
        self.last_action = 0.0
        self.sub_cmd = self.create_subscription(Float32, "cmd_action", self.cmd_action_cb, 10)

        self.pub_imu = self.create_publisher(Imu, "car_imu", 10)
        self.pub_odom = self.create_publisher(Odometry, "car_odom", 10)
        self.pub_obs_pose = self.create_publisher(PoseStamped, "obstacle_pose", 10)

        self.reset_srv = self.create_service(Trigger, "reset_car", self.reset_callback)
        self.timer = self.create_timer(self.ctrl_dt, self.timer_cb)

        self.get_logger().info(
            f"MujocoImuNode ready. START_FLIPPED={START_FLIPPED}. "
            "Subscribing /cmd_action, publishing /car_imu, /car_odom, /obstacle_pose."
        )

    # ------------------------------------------------------------
    # Helper: apply upright/flipped pose based on START_FLIPPED
    # ------------------------------------------------------------
    def _apply_start_pose_flag(self):
        if START_FLIPPED:
            quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=self.data.qpos.dtype)  # 180deg about X
            z0 = 0.24
        else:
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=self.data.qpos.dtype)  # upright
            z0 = 0.14

        # Set base pose (freejoint qpos: x,y,z,qw,qx,qy,qz)
        self.data.qpos[self.free_qpos_adr + 0] = 0.0
        self.data.qpos[self.free_qpos_adr + 1] = 0.0
        self.data.qpos[self.free_qpos_adr + 2] = z0
        self.data.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7] = quat

        # zero base velocity
        self.data.qvel[self.free_qvel_adr : self.free_qvel_adr + 6] = 0.0

        mj.mj_forward(self.model, self.data)

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------
    def cmd_action_cb(self, msg: Float32) -> None:
        self.last_action = float(msg.data)

    def reset_callback(self, request, response):
        mj.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel

        self._apply_start_pose_flag()
        self.data.ctrl[:] = 0.0

        response.success = True
        response.message = f"Car reset in MuJoCo (START_FLIPPED={START_FLIPPED})"
        self.get_logger().info(response.message)
        return response


    def _publish_pole_pose(self, stamp):
        # World position of the geom
        p = self.data.geom_xpos[self.pole_geom_id]  # (3,)

        # World rotation matrix of the geom (flattened 9 values)
        xmat = self.data.geom_xmat[self.pole_geom_id]  # (9,)

        # Convert rotation matrix -> quaternion [w, x, y, z]
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



    def timer_cb(self) -> None:
        # Apply action
        self.data.ctrl[:] = self.last_action

        # Step MuJoCo
        for _ in range(self.steps_per_ctrl):
            mj.mj_step(self.model, self.data)

        # Update viewer
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        stamp = self.get_clock().now().to_msg()

        # ---------------- Publish obstacle pose ----------------
        self._publish_pole_pose(stamp)

        # ---------------- Publish IMU ----------------
        qw, qx, qy, qz = self.data.qpos[self.qadr : self.qadr + 4]
        gyro = self.data.sensordata[self.gyro_adr : self.gyro_adr + self.gyro_dim]
        acc = self.data.sensordata[self.acc_adr : self.acc_adr + self.acc_dim]

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

        # ---------------- Get base pose + twist from FREE joint ----------------
        x, y, z = self.data.qpos[self.free_qpos_adr : self.free_qpos_adr + 3]
        qw, qx, qy, qz = self.data.qpos[self.free_qpos_adr + 3 : self.free_qpos_adr + 7]

        vx, vy, vz, wx, wy, wz = self.data.qvel[self.free_qvel_adr : self.free_qvel_adr + 6]

        # ---------------- Publish Odometry ----------------
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

        # Covariances unknown
        odom.pose.covariance[0] = -1.0
        odom.twist.covariance[0] = -1.0

        self.pub_odom.publish(odom)


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
