#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from SVGP.utils import geometry


class CarImuSubscriber(Node):
    def __init__(self):
        super().__init__('car_imu_subscriber')

        self.imu_data = None

        self.subscription = self.create_subscription(Imu, '/car_imu', self.imu_callback, qos_profile_sensor_data)

    def imu_callback(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        # fixed bug: use x, y, z correctly
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)

        (self.roll, 
        self.pitch, 
        self.yaw, 
        self.roll_dot, 
        self.pitch_dot, 
        self.yaw_dot) = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        print("Pitch: ", self.pitch)
        print("Roll: ", self.roll)
        print("Yaw: ", self.yaw)
        print(" ")

def main(args=None):
    rclpy.init(args=args)

    node = CarImuSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()