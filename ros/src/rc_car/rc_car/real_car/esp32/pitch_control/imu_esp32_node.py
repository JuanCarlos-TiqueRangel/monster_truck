#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3Stamped  # Used for Roll, Pitch, Yaw
from std_msgs.msg import Int32




class ImuSerialNode(Node):
    def __init__(self):
        super().__init__('imu_serial_node')
        
        # Declare parameters for configuration
        self.declare_parameter('port', '/dev/ttyACM0') 
        self.declare_parameter('baudrate', 115200)     
        self.declare_parameter('frame_id', 'imu_link')
        
        port = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        self.frame_id = self.get_parameter('frame_id').value
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1.0)
            self.get_logger().info(f"Opened port: {port} at {baudrate} baud.")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to open port {port}: {e}")
            raise e
            
        # Publishers
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.euler_pub = self.create_publisher(Vector3Stamped, 'imu/euler', 10)

        # Subscriber (In from network -> out to serial)
        self.pwm_sub = self.create_subscription(Int32, 'cmd_pwm', self.pwm_callback, 10)


        # Poll at 100Hz
        timer = 1/250
        self.timer = self.create_timer(timer, self.serial_read_callback) 

    def pwm_callback(self, msg):
        """Intercepts the calculation from your PID node and pipes it down the wire."""
        try:
            command_string = f"{msg.data}\n"
            self.ser.write(command_string.encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Failed to relay command to ESP32: {e}")
        


    def serial_read_callback(self):
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                data = [float(val) for val in line.split(',')]
                
                # Expecting 13 values: ax,ay,az, gx,gy,gz, qw,qx,qy,qz, roll,pitch,yaw
                if len(data) == 13:
                    current_time = self.get_clock().now().to_msg()
                    self.publish_imu(data, current_time)
                    self.publish_euler(data, current_time)
                    
            except ValueError:
                pass 
            except Exception as e:
                self.get_logger().error(f"Serial error: {e}")

                

    def publish_imu(self, data, current_time):
        msg = Imu()
        msg.header.stamp = current_time
        msg.header.frame_id = self.frame_id
        
        # Linear Acceleration
        msg.linear_acceleration.x = data[0]
        msg.linear_acceleration.y = data[1]
        msg.linear_acceleration.z = data[2]
        
        # Angular Velocity
        msg.angular_velocity.x = data[3]
        msg.angular_velocity.y = data[4]
        msg.angular_velocity.z = data[5]
        
        # Orientation Quaternion
        msg.orientation.w = data[6]
        msg.orientation.x = data[7]
        msg.orientation.y = data[8]
        msg.orientation.z = data[9]
        
        # Set variance defaults
        msg.orientation_covariance = [0.01] + [0.0]*8
        msg.angular_velocity_covariance = [0.01] + [0.0]*8
        msg.linear_acceleration_covariance = [0.01] + [0.0]*8
        
        self.imu_pub.publish(msg)

    def publish_euler(self, data, current_time):
        msg = Vector3Stamped()
        msg.header.stamp = current_time
        msg.header.frame_id = self.frame_id
        
        # Map values to x=roll, y=pitch, z=yaw (values are in radians)
        msg.vector.x = data[10] # Roll
        msg.vector.y = data[11] # Pitch
        msg.vector.z = data[12] # Yaw
        
        self.euler_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuSerialNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
