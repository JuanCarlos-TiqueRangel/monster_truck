#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Int32
import numpy as np

import time
from rclpy.signals import SignalHandlerOptions

class PitchPidController(Node):
    def __init__(self):
        super().__init__('pitch_pid_controller')
        
        # ============================================================
        # PID TUNING PARAMETERS
        # ============================================================
        self.Kp = 3.5
        self.Ki = 2.0
        self.Kd = 0.2
        
        self.target_pitch = -63.0
        
        # RC ESC Hardware Constants
        self.PWM_NEUTRAL = 1500
        self.PWM_MIN = 1000
        self.PWM_MAX = 2000
        
        # Fixed Loop Timing Constants (100Hz -> 0.01s)
        self.LOOP_PERIOD = 0.01 
        
        # Working State Variables
        self.current_pitch = 0.0
        self.integral = 0.0
        self.last_error = 0.0
        self.first_run = True
        
        # ROS 2 Pub/Sub
        self.euler_sub = self.create_subscription(Vector3Stamped, 'imu/euler', self.euler_callback, 10)
        self.pwm_pub = self.create_publisher(Int32, 'cmd_pwm', 10)
        
        # Dedicated ROS 2 Wall Timer to enforce a strict 100Hz loop rate
        self.timer = self.create_timer(self.LOOP_PERIOD, self.control_loop_callback)
        
        self.get_logger().info("Pitch PID Controller active at a strict 100Hz loop rate.")

    def euler_callback(self, msg):
        """Asynchronous callback: Simply catches data and saves it instantly."""
        self.current_pitch = msg.vector.y

    def control_loop_callback(self):
        """Synchronous 100Hz execution loop: Runs independent of serial arrival jitter."""
        # Use a fixed, stable dt representing our 100Hz loop interval
        dt = self.LOOP_PERIOD
        
        # Calculate tracking error
        error = self.current_pitch - self.target_pitch
        
        if self.first_run:
            self.last_error = error
            self.first_run = False
            return

        # 1. Accumulate raw error over time
        self.integral += error * dt
        
        # 2. Rate of change of error
        derivative = (error - self.last_error) / dt
        
        # 3. Calculate Total PID Output (Matching C++ layout)
        # pid_output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        pid_output = (self.Kp * error) + (self.Kd * derivative)
        
        # 4. Output Saturation
        pid_output = max(min(pid_output, 500.0), -500.0)
        
        # 5. Add Output to ESC center point & apply physical limits
        target_pwm = int(self.PWM_NEUTRAL + pid_output)
        target_pwm = max(min(target_pwm, self.PWM_MAX), self.PWM_MIN)
        
        # Fire down the command
        pwm_msg = Int32()
        pwm_msg.data = target_pwm
        self.pwm_pub.publish(pwm_msg)
        
        # Save states for next cycle
        self.last_error = error
        
        # DO NOT use print() inside a 100Hz loop. Use throttle loggers instead:
        self.get_logger().info(f"Error: {error:.2f} | Out: {target_pwm}", throttle_duration_sec=0.05)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = PitchPidController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.timer.cancel()

        pwm_msg = Int32()
        pwm_msg.data = 1500

        for _ in range(20):
            node.pwm_pub.publish(pwm_msg)
            time.sleep(0.02)

        node.get_logger().info("STOP: PWM = 1500")

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
