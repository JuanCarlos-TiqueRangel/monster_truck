#!/usr/bin/env python3

import matplotlib
matplotlib.rcParams['figure.raise_window'] = False

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class AccelerationPlotNode(Node):

    def __init__(self):
        super().__init__('acceleration_plot_node')

        # ------------------------------------------------------
        # SETTINGS
        # ------------------------------------------------------

        self.odom_topic = '/optitrack/odom_810'

        # Time window used to estimate acceleration
        self.regression_window = 0.20  # seconds

        # Plot last N seconds
        self.plot_window = 10.0

        # Theoretical wheelie threshold
        self.wheelie_threshold = 11.12  # m/s^2

        # ------------------------------------------------------
        # DATA
        # ------------------------------------------------------

        self.start_time = None

        # Data used for the regression
        self.velocity_times = deque()
        self.velocity_values = deque()

        # Data used for plotting
        self.time_data = deque(maxlen=3000)
        self.accel_data = deque(maxlen=3000)

        # ------------------------------------------------------
        # ROS
        # ------------------------------------------------------

        self.subscription = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        # ------------------------------------------------------
        # PLOT
        # ------------------------------------------------------

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(10, 5))

        self.accel_line, = self.ax.plot(
            [],
            [],
            linewidth=2,
            label='Estimated acceleration'
        )

        self.ax.axhline(
            self.wheelie_threshold,
            linestyle='--',
            linewidth=2,
            label='Wheelie threshold = 11.12 m/s²'
        )

        self.ax.set_xlabel('Time [s]')
        self.ax.set_ylabel(r'$a_x$ [m/s²]')
        self.ax.set_title('RC Car Longitudinal Acceleration')

        self.ax.set_xlim(0.0, self.plot_window)
        self.ax.set_ylim(-5.0, 20.0)

        self.ax.grid(True)
        self.ax.legend()

        self.fig.tight_layout()

        plt.show(block=False)


    def odom_callback(self, msg):

        vx = msg.twist.twist.linear.x
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.start_time is None:
            self.start_time = t

        elapsed_time = t - self.start_time

        # ------------------------------------------------------
        # SAVE VELOCITY SAMPLE
        # ------------------------------------------------------

        self.velocity_times.append(t)
        self.velocity_values.append(vx)

        # Remove samples older than regression window
        while self.velocity_times and (t - self.velocity_times[0]) > self.regression_window:
            self.velocity_times.popleft()
            self.velocity_values.popleft()

        # Need enough points for a reliable linear fit
        if len(self.velocity_times) < 5:
            return

        # ------------------------------------------------------
        # LINEAR REGRESSION
        #
        # vx(t) = ax*t + b
        #
        # slope = acceleration
        # ------------------------------------------------------

        times = np.asarray(self.velocity_times)
        velocities = np.asarray(self.velocity_values)

        # Shift time close to zero for numerical stability
        times = times - times[0]

        ax_estimated, _ = np.polyfit(times, velocities, 1)

        # ------------------------------------------------------
        # SAVE FOR PLOT
        # ------------------------------------------------------

        self.time_data.append(elapsed_time)
        self.accel_data.append(ax_estimated)


    def update_plot(self):

        if len(self.time_data) < 2:
            return

        time_data = np.asarray(self.time_data)
        accel_data = np.asarray(self.accel_data)

        self.accel_line.set_data(time_data, accel_data)

        current_time = time_data[-1]

        # ------------------------------------------------------
        # X AXIS
        # ------------------------------------------------------

        if current_time < self.plot_window:
            self.ax.set_xlim(0.0, self.plot_window)
        else:
            self.ax.set_xlim(current_time - self.plot_window, current_time)

        # ------------------------------------------------------
        # Y AXIS
        # ------------------------------------------------------

        a_min = np.min(accel_data)
        a_max = np.max(accel_data)

        y_min = min(-5.0, a_min - 2.0)
        y_max = max(20.0, a_max + 2.0)

        self.ax.set_ylim(y_min, y_max)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None):

    rclpy.init(args=args)

    node = AccelerationPlotNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            node.update_plot()
            plt.pause(0.001)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

    plt.ioff()
    plt.close('all')


if __name__ == '__main__':
    main()