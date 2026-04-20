#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import math
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

# File configuration
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

from utils.geometry import quat_to_euler_xyz

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # Fixed dataset sampling / random-command update period
    sample_dt: float = 0.1          # [s] 50 Hz logging + random input update

    # Re-publish held command at this rate
    cmd_pub_hz: float = 50.0         # [Hz]

    # Random input range
    u_min: float = -1.0
    u_max: float = 1.0

    # Run duration in wall-clock seconds
    duration: Optional[float] = 10.0

    # Random seed for reproducibility (None => different every run)
    random_seed: Optional[int] = None

    # Plot options
    online_plot: bool = False
    refresh_hz: float = 10.0
    show_final_plot: bool = True

    # Save path
    file_name = cfg_params.files.ini_data_file
    save_path: str = f"data/{file_name}"


# ============================================================
# Plot helpers
# ============================================================

def setup_figure(cfg: Config):
    plt.ion()
    #     x, xdot, pitch, pitchdot, u  
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(18, 10), sharex=True, constrained_layout=True
    )

    line_width = 2.5

    (line_xpos,) = ax1.plot([], [], lw=line_width, label="xpos")
    (line_xpos_dot,) = ax2.plot([], [], lw=line_width, label="xpos_dot")
    (line_pitch,) = ax3.plot([], [], lw=line_width, label="pitch")
    (line_pitch_dot,) = ax4.plot([], [], lw=line_width, label="pitch_dot")
    (line_u,) = ax5.plot([], [], lw=line_width, label="u")

    ax1.set_ylabel("xpos")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_ylabel("xpos_dot")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    ax3.set_ylabel("pitch")
    ax3.grid(True)
    ax3.legend(loc="upper right")

    ax4.set_ylabel("pitch_dot")
    ax4.grid(True)
    ax4.legend(loc="upper right")

    ax5.set_ylabel("u")
    ax5.set_xlabel("time [s]")
    ax5.set_ylim(cfg.u_min - 0.1, cfg.u_max + 0.1)
    ax5.grid(True)
    ax5.legend(loc="upper right")

    fig.suptitle(f"Random Input Data Collection (dt = {cfg.sample_dt:.3f} s)")
    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, (ax1, ax2, ax3, ax4, ax5), line_xpos, line_xpos_dot, line_pitch, line_pitch_dot, line_u


def update_plot(node, fig, axes, line_xpos, line_xpos_dot, line_pitch, line_pitch_dot, line_u):
    ax1, ax2, ax3, ax4, ax5 = axes

    if not node.t_wall_log:
        return

    t = np.asarray(node.t_wall_log, dtype=float)
    xpos = np.asarray(node.xpos_log, dtype=float)
    xpos_dot = np.asarray(node.xpos_dot_log, dtype=float)
    pitch = np.asarray(node.pitch_log, dtype=float)
    pitch_dot = np.asarray(node.pitch_dot_log, dtype=float)
    u = np.asarray(node.u_log, dtype=float)

    line_xpos.set_data(t, xpos)
    line_xpos_dot.set_data(t, xpos_dot)
    line_pitch.set_data(t, pitch)
    line_pitch_dot.set_data(t, pitch_dot)
    line_u.set_data(t, u)

    ax1.set_xlim(0.0, max(2.0, t[-1]))

    # autoscale y for signals
    if len(xpos) > 1:
        ymin, ymax = np.min(xpos), np.max(xpos)
        pad = max(0.05, 0.1 * max(1e-6, ymax - ymin))
        ax1.set_ylim(ymin - pad, ymax + pad)

    if len(xpos_dot) > 1:
        ymin, ymax = np.min(xpos_dot), np.max(xpos_dot)
        pad = max(0.05, 0.1 * max(1e-6, ymax - ymin))
        ax2.set_ylim(ymin - pad, ymax + pad)

    fig.canvas.draw()
    fig.canvas.flush_events()


# ============================================================
# Random driver + fixed-dt logger node
# ============================================================

class MujocoRandomCmdLogger(Node):
    def __init__(self, cfg: Config):
        super().__init__("mujoco_random_cmd_logger")
        self.cfg = cfg

        # Create save directory if needed
        save_dir = os.path.dirname(cfg.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # RNG
        self.rng = np.random.default_rng(cfg.random_seed)

        # Wall clock for timers
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)

        # Publisher
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)

        # Subscribers
        self.imu_sub = self.create_subscription(Imu, "car_imu", self.imu_cb, 50)
        self.odom_sub = self.create_subscription(Odometry, "car_odom", self.odom_cb, 10)

        # Run state
        self.stop_requested = False
        self.start_wall = time.perf_counter()

        # Command state
        self.current_cmd = 0.0
        self.last_u = 0.0

        # IMU decoded latest
        self.have_imu = False
        self.latest_t_sim = 0.0
        self.t0_sim: Optional[float] = None

        # Latest decoded signals
        self.xpos = 0.0
        self.xpos_dot = 0.0

        # Imu data
        self.roll = 0.0
        self.roll_dot = 0.0
        self.pitch = 0.0
        self.pitch_dot = 0.0
        self.yaw = 0.0
        self.yaw_dot = 0.0

        # Fixed-rate logs
        self.t_wall_log = []
        self.t_sim_log = []
        self.u_log = []
        self.xpos_log = []
        self.xpos_dot_log = []
        self.pitch_log = []
        self.pitch_dot_log = []

        # Timers
        # cmd_timer: re-publish currently held random command
        # sample_timer: draw new random command + log one sample
        self.cmd_timer = self.create_timer(
            1.0 / self.cfg.cmd_pub_hz, self.cmd_timer_cb, clock=self.wall_clock
        )
        self.sample_timer = self.create_timer(
            self.cfg.sample_dt, self.sample_timer_cb, clock=self.wall_clock
        )

        self.get_logger().info(
            "Random-input control + logging started.\n"
            f"Random command held for sample_dt = {self.cfg.sample_dt:.3f} s\n"
            f"u ~ Uniform[{self.cfg.u_min:.3f}, {self.cfg.u_max:.3f}]\n"
            f"Publishing cmd_action at {self.cfg.cmd_pub_hz:.1f} Hz (wall)\n"
            f"Logging at dt = {self.cfg.sample_dt:.3f} s (wall)\n"
            f"Saving to: {self.cfg.save_path}"
        )

    def request_stop(self, reason: str):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.get_logger().info(f"Stopping: {reason}")

        try:
            self.cmd_timer.cancel()
        except Exception:
            pass

        try:
            self.sample_timer.cancel()
        except Exception:
            pass

        self.publish_cmd(0.0)

    def publish_cmd(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)

    # ------------------ ROS callbacks ------------------

    def imu_cb(self, msg: Imu):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        if self.t0_sim is None:
            self.t0_sim = t
        self.latest_t_sim = float(t - self.t0_sim)

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
        self.yaw_dot) = quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        self.have_imu = True


    def odom_cb(self, msg: Odometry):
        self.xpos = float(msg.pose.pose.position.x)
        self.xpos_dot = float(msg.twist.twist.linear.x)

    # ------------------ timers ------------------

    def cmd_timer_cb(self):
        if self.stop_requested:
            return

        # Re-publish the currently held command
        self.publish_cmd(self.current_cmd)

    def sample_timer_cb(self):
        if self.stop_requested:
            return

        # optional auto-stop
        if self.cfg.duration is not None:
            if (time.perf_counter() - self.start_wall) >= self.cfg.duration:
                self.request_stop("duration reached")
                return

        # Wait until at least one IMU sample is available
        if not self.have_imu:
            return

        # 1) New random command, held until next sample tick
        self.current_cmd = float(self.rng.uniform(self.cfg.u_min, self.cfg.u_max))
        self.publish_cmd(self.current_cmd)

        # 2) Log exactly one sample at fixed dt
        t_wall = time.perf_counter() - self.start_wall

        self.t_wall_log.append(float(t_wall))
        self.t_sim_log.append(float(self.latest_t_sim))
        self.u_log.append(float(self.last_u))
        self.xpos_log.append(float(self.xpos))
        self.xpos_dot_log.append(float(self.xpos_dot))
        self.pitch_log.append(float(self.pitch))
        self.pitch_dot_log.append(float(self.pitch_dot))

    def save_npz(self):
        if not self.t_wall_log:
            self.get_logger().warn("No data collected, skipping NPZ save.")
            return

        np.savez(
            self.cfg.save_path,
            dt_wall=np.float32(self.cfg.sample_dt),
            t_wall=np.asarray(self.t_wall_log, dtype=np.float32),
            t_sim=np.asarray(self.t_sim_log, dtype=np.float32),
            u=np.asarray(self.u_log, dtype=np.float32),
            xpos=np.asarray(self.xpos_log, dtype=np.float32),
            xpos_dot=np.asarray(self.xpos_dot_log, dtype=np.float32),
            pitch=np.asarray(self.pitch_log, dtype=np.float32),
            pitch_dot=np.asarray(self.pitch_dot_log, dtype=np.float32),
        )

        self.get_logger().info(
            f"Saved data to NPZ: {self.cfg.save_path} (N={len(self.t_wall_log)})"
        )
        print(f"Done. Samples: {len(self.xpos_log)}  wall time: {self.t_wall_log[-1]:.3f}s")


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()

    rclpy.init()
    node = MujocoRandomCmdLogger(cfg)

    fig, axes, line_xpos, line_xpos_dot, line_pitch, line_pitch_dot, line_u = setup_figure(cfg)
    last_refresh_wall = time.perf_counter()

    try:
        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.01)

            now = time.perf_counter()
            if cfg.online_plot and (now - last_refresh_wall >= 1.0 / cfg.refresh_hz):
                update_plot(node, fig, axes, line_xpos, line_xpos_dot, line_pitch, line_pitch_dot, line_u)
                last_refresh_wall = now
                plt.pause(0.001)

    except KeyboardInterrupt:
        node.request_stop("KeyboardInterrupt (Ctrl+C)")

    finally:
        # Send one final zero command
        try:
            node.publish_cmd(0.0)
            time.sleep(0.02)
        except Exception:
            pass

        # Final plot refresh
        try:
            update_plot(node, fig, axes, line_xpos, line_xpos_dot, line_pitch, line_pitch_dot, line_u)
        except Exception:
            pass

        # Save
        node.save_npz()

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        if cfg.show_final_plot:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()