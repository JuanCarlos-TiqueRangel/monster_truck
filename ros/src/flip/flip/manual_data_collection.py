#!/usr/bin/env python3
import sys
import os
import select
import tty
import termios
import atexit
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

from utils.geometry import quat_to_R, up_and_updot_from_quat_gyro


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # Logging sampling (fixed wall dt)
    sample_dt: float = 0.1          # [s] logging at 10 Hz (wall-clock)

    # Command publish rate (responsive manual control)
    cmd_pub_hz: float = 50.0        # [Hz] publish /cmd_action at 50 Hz (wall-clock)

    # Manual command magnitude
    amplitude: float = 1.0          # W => +amplitude, S => -amplitude

    # If None: run until Q. If set: auto-stop after duration seconds (wall-clock)
    duration: Optional[float] = None

    # Save path
    save_path: str = "data/mujoco_manual_run_flip.npz"

    # Helps “hold” behavior because OS key-repeat can be slow
    key_hold_sec: float = 0.10


# ============================================================
# Terminal key reader with guaranteed restore
# ============================================================

class TerminalKeyReader:
    """
    Puts TTY into cbreak mode and provides non-blocking single-char reads.
    Always restores terminal settings on exit.
    """
    def __init__(self):
        self._old = None
        self._fd = None
        self._file = None
        self._owns_file = False
        self.enabled = False
        self._restored = False

        # Prefer stdin if it's a TTY; otherwise try /dev/tty
        if sys.stdin is not None and sys.stdin.isatty():
            self._file = sys.stdin
            self._fd = sys.stdin.fileno()
        else:
            try:
                self._file = open("/dev/tty", "r")
                self._fd = self._file.fileno()
                self._owns_file = True
            except Exception:
                self._file = None
                self._fd = None

        if self._fd is not None:
            try:
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.enabled = True
                atexit.register(self.restore)
            except Exception:
                self.enabled = False

    def restore(self):
        if self._restored:
            return
        self._restored = True
        try:
            if self.enabled and self._old is not None and self._fd is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass
        try:
            if self._owns_file and self._file is not None:
                self._file.close()
        except Exception:
            pass

    def read_key_nonblocking(self):
        if not self.enabled or self._file is None:
            return None
        try:
            dr, _, _ = select.select([self._file], [], [], 0.0)
            if dr:
                return self._file.read(1)
        except Exception:
            return None
        return None


# ============================================================
# Manual driver + fixed-dt logger node
# ============================================================

class MujocoKeyboardCmdLogger(Node):
    def __init__(self, cfg: Config):
        super().__init__("mujoco_keyboard_cmd_logger")
        self.cfg = cfg

        # Create save directory if needed
        save_dir = os.path.dirname(cfg.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Terminal
        self.keys = TerminalKeyReader()
        if not self.keys.enabled:
            self.get_logger().warn("Keyboard input not attached to a TTY. Controls may not work.")

        # Use a STEADY (wall) clock for timers so rates don't depend on /clock or use_sim_time
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
        self.last_w_time: Optional[float] = None
        self.last_s_time: Optional[float] = None

        # IMU decoded latest
        self.have_imu = False
        self.latest_t_sim = 0.0     # sim time from IMU header (relative)

        # IMU time reference (sim time)
        self.t0_sim: Optional[float] = None

        # Logs (fixed wall dt)
        self.t_wall_log = []     # wall time since start
        self.t_sim_log = []      # sim time since first imu
        self.u_log = []
        self.up_z_log = []
        self.up_z_dot_log = []

        # Timers (IMPORTANT: pass clock=self.wall_clock)
        self.cmd_timer = self.create_timer(1.0 / self.cfg.cmd_pub_hz, self.cmd_timer_cb, clock=self.wall_clock)
        self.log_timer = self.create_timer(self.cfg.sample_dt, self.log_timer_cb, clock=self.wall_clock)

        self.get_logger().info(
            "Manual control + logging started.\n"
            "Controls: Hold W=forward | Hold S=backward | Space=stop | Q=quit\n"
            f"Publishing cmd_action at {self.cfg.cmd_pub_hz:.1f} Hz (wall)\n"
            f"Logging at dt={self.cfg.sample_dt:.3f} s (wall)\n"
            f"Saving to: {self.cfg.save_path}\n"
            "(Click this terminal window to focus key input)"
        )

    def destroy_node(self):
        try:
            self.keys.restore()
        except Exception:
            pass
        super().destroy_node()

    def request_stop(self, reason: str):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.get_logger().info(f"Stopping: {reason}")

        # restore terminal immediately
        self.keys.restore()

        try:
            self.cmd_timer.cancel()
        except Exception:
            pass
        try:
            self.log_timer.cancel()
        except Exception:
            pass

        self.publish_cmd(0.0)

        # End spin cleanly
        if rclpy.ok():
            rclpy.shutdown()

    def publish_cmd(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)

    # ------------------ ROS callbacks ------------------

    def imu_cb(self, msg: Imu):
        # sim time from message header
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0_sim is None:
            self.t0_sim = t
        t_rel = t - self.t0_sim

        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)
        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.x)
        wz = float(msg.angular_velocity.x)

        self.up_z, self.up_z_dot, _   = up_and_updot_from_quat_gyro(qw, qx, qy, qz, wx, wy, wz)
        self.have_imu = True

    def odom_cb(self, msg: Odometry):
        self.x_pos = float(msg.pose.pose.position.x)
        self.line_speed_x = float(msg.twist.twist.linear.x)

    # ------------------ timers ------------------

    def cmd_timer_cb(self):
        if self.stop_requested:
            return

        # optional auto-stop (wall)
        if self.cfg.duration is not None:
            if (time.perf_counter() - self.start_wall) >= self.cfg.duration:
                self.request_stop("duration reached")
                return

        now = time.perf_counter()

        # Read all pending keys
        key = self.keys.read_key_nonblocking()
        got_space = False
        got_q = False
        while key is not None:
            k = key.lower()
            if k == "w":
                self.last_w_time = now
            elif k == "s":
                self.last_s_time = now
            elif k == " ":
                got_space = True
            elif k == "q":
                got_q = True
            key = self.keys.read_key_nonblocking()

        if got_q:
            self.request_stop("Q pressed")
            return

        # Space has priority
        if got_space:
            self.current_cmd = 0.0
        else:
            w_active = (self.last_w_time is not None) and ((now - self.last_w_time) <= self.cfg.key_hold_sec)
            s_active = (self.last_s_time is not None) and ((now - self.last_s_time) <= self.cfg.key_hold_sec)

            if w_active and not s_active:
                self.current_cmd = +self.cfg.amplitude
            elif s_active and not w_active:
                self.current_cmd = -self.cfg.amplitude
            else:
                self.current_cmd = 0.0

        self.publish_cmd(self.current_cmd)

    def log_timer_cb(self):
        if self.stop_requested:
            return
        if not self.have_imu:
            return

        # fixed wall-time log tick
        t_wall = time.perf_counter() - self.start_wall

        self.t_wall_log.append(float(t_wall))
        self.t_sim_log.append(float(self.latest_t_sim))
        self.u_log.append(self.last_u)
        self.up_z_log.append(self.up_z)
        self.up_z_dot_log.append(self.up_z_dot)

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
            up_z=np.asarray(self.up_z_log, dtype=np.float32),
            up_z_dot=np.asarray(self.up_z_dot_log, dtype=np.float32),
        )
        self.get_logger().info(f"Saved data to NPZ: {self.cfg.save_path} (N={len(self.t_wall_log)})")
        print(f"Done. Samples: {len(self.t_wall_log)}  wall time: {self.t_wall_log[-1]:.3f}s")


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()

    rclpy.init()
    node = MujocoKeyboardCmdLogger(cfg)

    try:
        # Spin until node requests stop (Q) -> it calls rclpy.shutdown()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.request_stop("KeyboardInterrupt (Ctrl+C)")
    finally:
        # make sure terminal is restored even if something weird happens
        try:
            node.keys.restore()
        except Exception:
            pass

        # stop cmd once more + save
        try:
            node.publish_cmd(0.0)
            time.sleep(0.02)
        except Exception:
            pass

        node.save_npz()

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
