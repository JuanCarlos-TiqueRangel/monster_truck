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
    save_path: str = "data/mujoco_manual_run.npz"

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
        self.latest_pitch = 0.0
        self.latest_flip_rel = 0.0
        self.latest_rate = 0.0
        self.latest_acc = 0.0
        self.latest_up_z = 0.0
        self.latest_up_x = 0.0

        # Odom latest
        self.x_pos = 0.0
        self.line_speed_x = 0.0

        # IMU time reference (sim time)
        self.t0_sim: Optional[float] = None

        # Angle unwrapping
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped = 0.0
        self.theta0: Optional[float] = None

        # Logs (fixed wall dt)
        self.t_wall_log = []     # wall time since start
        self.t_sim_log = []      # sim time since first imu
        self.pitch_log = []
        self.flip_rel_log = []
        self.u_log = []
        self.rate_log = []
        self.acc_log = []
        self.vz_log = []
        self.vx_log = []
        self.x_log = []
        self.linear_x_log = []

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

    # ------------------ math helpers ------------------

    @staticmethod
    def quat_to_R_and_pitch(qw, qx, qy, qz):
        R00 = 1 - 2*(qy*qy + qz*qz)
        R01 = 2*(qx*qy - qw*qz)
        R02 = 2*(qx*qz + qw*qy)

        R10 = 2*(qx*qy + qw*qz)
        R11 = 1 - 2*(qx*qx + qz*qz)
        R12 = 2*(qy*qz - qw*qx)

        R20 = 2*(qx*qz - qw*qy)
        R21 = 2*(qy*qz + qw*qx)
        R22 = 1 - 2*(qx*qx + qy*qy)

        pitch = -math.asin(max(-1.0, min(1.0, R20)))

        R = np.array([[R00, R01, R02],
                      [R10, R11, R12],
                      [R20, R21, R22]], dtype=float)
        return R, pitch

    @staticmethod
    def unwrap_angle(prev_angle, prev_unwrapped, angle):
        if prev_angle is None:
            return angle, angle
        d = angle - prev_angle
        if d > math.pi:
            angle_unwrapped = prev_unwrapped + (d - 2*math.pi)
        elif d < -math.pi:
            angle_unwrapped = prev_unwrapped + (d + 2*math.pi)
        else:
            angle_unwrapped = prev_unwrapped + d
        return angle, angle_unwrapped

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

        R, euler_pitch = self.quat_to_R_and_pitch(qw, qx, qy, qz)

        up_x, up_y, up_z = R[0, 2], R[1, 2], R[2, 2]
        theta = math.atan2(up_x, up_z)

        self.prev_theta, theta_unwrapped = self.unwrap_angle(
            self.prev_theta, self.prev_theta_unwrapped, theta
        )
        self.prev_theta_unwrapped = theta_unwrapped

        if self.theta0 is None:
            self.theta0 = theta_unwrapped
        flip_rel = theta_unwrapped - self.theta0

        pitch_rate = float(msg.angular_velocity.y)
        acc_imu = float(msg.linear_acceleration.x)

        self.latest_t_sim = float(t_rel)
        self.latest_pitch = float(euler_pitch)
        self.latest_flip_rel = float(flip_rel)
        self.latest_rate = float(pitch_rate)
        self.latest_acc = float(acc_imu)
        self.latest_up_z = float(up_z)
        self.latest_up_x = float(up_x)
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
        self.pitch_log.append(self.latest_pitch)
        self.flip_rel_log.append(self.latest_flip_rel)
        self.u_log.append(self.last_u)
        self.rate_log.append(self.latest_rate)
        self.acc_log.append(self.latest_acc)
        self.vz_log.append(self.latest_up_z)
        self.vx_log.append(self.latest_up_x)
        self.x_log.append(self.x_pos)
        self.linear_x_log.append(self.line_speed_x)

    def save_npz(self):
        if not self.t_wall_log:
            self.get_logger().warn("No data collected, skipping NPZ save.")
            return

        np.savez(
            self.cfg.save_path,
            dt_wall=np.float32(self.cfg.sample_dt),
            t_wall=np.asarray(self.t_wall_log, dtype=np.float32),
            t_sim=np.asarray(self.t_sim_log, dtype=np.float32),
            pitch=np.asarray(self.pitch_log, dtype=np.float32),
            flip=np.asarray(self.flip_rel_log, dtype=np.float32),
            u=np.asarray(self.u_log, dtype=np.float32),
            rate=np.asarray(self.rate_log, dtype=np.float32),
            acc=np.asarray(self.acc_log, dtype=np.float32),
            vz=np.asarray(self.vz_log, dtype=np.float32),
            vx=np.asarray(self.vx_log, dtype=np.float32),
            x_pose=np.asarray(self.x_log, dtype=np.float32),
            linear_speed_x=np.asarray(self.linear_x_log, dtype=np.float32),
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
