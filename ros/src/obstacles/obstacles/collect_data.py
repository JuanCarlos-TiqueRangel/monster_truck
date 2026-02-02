#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # We want a FIXED dataset dt for GP training
    sample_dt: float = 0.1        # [s] dataset sampling + command update rate (10 Hz)

    duration: float = 5.0        # [s] total run time
    u_min: float = -1.0
    u_max: float = 1.0

    refresh_hz: float = 5.0       # plot refresh
    online_plot: bool = False

    # save_path: str = "utils/data/mujoco_random_run.npz"
    save_path: str = "data/mujoco_random_wheelie.npz"
    # save_path: str = "utils/mujoco_random_run_dt0p2.npz"
    


# ============================================================
# Logger node
# ============================================================

class MujocoRandomCmdLogger(Node):
    """
    - Subscribes IMU at whatever rate it arrives (e.g., 100 Hz)
    - Keeps the latest decoded state in memory
    - Uses a timer at sample_dt (=0.1s) to:
        (1) publish a new random command u[k]
        (2) log ONE sample (t, flip, rate, u, etc.)
    So training data has a clean fixed dt.
    """
    def __init__(self, cfg: Config):
        super().__init__("mujoco_random_cmd_logger")
        self.cfg = cfg

        # Publisher for commands
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)

        # Subscriber for IMU
        self.imu_sub = self.create_subscription(Imu, "car_imu", self.imu_cb, 50)
        
        self.odom_sub = self.create_subscription(Odometry, "car_odom", self.odom_cb, 10)

        # ---- latest decoded signals from IMU (updated in imu_cb) ----
        self.have_imu: bool = False
        self.latest_t_rel: float = 0.0
        self.latest_pitch: float = 0.0
        self.latest_flip_rel: float = 0.0
        self.latest_rate: float = 0.0
        self.latest_acc: float = 0.0
        self.latest_up_z: float = 0.0
        self.latest_up_x: float = 0.0

        # odometry data variables
        self.x_pos = 0.0
        self.line_speed_x = 0.0

        # Time reference (IMU header time)
        self.t0: Optional[float] = None

        # Angle unwrapping state (for flip_rel)
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        # Last command used
        self.last_u: float = 0.0

        # ---- fixed-rate logs (appended in timer_cb) ----
        self.t_log = []
        self.pitch_log = []
        self.flip_rel_log = []
        self.u_log = []
        self.rate_log = []
        self.acc_log = []
        self.vz_log = []
        self.vx_log = []
        
        # state space for gp model odom
        self.x_log = []
        self.linear_x_log = []

        # Run control/log loop at fixed dt
        self.timer = self.create_timer(self.cfg.sample_dt, self.timer_cb)

        self.start_wall = time.perf_counter()
        self.get_logger().info(
            f"Logger initialized. Logging at fixed dt={self.cfg.sample_dt:.3f}s "
            f"({1.0/self.cfg.sample_dt:.1f} Hz) for {self.cfg.duration:.1f}s"
        )

    # ------------------ helpers: quat -> rotation + up vector ------------------
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

    # ---------------------------------------------------------------
    # IMU callback: update latest state only (NO logging here)
    # ---------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        t_rel = t - self.t0

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

        # Store latest
        self.latest_t_rel = float(t_rel)
        self.latest_pitch = float(euler_pitch)
        self.latest_flip_rel = float(flip_rel)
        self.latest_rate = float(pitch_rate)
        self.latest_acc = float(acc_imu)
        self.latest_up_z = float(up_z)
        self.latest_up_x = float(up_x)
        self.have_imu = True

    def odom_cb(self, msg: Odometry):
        x_pos = msg.pose.pose.position.x
        y_pos = msg.pose.pose.position.y
        z_pos = msg.pose.pose.position.z
        line_speed_x = msg.twist.twist.linear.x
        line_speed_y = msg.twist.twist.linear.y
        line_speed_z = msg.twist.twist.linear.z
        
        self.x_pos = float(x_pos)
        self.line_speed_x = float(line_speed_x)
        

    # ---------------------------------------------------------------
    # Fixed-rate timer: publish command + log one sample
    # ---------------------------------------------------------------
    def timer_cb(self):
        # stop after duration
        if (time.perf_counter() - self.start_wall) >= self.cfg.duration:
            self.publish_cmd(0.0)
            self.get_logger().info("Duration reached. Stopping logger timer.")
            self.timer.cancel()
            return

        # need at least one IMU sample before logging
        if not self.have_imu:
            return

        # 1) publish new random command (piecewise constant over next sample_dt)
        u = float(np.random.uniform(self.cfg.u_min, self.cfg.u_max))
        self.publish_cmd(u)

        # 2) log exactly ONE sample at this fixed dt
        self.t_log.append(self.latest_t_rel)
        self.pitch_log.append(self.latest_pitch)
        self.flip_rel_log.append(self.latest_flip_rel)
        self.u_log.append(self.last_u)              # this u is what we just published
        self.rate_log.append(self.latest_rate)
        self.acc_log.append(self.latest_acc)
        self.vz_log.append(self.latest_up_z)
        self.vx_log.append(self.latest_up_x)
        
        self.x_log.append(self.x_pos)
        self.linear_x_log.append(self.line_speed_x)

    # ---------------------------------------------------------------
    def publish_cmd(self, u: float) -> None:
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)


# ============================================================
# Plot helpers (unchanged style, but now uses fixed-rate logs)
# ============================================================

def setup_figure(cfg: Config):
    lfontsize = 30
    plt.ion()
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(25, 14), sharex=True, constrained_layout=True
    )

    line_width = 4.0
    (line_pitch,) = ax1.plot([], [], lw=line_width)
    (line_flip,)  = ax2.plot([], [], lw=line_width)
    (line_u,)     = ax3.plot([], [], lw=line_width)
    (line_rate,)  = ax4.plot([], [], lw=line_width)
    (line_acc,)   = ax5.plot([], [], lw=line_width)

    ax1.set_ylabel("Euler pitch [rad]", fontsize=lfontsize)
    ax1.set_ylim(-2.0, 2.0); ax1.tick_params(axis='both', labelsize=lfontsize); ax1.grid(True, linewidth=1.3)

    ax2.set_ylabel("flip angle rel [rad]", fontsize=lfontsize)
    ax2.set_ylim(-3.5, 3.5); ax2.tick_params(axis='both', labelsize=lfontsize); ax2.grid(True, linewidth=1.3)

    ax3.set_ylabel("u", fontsize=lfontsize)
    ax3.set_ylim(cfg.u_min - 0.1, cfg.u_max + 0.1); ax3.tick_params(axis='both', labelsize=lfontsize); ax3.grid(True, linewidth=1.3)

    ax4.set_ylabel("pitch rate [rad/s]", fontsize=lfontsize)
    ax4.set_ylim(-10, 10); ax4.tick_params(axis='both', labelsize=lfontsize); ax4.grid(True, linewidth=1.3)

    ax5.set_ylabel("acc imu", fontsize=lfontsize)
    ax5.set_ylim(-50, 50); ax5.tick_params(axis='both', labelsize=lfontsize)
    ax5.set_xlabel("time [s]", fontsize=lfontsize); ax5.grid(True, linewidth=1.3)

    ax_up = fig.add_axes([0.43, 0.42, 0.18, 0.18], projection="polar", zorder=10)
    ax_up.set_facecolor("white"); ax_up.patch.set_alpha(1.0)
    ax_up.set_theta_zero_location("S")
    ax_up.set_theta_direction(-1)
    (line_upvec,) = ax_up.plot([], [], lw=1.4)
    ax_up.set_rlim(0, 1.05); ax_up.set_rticks([])
    ax_up.set_thetagrids([0, 90, 180, 270], labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"], fontsize=lfontsize - 6)
    ax_up.set_title("Flip angle [rad]", fontsize=lfontsize)

    fig.suptitle(f"Collect Data (fixed dt={cfg.sample_dt:.2f}s)", fontsize=lfontsize)

    fig.canvas.draw()
    rect = patches.FancyBboxPatch(
        (0.48, 0.42), 0.08, 0.20,
        boxstyle="round,pad=0.02",
        facecolor="white", edgecolor="black", linewidth=1.5,
        transform=fig.transFigure, zorder=ax_up.get_zorder() - 1,
    )
    fig.patches.append(rect)
    fig.canvas.draw(); fig.canvas.flush_events()

    return fig, (ax1, ax2, ax3, ax4, ax5), line_pitch, line_flip, line_u, line_rate, line_acc, ax_up, line_upvec


def update_plot(node: MujocoRandomCmdLogger, fig, axes, line_pitch, line_flip, line_u, line_rate, line_acc, line_upvec):
    ax1, ax2, ax3, ax4, ax5 = axes
    if not node.t_log:
        return

    t = node.t_log
    line_pitch.set_data(t, node.pitch_log)
    line_flip.set_data(t, node.flip_rel_log)
    line_u.set_data(t, node.u_log)
    line_rate.set_data(t, node.rate_log)
    line_acc.set_data(t, node.acc_log)

    theta = np.mod(np.asarray(node.flip_rel_log), 2*np.pi)
    r = np.ones_like(theta)
    line_upvec.set_data(theta, r)

    ax1.set_xlim(0.0, max(2.0, t[-1]))
    fig.canvas.draw()
    fig.canvas.flush_events()


# ============================================================
# Main
# ============================================================

def main():
    cfg = Config()

    rclpy.init()
    node = MujocoRandomCmdLogger(cfg)

    fig, axes, line_pitch, line_flip, line_u, line_rate, line_acc, ax_up, line_upvec = setup_figure(cfg)
    last_refresh_wall = time.perf_counter()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            # refresh plot
            now = time.perf_counter()
            if cfg.online_plot and (now - last_refresh_wall >= 1.0 / cfg.refresh_hz):
                update_plot(node, fig, axes, line_pitch, line_flip, line_u, line_rate, line_acc, line_upvec)
                last_refresh_wall = now
                plt.pause(0.001)

            # stop when timer canceled
            if node.timer.is_canceled():
                break

    except KeyboardInterrupt:
        pass
    finally:
        # final update + stop cmd
        update_plot(node, fig, axes, line_pitch, line_flip, line_u, line_rate, line_acc, line_upvec)

        node.get_logger().info("Experiment finished, sending stop command (u=0.0)")
        node.publish_cmd(0.0)
        rclpy.spin_once(node, timeout_sec=0.1)

        # save NPZ
        if node.t_log:
            np.savez(
                cfg.save_path,
                # IMPORTANT: training should assume dt = cfg.sample_dt (fixed)
                dt      =           np.float32(cfg.sample_dt),
                t       =           np.asarray(node.t_log, dtype=np.float32),
                pitch   =           np.asarray(node.pitch_log, dtype=np.float32),
                flip    =           np.asarray(node.flip_rel_log, dtype=np.float32),
                u       =           np.asarray(node.u_log, dtype=np.float32),
                rate    =           np.asarray(node.rate_log, dtype=np.float32),
                acc     =           np.asarray(node.acc_log, dtype=np.float32),
                vz      =           np.asarray(node.vz_log, dtype=np.float32),
                vx      =           np.asarray(node.vx_log, dtype=np.float32),
                x_pose   =           np.asarray(node.x_log, dtype=np.float32),
                linear_speed_x  =   np.asarray(node.linear_x_log, dtype=np.float32),
            )
            node.get_logger().info(f"Saved data to NPZ: {cfg.save_path} (N={len(node.t_log)})")
            print(f"Done. Samples: {len(node.t_log)}  approx sim time: {node.t_log[-1]:.3f}s")
        else:
            node.get_logger().warn("No data collected, skipping NPZ save.")

        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
