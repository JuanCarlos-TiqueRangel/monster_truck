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
import matplotlib.patches as patches

import matplotlib.pyplot as plt


@dataclass
class Config:
    ctrl_dt: float = 0.4        # s between random commands
    duration: float = 60.0      # total logging duration [s]
    u_min: float = -1.0
    u_max: float = 1.0
    refresh_hz: float = 5.0     # plot refresh rate [Hz]
    save_path: str = "utils/mujoco_random_run.npz"   # file to save data
    online_plot: bool = True    # <--- NEW: True = live plot, False = offline-only


class MujocoRandomCmdLogger(Node):
    def __init__(self, cfg: Config):
        super().__init__("mujoco_random_cmd_logger")
        self.cfg = cfg

        # Publisher for commands
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)

        # Subscriber for IMU
        self.imu_sub = self.create_subscription(
            Imu, "car_imu", self.imu_cb, 10
        )

        # Logs: all lists are appended in imu_cb
        self.t_log = []
        self.pitch_log = []
        self.flip_rel_log = []
        self.u_log = []
        self.rate_log = []
        self.acc_log = []
        self.vz_log = []
        self.vx_log = []

        # Time reference (simulation/IMU time from header)
        self.t0: Optional[float] = None

        # Angle unwrapping state
        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        # Last command used (logged with IMU)
        self.last_u: float = 0.0

        # TEMPORAL for flip_policy
        self.t_test_flip = 0.0

        self.get_logger().info("MujocoRandomCmdLogger node initialized")

    # ------------------ flip controller (unused here) ------------------
    def flip_policy(self, t, U_MIN=-1.0, U_MAX=1.0):
        """
        Simple open-loop sequence to try to flip the truck.
        Tune the timings / signs for your model.
        """
        if t < 0.5:
            return U_MAX        # punch in one direction
        elif t < 1.5:
            return U_MIN        # rebound
        elif t < 2.5:
            return U_MAX        # second kick
        else:
            return 0.0          # let it fly / settle

    # ------------------ helpers: quat -> rotation + up vector ------------------
    def quat_to_R_and_pitch(self, qw, qx, qy, qz):
        """
        Returns:
        R      : 3x3 rotation matrix (body -> world)
        pitch  : standard Euler pitch (for debugging, may have singularities)
        """
        # Rotation matrix for unit quaternion (w, x, y, z)
        R00 = 1 - 2*(qy*qy + qz*qz)
        R01 = 2*(qx*qy - qw*qz)
        R02 = 2*(qx*qz + qw*qy)

        R10 = 2*(qx*qy + qw*qz)
        R11 = 1 - 2*(qx*qx + qz*qz)
        R12 = 2*(qy*qz - qw*qx)

        R20 = 2*(qx*qz - qw*qy)
        R21 = 2*(qy*qz + qw*qx)
        R22 = 1 - 2*(qx*qx + qy*qy)

        # "classic" Euler pitch just for debugging
        pitch = -math.asin(max(-1.0, min(1.0, R20)))

        R = np.array([[R00, R01, R02],
                      [R10, R11, R12],
                      [R20, R21, R22]], dtype=float)
        return R, pitch

    def unwrap_angle(self, prev_angle, prev_unwrapped, angle):
        """
        Incremental unwrap of an angle in [-pi, pi] so it becomes continuous.
        """
        if prev_angle is None:
            return angle, angle  # first call
        d = angle - prev_angle
        # handle wrap-around at ±pi
        if d > math.pi:
            angle_unwrapped = prev_unwrapped + (d - 2*math.pi)
        elif d < -math.pi:
            angle_unwrapped = prev_unwrapped + (d + 2*math.pi)
        else:
            angle_unwrapped = prev_unwrapped + d
        return angle, angle_unwrapped

    # ---------------------------------------------------------------
    # Command publishing
    # ---------------------------------------------------------------
    def publish_cmd(self, u: float) -> None:
        """Publish a specific control command and log it as last_u."""
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)

    def publish_random_cmd(self) -> None:
        # u = float(np.clip(self.flip_policy(self.t_test_flip, self.cfg.u_min, self.cfg.u_max),
        #                   self.cfg.u_min, self.cfg.u_max))
        u = float(np.random.uniform(self.cfg.u_min, self.cfg.u_max))
        self.t_test_flip += self.cfg.ctrl_dt
        self.publish_cmd(u)

    # ---------------------------------------------------------------
    # IMU callback: log data each time we get /car_imu
    # ---------------------------------------------------------------
    def imu_cb(self, msg: Imu):
        # Get time from header
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        t_rel = t - self.t0

        # Orientation (qw, qx, qy, qz)
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        # Use helper to get rotation matrix and Euler pitch
        R, euler_pitch = self.quat_to_R_and_pitch(qw, qx, qy, qz)

        # "Up" vector in world coordinates = 3rd column of R
        up_x, up_y, up_z = R[0, 2], R[1, 2], R[2, 2]

        # Angle of up vector in (z,x) plane
        theta = math.atan2(up_x, up_z)

        # Unwrap the angle over time
        self.prev_theta, theta_unwrapped = self.unwrap_angle(
            self.prev_theta,
            self.prev_theta_unwrapped,
            theta,
        )
        self.prev_theta_unwrapped = theta_unwrapped

        # Set reference so that flip_rel ~ 0 at the beginning
        if self.theta0 is None:
            self.theta0 = theta_unwrapped
        flip_rel = theta_unwrapped - self.theta0

        # IMU signals
        # Assuming angular_velocity.y corresponds to pitch rate in your setup
        pitch_rate = float(msg.angular_velocity.y)
        # Assuming linear_acceleration.x is the one you used as acc_imu
        acc_imu = float(msg.linear_acceleration.x)

        # Append to logs
        self.t_log.append(t_rel)
        self.pitch_log.append(euler_pitch)
        self.flip_rel_log.append(flip_rel)
        self.u_log.append(self.last_u)
        self.rate_log.append(pitch_rate)
        self.acc_log.append(acc_imu)
        self.vz_log.append(up_z)
        self.vx_log.append(up_x)




def setup_figure(cfg: Config):
    lfontsize = 30

    plt.ion()
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(25, 14), sharex=True, constrained_layout=True
    )

    line_width = 4.0

    (line_pitch,) = ax1.plot([], [], lw=line_width)
    (line_flip,) = ax2.plot([], [], lw=line_width)
    (line_u,) = ax3.plot([], [], lw=line_width)
    (line_rate,) = ax4.plot([], [], lw=line_width)
    (line_acc,) = ax5.plot([], [], lw=line_width)

    ax1.set_ylabel("Euler pitch [rad]", fontsize=lfontsize)
    ax1.set_ylim(-2.0, 2.0)
    ax1.tick_params(axis='both', labelsize=lfontsize)
    ax1.grid(True, linewidth=1.3)

    ax2.set_ylabel("flip angle rel [rad]", fontsize=lfontsize)
    ax2.set_ylim(-3.5, 3.5)
    ax2.tick_params(axis='both', labelsize=lfontsize)
    ax2.grid(True, linewidth=1.3)

    ax3.set_ylabel("u", fontsize=lfontsize)
    ax3.set_ylim(cfg.u_min - 0.1, cfg.u_max + 0.1)
    ax3.tick_params(axis='both', labelsize=lfontsize)
    ax3.grid(True, linewidth=1.3)

    ax4.set_ylabel("pitch rate [rad/s]", fontsize=lfontsize)
    ax4.set_ylim(-10, 10)
    ax4.tick_params(axis='both', labelsize=lfontsize)
    ax4.grid(True, linewidth=1.3)

    ax5.set_ylabel("acc imu", fontsize=lfontsize)
    ax5.set_ylim(-50, 50)
    ax5.tick_params(axis='both', labelsize=lfontsize)
    ax5.set_xlabel("time [s]", fontsize=lfontsize)
    ax5.grid(True, linewidth=1.3)


    ax_up = fig.add_axes([0.43, 0.42, 0.18, 0.18],
                        projection="polar", zorder=10)

    ax_up.set_facecolor("white")
    ax_up.patch.set_alpha(1.0)

    # 0 rad at the top, angles increase clockwise (like your flip)
    ax_up.set_theta_zero_location("S")   # "North" = up
    ax_up.set_theta_direction(-1)        # clockwise

    (line_upvec,) = ax_up.plot([], [], lw=1.4)

    # radius always ~1 (just a unit circle)
    ax_up.set_rlim(0, 1.05)
    ax_up.set_rticks([])  # hide radial ticks

    # nice angle labels in radians
    ax_up.set_thetagrids(
        [0, 90, 180, 270],
        labels=["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"],
        fontsize=lfontsize - 6,
    )

    ax_up.set_title("Flip angle [rad]", fontsize=lfontsize)



    fig.suptitle("Monster-Truck Collect Data Experiment", fontsize=lfontsize)

    # ---- draw a white rectangle behind the whole ax_up (including labels) ----
    fig.canvas.draw()  # need this so tightbbox is available
    bbox = ax_up.get_tightbbox(fig.canvas.get_renderer())
    bbox_fig = bbox.transformed(fig.transFigure.inverted())

    rect = patches.FancyBboxPatch(
        (0.48, 0.42),   # left, bottom in figure coords (0–1)
        0.08, 0.20,     # width, height in figure coords
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        transform=fig.transFigure,
        zorder=ax_up.get_zorder() - 1,  # just under ax_up, above other axes
    )
    fig.patches.append(rect)

    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, (ax1, ax2, ax3, ax4, ax5), line_pitch, line_flip, line_u, line_rate, line_acc, ax_up, line_upvec



def update_plot(node: MujocoRandomCmdLogger,
                fig,
                axes,
                line_pitch,
                line_flip,
                line_u,
                line_rate,
                line_acc,
                line_upvec):
    ax1, ax2, ax3, ax4, ax5 = axes
    if not node.t_log:
        return

    t = node.t_log
    line_pitch.set_data(t, node.pitch_log)
    line_flip.set_data(t, node.flip_rel_log)
    line_u.set_data(t, node.u_log)
    line_rate.set_data(t, node.rate_log)
    line_acc.set_data(t, node.acc_log)


    # use flip_rel as angle around the circle
    theta = np.asarray(node.flip_rel_log)

    # map to [0, 2π] if you like
    theta = np.mod(theta, 2*np.pi)

    r = np.ones_like(theta)
    line_upvec.set_data(theta, r)

    # line_upvec.set_data(node.vx_log, node.vz_log)

    ax1.set_xlim(0.0, max(2.0, t[-1]))

    fig.canvas.draw()
    fig.canvas.flush_events()


# -------------------------------------------------------------------
# Main loop: ROS spin + random commands + (optional) live plotting
# -------------------------------------------------------------------
def main():
    # Toggle online vs offline behavior here:
    #   online_plot=True  -> live updating during run
    #   online_plot=False -> only final plot after run
    cfg = Config()

    rclpy.init()
    node = MujocoRandomCmdLogger(cfg)

    # Set up plotting (we reuse same figure for both online+offline)
    fig, axes, line_pitch, line_flip, line_u, line_rate, line_acc, ax_up, line_upvec = setup_figure(cfg)

    start_wall = time.perf_counter()
    last_cmd_wall = start_wall
    last_refresh_wall = start_wall

    try:
        while rclpy.ok():
            now = time.perf_counter()
            elapsed = now - start_wall

            # Stop after DURATION seconds
            if elapsed >= cfg.duration:
                break

            # 1) Spin ROS (process incoming IMU messages)
            rclpy.spin_once(node, timeout_sec=0.01)

            # 2) Publish random command at CTRL_DT
            if now - last_cmd_wall >= cfg.ctrl_dt:
                node.publish_random_cmd()
                last_cmd_wall = now

            # 3) Refresh plot at REFRESH_HZ (only if online_plot is True)
            if cfg.online_plot and (now - last_refresh_wall >= 1.0 / cfg.refresh_hz):
                update_plot(
                    node,
                    fig,
                    axes,
                    line_pitch,
                    line_flip,
                    line_u,
                    line_rate,
                    line_acc,
                    line_upvec,
                )
                last_refresh_wall = now
                # let matplotlib process GUI events
                plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        # Final plot update (this is your "offline" plot if online_plot=False)
        update_plot(
            node,
            fig,
            axes,
            line_pitch,
            line_flip,
            line_u,
            line_rate,
            line_acc,
            line_upvec,
        )

        # --------- send a final stop command -----------
        node.get_logger().info("Experiment finished, sending stop command (u=0.0)")
        node.publish_cmd(0.0)
        # Give ROS a tiny moment to actually send it
        rclpy.spin_once(node, timeout_sec=0.1)
        # ------------------------------------------------

        # --------- SAVE DATA TO NPZ ---------------------
        if node.t_log:
            t_arr     = np.asarray(node.t_log,        dtype=np.float32)
            pitch_arr = np.asarray(node.pitch_log,    dtype=np.float32)
            flip_arr  = np.asarray(node.flip_rel_log, dtype=np.float32)
            u_arr     = np.asarray(node.u_log,        dtype=np.float32)
            rate_arr  = np.asarray(node.rate_log,     dtype=np.float32)
            acc_arr   = np.asarray(node.acc_log,      dtype=np.float32)
            vz_arr    = np.asarray(node.vz_log,       dtype=np.float32)
            vx_arr    = np.asarray(node.vx_log,       dtype=np.float32)

            np.savez(
                cfg.save_path,
                t=t_arr,
                pitch=pitch_arr,
                flip=flip_arr,
                u=u_arr,
                rate=rate_arr,
                acc=acc_arr,
                vz=vz_arr,
                vx=vx_arr,
            )

            node.get_logger().info(
                f"Saved data to NPZ: {cfg.save_path}  (N={len(t_arr)})"
            )
            print(
                f"Done. Samples: {len(t_arr)}  Sim time: {t_arr[-1]:.3f}s  "
                f"saved to {cfg.save_path}"
            )
        else:
            node.get_logger().warn("No data collected, skipping NPZ save.")
        # ------------------------------------------------

        node.destroy_node()
        rclpy.shutdown()
        # Keep figure open at the end
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
