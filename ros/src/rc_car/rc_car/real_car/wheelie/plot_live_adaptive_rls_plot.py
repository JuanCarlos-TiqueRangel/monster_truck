#!/usr/bin/env python3

import math
import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# ============================================================
# PLOT OPTIONS
# ============================================================

PLOT_V_ACC = True
PLOT_W_ACC = False
PLOT_V = True
PLOT_W = True

PLOT_WINDOW_S = 5.0
SHOW_MAIN_TITLE = True


class OnlineRLSPlotNode(Node):

    def __init__(self):
        super().__init__("online_rls_plot")

        self.lock = threading.Lock()
        self.t0 = None
        maxlen = 5000

        self.t = deque(maxlen=maxlen)
        self.v_dot_meas = deque(maxlen=maxlen)
        self.v_dot_pred = deque(maxlen=maxlen)
        self.w_dot_meas = deque(maxlen=maxlen)
        self.w_dot_pred = deque(maxlen=maxlen)
        self.v_meas = deque(maxlen=maxlen)
        self.v_pred = deque(maxlen=maxlen)
        self.w_meas = deque(maxlen=maxlen)
        self.w_pred = deque(maxlen=maxlen)
        self.theta = deque(maxlen=maxlen)
        self.drive = deque(maxlen=maxlen)
        self.updates = deque(maxlen=maxlen)

        self.prev_source_time = None
        self.prev_v = None
        self.prev_w = None
        self.prev_v_dot_pred = None
        self.prev_w_dot_pred = None

        self.diag_sub = self.create_subscription(
            Float64MultiArray,
            "/adaptive_wheelie/rls_diagnostics",
            self.diag_callback,
            20,
        )

    def diag_callback(self, msg):
        if len(msg.data) != 11:
            return

        data = np.asarray(msg.data, dtype=float)
        if not np.all(np.isfinite(data)):
            return

        source_time = float(data[0])

        v_dot_measured = float(data[1])
        w_dot_measured = float(data[2])
        v_dot_predicted = float(data[3])
        w_dot_predicted = float(data[4])

        theta = float(data[5])
        drive = float(data[6])
        updates = int(data[8])

        v_measured = float(data[9])
        w_measured = float(data[10])

        if self.t0 is None:
            self.t0 = source_time

        v_predicted = np.nan
        w_predicted = np.nan

        if self.prev_source_time is not None and source_time > self.prev_source_time:
            dt = source_time - self.prev_source_time

            if 0.0 < dt <= 0.1:
                v_predicted = self.prev_v + dt * self.prev_v_dot_pred
                w_predicted = self.prev_w + dt * self.prev_w_dot_pred

        with self.lock:
            self.t.append(source_time - self.t0)

            self.v_dot_meas.append(v_dot_measured)
            self.v_dot_pred.append(v_dot_predicted)

            self.w_dot_meas.append(w_dot_measured)
            self.w_dot_pred.append(w_dot_predicted)

            self.v_meas.append(v_measured)
            self.v_pred.append(v_predicted)

            self.w_meas.append(w_measured)
            self.w_pred.append(w_predicted)

            self.theta.append(theta)
            self.drive.append(drive)
            self.updates.append(updates)

        self.prev_source_time = source_time
        self.prev_v = v_measured
        self.prev_w = w_measured
        self.prev_v_dot_pred = v_dot_predicted
        self.prev_w_dot_pred = w_dot_predicted

    def snapshot(self):
        with self.lock:
            if not self.t:
                return None

            return {
                "t": np.asarray(self.t, dtype=float),
                "v_dot_meas": np.asarray(self.v_dot_meas, dtype=float),
                "v_dot_pred": np.asarray(self.v_dot_pred, dtype=float),
                "w_dot_meas": np.asarray(self.w_dot_meas, dtype=float),
                "w_dot_pred": np.asarray(self.w_dot_pred, dtype=float),
                "v_meas": np.asarray(self.v_meas, dtype=float),
                "v_pred": np.asarray(self.v_pred, dtype=float),
                "w_meas": np.asarray(self.w_meas, dtype=float),
                "w_pred": np.asarray(self.w_pred, dtype=float),
                "theta": np.asarray(self.theta, dtype=float),
                "drive": np.asarray(self.drive, dtype=float),
                "updates": np.asarray(self.updates, dtype=int),
            }


def rmse(measured, predicted):
    valid = np.isfinite(measured) & np.isfinite(predicted)

    if not np.any(valid):
        return math.nan

    return float(np.sqrt(np.mean((predicted[valid] - measured[valid])**2)))


def main():
    enabled = []

    if PLOT_V_ACC:
        enabled.append("v_acc")
    if PLOT_W_ACC:
        enabled.append("w_acc")
    if PLOT_V:
        enabled.append("v")
    if PLOT_W:
        enabled.append("w")

    if not enabled:
        raise RuntimeError("Enable at least one PLOT_* option.")

    rclpy.init()
    node = OnlineRLSPlotNode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    fig, axes = plt.subplots(
        len(enabled),
        1,
        figsize=(13, 3.2 * len(enabled)),
        sharex=True,
    )

    if len(enabled) == 1:
        axes = [axes]

    plot_items = {}

    for ax, name in zip(axes, enabled):
        measured_line, = ax.plot([], [], linewidth=1.8, label="Measured")
        predicted_line, = ax.plot([], [], linewidth=1.8, label="Online RLS")

        ax.grid(True, alpha=0.3)
        ax.legend()

        if name == "v_acc":
            ax.set_ylabel(r"$\dot{v}$ [m/s$^2$]")
        elif name == "w_acc":
            ax.set_ylabel(r"$\dot{\omega}$ [rad/s$^2$]")
        elif name == "v":
            ax.set_ylabel(r"$v$ [m/s]")
        elif name == "w":
            ax.set_ylabel(r"$\omega$ [rad/s]")

        plot_items[name] = (ax, measured_line, predicted_line)

    axes[-1].set_xlabel("Time [s]")

    def update(_):
        data = node.snapshot()

        if data is None:
            return tuple(
                line
                for _, measured, predicted in plot_items.values()
                for line in (measured, predicted)
            )

        t = data["t"]
        t_end = t[-1]
        t_start = max(0.0, t_end - PLOT_WINDOW_S)
        visible = t >= t_start
        tv = t[visible]

        datasets = {
            "v_acc": (
                data["v_dot_meas"][visible],
                data["v_dot_pred"][visible],
            ),
            "w_acc": (
                data["w_dot_meas"][visible],
                data["w_dot_pred"][visible],
            ),
            "v": (
                data["v_meas"][visible],
                data["v_pred"][visible],
            ),
            "w": (
                data["w_meas"][visible],
                data["w_pred"][visible],
            ),
        }

        titles = {
            "v_acc": ("Linear Acceleration", "m/s²"),
            "w_acc": ("Pitch Angular Acceleration", "rad/s²"),
            "v": ("Linear Velocity | one-step prediction", "m/s"),
            "w": ("Pitch Angular Velocity | one-step prediction", "rad/s"),
        }

        for name in enabled:
            ax, measured_line, predicted_line = plot_items[name]
            measured, predicted = datasets[name]

            measured_line.set_data(tv, measured)
            predicted_line.set_data(tv, predicted)

            error = rmse(measured, predicted)
            title, units = titles[name]

            if math.isfinite(error):
                ax.set_title(f"{title} | rolling RMSE = {error:.4f} {units}")
            else:
                ax.set_title(f"{title} | waiting for data")

            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)

        if len(tv):
            axes[0].set_xlim(t_start, max(t_start + 0.25, t_end))

        if SHOW_MAIN_TITLE:
            updates = int(data["updates"][-1])
            theta_deg = math.degrees(float(data["theta"][-1]))
            drive = float(data["drive"][-1])

            fig.suptitle(
                f"Controller Online AdaptiveRLS | updates={updates} | "
                f"pitch={theta_deg:.1f}° | drive={drive:.4f}",
                fontweight="bold",
            )

        return tuple(
            line
            for _, measured, predicted in plot_items.values()
            for line in (measured, predicted)
        )

    animation = FuncAnimation(
        fig,
        update,
        interval=50,
        cache_frame_data=False,
        blit=False,
    )

    plt.tight_layout()

    try:
        plt.show()

    finally:
        if rclpy.ok():
            rclpy.shutdown()

        ros_thread.join(timeout=1.0)
        node.destroy_node()


if __name__ == "__main__":
    main()
