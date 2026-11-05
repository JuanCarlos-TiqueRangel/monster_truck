#!/usr/bin/env python3
"""Display the CSV files recorded by model_ident.py. Requires numpy and matplotlib.

Run beside model_ident.py to open the latest recording:
    python3 plot_model_ident.py
Or select a recording folder:
    python3 plot_model_ident.py results/optitrack_<timestamp>

Every plot uses the same receive/publication time origin. Original source
stamps are shown through the timing plots. No figures are saved by default.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Settings
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOT_COVARIANCES = False
PLOT_TIMING = False
CSV_NAMES = ("accel", "odom", "rpy", "twist", "drive")


def select_folder(folder):
    if folder is not None:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Recording folder does not exist: {folder}")
        return folder

    runs = sorted(path for path in RESULTS_DIR.glob("optitrack_*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(
            f"No optitrack_* recording folders in {RESULTS_DIR}. "
            "Run model_ident.py first or pass a recording folder.")
    return runs[-1]


# Read CSVs. Keep nanosecond timestamps as integers to preserve their precision.
def load_recordings(folder):
    recordings = {}
    for name in CSV_NAMES:
        path = folder / f"{name}.csv"
        if not path.exists():
            print(f"Skipping missing {path.name}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"Skipping empty {path.name}")
            continue

        data = {}
        for column in rows[0]:
            values = [row[column] for row in rows]
            if column in ("frame_id", "child_frame_id"):
                data[column] = values
            elif column.endswith("_ns"):
                data[column] = np.array([int(value) for value in values], dtype=np.int64)
            else:
                data[column] = np.array(values, dtype=float)
        recordings[name] = data
        print(f"{path.name}: {len(rows)} samples")
        for column in ("frame_id", "child_frame_id"):
            if column in data:
                print(f"  {column}: {', '.join(dict.fromkeys(data[column]))}")

    if not recordings:
        raise ValueError(f"No recorded samples found in {folder}")
    return recordings


def receive_stamps(data):
    if "received_stamp_ns" in data:
        return data["received_stamp_ns"]
    return data["published_stamp_ns"]


def time_seconds(data, origin_ns):
    return (receive_stamps(data) - origin_ns) * 1e-9


# Plot each topic. Column names match the CSV headers exactly.
def plot_topic(name, data, origin_ns):
    linear = ["linear_x", "linear_y", "linear_z"]
    angular = ["angular_x", "angular_y", "angular_z"]
    groups = {
        "accel": [("Linear acceleration", linear, "m/s²"),
                  ("Angular acceleration", angular, "rad/s²")],
        "twist": [("Linear velocity", linear, "m/s"),
                  ("Angular velocity", angular, "rad/s")],
        "odom": [("Position", ["position_x", "position_y", "position_z"], "m"),
                 ("Orientation quaternion", ["orientation_x", "orientation_y",
                                             "orientation_z", "orientation_w"], "value"),
                 ("Odometry linear velocity", linear, "m/s"),
                 ("Odometry angular velocity", angular, "rad/s")],
        "rpy": [("Roll, pitch, yaw", ["roll", "pitch", "yaw"], "rad")],
        "drive": [("Published /drive PRBS", ["drive"], "command")],
    }[name]

    figure, axes = plt.subplots(len(groups), 1, figsize=(11, 2.7 * len(groups) + 1),
                                sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    time_s = time_seconds(data, origin_ns)
    title = f"{name}.csv"
    if "frame_id" in data:
        title += f" — frame: {', '.join(dict.fromkeys(data['frame_id']))}"
    figure.suptitle(title)
    for axis, (title, columns, units) in zip(axes, groups):
        for column in columns:
            if name == "drive":
                axis.step(time_s, data[column], where="post", label=column)
            else:
                axis.plot(time_s, data[column], label=column)
        axis.set_title(title)
        axis.set_ylabel(units)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", ncol=min(len(columns), 4))
    axes[-1].set_xlabel("Time since first received/published message (s)")
    return figure


# Show every entry of the two 6 × 6 covariance matrices, including off-diagonals.
def plot_covariance(data, origin_ns, kind):
    labels = ("x", "y", "z", "roll", "pitch", "yaw")
    if kind == "twist":
        labels = ("vx", "vy", "vz", "wx", "wy", "wz")
    figure, axes = plt.subplots(6, 6, figsize=(15, 10), sharex=True,
                                constrained_layout=True)
    figure.suptitle(f"Odometry {kind} covariance — all 36 entries, in recorded units")
    time_s = time_seconds(data, origin_ns)
    for row in range(6):
        for column in range(6):
            axis = axes[row, column]
            field = f"{kind}_covariance_{6 * row + column}"
            axis.plot(time_s, data[field], linewidth=1)
            axis.set_title(f"{labels[row]}, {labels[column]}", fontsize=9)
            axis.tick_params(labelsize=7)
            axis.grid(True, alpha=0.3)
    figure.supxlabel("Time since first received/published message (s)")
    return figure


# Timing data: arrival spacing for every topic, and receive minus source stamps.
def plot_timing(recordings, origin_ns):
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                constrained_layout=True)
    figure.suptitle("Recorded timestamps")
    for name, data in recordings.items():
        time_s = time_seconds(data, origin_ns)
        intervals_ms = np.diff(receive_stamps(data)) * 1e-6
        axes[0].plot(time_s[1:], intervals_ms, label=name)
        if "stamp_ns" in data:
            difference_ms = (data["received_stamp_ns"] - data["stamp_ns"]) * 1e-6
            axes[1].plot(time_s, difference_ms, label=name)
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].set_title("Interval between received messages / drive publications")
    axes[1].set_title("Receive timestamp − source timestamp")
    for axis in axes:
        axis.set_ylabel("ms")
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(loc="upper right")
        else:
            axis.text(0.5, 0.5, "No stamped sensor data", ha="center", transform=axis.transAxes)
    axes[-1].set_xlabel("Time since first received/published message (s)")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", help="Recording folder; default: latest optitrack_* run")
    args = parser.parse_args()
    try:
        folder = select_folder(args.folder)
        recordings = load_recordings(folder)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    origin_ns = min(int(receive_stamps(data).min()) for data in recordings.values())
    print(f"Recording: {folder}\nTime origin: {origin_ns} ns (ROS clock)")
    figures = [(name, plot_topic(name, data, origin_ns)) for name, data in recordings.items()]
    if PLOT_COVARIANCES and "odom" in recordings:
        for kind in ("pose", "twist"):
            figures.append((f"odom_{kind}_covariance",
                            plot_covariance(recordings["odom"], origin_ns, kind)))
    if PLOT_TIMING:
        figures.append(("timestamps", plot_timing(recordings, origin_ns)))

    for name, figure in figures:
        figure.canvas.manager.set_window_title(f"{folder.name} — {name}")
        # figure.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight")

    plt.show()  # Display all figures; close the windows to exit.


if __name__ == "__main__":
    main()
