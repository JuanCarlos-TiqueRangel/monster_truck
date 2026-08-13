"""Plot wheelie and flip results with measured and RLS-estimated accelerations.

Run this script from the project directory that contains the ``flip`` and
``wheelie`` folders. Edit only the configuration section when paths, columns,
episodes, or pitch references change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

FLIP_CSV = Path("flip/results/flip_mujoco.csv")
WHEELIE_CSV = Path("wheelie/results/wheelie_mujoco.csv")
OUTPUT_FILE = Path("wheelie_flip_acceleration_results.pdf")

# Use None to select the last available episode in each file.
FLIP_EPISODE = None
WHEELIE_EPISODE = None

# CSV column names
EPISODE_COLUMN = "episode"
TIME_COLUMN = "time"
PITCH_COLUMN = "raw_pitch_deg"
TORQUE_COLUMN = "tau_cmd"
RLS_UPDATE_COUNT_COLUMN = "rls_update_count"

LINEAR_ACCELERATION_MEASURED_COLUMN = "rls_v_dot_measured"
LINEAR_ACCELERATION_ESTIMATED_COLUMN = "rls_v_dot_hat"
ANGULAR_ACCELERATION_MEASURED_COLUMN = "rls_omega_dot_measured"
ANGULAR_ACCELERATION_ESTIMATED_COLUMN = "rls_omega_dot_hat"

# Nose-up is negative pitch for the wheelie vehicle.
WHEELIE_PITCH_REFERENCE_DEG = -70.0
FLIP_PITCH_REFERENCE_DEG = 0.0
PITCH_TOLERANCE_DEG = 5.0

# Five RLS updates correspond to approximately 0.10 s for these logs.
# Set this to 1 to plot the logged acceleration values without smoothing.
ACCELERATION_SMOOTHING_SAMPLES = 5
ACCELERATION_BAND_STANDARD_DEVIATIONS = 1.0
ACCELERATION_BAND_ALPHA = 0.15

# Display-only smoothing for dense 1 kHz signals.
PITCH_SMOOTHING_SECONDS = 0.1
TORQUE_SMOOTHING_SECONDS = 0.20
TORQUE_BAND_STANDARD_DEVIATIONS = 1.0
TORQUE_BAND_ALPHA = 0.20

SHOW_RMSE = False
SHOW_PLOT = True


# Plot colors
ACTUAL_COLOR = "#0072B2"
REFERENCE_COLOR = "#D55E00"
ANGULAR_MEASURED_COLOR = "#5A4A6E"
ANGULAR_ESTIMATED_COLOR = "#7B61A8"
LINEAR_MEASURED_COLOR = "#356C67"
LINEAR_ESTIMATED_COLOR = "#009E73"
TORQUE_COLOR = "#C74440"
PANEL_COLOR = "#F0EFEF"
GRID_COLOR = "#FAFAFA"
ZERO_LINE_COLOR = "#A8A8A8"


# =============================================================================
# DATA PREPARATION
# =============================================================================

REQUIRED_COLUMNS = [
    EPISODE_COLUMN,
    TIME_COLUMN,
    PITCH_COLUMN,
    TORQUE_COLUMN,
    RLS_UPDATE_COUNT_COLUMN,
    LINEAR_ACCELERATION_MEASURED_COLUMN,
    LINEAR_ACCELERATION_ESTIMATED_COLUMN,
    ANGULAR_ACCELERATION_MEASURED_COLUMN,
    ANGULAR_ACCELERATION_ESTIMATED_COLUMN,
]


def require_columns(data, file_path):
    """Report configured column names that are not present in a CSV file."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        missing_text = ", ".join(missing)
        available_text = ", ".join(data.columns)
        raise KeyError(
            f"{file_path} is missing: {missing_text}.\n"
            f"Available columns: {available_text}"
        )


def load_episode(file_path, requested_episode):
    """Load and sort one maneuver episode."""
    data = pd.read_csv(file_path, comment="#")
    require_columns(data, file_path)

    available_episodes = np.sort(data[EPISODE_COLUMN].dropna().unique())
    if len(available_episodes) == 0:
        raise ValueError(f"No episodes were found in {file_path}")

    episode = available_episodes[-1] if requested_episode is None else requested_episode
    episode_data = data[data[EPISODE_COLUMN] == episode].copy()

    if episode_data.empty:
        raise ValueError(
            f"Episode {episode} was not found in {file_path}. "
            f"Available episodes: {available_episodes.tolist()}"
        )

    episode_data = episode_data.sort_values(TIME_COLUMN)
    episode_data = episode_data.drop_duplicates(subset=TIME_COLUMN, keep="last")

    # Express time relative to the beginning of the selected episode.
    episode_data[TIME_COLUMN] = (
        pd.to_numeric(episode_data[TIME_COLUMN], errors="coerce")
        - pd.to_numeric(episode_data[TIME_COLUMN], errors="coerce").iloc[0]
    )

    numeric_columns = [column for column in REQUIRED_COLUMNS if column != EPISODE_COLUMN]
    for column in numeric_columns:
        episode_data[column] = pd.to_numeric(episode_data[column], errors="coerce")

    episode_data = episode_data.dropna(subset=[TIME_COLUMN, PITCH_COLUMN, TORQUE_COLUMN])
    return episode_data, episode


def smooth_signal_and_band(values):
    """Return a centered mean and local mean +/- standard-deviation band."""
    values = pd.Series(values, dtype=float)

    if ACCELERATION_SMOOTHING_SAMPLES <= 1:
        values = values.to_numpy(dtype=float)
        return values, values, values

    rolling = values.rolling(
        ACCELERATION_SMOOTHING_SAMPLES,
        center=True,
        min_periods=1,
    )
    mean = rolling.mean().to_numpy(dtype=float)
    standard_deviation = rolling.std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    half_width = ACCELERATION_BAND_STANDARD_DEVIATIONS * standard_deviation
    return mean, mean - half_width, mean + half_width


def centered_time_mean(time, values, window_seconds):
    """Apply a centered moving mean using a window specified in seconds."""
    if window_seconds <= 0 or len(time) < 3:
        return np.asarray(values, dtype=float)

    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 0]
    if len(positive_dt) == 0:
        return np.asarray(values, dtype=float)

    sample_time = np.median(positive_dt)
    window_samples = max(1, int(round(window_seconds / sample_time)))
    if window_samples % 2 == 0:
        window_samples += 1

    return (
        pd.Series(values)
        .rolling(window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def centered_time_mean_and_band(time, values, window_seconds, band_std):
    """Return a centered mean and a local mean +/- standard-deviation band."""
    if window_seconds <= 0 or len(time) < 3:
        values = np.asarray(values, dtype=float)
        return values, values, values

    positive_dt = np.diff(time)
    positive_dt = positive_dt[positive_dt > 0]
    if len(positive_dt) == 0:
        values = np.asarray(values, dtype=float)
        return values, values, values

    sample_time = np.median(positive_dt)
    window_samples = max(1, int(round(window_seconds / sample_time)))
    if window_samples % 2 == 0:
        window_samples += 1

    rolling = pd.Series(values).rolling(
        window_samples,
        center=True,
        min_periods=1,
    )
    mean = rolling.mean().to_numpy(dtype=float)
    standard_deviation = rolling.std(ddof=0).fillna(0.0).to_numpy(dtype=float)
    half_width = band_std * standard_deviation
    return mean, mean - half_width, mean + half_width


def acceleration_signals(data, measured_column, estimated_column):
    """Return one measured/estimated acceleration pair per RLS update."""
    update_count = data[RLS_UPDATE_COUNT_COLUMN].to_numpy(dtype=float)
    previous_count = np.r_[np.nan, update_count[:-1]]
    update_rows = (update_count > 0) & (update_count != previous_count)

    time = data.loc[update_rows, TIME_COLUMN].to_numpy(dtype=float)
    measured = data.loc[update_rows, measured_column].to_numpy(dtype=float)
    estimated = data.loc[update_rows, estimated_column].to_numpy(dtype=float)

    measured, measured_lower, measured_upper = smooth_signal_and_band(measured)
    estimated, estimated_lower, estimated_upper = smooth_signal_and_band(estimated)

    valid = (
        np.isfinite(time)
        & np.isfinite(measured)
        & np.isfinite(estimated)
        & np.isfinite(measured_lower)
        & np.isfinite(measured_upper)
        & np.isfinite(estimated_lower)
        & np.isfinite(estimated_upper)
    )
    return (
        time[valid],
        measured[valid],
        estimated[valid],
        measured_lower[valid],
        measured_upper[valid],
        estimated_lower[valid],
        estimated_upper[valid],
    )


# =============================================================================
# PLOTTING
# =============================================================================

def style_axis(axis):
    axis.set_facecolor(PANEL_COLOR)
    axis.set_axisbelow(True)
    axis.grid(True, color=GRID_COLOR, linewidth=3.0)#, alpha=0.80)
    axis.tick_params(axis="both", which="both", length=0, labelsize=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_acceleration(
    axis,
    time,
    measured,
    estimated,
    measured_lower,
    measured_upper,
    estimated_lower,
    estimated_upper,
    measured_color,
    estimated_color,
    y_label,
):
    """Plot the acceleration target and its RLS estimate."""
    axis.fill_between(
        time,
        measured_lower,
        measured_upper,
        color=measured_color,
        alpha=ACCELERATION_BAND_ALPHA,
        linewidth=0,
    )
    axis.fill_between(
        time,
        estimated_lower,
        estimated_upper,
        color=estimated_color,
        alpha=ACCELERATION_BAND_ALPHA,
        linewidth=0,
    )
    axis.plot(
        time,
        measured,
        color=measured_color,
        linestyle="--",
        linewidth=2.2,
        # alpha=0.7,
        label="Measured",
    )
    axis.plot(
        time,
        estimated,
        color=estimated_color,
        linewidth=2.2,
        label="RLS estimate",
    )
    axis.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=0.9, alpha=0.75)
    axis.set_ylabel(y_label)

    if SHOW_RMSE and len(measured) > 0:
        error = measured - estimated
        rmse = np.sqrt(np.mean(error**2))
        axis.text(
            0.97,
            0.95,
            f"RMSE = {rmse:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )

    axis.legend(loc="lower left", frameon=False, fontsize=8.5)


def plot_maneuver_row(axes, data, pitch_reference_deg):
    """Plot pitch, angular acceleration, linear acceleration, and torque."""
    time = data[TIME_COLUMN].to_numpy(dtype=float)
    pitch = data[PITCH_COLUMN].to_numpy(dtype=float)
    torque = data[TORQUE_COLUMN].to_numpy(dtype=float)
    pitch = centered_time_mean(time, pitch, PITCH_SMOOTHING_SECONDS)

    pitch_reference = np.full_like(time, pitch_reference_deg)
    axes[0].fill_between(
        time,
        pitch_reference - PITCH_TOLERANCE_DEG,
        pitch_reference + PITCH_TOLERANCE_DEG,
        color=REFERENCE_COLOR,
        alpha=0.10,
        linewidth=0,
        label="Target tolerance",
    )
    axes[0].plot(
        time,
        pitch_reference,
        color=REFERENCE_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="Reference",
    )
    axes[0].plot(time, pitch, color=ACTUAL_COLOR, linewidth=2.2, label="Pitch")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].legend(loc="best", frameon=False, fontsize=8.5)

    (
        angular_time,
        angular_measured,
        angular_estimated,
        angular_measured_lower,
        angular_measured_upper,
        angular_estimated_lower,
        angular_estimated_upper,
    ) = acceleration_signals(
        data,
        ANGULAR_ACCELERATION_MEASURED_COLUMN,
        ANGULAR_ACCELERATION_ESTIMATED_COLUMN,
    )
    plot_acceleration(
        axes[1],
        angular_time,
        angular_measured,
        angular_estimated,
        angular_measured_lower,
        angular_measured_upper,
        angular_estimated_lower,
        angular_estimated_upper,
        ANGULAR_MEASURED_COLOR,
        ANGULAR_ESTIMATED_COLOR,
        r"Angular acceleration (rad/s$^2$)",
    )

    (
        linear_time,
        linear_measured,
        linear_estimated,
        linear_measured_lower,
        linear_measured_upper,
        linear_estimated_lower,
        linear_estimated_upper,
    ) = acceleration_signals(
        data,
        LINEAR_ACCELERATION_MEASURED_COLUMN,
        LINEAR_ACCELERATION_ESTIMATED_COLUMN,
    )
    plot_acceleration(
        axes[2],
        linear_time,
        linear_measured,
        linear_estimated,
        linear_measured_lower,
        linear_measured_upper,
        linear_estimated_lower,
        linear_estimated_upper,
        LINEAR_MEASURED_COLOR,
        LINEAR_ESTIMATED_COLOR,
        r"Linear acceleration (m/s$^2$)",
    )

    torque_mean, torque_lower, torque_upper = centered_time_mean_and_band(
        time,
        torque,
        TORQUE_SMOOTHING_SECONDS,
        TORQUE_BAND_STANDARD_DEVIATIONS,
    )
    axes[3].fill_between(
        time,
        torque_lower,
        torque_upper,
        color=TORQUE_COLOR,
        alpha=TORQUE_BAND_ALPHA,
        linewidth=0,
        #label=r"Local variation ($\pm 1\sigma$)",
    )
    axes[3].plot(
        time,
        torque_mean,
        color=TORQUE_COLOR,
        linewidth=2.2,
        label="Applied Torque",
    )
    axes[3].axhline(
        0.0,
        color=ZERO_LINE_COLOR,
        linewidth=0.9,
        alpha=0.75,
    )
    axes[3].set_ylabel(r"Torque (N$\cdot$m)")
    axes[3].legend(loc="best", frameon=False, fontsize=8.0)

    for axis in axes:
        style_axis(axis)
        axis.set_xlabel("Time (s)")


def create_figure(flip_data, wheelie_data, flip_episode, wheelie_episode):
    """Create the combined two-maneuver results figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
        }
    )

    figure, axes = plt.subplots(2, 4, figsize=(17, 8))
    figure.patch.set_facecolor("#FAFAFA")

    titles = [
        "Pitch Tracking",
        "Angular Acceleration",
        "Linear Acceleration",
        "Control Input",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, pad=10, fontweight="medium")

    plot_maneuver_row(axes[0], wheelie_data, WHEELIE_PITCH_REFERENCE_DEG)
    plot_maneuver_row(axes[1], flip_data, FLIP_PITCH_REFERENCE_DEG)

    row_labels = [(axes[0, 0], "WHEELIE"), (axes[1, 0], "FLIP")]
    for axis, label in row_labels:
        axis.text(
            -0.32,
            0.5,
            label,
            transform=axis.transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#333333",
        )

    # figure.suptitle(
    #     "Wheelie and Flip Acceleration-Estimation Results",
    #     fontsize=20,
    #     fontweight="bold",
    #     #y=0.982,
    # )
    if ACCELERATION_SMOOTHING_SAMPLES > 1:
        smoothing_note = (
            f" A {ACCELERATION_SMOOTHING_SAMPLES}-update centered mean is shown."
        )
    else:
        smoothing_note = " No display smoothing is applied."

    # figure.text(
    #     0.5,
    #     0.018,
    #     "Acceleration panels compare logged RLS targets with online RLS estimates; "
    #     f"pre-update samples are omitted.{smoothing_note} "
    #     "Acceleration and torque shading show +/-1 local standard deviation, "
    #     "not run-to-run uncertainty.",
    #     ha="center",
    #     va="bottom",
    #     fontsize=9.5,
    #     color="#555555",
    # )

    # figure.subplots_adjust(top=0.86)

    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.85,
        bottom=0.105,
        wspace=0.42,
        hspace=0.40,
    )
    figure.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    return figure


def main():
    flip_data, flip_episode = load_episode(FLIP_CSV, FLIP_EPISODE)
    wheelie_data, wheelie_episode = load_episode(WHEELIE_CSV, WHEELIE_EPISODE)

    create_figure(flip_data, wheelie_data, flip_episode, wheelie_episode)

    print(f"Flip episode: {flip_episode}")
    print(f"Wheelie episode: {wheelie_episode}")
    print(f"Saved figure: {OUTPUT_FILE.resolve()}")

    if SHOW_PLOT:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
