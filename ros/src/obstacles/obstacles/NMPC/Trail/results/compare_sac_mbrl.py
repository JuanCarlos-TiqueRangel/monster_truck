"""Create a publication-style SAC-versus-MBRL comparison figure.

The SAC CSV is expected to contain one row per episode.
The MBRL CSV is expected to contain one row per simulation timestep.

Edit only the CONFIGURATION section below when filenames or columns change.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION: edit filenames, column names, and plot settings here
# =============================================================================

SAC_FILE = Path("sac_monitor/monitor.csv")
MBRL_FILE = Path("obstacle_mujoco_1000.csv")
OUTPUT_DIRECTORY = Path("images")

# SAC: one row per episode
SAC_TIME_COLUMN = "sim_time"       # Episode duration, not cumulative wall time "t"
SAC_SUCCESS_COLUMN = "reached"
SAC_EFFORT_COLUMN = "effort_rms"

# MBRL: many timestep rows per episode
MBRL_EPISODE_COLUMN = "episode"
MBRL_TIME_COLUMN = "time"
MBRL_POSITION_COLUMN = "x"
MBRL_EFFORT_COLUMN = "ctrl_cmd"    # Change to "tau_cmd" if that matches SAC units

# Leave this as None to infer success from the position threshold below.
# If the MBRL file later contains a success flag, use its name, e.g. "reached".
MBRL_SUCCESS_COLUMN = None

# An MBRL episode is successful if the car reaches x >= 9.85 at any time.
MBRL_SUCCESS_POSITION = 9.85

# Each curve is a trailing rolling average over this many episodes. Because the
# current comparison has one file per method, time and effort use rolling
# standard-deviation bands. Success uses a 95% Wilson confidence interval,
# which is appropriate for a rolling proportion of successful episodes.
ROLLING_WINDOW = 100
ROLLING_BAND_STD = 1.0
BAND_ALPHA = 0.20
SUCCESS_CONFIDENCE_Z = 1.96
SUCCESS_BAND_ALPHA = 0.12

# Set to an integer, such as 2000, to show only the first 2000 episodes.
MAX_EPISODES = 1000

# skip episodes
SKIP_FIRST_EPISODES = 5

SHOW_PLOTS = True
OUTPUT_NAME = "sac_vs_mbrl_comparison.pdf"

SAC_LABEL = "SAC"
MBRL_LABEL = "Proposed"
SAC_COLOR = "#0072B2"
MBRL_COLOR = "#D55E00"
SAC_LINESTYLE = "-"
MBRL_LINESTYLE = "--"              # Dashed so SAC remains visible at 100%


# =============================================================================
# DATA PREPARATION
# =============================================================================

def require_columns(data, required_columns, file_label):
    """Stop with a clear message when a configured column is missing."""
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        available = ", ".join(data.columns)
        missing_text = ", ".join(missing)
        raise KeyError(
            f"{file_label} is missing: {missing_text}.\n"
            f"Available columns: {available}"
        )


def as_boolean(series):
    """Convert True/False, 1/0, or text flags into Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)

    true_values = {"true", "1", "yes", "y"}
    return series.fillna("").astype(str).str.strip().str.lower().isin(true_values)


def load_sac_episode_data(file_path):
    """Load SAC's one-row-per-episode monitor file."""
    data = pd.read_csv(file_path, comment="#")

    require_columns(
        data,
        [SAC_TIME_COLUMN, SAC_SUCCESS_COLUMN, SAC_EFFORT_COLUMN],
        "SAC CSV",
    )

    episode_data = pd.DataFrame(
        {
            "episode": np.arange(len(data)),
            "time": pd.to_numeric(data[SAC_TIME_COLUMN], errors="coerce"),
            "success": as_boolean(data[SAC_SUCCESS_COLUMN]).astype(float),
            "effort": pd.to_numeric(data[SAC_EFFORT_COLUMN], errors="coerce"),
        }
    )

    return episode_data.dropna(subset=["time", "success", "effort"])


def mbrl_episode_success(episode_rows):
    """Determine whether one MBRL episode reached its goal."""
    if MBRL_SUCCESS_COLUMN is not None:
        return float(as_boolean(episode_rows[MBRL_SUCCESS_COLUMN]).any())

    position = pd.to_numeric(
        episode_rows[MBRL_POSITION_COLUMN], errors="coerce"
    ).dropna()

    if position.empty:
        return 0.0

    reached = position.max() >= MBRL_SUCCESS_POSITION
    return float(reached)


def load_mbrl_episode_data(file_path):
    """Aggregate MBRL's timestep data into one row per episode."""
    data = pd.read_csv(file_path, comment="#")

    required_columns = [
        MBRL_EPISODE_COLUMN,
        MBRL_TIME_COLUMN,
        MBRL_EFFORT_COLUMN,
    ]

    if MBRL_SUCCESS_COLUMN is None:
        required_columns.append(MBRL_POSITION_COLUMN)
    else:
        required_columns.append(MBRL_SUCCESS_COLUMN)

    require_columns(data, required_columns, "MBRL CSV")
    data = data.sort_values([MBRL_EPISODE_COLUMN, MBRL_TIME_COLUMN])

    episode_rows = []

    for episode, group in data.groupby(MBRL_EPISODE_COLUMN, sort=True):
        episode_time = pd.to_numeric(group[MBRL_TIME_COLUMN], errors="coerce")
        effort = pd.to_numeric(group[MBRL_EFFORT_COLUMN], errors="coerce")

        # MBRL time resets near zero at the beginning of each episode.
        duration = episode_time.max()
        effort_rms = np.sqrt(np.mean(np.square(effort.dropna())))

        episode_rows.append(
            {
                "episode": episode,
                "time": duration,
                "success": mbrl_episode_success(group),
                "effort": effort_rms,
            }
        )

    episode_data = pd.DataFrame(episode_rows)
    return episode_data.dropna(subset=["time", "success", "effort"])


def limit_episodes(data):
    """Skip initial episodes and optionally limit the displayed data."""
    data = data.iloc[SKIP_FIRST_EPISODES:].copy()

    if MAX_EPISODES is not None:
        data = data.iloc[:MAX_EPISODES].copy()

    return data


def rolling_statistics(data, column):
    """Return the rolling mean and rolling standard-deviation band."""
    rolling = data[column].rolling(ROLLING_WINDOW, min_periods=1)
    mean = rolling.mean()
    standard_deviation = rolling.std(ddof=0).fillna(0.0)
    lower = mean - ROLLING_BAND_STD * standard_deviation
    upper = mean + ROLLING_BAND_STD * standard_deviation
    return mean, lower, upper


def rolling_success_statistics(data):
    """Return rolling success rate and its 95% Wilson confidence interval."""
    rolling = data["success"].rolling(ROLLING_WINDOW, min_periods=1)
    successes = rolling.sum()
    sample_count = rolling.count()
    proportion = successes / sample_count

    z_squared = SUCCESS_CONFIDENCE_Z**2
    denominator = 1.0 + z_squared / sample_count
    center = (
        proportion + z_squared / (2.0 * sample_count)
    ) / denominator
    margin = (
        SUCCESS_CONFIDENCE_Z
        * np.sqrt(
            proportion * (1.0 - proportion) / sample_count
            + z_squared / (4.0 * sample_count**2)
        )
        / denominator
    )

    mean = 100.0 * proportion
    lower = 100.0 * np.clip(center - margin, 0.0, 1.0)
    upper = 100.0 * np.clip(center + margin, 0.0, 1.0)
    return mean, lower, upper


# =============================================================================
# PUBLICATION-STYLE PLOTTING
# =============================================================================

def format_episode_tick(value, _position):
    """Format large episode numbers as 5k, 20k, or 1M."""
    absolute_value = abs(value)

    if absolute_value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:g}M"
    if absolute_value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:g}k"
    return f"{value:g}"


def format_axes(axis, title, y_label):
    """Match the clean gray-background style of the reference figure."""
    axis.set_title(title, fontsize=19, pad=12)
    axis.set_xlabel("Episode")
    axis.set_ylabel(y_label)
    axis.set_axisbelow(True)
    axis.set_facecolor("#FBFAFA") #F0EFEF
    axis.grid(
        True,
        color="#E9E9E9",
        linewidth=1.0,
        #alpha=0.10,
    )
    axis.tick_params(axis="both", which="both", length=0, labelsize=12)
    axis.xaxis.set_major_formatter(plt.FuncFormatter(format_episode_tick))

    for spine in axis.spines.values():
        spine.set_visible(False)


def plot_method(
    axis,
    data,
    column,
    label,
    color,
    linestyle,
    percentage=False,
):
    """Draw one rolling mean curve and its translucent variability band."""
    if percentage:
        mean, lower, upper = rolling_success_statistics(data)
        band_alpha = SUCCESS_BAND_ALPHA
    else:
        mean, lower, upper = rolling_statistics(data, column)
        lower = np.maximum(lower, 0.0)
        band_alpha = BAND_ALPHA

    axis.fill_between(
        data["episode"],
        lower,
        upper,
        color=color,
        alpha=band_alpha,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        data["episode"],
        mean,
        color=color,
        linewidth=2.7,
        linestyle=linestyle,
        label=label,
        zorder=3,
    )


def create_comparison_figure(sac, mbrl):
    """Create the three side-by-side panels in one figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 13,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    figure.patch.set_facecolor("white")

    panels = [
        (axes[0], "time", "Episode Duration", "Episode duration (s)", False),
        (axes[1], "success", "Success Rate", "Success rate (%)", True),
        (axes[2], "effort", "Control Effort", "RMS control effort", False),
    ]

    for axis, column, title, y_label, percentage in panels:
        plot_method(
            axis,
            sac,
            column,
            SAC_LABEL,
            SAC_COLOR,
            SAC_LINESTYLE,
            percentage,
        )
        plot_method(
            axis,
            mbrl,
            column,
            MBRL_LABEL,
            MBRL_COLOR,
            MBRL_LINESTYLE,
            percentage,
        )
        format_axes(axis, title, y_label)

    # Leave space above 100% so converged curves are not clipped by the border.
    axes[1].set_ylim(0, 105)
    axes[1].set_yticks([0, 25, 50, 75, 100])

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    figure.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.2)
    figure.savefig(
        OUTPUT_DIRECTORY / OUTPUT_NAME,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    return figure


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    sac = limit_episodes(load_sac_episode_data(SAC_FILE))
    mbrl = limit_episodes(load_mbrl_episode_data(MBRL_FILE))

    create_comparison_figure(sac, mbrl)

    print(f"Saved figure: {(OUTPUT_DIRECTORY / OUTPUT_NAME).resolve()}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()