"""Compare PILCO, the proposed controller, MBPO, SAC, and PPO.

The main figure follows the visual structure of the MBPO paper: one row of
clean learning curves, a shared linear episode axis, and light shaded bands.
Each input currently contains one run, so the bands show the rolling standard
error among nearby episodes. They are not uncertainty across random seeds.

Edit only the CONFIGURATION section when filenames or columns change.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIRECTORY = Path(__file__).resolve().parent
SAC_FILE = RESULTS_DIRECTORY / "sac_monitor/monitor.csv"
PPO_MONITOR_DIRECTORY = RESULTS_DIRECTORY / "ppo_monitor"
MBRL_FILE = RESULTS_DIRECTORY / "obstacle_mujoco_1000.csv"
PILCO_FILE = RESULTS_DIRECTORY / "pilco_episodes.csv"
MBPO_FILE = RESULTS_DIRECTORY / "mbpo/episodes.csv"
OUTPUT_DIRECTORY = RESULTS_DIRECTORY / "images"

# SAC: one row per completed training episode.
SAC_TIME_COLUMN = "sim_time"
SAC_SUCCESS_COLUMN = "reached"
SAC_EFFORT_COLUMN = "effort_rms"
SAC_EPISODE_STEPS_COLUMN = "l"

# PPO: one Monitor file per parallel environment. Stable-Baselines3 counts all
# parallel transitions, so a worker's local steps are multiplied by the number
# of workers in the newest run.
PPO_TIME_COLUMN = "sim_time"
PPO_SUCCESS_COLUMN = "reached"
PPO_EFFORT_COLUMN = "effort_rms"
PPO_EPISODE_STEPS_COLUMN = "l"
PPO_WALL_TIME_COLUMN = "t"
PPO_RUN_START_TOLERANCE_SECONDS = 60.0

# PILCO: buffer_size is the cumulative number of real transitions collected.
# Random collection/validation trials train the model but do not show the
# learned policy, so the performance curve uses policy rows only.
PILCO_TIME_COLUMN = "sim_time"
PILCO_SUCCESS_COLUMN = "reached"
PILCO_EFFORT_COLUMN = "effort_rms"
PILCO_EPISODE_COLUMN = "episode"
PILCO_REAL_STEPS_COLUMN = "buffer_size"
PILCO_MODE_COLUMN = "mode"
PILCO_POLICY_MODE = "policy"
PILCO_POLICY_ONLY = True

# MBPO: real_timesteps excludes synthetic model rollouts. The loader detects an
# appended/restarted run whenever this column decreases and keeps the newest
# uninterrupted segment.
MBPO_TIME_COLUMN = "sim_time"
MBPO_SUCCESS_COLUMN = "reached"
MBPO_EFFORT_COLUMN = "effort_rms"
MBPO_REAL_STEPS_COLUMN = "real_timesteps"

# Proposed MBRL controller: the CSV logs every MuJoCo integration step, while a
# control command is held for 0.05 s. Fifty 0.001-s rows equal one transition.
MBRL_EPISODE_COLUMN = "episode"
MBRL_TIME_COLUMN = "time"
MBRL_POSITION_COLUMN = "x"
MBRL_EFFORT_COLUMN = "ctrl_cmd"  # Use "tau_cmd" if that matches RL action units.
MBRL_SUCCESS_COLUMN = None       # Otherwise set to a Boolean success column.
MBRL_SUCCESS_POSITION = 9.85
MBRL_SIMULATION_DT = 0.001
MBRL_CONTROL_DT = 0.05

# Choose "episodes" for improvement per rollout/reset, or "timesteps" for
# sample efficiency per real control interaction. The plotting code supports
# both representations so switching later requires changing only this value.
X_AXIS_MODE = "episodes"
X_AXIS_SETTINGS = {
    "episodes": {
        "column": "episode",
        "label": "episode",
        "main_limit": 1000,
        # "main_ticks": [0, 100, 200, 300, 400, 500],
        "main_ticks": [0, 250, 500, 750, 1000],
        "main_bin_width": 25,
        "inset_limit": 100,
        "inset_ticks": [0, 50, 100],
        "inset_bin_width": 10,
        "inset_title": "first 100 episodes",
    },
    "timesteps": {
        "column": "training_steps",
        "label": "real environment steps",
        "main_limit": 300_000,
        "main_ticks": [0, 100_000, 200_000, 300_000],
        "main_bin_width": 5_000,
        "inset_limit": 5_000,
        "inset_ticks": [0, 2_500, 5_000],
        "inset_bin_width": 1_000,
        "inset_title": "first 5k steps",
    },
}

# MBPO-paper-style smoothing. The band is one standard error across episodes
# inside the rolling window; it is not an across-seed uncertainty interval.
ROLLING_EPISODES = 200
ROLLING_MIN_PERIODS = 10
ROLLING_BAND_STANDARD_ERRORS = 0.1
BAND_STD_SCALE = 0.25

# Keep all recorded episodes by default. Limits do not renumber training steps.
SKIP_FIRST_EPISODES = 0
MAX_EPISODES = None

BAND_ALPHA = 0.15
SHOW_PLOTS = True
OUTPUT_NAME = "methods_comparison.png"
OUTPUT_PDF_NAME = "methods_comparison.pdf"
PER_METHOD_OUTPUT_NAME = "methods_individual_comparison.png"

# Choose "all" for duration, success rate, and control effort.
# Choose "selected" to plot only the metrics listed in SELECTED_METRICS.
# Available metrics: "time", "success", "effort".
PLOT_MODE = "selected"
SELECTED_METRICS = ("time", "success")
SELECTED_LEGEND_AXIS = 0
SELECTED_LEGEND_LOCATION = "upper right"

# Colors and solid strokes follow the visual language of the MBPO paper.
MBPO_LABEL, MBPO_COLOR = "MBPO", "#1F77B4"
SAC_LABEL, SAC_COLOR = "SAC", "#2CA02C"
PPO_LABEL, PPO_COLOR = "PPO", "#9467BD"
PILCO_LABEL, PILCO_COLOR = "PILCO", "#E377C2"
MBRL_LABEL, MBRL_COLOR = "Proposed", "#FF7F0E"


# =============================================================================
# DATA PREPARATION
# =============================================================================

def require_columns(data, required_columns, file_label):
    """Stop with a clear message when a configured column is missing."""
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        available = ", ".join(data.columns)
        raise KeyError(
            f"{file_label} is missing: {', '.join(missing)}.\n"
            f"Available columns: {available}"
        )


def as_boolean(series):
    """Convert True/False, 1/0, or common text flags to Boolean values."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).ne(0)

    true_values = {"true", "1", "yes", "y"}
    return series.fillna("").astype(str).str.strip().str.lower().isin(true_values)


def build_episode_frame(
    data,
    time_column,
    success_column,
    effort_column,
    file_label,
    *,
    episode_column=None,
    episode_steps_column=None,
    cumulative_steps_column=None,
):
    """Normalize one-row-per-episode logs to the plotting schema."""
    required = [time_column, success_column, effort_column]
    if episode_column is not None:
        required.append(episode_column)
    if episode_steps_column is not None:
        required.append(episode_steps_column)
    if cumulative_steps_column is not None:
        required.append(cumulative_steps_column)
    require_columns(data, required, file_label)

    if (episode_steps_column is None) == (cumulative_steps_column is None):
        raise ValueError(
            "Configure exactly one episode-step or cumulative-step column for "
            f"{file_label}."
        )

    if cumulative_steps_column is not None:
        training_steps = pd.to_numeric(
            data[cumulative_steps_column], errors="coerce"
        )
    else:
        per_episode_steps = pd.to_numeric(
            data[episode_steps_column], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        training_steps = per_episode_steps.cumsum()

    episode = (
        np.arange(len(data), dtype=float)
        if episode_column is None
        else pd.to_numeric(data[episode_column], errors="coerce")
    )
    episode_data = pd.DataFrame(
        {
            "episode": episode,
            "training_steps": training_steps,
            "time": pd.to_numeric(data[time_column], errors="coerce"),
            "success": as_boolean(data[success_column]).astype(float),
            "effort": pd.to_numeric(data[effort_column], errors="coerce"),
        }
    )
    episode_data = episode_data.dropna(
        subset=["episode", "training_steps", "time", "success", "effort"]
    )
    episode_data = episode_data[episode_data["training_steps"] >= 0]
    return episode_data.sort_values(
        ["training_steps", "episode"], kind="stable"
    ).reset_index(drop=True)



def load_sac_episode_data(file_path):
    """Load SAC and derive its exact cumulative real environment steps."""
    data = pd.read_csv(file_path, comment="#")
    return build_episode_frame(
        data,
        SAC_TIME_COLUMN,
        SAC_SUCCESS_COLUMN,
        SAC_EFFORT_COLUMN,
        "SAC CSV",
        episode_steps_column=SAC_EPISODE_STEPS_COLUMN,
    )




def ppo_monitor_start_time(file_path):
    """Read one Stable-Baselines3 Monitor file's JSON start-time header."""
    with file_path.open("r", encoding="utf-8") as monitor_file:
        metadata_line = monitor_file.readline().strip()

    if not metadata_line.startswith("#"):
        raise ValueError(f"PPO Monitor file has no JSON header: {file_path}")

    try:
        metadata = json.loads(metadata_line[1:])
        return float(metadata["t_start"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"PPO Monitor file has an invalid t_start header: {file_path}"
        ) from error



def load_ppo_episode_data(monitor_directory):
    """Load the newest PPO worker cohort on the exact SB3 timestep axis."""
    monitor_files = sorted(monitor_directory.glob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No PPO Monitor files found in: {monitor_directory}")

    starts = {
        file_path: ppo_monitor_start_time(file_path)
        for file_path in monitor_files
    }
    newest_start = max(starts.values())
    newest_run_files = [
        file_path
        for file_path in monitor_files
        if newest_start - starts[file_path] <= PPO_RUN_START_TOLERANCE_SECONDS
    ]
    number_of_environments = len(newest_run_files)

    worker_frames = []
    for file_path in newest_run_files:
        data = pd.read_csv(file_path, comment="#")
        require_columns(
            data,
            [
                PPO_WALL_TIME_COLUMN,
                PPO_TIME_COLUMN,
                PPO_SUCCESS_COLUMN,
                PPO_EFFORT_COLUMN,
                PPO_EPISODE_STEPS_COLUMN,
            ],
            f"PPO Monitor CSV ({file_path.name})",
        )

        try:
            worker_number = int(file_path.name.split(".", maxsplit=1)[0])
        except ValueError:
            worker_number = len(worker_frames)

        worker_data = data.copy()
        episode_steps = pd.to_numeric(
            worker_data[PPO_EPISODE_STEPS_COLUMN], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
        worker_data["_training_steps"] = (
            number_of_environments * episode_steps.cumsum()
        )
        worker_data["_episode_end_wall_time"] = (
            starts[file_path]
            + pd.to_numeric(worker_data[PPO_WALL_TIME_COLUMN], errors="coerce")
        )
        worker_data["_worker"] = worker_number
        worker_data["_worker_row"] = np.arange(len(worker_data))
        worker_frames.append(worker_data)

    combined = pd.concat(worker_frames, ignore_index=True)
    combined = combined.dropna(
        subset=["_training_steps", "_episode_end_wall_time"]
    )
    combined = combined.sort_values(
        ["_training_steps", "_episode_end_wall_time", "_worker", "_worker_row"],
        kind="stable",
    ).reset_index(drop=True)

    episode_data = pd.DataFrame(
        {
            "episode": np.arange(len(combined), dtype=float),
            "training_steps": combined["_training_steps"],
            "time": pd.to_numeric(combined[PPO_TIME_COLUMN], errors="coerce"),
            "success": as_boolean(combined[PPO_SUCCESS_COLUMN]).astype(float),
            "effort": pd.to_numeric(
                combined[PPO_EFFORT_COLUMN], errors="coerce"
            ),
        }
    )
    return episode_data.dropna(
        subset=["training_steps", "time", "success", "effort"]
    ).reset_index(drop=True)


def load_pilco_episode_data(file_path):
    """Load PILCO policy trials while preserving all real collected steps."""
    data = pd.read_csv(file_path, comment="#")
    require_columns(data, [PILCO_REAL_STEPS_COLUMN], "PILCO CSV")

    if PILCO_POLICY_ONLY:
        require_columns(data, [PILCO_MODE_COLUMN], "PILCO CSV")
        mode = data[PILCO_MODE_COLUMN].astype(str).str.strip().str.lower()
        data = data.loc[mode.eq(PILCO_POLICY_MODE.lower())].copy()
        if data.empty:
            raise ValueError(
                f"PILCO CSV contains no {PILCO_POLICY_MODE!r} rows in "
                f"{PILCO_MODE_COLUMN!r}."
            )

    return build_episode_frame(
        data,
        PILCO_TIME_COLUMN,
        PILCO_SUCCESS_COLUMN,
        PILCO_EFFORT_COLUMN,
        "PILCO CSV",
        episode_column=PILCO_EPISODE_COLUMN,
        cumulative_steps_column=PILCO_REAL_STEPS_COLUMN,
    )


def load_mbpo_episode_data(file_path):
    """Load the newest uninterrupted MBPO run on real environment steps."""
    data = pd.read_csv(file_path, comment="#")
    require_columns(data, [MBPO_REAL_STEPS_COLUMN], "MBPO CSV")

    real_steps = pd.to_numeric(data[MBPO_REAL_STEPS_COLUMN], errors="coerce")
    reset_positions = np.flatnonzero(real_steps.diff().lt(0).to_numpy())
    discarded_rows = 0
    if reset_positions.size:
        discarded_rows = int(reset_positions[-1])
        data = data.iloc[discarded_rows:].copy()

    episode_data = build_episode_frame(
        data,
        MBPO_TIME_COLUMN,
        MBPO_SUCCESS_COLUMN,
        MBPO_EFFORT_COLUMN,
        "MBPO CSV",
        cumulative_steps_column=MBPO_REAL_STEPS_COLUMN,
    )
    episode_data.attrs["discarded_rows"] = discarded_rows
    return episode_data


def load_mbrl_episode_data(file_path):
    """Aggregate the proposed controller's physics rows by episode."""
    data = pd.read_csv(file_path, comment="#")
    required = [MBRL_EPISODE_COLUMN, MBRL_TIME_COLUMN, MBRL_EFFORT_COLUMN]
    required.append(
        MBRL_POSITION_COLUMN
        if MBRL_SUCCESS_COLUMN is None
        else MBRL_SUCCESS_COLUMN
    )
    require_columns(data, required, "Proposed-controller CSV")

    prepared = pd.DataFrame(
        {
            "episode": pd.to_numeric(
                data[MBRL_EPISODE_COLUMN], errors="coerce"
            ),
            "time": pd.to_numeric(data[MBRL_TIME_COLUMN], errors="coerce"),
            "effort": pd.to_numeric(data[MBRL_EFFORT_COLUMN], errors="coerce"),
        }
    )
    if MBRL_SUCCESS_COLUMN is None:
        prepared["success_value"] = pd.to_numeric(
            data[MBRL_POSITION_COLUMN], errors="coerce"
        )
    else:
        prepared["success_value"] = as_boolean(
            data[MBRL_SUCCESS_COLUMN]
        ).astype(float)

    prepared = prepared.dropna(subset=["episode", "time", "effort"])
    prepared["effort_squared"] = np.square(prepared["effort"])
    grouped = prepared.groupby("episode", sort=True)
    summary = grouped.agg(
        time=("time", "max"),
        mean_effort_squared=("effort_squared", "mean"),
        physics_samples=("time", "size"),
        success_value=("success_value", "max"),
    ).reset_index()
    summary["effort"] = np.sqrt(summary["mean_effort_squared"])
    if MBRL_SUCCESS_COLUMN is None:
        summary["success"] = (
            summary["success_value"] >= MBRL_SUCCESS_POSITION
        ).astype(float)
    else:
        summary["success"] = summary["success_value"].fillna(0.0).ne(0).astype(float)

    physics_steps_per_control = int(round(MBRL_CONTROL_DT / MBRL_SIMULATION_DT))
    if physics_steps_per_control <= 0:
        raise ValueError("MBRL_CONTROL_DT must be at least MBRL_SIMULATION_DT.")
    summary["episode_control_steps"] = np.maximum(
        1,
        np.ceil(summary["physics_samples"] / physics_steps_per_control),
    )
    summary["training_steps"] = summary["episode_control_steps"].cumsum()

    return summary[
        ["episode", "training_steps", "time", "success", "effort"]
    ].dropna().reset_index(drop=True)


def limit_episodes(data):
    """Apply optional row limits without changing cumulative step coordinates."""
    limited = data.iloc[SKIP_FIRST_EPISODES:].copy()
    if MAX_EPISODES is not None:
        limited = limited.iloc[:MAX_EPISODES].copy()
    return limited.reset_index(drop=True)


# =============================================================================
# PUBLICATION-STYLE PLOTTING
# =============================================================================

def format_step_tick(value, _position):
    """Format real environment steps as 5k, 300k, or 3M."""
    if abs(value) < 1e-9:
        return "0"
    absolute_value = abs(value)
    if absolute_value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if absolute_value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def step_window_statistics(data, column, step_window, percentage=False):
    """Compute trailing summaries over a fixed span of real training steps."""
    x = pd.to_numeric(data["training_steps"], errors="coerce").to_numpy()
    values = pd.to_numeric(data[column], errors="coerce").to_numpy()
    time_index = pd.to_timedelta(x, unit="s")
    series = pd.Series(values, index=time_index)
    rolling = series.rolling(
        pd.Timedelta(max(float(step_window), 1.0), unit="s"),
        min_periods=1,
    )

    if percentage:
        return 100.0 * rolling.mean().to_numpy(), None, None

    center = rolling.median().to_numpy()
    lower = np.maximum(rolling.quantile(0.25).to_numpy(), 0.0)
    upper = rolling.quantile(0.75).to_numpy()
    return center, lower, upper


def style_axis(axis, title, y_label, *, x_limit=None, x_ticks=None):
    """Apply the white-figure/light-gray-panel styling of the reference."""
    axis.set_title(title, fontsize=15, pad=9)
    axis.set_xlabel("real environment steps", fontsize=10.5)
    axis.set_ylabel(y_label, fontsize=11)
    axis.set_facecolor("#FBFAFA") # EFEFEF
    axis.set_axisbelow(True)
    axis.grid(True, color="#E9E9E9", linewidth=1.2)
    axis.tick_params(axis="both", which="both", length=0, labelsize=9.5, pad=4)
    axis.xaxis.set_major_formatter(FuncFormatter(format_step_tick))
    for spine in axis.spines.values():
        spine.set_visible(False)

    if x_limit is not None:
        axis.set_xlim(0, x_limit)
    if x_ticks is not None:
        axis.set_xticks(x_ticks)
    else:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))


def configure_metric_axis(axis, metric):
    """Keep each metric on a consistent scale in every panel."""
    if metric == "time":
        axis.set_ylim(0, 21)
        axis.set_yticks([0, 5, 10, 15, 20])
    elif metric == "success":
        axis.set_ylim(0, 103)
        axis.set_yticks([0, 25, 50, 75, 100])
    elif metric == "effort":
        axis.set_ylim(0, 8)
        axis.set_yticks([0, 2, 4, 6, 8])


def configure_main_metric_axis(axis, metric):
    """Use compact, shared ranges that keep method differences legible."""
    if metric == "time":
        axis.set_ylim(0, 20.5)
        axis.set_yticks([0, 5, 10, 15, 20])
    elif metric == "success":
        axis.set_ylim(0, 105)
        axis.set_yticks([0, 25, 50, 75, 100])
    elif metric == "effort":
        axis.set_ylim(3.4, 7.6)
        axis.set_yticks([4, 5, 6, 7])


def plot_method(
    axis,
    data,
    metric,
    label,
    color,
    step_window,
    *,
    step_limit=None,
    percentage=False,
    linewidth=2.25,
):
    """Draw one within-run training curve and an optional local IQR band."""
    visible = data
    if step_limit is not None:
        visible = data[data["training_steps"] <= step_limit]
    if visible.empty:
        return

    visible = visible.sort_values("training_steps", kind="stable")
    x = visible["training_steps"].to_numpy()
    center, lower, upper = step_window_statistics(
        visible, metric, step_window, percentage=percentage
    )

    if lower is not None:
        axis.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    axis.plot(
        x,
        center,
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="round",
        label=label,
        zorder=3,
    )

    # A dot means the recorded run ends here. Curves crossing the panel edge
    # continue beyond the displayed comparison budget.
    recorded_end = float(data["training_steps"].max())
    if step_limit is None or recorded_end <= step_limit:
        axis.plot(
            x[-1],
            center[-1],
            marker="o",
            markersize=4.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.75,
            linestyle="none",
            zorder=5,
        )


# def rolling_episode_statistics(data, metric, label, *, percentage=False, successful_only=False,):
#     """Return a rolling mean and one-standard-error episode band."""
#     values = pd.to_numeric(data[metric], errors="coerce")
#     if successful_only:
#         values = values.where(data["success"] >= 0.5)

#     rolling = values.rolling(
#         ROLLING_EPISODES,
#         min_periods=ROLLING_MIN_PERIODS,
#     )
#     mean = rolling.mean()
#     count = rolling.count().clip(lower=1)
#     standard_error = rolling.std(ddof=1).fillna(0.0) / np.sqrt(count)
#     half_width = ROLLING_BAND_STANDARD_ERRORS * standard_error

#     if percentage:
#         mean = 100.0 * mean
#         half_width = 100.0 * half_width
#         lower = (mean - half_width).clip(lower=0.0, upper=100.0)
#         upper = (mean + half_width).clip(lower=0.0, upper=100.0)
#     else:
#         lower = (mean - half_width).clip(lower=0.0)
#         upper = mean + half_width
#     return mean.to_numpy(), lower.to_numpy(), upper.to_numpy()




def rolling_episode_statistics(data, metric, label, *, percentage=False, successful_only=False):
    values = pd.to_numeric(data[metric], errors="coerce")

    if successful_only:
        values = values.where(data["success"] >= 0.5)

    rolling = values.rolling(
        ROLLING_EPISODES,
        min_periods=ROLLING_MIN_PERIODS,
    )

    mean = rolling.mean()
    std = rolling.std(ddof=1).fillna(0.0)

    if percentage:
        mean = 100.0 * mean
        std = 100.0 * std
        lower = (mean - BAND_STD_SCALE * std).clip(0.0, 100.0)
        upper = (mean + BAND_STD_SCALE * std).clip(0.0, 100.0)
    else:
        lower = (mean - BAND_STD_SCALE * std).clip(lower=0.0)
        upper = mean + BAND_STD_SCALE * std

    return mean.to_numpy(), lower.to_numpy(), upper.to_numpy()







def plot_paper_method(
    axis,
    data,
    metric,
    label,
    color,
    linewidth,
    x_column,
    x_limit,
    *,
    percentage=False,
    successful_only=False,
):
    visible = data[
        (data[x_column] >= 0)
        & (data[x_column] <= x_limit)
    ].copy()

    if visible.empty:
        return

    visible = visible.sort_values("episode", kind="stable").reset_index(drop=True)

    # Define x here
    x = pd.to_numeric(visible[x_column], errors="coerce").to_numpy()

    mean, lower, upper = rolling_episode_statistics(
        visible,
        metric,
        label,
        percentage=percentage,
        successful_only=successful_only,
    )

    if lower is not None:
        axis.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        axis.plot(x, lower, color=color, alpha=0.30, linewidth=1.9, zorder=2)
        axis.plot(x, upper, color=color, alpha=0.30, linewidth=1.9, zorder=2)

    axis.plot(
        x,
        mean,
        color=color,
        linewidth=linewidth,
        linestyle="-",
        solid_capstyle="round",
        zorder=3,
    )







# def plot_paper_method(
#     axis,
#     data,
#     metric,
#     label,
#     color,
#     linewidth,
#     x_column,
#     x_limit,
#     *,
#     percentage=False,
#     successful_only=False,
# ):
#     """Draw one smooth solid curve and a light MBPO-style episode band."""
#     visible = data[(data[x_column] >= 0) & (data[x_column] <= x_limit)].copy()
    
#     if visible.empty:
#         return
    
#     visible = visible.sort_values("episode", kind="stable").reset_index(drop=True)
#     mean, lower, upper = rolling_episode_statistics(
#         visible,
#         metric,
#         label,
#         percentage=percentage,
#         successful_only=successful_only,
#     )
#     x = pd.to_numeric(visible[x_column], errors="coerce").to_numpy()
#     axis.fill_between(
#         x,
#         lower,
#         upper,
#         color=color,
#         alpha=BAND_ALPHA,
#         linewidth=0,
#         zorder=2,
#     )
#     axis.plot(
#         x,
#         mean,
#         color=color,
#         linewidth=linewidth,
#         linestyle="-",
#         solid_capstyle="round",
#         # Draw Proposed beneath coincident baselines so their convergence at
#         # 100% remains visible instead of being covered by the orange curve.
#         zorder=2.4 if label == MBRL_LABEL else 3,
#     )


def method_specs(sac, mbrl, pilco, mbpo, ppo):
    """Return methods in the same visual order as the MBPO reference."""
    return [
        (mbpo, MBPO_LABEL, MBPO_COLOR, 2.35),
        (sac, SAC_LABEL, SAC_COLOR, 2.25),
        (ppo, PPO_LABEL, PPO_COLOR, 2.25),
        (pilco, PILCO_LABEL, PILCO_COLOR, 2.25),
        (mbrl, MBRL_LABEL, MBRL_COLOR, 2.70),
    ]


def metric_specs():
    return [
        ("time", "Episode duration ↓", "duration (s)", False),
        ("success", "Success rate ↑", "success rate (%)", True),
        ("effort", "Control effort ↓", "RMS torque (N·m)", False),
    ]


def main_metric_specs():
    """Metrics used in the paper comparison."""
    metrics = [
        ("time", "Episode Duration", "episode duration (s)", False),
        ("success", "Success Rate", "success rate (%)", True),
        ("effort", "Control Effort", "RMS torque (N·m)", False),
    ]
    if PLOT_MODE == "all":
        return metrics
    if PLOT_MODE != "selected":
        raise ValueError('PLOT_MODE must be either "all" or "selected".')

    metric_by_name = {metric[0]: metric for metric in metrics}
    unknown_metrics = [metric for metric in SELECTED_METRICS if metric not in metric_by_name]
    if unknown_metrics:
        choices = ", ".join(metric_by_name)
        raise ValueError(
            f"Unknown SELECTED_METRICS entries: {unknown_metrics}. "
            f"Available metrics: {choices}."
        )
    return [metric_by_name[metric] for metric in SELECTED_METRICS]


def legend_handles(pilco_trials=None):
    pilco_legend_label = (
        PILCO_LABEL
        if pilco_trials is None
        else f"{PILCO_LABEL}"
    )
    return [
        Line2D([0], [0], color=color, lw=linewidth, label=label)
        for label, color, linewidth in [
            (SAC_LABEL, SAC_COLOR, 2.25),
            (PPO_LABEL, PPO_COLOR, 2.25),
            (pilco_legend_label, PILCO_COLOR, 2.25),
            (MBPO_LABEL, MBPO_COLOR, 2.35),
            (MBRL_LABEL, MBRL_COLOR, 2.70),
        ]
    ]


def create_comparison_figure(sac, mbrl, pilco, mbpo, ppo):
    """Create the single-row, common-budget paper comparison."""
    if X_AXIS_MODE not in X_AXIS_SETTINGS:
        choices = ", ".join(sorted(X_AXIS_SETTINGS))
        raise ValueError(f"X_AXIS_MODE must be one of: {choices}")
    axis_settings = X_AXIS_SETTINGS[X_AXIS_MODE]

    metrics = main_metric_specs()
    figure_width = 4.0 * len(metrics)
    figure, axes = plt.subplots(1, len(metrics), figsize=(figure_width, 4.0), squeeze=False)
    axes = axes[0]
    figure.patch.set_facecolor("white")
    methods = method_specs(sac, mbrl, pilco, mbpo, ppo)

    for axis, (metric, title, y_label, percentage) in zip(axes, metrics):
        for data, label, color, linewidth in methods:
            plot_paper_method(
                axis,
                data,
                metric,
                label,
                color,
                linewidth,
                axis_settings["column"],
                axis_settings["main_limit"],
                percentage=percentage,
                successful_only=False,
            )

        style_axis(
            axis,
            title,
            y_label,
            x_limit=axis_settings["main_limit"],
            x_ticks=axis_settings["main_ticks"],
        )
        axis.set_xlabel(axis_settings["label"])
        configure_main_metric_axis(axis, metric)

    handles = legend_handles(len(pilco))
    if PLOT_MODE == "selected":
        legend_axis_index = max(
            0,
            min(int(SELECTED_LEGEND_AXIS), len(axes) - 1),
        )
        axes[legend_axis_index].legend(
            handles=handles,
            loc=SELECTED_LEGEND_LOCATION,
            ncol=1,
            frameon=True,
            framealpha=0.86,
            facecolor="white",
            edgecolor="none",
            fontsize=8.8,
            handlelength=2.2,
        )
    else:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=5,
            frameon=False,
            fontsize=10.5,
            handlelength=2.8,
            columnspacing=1.8,
        )
    figure.subplots_adjust(
        left=0.065 if len(metrics) == 3 else 0.085,
        right=0.992,
        top=0.90,
        bottom=0.235 if PLOT_MODE == "all" else 0.15,
        wspace=0.30 if len(metrics) == 3 else 0.24,
    )
    figure.savefig(
        OUTPUT_DIRECTORY / OUTPUT_NAME,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIRECTORY / OUTPUT_PDF_NAME,
        bbox_inches="tight",
        facecolor="white",
    )
    return figure


def create_per_method_figure(sac, mbrl, pilco, mbpo, ppo):
    """Show each algorithm over its complete recorded training budget."""
    methods = method_specs(sac, mbrl, pilco, mbpo, ppo)
    metrics = metric_specs()
    figure, axes = plt.subplots(
        len(methods),
        len(metrics),
        figsize=(13.2, 14.2),
        sharex=False,
        sharey="col",
        squeeze=False,
    )
    figure.patch.set_facecolor("white")

    for row, (data, label, color, linewidth) in enumerate(methods):
        recorded_end = float(data["training_steps"].max())
        smoothing_steps = max(250.0, 0.04 * recorded_end)
        x_limit = max(1.0, 1.025 * recorded_end)

        for column, (metric, title, y_label, percentage) in enumerate(metrics):
            axis = axes[row, column]
            plot_method(
                axis,
                data,
                metric,
                label,
                color,
                smoothing_steps,
                percentage=percentage,
                linewidth=linewidth,
            )
            style_axis(
                axis,
                title if row == 0 else "",
                y_label,
                x_limit=x_limit,
            )
            configure_metric_axis(axis, metric)

        axes[row, 0].text(
            0.035,
            0.88,
            label,
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
            },
            zorder=8,
        )

    figure.legend(
        handles=legend_handles(len(pilco)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.046),
        ncol=5,
        frameon=False,
        fontsize=11,
        handlelength=3.0,
        columnspacing=2.0,
    )
    figure.text(
        0.5,
        0.012,
        "Each row uses that method's complete recorded real-step range; y-scales "
        "remain fixed by metric. Shading is within-run rolling IQR.",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color="#555555",
    )
    figure.tight_layout(rect=(0.035, 0.09, 1, 0.995), h_pad=1.75, w_pad=1.45)
    figure.savefig(
        OUTPUT_DIRECTORY / PER_METHOD_OUTPUT_NAME,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    return figure


def configure_plot_defaults():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )


def print_data_summary(methods, discarded_mbpo_rows):
    """Report exactly which real-step ranges feed the saved figures."""
    if discarded_mbpo_rows:
        print(
            f"MBPO: ignored {discarded_mbpo_rows:,} rows from an older appended "
            "run before the real_timesteps reset."
        )
    for data, label, _color, _linewidth in methods:
        print(
            f"{label:8s}: {len(data):6,d} plotted episodes, "
            f"{data['training_steps'].min():,.0f} to "
            f"{data['training_steps'].max():,.0f} real steps"
        )


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_plot_defaults()

    sac = limit_episodes(load_sac_episode_data(SAC_FILE))
    ppo = limit_episodes(load_ppo_episode_data(PPO_MONITOR_DIRECTORY))
    mbrl = limit_episodes(load_mbrl_episode_data(MBRL_FILE))
    pilco = limit_episodes(load_pilco_episode_data(PILCO_FILE))
    mbpo_loaded = load_mbpo_episode_data(MBPO_FILE)
    discarded_mbpo_rows = mbpo_loaded.attrs.get("discarded_rows", 0)
    mbpo = limit_episodes(mbpo_loaded)

    methods = method_specs(sac, mbrl, pilco, mbpo, ppo)
    print_data_summary(methods, discarded_mbpo_rows)
    create_comparison_figure(sac, mbrl, pilco, mbpo, ppo)

    print(f"Saved figure: {(OUTPUT_DIRECTORY / OUTPUT_NAME).resolve()}")
    print(f"Saved vector figure: {(OUTPUT_DIRECTORY / OUTPUT_PDF_NAME).resolve()}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
