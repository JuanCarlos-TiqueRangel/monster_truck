"""Publication-style comparison plot for the obstacle-avoidance methods.

This script intentionally reuses the data loaders from compare_methods.py and
only replaces the figure design.  It plots the common training budget
episode 0 through episode 999, because the proposed method has 1000 episodes.

Outputs:
    images/methods_comparison_plot2.png
    images/methods_comparison_plot2.pdf
    images/methods_comparison_plot2_summary.csv
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

import compare_methods as cm


# =============================================================================
# Configuration
# =============================================================================

RESULTS_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = RESULTS_DIRECTORY / "images"

EPISODE_BUDGET = 1000
ROLLING_WINDOW_EPISODES = 75
ROLLING_MIN_PERIODS = 8

OUTPUT_PNG = OUTPUT_DIRECTORY / "methods_comparison_plot2.png"
OUTPUT_PDF = OUTPUT_DIRECTORY / "methods_comparison_plot2.pdf"
OUTPUT_SUMMARY = OUTPUT_DIRECTORY / "methods_comparison_plot2_summary.csv"

PANEL_BACKGROUND = "#eeeeee"
GRID_COLOR = "#ffffff"
TEXT_COLOR = "#1f1f1f"
MUTED_TEXT_COLOR = "#555555"


@dataclass(frozen=True)
class MethodStyle:
    label: str
    color: str
    linewidth: float


METHOD_STYLES = {
    "MBPO": MethodStyle("MBPO", "#1f77b4", 2.45),
    "SAC": MethodStyle("SAC", "#2ca02c", 2.35),
    "PPO": MethodStyle("PPO", "#9467bd", 2.35),
    "PILCO": MethodStyle("PILCO", "#e377c2", 2.35),
    "Proposed": MethodStyle("Proposed", "#ff7f0e", 2.85),
}


METRICS = [
    ("time", "Episode duration", "duration (s)", False, (1.6, 8.4), [2, 4, 6, 8]),
    ("success", "Success rate", "success (%)", True, (0.0, 103.0), [0, 25, 50, 75, 100]),
    ("effort", "Control effort", "RMS control effort", False, (4.0, 7.4), [4, 5, 6, 7]),
]


# =============================================================================
# Data loading
# =============================================================================

def load_methods():
    """Return methods in the same visual order as the MBPO paper reference."""
    sac = cm.load_sac_episode_data(cm.SAC_FILE)
    ppo = cm.load_ppo_episode_data(cm.PPO_MONITOR_DIRECTORY)
    proposed = cm.load_mbrl_episode_data(cm.MBRL_FILE)
    pilco = cm.load_pilco_episode_data(cm.PILCO_FILE)
    mbpo = cm.load_mbpo_episode_data(cm.MBPO_FILE)

    return [
        (mbpo, METHOD_STYLES["MBPO"]),
        (sac, METHOD_STYLES["SAC"]),
        (ppo, METHOD_STYLES["PPO"]),
        (pilco, METHOD_STYLES["PILCO"]),
        (proposed, METHOD_STYLES["Proposed"]),
    ]


def common_budget(data):
    """Keep only episodes inside the fair 1000-episode comparison budget."""
    visible = data[(data["episode"] >= 0) & (data["episode"] < EPISODE_BUDGET)].copy()
    return visible.sort_values("episode", kind="stable").reset_index(drop=True)


# =============================================================================
# Statistics
# =============================================================================

def rolling_curve(data, metric, *, percentage=False):
    """Compute a causal rolling mean and a light local variability band."""
    values = pd.to_numeric(data[metric], errors="coerce")
    rolling = values.rolling(
        ROLLING_WINDOW_EPISODES,
        min_periods=ROLLING_MIN_PERIODS,
    )

    center = rolling.mean()
    count = rolling.count().clip(lower=1.0)

    if percentage:
        center = 100.0 * center
        p = center / 100.0
        half_width = 100.0 * np.sqrt((p * (1.0 - p)).clip(lower=0.0) / count)
        lower = (center - half_width).clip(lower=0.0, upper=100.0)
        upper = (center + half_width).clip(lower=0.0, upper=100.0)
    else:
        std = rolling.std(ddof=1).fillna(0.0)
        half_width = 0.35 * std
        lower = (center - half_width).clip(lower=0.0)
        upper = center + half_width

    return center.to_numpy(), lower.to_numpy(), upper.to_numpy()


def first_success_streak_episode(data, streak_length=100):
    """Return the episode coordinate where the first success streak ends."""
    run_length = 0
    for _, row in data.iterrows():
        if row["success"] >= 0.5:
            run_length += 1
            if run_length == streak_length:
                return int(row["episode"])
        else:
            run_length = 0
    return np.nan


def build_summary(methods):
    """Save a numerical companion table using the same <=1000-episode budget."""
    rows = []
    for data, style in methods:
        visible = common_budget(data)
        successful = visible[visible["success"] >= 0.5]
        rows.append(
            {
                "method": style.label,
                "episodes_in_budget": len(visible),
                "success_rate_percent": 100.0 * visible["success"].mean(),
                "mean_successful_completion_time_s": successful["time"].mean(),
                "mean_episode_rms_control_effort": visible["effort"].mean(),
                "episode_ending_first_100_success_streak": first_success_streak_episode(visible),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def format_episode_tick(value, _position):
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:g}k"
    return f"{int(value)}"


def style_axis(axis, title, y_label, y_limits, y_ticks):
    axis.set_facecolor(PANEL_BACKGROUND)
    axis.grid(True, color=GRID_COLOR, linewidth=1.35)
    axis.set_axisbelow(True)
    axis.set_title(title, pad=10)
    axis.set_xlabel("episode")
    axis.set_ylabel(y_label)
    axis.set_xlim(0, EPISODE_BUDGET + 95)
    axis.set_xticks([0, 250, 500, 750, 1000])
    axis.xaxis.set_major_formatter(FuncFormatter(format_episode_tick))
    axis.set_ylim(*y_limits)
    axis.set_yticks(y_ticks)
    axis.tick_params(axis="both", which="both", length=0, colors=TEXT_COLOR)

    for spine in axis.spines.values():
        spine.set_visible(False)


def label_text(metric, value):
    if not np.isfinite(value):
        return ""
    if metric == "success":
        return f"{value:.0f}%"
    if metric == "time":
        return f"{value:.1f}s"
    return f"{value:.2f}"


def spread_endpoint_labels(axis, labels):
    """Place right-edge labels with simple vertical collision avoidance."""
    if not labels:
        return []

    y_min, y_max = axis.get_ylim()
    min_gap = 0.055 * (y_max - y_min)
    padded_min = y_min + 0.025 * (y_max - y_min)
    padded_max = y_max - 0.025 * (y_max - y_min)

    ordered = sorted(labels, key=lambda item: item["y"])
    adjusted = []
    last_y = -np.inf
    for item in ordered:
        new_item = item.copy()
        new_item["label_y"] = min(max(new_item["y"], padded_min), padded_max)
        if new_item["label_y"] - last_y < min_gap:
            new_item["label_y"] = last_y + min_gap
        adjusted.append(new_item)
        last_y = new_item["label_y"]

    overflow = adjusted[-1]["label_y"] - padded_max
    if overflow > 0:
        for item in adjusted:
            item["label_y"] -= overflow

    return adjusted


def add_endpoint_labels(axis, labels, metric):
    labels = spread_endpoint_labels(axis, labels)
    for item in labels:
        text = f"{item['style'].label} {label_text(metric, item['y'])}"
        axis.plot(
            [item["x"], EPISODE_BUDGET + 18],
            [item["y"], item["label_y"]],
            color=item["style"].color,
            linewidth=0.75,
            alpha=0.62,
            clip_on=False,
        )
        axis.text(
            EPISODE_BUDGET + 28,
            item["label_y"],
            text,
            ha="left",
            va="center",
            fontsize=8.2,
            color=item["style"].color,
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=2.7, foreground="white")],
        )


def plot_metric(axis, methods, metric, title, y_label, percentage, y_limits, y_ticks):
    style_axis(axis, title, y_label, y_limits, y_ticks)
    endpoint_labels = []

    for data, style in methods:
        visible = common_budget(data)
        if visible.empty:
            continue

        x = visible["episode"].to_numpy(dtype=float)
        center, lower, upper = rolling_curve(visible, metric, percentage=percentage)
        valid = np.isfinite(center)
        if not valid.any():
            continue

        axis.fill_between(
            x,
            lower,
            upper,
            where=np.isfinite(lower) & np.isfinite(upper),
            color=style.color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            center,
            color=style.color,
            linewidth=style.linewidth,
            solid_capstyle="round",
            zorder=3 if style.label != "Proposed" else 4,
        )

        last_index = np.flatnonzero(valid)[-1]
        axis.scatter(
            x[last_index],
            center[last_index],
            s=24 if style.label != "Proposed" else 34,
            color=style.color,
            edgecolor="white",
            linewidth=0.65,
            zorder=5,
        )
        endpoint_labels.append(
            {
                "x": float(x[last_index]),
                "y": float(center[last_index]),
                "style": style,
            }
        )

    if metric == "success":
        axis.axhline(100, color="#3d3d3d", linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.55)
    add_endpoint_labels(axis, endpoint_labels, metric)


def legend_handles(methods):
    return [
        Line2D([0], [0], color=style.color, lw=style.linewidth, label=style.label)
        for _, style in methods
    ]


def create_figure(methods):
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 3.95), squeeze=False)
    axes = axes[0]
    figure.patch.set_facecolor("white")

    for axis, metric_spec in zip(axes, METRICS):
        plot_metric(axis, methods, *metric_spec)

    figure.legend(
        handles=legend_handles(methods),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.7,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Common budget: episodes 0-{EPISODE_BUDGET - 1}. "
            f"Solid curves are trailing {ROLLING_WINDOW_EPISODES}-episode means; "
            "shading shows local variability within the same run."
        ),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=MUTED_TEXT_COLOR,
    )
    figure.subplots_adjust(left=0.06, right=0.905, top=0.86, bottom=0.27, wspace=0.34)
    return figure


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    methods = load_methods()
    summary = build_summary(methods)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    figure = create_figure(methods)
    figure.savefig(OUTPUT_PNG, dpi=350, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(figure)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved figure: {OUTPUT_PNG.resolve()}")
    print(f"Saved vector figure: {OUTPUT_PDF.resolve()}")
    print(f"Saved summary: {OUTPUT_SUMMARY.resolve()}")


if __name__ == "__main__":
    main()
