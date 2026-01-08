import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

def load_metrics(path: str, episode_offset: int = 0):
    df = pd.read_csv(path)
    df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
    df["flip_time_sec"] = pd.to_numeric(df["flip_time_sec"], errors="coerce")
    df = df.dropna(subset=["episode", "flip_time_sec"])
    df = df.groupby("episode", as_index=False)["flip_time_sec"].mean()
    df = df.sort_values("episode").reset_index(drop=True)
    df["episode_plot"] = df["episode"] + episode_offset
    return df

def add_run(ax, df, label, color, window=50):
    x = df["episode_plot"].to_numpy(float)
    y = df["flip_time_sec"].to_numpy(float)
    s = pd.Series(y)

    ax.scatter(x, y, s=10, alpha=0.08, color=color, label=f"{label} (raw)")

    med = s.rolling(window=window, min_periods=1).median().to_numpy()
    q25 = s.rolling(window=window, min_periods=1).quantile(0.25).to_numpy()
    q75 = s.rolling(window=window, min_periods=1).quantile(0.75).to_numpy()

    ax.plot(x, med, color=color, linewidth=3.5, label=f"{label} (rolling median)")
    ax.fill_between(x, q25, q75, color=color, alpha=0.12, linewidth=0)

runs = [
    ("episode_metrics_base.csv",       "Model Base",       "black",  0),
    ("episode_metrics_50_Entropy.csv", "Model With Entropy", "green",  100),
    ("episode_metrics_NE_1000.csv",    "Mean NE",         "blue",   150),
    ("episode_metrics.csv",            "Mean with Entropy",     "purple", 150),
]

fig, ax = plt.subplots(figsize=(14, 6))

for path, label, color, offset in runs:
    df = load_metrics(path, episode_offset=offset)
    add_run(ax, df, label, color, window=50)

for v, txt, c in [(100, "100", "black"), (150, "150", "green"), (1000, "1000", "blue")]:
    ax.axvline(v, color=c, alpha=0.2, linewidth=2.5)
    ax.text(v, ax.get_ylim()[0], f" {txt}", color=c, va="bottom", fontsize=14)

ax.set_xlabel("Episodes")
ax.set_ylabel("Flip time (s)")
ax.set_title("Flip time vs Episode")
ax.grid(True, alpha=0.2)
ax.legend(ncol=2, frameon=True)

plt.tight_layout()
plt.show()
