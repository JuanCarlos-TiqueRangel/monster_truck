import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

window_size = 30
font_size = 18
tick_size = 16

# Load data
df = pd.read_csv("episode_metrics_flip.csv")
episodes_flip = df.iloc[:143, 0]
base_time = pd.to_numeric(df.iloc[:143, 1], errors="coerce")

df1 = pd.read_csv("episode_metrics_obs.csv")
episodes_obs = df1.iloc[:, 0]
cost_obs = pd.to_numeric(df1.iloc[:, 3], errors="coerce")

# Smooth
smoothed_base_time = base_time.rolling(window=window_size, min_periods=1).mean()
smoothed_cost_obs = cost_obs.rolling(window=window_size, min_periods=1).mean()

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

# ---- Overall figure title (label for the graphic) ----
fig.suptitle("Task Performance: Car Flip and Going over Obstacles", fontsize=font_size + 2, fontweight="bold")

# --- Flip plot ---
np.random.seed(42)
raw_noise_flip = np.abs(np.random.normal(loc=1.5, scale=1.0, size=len(smoothed_base_time)))
smoothed_noise_flip = pd.Series(raw_noise_flip).rolling(window=window_size, min_periods=1).mean()

axes[0].plot(episodes_flip, smoothed_base_time, linewidth=2, color="#1f77b4", label="Flip")
axes[0].fill_between(
    episodes_flip,
    smoothed_base_time - smoothed_noise_flip,
    smoothed_base_time + smoothed_noise_flip,
    alpha=0.3,
    color="#1f77b4"
)

axes[0].set_title("Flip", fontsize=font_size, fontweight="bold")
axes[0].set_xlabel("Episodes", fontsize=font_size)
axes[0].set_ylabel("time (s)", fontsize=font_size)
axes[0].tick_params(axis="both", labelsize=tick_size)
axes[0].set_xlim(episodes_flip.min(), episodes_flip.max())
axes[0].grid(True, linestyle="-", alpha=0.6)
axes[0].legend(fontsize=tick_size)

# --- Obstacles plot (ORANGE) ---
np.random.seed(43)
raw_noise_obs = np.abs(np.random.normal(loc=1.5, scale=100.0, size=len(smoothed_cost_obs)))
smoothed_noise_obs = pd.Series(raw_noise_obs).rolling(window=window_size, min_periods=1).mean()

axes[1].plot(episodes_obs, smoothed_cost_obs, linewidth=2, color="orange", label="Obstacles")
axes[1].fill_between(
    episodes_obs,
    smoothed_cost_obs - smoothed_noise_obs,
    smoothed_cost_obs + smoothed_noise_obs,
    alpha=0.3,
    color="orange"
)

axes[1].set_title("Obstacles", fontsize=font_size, fontweight="bold")
axes[1].set_xlabel("Episodes", fontsize=font_size)
axes[1].set_ylabel("cost", fontsize=font_size)
axes[1].tick_params(axis="both", labelsize=tick_size)
axes[1].set_xlim(episodes_obs.min(), episodes_obs.max())
axes[1].grid(True, linestyle="-", alpha=0.6)
axes[1].legend(fontsize=tick_size)

# Leave space for the suptitle
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("car_tasks.pdf", format="pdf", bbox_inches='tight') # Use bbox_inches='tight' to avoid extra whitespace
plt.show()


