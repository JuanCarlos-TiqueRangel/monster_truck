import pandas as pd
import matplotlib.pyplot as plt

csv_path = "episode_metrics.csv"
rolling_window = 5

df = pd.read_csv(csv_path)

data = pd.DataFrame({
    "episode": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
    "success": pd.to_numeric(df.iloc[:, -1], errors="coerce")
}).dropna()

data = data[data["success"].isin([0, 1])].copy()
data["success"] = data["success"].astype(int)
data = data.sort_values("episode").reset_index(drop=True)

data["rolling_success_rate"] = data["success"].rolling(rolling_window, min_periods=1).mean()
data["cumulative_success_rate"] = data["success"].expanding().mean()

fig, ax = plt.subplots(figsize=(12, 5))

# ax.plot(data["episode"], data["rolling_success_rate"], linewidth=2,
#         label=f"Rolling success rate ({rolling_window} episodes)")
ax.plot(data["episode"], data["cumulative_success_rate"], linewidth=2,
        linestyle="--", label="Cumulative success rate")

success_points = data[data["success"] == 1]
fail_points = data[data["success"] == 0]

ax.scatter(success_points["episode"], success_points["success"],
           s=18, alpha=0.5, label="Success episodes")
ax.scatter(fail_points["episode"], fail_points["success"],
           s=18, alpha=0.25, label="Failure episodes")

ax.set_title("Success Rate Across Episodes")
ax.set_xlabel("Episode")
ax.set_ylabel("Rate / Outcome")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()

plt.savefig("success_rate_plot.png", dpi=200, bbox_inches="tight")
plt.show()

print(f"Overall success rate: {data['success'].mean():.2%}")