#!/usr/bin/env python3
"""
plot_results.py
----------------
Training curves for the RL baselines (obstacle_mujoco_sac.py / _ppo.py / _PETS.py /
_PILCO.py), all on the same obstacle wheelie-climb task as the MPPI/NMPC controllers.

Just change METHOD below and re-run -- everything else is picked up automatically:
  "sac"   -> results/sac_monitor/*.monitor.csv  (written by obstacle_mujoco_sac.py)
  "ppo"   -> results/ppo_monitor/*.monitor.csv  (one file per parallel worker,
                                                  written by obstacle_mujoco_ppo.py)
  "pets"  -> results/pets_episodes.csv          (written by obstacle_mujoco_PETS.py)
  "pilco" -> results/pilco_episodes.csv         (written by obstacle_mujoco_PILCO.py)

Five panels, all against cumulative training timesteps (PPO's workers are merged and
sorted by wall-clock time first, matching stable_baselines3's own results_plotter
convention):

  1. episode return           -> raw + rolling mean; UP = learning.
  2. x_final vs goal          -> green = reached goal, red = crashed, grey = neither.
  3. rolling success rate     -> fraction of the last ROLLING_WINDOW episodes reached.
  4. effort (RMS torque)      -> raw + rolling mean; how hard the controller works.
  5. time to reach goal       -> reached episodes only; DOWN = getting there faster.

Panels 4-5 need the "sim_time"/"effort_rms" columns, only logged after this script's
env/training-script update -- re-run training once if your log predates that.

Usage:
    python plot_results.py
The figure is always saved as a PNG under results/images/; this script never opens an
interactive window (the "Agg" backend below is headless by construction), since a
container's Qt/X11 setup is a separate, unrelated thing to depend on for saving a plot.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: render straight to PNG, no GUI toolkit involved

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD = "sac"   # <-- change this: "sac" | "ppo" | "pets" | "pilco"

ROLLING_WINDOW = 10   # episodes, for the smoothed return / success-rate lines

_ROOT = Path(__file__).resolve().parent.parent   # Trail/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from obstacle_gym_env import GOAL_X  # noqa: E402


def _load_monitor_dir(monitor_dir, run_script):
    from stable_baselines3.common.monitor import load_results

    if not monitor_dir.exists() or not any(monitor_dir.glob("*monitor.csv")):
        raise SystemExit(f"No monitor logs found in {monitor_dir}\nRun {run_script} first.")

    df = load_results(str(monitor_dir))
    df = df.rename(columns={"r": "return", "l": "steps", "x": "x_final"})
    df["episode"] = np.arange(len(df))
    df["reached"] = df["reached"].astype(bool)
    df["crashed"] = df["crashed"].astype(bool)
    return df


def _load_episode_csv(path, run_script):
    if not path.exists():
        raise SystemExit(f"No log found at {path}\nRun {run_script} first.")
    df = pd.read_csv(path)
    df["reached"] = df["reached"].astype(bool)
    df["crashed"] = df["crashed"].astype(bool)
    return df


def load(method):
    here = Path(__file__).parent
    method = method.lower()
    if method == "sac":
        return _load_monitor_dir(here / "sac_monitor", "obstacle_mujoco_sac.py"), "SAC"
    if method == "ppo":
        return _load_monitor_dir(here / "ppo_monitor", "obstacle_mujoco_ppo.py"), "PPO"
    if method == "pets":
        return _load_episode_csv(here / "pets_episodes.csv", "obstacle_mujoco_PETS.py"), "PETS"
    if method == "pilco":
        return _load_episode_csv(here / "pilco_episodes.csv", "obstacle_mujoco_PILCO.py"), "PILCO"
    raise SystemExit(f"METHOD must be 'sac', 'ppo', 'pets' or 'pilco', got {method!r}")


def rolling(series):
    return series.rolling(ROLLING_WINDOW, min_periods=1).mean()


def main():
    df, title = load(METHOD)
    # Cumulative env steps at the end of each episode -- same convention as
    # stable_baselines3.common.results_plotter's X_TIMESTEPS mode. The rolling
    # window below is still counted in EPISODES (smoothing "the last N episodes"),
    # only the x-axis position of each point changes from episode index to this.
    df["timesteps"] = df["steps"].cumsum()

    needed = {"effort_rms", "sim_time"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(
            f"{title} log is missing {sorted(missing)} -- these columns were added "
            f"after your last training run logged this file. Re-run the training "
            f"script once to regenerate the log with them, then plot again."
        )

    reached_pct = 100.0 * df["reached"].mean()
    crashed_pct = 100.0 * df["crashed"].mean()
    print(f"{title}: {len(df)} episodes | reached goal: {reached_pct:.0f}% | "
          f"crashed: {crashed_pct:.0f}% | best return: {df['return'].max():.1f}")

    fig, (a1, a2, a3, a4, a5) = plt.subplots(5, 1, figsize=(9, 15), sharex=True)

    # 1) episode return
    a1.plot(df["timesteps"], df["return"], color="0.8", lw=1.0, label="return")
    a1.plot(df["timesteps"], rolling(df["return"]), "C0", lw=2.0,
            label=f"rolling mean ({ROLLING_WINDOW} ep)")
    a1.set_title("1) episode return")
    a1.set_ylabel("return"); a1.grid(True); a1.legend(fontsize=8)

    # 2) x_final vs goal, colored by outcome (same green/red convention as
    # plot_obstacle_episodes.py: green = reached, red = crashed)
    neither = ~(df["reached"] | df["crashed"])
    a2.scatter(df.loc[neither, "timesteps"], df.loc[neither, "x_final"],
               s=16, c="0.6", label="neither")
    a2.scatter(df.loc[df["crashed"], "timesteps"], df.loc[df["crashed"], "x_final"],
               s=16, c="C3", label="crashed")
    a2.scatter(df.loc[df["reached"], "timesteps"], df.loc[df["reached"], "x_final"],
               s=16, c="C2", label="reached goal")
    a2.axhline(GOAL_X, color="green", ls="--", lw=1.0)
    a2.set_title("2) x_final per episode (dashed = goal)")
    a2.set_ylabel("x_final [m]"); a2.grid(True); a2.legend(fontsize=8, ncol=3)

    # 3) rolling success rate
    success = rolling(df["reached"].astype(float))
    a3.plot(df["timesteps"], success, "C2", lw=2.0)
    a3.fill_between(df["timesteps"], 0, success, color="C2", alpha=0.15)
    a3.set_ylim(-0.05, 1.05)
    a3.set_title(f"3) rolling success rate ({ROLLING_WINDOW} ep)")
    a3.set_xlabel("timesteps"); a3.set_ylabel("success rate"); a3.grid(True)

    # 4) effort: RMS torque over each episode (how hard the controller worked)
    a4.plot(df["timesteps"], df["effort_rms"], color="0.8", lw=1.0, label="effort_rms")
    a4.plot(df["timesteps"], rolling(df["effort_rms"]), "C1", lw=2.0,
            label=f"rolling mean ({ROLLING_WINDOW} ep)")
    a4.set_title("4) effort (RMS torque per episode)")
    a4.set_ylabel("effort_rms [N*m]"); a4.grid(True); a4.legend(fontsize=8)

    # 5) time to reach the goal -- only defined for episodes that actually got there
    reached_df = df.loc[df["reached"]]
    if len(reached_df) == 0:
        a5.text(0.5, 0.5, "no episodes reached the goal yet", ha="center", va="center",
                transform=a5.transAxes, color="0.5")
    else:
        a5.scatter(reached_df["timesteps"], reached_df["sim_time"], s=16, c="C2",
                   label="reached")
        if len(reached_df) >= 2:
            roll_time = reached_df["sim_time"].rolling(ROLLING_WINDOW, min_periods=1).mean()
            a5.plot(reached_df["timesteps"], roll_time, "C2", lw=2.0,
                    label=f"rolling mean ({ROLLING_WINDOW} reached ep)")
        a5.legend(fontsize=8)
    a5.set_title("5) time to reach goal (reached episodes only)")
    a5.set_xlabel("timesteps"); a5.set_ylabel("time to goal [s]"); a5.grid(True)

    fig.suptitle(f"{title} training curves -- obstacle wheelie-climb task")
    fig.tight_layout()

    img_dir = Path(__file__).parent / "images"
    try:
        img_dir.mkdir(exist_ok=True)
        out = img_dir / f"{METHOD.lower()}_training_curves.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"Saved figure: {out}")
    except PermissionError:
        out = Path(f"/tmp/{METHOD.lower()}_training_curves.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"[warn] {img_dir} not writable; saved to {out}")


if __name__ == "__main__":
    main()
