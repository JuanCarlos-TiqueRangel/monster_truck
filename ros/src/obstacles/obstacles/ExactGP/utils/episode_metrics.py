# utils/episode_metrics.py
import os
import csv


class EpisodeMetricsWriter:
    def __init__(self, log_dir: str, plotter=None, logger=None):
        self.log_dir = str(log_dir)
        self.plotter = plotter
        self.logger = logger

        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.log_dir, "episode_metrics.csv")

        # If file doesn't exist, write header INCLUDING cost
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "time_to_goal_sec", "retrain_started", "cost"])


    def write(self, episode: int, time_to_goal_sec: float, retrain_started: bool, cost: float):
            ep = int(episode)
            dt = float(time_to_goal_sec)
            rs = int(bool(retrain_started))
            c  = float(cost)

            if self.logger:
                self.logger.info(
                    f"Episode {ep} time_to_goal: {dt:.3f} s | retrain_started={rs} | cost={c:.3f}"
                )

            with open(self.metrics_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([ep, dt, rs, c])

            if self.plotter is not None:
                self.plotter.update(ep, dt, c)

    # def write(self, episode: int, time_to_goal_sec: float, retrain_started: bool, cost: float, success: int):
    #     ep = int(episode)
    #     dt = float(time_to_goal_sec)
    #     rs = int(bool(retrain_started))
    #     c  = float(cost)
    #     succ = int(success)

    #     if self.logger:
    #         self.logger.info(
    #             f"Episode {ep} time_to_goal: {dt:.3f} s | retrain_started={rs} | cost={c:.3f} | Success = {succ}"
    #         )

    #     with open(self.metrics_path, "a", newline="") as f:
    #         w = csv.writer(f)
    #         w.writerow([ep, dt, rs, c, succ])

    #     if self.plotter is not None:
    #         self.plotter.update(ep, dt, c)
