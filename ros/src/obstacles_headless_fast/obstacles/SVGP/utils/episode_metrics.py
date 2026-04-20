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
        self.header = ["episode", "time_to_goal_sec", "retrain_started", "cost", "success"]
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
            return

        try:
            with open(self.metrics_path, "r", newline="") as f:
                rows = list(csv.reader(f))
        except Exception:
            rows = []

        if not rows:
            with open(self.metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(self.header)
            return

        if rows[0] == self.header:
            return

        bak_path = self.metrics_path + ".bak"
        try:
            os.replace(self.metrics_path, bak_path)
            if self.logger:
                self.logger.warn(
                    f"episode_metrics.csv had an old header. Previous file moved to {bak_path}"
                )
        except Exception:
            pass

        with open(self.metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    def write(self, episode: int, time_to_goal_sec: float, retrain_started: bool, cost: float, success: int):
        ep = int(episode)
        dt = float(time_to_goal_sec)
        rs = int(bool(retrain_started))
        c = float(cost)
        succ = int(success)

        if self.logger:
            self.logger.info(
                f"Episode {ep} time_to_goal: {dt:.3f} s | retrain_started={rs} | cost={c:.3f} | success={succ}"
            )

        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, dt, rs, c, succ])

        if self.plotter is not None:
            self.plotter.update(ep, dt, c)
