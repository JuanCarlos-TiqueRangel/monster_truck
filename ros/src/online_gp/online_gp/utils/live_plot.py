# utils/live_plot.py
import os
import threading

class LivePlotter:
    """
    Live plotter with modes:

      mode = "cost" : single plot (cost vs episode)
      mode = "time" : single plot (time_to_goal_sec vs episode)
      mode = "both" : 2 subplots in the same figure:
                     top = cost vs episode
                     bottom = time vs episode
    """

    def __init__(self, enabled: bool, save_png: bool, out_dir: str, mode: str = "cost", logger=None):
        self.enabled = bool(enabled)
        self.save_png = bool(save_png)
        self.out_dir = str(out_dir)
        self.mode = str(mode).lower().strip()
        self.logger = logger

        self.ok = False
        self.ep_hist = []
        self.t_hist = []
        self.c_hist = []

        self._warned_non_main = False


        if self.enabled:
            self._init_plot()

    def _init_plot(self):
        try:
            import matplotlib.pyplot as plt
            self._plt = plt

            plt.ion()

            if self.mode == "both":
                # Two stacked plots, shared x-axis
                self.fig, (self.ax_cost, self.ax_time) = plt.subplots(
                    2, 1, sharex=True, figsize=(10, 6)
                )

                (self.line_cost,) = self.ax_cost.plot([], [], marker="o", label="cost")
                self.ax_cost.set_ylabel("Cost")
                self.ax_cost.grid(True)
                self.ax_cost.legend(loc="best")

                (self.line_time,) = self.ax_time.plot([], [], marker="o", label="time_to_goal_sec", color="tab:orange")
                self.ax_time.set_xlabel("Episode")
                self.ax_time.set_ylabel("Time (s)")
                self.ax_time.grid(True)
                self.ax_time.legend(loc="best")

                title = "Learning Curve (Cost + Time)"
                try:
                    self.fig.canvas.manager.set_window_title(title)
                except Exception:
                    pass

                self.fig.tight_layout()

            else:
                # Single plot
                self.fig, self.ax = plt.subplots()
                (self.line,) = self.ax.plot([], [], marker="o")
                self.ax.set_xlabel("Episode")
                self.ax.grid(True)

                if self.mode == "time":
                    self.ax.set_ylabel("Time to goal (s)")
                    title = "Learning Curve (Time)"
                else:
                    self.ax.set_ylabel("Cost")
                    title = "Learning Curve (Cost)"

                try:
                    self.fig.canvas.manager.set_window_title(title)
                except Exception:
                    pass

                self.fig.tight_layout()

            self.ok = True
            if self.logger:
                self.logger.info(f"Live plot enabled (matplotlib). mode={self.mode}")

        except Exception as e:
            self.ok = False
            if self.logger:
                self.logger.warn(f"Live plot disabled (matplotlib init failed): {e}")

    def update(self, ep: int, time_to_goal_sec: float, cost: float):
        if not self.ok:
            return

        # --- SAFETY: TkAgg must be updated from the main thread ---
        if threading.current_thread() is not threading.main_thread():
            if self.logger and not self._warned_non_main:
                self.logger.warn(
                    "LivePlotter.update() called from non-main thread; "
                    "skipping interactive update to avoid Tk crash."
                )
                self._warned_non_main = True

            # You can still store history if you want, but do NOT call draw/flush/pause/savefig here.
            self.ep_hist.append(int(ep))
            self.t_hist.append(float(time_to_goal_sec))
            self.c_hist.append(float(cost))
            return

        # ---- Normal (main-thread) plotting below ----
        self.ep_hist.append(int(ep))
        self.t_hist.append(float(time_to_goal_sec))
        self.c_hist.append(float(cost))

        try:
            if self.mode == "both":
                self.line_cost.set_data(self.ep_hist, self.c_hist)
                self.ax_cost.relim()
                self.ax_cost.autoscale_view()

                self.line_time.set_data(self.ep_hist, self.t_hist)
                self.ax_time.relim()
                self.ax_time.autoscale_view()

            elif self.mode == "time":
                self.line.set_data(self.ep_hist, self.t_hist)
                self.ax.relim()
                self.ax.autoscale_view()

            else:  # "cost"
                self.line.set_data(self.ep_hist, self.c_hist)
                self.ax.relim()
                self.ax.autoscale_view()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._plt.pause(0.001)

            if self.save_png:
                os.makedirs(self.out_dir, exist_ok=True)
                out = os.path.join(self.out_dir, "learning_curve.png")
                self.fig.savefig(out, dpi=150)

        except Exception as e:
            # If anything GUI-related still fails, disable live plot instead of crashing your system
            self.ok = False
            if self.logger:
                self.logger.warn(f"Live plot disabled (update failed): {e}")

