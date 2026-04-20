# utils/gp_retrain_manager.py
import os
import time
import threading
import traceback

from gp.gp_dynamics import GPManager
from gp.re_train_dynamics_gp import train_dynamics_gp_from_arrays


class GPRetrainManager:
    def __init__(self, cfg, device, model_lock, logger=None):
        self.cfg = cfg
        self.device = device
        self.model_lock = model_lock
        self.logger = logger

        self.training = False
        self.reload_pending = False
        self.train_thread = None

        # committed AFTER successful train+save
        self.last_train_size = 0

        # optional: store last reload file stats
        self.last_reload_stats = None

    def maybe_start_retrain_async(self, dataset, episode_id: int, force: bool = False) -> bool:
        if self.training:
            if self.logger:
                self.logger.info("Retrain requested but training is already running; skipping.")
            return False

        n = dataset.n_points()

        if not force:
            if n < self.cfg.min_points_to_train:
                if self.logger:
                    self.logger.info(f"Not enough data to retrain yet: {n} < {self.cfg.min_points_to_train}")
                return False

            if (n - self.last_train_size) < self.cfg.min_new_points_between_trains:
                if self.logger:
                    self.logger.info("Not enough new data since last train; skipping.")
                return False

        # snapshot + save
        flip, rate, x, vx, u, ep = dataset.snapshot()
        dataset.save_npz(episode_id, flip, rate, x, vx, u, ep)

        # cap training window
        M = int(self.cfg.max_points_for_train)
        flip, rate, x, vx, u, ep = dataset.cap_window(M, flip, rate, x, vx, u, ep)

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(flip, rate, x, vx, u, ep, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info("Started GP retraining thread.")
        return True

    def _train_worker(self, flip, rate, x, vx, u, ep, n_at_start: int):
        t0 = time.perf_counter()

        signals = {
            "x_pose": x,
            "linear_speed_x": vx,
            "flip": flip,
            "rate": rate,
            "u": u,
        }

        input_  = ["x_pose", "linear_speed_x", "flip", "rate", "u"]
        output_ = ["x_pose", "linear_speed_x", "flip", "rate"]

        try:
            gps, X, Y, y_names = train_dynamics_gp_from_arrays(
                signals_new=signals,
                dt=self.cfg.ctrl_dt,
                input_keys=input_,
                output_keys=output_,
                episode_id=ep,
                target_mode="derivative",
                kernel=self.cfg.train_kernel,
                iters=self.cfg.train_iters,
                seed_npz_path=self.cfg.seed_npz_path,
                keep_seed=self.cfg.keep_seed,
            )

            paths = [self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]
            assert len(gps) == len(paths)

            # Atomic save
            for gp, out_path in zip(gps, paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            # commit size only after success
            self.last_train_size = n_at_start

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"GP retraining finished in {elapsed:.2f}s | "
                    f"N_used={len(flip)} | kernel={self.cfg.train_kernel} | iters={self.cfg.train_iters}"
                )

            self.reload_pending = True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.error(f"Retraining failed after {elapsed:.2f}s: {e}")
                self.logger.error(traceback.format_exc())

        finally:
            self.training = False

    def reload_models_if_ready(self):
        """
        Returns (gp_pose_x, gp_vx, gp_flip, gp_rate) or None.
        """
        if not self.reload_pending or self.training:
            return None

        with self.model_lock:
            gp_flip   = GPManager.load(self.cfg.gp_flip_path)
            gp_rate   = GPManager.load(self.cfg.gp_rate_path)
            gp_pose_x = GPManager.load(self.cfg.gp_pos_x_path)
            gp_vx     = GPManager.load(self.cfg.gp_vx_path)

            gp_flip.device = self.device
            gp_rate.device = self.device
            gp_pose_x.device = self.device
            gp_vx.device = self.device

        self.reload_pending = False

        # stats (optional)
        try:
            stats = []
            for p in [self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]:
                stats.append((p, os.path.getmtime(p), os.path.getsize(p)))
            self.last_reload_stats = stats
        except Exception:
            self.last_reload_stats = None

        if self.logger:
            self.logger.info("Reloaded GP models (hot swap).")

        return gp_pose_x, gp_vx, gp_flip, gp_rate
