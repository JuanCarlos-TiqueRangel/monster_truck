# utils/svgp_retrain_manager.py
import os
import time
import threading
import traceback
import numpy as np

from gp.svgp_dynamics import SVGPManager


def _extract_npz_arrays(npz: dict):
    """
    Support both your snapshot format and your seed dataset.
    Expected keys (your snapshots):
      flip, rate, x_pose, linear_speed_x, u, episode_id, dt
    """
    def pick(*keys):
        for k in keys:
            if k in npz:
                return npz[k]
        return None

    flip = pick("flip")
    rate = pick("rate")
    x    = pick("x_pose", "x", "pose_x")
    vx   = pick("linear_speed_x", "vx", "speed_x")
    u    = pick("u", "action")
    ep   = pick("episode_id", "ep")
    dt   = pick("dt")

    if flip is None or rate is None or x is None or vx is None or u is None:
        raise RuntimeError(f"Seed/NPZ missing required keys. Found keys: {list(npz.keys())}")

    if ep is None:
        # If seed file doesn't have episode_id, make one episode
        ep = np.zeros_like(flip, dtype=np.int64)

    if dt is None:
        dt = None

    return (
        np.asarray(flip).astype(np.float32),
        np.asarray(rate).astype(np.float32),
        np.asarray(x).astype(np.float32),
        np.asarray(vx).astype(np.float32),
        np.asarray(u).astype(np.float32),
        np.asarray(ep).astype(np.int64),
        dt,
    )


def build_derivative_dataset(flip, rate, x, vx, u, ep, dt: float):
    """
    Build supervised dataset for derivative targets:
      X_t = [x, vx, flip, rate, u] at t
      y = (next - current)/dt   (only within same episode)
    """
    if len(flip) < 2:
        return None

    same_ep = (ep[:-1] == ep[1:])

    x0    = x[:-1][same_ep]
    vx0   = vx[:-1][same_ep]
    flip0 = flip[:-1][same_ep]
    rate0 = rate[:-1][same_ep]
    u0    = u[:-1][same_ep]

    x1    = x[1:][same_ep]
    vx1   = vx[1:][same_ep]
    flip1 = flip[1:][same_ep]
    rate1 = rate[1:][same_ep]

    X = np.stack([x0, vx0, flip0, rate0, u0], axis=1).astype(np.float32)

    y_dx   = ((x1    - x0)    / dt).astype(np.float32)
    y_dvx  = ((vx1   - vx0)   / dt).astype(np.float32)
    y_dflip= ((flip1 - flip0) / dt).astype(np.float32)
    y_drate= ((rate1 - rate0) / dt).astype(np.float32)

    # drop non-finite rows
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y_dx) & np.isfinite(y_dvx) & np.isfinite(y_dflip) & np.isfinite(y_drate)
    X = X[finite]
    y_dx = y_dx[finite]
    y_dvx = y_dvx[finite]
    y_dflip = y_dflip[finite]
    y_drate = y_drate[finite]

    if X.shape[0] == 0:
        return None

    return X, y_dx, y_dvx, y_dflip, y_drate


class SVGPRetrainManager:
    """
    Same external behavior as your previous GPRetrainManager:
      - .training, .reload_pending, .last_train_size
      - maybe_start_retrain_async(dataset, episode_id, force=False) -> bool
      - reload_models_if_ready() -> (gp_pose_x, gp_vx, gp_flip, gp_rate) or None

    This version warm-updates SVGP checkpoints instead of retraining ExactGP from scratch.
    """

    def __init__(self, cfg, device, model_lock, logger=None):
        self.cfg = cfg
        self.device = device
        self.model_lock = model_lock
        self.logger = logger

        self.training = False
        self.reload_pending = False
        self.train_thread = None

        self.last_train_size = 0
        self.last_reload_stats = None

    def maybe_start_retrain_async(self, dataset, episode_id: int, force: bool = False) -> bool:
        if self.training:
            if self.logger:
                self.logger.info("SVGP update requested but training is already running; skipping.")
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

        # Snapshot + save NPZ for debugging (same as before)
        flip, rate, x, vx, u, ep = dataset.snapshot()
        dataset.save_npz(episode_id, flip, rate, x, vx, u, ep)

        # Cap training window
        M = int(self.cfg.max_points_for_train)
        if len(flip) > M:
            flip = flip[-M:]
            rate = rate[-M:]
            x    = x[-M:]
            vx   = vx[-M:]
            u    = u[-M:]
            ep   = ep[-M:]

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(flip, rate, x, vx, u, ep, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info("Started SVGP warm-update thread.")
        return True

    def _train_worker(self, flip, rate, x, vx, u, ep, n_at_start: int):
        t0 = time.perf_counter()

        dt = float(self.cfg.ctrl_dt)

        # Build dataset from current window
        built = build_derivative_dataset(flip, rate, x, vx, u, ep, dt)
        if built is None:
            if self.logger:
                self.logger.warn("SVGP update: not enough valid consecutive samples to build derivative dataset.")
            self.training = False
            return

        Xw, y_dx_w, y_dvx_w, y_dflip_w, y_drate_w = built

        # Optionally include seed once per update (BUT only if model has no replay buffer stored)
        X_seed = y_dx_seed = y_dvx_seed = y_dflip_seed = y_drate_seed = None
        if bool(getattr(self.cfg, "keep_seed", False)) and os.path.exists(self.cfg.seed_npz_path):
            try:
                npz = np.load(self.cfg.seed_npz_path)
                s_flip, s_rate, s_x, s_vx, s_u, s_ep, s_dt = _extract_npz_arrays(npz)
                seed_dt = float(s_dt) if s_dt is not None else dt
                built_seed = build_derivative_dataset(s_flip, s_rate, s_x, s_vx, s_u, s_ep, seed_dt)
                if built_seed is not None:
                    X_seed, y_dx_seed, y_dvx_seed, y_dflip_seed, y_drate_seed = built_seed
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"Could not load/parse seed_npz for SVGP update: {e}")

        def combine(Xa, ya, Xb, yb):
            if Xa is None:
                return Xb, yb
            if Xb is None:
                return Xa, ya
            X = np.concatenate([Xa, Xb], axis=0)
            y = np.concatenate([ya, yb], axis=0)
            return X, y

        # Combine seed + window
        X_dx,   y_dx   = combine(X_seed, y_dx_seed, Xw, y_dx_w)
        X_dvx,  y_dvx  = combine(X_seed, y_dvx_seed, Xw, y_dvx_w)
        X_dflip,y_dflip= combine(X_seed, y_dflip_seed, Xw, y_dflip_w)
        X_drate,y_drate= combine(X_seed, y_drate_seed, Xw, y_drate_w)

        warm_steps = int(getattr(self.cfg, "svgp_warm_steps", 200))

        # Load, warm-update, save atomically
        paths = [self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]

        try:
            gp_pose_x = SVGPManager.load(paths[0], device=self.device)
            gp_vx     = SVGPManager.load(paths[1], device=self.device)
            gp_flip   = SVGPManager.load(paths[2], device=self.device)
            gp_rate   = SVGPManager.load(paths[3], device=self.device)

            # Warm update using *current* window (and seed) as the replay buffer for this update
            gp_pose_x.add_data(X_dx,   y_dx,   retrain=True, warm_steps=warm_steps, max_points=self.cfg.max_points_for_train, keep_raw=False)
            gp_vx.add_data(    X_dvx,  y_dvx,  retrain=True, warm_steps=warm_steps, max_points=self.cfg.max_points_for_train, keep_raw=False)
            gp_flip.add_data(  X_dflip,y_dflip,retrain=True, warm_steps=warm_steps, max_points=self.cfg.max_points_for_train, keep_raw=False)
            gp_rate.add_data(  X_drate,y_drate,retrain=True, warm_steps=warm_steps, max_points=self.cfg.max_points_for_train, keep_raw=False)

            # Atomic save
            for gp, out_path in zip([gp_pose_x, gp_vx, gp_flip, gp_rate], paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            self.last_train_size = n_at_start

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"SVGP warm-update finished in {elapsed:.2f}s | "
                    f"N_pairs_window={Xw.shape[0]} | warm_steps={warm_steps}"
                )

            self.reload_pending = True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.error(f"SVGP update failed after {elapsed:.2f}s: {e}")
                self.logger.error(traceback.format_exc())

        finally:
            self.training = False

    def reload_models_if_ready(self):
        """
        Returns (gp_pose_x, gp_vx, gp_flip, gp_rate) or None
        """
        if not self.reload_pending or self.training:
            return None

        # Hot-swap: load fresh managers onto device under model_lock
        with self.model_lock:
            gp_flip   = SVGPManager.load(self.cfg.gp_flip_path, device=self.device)
            gp_rate   = SVGPManager.load(self.cfg.gp_rate_path, device=self.device)
            gp_pose_x = SVGPManager.load(self.cfg.gp_pos_x_path, device=self.device)
            gp_vx     = SVGPManager.load(self.cfg.gp_vx_path, device=self.device)

        self.reload_pending = False

        try:
            stats = []
            for p in [self.cfg.gp_pos_x_path, self.cfg.gp_vx_path, self.cfg.gp_flip_path, self.cfg.gp_rate_path]:
                stats.append((p, os.path.getmtime(p), os.path.getsize(p)))
            self.last_reload_stats = stats
        except Exception:
            self.last_reload_stats = None

        if self.logger:
            self.logger.info("Reloaded SVGP models (hot swap).")

        return gp_pose_x, gp_vx, gp_flip, gp_rate
