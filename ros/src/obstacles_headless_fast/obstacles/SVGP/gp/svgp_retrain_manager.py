# utils/gp_retrain_manager.py
import os
import time
import threading
import traceback
import numpy as np

from typing import Dict, List, Optional, Tuple

from gp.svgp_dynamics import SVGPManager
# IMPORTANT:
# SVGPManager must now be your SVGP-based version
# (ApproximateGP + VariationalELBO + inducing points)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params


# -----------------------------
# Utilities
# -----------------------------
def _as_2d(a: np.ndarray) -> np.ndarray:
    """Ensure array is (N, D). Accepts (N,) or (N, D)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"Signal array must be 1D or 2D. Got shape {a.shape}.")


def _align_signals(signals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Chop all signals to same length N = min length across keys."""
    N = min(v.shape[0] for v in signals.values())
    if N < 2:
        raise ValueError(f"Need at least 2 samples to form transitions. N={N}")
    return {k: v[:N] for k, v in signals.items()}


def _valid_mask(N: int, episode_id: Optional[np.ndarray]) -> np.ndarray:
    """valid[t] True if transition (t->t+1) stays in same episode."""
    if episode_id is None:
        return np.ones(N - 1, dtype=bool)

    ep = np.asarray(episode_id, dtype=np.int64).reshape(-1)
    if len(ep) < N:
        raise ValueError(f"episode_id shorter than data: {len(ep)} < {N}")
    ep = ep[:N]
    return (ep[1:] == ep[:-1])


def _build_X(signals: Dict[str, np.ndarray], input_keys: List[str], valid: np.ndarray) -> np.ndarray:
    parts = []
    for k in input_keys:
        if k not in signals:
            raise KeyError(f"Missing input key '{k}'. Available keys: {list(signals.keys())}")
        parts.append(signals[k][:-1][valid])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_Y(
    signals: Dict[str, np.ndarray],
    output_keys: List[str],
    dt: float,
    mode: str,
    valid: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    mode = mode.lower().strip()
    if mode not in ("derivative", "delta", "next"):
        raise ValueError(f"Unsupported target_mode='{mode}'. Use derivative|delta|next.")

    Y_parts: List[np.ndarray] = []
    y_names: List[str] = []

    for k in output_keys:
        if k not in signals:
            raise KeyError(f"Missing output key '{k}'. Available keys: {list(signals.keys())}")

        s = signals[k]  # (N, Dk)

        if mode == "derivative":
            yk = (s[1:] - s[:-1]) / float(dt)
            suffix = "d_dt"
        elif mode == "delta":
            yk = (s[1:] - s[:-1])
            suffix = "delta"
        else:  # next
            yk = s[1:]
            suffix = "next"

        yk = yk[valid].astype(np.float32)
        Y_parts.append(yk)

        Dk = yk.shape[1]
        if Dk == 1:
            y_names.append(f"{k}_{suffix}")
        else:
            for j in range(Dk):
                y_names.append(f"{k}{j}_{suffix}")

    Y = np.concatenate(Y_parts, axis=1).astype(np.float32)
    return Y, y_names


def _maybe_check_dt(D: np.lib.npyio.NpzFile, dt: float, context: str):
    if "dt" in D.files:
        dt_file = float(np.asarray(D["dt"]).reshape(()))
        if not np.isclose(dt_file, float(dt), rtol=1e-3, atol=1e-6):
            raise ValueError(f"{context} dt mismatch: file dt={dt_file} vs requested dt={dt}")


def _load_seed_npz( seed_npz_path: str, keys_needed: List[str], 
                   dt: float, seed_episode_id: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Loads a seed NPZ with arbitrary keys_needed.
    Returns: (signals_seed_dict, episode_id_seed_array)
    """
    D = np.load(seed_npz_path)
    _maybe_check_dt(D, dt, context="Seed NPZ")

    missing = [k for k in keys_needed if k not in D.files]
    if missing:
        raise KeyError(
            f"Seed NPZ missing keys {missing}. Needed={keys_needed}. Available={list(D.files)}"
        )

    signals_seed = {k: _as_2d(D[k]) for k in keys_needed}
    signals_seed = _align_signals(signals_seed)
    Ns = next(iter(signals_seed.values())).shape[0]

    if "episode_id" in D.files:
        ep_seed = np.asarray(D["episode_id"], dtype=np.int64).reshape(-1)[:Ns]
    else:
        ep_seed = np.full(Ns, int(seed_episode_id), dtype=np.int64)

    return signals_seed, ep_seed



# --------------------------------------------------------------
# MAIN MODULAR API
# --------------------------------------------------------------
def train_dynamics_gp_from_arrays(
    signals_new: Dict[str, np.ndarray],
    dt: float,
    input_keys: List[str],
    output_keys: List[str],
    episode_id: Optional[np.ndarray] = None,
    target_mode: str = "derivative",
    kernel: str = "RQ",
    lr: float = 0.03,
    iters: int = 300,
    num_inducing: int = 128,
    batch_size: Optional[int] = 256,
    seed_npz_path: Optional[str] = None,
    seed_episode_id: int = -1,
    keep_seed: bool = True,
) -> Tuple[List[SVGPManager], np.ndarray, np.ndarray, List[str]]:
    """
    Trains on ALL valid transitions from:
    - optional seed NPZ
    - new online signals (signals_new)

    Returns:
    gps (1 per output dim), X, Y, y_names
    """
    if not (float(dt) > 0.0):
        raise ValueError(f"dt must be > 0. Got dt={dt}")

    # Normalize new signals
    signals_new = {k: _as_2d(v) for k, v in signals_new.items()}
    signals_new = _align_signals(signals_new)
    Nn = next(iter(signals_new.values())).shape[0]

    valid_new = _valid_mask(Nn, episode_id)
    X_new = _build_X(signals_new, input_keys, valid_new)
    Y_new, y_names = _build_Y(signals_new, output_keys, dt, target_mode, valid_new)

    if X_new.shape[0] == 0:
        raise ValueError("No valid NEW transitions after episode filtering.")

    # Seed (optional)
    X_seed = Y_seed = None
    n_seed = 0
    if keep_seed and seed_npz_path is not None and os.path.exists(seed_npz_path):
        keys_needed = sorted(set(input_keys + output_keys))
        signals_seed, ep_seed = _load_seed_npz(seed_npz_path, keys_needed, dt, seed_episode_id)
        Ns = next(iter(signals_seed.values())).shape[0]
        valid_seed = _valid_mask(Ns, ep_seed)

        X_seed = _build_X(signals_seed, input_keys, valid_seed)
        Y_seed, _ = _build_Y(signals_seed, output_keys, dt, target_mode, valid_seed)
        n_seed = int(X_seed.shape[0])

        if X_seed.shape[0] == 0:
            raise ValueError("Seed NPZ produced 0 valid transitions (episode filtering removed all).")

    # Stack seed + new
    if X_seed is not None:
        X = np.vstack([X_seed, X_new]).astype(np.float32)
        Y = np.vstack([Y_seed, Y_new]).astype(np.float32)
    else:
        X = X_new.astype(np.float32)
        Y = Y_new.astype(np.float32)

    print(f"[TRAIN DATA] seed_transitions={n_seed} | new_transitions={len(X_new)} | total={len(X)}")
    print(
        f"[TRAIN DATA] X={X.shape} | Y={Y.shape} | mode={target_mode} | "
        f"kernel={kernel} | lr={lr} | iters={iters} | "
        f"num_inducing={num_inducing} | batch_size={batch_size}"
    )
    print(f"[TRAIN DATA] inputs={input_keys}")
    print(f"[TRAIN DATA] outputs={output_keys}")

    # Train 1 SVGP per output dimension
    Dy = Y.shape[1]
    gps = [
        SVGPManager(
            kernel=kernel,
            lr=lr,
            iters=iters,
            num_inducing=num_inducing,
            batch_size=batch_size,
        )
        for _ in range(Dy)
    ]

    for j in range(Dy):
        gps[j].fit(X, Y[:, j])
        print(f"[TRAIN] SVGP[{j}] trained for '{y_names[j]}' with {len(X)} samples.")

    return gps, X, Y, y_names


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

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(flip, rate, x, vx, u, ep, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info("Started SVGP retraining thread.")
        return True




    def _train_worker(self, pitch, pitch_dot, xpos, xpos_dot, u, ep, n_at_start: int):
        t0 = time.perf_counter()

        signals = {
            "xpos": xpos,
            "xpos_dot": xpos_dot,
            "pitch": pitch,
            "pitch_dot": pitch_dot,
            "u": u,
        }

        ep_min = int(np.min(ep)) if len(ep) > 0 else -1
        ep_max = int(np.max(ep)) if len(ep) > 0 else -1

        input_ = ["xpos", "xpos_dot", "pitch", "pitch_dot", "u"]
        output_ = ["xpos", "xpos_dot", "pitch", "pitch_dot"]
        model_mode = cfg_params.gp.type_of_data

        paths = [
            self.cfg.gp_xpos_path,
            self.cfg.gp_xpos_dot_path,
            self.cfg.gp_pitch_path,
            self.cfg.gp_pitch_dot_path,
        ]

        try:
            # -------------------------------------------------
            # Build WINDOW dataset (for FULL retrain)
            # -------------------------------------------------
            signals_window = {k: _as_2d(v) for k, v in signals.items()}
            signals_window = _align_signals(signals_window)
            ep_window = np.asarray(ep, dtype=np.int64).reshape(-1)[: next(iter(signals_window.values())).shape[0]]

            # -------------------------------------------------
            # Build ONLY the latest-episode dataset (for PARTIAL update)
            # -------------------------------------------------
            latest_ep = int(ep_window.max()) if ep_window.size > 0 else 0
            latest_mask = (ep_window == latest_ep)

            if latest_mask.sum() < 2:
                raise ValueError(f"Latest episode {latest_ep} has fewer than 2 samples.")

            signals_latest = {
                k: np.asarray(v)[latest_mask] for k, v in signals_window.items()
            }
            signals_latest = _align_signals(signals_latest)

            ep_latest = ep_window[latest_mask]
            Nl = next(iter(signals_latest.values())).shape[0]
            valid_latest = _valid_mask(Nl, ep_latest)

            X_latest = _build_X(signals_latest, input_, valid_latest)
            Y_latest, _ = _build_Y(signals_latest, output_, self.cfg.ctrl_dt, model_mode, valid_latest)

            if X_latest.shape[0] == 0:
                raise ValueError(f"No valid transitions for latest episode {latest_ep}.")

            # -------------------------------------------------
            # Decide: periodic full retrain or warm update
            # -------------------------------------------------
            full_every = int(getattr(self.cfg, "full_retrain_every_episodes", 0))

            model_files_exist = all(os.path.exists(p) for p in paths)
            do_full_retrain = (not model_files_exist) or (full_every > 0 and ((latest_ep + 1) % full_every == 0))

            if do_full_retrain:
                gps, X_used, Y_used, y_names = train_dynamics_gp_from_arrays(
                    signals_new=signals,
                    dt=self.cfg.ctrl_dt,
                    input_keys=input_,
                    output_keys=output_,
                    episode_id=ep,
                    target_mode=model_mode,
                    kernel=self.cfg.train_kernel,
                    lr=self.cfg.train_lr,
                    iters=self.cfg.train_iters,
                    num_inducing=self.cfg.train_num_inducing,
                    batch_size=self.cfg.train_batch_size,
                    seed_npz_path=self.cfg.seed_npz_path,
                    keep_seed=self.cfg.keep_seed,
                )

                n_used_for_log = len(X_used)
                mode_used = "FULL"

            else:
                # -------------------------------------------------
                # Warm-update the CURRENT saved SVGPs
                # -------------------------------------------------
                gp_xpos = SVGPManager.load(self.cfg.gp_xpos_path, device=self.device)
                gp_xpos_dot = SVGPManager.load(self.cfg.gp_xpos_dot_path, device=self.device)
                gp_pitch = SVGPManager.load(self.cfg.gp_pitch_path, device=self.device)
                gp_pitch_dot = SVGPManager.load(self.cfg.gp_pitch_dot_path, device=self.device)

                # online_steps = how long to update
                # replay_size = how much old data to mix in
                # max_keep = how much total history to retain

                online_steps = int(getattr(self.cfg, "online_update_steps", 50))
                replay_size = int(getattr(self.cfg, "online_replay_size", 256))
                max_keep = getattr(self.cfg, "online_max_keep_points", None)

                gp_xpos.partial_fit(
                    X_latest, Y_latest[:, 0],
                    steps=online_steps,
                    replay_size=replay_size,
                    batch_size=self.cfg.train_batch_size,
                    max_keep=max_keep,
                )
                gp_xpos_dot.partial_fit(
                    X_latest, Y_latest[:, 1],
                    steps=online_steps,
                    replay_size=replay_size,
                    batch_size=self.cfg.train_batch_size,
                    max_keep=max_keep,
                )
                gp_pitch.partial_fit(
                    X_latest, Y_latest[:, 2],
                    steps=online_steps,
                    replay_size=replay_size,
                    batch_size=self.cfg.train_batch_size,
                    max_keep=max_keep,
                )
                gp_pitch_dot.partial_fit(
                    X_latest, Y_latest[:, 3],
                    steps=online_steps,
                    replay_size=replay_size,
                    batch_size=self.cfg.train_batch_size,
                    max_keep=max_keep,
                )

                gps = [gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot]
                n_used_for_log = len(X_latest)
                mode_used = "PARTIAL"

            assert len(gps) == len(paths)

            # -------------------------------------------------
            # Atomic save
            # -------------------------------------------------
            for gp, out_path in zip(gps, paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            self.last_train_size = n_at_start

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"SVGP {mode_used} update finished in {elapsed:.2f}s | "
                    f"N_used={n_used_for_log} | episodes=[{ep_min}, {ep_max}] | "
                    f"kernel={self.cfg.train_kernel} | "
                    f"iters={self.cfg.train_iters} | num_inducing={self.cfg.train_num_inducing} | "
                    f"batch_size={self.cfg.train_batch_size}"
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
        Returns (gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot) or None.
        """
        if not self.reload_pending or self.training:
            return None

        with self.model_lock:
            gp_pitch = SVGPManager.load(self.cfg.gp_pitch_path)
            gp_pitch_dot = SVGPManager.load(self.cfg.gp_pitch_dot_path)
            gp_xpos = SVGPManager.load(self.cfg.gp_xpos_path)
            gp_xpos_dot = SVGPManager.load(self.cfg.gp_xpos_dot_path)

            gp_pitch.device = self.device
            gp_pitch_dot.device = self.device
            gp_xpos.device = self.device
            gp_xpos_dot.device = self.device

        self.reload_pending = False

        try:
            stats = []
            for p in [
                self.cfg.gp_xpos_path,
                self.cfg.gp_xpos_dot_path,
                self.cfg.gp_pitch_path,
                self.cfg.gp_pitch_dot_path,
            ]:
                stats.append((p, os.path.getmtime(p), os.path.getsize(p)))
            self.last_reload_stats = stats
        except Exception:
            self.last_reload_stats = None

        if self.logger:
            self.logger.info("Reloaded SVGP models (hot swap).")

        return gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot