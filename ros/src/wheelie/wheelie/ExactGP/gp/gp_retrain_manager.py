# utils/gp_retrain_manager.py
import os
import time
import threading
import traceback
import numpy as np

from typing import Dict, List, Optional, Tuple

import os
import numpy as np

from gp.gp_dynamics import GPManager
# from gp.re_train_dynamics_gp import train_dynamics_gp_from_arrays


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


def _load_seed_npz(
    seed_npz_path: str,
    keys_needed: List[str],
    dt: float,
    seed_episode_id: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
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
# MAIN MODULAR API (this is what you wanted)
# --------------------------------------------------------------
def train_dynamics_gp_from_arrays(
    signals_new: Dict[str, np.ndarray],
    dt: float,
    input_keys: List[str],
    output_keys: List[str],
    episode_id: Optional[np.ndarray] = None,
    target_mode: str = "derivative",
    kernel: str = "RQ",
    iters: int = 300,
    seed_npz_path: Optional[str] = None,
    seed_episode_id: int = -1,
    keep_seed: bool = True,
) -> Tuple[List[GPManager], np.ndarray, np.ndarray, List[str]]:
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
    print(f"[TRAIN DATA] X={X.shape} | Y={Y.shape} | mode={target_mode} | kernel={kernel} | iters={iters}")
    print(f"[TRAIN DATA] inputs={input_keys}")
    print(f"[TRAIN DATA] outputs={output_keys}")

    # Train 1 GP per output dimension
    Dy = Y.shape[1]
    gps = [GPManager(kernel=kernel, iters=iters) for _ in range(Dy)]
    for j in range(Dy):
        gps[j].fit(X, Y[:, j])
        print(f"[TRAIN] GP[{j}] trained for '{y_names[j]}' with {len(X)} samples.")

    return gps, X, Y, y_names


def _configured_keys(cfg, cfg_attr: str, yaml_attr: str) -> List[str]:
    value = getattr(cfg, cfg_attr, None)
    if value is None:
        value = getattr(cfg_params.gp, yaml_attr, None)
    if value is None:
        raise ValueError(f"No GP keys configured for {cfg_attr}/{yaml_attr}.")
    keys = list(value)
    if not keys:
        raise ValueError(f"No GP keys configured for {cfg_attr}/{yaml_attr}.")
    return keys


def _required_signal_keys(input_keys: List[str], output_keys: List[str]) -> List[str]:
    return list(dict.fromkeys([*input_keys, *output_keys]))


def _target_base_name(y_name: str) -> str:
    for suffix in ("_d_dt", "_delta", "_next"):
        if y_name.endswith(suffix):
            return y_name[: -len(suffix)]
    return y_name



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
        self.last_y_names = None
    
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

        input_keys, output_keys = self._training_keys()
        required_keys = _required_signal_keys(input_keys, output_keys)

        signals, ep = self._snapshot_dataset(dataset)
        self._check_required_signals(signals, required_keys, context="online dataset")

        dataset.save_npz_from_dict(episode_id, signals, ep)

        # cap training window
        M = int(self.cfg.max_points_for_train)
        signals, ep = dataset.cap_window_dict(M, signals, ep)
        train_signals = {k: signals[k] for k in required_keys}

        if self.logger:
            shapes = ", ".join(f"{k}{np.asarray(v).shape}" for k, v in train_signals.items())
            self.logger.info(
                f"Preparing GP retrain | inputs={input_keys} | outputs={output_keys} | "
                f"signals={shapes}"
            )

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(train_signals, ep, input_keys, output_keys, n_at_start),
            daemon=True,
        )
        self.train_thread.start()

        if self.logger:
            self.logger.info("Started GP retraining thread.")
        return True

    def _training_keys(self) -> Tuple[List[str], List[str]]:
        input_keys = _configured_keys(self.cfg, "gp_input_keys", "input_keys")
        output_keys = _configured_keys(self.cfg, "gp_output_keys", "output_keys")
        return input_keys, output_keys

    def _snapshot_dataset(self, dataset) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        if not hasattr(dataset, "snapshot_dict"):
            raise TypeError(
                "Dataset object must implement snapshot_dict() returning "
                "(signals: dict[str, ndarray], episode_id: ndarray)."
            )
        signals, ep = dataset.snapshot_dict()
        return dict(signals), np.asarray(ep, dtype=np.int64)

    @staticmethod
    def _check_required_signals(signals: Dict[str, np.ndarray], required_keys: List[str], context: str):
        missing = [k for k in required_keys if k not in signals]
        if missing:
            raise KeyError(
                f"Missing {context} signals {missing}. "
                f"Needed={required_keys}. Available={list(signals.keys())}"
            )

    def _model_paths_for_outputs(self, y_names: List[str]) -> List[str]:
        model_paths = getattr(self.cfg, "gp_model_paths", None)
        paths = []

        for y_name in y_names:
            candidates = [y_name, _target_base_name(y_name)]
            path = None

            if model_paths is not None:
                for key in candidates:
                    if key in model_paths:
                        path = model_paths[key]
                        break

            if path is None:
                for key in candidates:
                    attr = f"gp_{key}_path"
                    if hasattr(self.cfg, attr):
                        path = getattr(self.cfg, attr)
                        break

            if path is None:
                raise KeyError(
                    f"No model path configured for output '{y_name}'. "
                    f"Tried keys/attrs: {candidates} and gp_<key>_path."
                )

            paths.append(str(path))

        return paths

    def _train_worker(self,
        signals: Dict[str, np.ndarray],
        ep: np.ndarray,
        input_keys: List[str],
        output_keys: List[str],
        n_at_start: int,
    ):
        t0 = time.perf_counter()
        model_mode = getattr(self.cfg, "gp_target_mode", cfg_params.gp.type_of_data)
        seed_npz_path = self.cfg.seed_npz_path

        if self.cfg.keep_seed and seed_npz_path and not os.path.exists(seed_npz_path):
            if self.logger:
                self.logger.warn(f"Seed NPZ not found; retraining only online data: {seed_npz_path}")

        try:
            gps, X, Y, y_names = train_dynamics_gp_from_arrays(
                signals_new=signals,
                dt=self.cfg.ctrl_dt,
                input_keys=input_keys,
                output_keys=output_keys,
                episode_id=ep,
                target_mode=model_mode,
                kernel=self.cfg.train_kernel,
                iters=self.cfg.train_iters,
                seed_npz_path=seed_npz_path,
                keep_seed=self.cfg.keep_seed,
            )

            paths = self._model_paths_for_outputs(y_names)
            if len(gps) != len(paths):
                raise RuntimeError(f"Expected {len(paths)} GP models, got {len(gps)}.")

            # Atomic save
            for gp, out_path in zip(gps, paths):
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp_path = out_path + ".tmp"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            # commit size only after success
            self.last_train_size = n_at_start
            self.last_y_names = list(y_names)

            elapsed = time.perf_counter() - t0
            if self.logger:
                self.logger.info(
                    f"GP retraining finished in {elapsed:.2f}s | "
                    f"N_used={len(X)} | kernel={self.cfg.train_kernel} | iters={self.cfg.train_iters}"
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
        Returns a dict mapping output signal names to loaded GPManager objects,
        or None.
        """
        if not self.reload_pending or self.training:
            return None

        y_names = self.last_y_names
        if y_names is None:
            _, output_keys = self._training_keys()
            y_names = output_keys

        paths = self._model_paths_for_outputs(y_names)

        with self.model_lock:
            loaded_models = {
                _target_base_name(y_name): GPManager.load(path, device=self.device)
                for y_name, path in zip(y_names, paths)
            }

        self.reload_pending = False

        # stats (optional)
        try:
            stats = []
            for p in paths:
                stats.append((p, os.path.getmtime(p), os.path.getsize(p)))
            self.last_reload_stats = stats
        except Exception:
            self.last_reload_stats = None

        if self.logger:
            self.logger.info("Reloaded GP models (hot swap).")

        return loaded_models
