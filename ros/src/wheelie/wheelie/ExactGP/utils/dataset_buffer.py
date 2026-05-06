# utils/dataset_buffer.py
import os
import threading
from collections import deque
from typing import Dict, Iterable, Optional, Sequence

import numpy as np


class DatasetBuffer:
    def __init__(
        self,
        maxlen: int,
        log_dir: str,
        ctrl_dt: float,
        signal_keys: Sequence[str],
        logger=None,
    ):
        self.logger = logger
        self.log_dir = str(log_dir)
        self.ctrl_dt = float(ctrl_dt)
        os.makedirs(self.log_dir, exist_ok=True)

        self.signal_keys = list(dict.fromkeys(signal_keys))
        if not self.signal_keys:
            raise ValueError("DatasetBuffer requires at least one signal key.")

        self.lock = threading.Lock()
        self.log_signals = {
            key: deque(maxlen=maxlen)
            for key in self.signal_keys
        }
        self.log_ep = deque(maxlen=maxlen)


    def get_transition_pairs(self, state_keys: Sequence[str], action_keys: Optional[Sequence[str]] = None):
        """
        Return same-episode consecutive transition pairs.

        Returns:
            s0: np.ndarray [N, len(state_keys)]
            u0: np.ndarray [N, len(action_keys)] or None
            s1: np.ndarray [N, len(state_keys)]
            ep0: np.ndarray [N]
        """
        action_keys = [] if action_keys is None else list(action_keys)
        state_keys = list(state_keys)

        signals, ep = self.snapshot_dict(keys=list(dict.fromkeys([*state_keys, *action_keys])))
        if len(ep) < 2:
            return None

        same_ep = (ep[:-1] == ep[1:])
        if not np.any(same_ep):
            return None

        s0 = np.stack(
            [signals[k][:-1][same_ep] for k in state_keys],
            axis=1,
        ).astype(np.float32)

        s1 = np.stack(
            [signals[k][1:][same_ep] for k in state_keys],
            axis=1,
        ).astype(np.float32)

        if action_keys:
            u0 = np.stack(
                [signals[k][:-1][same_ep] for k in action_keys],
                axis=1,
            ).astype(np.float32)
        else:
            u0 = None

        ep0 = ep[:-1][same_ep].astype(np.int64)
        return s0, u0, s1, ep0

    def n_points(self) -> int:
        with self.lock:
            return len(self.log_ep)

    def append_step(self, episode_id: int, signals: Optional[Dict[str, float]] = None, **signal_values):
        """
        Append one sample.

        Preferred call:
            append_step(episode_id=ep, signals={name: value, ...})

        Keyword signal values are also accepted for convenience.
        """
        sample = {}
        if signals is not None:
            sample.update(signals)
        sample.update(signal_values)

        missing = [k for k in self.signal_keys if k not in sample]
        if missing:
            raise KeyError(
                f"DatasetBuffer append missing signals {missing}. "
                f"Required={self.signal_keys}. Provided={list(sample.keys())}"
            )

        with self.lock:
            for key in self.signal_keys:
                self.log_signals[key].append(float(sample[key]))
            self.log_ep.append(int(episode_id))


    def drop_episode(self, episode_id: int) -> int:
        """
        Remove all samples that belong to one episode.

        Returns:
            number of removed samples
        """
        with self.lock:
            old_signals = {
                key: list(values)
                for key, values in self.log_signals.items()
            }
            old_ep = list(self.log_ep)

            keep_idx = [i for i, ep in enumerate(old_ep) if ep != int(episode_id)]
            removed = len(old_ep) - len(keep_idx)

            if removed == 0:
                return 0

            maxlen = self.log_ep.maxlen

            self.log_signals = {
                key: deque((values[i] for i in keep_idx), maxlen=maxlen)
                for key, values in old_signals.items()
            }
            self.log_ep = deque((old_ep[i] for i in keep_idx), maxlen=maxlen)

            if self.logger is not None:
                self.logger.info(f"Dropped episode {episode_id} from dataset buffer ({removed} samples removed).")

            return removed


    def snapshot(self, keys: Optional[Iterable[str]] = None):
        signals, ep = self.snapshot_dict(keys=keys)
        ordered_keys = self.signal_keys if keys is None else list(keys)
        return tuple(signals[k] for k in ordered_keys) + (ep,)

    def snapshot_dict(self, keys: Optional[Iterable[str]] = None):
        keys = self.signal_keys if keys is None else list(keys)
        self._check_known_keys(keys, context="snapshot")
        with self.lock:
            signals = {
                key: np.asarray(list(self.log_signals[key]), dtype=np.float32)
                for key in keys
            }
            ep = np.asarray(list(self.log_ep), dtype=np.int64)
        return signals, ep

    def save_npz(self, episode_id: int, signals: dict, ep, keys=None):
        return self.save_npz_from_dict(episode_id, signals, ep, keys=keys)

    def save_npz_from_dict(self, episode_id: int, signals: dict, ep, keys=None):
        keys = list(signals.keys()) if keys is None else list(keys)
        missing = [k for k in keys if k not in signals]
        if missing:
            raise KeyError(f"Cannot save dataset snapshot. Missing signals: {missing}")

        out = os.path.join(self.log_dir, f"dataset_ep{int(episode_id):04d}.npz")
        payload = {
            k: np.asarray(signals[k], dtype=np.float32)
            for k in keys
        }
        payload.update(
            episode_id=np.asarray(ep, dtype=np.int64),
            dt=np.array(self.ctrl_dt, dtype=np.float32),
        )
        np.savez_compressed(out, **payload)
        if self.logger is not None:
            self.logger.info(f"Saved dataset snapshot: {out}")
        return out

    @staticmethod
    def cap_window(M: int, signals: dict, ep):
        return DatasetBuffer.cap_window_dict(M, signals, ep)

    @staticmethod
    def cap_window_dict(M: int, signals: dict, ep):
        if len(ep) <= M:
            return signals, ep
        return {
            k: np.asarray(v)[-M:]
            for k, v in signals.items()
        }, np.asarray(ep)[-M:]

    def _check_known_keys(self, keys: Iterable[str], context: str):
        missing = [k for k in keys if k not in self.log_signals]
        if missing:
            raise KeyError(
                f"Unknown signals for {context}: {missing}. "
                f"Available={self.signal_keys}"
            )
