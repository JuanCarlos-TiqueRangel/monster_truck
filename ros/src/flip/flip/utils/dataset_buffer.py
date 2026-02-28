# utils/dataset_buffer.py
import os
import threading
from collections import deque
import numpy as np


class DatasetBuffer:
    """
    Flip-only dataset buffer.

    We keep the public method signature:
        append_step(flip_rel, rate, x, vx, u, episode_id)
    for backwards compatibility with your existing code,
    but semantically:
        flip_rel -> up_z
        rate     -> up_z_dot
        u        -> action (motor command)
    """

    def __init__(self, maxlen: int, log_dir: str, ctrl_dt: float, logger=None):
        self.logger = logger
        self.log_dir = str(log_dir)
        self.ctrl_dt = float(ctrl_dt)
        os.makedirs(self.log_dir, exist_ok=True)

        self.lock = threading.Lock()

        self.log_up_z     = deque(maxlen=maxlen)
        self.log_up_z_dot = deque(maxlen=maxlen)
        self.log_a        = deque(maxlen=maxlen)
        self.log_ep       = deque(maxlen=maxlen)

    def n_points(self) -> int:
        with self.lock:
            return len(self.log_up_z)

    def append_step(self, up_z: float, up_z_dot: float, u: float, episode_id: int):
        # flip_rel == up_z, rate == up_z_dot
        with self.lock:
            self.log_up_z.append(float(up_z))
            self.log_up_z_dot.append(float(up_z_dot))
            self.log_a.append(float(u))
            self.log_ep.append(int(episode_id))

    def snapshot(self):
        with self.lock:
            up_z     = np.asarray(self.log_up_z, dtype=np.float32)
            up_z_dot = np.asarray(self.log_up_z_dot, dtype=np.float32)
            a        = np.asarray(self.log_a, dtype=np.float32)
            ep       = np.asarray(self.log_ep, dtype=np.int64)

        # For compatibility with existing retrain manager naming:
        # returns (flip, rate, x, vx, u, ep)
        return up_z, up_z_dot, a, ep

    def save_npz(self, episode_id: int, up_z, up_z_dot, u, ep):
        """
        Saves a snapshot. In flip-only meaning:
            flip -> up_z
            rate -> up_z_dot
            u    -> action a
        """
        out = os.path.join(self.log_dir, f"flip_dataset_ep{int(episode_id):04d}.npz")
        np.savez_compressed(
            out,
            up_z=up_z,
            up_z_dot=up_z_dot,
            a=u,
            episode_id=ep,
            dt=np.array(self.ctrl_dt, dtype=np.float32),
        )
        if self.logger is not None:
            self.logger.info(f"Saved dataset snapshot: {out}")
        return out

    @staticmethod
    def cap_window(M: int, up_z, up_z_dot, u, ep):
        if len(up_z) <= M:
            return up_z, up_z_dot, u, ep
        return up_z[-M:], up_z_dot[-M:], u[-M:], ep[-M:]