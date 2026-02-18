# utils/dataset_buffer.py
import os
import threading
from collections import deque

import numpy as np


class DatasetBuffer:
    def __init__(self, maxlen: int, log_dir: str, ctrl_dt: float, logger=None):
        self.logger = logger
        self.log_dir = str(log_dir)
        self.ctrl_dt = float(ctrl_dt)
        os.makedirs(self.log_dir, exist_ok=True)

        self.lock = threading.Lock()
        self.log_flip = deque(maxlen=maxlen)
        self.log_rate = deque(maxlen=maxlen)
        self.log_u    = deque(maxlen=maxlen)
        self.log_ep   = deque(maxlen=maxlen)
        self.log_x    = deque(maxlen=maxlen)
        self.log_vx   = deque(maxlen=maxlen)

    def n_points(self) -> int:
        with self.lock:
            return len(self.log_flip)

    def append_step(self, flip_rel: float, rate: float, x: float, vx: float, u: float, episode_id: int):
        with self.lock:
            self.log_flip.append(float(flip_rel))
            self.log_rate.append(float(rate))
            self.log_x.append(float(x))
            self.log_vx.append(float(vx))
            self.log_u.append(float(u))
            self.log_ep.append(int(episode_id))

    def snapshot(self):
        with self.lock:
            flip = np.asarray(list(self.log_flip), dtype=np.float32)
            rate = np.asarray(list(self.log_rate), dtype=np.float32)
            x    = np.asarray(list(self.log_x), dtype=np.float32)
            vx   = np.asarray(list(self.log_vx), dtype=np.float32)
            u    = np.asarray(list(self.log_u), dtype=np.float32)
            ep   = np.asarray(list(self.log_ep), dtype=np.int64)
        return flip, rate, x, vx, u, ep

    def save_npz(self, episode_id: int, flip, rate, x, vx, u, ep):
        out = os.path.join(self.log_dir, f"dataset_ep{int(episode_id):04d}.npz")
        np.savez_compressed(
            out,
            flip=flip,
            rate=rate,
            x_pose=x,
            linear_speed_x=vx,
            u=u,
            episode_id=ep,
            dt=np.array(self.ctrl_dt, dtype=np.float32),
        )
        if self.logger is not None:
            self.logger.info(f"Saved dataset snapshot: {out}")
        return out

    @staticmethod
    def cap_window(M: int, flip, rate, x, vx, u, ep):
        if len(flip) <= M:
            return flip, rate, x, vx, u, ep
        return flip[-M:], rate[-M:], x[-M:], vx[-M:], u[-M:], ep[-M:]
