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
        self.log_xpos    = deque(maxlen=maxlen)
        self.log_xpos_dot   = deque(maxlen=maxlen)
        self.log_pitch = deque(maxlen=maxlen)
        self.log_pitch_dot = deque(maxlen=maxlen)
        self.log_u    = deque(maxlen=maxlen)
        self.log_ep   = deque(maxlen=maxlen)


    def get_transition_pairs(self):
        """
        Return same-episode consecutive transition pairs.

        Returns:
            s0: np.ndarray [N,4]
            u0: np.ndarray [N,1]
            s1: np.ndarray [N,4]
            ep0: np.ndarray [N]
        """
        pitch, pitch_dot, xpos, xpos_dot, u, ep = self.snapshot()
        if len(pitch) < 2:
            return None

        same_ep = (ep[:-1] == ep[1:])
        if not np.any(same_ep):
            return None

        s0 = np.stack(
            [xpos[:-1][same_ep], xpos_dot[:-1][same_ep], pitch[:-1][same_ep], pitch_dot[:-1][same_ep]],
            axis=1,
        ).astype(np.float32)

        s1 = np.stack(
            [xpos[1:][same_ep], xpos_dot[1:][same_ep], pitch[1:][same_ep], pitch_dot[1:][same_ep]],
            axis=1,
        ).astype(np.float32)

        u0 = u[:-1][same_ep].reshape(-1, 1).astype(np.float32)
        ep0 = ep[:-1][same_ep].astype(np.int64)
        return s0, u0, s1, ep0




    def n_points(self) -> int:
        with self.lock:
            return len(self.log_pitch)

    def append_step(self, pitch: float, pitch_dot: float, xpos: float, xpos_dot: float, u: float, episode_id: int):
        with self.lock:
            self.log_xpos.append(float(xpos))
            self.log_xpos_dot.append(float(xpos_dot))
            self.log_pitch.append(float(pitch))
            self.log_pitch_dot.append(float(pitch_dot))
            self.log_u.append(float(u))
            self.log_ep.append(int(episode_id))

    def snapshot(self):
        with self.lock:
            xpos    = np.asarray(list(self.log_xpos), dtype=np.float32)
            xpos_dot   = np.asarray(list(self.log_xpos_dot), dtype=np.float32)
            pitch = np.asarray(list(self.log_pitch), dtype=np.float32)
            pitch_dot = np.asarray(list(self.log_pitch_dot), dtype=np.float32)
            u    = np.asarray(list(self.log_u), dtype=np.float32)
            ep   = np.asarray(list(self.log_ep), dtype=np.int64)
        return pitch, pitch_dot, xpos, xpos_dot, u, ep

    def save_npz(self, episode_id: int, pitch, pitch_dot, xpos, xpos_dot, u, ep):
        out = os.path.join(self.log_dir, f"dataset_ep{int(episode_id):04d}.npz")
        np.savez_compressed(
            out,
            xpos=xpos,
            xpos_dot=xpos_dot,
            pitch=pitch,
            pitch_dot=pitch_dot,
            u=u,
            episode_id=ep,
            dt=np.array(self.ctrl_dt, dtype=np.float32),
        )
        if self.logger is not None:
            self.logger.info(f"Saved dataset snapshot: {out}")
        return out

    @staticmethod
    def cap_window(M: int, pitch, pitch_dot, xpos, xpos_dot, u, ep):
        if len(pitch) <= M:
            return pitch, pitch_dot, xpos, xpos_dot, u, ep
        return pitch[-M:], pitch_dot[-M:], xpos[-M:], xpos_dot[-M:], u[-M:], ep[-M:]
