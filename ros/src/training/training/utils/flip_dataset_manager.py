#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict
import os
import numpy as np


@dataclass
class FlipEpisodeDataset:
    """
    Holds a single episode of data for the flip task and can append it to
    a global NPZ dataset (e.g. mujoco_random_run.npz).

    Only stores the signals used by training:
      - 'flip' : flip_rel
      - 'rate' : pitch rate
      - 'u'    : control input
    """
    flip: List[float] = field(default_factory=list)
    rate: List[float] = field(default_factory=list)
    u:    List[float] = field(default_factory=list)

    # --------- logging API ---------

    def reset(self) -> None:
        """Clear the current episode buffer."""
        self.flip.clear()
        self.rate.clear()
        self.u.clear()

    def log_step(self, flip: float, rate: float, u: float) -> None:
        """Log a single timestep."""
        self.flip.append(float(flip))
        self.rate.append(float(rate))
        self.u.append(float(u))

    # --------- NPZ interaction ---------

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Return the current episode as numpy arrays."""
        return {
            "flip": np.asarray(self.flip, dtype=np.float32),
            "rate": np.asarray(self.rate, dtype=np.float32),
            "u":    np.asarray(self.u,    dtype=np.float32),
        }

    def append_to_npz(self, path: str) -> int:
        """
        Append this episode to the NPZ at `path`.

        - If the file does not exist, it is created.
        - If it exists, 'flip', 'rate', 'u' are concatenated.
        - The file is overwritten.
        - Prints old size, added size, and new total size.
        - Returns the new total number of samples (len(flip)) in the file.
        """
        new_data = self.as_arrays()
        N_new = new_data["flip"].shape[0]

        if N_new == 0:
            print("[FlipEpisodeDataset] No samples to append (episode empty).")
            return 0

        if os.path.exists(path):
            with np.load(path) as D:
                data = {k: D[k] for k in D.files}
            old_N = int(data.get("flip", np.zeros(0, dtype=np.float32)).shape[0])

            for key in ("flip", "rate", "u"):
                if key in data:
                    data[key] = np.concatenate([data[key], new_data[key]], axis=0)
                else:
                    data[key] = new_data[key]
        else:
            data = new_data
            old_N = 0

        np.savez(path, **data)

        total_N = int(data["flip"].shape[0])
        print(
            f"[FlipEpisodeDataset] {os.path.basename(path)}:"
            f" old N={old_N}, added={N_new}, total N={total_N}"
        )
        return total_N
