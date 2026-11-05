#!/usr/bin/env python3
"""Offline wall-clock benchmark; no ROS and no commands sent.

CUDA solves return a host scalar and trajectory, so the measured interval
includes completion of those device operations. First warm-up is excluded.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from params_mppi import MPPIConfig, WheelieParams
from mppi import MPPITorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent/'wheelie_nn'/'rls_learned_weights_v2.npz')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    cfg = MPPIConfig()
    cfg.compile_rollouts = args.compile
    with np.load(args.model, allow_pickle=False) as data:
        a = data['a'].copy()
        cfg.input_delay_steps = int(data['input_delay_steps'])
    controller = MPPITorch(WheelieParams(), cfg, a, device=args.device)
    reference = np.array((0.0, 0.0, np.deg2rad(-55.0), 0.0))
    controller.solve(np.zeros(4), reference, 0.0, [0.0]*cfg.input_delay_steps)
    durations, statuses = [], []
    for _ in range(args.runs):
        begin = time.perf_counter()
        u, info = controller.solve(np.zeros(4), reference, 0.0, [0.0]*cfg.input_delay_steps)
        durations.append(1000*(time.perf_counter()-begin))
        statuses.append(info.get('success', False))
    result = {'device': args.device, 'torch_version': torch.__version__, 'compiled': args.compile, 'K': cfg.K, 'N': cfg.N, 'dt_ms': cfg.dt*1000, 'runs': args.runs, 'median_ms': float(np.median(durations)), 'p95_ms': float(np.percentile(durations, 95)), 'max_ms': float(np.max(durations)), 'successful_solves': sum(statuses), 'hardware_control_test': False}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
