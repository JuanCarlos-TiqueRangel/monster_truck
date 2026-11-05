#!/usr/bin/env python3
"""Reproduce the original best-sample/returned-plan discrepancy, OFFLINE ONLY.

The only source alteration is replacing the original forced CUDA device with
CPU, because CUDA was unavailable in the audit environment. This script runs
archived source files supplied by the user, never a ROS node.
"""
import json
import sys
import types
from pathlib import Path
import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parent
    originals = root/'audit'/'originals'
    for name in ('params_mppi', 'rls', 'mppi'):
        path = originals/f'{name}.py'
        source = path.read_text()
        if name == 'mppi':
            old = 'self.device = torch.device("cuda")'
            if source.count(old) != 1:
                raise ValueError('Original device line has changed; review the source before replay.')
            source = source.replace(old, 'self.device = torch.device("cpu")')
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
    config, mppi = sys.modules['params_mppi'], sys.modules['mppi']
    with np.load(root/'audit'/'original_script_001_to_145.npz', allow_pickle=False) as data:
        a = data['a'].copy()
    torch.set_num_threads(1)
    records = []
    for seed in range(5):
        torch.manual_seed(seed)
        controller = mppi.MPPITorch(config.WheelieParams(), config.MPPIConfig(), a, live_plot=False)
        u, info = controller.solve(np.zeros(4), np.array((0., 0., np.deg2rad(-55), 0.)), 0.)
        state = torch.zeros((1, 4), dtype=controller.dtype)
        with torch.no_grad():
            for command in controller.last_solution:
                state = controller._step(state, command.reshape(1), controller.cfg.dt)
        records.append(dict(seed=seed, u=u, logged_best_theta_N_deg=info['best_theta_N_deg'], returned_plan_theta_N_deg=float(torch.rad2deg(state[0, 2])), plan_u_min=float(controller.last_solution.min()), plan_u_max=float(controller.last_solution.max())))
    output = root/'audit'/'original_mean_vs_best.json'
    output.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
