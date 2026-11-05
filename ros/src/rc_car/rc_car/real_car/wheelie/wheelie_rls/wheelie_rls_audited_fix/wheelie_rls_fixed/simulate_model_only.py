#!/usr/bin/env python3
"""Controller consistency test on its OWN model. Not a real-car validation."""
import argparse
import json
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from params_mppi import WheelieParams, MPPIConfig
from rls import rls_step_torch
from mppi import MPPITorch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent/'wheelie_nn'/'rls_learned_weights_v2.npz')
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.manual_seed(42)
    cfg = MPPIConfig()
    with np.load(args.model, allow_pickle=False) as data:
        a, cfg.input_delay_steps = data['a'].copy(), int(data['input_delay_steps'])
    ctrl = MPPITorch(WheelieParams(), cfg, a, device=args.device)
    parameters = torch.tensor(a, dtype=torch.float64)
    state, ref = np.zeros(4), np.array((0, 0, np.deg2rad(-55), 0))
    history, u_prev, records = deque([0.0]*cfg.input_delay_steps, maxlen=cfg.input_delay_steps), 0.0, []
    for k in range(args.steps):
        u, info = ctrl.solve(state, ref, u_prev, list(history))
        records.append({'t': k*cfg.dt, 'v': state[1], 'theta_deg': np.rad2deg(state[2]), 'omega': state[3], 'u': u, 'success': info['success'], 'pred_theta_N_deg': info.get('pred_theta_N_deg', np.nan)})
        effective = history[0] if cfg.input_delay_steps else u
        state = rls_step_torch(torch.tensor(state)[None], torch.tensor([effective], dtype=torch.float64), cfg.dt, parameters)[0].numpy()
        history.append(u)
        u_prev = u
    output = Path(__file__).parent/'audit'
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output/'model_only_closed_loop.csv', index=False)
    result = {'test': 'controller_on_its_own_model_only', 'physical_validation': False, 'steps': args.steps, 'failed_solves': int((~frame.success).sum()), 'last_second_pitch_rmse_deg': float(np.sqrt(np.mean((frame.theta_deg.tail(round(1/cfg.dt))+55)**2))), 'final_pitch_deg': float(np.rad2deg(state[2])), 'final_v': float(state[1])}
    (output/'model_only_closed_loop.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
