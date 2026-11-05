#!/usr/bin/env python3
"""Frozen free-running validation, using the SAME rls_step_torch as MPPI.

No measured-state replacement after initialization, and no online weight
updates. A fixed start floor of four samples makes lag comparisons use
common windows. Report both screened and unscreened held-out windows.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from train_rls import load_experiments
from rls import MODEL_TYPE, rls_features_numpy, rls_step_torch


def evaluate(experiments, names, a, delay, steps, screened=True, device='cpu', start_floor=4):
    states, controls, truths, times, meta = [], [], [], [], []
    for name in names:
        d = experiments[name]
        for start in range(max(delay, start_floor), len(d['state'])-steps):
            idx = np.arange(start, start+steps)
            if screened and not d['valid'][idx].all():
                continue
            if not np.isfinite(d['state'][start:start+steps+1]).all() or not (d['dt'][idx] > 0).all():
                continue
            states.append(d['state'][start])
            controls.append(d['u'][start-delay:start+steps-delay])
            truths.append(d['state'][start+1:start+steps+1])
            times.append(d['dt'][idx])
            meta.append((name, start))
    if not states:
        return {'steps': steps, 'screened': screened, 'windows': 0}, pd.DataFrame(), None
    state = torch.as_tensor(np.array(states), dtype=torch.float64, device=device).clone()
    state[:, 2].clamp_(max=0.0)
    parameters = torch.as_tensor(a, dtype=state.dtype, device=state.device)
    controls = torch.as_tensor(np.array(controls), dtype=state.dtype, device=state.device)
    dt = torch.as_tensor(np.array(times), dtype=state.dtype, device=state.device)
    predictions = []
    with torch.no_grad():
        for k in range(steps):
            state = rls_step_torch(state, controls[:, k], dt[:, k], parameters)
            predictions.append(state.cpu().numpy().copy())
    prediction = np.stack(predictions, axis=1)
    truth = np.array(truths)
    error = prediction-truth
    finite = np.isfinite(error).all((1, 2))
    domain = (np.abs(prediction[:, :, 1]) <= 5).all(1) & (np.abs(prediction[:, :, 3]) <= 12).all(1) & (prediction[:, :, 2] >= -np.pi/2).all(1)
    with np.errstate(over='ignore', invalid='ignore'):
        rmse = np.sqrt(np.mean(error**2, axis=1))
    rows = pd.DataFrame({'file': [m[0] for m in meta], 'start': [m[1] for m in meta], 'steps': steps, 'screened': screened, 'finite': finite, 'within_model_domain': domain & finite, 'pitch_rmse_deg': np.rad2deg(rmse[:, 2]), 'final_pitch_error_deg': np.rad2deg(error[:, -1, 2]), 'v_rmse': rmse[:, 1], 'omega_rmse': rmse[:, 3]})
    usable = finite & np.isfinite(rmse).all(1)
    good_error = error[usable]
    balanced = rows.loc[usable].assign(mse=rows.loc[usable, 'pitch_rmse_deg']**2).groupby('file').mse.mean()
    summary = {'steps': steps, 'screened': screened, 'windows': len(meta), 'files': len(set(m[0] for m in meta)), 'nonfinite_windows': int((~usable).sum()), 'domain_violation_windows': int((~(domain & usable)).sum())}
    values = {'pitch_rmse_deg_finite_windows': None, 'file_balanced_pitch_rmse_deg': None, 'median_window_pitch_rmse_deg': None, 'p95_window_pitch_rmse_deg': None, 'v_rmse_finite_windows': None, 'omega_rmse_finite_windows': None}
    if usable.any():
        values.update(pitch_rmse_deg_finite_windows=float(np.rad2deg(np.sqrt(np.mean(good_error[:, :, 2]**2)))), file_balanced_pitch_rmse_deg=float(np.sqrt(balanced.mean())), median_window_pitch_rmse_deg=float(np.median(rows.loc[usable, 'pitch_rmse_deg'])), p95_window_pitch_rmse_deg=float(np.percentile(rows.loc[usable, 'pitch_rmse_deg'], 95)), v_rmse_finite_windows=float(np.sqrt(np.mean(good_error[:, :, 1]**2))), omega_rmse_finite_windows=float(np.sqrt(np.mean(good_error[:, :, 3]**2))))
    summary.update(values)
    return summary, rows, {'files': np.asarray([m[0] for m in meta]), 'starts': np.asarray([m[1] for m in meta]), 'prediction': prediction, 'truth': truth, 'dt': np.array(times)}


def equilibrium_diagnostic(a, theta_deg=-55.0):
    theta = np.deg2rad(theta_deg)
    def residual(z):
        v, u = z
        fv, fw = rls_features_numpy(np.array((0.0, v, theta, 0.0)), u)
        return np.array((fv @ a[:7], fw @ a[7:]))
    result = least_squares(residual, np.array((1.0, 0.2)), bounds=([0.0, 0.0], [5.0, 0.30]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    fv, fw = rls_features_numpy(np.array((0.0, 1.0, theta, 0.0)), 0.0)
    gain = float(a[7]+a[13]*np.cos(theta))
    root = float(-(fw @ a[7:])/gain) if abs(gain) > 1e-10 else None
    return {'reference_pitch_deg': theta_deg, 'angular_equilibrium_u_at_v_1': root, 'angular_input_gain_at_reference': gain, 'joint_trim_v': float(result.x[0]), 'joint_trim_u': float(result.x[1]), 'joint_trim_v_dot': float(result.fun[0]), 'joint_trim_omega_dot': float(result.fun[1]), 'model_has_joint_trim': bool(np.max(np.abs(result.fun)) < 1e-3), 'hardware_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent/'wheelie_nn'/'dataset')
    parser.add_argument('--model', type=Path, default=Path(__file__).parent/'wheelie_nn'/'rls_learned_weights_v2.npz')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'audit'/'validation')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    experiments = load_experiments(args.dataset)
    with np.load(args.model, allow_pickle=False) as data:
        if str(data['model_type'].item()) != MODEL_TYPE:
            raise ValueError('Expected the v2 model.')
        a, delay = np.array(data['a']), int(data['input_delay_steps'])
        names = [str(n) for n in data['holdout_files']]
        expected = dict(zip(names, map(str, data['holdout_sha256'])))
        if any(n not in experiments or experiments[n]['sha256'] != expected[n] for n in names):
            raise ValueError('Held-out files are missing or changed since training.')
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, frames = [], []
    for screened in (True, False):
        for steps in (1, 5, 10, 20):
            summary, rows, traces = evaluate(experiments, names, a, delay, steps, screened, args.device)
            summaries.append(summary)
            frames.append(rows)
            if traces is not None:
                np.savez_compressed(args.output/f'traces_{steps}_{"screened" if screened else "all"}.npz', **traces)
            print(json.dumps(summary))
    metrics = {'input_delay_steps': delay, 'model_type': MODEL_TYPE, 'held_out_files': names, 'validation': summaries, 'equilibrium': equilibrium_diagnostic(a), 'hardware_validated': False}
    (args.output/'metrics.json').write_text(json.dumps(metrics, indent=2))
    pd.concat(frames, ignore_index=True).to_csv(args.output/'validation_windows.csv', index=False)
    print(json.dumps(metrics['equilibrium']))


if __name__ == '__main__':
    main()
