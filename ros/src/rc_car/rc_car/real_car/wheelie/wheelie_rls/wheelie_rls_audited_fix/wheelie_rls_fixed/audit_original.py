#!/usr/bin/env python3
"""Reproduce original learning on the same fit-file split, then freeze it.

This reconstructs a model from zero. The actual deployed .npz was not
uploaded, and its initialization/online weight history cannot be recovered.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from rls import rls_features_numpy
from train_rls import load_experiments


def original_derivative(state, u, a):
    fv, fw = rls_features_numpy(state, u)
    dv, dw = fv @ a[:7], fw @ a[7:]
    dw = np.where((state[..., 2] >= 0) & (dw > 0), 0.0, dw)
    return np.stack((state[..., 1], dv, state[..., 3], dw), axis=-1)


def original_step(state, u, dt, a):
    k1 = original_derivative(state, u, a)
    k2 = original_derivative(state+0.5*dt*k1, u, a)
    k3 = original_derivative(state+0.5*dt*k2, u, a)
    k4 = original_derivative(state+dt*k3, u, a)
    result = state+dt*(k1+2*k2+2*k3+k4)/6
    result[..., 1] = result[..., 1].clip(-15, 15)
    result[..., 2] = result[..., 2].clip(-2*np.pi, 2*np.pi)
    result[..., 3] = result[..., 3].clip(-50, 50)
    return np.nan_to_num(result, nan=0, posinf=10, neginf=-10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    output = root/'audit'
    experiments = load_experiments(args.dataset)
    spec = importlib.util.spec_from_file_location('original_rls', root/'audit'/'originals'/'rls.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    results = {}
    for name_set in ('original_script_001_to_145', 'same_training_split'):
        a, prior, posterior, targets = np.zeros(18), [], [], []
        selected = [n for n in experiments if int(Path(n).stem.split('_')[-1]) <= 145] if name_set.startswith('original_script') else [n for n in experiments if int(Path(n).stem.split('_')[-1]) % 5 != 0]
        history = []
        for name in selected:
            d = experiments[name]
            b, c = butter(2, 0.4)
            y = filtfilt(b, c, np.gradient(d['state'][:, [1, 3]], d['t'], axis=0), axis=0)
            future_u = pd.Series(d['u']).shift(-2).fillna(0.0).to_numpy()
            P = np.eye(18)*1000
            for state, u, target in zip(d['state'], future_u, y):
                a, P, info = old.rls_update(state, u, target[0], target[1], a, P, forgetting_factor=0.95, sigma_v_dot=2.0, sigma_omega_dot=5.0)
                prior.append(target-np.array((info['v_dot_error'], info['omega_dot_error'])))
                posterior.append((info['v_dot_hat'], info['omega_dot_hat']))
                targets.append(target)
            history.append(dict(file=name, **{f'a{i}': float(x) for i, x in enumerate(a)}))
        pd.DataFrame(history).to_csv(output/f'{name_set}_weights.csv', index=False)
        np.savez(output/f'{name_set}.npz', a=a, P=P)
        target = np.array(targets)
        metrics = {'fit_files': len(selected), 'prior_training_rmse_v_w': np.sqrt(np.mean((target-prior)**2, axis=0)).tolist(), 'posterior_training_rmse_v_w': np.sqrt(np.mean((target-posterior)**2, axis=0)).tolist(), 'reference_wdot_u0_v1': float(original_derivative(np.array((0.0, 1.0, np.deg2rad(-55), 0.0)), np.asarray(0.0), a)[3]), 'reference_input_gain': float(a[7]+a[13]*np.cos(np.deg2rad(-55)))}
        if name_set == 'same_training_split':
            summaries = []
            for steps in (1, 5, 10, 20):
                data = np.load(output/'validation'/f'traces_{steps}_screened.npz')
                files, starts, truth, dt = data['files'], data['starts'], data['truth'], data['dt']
                state = np.array([experiments[str(n)]['state'][k] for n, k in zip(files, starts)])
                controls = np.array([experiments[str(n)]['u'][k:k+steps] for n, k in zip(files, starts)])
                prediction = []
                for j in range(steps):
                    state = original_step(state, controls[:, j], dt[:, j, None], a)
                    prediction.append(state.copy())
                prediction = np.stack(prediction, axis=1)
                error = prediction-truth
                rmse = np.rad2deg(np.sqrt(np.mean(error[:, :, 2]**2, axis=1)))
                rows = pd.DataFrame(dict(file=files, start=starts, pitch_rmse_deg=rmse))
                rows.to_csv(output/f'original_same_split_windows_{steps}.csv', index=False)
                np.savez_compressed(output/f'original_traces_{steps}.npz', files=files, starts=starts, prediction=prediction, truth=truth, dt=dt)
                summaries.append(dict(steps=steps, windows=len(files), files=len(set(files)), pitch_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(error[:, :, 2]**2)))), median_window_pitch_rmse_deg=float(np.median(rmse)), p95_window_pitch_rmse_deg=float(np.percentile(rmse, 95))))
            metrics['screened_test'] = summaries
        results[name_set] = metrics
    (output/'original_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
