#!/usr/bin/env python3
"""Fit a frozen, causal whole-dynamics RLS model; never overwrite CSVs.

The command convention is: u[k] is issued at row k, so u[k-delay_steps]
explains the transition k -> k+1. The physical logger convention still needs
hardware verification. The chosen 3-step lag is an empirical pipeline lag,
NOT a measured ESC latency.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from rls import MODEL_TYPE, V_FEATURES, W_FEATURES, rls_features_numpy, rls_update

EXCLUDED_FILES = ('rls_id_003.csv', 'rls_id_011.csv', 'rls_id_079.csv')


def load_experiments(folder):
    paths = sorted(Path(folder).glob('rls_id_*.csv'))  # Do not read backup_dataset a second time.
    if not paths:
        raise FileNotFoundError(f'No rls_id_*.csv files directly inside {folder}')
    experiments = {}
    for path in paths:
        df = pd.read_csv(path)
        required = ('elapsed_s', 'u', 'x', 'v', 'theta_rad', 'omega')
        if not set(required).issubset(df.columns) or len(df) < 2:
            raise ValueError(f'{path.name}: missing required columns or fewer than two rows.')
        t = df['elapsed_s'].to_numpy(float)
        state = df[['x', 'v', 'theta_rad', 'omega']].to_numpy(float)
        u = df['u'].to_numpy(float)
        dt = np.diff(t)
        with np.errstate(divide='ignore', invalid='ignore'):
            target = np.diff(state[:, [1, 3]], axis=0) / dt[:, None]
        defect = np.abs(np.diff(state[:, 2]) - 0.5*dt*(state[:-1, 3]+state[1:, 3]))
        valid = np.isfinite(state[:-1]).all(1) & np.isfinite(state[1:]).all(1) & np.isfinite(target).all(1)
        valid &= (dt > 0.02) & (dt < 0.06) & np.isfinite(u[:-1])
        valid &= (np.abs(state[:-1, 1]) <= 5.0) & (np.abs(state[1:, 1]) <= 5.0)
        valid &= (np.abs(state[:-1, 3]) <= 12.0) & (np.abs(state[1:, 3]) <= 12.0)
        valid &= (defect < np.deg2rad(8.0)) & (np.abs(target[:, 0]) < 40.0) & (np.abs(target[:, 1]) < 200.0)
        if path.name in EXCLUDED_FILES:
            valid[:] = False
        experiments[path.name] = dict(state=state, u=u, t=t, dt=dt, target=target, valid=valid, path=path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return experiments


def fit_rls(experiments, names, delay_steps=3, ridge=10.0):
    if delay_steps < 0 or ridge <= 0.0:
        raise ValueError('delay_steps must be nonnegative and ridge must be positive.')
    blocks, targets, used_names = [], [], []
    for name in names:
        d = experiments[name]
        k = np.flatnonzero(d['valid'] & (np.arange(len(d['valid'])) >= delay_steps))
        k = k[np.isfinite(d['u'][k-delay_steps])]
        if not len(k):
            continue
        blocks.append(rls_features_numpy(d['state'][k], d['u'][k-delay_steps]))
        targets.append(d['target'][k])
        used_names.append(name)
    if not blocks:
        raise ValueError('No usable training transitions after screening.')
    fv = np.concatenate([b[0] for b in blocks])
    fw = np.concatenate([b[1] for b in blocks])
    y = np.concatenate(targets)
    scale = np.maximum(np.sqrt(np.mean(np.concatenate((fv, fw), axis=1)**2, axis=0)), 1e-8)
    a, P = np.zeros(18), np.eye(18) / ridge
    for i in range(len(y)):
        a, P, _ = rls_update(np.zeros(4), 0.0, y[i, 0], y[i, 1], a, P, forgetting_factor=1.0, features=(fv[i]/scale[:7], fw[i]/scale[7:]))
    # Export physical-unit coefficients; rollout evaluation needs no hidden scaler.
    a /= scale
    P /= np.outer(scale, scale)
    return a, P, scale, len(y), used_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path(__file__).parent/'wheelie_nn'/'dataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'wheelie_nn'/'rls_learned_weights_v2.npz')
    parser.add_argument('--delay-steps', type=int, default=3)
    parser.add_argument('--ridge', type=float, default=10.0)
    args = parser.parse_args()
    experiments = load_experiments(args.dataset)
    holdout = [n for n in experiments if int(Path(n).stem.split('_')[-1]) % 5 == 0]
    training = [n for n in experiments if n not in holdout]
    a, P, scale, count, used = fit_rls(experiments, training, args.delay_steps, args.ridge)
    if not holdout:
        raise ValueError('No held-out experiments (IDs divisible by 5).')
    rows = []
    for name, d in experiments.items():
        rows.append(dict(file=name, split='test' if name in holdout else 'train', rows=len(d['state']), accepted_transitions=int(d['valid'].sum()), total_transitions=len(d['valid']), excluded_file=name in EXCLUDED_FILES, sha256=d['sha256']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trained_u = np.concatenate([experiments[n]['u'] for n in used])
    np.savez_compressed(args.output, model_type=MODEL_TYPE, format_version=2, a=a, P=P, feature_scale=scale, v_features=np.asarray(V_FEATURES), w_features=np.asarray(W_FEATURES), target='forward_velocity_difference_per_second', integrator='forward_trapezoidal_kinematics', input_delay_steps=args.delay_steps, command_convention='u[k] issued at row k; transition uses u[k-delay]', nominal_sample_period=0.04, ridge=args.ridge, forgetting_factor=1.0, training_files=np.asarray(used), holdout_files=np.asarray(holdout), training_sha256=np.asarray([experiments[n]['sha256'] for n in used]), holdout_sha256=np.asarray([experiments[n]['sha256'] for n in holdout]), excluded_files=np.asarray(EXCLUDED_FILES), observed_u_min=float(trained_u.min()), observed_u_max=float(trained_u.max()), training_transitions=count, hardware_validated=False)
    pd.DataFrame(rows).to_csv(args.output.with_suffix('.screening.csv'), index=False)
    print(f'Trained on {count} transitions from {len(used)} experiments; held out {len(holdout)} complete experiments.')
    print(f'Saved {args.output}; delay={args.delay_steps} samples, ridge={args.ridge:g}.')
    print('Run validate_rls.py before enabling drive. Offline fit is not a hardware validation.')


if __name__ == '__main__':
    main()
