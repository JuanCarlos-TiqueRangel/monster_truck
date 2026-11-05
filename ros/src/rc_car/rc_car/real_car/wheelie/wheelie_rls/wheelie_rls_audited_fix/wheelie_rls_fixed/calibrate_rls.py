#!/usr/bin/env python3
"""Explore causal lag and ridge on calibration experiments, never test IDs.

The winner must have a modeled trim at -55 deg inside the command bounds
and no predicted domain violations on screened calibration windows. These
conditions do NOT establish real-car stability or sufficient model accuracy.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from train_rls import load_experiments, fit_rls
from validate_rls import evaluate, equilibrium_diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'audit'/'calibration_candidates.csv')
    args = parser.parse_args()
    data = load_experiments(args.dataset)
    train = [n for n in data if int(Path(n).stem.split('_')[-1]) % 5 not in (0, 1)]
    calibration = [n for n in data if int(Path(n).stem.split('_')[-1]) % 5 == 1]
    rows = []
    for delay in range(5):
        for ridge in (1.0, 10.0, 100.0):
            a, _, _, _, _ = fit_rls(data, train, delay, ridge)
            m10, _, _ = evaluate(data, calibration, a, delay, 10)
            m20, _, _ = evaluate(data, calibration, a, delay, 20)
            trim = equilibrium_diagnostic(a)
            if not m10['windows'] or not m20['windows']:
                raise ValueError('Not enough calibration windows for this search.')
            eligible = trim['model_has_joint_trim'] and trim['angular_input_gain_at_reference'] < 0.0 and m10['domain_violation_windows'] == 0 and m20['domain_violation_windows'] == 0
            score = 0.5*(m10['file_balanced_pitch_rmse_deg']+m20['file_balanced_pitch_rmse_deg']) if eligible else float('inf')
            rows.append(dict(delay_steps=delay, ridge=ridge, eligible=eligible, selection_score=score, pitch_rmse_10_deg=m10['pitch_rmse_deg_finite_windows'], pitch_rmse_20_deg=m20['pitch_rmse_deg_finite_windows'], windows_10=m10['windows'], windows_20=m20['windows'], files_10=m10['files'], files_20=m20['files'], joint_trim_u=trim['joint_trim_u']))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    accepted = frame[frame.eligible].sort_values('selection_score')
    print('No eligible model.' if accepted.empty else json.dumps(accepted.iloc[0].to_dict(), indent=2))


if __name__ == '__main__':
    main()
