#!/usr/bin/env python3
"""Read-only screening of x/v consistency in the files named by a saved model.

Assumption being checked: recorded x and v share a coordinate/frame and units,
and x_dot = v is the intended state definition. No fitting, exclusion, file
changes, ROS connection, or commands. A flag is a review item, not a verdict.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def check_file(path: Path, tolerance: float, dt_min: float, dt_max: float) -> dict:
    df = pd.read_csv(path)
    required = ["elapsed_s", "x", "v"]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    values = df[required].to_numpy(dtype=np.float64)
    if len(values) < 2:
        raise ValueError("need at least two data rows")
    t, x, v = values.T
    dt = np.diff(t)
    finite = np.isfinite(values).all(axis=1)
    valid = finite[:-1] & finite[1:] & np.isfinite(dt) & (dt >= dt_min) & (dt <= dt_max)
    bad_intervals = int((~valid).sum())
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise ValueError(f"no eligible intervals; {bad_intervals} nonfinite/out-of-cadence intervals")
    with np.errstate(over="ignore", invalid="ignore"):
        position_speed = (x[ids + 1] - x[ids]) / dt[ids]
        mean_velocity = 0.5 * v[ids] + 0.5 * v[ids + 1]
        errors = np.abs(position_speed - mean_velocity)
        acceleration = (v[ids + 1] - v[ids]) / dt[ids]
    if not np.isfinite(np.column_stack((position_speed, mean_velocity, errors, acceleration))).all():
        raise ValueError("arithmetic overflow in displacement/velocity check")
    worst = int(np.argmax(errors))
    jump = int(np.argmax(np.abs(acceleration)))
    row = int(ids[worst])
    return {
        "file": path.name, "valid": len(ids), "bad_dt": bad_intervals,
        "flagged": int(np.count_nonzero(errors > tolerance)), "worst_row": row,
        "max_error": float(errors[worst]), "median_error": float(np.median(errors)),
        "position_speed": float(position_speed[worst]), "mean_velocity": float(mean_velocity[worst]),
        "v0": float(v[row]), "v1": float(v[row + 1]), "time0": float(t[row]), "time1": float(t[row + 1]),
        "max_abs_forward_acc": float(abs(acceleration[jump])), "jump_row": int(ids[jump]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help=".npz with training_files metadata")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0, help="review threshold on speed disagreement, m/s; not a safety or removal limit")
    parser.add_argument("--min-dt", type=float, default=0.02)
    parser.add_argument("--max-dt", type=float, default=0.06)
    args = parser.parse_args()
    if not args.model.is_file() or not args.dataset.is_dir():
        parser.error("model must be an existing .npz and --dataset an existing directory")
    if not np.isfinite([args.tolerance, args.min_dt, args.max_dt]).all() or args.tolerance <= 0 or not 0 < args.min_dt <= args.max_dt:
        parser.error("require finite positive tolerance and 0 < min-dt <= max-dt")
    with np.load(args.model, allow_pickle=False) as saved:
        if "training_files" not in saved.files:
            parser.error("model has no training_files metadata")
        paths = [str(p) for p in saved["training_files"].reshape(-1)]
    names = [Path(p).name for p in paths]
    if len(set(names)) != len(names):
        parser.error("duplicate training basenames: cannot unambiguously map them into --dataset")
    if not names:
        parser.error("training_files metadata is empty")

    print("READ-ONLY SCREEN: no ROS, fitting, exclusions, CSV edits, or model changes.")
    print("Requires x_dot=v with consistent frame, units, and timing. Flags need review.")
    print(f"Metric: abs((x[k+1]-x[k])/dt - (v[k]+v[k+1])/2). Review threshold={args.tolerance:g} m/s.")
    print("The threshold is diagnostic, not an automatic rejection or physical-safety bound.")
    reports, failures = [], []
    for name in names:
        try:
            reports.append(check_file(args.dataset / name, args.tolerance, args.min_dt, args.max_dt))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            failures.append((name, str(exc)))
    reports.sort(key=lambda r: r["max_error"], reverse=True)
    flagged = [r for r in reports if r["flagged"]]
    print("\nFiles with flagged intervals, largest discrepancy first (zero-based data rows):")
    print("file                         flagged/valid  max_err_mps  median_mps  worst_pair    dx_dt    mean_v  max_abs_dv_dt")
    for r in flagged:
        pair = f"{r['worst_row']}->{r['worst_row'] + 1}"
        counts = f"{r['flagged']}/{r['valid']}"
        print(f"{r['file']:28s} {counts:>12s} {r['max_error']:12.3f} {r['median_error']:11.3f} {pair:>11s} {r['position_speed']:8.3f} {r['mean_velocity']:9.3f} {r['max_abs_forward_acc']:14.3f}")
        print(f"  worst pair: t={r['time0']:.6f}->{r['time1']:.6f} s; v={r['v0']:+.6f}->{r['v1']:+.6f} m/s; largest |dv/dt| at row {r['jump_row']}->{r['jump_row'] + 1}.")
    if not flagged:
        print("No eligible intervals exceeded the selected threshold. This is not a complete data-quality pass.")
    for r in reports:
        if r["bad_dt"]:
            print(f"INTERVAL CHECK: {r['file']}: {r['bad_dt']} nonfinite/out-of-cadence intervals not scored.")
    for name, error in failures:
        print(f"UNREADABLE: {name}: {error}")
    print(f"\nTraining files listed={len(names)}, read={len(reports)}, flagged_files={len(flagged)}, unreadable={len(failures)}.")
    print("A large dv/dt alone is not the flag criterion. No negative velocities or zero-input rows were removed.")
    print("Small discrepancies can reflect sampling/filtering; unflagged data are not certified, and angular signals are not checked here.")


if __name__ == "__main__":
    main()
