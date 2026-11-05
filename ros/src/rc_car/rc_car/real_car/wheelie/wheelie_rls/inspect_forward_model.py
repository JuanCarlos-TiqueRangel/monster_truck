#!/usr/bin/env python3
"""Read-only, offline checks of the history model and one recorded experiment.

Uses CPU/CUDA float64. No ROS, motor commands, training, or input-file changes.
The projection below exactly matches the supplied mppi_history.py _step.
--dataset searches the saved training file names for the most negative-velocity
kernel landmark's five state/input rows. A match locates data; it does not prove
that the record is erroneous. It never excludes or deletes records.
"""

import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def projection(state):
    landing = (state[:, 2] >= 0.0) & (state[:, 3] > 0.0)
    return torch.stack((state[:, 0], state[:, 1].clamp(-10.0, 10.0), state[:, 2].clamp(max=0.0),
                        torch.where(landing, torch.zeros_like(state[:, 3]), state[:, 3].clamp(-30.0, 30.0))), dim=-1)


@torch.no_grad()
def rollout(model, measurements, controls, times, start, steps, measured_velocity=False):
    inds = start - np.arange(1, 5)
    state = torch.as_tensor(measurements[start:start + 1].copy(), dtype=model.dtype, device=model.device)
    past = torch.as_tensor(controls[inds][None].copy(), dtype=model.dtype, device=model.device)
    ages = torch.as_tensor((times[start] - times[inds])[None], dtype=model.dtype, device=model.device)
    predicted = [state[0].cpu().numpy().copy()]
    for k in range(start, start + steps):
        u = torch.tensor([controls[k, 0]], dtype=model.dtype, device=model.device)
        state, past, ages = model.step(state, u, past, ages, dt=float(times[k + 1] - times[k]))
        state = projection(state)
        if measured_velocity:
            # Offline diagnostic ONLY: future measured v is unavailable to a controller.
            state[:, 1] = float(measurements[k + 1, 1])
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError(f"Nonfinite rollout starting at row {start}, at predicted row {k + 1}.")
        predicted.append(state[0].cpu().numpy().copy())
    return np.asarray(predicted)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--dataset", type=Path, help="Folder with the actual training CSVs; optional exact landmark search.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.model.is_file() or not args.csv.is_file():
        parser.error("Model and CSV must be existing files.")
    if args.steps < 1 or (args.dataset is not None and not args.dataset.is_dir()):
        parser.error("Require positive --steps and an existing --dataset directory when specified.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")
    sys.path.insert(0, str(args.code_dir.resolve()))
    from rls_history_mppi import RLSHistoryDynamics
    torch.set_num_threads(1)
    model = RLSHistoryDynamics(args.model, device=args.device, dtype=torch.float64)
    df = pd.read_csv(args.csv)
    columns = ["elapsed_s", "u", "x", "v", "theta_rad", "omega"]
    missing = set(columns).difference(df.columns)
    if missing:
        parser.error(f"Missing CSV fields: {sorted(missing)}")
    if not np.isfinite(df[columns].to_numpy(dtype=float)).all():
        parser.error("CSV measurements must be finite.")
    t = df.elapsed_s.to_numpy(dtype=float)
    if np.any(np.diff(t) <= 0.0):
        parser.error("CSV timestamps must be strictly increasing.")
    s = df[["x", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    z = df[["u", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    print("OFFLINE ONLY: no ROS, published commands, retraining, or changes to model/CSV files.")
    print(f"Model SHA256: {hashlib.sha256(args.model.read_bytes()).hexdigest()}")
    print(f"Device={args.device}; dtype=float64; controller projection enabled; horizon={args.steps} steps")
    print(f"CSV rows={len(df)}; measured pitch range=[{np.degrees(s[:, 2]).min():.3f}, {np.degrees(s[:, 2]).max():.3f}] deg")
    print("\nAll admissible full-horizon windows (zero-based CSV data-row indices):")
    print(" start end   pitch_RMSE   max_error  measured_end predicted_end   v_RMSE omega_RMSE")
    records = []
    for start in range(4, len(df) - args.steps):
        intervals = np.diff(t[start - 4:start + args.steps + 1])
        if np.any((intervals < 0.5 * model.nominal_dt) | (intervals > 1.5 * model.nominal_dt)):
            continue
        pred = rollout(model, s, z, t, start, args.steps)
        actual = s[start:start + args.steps + 1]
        e = pred - actual
        pitch_error = np.degrees(e[:, 2])
        pitch_rmse = float(np.sqrt(np.mean(pitch_error**2)))
        values = (start, start + args.steps, pitch_rmse, float(np.max(np.abs(pitch_error))),
                  float(np.degrees(actual[-1, 2])), float(np.degrees(pred[-1, 2])),
                  float(np.sqrt(np.mean(e[:, 1]**2))), float(np.sqrt(np.mean(e[:, 3]**2))))
        records.append(values)
        print("%6d %3d %12.3f %11.3f %13.3f %13.3f %8.3f %10.3f" % values)
    if not records:
        parser.error("No admissible full-horizon windows in this file.")
    print(f"Evaluated {len(records)} overlapping windows, not {len(records)} independent experiments.")

    # Re-run a preselected active-maneuver window with future measured velocity.
    start = 20
    if any(row[0] == start for row in records):
        normal = rollout(model, s, z, t, start, args.steps)
        oracle = rollout(model, s, z, t, start, args.steps, measured_velocity=True)
        tr = s[start:start + args.steps + 1]
        en = np.degrees(normal[:, 2] - tr[:, 2])
        eo = np.degrees(oracle[:, 2] - tr[:, 2])
        print(f"\nRows {start}..{start + args.steps}: pitch RMSE normal={np.sqrt(np.mean(en**2)):.3f} deg, "
              f"with measured future v={np.sqrt(np.mean(eo**2)):.3f} deg")
        print("Measured-future-v variant is an isolation test, NOT deployable dynamics.")

    print("\nZero-input, flat/stationary rollout, with the SAME controller projection:")
    state = torch.zeros(1, 4, dtype=model.dtype, device=model.device)
    past = torch.zeros(1, 4, 4, dtype=model.dtype, device=model.device)
    ages = torch.arange(1, 5, dtype=model.dtype, device=model.device)[None] * model.nominal_dt
    u = torch.zeros(1, dtype=model.dtype, device=model.device)
    print(" time_s    velocity_mps  pitch_deg    v_dot_mps2")
    for step in range(args.steps + 1):
        current = torch.cat((u[:, None], state[:, 1:]), dim=-1)
        h = torch.cat((current[:, None, :], past), dim=1)
        accel = model.predict(h, ages)
        print(f"{step * model.nominal_dt:7.3f} {state[0, 1].item():15.6f} "
              f"{torch.rad2deg(state[0, 2]).item():10.3f} {accel[0, 0].item():13.6f}")
        if step < args.steps:
            state, past, ages = model.step(state, u, past, ages, dt=model.nominal_dt)
            state = projection(state)

    with np.load(args.model, allow_pickle=False) as saved:
        raw = saved["kernel_centers"] * saved["v_scale"] + saved["v_mean"]
        j = int(np.argmin(raw[:, 1]))
        target_history = raw[j, :20].reshape(5, 4)
        training_names = [Path(str(p)).name for p in saved["training_files"]]
    print(f"\nMost negative current-velocity saved landmark: index {j} (zero-based).")
    print("These are recovered TRAINING INPUTS, not simulated states; oldest row first:")
    print(" lag    u       v_mps      theta_deg    omega_radps")
    for lag in range(4, -1, -1):
        row = target_history[lag]
        print(f"{-lag:4d} {row[0]:7.4f} {row[1]:11.6f} {np.degrees(row[2]):13.6f} {row[3]:14.6f}")
    if args.dataset is None:
        print("Add --dataset to locate this exact five-row history in the saved training files.")
        return

    matches, checked, unavailable = 0, 0, []
    for name in dict.fromkeys(training_names):
        path = args.dataset / name
        if not path.is_file():
            unavailable.append(name)
            continue
        block = pd.read_csv(path)
        needed = ["elapsed_s", "u", "v", "theta_rad", "omega"]
        if set(needed).difference(block.columns) or len(block) < 5:
            unavailable.append(name)
            continue
        source_z = block[["u", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
        indices = np.arange(4, len(block))[:, None] - np.arange(5)
        # Exact state/input-history match after roundoff from unstandardization.
        equal = np.isclose(source_z[indices], target_history[None], rtol=1e-7, atol=1e-8).all(axis=(1, 2))
        checked += 1
        for k in np.flatnonzero(equal) + 4:
            matches += 1
            print(f"\nMATCH: {path.resolve()} | zero-based data row {k} | CSV text line {k + 2}")
            view = block.iloc[max(0, k - 4):min(len(block), k + 2)].copy()
            view.insert(0, "data_row", view.index)
            view["pitch_deg"] = np.degrees(view["theta_rad"])
            names = ["data_row", "elapsed_s", "u"] + (["x"] if "x" in view.columns else []) + ["v", "pitch_deg", "omega"]
            print(view[names].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
            if "x" in view.columns and len(view) > 1:
                dt = np.diff(view["elapsed_s"].to_numpy(dtype=float))
                if np.all(dt > 0):
                    position_rate = np.diff(view["x"].to_numpy(dtype=float)) / dt
                    print("Forward dx/dt from recorded x [m/s]:", np.array2string(position_rate, precision=5))
                    print("Compare with v and the recording; filtering may cause differences, so this is not an automatic rejection test.")
    print(f"\nExact matching histories: {matches}; searched training files: {checked}; unavailable/unusable files: {len(unavailable)}")
    if matches == 0:
        print("No match found. Check that --dataset is the unchanged data collection used for this saved model.")
    print("No rows were removed. Review data validity and whether any external motion/forces occurred before excluding a trial.")


if __name__ == "__main__":
    main()
