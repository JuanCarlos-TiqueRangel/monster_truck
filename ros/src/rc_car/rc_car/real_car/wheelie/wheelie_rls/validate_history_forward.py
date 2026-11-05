#!/usr/bin/env python3
"""Score forward velocity targets and free-running pitch, using a frozen .npz.

No ROS, motor commands, weight updates, or CSV changes.
Works with the old and the forward-trained history models; always scores the
same forward targets, so comparisons use the same definitions and window.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rls_history_mppi import RLSHistoryDynamics


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--controller-projection", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file() or not args.csv.is_file():
        parser.error("Model and CSV arguments must be existing files, not placeholder filenames.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this environment.")
    torch.set_num_threads(1)
    df = pd.read_csv(args.csv)
    columns = ["elapsed_s", "u", "x", "v", "theta_rad", "omega"]
    missing = [name for name in columns if name not in df.columns]
    if missing:
        parser.error(f"Missing columns: {missing}")
    values = df[columns].to_numpy(dtype=float)
    if len(df) < 6 or not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        parser.error("Need at least six finite rows with strictly increasing timestamps.")
    first, last = args.start, args.start + args.steps
    if first < 4 or args.steps < 1 or last >= len(df):
        parser.error(f"Require start >= 4 and start + steps < {len(df)}. This script does not silently shorten the horizon.")

    model = RLSHistoryDynamics(args.model, device=args.device, dtype=torch.float64)
    with np.load(args.model, allow_pickle=False) as saved:
        target_type = str(saved["target_type"].item()) if "target_type" in saved.files else "recorded_acceleration_columns (legacy)"
    print(f"Model target metadata: {target_type}; device={args.device}, dtype=float64")
    t = df.elapsed_s.to_numpy(dtype=float)
    z = df[["u", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    measured = df[["x", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    dt_all = np.diff(t)
    targets = np.full((len(df), 2), np.nan)
    targets[:-1] = np.diff(measured[:, [1, 3]], axis=0) / dt_all[:, None]
    good_dt = np.r_[(dt_all >= 0.5 * model.nominal_dt) & (dt_all <= 1.5 * model.nominal_dt), False]
    raw_idx = np.arange(len(df))[:, None] - np.arange(5)
    idx = np.maximum(raw_idx, 0)
    valid = (raw_idx >= 0).all(axis=1) & good_dt[idx].all(axis=1) & np.isfinite(targets).all(axis=1)
    if not valid[first:last].all():
        parser.error("Selected history/rollout crosses an inadmissible timestamp interval; choose a contiguous 40-ms segment.")

    histories = torch.as_tensor(z[idx[valid]], dtype=model.dtype, device=model.device)
    ages = torch.as_tensor(t[valid, None] - t[idx[valid, 1:]], dtype=model.dtype, device=model.device)
    fitted = np.full((len(df), 2), np.nan)
    fitted[valid] = model.predict(histories, ages).cpu().numpy()
    rmse = np.sqrt(np.mean((fitted[valid] - targets[valid])**2, axis=0))
    window_rmse = np.sqrt(np.mean((fitted[first:last] - targets[first:last])**2, axis=0))
    print(f"Measured-history forward-target RMSE over {valid.sum()} transitions [m/s^2, rad/s^2]: {rmse}")
    print(f"Same metric over the selected {args.steps}-step window: {window_rmse}")
    omega_one_step = measured[first:last, 3] + dt_all[first:last] * fitted[first:last, 1]
    print(f"Measured-history one-step omega RMSE: {np.sqrt(np.mean((omega_one_step - measured[first + 1:last + 1, 3])**2)):.6f} rad/s")

    past_idx = first - np.arange(1, 5)
    state = torch.as_tensor(measured[first:first + 1].copy(), dtype=model.dtype, device=model.device)
    past = torch.as_tensor(z[past_idx][None].copy(), dtype=model.dtype, device=model.device)
    ages = torch.as_tensor((t[first] - t[past_idx])[None], dtype=model.dtype, device=model.device)
    predicted = [state[0].cpu().numpy().copy()]
    for k in range(first, last):
        u = torch.tensor([z[k, 0]], dtype=model.dtype, device=model.device)
        state, past, ages = model.step(state, u, past, ages, dt=float(dt_all[k]))
        if args.controller_projection:
            landing = (state[:, 2] >= 0.0) & (state[:, 3] > 0.0)
            state = torch.stack((state[:, 0], state[:, 1].clamp(-10.0, 10.0), state[:, 2].clamp(max=0.0),
                                 torch.where(landing, torch.zeros_like(state[:, 3]), state[:, 3].clamp(-30.0, 30.0))), dim=-1)
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError(f"Nonfinite free-running prediction at CSV row {k + 1}; model is not ready for control.")
        predicted.append(state[0].cpu().numpy().copy())
    predicted = np.asarray(predicted)
    truth = measured[first:last + 1]

    rmse_v, rmse_omega = np.sqrt(np.mean((predicted[:, [1, 3]] - truth[:, [1, 3]])**2, axis=0))
    print(f"Free-running RMSE: v={rmse_v:.3f} m/s | omega={rmse_omega:.3f} rad/s")
    print(f"Final v: measured={truth[-1, 1]:+.3f}, predicted={predicted[-1, 1]:+.3f} m/s")

    pitch_error_deg = np.degrees(predicted[:, 2] - truth[:, 2])
    print(f"Free-running window: rows {first}..{last}, {args.steps} steps, {t[last] - t[first]:.3f} s")
    print(f"Pitch RMSE={np.sqrt(np.mean(pitch_error_deg**2)):.3f} deg | max_abs={np.max(np.abs(pitch_error_deg)):.3f} deg")
    print(f"Final pitch: measured={np.degrees(truth[-1, 2]):+.3f} deg, predicted={np.degrees(predicted[-1, 2]):+.3f} deg")

    # Data consistency checks, NOT model outputs and NOT training targets.
    duration = dt_all[first:last]
    theta_from_omega = truth[0, 2] + np.r_[0.0, np.cumsum(0.5 * (truth[:-1, 3] + truth[1:, 3]) * duration)]
    theta_euler = truth[0, 2] + np.r_[0.0, np.cumsum(truth[:-1, 3] * duration)]
    e_omega_deg = np.degrees(theta_from_omega - truth[:, 2])
    e_euler_deg = np.degrees(theta_euler - truth[:, 2])
    print(f"Recorded-omega trapezoid check: pitch_RMSE={np.sqrt(np.mean(e_omega_deg**2)):.3f} deg | final={np.degrees(theta_from_omega[-1]):+.3f} deg")
    print(f"Recorded-omega Euler check: pitch_RMSE={np.sqrt(np.mean(e_euler_deg**2)):.3f} deg | final={np.degrees(theta_euler[-1]):+.3f} deg")
    integrated_theta = None
    if "omega_dot" in df.columns:
        recorded_acc = df["omega_dot"].to_numpy(dtype=float)[first:last + 1]
        if np.isfinite(recorded_acc).all():
            integrated_w = truth[0, 3] + np.r_[0.0, np.cumsum(0.5 * (recorded_acc[:-1] + recorded_acc[1:]) * duration)]
            integrated_theta = truth[0, 2] + np.r_[0.0, np.cumsum(0.5 * (integrated_w[:-1] + integrated_w[1:]) * duration)]
            print(f"Recorded-acceleration trapezoid check (not forward targets): omega_end={integrated_w[-1]:+.3f} versus measured={truth[-1, 3]:+.3f} rad/s")
            print(f"Recorded-acceleration trapezoid check: pitch_end={np.degrees(integrated_theta[-1]):+.3f} versus measured={np.degrees(truth[-1, 2]):+.3f} deg")

    if not args.no_plot:
        import matplotlib.pyplot as plt
        time_window = t[first:last + 1] - t[first]
        plt.figure(figsize=(10, 5))
        plt.plot(time_window, np.degrees(truth[:, 2]), label="Measured pitch")
        plt.plot(time_window, np.degrees(predicted[:, 2]), "--", label="Free-running model pitch")
        if integrated_theta is not None:
            plt.plot(time_window, np.degrees(integrated_theta), ":", label="Integrated recorded angular acceleration")
        plt.plot(time_window, np.degrees(theta_from_omega), "-.", label="Integrated recorded angular velocity")
        plt.xlabel("Time from prediction start [s]")
        plt.ylabel("Pitch [deg]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
