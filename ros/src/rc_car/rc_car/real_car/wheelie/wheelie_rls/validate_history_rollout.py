#!/usr/bin/env python3
"""Compare frozen acceleration fits and recursively integrated pitch on a CSV.

Example:
python validate_history_rollout.py rls_history_model.npz held_out.csv --start 4 --steps 20

--active-only filters sequence_index==-1 BEFORE constructing history. Use it
only when the actual training input files were filtered the same way.
No controller, ROS connection, output commands, online RLS or model refitting.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rls_history_mppi import RLSHistoryDynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--controller-projection", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    if args.active_only:
        df = df.loc[df["sequence_index"].ne(-1)].reset_index(drop=True)
    columns = ["elapsed_s", "u", "x", "v", "theta_rad", "omega", "v_dot", "omega_dot"]
    values = df[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("CSV values must be finite and timestamps strictly increasing.")
    if not 0 <= args.start < len(df) - 1 or args.steps < 1:
        raise ValueError("Require a start row with at least one future sample and positive --steps.")

    model = RLSHistoryDynamics(args.model, device="cpu", dtype=torch.float64)
    t = df.elapsed_s.to_numpy(dtype=float)
    z = df[["u", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    measured = df[["x", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
    target_acc = df[["v_dot", "omega_dot"]].to_numpy(dtype=float)
    idx = np.maximum(np.arange(len(df))[:, None] - np.arange(5), 0)
    histories = torch.as_tensor(z[idx], dtype=model.dtype)
    ages = torch.as_tensor(t[:, None] - t[idx[:, 1:]], dtype=model.dtype)
    fitted_acc = model.predict(histories, ages).numpy()
    print("Measured-history acceleration RMSE [m/s^2, rad/s^2]:", np.sqrt(np.mean((fitted_acc - target_acc)**2, axis=0)))

    first, last = args.start, min(args.start + args.steps, len(df) - 1)
    past_idx = np.maximum(first - np.arange(1, 5), 0)
    state = torch.as_tensor(measured[first:first + 1].copy(), dtype=model.dtype)
    past = torch.as_tensor(z[past_idx][None].copy(), dtype=model.dtype)
    ages = torch.as_tensor((t[first] - t[past_idx])[None], dtype=model.dtype)
    predicted = [state[0].numpy().copy()]
    for k in range(first, last):
        dt = float(t[k + 1] - t[k])
        if not 0.5 * model.nominal_dt <= dt <= 1.5 * model.nominal_dt:
            raise ValueError("Selected rollout crosses a gap incompatible with the learned history cadence.")
        u = torch.tensor([z[k, 0]], dtype=model.dtype)
        state, past, ages = model.step(state, u, past, ages, dt=dt)
        if args.controller_projection:
            landing = (state[:, 2] >= 0.0) & (state[:, 3] > 0.0)
            state = torch.stack((state[:, 0], state[:, 1].clamp(-10.0, 10.0), state[:, 2].clamp(max=0.0),
                                 torch.where(landing, torch.zeros_like(state[:, 3]), state[:, 3].clamp(-30.0, 30.0))), dim=-1)
        predicted.append(state[0].numpy().copy())
    predicted = np.asarray(predicted)
    truth = measured[first:last + 1]
    pitch_error_deg = np.degrees(predicted[:, 2] - truth[:, 2])
    print(f"Free-running window: rows {first}..{last}, {last - first} steps, {t[last] - t[first]:.3f} s")
    print(f"Pitch RMSE={np.sqrt(np.mean(pitch_error_deg**2)):.3f} deg | max_abs={np.max(np.abs(pitch_error_deg)):.3f} deg")
    print(f"Final pitch: measured={np.degrees(truth[-1, 2]):+.3f} deg, predicted={np.degrees(predicted[-1, 2]):+.3f} deg")

    # Independent consistency check using recorded acceleration, not the learned model.
    duration = np.diff(t[first:last + 1])
    acc = target_acc[first:last + 1, 1]
    integrated_w = truth[0, 3] + np.r_[0.0, np.cumsum(0.5 * (acc[:-1] + acc[1:]) * duration)]
    integrated_theta = truth[0, 2] + np.r_[0.0, np.cumsum(0.5 * (integrated_w[:-1] + integrated_w[1:]) * duration)]
    theta_from_omega = truth[0, 2] + np.r_[0.0, np.cumsum(0.5 * (truth[:-1, 3] + truth[1:, 3]) * duration)]
    error_from_omega = np.degrees(theta_from_omega - truth[:, 2])
    print(f"Recorded-omega integration: pitch_RMSE={np.sqrt(np.mean(error_from_omega**2)):.3f} deg | final={np.degrees(theta_from_omega[-1]):+.3f} deg")
    print(f"Recorded-acceleration trapezoid check: omega_end={integrated_w[-1]:+.3f} versus measured={truth[-1, 3]:+.3f} rad/s")
    print(f"Recorded-acceleration trapezoid check: pitch_end={np.degrees(integrated_theta[-1]):+.3f} versus measured={np.degrees(truth[-1, 2]):+.3f} deg")

    if not args.no_plot:
        time_window = t[first:last + 1] - t[first]
        plt.figure(figsize=(10, 5))
        plt.plot(time_window, np.degrees(truth[:, 2]), label="Measured pitch")
        plt.plot(time_window, np.degrees(predicted[:, 2]), "--", label="Free-running model pitch")
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
