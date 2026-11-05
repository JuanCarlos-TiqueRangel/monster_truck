# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics.pairwise import rbf_kernel
# from rls import history_features

# ROOT = Path(__file__).resolve().parent
# EXPERIMENT = "rls_id_014.csv"

# with np.load(ROOT / "rls_history_model.npz", allow_pickle=False) as archive:
#     model = {k: archive[k] for k in archive.files if k != "covariance"}


# df = pd.read_csv(ROOT / "wheelie_nn" / "dataset" / EXPERIMENT)
# training_names = {Path(s).name for s in model["training_files"].tolist()}


# # Build exactly the same causal features used during training.
# columns = model["feature_columns"].tolist()
# xv, xw = history_features(df[columns], df["elapsed_s"])
# zv = (xv - model["v_mean"]) / model["v_scale"]
# zw = (xw - model["w_mean"]) / model["w_scale"]

# kernels = rbf_kernel( zv, model["kernel_centers"], gamma=float(model["gamma"]) )
# FV = np.column_stack(( kernels @ model["kernel_normalization"].T, np.ones(len(df)) ))
# FW = np.column_stack((zw, np.ones(len(df))))

# # Frozen RLS prediction: no fitting and no coefficient updates.
# a = model["learned_parameters"]
# nv = int(model["n_v_features"])
# pred = np.column_stack((FV @ a[:nv], FW @ a[nv:]))

# actual = df[["v_dot", "omega_dot"]].to_numpy()
# active = df["sequence_index"].to_numpy() != -1
# rmse = np.sqrt(np.mean((pred[active] - actual[active]) ** 2, axis=0))

# print(f"Linear RMSE, active samples:  {rmse[0]:.4f} m/s^2")
# print(f"Angular RMSE, active samples: {rmse[1]:.4f} rad/s^2")

# fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
# labels = ["Linear acceleration (m/s^2)", "Angular acceleration (rad/s^2)"]

# for j, label in enumerate(labels):
#     shown = np.ones(len(df), dtype=bool) if j == 0 else active
#     axes[j].plot(df.elapsed_s, np.where(shown, actual[:, j], np.nan), color="black", label="Measured")
#     axes[j].plot(df.elapsed_s, np.where(shown, pred[:, j], np.nan), "--", color="cyan", label="Frozen RLS prediction")
#     axes[j].set_ylabel(label)
#     axes[j].legend()
#     axes[j].grid(alpha=0.3)
# axes[-1].set_xlabel("Elapsed time (s)")
# fig.suptitle(EXPERIMENT)
# fig.tight_layout()
# plt.show()














from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rls import rls_dynamics_torch


# ========================= CONFIGURATION =========================
CSV_PATH = Path('wheelie_nn/dataset/rls_id_102.csv')
WEIGHTS_PATH = Path('wheelie_nn/rls_learned_weights.npz')
DELAY_S = 0.0
OUTPUT_PATH = Path('rls_rollout_validation.png')
# ================================================================


def rk4_step(state, pwm, dt, weights):
    k1 = rls_dynamics_torch(state, pwm, weights)
    k2 = rls_dynamics_torch(state + 0.5 * dt * k1, pwm, weights)
    k3 = rls_dynamics_torch(state + 0.5 * dt * k2, pwm, weights)
    k4 = rls_dynamics_torch(state + dt * k3, pwm, weights)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def main():
    # 1. Load recorded data and frozen RLS weights
    df = pd.read_csv(CSV_PATH)
    t = df['elapsed_s'].to_numpy(dtype=float)
    u = df['u'].to_numpy(dtype=float)
    measured = df[['x', 'v', 'theta_rad', 'omega']].to_numpy(dtype=float)

    with np.load(WEIGHTS_PATH) as model:
        a = model['a'].copy()

    if len(a) != 18:
        raise ValueError('Expected 18 RLS parameters (7 linear + 11 angular).')
    if len(t) < 3 or not np.all(np.diff(t) > 0):
        raise ValueError('CSV timestamps must be strictly increasing.')
    if DELAY_S < 0:
        raise ValueError('DELAY_S must be nonnegative.')

    weights = torch.tensor(a, dtype=torch.float64)
    predicted = np.full_like(measured, np.nan)
    acceleration = np.full((len(t), 2), np.nan)

    # 2. Initialize from the first sample with known delayed PWM history
    start = int(np.searchsorted(t, t[0] + DELAY_S, side='left'))
    if start >= len(t) - 1:
        raise ValueError('Experiment is too short for the selected delay.')
    predicted[start] = measured[start]

    # 3. Free-running RK4 rollout with recorded PWM (no weight updates)
    for i in range(start, len(t) - 1):
        state = torch.tensor(predicted[i], dtype=torch.float64)
        dt = float(t[i + 1] - t[i])
        current_time = float(t[i])

        while current_time < t[i + 1] - 1e-12:
            command_index = int(np.searchsorted(t, current_time - DELAY_S + 1e-10, side='right') - 1)
            command_index = max(command_index, 0)
            pwm = torch.tensor(u[command_index], dtype=torch.float64)
            next_time = float(t[i + 1])

            if command_index + 1 < len(t):
                switch_time = float(t[command_index + 1] + DELAY_S)
                if current_time + 1e-10 < switch_time < next_time:
                    next_time = switch_time

            if current_time == t[i]:
                derivative = rls_dynamics_torch(state, pwm, weights)
                acceleration[i] = derivative[[1, 3]].numpy()

            state = rk4_step(state, pwm, next_time - current_time, weights)
            current_time = next_time

        if not torch.isfinite(state).all() or torch.max(torch.abs(state)) > 1e6:
            print(f'Prediction diverged at t={t[i + 1]:.3f} s')
            break
        predicted[i + 1] = state.numpy()

    # 4. Calculate errors over the completed rollout
    valid = np.isfinite(predicted[:, 0])
    error = predicted[valid] - measured[valid]
    if len(error) == 0:
        raise RuntimeError('No valid predictions.')
    rmse = np.sqrt(np.mean(error ** 2, axis=0))
    print(f'Experiment: {CSV_PATH.name}')
    print(f'Predicted samples: {np.count_nonzero(valid)} / {len(t) - start}')
    print(f'Position RMSE: {rmse[0]:.4f} m')
    print(f'Velocity RMSE: {rmse[1]:.4f} m/s')
    print(f'Pitch RMSE: {np.rad2deg(rmse[2]):.4f} deg')
    print(f'Angular velocity RMSE: {rmse[3]:.4f} rad/s')

    # 5. Compare all four states and both accelerations
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    labels = ['Position (m)', 'Velocity (m/s)', 'Pitch (deg)', 'Angular velocity (rad/s)']

    for j in range(4):
        ax = axes.flat[j]
        actual = measured[:, j]
        estimate = predicted[:, j]
        if j == 2:
            actual = np.rad2deg(actual)
            estimate = np.rad2deg(estimate)
        ax.plot(t, actual, label='Measured', color='black')
        ax.plot(t, estimate, '--', label='RLS + RK4', color='tab:blue')
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.3)
        ax.legend()

    accel_columns = ['v_dot', 'omega_dot']
    accel_labels = ['Linear acceleration (m/s²)', 'Angular acceleration (rad/s²)']
    for j in range(2):
        ax = axes[2, j]
        if accel_columns[j] in df.columns:
            ax.plot(t, df[accel_columns[j]], label='Recorded acceleration', color='black')
        ax.plot(t, acceleration[:, j], '--', label='RLS predicted', color='tab:blue')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(accel_labels[j])
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.show()


if __name__ == '__main__':
    main()
