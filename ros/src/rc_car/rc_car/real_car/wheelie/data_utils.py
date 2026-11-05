"""Strict experiment loading and frozen-rollout metrics; no smoothing or row shuffling."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch


@dataclass
class Experiment:
    number: int
    frame: pd.DataFrame
    state: np.ndarray
    u: np.ndarray
    time: np.ndarray
    accel: np.ndarray


def load_experiments(directory, ids):
    experiments = []
    required = ["elapsed_s", "x", "v", "theta_rad", "omega", "u", "v_dot", "omega_dot"]
    for number in ids:
        path = Path(directory) / f"rls_id_{number:02d}.csv"
        df = pd.read_csv(path)
        missing = set(required) - set(df)
        if missing:
            raise ValueError(f"{path.name}: missing {sorted(missing)}")
        data = df[required].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if len(data) < 2 or not np.isfinite(data).all():
            raise ValueError(f"{path.name}: requires >=2 finite rows; data are not silently deleted.")
        time = data[:, 0] - data[0, 0]
        if not (np.diff(time) > 0).all():
            raise ValueError(f"{path.name}: timestamps must strictly increase in recorded order.")
        experiments.append(Experiment(number, df, data[:, 1:5], data[:, 5], time, data[:, 6:8]))
    return experiments


def state_scales(experiments):
    states = np.concatenate([e.state[:, 1:] for e in experiments])
    return np.maximum(np.std(states, axis=0), [0.25, 0.15, 0.5])


def pack_windows(experiments, horizon=None, every_start=False, device="cpu", dtype=torch.float32):
    windows = []
    for e in experiments:
        starts = range(len(e.time) - 1) if every_start else [0]
        for start in starts:
            end = len(e.time) if horizon is None else min(start + horizon + 1, len(e.time))
            windows.append((e, start, end))
    H = max(end - start - 1 for _, start, end in windows)
    N = len(windows)
    states = np.zeros((N, H + 1, 4))
    u = np.zeros((N, H))
    dt = np.zeros((N, H, 1))
    mask = np.zeros((N, H))
    # Normalize each run by that run's measured variance, with fixed floors.
    # In training only, these are loss weights, not information supplied at inference.
    scales = np.zeros((N, 3))
    lengths = np.zeros(N, dtype=int)
    for i, (e, start, end) in enumerate(windows):
        h = end - start - 1
        states[i, :h + 1] = e.state[start:end]
        states[i, h + 1:] = e.state[end - 1]
        u[i, :h] = e.u[start:end - 1]
        dt[i, :h, 0] = np.diff(e.time[start:end])
        mask[i, :h] = 1.0
        scales[i] = np.maximum(np.std(e.state[1:, 1:], axis=0), [0.25, 0.15, 0.5])
        lengths[i] = h
    to_tensor = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    return tuple(to_tensor(x) for x in (states, u, dt, mask, scales, lengths))


def simulation_loss(model, batch):
    truth, inputs, dts, mask, scales, lengths = batch
    state = truth[:, 0]
    terms = torch.zeros_like(scales)
    for k in range(inputs.shape[1]):
        prediction = model.step(state, inputs[:, k], dts[:, k])
        state = torch.where(mask[:, k:k + 1] > 0, prediction, state)
        error = (state[:, 1:] - truth[:, k + 1, 1:]) / scales
        terms = terms + error.square() * mask[:, k:k + 1]
    return (terms / lengths[:, None]).mean()


def score(y, yhat):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if not np.isfinite(yhat).all():
        return {"RMSE": float("inf"), "MAE": float("inf"), "bias": float("nan"), "R2": float("-inf")}
    err = yhat - y
    sst = np.sum((y - y.mean())**2)
    return {"RMSE": float(np.sqrt(np.mean(err**2))), "MAE": float(np.mean(abs(err))),
            "bias": float(err.mean()), "R2": float(1 - np.sum(err**2) / sst) if sst > 1e-12 else float("nan")}


@torch.no_grad()
def evaluate_model(model, experiments, name, split, device="cpu", dtype=torch.float32):
    rows, prediction_rows = [], []
    model.eval()
    for e in experiments:
        s = torch.as_tensor(e.state, device=device, dtype=dtype)
        u = torch.as_tensor(e.u, device=device, dtype=dtype)
        dt = torch.as_tensor(np.diff(e.time), device=device, dtype=dtype)
        states = [s[0]]
        for k in range(len(dt)):
            states.append(model.step(states[-1], u[k], dt[k]))
        predicted = torch.stack(states)
        mapping = model(s, u)
        rollout_deriv = model(predicted, u)
        pred, mapped, rolled = (t.detach().cpu().numpy() for t in (predicted, mapping, rollout_deriv))
        arrays = {"v": (e.state[:, 1], pred[:, 1]), "omega": (e.state[:, 3], pred[:, 3]),
                  "pitch_deg": (np.rad2deg(e.state[:, 2]), np.rad2deg(pred[:, 2])),
                  "v_dot_mapping": (e.accel[:, 0], mapped[:, 1]),
                  "omega_dot_mapping": (e.accel[:, 1], mapped[:, 3]),
                  "v_dot_rollout": (e.accel[:, 0], rolled[:, 1]),
                  "omega_dot_rollout": (e.accel[:, 1], rolled[:, 3])}
        for signal, (y, yp) in arrays.items():
            # Do not count the deliberately matched initial state as a prediction.
            start = 1 if signal in ("v", "omega", "pitch_deg") else 0
            rows.append({"model": name, "split": split, "experiment": e.number, "signal": signal,
                         "n": len(y) - start, **score(y[start:], yp[start:])})
        for k in range(len(e.time)):
            row = {"model": name, "split": split, "experiment": e.number, "k": k,
                   "time": e.time[k], "u": e.u[k]}
            for signal, (y, yp) in arrays.items():
                row[signal + "_measured"] = float(y[k])
                row[signal + "_predicted"] = float(yp[k])
            prediction_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(prediction_rows)
