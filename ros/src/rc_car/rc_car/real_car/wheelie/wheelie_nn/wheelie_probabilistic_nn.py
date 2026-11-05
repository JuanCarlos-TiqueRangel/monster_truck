"""Three Gaussian-NN experiments: acceleration fit, integrated states, deltas.

Run: python3 wheelie_probabilistic_nn.py
Edit VALIDATION_EXPERIMENTS below. No ROS or controller code is imported.
Validation chooses checkpoints; its metrics are validation, not blind-test scores.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- EDIT THESE SETTINGS -----------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
VALIDATION_EXPERIMENTS = [45]       # e.g. [17, 18, 19]; never fitted by gradients
EXCLUDED_EXPERIMENTS = []          # remaining available CSVs are training runs
OUTPUT_DIR = BASE_DIR / "probabilistic_results"
HISTORY = 5                       # current observation plus four past samples
HIDDEN = 128
ENSEMBLE_SIZE = 3
EPOCHS = 2000
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 200
NLL_WEIGHT = 0.1                  # PNN loss = MSE + this * Gaussian NLL
ACCEL_FIT_ROLLOUT_WEIGHT = 0.25    # experiment 1: emphasize logged acceleration fit
ACCEL_INTEGRATION_ROLLOUT_WEIGHT = 2.0  # experiment 2: emphasize integrated states
DELTA_ROLLOUT_WEIGHT = 0.25        # experiment 3: discrete velocity increments
ROLLOUT_STEPS = 5
ACCEL_TARGET_OFFSET = 0           # 0: original same-row labels; 1: next-row labels
LOGVAR_MIN, LOGVAR_MAX = -8.0, 2.0 # bounds in normalized target units
PARTICLES = 500                   # for probabilistic trajectory intervals
DEVICE = "cuda"                   # tiny dataset; may change to "cuda"
SEED = 42
# ------------------------------------------------------------------


def load_data(directory):
    data = {}
    for path in sorted(Path(directory).glob("rls_id_*.csv")):
        i = int(path.stem.rsplit("_", 1)[1])
        df = pd.read_csv(path)
        t = df.elapsed_s.to_numpy(float)
        z = df[["u", "v", "theta_rad", "omega"]].to_numpy(np.float32)
        a = df[["v_dot", "omega_dot"]].to_numpy(np.float32)
        if not np.isfinite(np.c_[t, z, a]).all() or not (np.diff(t) > 0).all():
            raise ValueError(f"Invalid required data in {path}")
        data[i] = dict(t=t-t[0], dt=np.diff(t).astype(np.float32), z=z, a=a)
    return data


def context(run, k, history=None):
    history = HISTORY if history is None else history
    indices = np.arange(k-history+1, k+1)
    h = run["z"][np.maximum(indices, 0)].copy()
    h[indices < 0, 0] = 0.0  # startup padding assumes neutral pre-record commands
    return h


def pack(data, ids, target, horizon=1, device=DEVICE):
    hs, us, ds, ys, states, weights = [], [], [], [], [], []
    for i in ids:
        r = data[i]
        n = len(r["dt"]) - horizon + 1
        for k in range(max(n, 0)):
            hs.append(context(r, k)); us.append(r["z"][k:k+horizon, 0])
            ds.append(r["dt"][k:k+horizon]); states.append(r["z"][k+1:k+horizon+1, 1:])
            if target == "acceleration":
                ys.append(r["a"][k+ACCEL_TARGET_OFFSET])
            else:
                ys.append(r["z"][k+1, [1, 3]] - r["z"][k, [1, 3]])
            weights.append(1/max(n, 1))  # equal total weight per experiment
    if not hs:
        raise ValueError("Experiments are too short for the chosen history/horizon")
    return tuple(torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
                 for x in (hs, us, ds, ys, states, weights))


class GaussianNN(nn.Module):
    def __init__(self, stats, history=None, hidden=None, bounds=None):
        super().__init__()
        self.history = HISTORY if history is None else history
        hidden = HIDDEN if hidden is None else hidden
        self.bounds = (LOGVAR_MIN, LOGVAR_MAX) if bounds is None else bounds
        for name, values in stats.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(4*self.history+1, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, 2)
        self.logvar = nn.Linear(hidden, 2)
        nn.init.normal_(self.mean.weight, std=0.01); nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.logvar.weight)
        p = -self.bounds[0]/(self.bounds[1]-self.bounds[0])
        nn.init.constant_(self.logvar.bias, math.log(p/(1-p)))  # initial variance 1


    def forward(self, history, dt):
        dt = torch.as_tensor(dt, dtype=history.dtype, device=history.device)
        dt = torch.broadcast_to(dt, history.shape[:-2])
        x = torch.cat([((history-self.x_mean)/self.x_std).flatten(-2), (dt/0.04).unsqueeze(-1)], -1)
        h = self.body(x)
        return self.mean(h), self.bounds[0]+(self.bounds[1]-self.bounds[0])*torch.sigmoid(self.logvar(h))


    def physical(self, history, dt):
        mu, logvar = self(history, dt)
        return mu*self.y_std+self.y_mean, logvar.exp()*self.y_std.square()



def load_models(path, probabilistic=True, device="cpu"):
    """Reload architecture/normalization from the checkpoint, even after settings change."""
    saved = torch.load(path, map_location=device, weights_only=True)
    cfg = saved["config"]
    models = []
    for state in saved["pnn" if probabilistic else "deterministic"]:
        m = GaussianNN(saved["stats"], cfg["HISTORY"], cfg["HIDDEN"], (cfg["LOGVAR_MIN"], cfg["LOGVAR_MAX"]))
        m.load_state_dict(state); models.append(m.to(device).eval())
    return models, saved["target"], cfg


def integrate(state, output, dt, target):
    """[v, theta, omega]; exact constant-acceleration/trapezoidal rate step."""
    dt = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
    change = output*dt[..., None] if target == "acceleration" else output
    v = state[..., 0]+change[..., 0]
    w = state[..., 2]+change[..., 1]
    theta = state[..., 1]+0.5*dt*(state[..., 2]+w)
    return torch.stack([v, theta, w], -1)


def advance(h, state, command):
    return torch.cat([h[..., 1:, :], torch.cat([command[..., None], state], -1).unsqueeze(-2)], -2)


def training_rollout(model, batch, target):
    h, u, dt, _, truth, weights = batch
    error = torch.zeros_like(weights)
    for k in range(u.shape[1]):
        value, _ = model.physical(h, dt[:, k])
        state = integrate(h[:, -1, 1:], value, dt[:, k], target)
        error = error + ((state-truth[:, k])/model.state_std).square().mean(-1)/u.shape[1]
        h = advance(h, state, u[:, min(k+1, u.shape[1]-1)])
    return (error*weights).sum()/weights.sum()


@torch.no_grad()
def rollout(models, run, target, samples=0):
    # Initialize with measured history only; no future measured states enter.
    start = models[0].history-1
    h = torch.tensor(context(run, start, models[0].history), device=next(models[0].parameters()).device)[None]
    h = h.repeat(samples or 1, 1, 1)
    paths = [h[:, -1, 1:].clone()]
    for k in range(start, len(run["dt"])):
        dt = float(run["dt"][k])
        if samples:
            output = torch.empty((samples, 2), device=h.device)
            for j, model in enumerate(models):
                mu, var = model.physical(h[j::len(models)], dt)
                output[j::len(models)] = mu + torch.randn_like(mu)*var.sqrt()
        else:
            output = torch.stack([m.physical(h, dt)[0] for m in models]).mean(0)
        state = integrate(h[:, -1, 1:], output, dt, target)
        command = torch.full((len(h),), float(run["z"][k+1, 0]), device=h.device)
        h = advance(h, state, command); paths.append(state)
    return torch.stack(paths, 1).cpu().numpy()


@torch.no_grad()
def validation_score(model, data, ids, target):
    b = pack(data, ids, target, device=next(model.parameters()).device)
    mu, _ = model(b[0], b[2][:, 0])
    direct = ((mu-(b[3]-model.y_mean)/model.y_std).square().mean(-1)*b[5]).sum()/b[5].sum()
    errors = []
    for i in ids:
        pred = rollout([model], data[i], target)[0, 1:]
        truth = data[i]["z"][HISTORY:, 1:]
        errors.append(np.mean(((pred-truth)/model.state_std.cpu().numpy())**2))
    return float(direct) + float(np.mean(errors))


def train(data, train_ids, validation_ids, target, probabilistic, epochs, device, rollout_weight=0.25):
    warmup = min(WARMUP_EPOCHS, epochs//4)
    b = pack(data, train_ids, target, device=device)
    x = torch.tensor(np.concatenate([data[i]["z"] for i in train_ids]), device=device)
    stats = dict(x_mean=x.mean(0).tolist(), x_std=x.std(0).clamp_min(1e-3).tolist(),
                 y_mean=b[3].mean(0).tolist(), y_std=b[3].std(0).clamp_min(1e-4).tolist(),
                 state_std=x[:, 1:].std(0).clamp_min(0.05).tolist())
    rb = pack(data, train_ids, target, ROLLOUT_STEPS, device)
    models, logs = [], []
    label = "PNN" if probabilistic else "deterministic"
    for member in range(ENSEMBLE_SIZE):
        torch.manual_seed(SEED+member)
        model = GaussianNN(stats).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=LEARNING_RATE/20)
        best, selected, best_epoch = float("inf"), None, 0
        h, _, dt, y, _, weight = b
        yn = (y-model.y_mean)/model.y_std
        for ep in range(1, epochs+1):
            mu, lv = model(h, dt[:, 0])
            mse = (mu-yn).square().mean(-1)
            nll = 0.5*(lv+(mu-yn).square()*(-lv).exp()).mean(-1)
            row_loss = mse + (NLL_WEIGHT*nll if probabilistic and ep > warmup else 0)
            loss = (weight*row_loss).sum()/weight.sum()
            if ep > warmup and rollout_weight:
                loss = loss + rollout_weight*training_rollout(model, rb, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); scheduler.step()
            if ep % 50 == 0 or ep == epochs:
                score = validation_score(model, data, validation_ids, target)
                logs.append(dict(target=target, model=label, member=member, epoch=ep,
                                 train_mse=float(mse.mean().detach()), train_nll=float(nll.mean().detach()),
                                 objective=float(loss.detach()), validation_score=score))
                if ep > warmup and np.isfinite(score) and score < best:
                    best, best_epoch = score, ep
                    selected = copy.deepcopy(model.state_dict())
        if selected is None:
            raise FloatingPointError(f"No finite validation checkpoint: {target}/{label}")
        model.load_state_dict(selected); model.selected_epoch = best_epoch
        models.append(model.eval())
        print(f"{target:12s} {label:13s} member {member+1}: epoch {best_epoch}, validation score {best:.5f}", flush=True)
    return models, logs, stats


def rmse(a, b):
    return np.sqrt(np.mean((np.asarray(a, float)-np.asarray(b, float))**2, axis=0))


@torch.no_grad()
def evaluate(models, baseline, data, ids, target, out):
    rows, predictions = [], []
    device = next(models[0].parameters()).device
    for i in ids:
        r = data[i]; start = HISTORY-1
        h = torch.tensor(np.array([context(r, k) for k in range(start, len(r["dt"]))]), device=device)
        dt = torch.tensor(r["dt"][start:], device=device)
        y = r["a"][start+ACCEL_TARGET_OFFSET:len(r["dt"])+ACCEL_TARGET_OFFSET] if target == "acceleration" else np.diff(r["z"][:, [1, 3]], axis=0)[start:]
        means, variances = zip(*[m.physical(h, dt) for m in models])
        means, variances = torch.stack(means), torch.stack(variances)
        mu = means.mean(0).cpu().numpy()
        # Exact mixture likelihood, not the average member NLL.
        yy = torch.tensor(y, device=device)
        logp = -0.5*(((yy-means)**2/variances)+variances.log()+math.log(2*math.pi)).sum(-1)
        mixture_nll = float(-(torch.logsumexp(logp, dim=0)-math.log(len(models))).mean())
        draw = means[:, :, None, :] + variances.sqrt()[:, :, None, :]*torch.randn((*means.shape[:2], PARTICLES, 2), device=device)
        draw = draw.permute(1, 0, 2, 3).reshape(len(y), -1, 2).cpu().numpy()
        lower, upper = np.quantile(draw, [0.05, 0.95], axis=1)
        base_mu = torch.stack([m.physical(h, dt)[0] for m in baseline]).mean(0).cpu().numpy()
        pred = rollout(models, r, target)[0]
        base_pred = rollout(baseline, r, target)[0]
        trajectories = rollout(models, r, target, PARTICLES)
        lo, hi = np.quantile(trajectories, [0.05, 0.95], axis=0)
        truth = r["z"][start:, 1:]
        for name, direct, path in [("PNN", mu, pred), ("deterministic", base_mu, base_pred)]:
            aerr = rmse(y, direct); serr = rmse(truth[1:], path[1:])
            rows.append(dict(experiment=i, target=target, model=name, direct_linear_rmse=aerr[0], direct_angular_rmse=aerr[1],
                             rollout_v_rmse=serr[0], rollout_pitch_rmse_deg=np.rad2deg(serr[1]), rollout_omega_rmse=serr[2],
                             direct_nll=mixture_nll if name == "PNN" else np.nan,
                             coverage_linear=float(np.mean((y[:, 0]>=lower[:, 0]) & (y[:, 0]<=upper[:, 0]))) if name == "PNN" else np.nan,
                             coverage_angular=float(np.mean((y[:, 1]>=lower[:, 1]) & (y[:, 1]<=upper[:, 1]))) if name == "PNN" else np.nan,
                             rollout_pitch_coverage=float(np.mean((truth[1:, 1]>=lo[1:, 1]) & (truth[1:, 1]<=hi[1:, 1]))) if name == "PNN" else np.nan,
                             rollout_pitch_width_deg=float(np.rad2deg((hi[1:, 1]-lo[1:, 1]).mean())) if name == "PNN" else np.nan))
        direct_t = r["t"][start+ACCEL_TARGET_OFFSET:len(r["dt"])+ACCEL_TARGET_OFFSET] if target == "acceleration" else r["t"][start+1:]
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, constrained_layout=True)
        labels = ["Linear acceleration (m/s²)", "Angular acceleration (rad/s²)"] if target == "acceleration" else ["Δ velocity (m/s)", "Δ angular rate (rad/s)"]
        for j, ax in enumerate(axes):
            ax.plot(direct_t, y[:, j], label="Measured", color="black")
            ax.plot(direct_t, mu[:, j], label="Probabilistic NN", color="C0")
            ax.plot(direct_t, base_mu[:, j], "--", label="Deterministic NN", color="C1")
            ax.fill_between(direct_t, lower[:, j], upper[:, j], alpha=.2, color="C0", label="PNN 90% interval")
            ax.set_ylabel(labels[j]); ax.grid(alpha=.25)
        axes[0].legend(); axes[-1].set_xlabel("Time since first recording (s)")
        fig.suptitle(f"Run {i}: {target} predictions from measured histories")
        fig.savefig(out/f"{target}_predictions_{i:02d}.png", dpi=150); plt.close(fig)
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
        for j, (ax, label, factor) in enumerate(zip(axes, ["Velocity (m/s)", "Pitch (degrees)", "Angular rate (rad/s)"], [1, 180/np.pi, 1])):
            ax.plot(r["t"], r["z"][:, j+1]*factor, color="black", label="Measured")
            ax.plot(r["t"][start:], pred[:, j]*factor, label="Probabilistic NN", color="C0")
            ax.plot(r["t"][start:], base_pred[:, j]*factor, "--", label="Deterministic NN", color="C1")
            ax.fill_between(r["t"][start:], lo[:, j]*factor, hi[:, j]*factor, alpha=.2, label="PNN 90% interval", color="C0")
            ax.axvspan(r["t"][0], r["t"][start], color="gray", alpha=.12, label="Measured initialization")
            ax.set_ylabel(label); ax.grid(alpha=.25)
        axes[0].legend(fontsize=8); axes[-1].set_xlabel("Time since first recording (s)")
        fig.suptitle(f"Run {i}: recursive {target} rollout; predictions fed back after initialization")
        fig.savefig(out/f"{target}_rollout_{i:02d}.png", dpi=150); plt.close(fig)
        for k in range(1, len(pred)):
            row = dict(experiment=i, target=target, time=r["t"][start+k])
            for j, key in enumerate(["v", "theta", "omega"]):
                row.update({key+"_true": truth[k, j], key+"_pnn": pred[k, j], key+"_deterministic": base_pred[k, j],
                            key+"_lower90": lo[k, j], key+"_upper90": hi[k, j]})
            predictions.append(row)
    return rows, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--validation", type=int, nargs="+", default=VALIDATION_EXPERIMENTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.epochs < 1 or HISTORY < 1 or ACCEL_TARGET_OFFSET not in [0, 1]:
        raise ValueError("Invalid epochs, history or acceleration offset")
    data = load_data(args.data)
    validation = sorted(set(args.validation))
    train_ids = sorted(set(data)-set(validation)-set(EXCLUDED_EXPERIMENTS))
    if not train_ids or not validation or not set(validation).issubset(data):
        raise ValueError(f"Invalid split. Available experiments: {sorted(data)}")
    if any(len(data[i]["dt"]) < max(HISTORY, ROLLOUT_STEPS) for i in train_ids+validation):
        raise ValueError("A selected experiment is too short")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, list))}
    config.update(train_ids=train_ids, validation_ids=validation, epochs=args.epochs, device=args.device,
                  effective_warmup=min(WARMUP_EPOCHS, args.epochs//4), data_dir=str(args.data.resolve()),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), torch_version=str(torch.__version__),
                  selection="Validation chooses checkpoints; no blind-test claim", state_order=["v", "theta_rad", "omega"])
    (out/"config.json").write_text(json.dumps(config, indent=2))
    (out/"wheelie_probabilistic_nn.py").write_text(Path(__file__).read_text())
    print(f"Train {train_ids}; validate {validation}; device {args.device}", flush=True)
    all_rows, all_predictions, all_logs = [], [], []
    studies = [("1_acceleration_fit", "acceleration", ACCEL_FIT_ROLLOUT_WEIGHT),
               ("2_acceleration_integrated", "acceleration", ACCEL_INTEGRATION_ROLLOUT_WEIGHT),
               ("3_delta", "delta", DELTA_ROLLOUT_WEIGHT)]
    for study, target, weight in studies:
        destination = out/study
        destination.mkdir()
        print(f"Starting {study}; rollout loss weight {weight}", flush=True)
        ensemble, logs, stats = train(data, train_ids, validation, target, True, args.epochs, args.device, weight)
        baseline, base_logs, _ = train(data, train_ids, validation, target, False, args.epochs, args.device, weight)
        torch.save(dict(config=dict(config, study=study, rollout_weight=weight), target=target, stats=stats,
                        selected_epochs=dict(pnn=[m.selected_epoch for m in ensemble], deterministic=[m.selected_epoch for m in baseline]),
                        pnn=[m.cpu().state_dict() for m in ensemble], deterministic=[m.cpu().state_dict() for m in baseline]),
                   destination/f"{target}_models.pt")
        ensemble = [m.to(args.device) for m in ensemble]; baseline = [m.to(args.device) for m in baseline]
        torch.manual_seed(SEED+1000)
        rows, predictions = evaluate(ensemble, baseline, data, validation, target, destination)
        for row in rows+predictions+logs+base_logs:
            row["study"] = study
        pd.DataFrame(rows).to_csv(destination/"metrics.csv", index=False)
        all_rows += rows; all_predictions += predictions; all_logs += logs+base_logs
    pd.DataFrame(all_rows).to_csv(out/"metrics.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(out/"predictions.csv", index=False)
    pd.DataFrame(all_logs).to_csv(out/"training.csv", index=False)
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)
    print(f"Saved results: {out}", flush=True)


if __name__ == "__main__":
    main()
