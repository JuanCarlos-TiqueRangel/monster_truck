#!/usr/bin/env python3
"""Offline trajectory-error refinement of the existing 21 angular coefficients.

The linear model, feature maps, and all input files stay unchanged. This is an
Adam refinement after RLS initialization, NOT another RLS update. Prediction
still uses the same saved history model and the same controller projections.
Only CSV filenames in the initial model's training manifest may train weights;
the validation CSV never participates in the objective or checkpoint selection.
No ROS connection or published commands. A saved candidate is NOT a deployment
approval. Compare whole held-out trajectories and the zero-input audit afterward.
"""

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rls_history_mppi import RLSHistoryDynamics


def load_windows(paths, horizon, nominal_dt, min_steps):
    """Use real histories, original rows, and contiguous variable-length futures."""
    windows = []
    features = []
    counts = []
    for path in paths:
        df = pd.read_csv(path)
        cols = ["elapsed_s", "u", "x", "v", "theta_rad", "omega"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        a = df[cols].to_numpy(dtype=np.float64)
        if len(a) < 6 or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError(f"{path.name}: need at least six finite, time-ordered rows")
        t, u, states = a[:, 0], a[:, 1], a[:, 2:]
        z = np.column_stack((u, states[:, 1:]))
        dt = np.diff(t)
        good = np.r_[(dt >= 0.5 * nominal_dt) & (dt <= 1.5 * nominal_dt), False]
        ids = np.arange(len(a))[:, None] - np.arange(5)
        valid = (ids >= 0).all(axis=1) & good[np.maximum(ids, 0)].all(axis=1)
        features.append(z[ids[valid]].reshape(-1, 20))
        counts.append(int(valid.sum()))
        for start in np.flatnonzero(valid):
            length = 0
            while length < horizon and start + length < len(a) - 1 and valid[start + length]:
                length += 1
            if length < min_steps:
                continue
            truth = np.repeat(states[start + length][None], horizon + 1, axis=0)
            truth[:length + 1] = states[start:start + length + 1]
            controls = np.zeros(horizon)
            controls[:length] = u[start:start + length]
            periods = np.full(horizon, nominal_dt)
            periods[:length] = dt[start:start + length]
            mask = np.arange(horizon) < length
            past_ids = start - np.arange(1, 5)
            target_w = (states[start + 1, 3] - states[start, 3]) / dt[start]
            windows.append((truth, controls, periods, mask, z[past_ids], t[start] - t[past_ids], target_w))
    if not windows:
        raise ValueError(f"No windows with at least {min_steps} admissible future intervals")
    arrays = [np.stack([row[i] for row in windows]) for i in range(7)]
    return arrays, np.vstack(features), np.asarray(counts, dtype=np.int64)


class AngularRollouts(torch.nn.Module):
    def __init__(self, base, saved):
        super().__init__()
        # These buffers stay fixed. Only alpha is an optimizer parameter.
        for name in ("centers", "center_norm", "v_mean", "v_scale", "v_beta", "v_bias"):
            self.register_buffer(name, getattr(base, name).clone())
        for name in ("w_mean", "w_scale"):
            self.register_buffer(name, torch.as_tensor(saved[name], device=base.centers.device, dtype=torch.float64))
        nv = int(saved["n_v_features"])
        alpha = torch.as_tensor(saved["learned_parameters"][nv:], device=base.centers.device, dtype=torch.float64)
        if alpha.shape != (21,):
            raise ValueError("This refiner requires the original affine 21-feature angular branch")
        self.alpha = torch.nn.Parameter(alpha.clone())
        self.register_buffer("initial_alpha", alpha.clone())
        self.gamma, self.dt_floor = base.gamma, base.dt_floor

    def acceleration(self, history, ages):
        h = history.flatten(start_dim=-2)
        rates = (history[:, :1, 1:] - history[:, 1:, 1:]) / ages.clamp_min(self.dt_floor).unsqueeze(-1)
        xv = (torch.cat((h, rates.flatten(start_dim=-2)), dim=-1) - self.v_mean) / self.v_scale
        d2 = (xv.square().sum(dim=-1, keepdim=True) + self.center_norm - 2.0 * (xv @ self.centers.T)).clamp_min(0)
        av = torch.exp(-self.gamma * d2) @ self.v_beta + self.v_bias
        aw = ((h - self.w_mean) / self.w_scale) @ self.alpha[:-1] + self.alpha[-1]
        return av, aw

    def step(self, state, u, past, ages, dt):
        h = torch.cat((torch.cat((u[:, None], state[:, 1:]), dim=-1)[:, None], past), dim=1)
        av, aw = self.acceleration(h, ages)
        x = state[:, 0] + dt * state[:, 1]
        v = state[:, 1] + dt * av
        theta = state[:, 2] + dt * state[:, 3]
        omega = state[:, 3] + dt * aw
        landing = (theta >= 0.0) & (omega > 0.0)
        next_state = torch.stack((x, v.clamp(-10.0, 10.0), theta.clamp(max=0.0),
                                  torch.where(landing, torch.zeros_like(omega), omega.clamp(-30.0, 30.0))), dim=-1)
        next_ages = torch.cat((torch.zeros_like(ages[:, :1]), ages[:, :3]), dim=-1) + dt[:, None]
        return next_state, h[:, :4], next_ages

    def objective(self, batch, theta_scale, omega_scale, point_weight, anchor_weight):
        truth, controls, dt, mask, past, ages, targets = batch
        state = truth[:, 0]
        h0 = torch.cat((torch.cat((controls[:, 0, None], state[:, 1:]), dim=-1)[:, None], past), dim=1)
        _, aw = self.acceleration(h0, ages)
        point_loss = ((dt[:, 0] * (aw - targets)) / omega_scale).square().mean()
        accumulated = torch.zeros(len(state), device=state.device, dtype=state.dtype)
        for j in range(controls.shape[1]):
            candidate, new_past, new_ages = self.step(state, controls[:, j], past, ages, dt[:, j])
            active = mask[:, j]
            state = torch.where(active[:, None], candidate, state)
            past = torch.where(active[:, None, None], new_past, past)
            ages = torch.where(active[:, None], new_ages, ages)
            e_theta = (state[:, 2] - truth[:, j + 1, 2]) / theta_scale
            e_omega = (state[:, 3] - truth[:, j + 1, 3]) / omega_scale
            accumulated = accumulated + active * (e_theta.square() + e_omega.square())
        rollout_loss = (accumulated / mask.sum(dim=1).clamp_min(1)).mean()
        anchor_loss = (self.alpha - self.initial_alpha).square().mean()
        return rollout_loss + point_weight * point_loss + anchor_weight * anchor_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("rls_history_angular_rollout.npz"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--min-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--theta-scale", type=float, default=0.10, help="Angle-error normalization, radians")
    parser.add_argument("--omega-scale", type=float, default=1.0, help="Angular-rate-error normalization, rad/s")
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 1 <= args.min_steps <= args.horizon:
        parser.error("Require positive steps/batch size and 1 <= min-steps <= horizon")
    if min(args.learning_rate, args.theta_scale, args.omega_scale) <= 0 or min(args.point_weight, args.anchor_weight) < 0:
        parser.error("Learning rate and scales must be positive; loss weights must be nonnegative")
    if not args.model.is_file() or not args.dataset.is_dir():
        parser.error("Model file and dataset directory must exist")
    if args.output.resolve() == args.model.resolve() or args.output.exists():
        parser.error("Use a NEW output path; this script never overwrites a model")
    if args.output.suffix != ".npz" or not args.output.parent.is_dir():
        parser.error("Output must be an .npz in an existing directory")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.model, allow_pickle=False) as source:
        saved = {name: source[name].copy() for name in source.files}
    if str(saved.get("target_type", "")) != "forward_velocity_difference":
        parser.error("Use the forward-trained starting model, not the legacy acceleration-target model")
    names = [Path(str(name)).name for name in saved["training_files"]]
    validation = str(saved.get("validation_file", "rls_id_015.csv"))
    if len(set(names)) != len(names) or validation in names:
        parser.error("Invalid manifest: duplicate names or validation file present in training")
    paths = [args.dataset / name for name in names]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        parser.error(f"Missing retained training CSVs: {missing}. No glob fallback or partial-data fitting is allowed.")
    base = RLSHistoryDynamics(args.model, device=args.device, dtype=torch.float64)
    arrays, xw, counts = load_windows(paths, args.horizon, base.nominal_dt, args.min_steps)
    if "training_update_counts" in saved and not np.array_equal(counts, saved["training_update_counts"]):
        parser.error("Eligible-transition counts differ from the base training run; review the data before refining")
    if not np.allclose(xw.mean(axis=0), saved["w_mean"], rtol=1e-6, atol=1e-8) or not np.allclose(xw.std(axis=0), saved["w_scale"], rtol=1e-6, atol=1e-8):
        parser.error("Training history statistics differ from the base model. Use the same retained CSVs, without further filtering or edits.")
    dev = base.centers.device
    data = tuple(torch.as_tensor(a, device=dev, dtype=torch.bool if i == 3 else torch.float64) for i, a in enumerate(arrays))
    net = AngularRollouts(base, saved)
    kwargs = dict(theta_scale=args.theta_scale, omega_scale=args.omega_scale, point_weight=args.point_weight, anchor_weight=args.anchor_weight)
    check_ids = torch.as_tensor(rng.choice(len(data[0]), size=min(256, len(data[0])), replace=False), device=dev)
    check = tuple(a[check_ids] for a in data)
    optimizer = torch.optim.Adam([net.alpha], lr=args.learning_rate)
    with torch.no_grad():
        best_loss = float(net.objective(check, **kwargs).item())
        best_alpha = net.alpha.detach().clone()
    if not np.isfinite(best_loss):
        raise RuntimeError("Nonfinite initial training rollout loss; no model saved")
    initial_loss, best_iteration = best_loss, 0
    print("OFFLINE REFINEMENT: Adam on 21 angular coefficients after RLS; linear model stays fixed.")
    print(f"Training files={len(paths)}, eligible transitions={counts.sum()}, rollout starts={len(data[0])}, horizon<= {args.horizon}, device={dev}, dtype=float64")
    print(f"Held-out file {validation!r} is NOT read for fitting or checkpoint selection.")
    print(f"Fixed TRAINING-window objective, initial={initial_loss:.6f}")
    for iteration in range(1, args.steps + 1):
        ids = torch.as_tensor(rng.integers(len(data[0]), size=args.batch_size), device=dev)
        batch = tuple(a[ids] for a in data)
        optimizer.zero_grad(set_to_none=True)
        loss = net.objective(batch, **kwargs)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration}; no model saved")
        loss.backward()
        if net.alpha.grad is None or not bool(torch.isfinite(net.alpha.grad).all()):
            raise RuntimeError(f"Nonfinite/missing gradients at iteration {iteration}; no model saved")
        torch.nn.utils.clip_grad_norm_([net.alpha], 10.0)
        optimizer.step()
        if iteration % 100 == 0 or iteration == args.steps:
            with torch.no_grad():
                score = float(net.objective(check, **kwargs).item())
            if np.isfinite(score) and score < best_loss:
                best_loss, best_iteration = score, iteration
                best_alpha = net.alpha.detach().clone()
            print(f"Iteration {iteration}: fixed TRAINING objective={score:.6f}, best={best_loss:.6f}")
    if best_iteration == 0:
        print("No improvement on the fixed training-window objective. No candidate model was saved.")
        return
    nv = int(saved["n_v_features"])
    original = saved["learned_parameters"].copy()
    saved["learned_parameters"][nv:] = best_alpha.cpu().numpy()
    if not np.array_equal(original[:nv], saved["learned_parameters"][:nv]):
        raise RuntimeError("Linear parameters changed unexpectedly; no output saved")
    if "covariance" in saved:
        # This old RLS covariance is not valid for continuing RLS after Adam.
        saved["rls_initial_covariance"] = saved.pop("covariance")
    saved.update(fit_method=np.asarray("RLS_initialization_plus_angular_Adam_rollout_refinement"),
                 supports_online_rls_resume=np.asarray(False), initial_angular_parameters=original[nv:],
                 refinement_base_sha256=np.asarray(hashlib.sha256(args.model.read_bytes()).hexdigest()),
                 refinement_steps=np.asarray(args.steps), selected_iteration=np.asarray(best_iteration),
                 refinement_horizon=np.asarray(args.horizon), refinement_min_steps=np.asarray(args.min_steps),
                 refinement_learning_rate=np.asarray(args.learning_rate), refinement_theta_scale=np.asarray(args.theta_scale),
                 refinement_omega_scale=np.asarray(args.omega_scale), refinement_point_weight=np.asarray(args.point_weight),
                 refinement_anchor_weight=np.asarray(args.anchor_weight), refinement_seed=np.asarray(args.seed),
                 initial_training_objective=np.asarray(initial_loss), selected_training_objective=np.asarray(best_loss),
                 refinement_linear_branch_frozen=np.asarray(True))
    with args.output.open("xb") as output:
        np.savez_compressed(output, **saved)
    # Confirm the existing inference class can read the candidate.
    RLSHistoryDynamics(args.output, device="cpu", dtype=torch.float64)
    print(f"Saved candidate: {args.output.resolve()}")
    print("Training-loss improvement is NOT validation. Run inspect_forward_model.py and the zero-input audit on this candidate before any drive-enabled test.")
    print("The old RLS covariance is retained as rls_initial_covariance only; do not resume online RLS with it.")


if __name__ == "__main__":
    main()
