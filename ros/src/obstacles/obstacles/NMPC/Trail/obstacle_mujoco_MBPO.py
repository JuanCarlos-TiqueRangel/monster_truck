#!/usr/bin/env python3
"""MBPO baseline using the same MuJoCo task, observation and reward as SAC.

Janner et al., 'When to Trust Your Model: Model-Based Policy Optimization'
(NeurIPS 2019), https://arxiv.org/abs/1906.08253. This task adaptation uses
SB3 SAC, a bootstrapped probabilistic dynamics ensemble with held-out elites,
and short stochastic model rollouts branched from REAL replay observations.
The known task reward/termination is reused rather than learned. Dynamics
predicts the four state-observation deltas; torque memory is updated exactly.
The five-observation state, inherited from SAC, omits some MuJoCo state, so
this is an approximate learned model; one-step rollouts are the default.

    python3 obstacle_mujoco_MBPO.py --device cpu --total-timesteps 300000
    python3 obstacle_mujoco_MBPO.py --resume --total-timesteps 100000
    python3 obstacle_mujoco_MBPO.py --eval-only --eval-episodes 100
    python3 obstacle_mujoco_MBPO.py --eval-only --render-eval --render-speed 0.25

Headless by default; --render-eval opens the viewer. On resume, timesteps are
ADDITIONAL real environment steps. Checkpoints contain SAC, real replay,
ensemble/optimizers, RNG states and configuration. Resume starts a fresh real
episode and rebuilds synthetic replay; it does not restore a MuJoCo episode.
Only local checkpoints created by this script should be loaded.
Rendered evaluation uses wall-clock pacing: --render-speed 1 is real time,
0.5 is half speed and 0.25 is quarter speed. Simulation time and control
frequency are unchanged; headless episodes run without pacing.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from obstacle_gym_env import (
    ObstacleWheelieEnv, RESULTS_DIR, XML_PATH, compute_step_reward, is_crashed,
)


class MixedReplayBuffer(ReplayBuffer):
    """Primary storage is REAL only; SAC samples a mixture with synthetic data."""

    def __init__(self, *args, real_ratio=0.1, model_buffer_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise ValueError("MBPO requires one environment and explicit next observations")
        self.real_ratio = real_ratio
        self.synthetic = ReplayBuffer(
            model_buffer_size, self.observation_space, self.action_space,
            device=self.device, handle_timeout_termination=False,
        )

    def sample(self, batch_size, env=None):
        if not self.synthetic.size() or self.real_ratio == 1:
            return super().sample(batch_size, env)
        n_real = max(1, math.ceil(batch_size * self.real_ratio))
        n_model = batch_size - n_real
        if n_model == 0:
            return super().sample(batch_size, env)
        real = super().sample(n_real, env)
        model = self.synthetic.sample(n_model, env)
        return ReplayBufferSamples(*(
            None if a is None else torch.cat((a, b), dim=0)
            for a, b in zip(real, model)
        ))

    def add_synthetic(self, obs, actions, next_obs, rewards, dones):
        """Vectorized insertion into SB3's single-environment ring buffer."""
        buffer = self.synthetic
        n = min(len(obs), buffer.buffer_size)
        if n == 0:
            return
        indices = (np.arange(n) + buffer.pos) % buffer.buffer_size
        for key, values in (
            ("observations", obs), ("actions", actions),
            ("next_observations", next_obs), ("rewards", rewards), ("dones", dones),
        ):
            getattr(buffer, key)[indices, 0] = values[-n:]
        buffer.full = buffer.full or buffer.pos + n >= buffer.buffer_size
        buffer.pos = (buffer.pos + n) % buffer.buffer_size


class GaussianDynamics(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 8),
        )

    def forward(self, x):
        mean, raw_logvar = self.net(x).chunk(2, dim=-1)
        logvar = 0.5 - F.softplus(0.5 - raw_logvar)
        logvar = -10.0 + F.softplus(logvar + 10.0)
        return mean, logvar


class DynamicsEnsemble(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.members = nn.ModuleList([
            GaussianDynamics(args.model_hidden) for _ in range(args.ensemble_size)
        ])
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))
        self.register_buffer("target_mean", torch.zeros(4))
        self.register_buffer("target_std", torch.ones(4))
        self.to(device)
        self.optimizers = [torch.optim.Adam(
            member.parameters(), lr=args.model_lr, weight_decay=1e-5,
        ) for member in self.members]
        self.elites = list(range(args.elite_size))
        self.fitted = False

    @property
    def device(self):
        return self.input_mean.device

    def fit(self, replay, args):
        # Never call replay.sample here: that method intentionally mixes sources.
        n = replay.size()
        indices = np.random.choice(n, min(n, args.max_model_data), replace=False)
        x = np.concatenate((replay.observations[indices, 0], replay.actions[indices, 0]), axis=1)
        y = replay.next_observations[indices, 0, :4] - replay.observations[indices, 0, :4]
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x, y = x[finite], y[finite]
        if len(x) < 32:
            return None
        order = np.random.permutation(len(x))
        n_holdout = max(5, int(0.2 * len(x)))
        valid, train = order[:n_holdout], order[n_holdout:]
        x, y = torch.as_tensor(x, device=self.device), torch.as_tensor(y, device=self.device)
        with torch.no_grad():
            self.input_mean.copy_(x[train].mean(0))
            self.input_std.copy_(x[train].std(0).clamp_min(1e-3))
            self.target_mean.copy_(y[train].mean(0))
            self.target_std.copy_(y[train].std(0).clamp_min(1e-3))
        x = (x - self.input_mean) / self.input_std
        y = (y - self.target_mean) / self.target_std
        validation = []
        epochs_used = []
        for member, optimizer in zip(self.members, self.optimizers):
            # Different bootstrap sample, weights and optimizer for every member.
            bootstrap = np.random.choice(train, len(train), replace=True)
            best_loss, patience = float("inf"), 0
            best_state = best_optimizer = None
            for epoch in range(args.model_epochs):
                member.train()
                for batch in np.array_split(np.random.permutation(bootstrap),
                                            math.ceil(len(bootstrap) / args.model_batch_size)):
                    mean, logvar = member(x[batch])
                    loss = ((mean - y[batch]).square() * torch.exp(-logvar) + logvar).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite dynamics loss; inspect real transitions")
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(member.parameters(), 10.0)
                    optimizer.step()
                member.eval()
                with torch.no_grad():
                    score = float((member(x[valid])[0] - y[valid]).square().mean())
                if not math.isfinite(score):
                    raise FloatingPointError("Non-finite dynamics validation error")
                significant = score < best_loss * 0.99
                if score < best_loss:
                    best_loss = score
                    best_state = copy.deepcopy(member.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                patience = 0 if significant else patience + 1
                if patience >= args.model_patience:
                    break
            member.load_state_dict(best_state)
            optimizer.load_state_dict(best_optimizer)
            validation.append(best_loss)
            epochs_used.append(epoch + 1)
        self.elites = np.argsort(validation)[:args.elite_size].tolist()
        self.fitted = True
        return {
            "train_samples": len(train), "holdout_samples": len(valid),
            "validation_mse": float(np.mean(validation)),
            "elite_validation_mse": float(np.mean(np.asarray(validation)[self.elites])),
            "epochs": max(epochs_used), "elites": json.dumps(self.elites),
        }

    @torch.no_grad()
    def predict_delta(self, obs, actions):
        x = (torch.cat((obs, actions), dim=1) - self.input_mean) / self.input_std
        predictions = [self.members[i](x) for i in self.elites]
        means = torch.stack([p[0] for p in predictions])
        logvars = torch.stack([p[1] for p in predictions])
        member = torch.randint(len(self.elites), (len(obs),), device=self.device)
        row = torch.arange(len(obs), device=self.device)
        delta = means[member, row] + torch.randn_like(obs[:, :4]) * torch.exp(0.5 * logvars[member, row])
        return delta * self.target_std + self.target_mean


def imagined_transition(obs, actions, delta, env):
    """Pure tensor calculation: no simulator calls in imagined transitions."""
    next_obs = obs.clone()
    next_obs[:, :4] += delta
    tau_min, tau_max = env.p.tau_min, env.p.tau_max
    tau = tau_min + (actions[:, 0] + 1.0) * 0.5 * (tau_max - tau_min)
    next_obs[:, 4] = tau / tau_max
    theta = next_obs[:, 2] * math.radians(100.0)
    reward = compute_step_reward(
        (obs[:, 0] - next_obs[:, 0]) * env.goal_x, next_obs[:, 1] * 5.0,
        theta, next_obs[:, 3] * 5.0, tau, obs[:, 4] * tau_max,
        env.ctrl_dt, env.reward_cfg, tau_min, tau_max,
    )
    reached = next_obs[:, 0] <= 0.0
    crashed = is_crashed(theta * (180.0 / math.pi), env.reward_cfg)
    reward += reached * env.reward_cfg.goal_bonus + crashed * env.reward_cfg.crash_penalty
    # Rollout length and environment time limits are NOT true terminals.
    return next_obs, reward, reached | crashed


@torch.no_grad()
def generate_rollouts(model, dynamics, env, args):
    replay = model.replay_buffer
    replay.synthetic.reset()  # Avoid stale transitions from older fitted models.
    indices = np.random.randint(replay.size(), size=args.rollout_batch_size)
    obs = torch.as_tensor(replay.observations[indices, 0], device=model.device)
    added, rejected = 0, 0
    model.policy.set_training_mode(False)
    for _ in range(args.rollout_length):
        actions = model.actor(obs, deterministic=False)
        next_obs, rewards, dones = imagined_transition(
            obs, actions, dynamics.predict_delta(obs, actions), env,
        )
        finite = torch.isfinite(next_obs).all(dim=1) & torch.isfinite(rewards)
        rejected += int((~finite).sum())
        replay.add_synthetic(*[
            value[finite].cpu().numpy() for value in (obs, actions, next_obs, rewards, dones)
        ])
        added += int(finite.sum())
        obs = next_obs[finite & ~dones]
        if not len(obs):
            break
    return added, rejected


def append_csv(path, row):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def task_signature(env):
    return {
        "xml": XML_PATH.name, "xml_sha256": hashlib.sha256(XML_PATH.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(Path(__file__).with_name("obstacle_gym_env.py").read_bytes()).hexdigest(),
        "goal_x": env.goal_x, "ctrl_dt": env.ctrl_dt, "sim_time": env.sim_time,
        "tau_min": env.p.tau_min, "tau_max": env.p.tau_max, "reward": asdict(env.reward_cfg),
    }


def save_checkpoint(model, dynamics, args, env, last_fit):
    """Commit an entire generation before atomically publishing its pointer."""
    name = f"step_{model.num_timesteps:010d}_{time.time_ns()}"
    directory = args.output_dir / "checkpoints" / name
    temporary = directory.with_name("." + name)
    temporary.mkdir(parents=True)
    model.save(temporary / "sac.zip")
    replay = model.replay_buffer
    n = replay.size()
    # Arrays retain ring-buffer order; pos/full are saved explicitly.
    np.savez_compressed(temporary / "real_replay.npz", **{
        key: getattr(replay, key)[:n] for key in (
            "observations", "next_observations", "actions", "rewards", "dones", "timeouts",
        )
    }, pos=replay.pos, full=replay.full, capacity=replay.buffer_size)
    torch.save({
        "state_dict": dynamics.state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in dynamics.optimizers],
        "elites": dynamics.elites, "fitted": dynamics.fitted, "last_fit": last_fit,
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
        "action_rng": env.action_space.np_random.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, temporary / "dynamics.pt")
    metadata = {
        "format_version": 1, "algorithm": "MBPO", "real_timesteps": model.num_timesteps,
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
        "task": task_signature(env), "synthetic_replay": "regenerated on resume",
    }
    (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary, directory)
    pointer = args.output_dir / ".latest.json.tmp"
    pointer.write_text(json.dumps({"checkpoint": str(directory.relative_to(args.output_dir))}) + "\n")
    os.replace(pointer, args.output_dir / "latest.json")
    print(f"[mbpo] checkpoint: {directory}", flush=True)
    return directory


def resolve_checkpoint(path):
    if (path / "metadata.json").is_file():
        return path
    pointer = path / "latest.json"
    if not pointer.is_file():
        raise FileNotFoundError(f"No MBPO checkpoint found in {path}")
    return path / json.loads(pointer.read_text())["checkpoint"]


def restore_replay(model, directory):
    replay = model.replay_buffer
    with np.load(directory / "real_replay.npz", allow_pickle=False) as data:
        if replay.buffer_size != int(data["capacity"]):
            raise ValueError("Replay buffer capacity differs from saved checkpoint")
        for key in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
            getattr(replay, key)[:len(data[key])] = data[key]
        replay.pos, replay.full = int(data["pos"]), bool(data["full"])


class MBPOCallback(BaseCallback):
    def __init__(self, dynamics, env, args, last_fit=0):
        super().__init__()
        self.dynamics, self.env, self.args = dynamics, env, args
        self.last_fit = last_fit
        self.last_checkpoint = 0

    def _on_training_start(self):
        self.last_checkpoint = self.model.num_timesteps
        if self.dynamics.fitted and self.model.replay_buffer.size():
            generate_rollouts(self.model, self.dynamics, self.env, self.args)

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                ep = info["episode"]
                row = {"real_timesteps": self.num_timesteps, "return": ep["r"],
                       "steps": ep["l"], **{key: info[key] for key in
                        ("x", "reached", "crashed", "sim_time", "effort_rms")}}
                append_csv(self.args.output_dir / "episodes.csv", row)
        return True

    def _on_rollout_end(self):
        # SB3 invokes this AFTER storing the real transition, before SAC.train.
        ready = self.model.replay_buffer.size() >= max(32, self.args.learning_starts)
        if ready and self.num_timesteps - self.last_fit >= self.args.model_train_freq:
            start = time.monotonic()
            metrics = self.dynamics.fit(self.model.replay_buffer, self.args)
            if metrics is not None:
                added, rejected = generate_rollouts(self.model, self.dynamics, self.env, self.args)
                self.last_fit = self.num_timesteps
                row = {"real_timesteps": self.num_timesteps, **metrics,
                       "synthetic_samples": added, "rejected_nonfinite": rejected,
                       "fit_and_rollout_seconds": time.monotonic() - start}
                append_csv(self.args.output_dir / "model_fits.csv", row)
                print(f"[mbpo] real={self.num_timesteps} holdout_mse={metrics['elite_validation_mse']:.5g} "
                      f"synthetic={added} fit_seconds={row['fit_and_rollout_seconds']:.2f}", flush=True)
        if self.num_timesteps - self.last_checkpoint >= self.args.checkpoint_freq:
            save_checkpoint(self.model, self.dynamics, self.args, self.env, self.last_fit)
            self.last_checkpoint = self.num_timesteps


def evaluate(model, args):
    env = ObstacleWheelieEnv(render_mode="human" if args.render_eval else None)
    render_period = env.ctrl_dt / args.render_speed if args.render_eval else None
    successes = 0
    try:
        for episode in range(args.eval_episodes):
            obs, _ = env.reset(seed=args.seed + 10000 + episode)
            total, steps = 0.0, 0
            while True:
                step_started = time.perf_counter() if render_period is not None else None
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                steps += 1
                if render_period is not None:
                    time.sleep(max(0.0, render_period - (time.perf_counter() - step_started)))
                if terminated or truncated:
                    break
            successes += int(info["reached"])
            row = {"real_timesteps": model.num_timesteps, "seed": args.seed + 10000 + episode,
                   "episode": episode, "return": total, "steps": steps,
                   **{key: info[key] for key in ("x", "reached", "crashed", "sim_time", "effort_rms")}}
            append_csv(args.output_dir / "evaluation.csv", row)
            print(f"[mbpo eval {episode}] return={total:.3f} x={info['x']:.3f} "
                  f"reached={info['reached']} crashed={info['crashed']} steps={steps}", flush=True)
    finally:
        env.close()
    if args.eval_episodes:
        print(f"[mbpo eval] reached goal in {successes}/{args.eval_episodes} episodes")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "mbpo")
    parser.add_argument("--resume", nargs="?", const="auto", help="Resume latest checkpoint, or one in PATH")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--render-speed", type=float, default=1.0,
                        help="rendered evaluation speed: 1=real time, 0.5=half speed, 0.25=quarter speed")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    defaults = {
        "total-timesteps": 3000000, "learning-starts": 2000, "buffer-size": 3000000,
        "batch-size": 256, "gradient-steps": 20, "model-train-freq": 250,
        "model-epochs": 20, "model-patience": 5, "model-batch-size": 256,
        "max-model-data": 50000, "model-hidden": 200, "ensemble-size": 5,
        "elite-size": 3, "rollout-batch-size": 10000, "rollout-length": 1,
        "checkpoint-freq": 25000, "eval-episodes": 10, "threads": 1, "seed": 0,
    }
    for name, default in defaults.items():
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--real-ratio", type=float, default=0.1)
    parser.add_argument("--model-lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not math.isfinite(args.render_speed) or args.render_speed <= 0:
        parser.error("--render-speed must be finite and positive")
    for name in defaults:
        value = getattr(args, name.replace("-", "_"))
        minimum = 0 if name in ("total-timesteps", "learning-starts", "eval-episodes", "seed") else 1
        if value < minimum:
            parser.error(f"--{name} must be >= {minimum}")
    if not 0 < args.real_ratio <= 1 or not 1 <= args.rollout_length <= 5:
        parser.error("Require 0 < --real-ratio <= 1 and 1 <= --rollout-length <= 5")
    if args.elite_size > args.ensemble_size or args.max_model_data < 32 or args.model_lr <= 0:
        parser.error("Require elites <= ensemble size, max-model-data >= 32, and positive model-lr")
    if args.buffer_size < max(32, args.learning_starts):
        parser.error("--buffer-size must be >= max(32, --learning-starts)")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if args.resume or args.eval_only:
        source = args.output_dir if args.resume in (None, "auto") else Path(args.resume).expanduser().resolve()
        checkpoint = resolve_checkpoint(source)
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        # Retain training settings on resume; runtime/output/evaluation controls may change.
        runtime = {key: getattr(args, key) for key in (
            "total_timesteps", "output_dir", "resume", "eval_only", "render_eval", "render_speed",
            "eval_episodes", "device", "threads",
        )}
        vars(args).update(metadata["configuration"])
        vars(args).update(runtime)
        print(f"[mbpo] loading {checkpoint}; saved training settings retained", flush=True)
    elif (args.output_dir / "latest.json").exists():
        raise FileExistsError("Output already contains a run; use --resume or a new --output-dir")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = ObstacleWheelieEnv()
    try:
        if checkpoint and metadata["task"] != task_signature(env):
            raise ValueError("Task/XML/reward differs from checkpoint; use a separate training run")
        monitor = Monitor(env, info_keywords=("x", "reached", "crashed", "sim_time", "effort_rms"))
        if checkpoint:
            model = SAC.load(checkpoint / "sac.zip", env=monitor, device=args.device,
                             custom_objects={"replay_buffer_class": MixedReplayBuffer})
        else:
            model = SAC(
                "MlpPolicy", monitor, learning_rate=3e-4, buffer_size=args.buffer_size,
                learning_starts=args.learning_starts, batch_size=args.batch_size,
                tau=0.005, gamma=0.99, train_freq=1, gradient_steps=args.gradient_steps,
                policy_kwargs={"net_arch": [256, 256]}, seed=args.seed, device=args.device,
                replay_buffer_class=MixedReplayBuffer, replay_buffer_kwargs={
                    "real_ratio": args.real_ratio,
                    "model_buffer_size": args.rollout_batch_size * args.rollout_length,
                }, verbose=1,
            )
        if args.eval_only:
            evaluate(model, args)
            return
        dynamics = DynamicsEnsemble(args, model.device)
        last_fit = 0
        if checkpoint:
            restore_replay(model, checkpoint)
            saved = torch.load(checkpoint / "dynamics.pt", map_location=model.device, weights_only=False)
            dynamics.load_state_dict(saved["state_dict"])
            for optimizer, state in zip(dynamics.optimizers, saved["optimizers"]):
                optimizer.load_state_dict(state)
            dynamics.elites, dynamics.fitted = saved["elites"], saved["fitted"]
            last_fit = saved["last_fit"]
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if "action_rng" in saved:
                env.action_space.np_random.bit_generator.state = saved["action_rng"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            if saved["cuda_rng"] is not None and torch.cuda.is_available():
                # Restore matching GPUs only, allowing a different machine/device count.
                for index, state in enumerate(saved["cuda_rng"][:torch.cuda.device_count()]):
                    torch.cuda.set_rng_state(state.cpu(), device=index)
        callback = MBPOCallback(dynamics, env, args, last_fit)
        if args.total_timesteps:
            model.learn(total_timesteps=args.total_timesteps, callback=callback,
                        reset_num_timesteps=checkpoint is None, progress_bar=False)
        save_checkpoint(model, dynamics, args, env, callback.last_fit)
        evaluate(model, args)
    finally:
        env.close()


if __name__ == "__main__":
    main()
