#!/usr/bin/env python3
"""PETS (Probabilistic Ensembles with Trajectory Sampling, Chua et al. 2018,
"Deep RL in a Handful of Trials using Probabilistic Dynamics Models") on the
obstacle wheelie-climb task, for comparison against the MPPI+CBF / NMPC
controllers and the model-free SAC/PPO baselines in this folder.

Unlike SAC/PPO (which learn a policy), PETS is model-based: it fits a bootstrap
ensemble of probabilistic neural nets s_{t+1} ~ N(f_b(s_t, a_t)) to REAL
transitions collected from the MuJoCo plant, then at every control step plans
a short action sequence with CEM by rolling PARTICLES out through the learned
ensemble (never the real simulator) and picks the first action of the best
sequence -- i.e. it is itself an MPC controller, just one whose dynamics model
is learned online instead of hand-derived (contrast with mppi/mppi_rls_gp.py,
whose dynamics come from RLS + a GP residual on an analytic nominal model).

Trajectory sampling variant: TS-inf (each imagined particle is assigned ONE
ensemble member for its entire rollout, resampled fresh every planning call).
The PETS paper found this performs on par with TS1 (resampling every imagined
step) at a fraction of the compute, which is what makes CEM-over-an-ensemble
run in real time here.

Usage:
    python3 obstacle_mujoco_PETS.py            # train (headless) then render eval episodes
Toggle TRAIN / RENDER / LOAD_MODEL below to resume, or to just watch a saved ensemble.
"""

import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from obstacle_gym_env import (
    ObstacleWheelieEnv, RewardConfig, RESULTS_DIR,
    compute_step_reward, is_crashed,
)

STATE_DIM = 4     # [x, v, theta, omega]
ACTION_DIM = 1    # [tau]

# ============================================================
# Settings
# ============================================================
TRAIN = True
LOAD_MODEL = False
RENDER_EVAL = True

DEVICE = torch.device("cuda")
SEED = 0

WARMUP_EPISODES = 3       # random-action episodes to seed the replay buffer
N_EPISODES = 1000           # total episodes (including warmup)
EVAL_EPISODES = 3

ENSEMBLE_SIZE = 5
HIDDEN = 200
N_HIDDEN_LAYERS = 3
TRAIN_EPOCHS_PER_EPISODE = 100
BATCH_SIZE = 256
LR = 1e-3

PLAN_HORIZON = 25
N_CANDIDATES = 400
N_ELITES = 40
N_CEM_ITERS = 5
N_PARTICLES = 20          # imagined particles per candidate sequence

# Wall-clock note: measured on this machine, planner.plan() costs ~115-170ms per call
# regardless of whether (horizon, iters, candidates, particles) are tiny or at the
# literature-typical values above -- the batch (candidates*particles) is nearly free on
# a GPU, and the cost is dominated by a fixed per-call latency (consistent with the GPU
# clocking down between calls -- each real control step has an ~1.5ms MuJoCo gap before
# the next plan() call -- and paying a wake-up/reclock cost on the next CUDA launch; a
# tight loop of back-to-back plan() calls with no such gap runs 10-30x faster). So these
# defaults are kept at the literature-appropriate values rather than watered down: ~48s
# per 400-step (20s sim) episode once the planner is active, i.e. roughly 30 minutes for
# the full N_EPISODES run below. If this GPU-wake latency matters to you and you have
# root, `sudo nvidia-smi -pm 1` (persistence mode) typically removes it; that's a
# system-wide setting outside this script's scope, so it isn't done here.
EPISODE_LOG_PATH = RESULTS_DIR / "pets_episodes.csv"   # read by results/plot_results.py

MODEL_PATH = RESULTS_DIR / "pets_ensemble.pt"
BUFFER_PATH = RESULTS_DIR / "pets_buffer.npz"


# ============================================================
# Replay buffer of REAL (s, a, s') transitions
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.s = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.a = np.zeros((capacity, ACTION_DIM), dtype=np.float32)
        self.s_next = np.zeros((capacity, STATE_DIM), dtype=np.float32)
        self.size = 0
        self._ptr = 0

    def add(self, s, a, s_next):
        i = self._ptr
        self.s[i] = s
        self.a[i] = a
        self.s_next[i] = s_next
        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def as_tensors(self, device):
        s = torch.as_tensor(self.s[: self.size], device=device)
        a = torch.as_tensor(self.a[: self.size], device=device)
        s_next = torch.as_tensor(self.s_next[: self.size], device=device)
        return s, a, s_next

    def save(self, path):
        np.savez(path, s=self.s[: self.size], a=self.a[: self.size], s_next=self.s_next[: self.size])

    def load(self, path):
        data = np.load(path)
        n = data["s"].shape[0]
        self.s[:n], self.a[:n], self.s_next[:n] = data["s"], data["a"], data["s_next"]
        self.size = n
        self._ptr = n % self.capacity


# ============================================================
# Probabilistic ensemble dynamics model
# ============================================================

class EnsembleLinear(nn.Module):
    """A linear layer applied independently to `ensemble_size` sets of weights
    via ONE batched matmul, instead of a Python loop over separate nn.Linear
    layers. The planner calls the ensemble at every real control step (it's an
    MPC), so collapsing `ensemble_size` sequential kernel launches into a
    single torch.bmm is what keeps CEM-over-an-ensemble fast enough to replan
    in real time."""

    def __init__(self, ensemble_size, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, out_dim))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x: (ensemble_size, batch, in_dim) -> (ensemble_size, batch, out_dim)
        return torch.bmm(x, self.weight) + self.bias


class ProbabilisticEnsemble(nn.Module):
    """B bootstrap members packed into batched EnsembleLinear layers (one
    forward call evaluates all B at once), each trained on its own bootstrap
    resample of the replay buffer, with the learnable log-variance bounds
    trick from PETS/Kurutach et al. (keeps the Gaussian NLL numerically stable
    early in training). Inputs/outputs are standardized using running
    statistics fit on the current dataset (recomputed each retrain call, as in
    the reference PETS implementation)."""

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM,
                 ensemble_size=ENSEMBLE_SIZE, hidden=HIDDEN, n_layers=N_HIDDEN_LAYERS):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        in_dim = state_dim + action_dim
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            EnsembleLinear(ensemble_size, dims[i], dims[i + 1]) for i in range(n_layers)
        )
        self.head = EnsembleLinear(ensemble_size, hidden, 2 * state_dim)
        self.max_logvar = nn.Parameter(torch.full((ensemble_size, 1, state_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((ensemble_size, 1, state_dim), -10.0))

        self.register_buffer("in_mean", torch.zeros(in_dim))
        self.register_buffer("in_std", torch.ones(in_dim))
        self.register_buffer("out_mean", torch.zeros(state_dim))
        self.register_buffer("out_std", torch.ones(state_dim))

    def fit_normalizer(self, s, a, s_next):
        x = torch.cat([s, a], dim=-1)
        delta = s_next - s
        self.in_mean.copy_(x.mean(0))
        self.in_std.copy_(x.std(0).clamp_min(1e-3))
        self.out_mean.copy_(delta.mean(0))
        self.out_std.copy_(delta.std(0).clamp_min(1e-3))

    def _normalize_in(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return (x - self.in_mean) / self.in_std

    def _forward_ensemble(self, x_ens):
        """x_ens: (B, batch, in_dim), already normalized -> mean_n, logvar_n,
        each (B, batch, state_dim)."""
        h = x_ens
        for layer in self.layers:
            h = F.silu(layer(h))
        mean_n, logvar_raw = self.head(h).chunk(2, dim=-1)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar_raw)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean_n, logvar

    def predict_delta_all(self, s, a):
        """s, a: (batch, dim). Returns mean, var of shape (B, batch, state_dim)
        in REAL (unnormalized) units -- every ensemble member evaluated on the
        SAME batch in one batched op (the planner later gathers, per particle,
        the one member it was assigned for TS-inf)."""
        x = self._normalize_in(s, a)
        x_ens = x.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        mean_n, logvar_n = self._forward_ensemble(x_ens)
        mean = mean_n * self.out_std + self.out_mean
        var = torch.exp(logvar_n) * self.out_std ** 2
        return mean, var

    def loss(self, s, a, s_next):
        """Gaussian NLL summed over ensemble members, each member trained on
        its OWN bootstrap resample of the batch -- all B members' losses
        computed in one batched forward pass."""
        batch = s.shape[0]
        device = s.device
        x = self._normalize_in(s, a)
        target = ((s_next - s) - self.out_mean) / self.out_std

        idx = torch.randint(0, batch, (self.ensemble_size, batch), device=device)
        x_ens = x[idx]            # (B, batch, in_dim)
        tgt_ens = target[idx]     # (B, batch, state_dim)

        mean_n, logvar_n = self._forward_ensemble(x_ens)
        inv_var = torch.exp(-logvar_n)
        nll = 0.5 * (((tgt_ens - mean_n) ** 2) * inv_var + logvar_n).sum(-1).mean()
        # small penalty keeping the learnable logvar bounds from drifting apart
        # without limit (standard PETS regularizer).
        bound_reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll + bound_reg


def train_ensemble(ensemble: ProbabilisticEnsemble, buffer: ReplayBuffer,
                    epochs=TRAIN_EPOCHS_PER_EPISODE, batch_size=BATCH_SIZE, lr=LR):
    s, a, s_next = buffer.as_tensors(DEVICE)
    ensemble.fit_normalizer(s, a, s_next)
    opt = torch.optim.Adam(ensemble.parameters(), lr=lr)
    n = s.shape[0]
    bs = min(batch_size, n)
    last_loss = None
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            if idx.numel() < 2:
                continue
            loss = ensemble.loss(s[idx], a[idx], s_next[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
    return last_loss


# ============================================================
# CEM planner: imagines batched rollouts through the ensemble (TS-inf)
# ============================================================

class CEMPlanner:
    def __init__(self, ensemble, tau_min, tau_max, reward_cfg,
                 horizon=PLAN_HORIZON, n_candidates=N_CANDIDATES, n_elites=N_ELITES,
                 n_iters=N_CEM_ITERS, n_particles=N_PARTICLES, goal_x=None, dt=None):
        self.ensemble = ensemble
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.reward_cfg = reward_cfg
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.n_particles = n_particles
        self.goal_x = goal_x
        self.dt = dt

    @torch.no_grad()
    def plan(self, state0, tau_prev0, init_mean=None):
        H, M, P = self.horizon, self.n_candidates, self.n_particles
        total = M * P
        device = DEVICE

        state0_t = torch.as_tensor(np.asarray(state0, dtype=np.float32), device=device)
        s0 = state0_t.unsqueeze(0).repeat(total, 1)

        # TS-inf: one fixed ensemble member per particle for the whole rollout.
        model_idx = torch.randint(0, self.ensemble.ensemble_size, (total,), device=device)

        span = 0.5 * (self.tau_max - self.tau_min)
        mean = init_mean.clone() if init_mean is not None else torch.zeros(H, device=device)
        std = torch.full((H,), span * 0.5, device=device)

        best_action0 = 0.0
        for it in range(self.n_iters):
            noise = torch.randn(M, H, device=device)
            samples = (mean.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(self.tau_min, self.tau_max)
            actions = samples.repeat_interleave(P, dim=0)   # (total, H)

            states = s0.clone()
            tau_prev = torch.full((total,), float(tau_prev0), device=device)
            returns = torch.zeros(total, device=device)
            alive = torch.ones(total, device=device)

            for h in range(H):
                a_h = actions[:, h]
                mean_d, var_d = self.ensemble.predict_delta_all(states, a_h.unsqueeze(-1))
                mean_d = mean_d.gather(0, model_idx.view(1, -1, 1).expand(1, total, STATE_DIM)).squeeze(0)
                var_d = var_d.gather(0, model_idx.view(1, -1, 1).expand(1, total, STATE_DIM)).squeeze(0)
                delta = mean_d + torch.sqrt(var_d.clamp_min(1e-8)) * torch.randn_like(mean_d)
                next_states = states + delta

                dx = next_states[:, 0] - states[:, 0]
                theta_deg = next_states[:, 2] * (180.0 / math.pi)
                r = compute_step_reward(
                    dx, next_states[:, 1], next_states[:, 2], next_states[:, 3],
                    a_h, tau_prev, self.dt, self.reward_cfg, self.tau_min, self.tau_max,
                )
                crashed = is_crashed(theta_deg, self.reward_cfg).float()
                reached = (next_states[:, 0] >= self.goal_x).float()
                r = r + reached * self.reward_cfg.goal_bonus + crashed * self.reward_cfg.crash_penalty

                returns = returns + r * alive
                alive = alive * (1.0 - crashed) * (1.0 - reached)
                tau_prev = a_h
                states = next_states

            returns = returns.view(M, P).mean(dim=1)
            elite_idx = torch.topk(returns, self.n_elites).indices
            elites = samples[elite_idx]
            mean = elites.mean(0)
            std = elites.std(0).clamp_min(1e-2)
            best_action0 = float(mean[0].item())

        warm_start = torch.cat([mean[1:], mean[-1:]])
        return float(np.clip(best_action0, self.tau_min, self.tau_max)), warm_start


def _to_action_norm(tau, tau_min, tau_max):
    """Map a real torque back to the env's [-1, 1] action convention."""
    return 2.0 * (tau - tau_min) / (tau_max - tau_min) - 1.0


# ============================================================
# Main loop: collect real data, retrain the ensemble, replan every episode.
# ============================================================

def run_episode(env, planner, rng, use_planner: bool):
    obs, info = env.reset()
    state = env.get_state()
    tau_prev = 0.0
    warm_mean = None
    transitions = []
    ep_return = 0.0
    done = truncated = False

    while not (done or truncated):
        if use_planner:
            tau, warm_mean = planner.plan(state, tau_prev, init_mean=warm_mean)
        else:
            tau = float(rng.uniform(env.p.tau_min, env.p.tau_max))

        a_norm = np.array([_to_action_norm(tau, env.p.tau_min, env.p.tau_max)], dtype=np.float32)
        obs, reward, done, truncated, info = env.step(a_norm)
        next_state = env.get_state()

        transitions.append((state.copy(), np.array([tau], dtype=np.float32), next_state.copy()))
        ep_return += reward
        state = next_state
        tau_prev = tau

    return transitions, ep_return, info


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    env = ObstacleWheelieEnv(reward_cfg=RewardConfig())
    reward_cfg = env.reward_cfg
    p = env.p

    ensemble = ProbabilisticEnsemble().to(DEVICE)
    buffer = ReplayBuffer()
    planner = CEMPlanner(
        ensemble, p.tau_min, p.tau_max, reward_cfg,
        goal_x=env.goal_x, dt=env.ctrl_dt,
    )

    if LOAD_MODEL and MODEL_PATH.exists():
        ensemble.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        if BUFFER_PATH.exists():
            buffer.load(BUFFER_PATH)
        print(f"[pets] resumed ensemble from {MODEL_PATH} ({buffer.size} buffered transitions)")

    if TRAIN:
        episode_rows = []
        for ep in range(N_EPISODES):
            use_planner = (ep >= WARMUP_EPISODES) and buffer.size > 0
            t0 = time.time()
            transitions, ep_return, info = run_episode(env, planner, rng, use_planner)
            for s, a, s_next in transitions:
                buffer.add(s, a, s_next)

            loss = None
            if buffer.size >= BATCH_SIZE:
                loss = train_ensemble(ensemble, buffer)

            mode = "planner" if use_planner else "random-warmup"
            wall = time.time() - t0
            taus = np.array([t[1][0] for t in transitions], dtype=np.float64)
            effort_rms = float(np.sqrt(np.mean(taus ** 2))) if taus.size else float("nan")
            sim_time = len(transitions) * env.ctrl_dt
            print(
                f"[episode {ep:02d}] mode={mode:14s} x_final={info['x']:6.2f} | "
                f"reached={info['reached']} | crashed={info['crashed']} | "
                f"return={ep_return:8.2f} | steps={len(transitions):3d} | "
                f"buffer={buffer.size:6d} | ensemble_loss={loss} | "
                f"wall={wall:5.1f}s"
            )

            episode_rows.append({
                "episode": ep,
                "mode": mode,
                "x_final": info["x"],
                "reached": int(info["reached"]),
                "crashed": int(info["crashed"]),
                "return": ep_return,
                "steps": len(transitions),
                "sim_time": sim_time,
                "effort_rms": effort_rms,
                "buffer_size": buffer.size,
                "ensemble_loss": loss,
                "wall_seconds": wall,
            })
            # Overwritten every episode (cheap: N_EPISODES rows) so a killed run still
            # leaves a plottable log, matching the per-episode checkpointing below.
            with open(EPISODE_LOG_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
                writer.writeheader()
                writer.writerows(episode_rows)

            torch.save(ensemble.state_dict(), MODEL_PATH)
            buffer.save(BUFFER_PATH)

        print(f"[pets] saved ensemble to {MODEL_PATH}, buffer to {BUFFER_PATH}, "
              f"episode log to {EPISODE_LOG_PATH}")

    # ---- evaluation with rendering ----
    eval_env = ObstacleWheelieEnv(render_mode="human" if RENDER_EVAL else None, reward_cfg=RewardConfig())
    n_reached = 0
    for ep in range(EVAL_EPISODES):
        transitions, ep_return, info = run_episode(eval_env, planner, rng, use_planner=True)
        n_reached += int(info["reached"])
        print(
            f"[eval ep {ep}] x_final={info['x']:.2f} | reached={info['reached']} | "
            f"crashed={info['crashed']} | return={ep_return:.2f} | steps={len(transitions)}"
        )
    print(f"[eval] reached goal in {n_reached}/{EVAL_EPISODES} episodes")
    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
