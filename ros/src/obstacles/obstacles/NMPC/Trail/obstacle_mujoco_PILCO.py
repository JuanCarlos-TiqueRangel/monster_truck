#!/usr/bin/env python3
"""PILCO-style model-based policy search for the MuJoCo obstacle task.

The controller fits exact independent Gaussian-process dynamics models, then
optimizes an RBF policy through analytic Gaussian moment propagation. Gaussian
belief propagation and the Gaussian approximation before sine squashing remain
modeling approximations; autograd differentiates those moment equations.

The implementation follows Deisenroth, Fox and Rasmussen (2015), "Gaussian
Processes for Data-Efficient Learning in Robotics and Control". It learns all
four discrete state differences with independent exact GPs. A deterministic,
diversity-preserving subset replaces uniform replay sampling; complete episodes
are held out to decide whether the model is accurate enough to improve the
policy. GP inputs and targets are scaled for fitting, with the affine transforms
carried exactly through analytic moment matching.

Kernel-product moments are evaluated in a form without exponential overflow.
Covariances use detached diagonal jitter only when needed for numerical
stability. Policy improvement is offline; applying the saved policy requires
only a forward pass, unlike an MPC planner.

Usage:
    python3 obstacle_mujoco_PILCO.py --trials 50
    python3 obstacle_mujoco_PILCO.py --eval-only --render
    python3 obstacle_mujoco_PILCO.py --eval-only --eval-best --render
    python3 obstacle_mujoco_PILCO.py --eval-only --render --render-speed 0.25
    python3 obstacle_mujoco_PILCO.py --resume --trials 10

Rendered evaluation is paced in wall-clock time: --render-speed 1 is real time,
0.5 is half speed and 0.25 is quarter speed. It does not change simulation time
or control frequency; headless episodes run without pacing.

Use --output-dir for a separate experiment or a short smoke test.
The default cost widths emphasize forward progress, with a three-second
prediction horizon and data-scaled initialization. CUDA is the default and the
program fails clearly if it is unavailable; pass --device cpu explicitly for a
CPU run. A short run is useful for checking execution, not convergence.
The latest model is resumable; pilco_best_policy.pt separately retains the
policy with the highest observed real trial return (excluding random warmup).
"""

import argparse
import csv
from pathlib import Path
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from obstacle_gym_env import ObstacleWheelieEnv, RewardConfig, RESULTS_DIR
from obstacle_mujoco_PETS import (
    ReplayBuffer, _to_action_norm,
)

DTYPE = torch.float64   # exact-GP + moment-matching needs double precision (see ExactGP)
DEVICE = torch.device("cuda")
SEED = 0

STATE_DIM = 4       # D in the paper: [x, v, theta, omega]
ACTION_DIM = 1       # F in the paper: [tau]
GP_INPUT_DIM = STATE_DIM + ACTION_DIM   # tilde_x = [x,v,theta,omega,tau], D+F
STATE_NAMES = ("x", "v", "theta", "omega")

# ============================================================
# Settings
# ============================================================
TRAIN = True
LOAD_MODEL = False
RENDER_EVAL = False

WARMUP_TRIALS = 5        # random-action trials to seed the GP dataset
N_TRIALS = 50             # total real trials (PILCO's whole selling point: very few)
EVAL_EPISODES = 3

MAX_GP_POINTS = 300        # deterministic diverse subset for exact-GP tractability
GP_TRAIN_ITERS = 80
GP_LR = 0.01
GP_RESTARTS = 3

# Validation gates policy improvement. The holdout contains complete episodes.
VALIDATION_FRACTION = 0.20
VALIDATION_MIN_POINTS = 32
VALIDATION_MIN_ACTIVE = 16
VALIDATION_MIN_SKILL = 0.0       # strictly beat the zero-difference predictor
ACTIVE_DELTA_THRESHOLD = 0.05   # in target-scaled units
OBSTACLE_APPROACH_X = (0.8, 2.8)

POLICY_N_BASIS = 24       # N in eq (41)
# One trusted model-gradient step is taken between real trials. Repeating many
# steps against a fixed one-step GP validation set caused model exploitation;
# later real trials provide the feedback for further steps.
POLICY_GRAD_STEPS = 1
POLICY_LR = 0.01
PLAN_HORIZON = 60         # three seconds at the default control period

# Cost (44): target state and task-specific per-dimension widths.
SIGMA_X = 8.0
SIGMA_V = 50.0            # weak speed penalty: reaching 10 m requires acceleration
SIGMA_THETA = 3.0         # allow the roughly 90-degree pitch needed to cross
SIGMA_OMEGA = 20.0

INIT_STATE_VAR = 1e-6    # regularizing epsilon for the (near-)Dirac initial belief, see main()

# Wall-clock note: this is the analytic (moment-matching) pipeline, not particle
# sampling -- there is no particle dimension at all, a SINGLE deterministic
# (given the policy parameters) trajectory of Gaussian moments is propagated per
# gradient step. Cost per step is dominated by O(MAX_GP_POINTS^2) Q-matrix
# evaluations (Eq 26), a handful of which are needed per timestep.
EPISODE_LOG_PATH = RESULTS_DIR / "pilco_episodes.csv"   # read by results/plot_results.py

MODEL_PATH = RESULTS_DIR / "pilco_model.pt"
BUFFER_PATH = RESULTS_DIR / "pilco_buffer.npz"


# ============================================================
# Shared moment-matching primitives (Eqs 17-32), used by BOTH the dynamics
# GPs and the policy's own "deterministic GP" -- identical math either way.
# ============================================================

MIN_LENGTHSCALE = 0.05   # floor, native units (rad for theta/omega, m for x, etc.)


def _floored_ell(log_lengthscale):
    # A smooth positive floor prevents degenerate ARD lengthscales.
    return torch.sqrt(torch.exp(2.0 * log_lengthscale) + MIN_LENGTHSCALE ** 2)


def _q_and_nu(X_train, Lambda, sf2, mu, Sigma):
    """Expected RBF kernels for Gaussian inputs, evaluated with Cholesky solves."""
    nu = X_train - mu
    S = Lambda + 0.5 * (Sigma + Sigma.T)
    L = torch.linalg.cholesky(S)
    whitened = torch.linalg.solve_triangular(L, nu.T, upper=False)
    log_factor = (0.5 * torch.log(torch.diagonal(Lambda)).sum()
                  - torch.log(torch.diagonal(L)).sum())
    q = sf2 * torch.exp(log_factor - 0.5 * whitened.square().sum(0))
    return q, nu


def _kernel_to_point(X, mu, Lambda, sf2):
    """RBF kernels at a deterministic input, used when applying the policy."""
    d = X - mu
    Minv = torch.linalg.inv(Lambda)
    quad = torch.einsum('ni,ij,nj->n', d, Minv, d)
    return sf2 * torch.exp(-0.5 * quad)


def _Q_matrix(nu_a, nu_b, Lambda_a, Lambda_b, sf2_a, sf2_b, Sigma):
    """Expected product of two ARD RBF kernels under a Gaussian input.

    Completing the square first gives only nonpositive quadratic exponents.
    Evaluating point kernels times an exponentially large correction instead
    causes underflow/overflow; clipping that correction changes the moments.
    Lambda_a and Lambda_b are diagonal lengthscale-squared matrices.
    """
    la = torch.diagonal(Lambda_a)
    lb = torch.diagonal(Lambda_b)
    summed = la + lb
    h = 1.0 / (1.0 / la + 1.0 / lb)
    center = (nu_a[:, None, :] * (lb / summed)
              + nu_b[None, :, :] * (la / summed))
    separation = ((nu_a[:, None, :] - nu_b[None, :, :]).square() / summed).sum(-1)
    S = torch.diag(h) + 0.5 * (Sigma + Sigma.T)
    L = torch.linalg.cholesky(S)
    flat_center = center.reshape(-1, center.shape[-1])
    whitened = torch.linalg.solve_triangular(L, flat_center.T, upper=False)
    center_quad = whitened.square().sum(0).reshape(center.shape[:-1])
    log_factor = 0.5 * torch.log(h).sum() - torch.log(torch.diagonal(L)).sum()
    return sf2_a * sf2_b * torch.exp(log_factor - 0.5 * (separation + center_quad))


def _cross_cov_input_output(nu_a, q_a, beta_a, Sigma, Lambda_a):
    """Covariance between the uncertain GP input and one output dimension."""
    S = Lambda_a + 0.5 * (Sigma + Sigma.T)
    L = torch.linalg.cholesky(S)
    weighted = ((beta_a * q_a).unsqueeze(-1) * nu_a).sum(0)
    solved = torch.cholesky_solve(weighted.unsqueeze(-1), L).squeeze(-1)
    return Sigma @ solved


# ============================================================
# Exact GP dynamics model (one per state-difference output dimension)
# ============================================================

class ExactGP(nn.Module):
    """Normalized single-output exact GP with physical-unit prediction methods.

    The stored training inputs/targets and all kernel hyperparameters live in
    normalized coordinates. ``predict_point`` and ``gp_moment_match`` expose
    physical units, so the rest of PILCO never mixes normalized and raw state
    covariances. Targets are scaled without subtracting their mean: this keeps
    PILCO's zero prior over state differences outside observed data.
    """

    def __init__(self, in_dim):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.zeros(in_dim, dtype=DTYPE))
        self.log_signal_var = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.log_noise_var = nn.Parameter(torch.full((), -2.0, dtype=DTYPE))
        self.register_buffer("X_train", torch.zeros(0, in_dim, dtype=DTYPE))
        self.register_buffer("y_train", torch.zeros(0, dtype=DTYPE))
        self.register_buffer("L", torch.zeros(0, 0, dtype=DTYPE))
        self.register_buffer("alpha", torch.zeros(0, dtype=DTYPE))
        self.register_buffer("Kinv", torch.zeros(0, 0, dtype=DTYPE))
        self.register_buffer("input_mean", torch.zeros(in_dim, dtype=DTYPE))
        self.register_buffer("input_scale", torch.ones(in_dim, dtype=DTYPE))
        self.register_buffer("target_scale", torch.ones((), dtype=DTYPE))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Fitted data buffers have variable size, unlike their empty initial values.
        # Older checkpoints do not contain the derived inverse cache or
        # normalization buffers. Identity scaling preserves their predictions.
        if prefix + "Kinv" not in state_dict and prefix + "L" in state_dict:
            saved_L = state_dict[prefix + "L"]
            state_dict[prefix + "Kinv"] = (torch.cholesky_inverse(saved_L)
                                            if saved_L.numel() else saved_L.clone())
        for name, default in (
            ("input_mean", self.input_mean),
            ("input_scale", self.input_scale),
            ("target_scale", self.target_scale),
        ):
            state_dict.setdefault(prefix + name, default.clone())
        for name in ("X_train", "y_train", "L", "alpha", "Kinv",
                     "input_mean", "input_scale", "target_scale"):
            key = prefix + name
            if key in state_dict:
                current = self._buffers[name]
                self._buffers[name] = torch.empty(state_dict[key].shape,
                                                   dtype=current.dtype, device=current.device)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def Lambda(self):
        return torch.diag(_floored_ell(self.log_lengthscale) ** 2)

    def sf2(self):
        return torch.exp(self.log_signal_var)

    def _noise_var(self):
        # A positive noise floor regularizes nearly repeated trajectory samples.
        return torch.exp(self.log_noise_var) + 1e-5

    def _kernel(self, X1, X2):
        ell = _floored_ell(self.log_lengthscale)
        d = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
        return self.sf2() * torch.exp(-0.5 * d.square().sum(-1))

    def _neg_log_marginal_likelihood(self, X, y):
        n = X.shape[0]
        K = self._kernel(X, X) + self._noise_var() * torch.eye(n, dtype=DTYPE, device=X.device)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        return (0.5 * y @ alpha + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * math.log(2 * math.pi))

    @staticmethod
    def input_statistics(X):
        mean = X.mean(0)
        scale = X.std(0, unbiased=False)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        return mean, scale

    @staticmethod
    def target_statistics(y):
        # Scaling about physical zero preserves the m(tilde_x)=0 prior for
        # differences used by PILCO outside the data support.
        rms = y.square().mean().sqrt()
        return rms.clamp_min(1e-6)

    def _posterior_from_normalized(self, X, y):
        n = X.shape[0]
        K = self._kernel(X, X) + self._noise_var() * torch.eye(
            n, dtype=DTYPE, device=X.device)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        inverse = torch.cholesky_inverse(L)
        loss = (0.5 * y @ alpha + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * math.log(2 * math.pi))
        return loss, L, alpha, inverse

    def fit(self, X, y, *, input_mean=None, input_scale=None,
            iters=GP_TRAIN_ITERS, lr=GP_LR, restarts=GP_RESTARTS):
        X = X.to(dtype=DTYPE, device=self.log_lengthscale.device)
        y = y.to(dtype=DTYPE, device=X.device)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
            raise ValueError("GP fitting requires a nonempty input matrix and matching targets")
        if not torch.isfinite(X).all() or not torch.isfinite(y).all():
            raise ValueError("GP training data contains nonfinite values")
        if iters < 1:
            raise ValueError("GP fitting iterations must be positive")
        if restarts < 1:
            raise ValueError("GP fitting restarts must be positive")
        if input_mean is None or input_scale is None:
            input_mean, input_scale = self.input_statistics(X)
        input_mean = input_mean.to(dtype=DTYPE, device=X.device)
        input_scale = input_scale.to(dtype=DTYPE, device=X.device)
        if input_mean.shape != (X.shape[1],) or input_scale.shape != (X.shape[1],):
            raise ValueError("GP input normalization has the wrong shape")
        if not torch.isfinite(input_mean).all() or not torch.isfinite(input_scale).all() or \
                torch.any(input_scale <= 0):
            raise ValueError("GP input normalization must be finite and positive")

        target_scale = self.target_statistics(y)
        X_normalized = (X - input_mean) / input_scale
        y_normalized = y / target_scale
        old_parameters = tuple(parameter.detach().clone() for parameter in self.parameters())
        best_parameters = None
        best_loss = float("inf")
        # Fixed, data-scale-independent initializations make restarts exactly
        # reproducible under the saved torch RNG state.
        initial_lengthscales = (0.5, 1.0, 2.0)
        for restart in range(restarts):
            try:
                family = restart % len(initial_lengthscales)
                cycle = restart // len(initial_lengthscales)
                base = initial_lengthscales[family] * (1.4 ** cycle)
                offset = torch.linspace(-0.12, 0.12, X.shape[1], dtype=DTYPE,
                                        device=X.device)
                offset = offset * (family - 1) * (1.0 + 0.25 * cycle)
                with torch.no_grad():
                    self.log_lengthscale.copy_(math.log(base) + offset)
                    self.log_signal_var.zero_()
                    self.log_noise_var.fill_(math.log(1e-2))
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
                for _ in range(iters):
                    optimizer.zero_grad(set_to_none=True)
                    loss = self._neg_log_marginal_likelihood(X_normalized, y_normalized)
                    if not torch.isfinite(loss):
                        break
                    value = float(loss.detach().item())
                    if value < best_loss:
                        best_loss = value
                        best_parameters = tuple(
                            parameter.detach().clone() for parameter in self.parameters())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), 10.0, error_if_nonfinite=True)
                    optimizer.step()
                with torch.no_grad():
                    final = self._neg_log_marginal_likelihood(X_normalized, y_normalized)
                    if torch.isfinite(final) and float(final.item()) < best_loss:
                        best_loss = float(final.item())
                        best_parameters = tuple(
                            parameter.detach().clone() for parameter in self.parameters())
            except (RuntimeError, FloatingPointError):
                # A failed restart is harmless if another one yielded a finite fit.
                continue
        if best_parameters is None:
            with torch.no_grad():
                for parameter, previous in zip(self.parameters(), old_parameters):
                    parameter.copy_(previous)
            raise FloatingPointError("Every GP hyperparameter restart failed")

        with torch.no_grad():
            for parameter, best in zip(self.parameters(), best_parameters):
                parameter.copy_(best)
            final_loss, L, alpha, inverse = self._posterior_from_normalized(
                X_normalized, y_normalized)
            if not torch.isfinite(final_loss) or not torch.isfinite(inverse).all():
                raise FloatingPointError("GP fitting produced a nonfinite posterior")
        self.input_mean = input_mean.detach().clone()
        self.input_scale = input_scale.detach().clone()
        self.target_scale = target_scale.detach().clone()
        self.X_train = X_normalized.detach().clone()
        self.y_train = y_normalized.detach().clone()
        self.L, self.alpha, self.Kinv = L, alpha, inverse
        return float(final_loss.item())

    def predict_point(self, x_star):
        """Posterior latent-function mean and variance in physical units."""
        if self.X_train.shape[0] == 0:
            raise RuntimeError("Fit or load the GP before predicting")
        x_star = x_star.to(dtype=DTYPE, device=self.X_train.device)
        vector_input = x_star.ndim == 1
        if vector_input:
            x_star = x_star.unsqueeze(0)
        x_normalized = (x_star - self.input_mean) / self.input_scale
        k_star = self._kernel(x_normalized, self.X_train)
        mean_normalized = k_star @ self.alpha
        v = torch.linalg.solve_triangular(self.L, k_star.T, upper=False)
        var_normalized = (self.sf2() - v.square().sum(0)).clamp_min(0.0)
        mean = self.target_scale * mean_normalized
        var = self.target_scale.square() * var_normalized
        if vector_input:
            return mean.squeeze(0), var.squeeze(0)
        return mean, var


def gp_moment_match(gps, mu_in, Sigma_in):
    """Physical GP moments for a Gaussian physical-unit input distribution."""
    if any(gp.X_train.shape[0] == 0 for gp in gps):
        raise RuntimeError("Fit or load every GP before moment matching")
    reference_mean = gps[0].input_mean
    reference_scale = gps[0].input_scale
    if any(not torch.allclose(gp.input_mean, reference_mean) or
           not torch.allclose(gp.input_scale, reference_scale) for gp in gps[1:]):
        raise ValueError("All dynamics GPs must share the same input normalization")

    mu_normalized = (mu_in - reference_mean) / reference_scale
    Sigma_normalized = (Sigma_in / reference_scale[:, None]
                        / reference_scale[None, :])
    E = len(gps)
    lambdas = [gp.Lambda() for gp in gps]
    signals = [gp.sf2() for gp in gps]
    moments = [_q_and_nu(gp.X_train, lam, signal, mu_normalized, Sigma_normalized)
               for gp, lam, signal in zip(gps, lambdas, signals)]
    qs, nus = zip(*moments)
    mean_normalized = torch.stack([gp.alpha @ q for gp, q in zip(gps, qs)])
    entries = {}
    for a in range(E):
        for b in range(a, E):
            Qab = _Q_matrix(nus[a], nus[b], lambdas[a], lambdas[b],
                            signals[a], signals[b], Sigma_normalized)
            val = (gps[a].alpha @ Qab @ gps[b].alpha
                   - mean_normalized[a] * mean_normalized[b])
            if a == b:
                # The inverse is fixed between dynamics refits. The Frobenius
                # contraction evaluates trace(Kinv @ Q) without a cubic matmul.
                Kinv = gps[a].Kinv
                val = val + signals[a] - (Kinv * Qab.T).sum() + gps[a]._noise_var()
                val = val.clamp_min(0.0)
            entries[a, b] = val
            entries[b, a] = val
    Sigma_delta = torch.stack([torch.stack([entries[a, b] for b in range(E)])
                               for a in range(E)])
    cross_normalized = torch.stack([
        _cross_cov_input_output(
            nus[a], qs[a], gps[a].alpha, Sigma_normalized, lambdas[a])
        for a in range(E)
    ], dim=1)
    target_scales = torch.stack([gp.target_scale for gp in gps])
    mu_delta = target_scales * mean_normalized
    Sigma_delta = (target_scales[:, None] * Sigma_delta
                   * target_scales[None, :])
    cross_cov = (reference_scale[:, None] * cross_normalized
                 * target_scales[None, :])
    return mu_delta, Sigma_delta, cross_cov


class PILCOReplayBuffer(ReplayBuffer):
    """Real replay with episode identities for leakage-free validation."""

    def __init__(self, capacity=200_000):
        super().__init__(capacity=capacity)
        self.episode_id = np.full(capacity, -1, dtype=np.int64)

    def add(self, s, a, s_next, episode_id=-1):
        index = self._ptr
        super().add(s, a, s_next)
        self.episode_id[index] = int(episode_id)

    def save(self, path):
        # A file object prevents numpy from appending ".npz" to staging names.
        with Path(path).open("wb") as stream:
            np.savez(
                stream,
                s=self.s[:self.size],
                a=self.a[:self.size],
                s_next=self.s_next[:self.size],
                episode_id=self.episode_id[:self.size],
            )

    def load(self, path):
        data = np.load(path)
        n = int(data["s"].shape[0])
        if n > self.capacity:
            raise ValueError(f"Replay contains {n} rows, capacity is {self.capacity}")
        self.s[:n] = data["s"]
        self.a[:n] = data["a"]
        self.s_next[:n] = data["s_next"]
        self.episode_id[:n] = data["episode_id"] if "episode_id" in data else -1
        self.size = n
        self._ptr = n % self.capacity

    def truncate(self, size):
        """Discard an uncommitted suffix recovered after an interrupted save."""
        size = int(size)
        if not 0 <= size <= self.size:
            raise ValueError(f"Cannot truncate replay of size {self.size} to {size}")
        old_size = self.size
        self.episode_id[size:old_size] = -1
        self.size = size
        self._ptr = size % self.capacity

    def restore_legacy_episode_ids(self, episode_rows):
        """Reconstruct IDs for old buffers whose CSV contains episode lengths."""
        if self.size == 0 or np.all(self.episode_id[:self.size] >= 0):
            return
        cursor = 0
        for row in episode_rows:
            steps = int(float(row["steps"]))
            stop = cursor + steps
            if stop > self.size:
                raise ValueError("Episode log describes more transitions than the replay")
            self.episode_id[cursor:stop] = int(float(row["episode"]))
            cursor = stop
        if cursor != self.size:
            raise ValueError(
                "Legacy replay cannot be split by complete episode: CSV step counts "
                f"cover {cursor} of {self.size} transitions. Start a fresh output directory."
            )


def _robust_standardize(values):
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = (q75 - q25) / 1.349
    standard = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, standard)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return np.clip((values - center) / scale, -10.0, 10.0)


def _farthest_indices(features, candidates, count, seeds=()):
    """Deterministic greedy maximin selection with stable index tie-breaking."""
    candidates = np.asarray(sorted(set(map(int, candidates))), dtype=np.int64)
    count = min(int(count), len(candidates))
    if count <= 0:
        return []
    available = np.ones(len(candidates), dtype=bool)
    minimum_distance = np.full(len(candidates), np.inf, dtype=np.float64)
    for seed in seeds:
        minimum_distance = np.minimum(
            minimum_distance,
            np.square(features[candidates] - features[int(seed)]).sum(axis=1),
        )
    selected = []
    if not tuple(seeds):
        distance_from_center = np.square(features[candidates]).sum(axis=1)
        position = int(np.argmax(distance_from_center))
    else:
        position = int(np.argmax(minimum_distance))
    for _ in range(count):
        if not available[position]:
            break
        index = int(candidates[position])
        selected.append(index)
        available[position] = False
        distance = np.square(features[candidates] - features[index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        scores = np.where(available, minimum_distance, -np.inf)
        position = int(np.argmax(scores))
    return selected


def select_representative_indices(X, Y, max_points, tau_max):
    """Cover changing, loaded-stall, and quiet dynamics without random redraws."""
    n = len(X)
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    y_scale = np.sqrt(np.mean(np.square(Y), axis=0))
    y_scale = np.where(y_scale > 1e-8, y_scale, 1.0)
    activity = np.max(np.abs(Y) / y_scale, axis=1) > ACTIVE_DELTA_THRESHOLD
    loaded_stall = (~activity) & (np.abs(X[:, -1]) >= 0.25 * abs(tau_max))
    quiet = ~(activity | loaded_stall)
    features = _robust_standardize(np.concatenate([X, Y], axis=1))
    quotas = (
        (np.flatnonzero(activity), round(0.60 * max_points)),
        (np.flatnonzero(loaded_stall), round(0.25 * max_points)),
        (np.flatnonzero(quiet), max_points),
    )
    selected = []
    for candidates, quota in quotas:
        remaining = max_points - len(selected)
        if remaining <= 0:
            break
        selected.extend(_farthest_indices(
            features, candidates, min(quota, remaining), seeds=selected))
    if len(selected) < max_points:
        remaining = np.setdiff1d(
            np.arange(n, dtype=np.int64), np.asarray(selected, dtype=np.int64),
            assume_unique=False)
        selected.extend(_farthest_indices(
            features, remaining, max_points - len(selected), seeds=selected))
    return np.asarray(selected, dtype=np.int64)


def _angular_state_differences(states, next_states):
    differences = next_states - states
    # The IMU angle is represented on [-pi, pi]; use the shortest difference.
    differences[:, 2] = np.arctan2(
        np.sin(differences[:, 2]), np.cos(differences[:, 2]))
    return differences


def _choose_initial_validation_episode(buffer, warmup_trials):
    """Choose a representative whole warmup episode as the initial holdout."""
    summaries, identifiers = [], []
    for episode in range(warmup_trials):
        indices = np.flatnonzero(buffer.episode_id[:buffer.size] == episode)
        if len(indices) == 0:
            return None
        states, actions = buffer.s[indices], buffer.a[indices, 0]
        summaries.append([
            buffer.s_next[indices[-1], 0], states[:, 0].min(), states[:, 0].max(),
            np.mean(np.abs(states[:, 1])), np.mean(np.abs(states[:, 3])),
            np.mean(np.abs(actions)), len(indices),
        ])
        identifiers.append(episode)
    standardized = _robust_standardize(np.asarray(summaries, dtype=np.float64))
    pairwise = np.square(standardized[:, None] - standardized[None, :]).sum(axis=2)
    return identifiers[int(np.argmin(pairwise.sum(axis=1)))]


def update_validation_episode_ids(buffer, current_ids, warmup_trials,
                                  validation_fraction=VALIDATION_FRACTION):
    identifiers = set(map(int, current_ids))
    if not identifiers and warmup_trials > 0:
        medoid = _choose_initial_validation_episode(buffer, warmup_trials)
        if medoid is not None:
            identifiers.add(medoid)
    observed = set(map(int, buffer.episode_id[:buffer.size]))
    for episode in observed:
        if _is_scheduled_validation_episode(
                episode, warmup_trials, validation_fraction):
            identifiers.add(episode)
    return sorted(identifiers)


def _is_scheduled_validation_episode(episode, warmup_trials,
                                     validation_fraction=VALIDATION_FRACTION):
    stride = max(2, round(1.0 / validation_fraction))
    return (episode >= warmup_trials
            and (episode - warmup_trials + 1) % stride == 0)


def _maximum_replay_x(buffer, excluded_episode_ids=()):
    if buffer.size == 0:
        return float("-inf")
    included = ~np.isin(
        buffer.episode_id[:buffer.size], list(excluded_episode_ids))
    if not np.any(included):
        return float("-inf")
    return float(np.max(buffer.s_next[:buffer.size][included, 0]))


def build_gp_datasets(buffer, validation_episode_ids, max_points, tau_max, device):
    n = buffer.size
    if n == 0:
        raise ValueError("Cannot build GP data from an empty replay")
    episode_ids = buffer.episode_id[:n]
    validation_mask = np.isin(episode_ids, validation_episode_ids)
    training_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if len(training_indices) == 0 or len(validation_indices) == 0:
        return None
    all_X = np.concatenate([buffer.s[:n], buffer.a[:n]], axis=1).astype(np.float64)
    all_Y = _angular_state_differences(
        buffer.s[:n].astype(np.float64), buffer.s_next[:n].astype(np.float64))
    local = select_representative_indices(
        all_X[training_indices], all_Y[training_indices], max_points, tau_max)
    chosen = training_indices[local]
    to_tensor = lambda value: torch.as_tensor(value, dtype=DTYPE, device=device)
    return (
        to_tensor(all_X[chosen]), to_tensor(all_Y[chosen]),
        to_tensor(all_X[validation_indices]), to_tensor(all_Y[validation_indices]),
        chosen,
    )


@torch.no_grad()
def validate_dynamics(gps, X, Y, tau_max):
    """Require held-out skill over a physical zero-difference baseline."""
    if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
        return {
            "passed": False,
            "reason": "held-out inputs or targets contain nonfinite values",
            "points": len(X),
            "active_points": 0,
            "overall_skill": float("nan"),
            "active_skill": float("nan"),
            "obstacle_skill": float("nan"),
            "output_skill": [float("nan")] * STATE_DIM,
        }
    means = []
    batch_size = 2048
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        means.append(torch.stack([gp.predict_point(batch)[0] for gp in gps], dim=1))
    prediction = torch.cat(means, dim=0)
    scales = torch.stack([gp.target_scale for gp in gps]).clamp_min(1e-6)
    normalized_error = (prediction - Y) / scales
    normalized_zero_error = Y / scales
    active = normalized_zero_error.abs().amax(dim=1) > ACTIVE_DELTA_THRESHOLD
    obstacle = ((X[:, 0] >= OBSTACLE_APPROACH_X[0])
                & (X[:, 0] <= OBSTACLE_APPROACH_X[1]) & active)
    loaded_stall = ((~active) & (X[:, -1].abs() >= 0.25 * abs(tau_max)))

    def scores(mask):
        model_mse = normalized_error[mask].square().mean()
        zero_mse = normalized_zero_error[mask].square().mean()
        skill = 1.0 - model_mse / zero_mse.clamp_min(1e-12)
        output_model = normalized_error[mask].square().mean(0)
        output_zero = normalized_zero_error[mask].square().mean(0)
        output_skill = 1.0 - output_model / output_zero.clamp_min(1e-12)
        signal = output_zero > 1e-10
        return float(skill.item()), output_skill, signal

    reasons = []
    if len(X) < VALIDATION_MIN_POINTS:
        reasons.append(f"only {len(X)} held-out points")
    if int(active.sum()) < VALIDATION_MIN_ACTIVE:
        reasons.append(f"only {int(active.sum())} active held-out points")
    overall_skill, output_skill, _ = scores(
        torch.ones(len(X), dtype=torch.bool, device=X.device))
    active_skill = float("nan")
    obstacle_skill = float("nan")
    if math.isfinite(overall_skill) and overall_skill <= VALIDATION_MIN_SKILL:
        reasons.append(f"overall skill {overall_skill:.3f} <= {VALIDATION_MIN_SKILL:.3f}")
    if int(active.sum()):
        active_skill, active_output_skill, active_signal = scores(active)
        if active_skill <= VALIDATION_MIN_SKILL:
            reasons.append(f"active skill {active_skill:.3f} <= {VALIDATION_MIN_SKILL:.3f}")
        failed_outputs = torch.nonzero(
            active_signal & (active_output_skill <= VALIDATION_MIN_SKILL),
            as_tuple=False).flatten().tolist()
        if failed_outputs:
            names = ",".join(STATE_NAMES[index]
                             for index in failed_outputs)
            reasons.append(f"active outputs not better than zero: {names}")
    if int(obstacle.sum()) >= VALIDATION_MIN_ACTIVE:
        obstacle_skill, _, _ = scores(obstacle)
        if obstacle_skill <= VALIDATION_MIN_SKILL:
            reasons.append(
                f"obstacle-motion skill {obstacle_skill:.3f} <= {VALIDATION_MIN_SKILL:.3f}")
    if int(loaded_stall.sum()) >= VALIDATION_MIN_ACTIVE:
        model_rmse = normalized_error[loaded_stall].square().mean().sqrt()
        zero_rmse = normalized_zero_error[loaded_stall].square().mean().sqrt()
        if model_rmse > zero_rmse + 0.05:
            reasons.append(
                f"loaded-stall RMSE {model_rmse:.3f} exceeds {zero_rmse:.3f}+0.05")
    if not torch.isfinite(prediction).all():
        reasons.append("nonfinite validation prediction")
    return {
        "passed": not reasons,
        "reason": "; ".join(reasons) if reasons else "passed",
        "points": len(X),
        "active_points": int(active.sum()),
        "overall_skill": overall_skill,
        "active_skill": active_skill,
        "obstacle_skill": obstacle_skill,
        "output_skill": output_skill.detach().cpu().tolist(),
    }


# ============================================================
# Trigonometric moments of the sine-squashing function (Eq 36 + Appendix A)
# ============================================================
# Appendix A only tabulates E[sin(u)], E[cos(u)], E[sin^2(u)], E[cos^2(u)],
# E[sin(u)cos(u)] for u~N(m,s2) -- the single-frequency case. sigma(u) in (36)
# is a SUM of sin(u) and sin(3u), so E[sigma(u)^2] and Cov[u,sigma(u)] need the
# moments of PRODUCTS at different frequencies, which the appendix doesn't
# spell out directly. Product-to-sum identities and Gaussian sine/cosine
# expectations give the moments used below.

def _sin_moment(k, m, s2):
    return torch.exp(-0.5 * k ** 2 * s2) * torch.sin(k * m)


def _cos_moment(k, m, s2):
    return torch.exp(-0.5 * k ** 2 * s2) * torch.cos(k * m)


def _E_sigma(m, s2):
    return 9.0 / 8.0 * _sin_moment(1, m, s2) + 1.0 / 8.0 * _sin_moment(3, m, s2)


def _E_sigma_sq(m, s2):
    # sin(i u) sin(j u) = 0.5*[cos((i-j)u) - cos((i+j)u)]
    def sin_sin(i, j):
        return 0.5 * (_cos_moment(i - j, m, s2) - _cos_moment(i + j, m, s2))
    return ((9.0 / 8.0) ** 2 * sin_sin(1, 1)
             + 2 * (9.0 / 8.0) * (1.0 / 8.0) * sin_sin(1, 3)
             + (1.0 / 8.0) ** 2 * sin_sin(3, 3))


def _E_u_sin_ku(k, m, s2):
    # derived via u = m + s*z, z~N(0,1), using E[z sin(cz)] = c*exp(-c^2/2):
    # E[u sin(ku)] = exp(-k^2 s2/2) * (m*sin(km) + k*s2*cos(km))
    return torch.exp(-0.5 * k ** 2 * s2) * (m * torch.sin(k * m) + k * s2 * torch.cos(k * m))


def _cov_u_sigma(m, s2):
    Eu_sigma = 9.0 / 8.0 * _E_u_sin_ku(1, m, s2) + 1.0 / 8.0 * _E_u_sin_ku(3, m, s2)
    return Eu_sigma - m * _E_sigma(m, s2)


# ============================================================
# Policy: Sec 5.3.2's "deterministic GP" (regularized RBF network), squashed
# through Eq (36)-(37).
# ============================================================

def _kernel_matrix(X, Lambda, sf2):
    ell = torch.diagonal(Lambda).sqrt()
    d = (X.unsqueeze(1) - X.unsqueeze(0)) / ell
    return sf2 * torch.exp(-0.5 * (d ** 2).sum(-1))


class RBFPolicy(nn.Module):
    """Eqs (39)-(43): a degenerate/"deterministic" GP -- fixed signal variance
    1 and noise variance 0.01 (the paper's own stated choice in Sec 5.3.2, not
    learned), sharing the SAME q/Q moment-matching machinery as the dynamics
    GPs above. Only the centers M, shared length-scales, and "virtual targets"
    t are learned; alpha = (K+0.01I)^-1 t exactly as Eq (41)."""

    SF2 = 1.0
    NOISE_VAR = 0.01

    def __init__(self, state_dim, n_basis, tau_max):
        super().__init__()
        self.M = nn.Parameter(torch.randn(n_basis, state_dim, dtype=DTYPE))
        self.log_lengthscale = nn.Parameter(torch.zeros(state_dim, dtype=DTYPE))
        self.t = nn.Parameter(torch.randn(n_basis, dtype=DTYPE) * 0.1)
        self.tau_max = float(tau_max)

    def Lambda(self):
        return torch.diag(_floored_ell(self.log_lengthscale) ** 2)

    def alpha(self):
        n = self.M.shape[0]
        K = (_kernel_matrix(self.M, self.Lambda(), self.SF2)
             + self.NOISE_VAR * torch.eye(n, dtype=DTYPE, device=self.M.device))
        return torch.linalg.solve(K, self.t)

    def forward_point(self, x):
        """Deterministic torque for a single OBSERVED (point, not distributional)
        real state x (D,) -- used only to act on the real system; no moment
        matching needed for a point input."""
        k = _kernel_to_point(self.M, x, self.Lambda(), self.SF2)
        pi_tilde = k @ self.alpha()
        sigma = 9.0 / 8.0 * torch.sin(pi_tilde) + 1.0 / 8.0 * torch.sin(3 * pi_tilde)
        return self.tau_max * sigma

    def moment_match(self, mu_in, Sigma_in):
        """Gaussian moments of the RBF policy followed by sine squashing."""
        alpha = self.alpha()
        Lam = self.Lambda()
        q, nu = _q_and_nu(self.M, Lam, self.SF2, mu_in, Sigma_in)
        mean_pi = alpha @ q
        Q = _Q_matrix(nu, nu, Lam, Lam, self.SF2, self.SF2, Sigma_in)
        # This deterministic RBF policy has no additional GP model variance.
        var_pi = (alpha @ Q @ alpha - mean_pi.square()).clamp_min(0.0)
        cov_x_pi = _cross_cov_input_output(nu, q, alpha, Sigma_in, Lam)
        mean_sigma = _E_sigma(mean_pi, var_pi)
        var_sigma = (_E_sigma_sq(mean_pi, var_pi) - mean_sigma.square()).clamp_min(0.0)
        mean_u = self.tau_max * mean_sigma
        var_u = self.tau_max ** 2 * var_sigma
        # Stein's identity cancels var_pi analytically in cov[x,pi]/var_pi
        # times cov[pi,u], retaining the correct limit at zero variance.
        expected_slope = (9.0 / 8.0 * _cos_moment(1, mean_pi, var_pi)
                          + 3.0 / 8.0 * _cos_moment(3, mean_pi, var_pi))
        cov_x_u = self.tau_max * cov_x_pi * expected_slope
        return mean_u, var_u, cov_x_u


# ============================================================
# Closed-form saturating cost (Eqs 44-47)
# ============================================================

def expected_saturating_cost(mu, Sigma, target, W):
    """Expected saturating quadratic cost, using a symmetric Cholesky solve."""
    D = mu.shape[0]
    root_W = torch.linalg.cholesky(W)
    S = (torch.eye(D, dtype=mu.dtype, device=mu.device)
         + root_W.T @ (0.5 * (Sigma + Sigma.T)) @ root_W)
    L = torch.linalg.cholesky(S)
    difference = root_W.T @ (mu - target)
    whitened = torch.linalg.solve_triangular(L, difference.unsqueeze(-1), upper=False)
    log_expectation = -torch.log(torch.diagonal(L)).sum() - 0.5 * whitened.square().sum()
    return -torch.expm1(log_expectation)


# ============================================================
# Successor-state distribution for full state-difference GPs (Eqs 9-10).
# ============================================================

def _project_to_valid_covariance(Sigma, min_eig=1e-10):
    """Symmetrize a covariance and add only the diagonal jitter it needs.

    The jitter is a detached numerical safeguard, not part of the analytic
    moment derivative. Avoid differentiating through eigenvectors: their
    gradients are undefined at repeated eigenvalues, including isotropic
    initial beliefs. No repair is applied when the covariance is already valid.
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    with torch.no_grad():
        if not torch.isfinite(Sigma).all():
            raise FloatingPointError("PILCO propagated a nonfinite covariance")
        smallest = torch.linalg.eigvalsh(Sigma.detach()).amin()
        jitter = (min_eig - smallest).clamp_min(0.0)
    return Sigma + jitter * torch.eye(Sigma.shape[0], dtype=Sigma.dtype, device=Sigma.device)


def _successor_distribution(mu_t, Sigma_t, cross_cov_state, mu_delta, Sigma_delta):
    mu_next = mu_t + mu_delta
    Sigma_next = Sigma_t + Sigma_delta + cross_cov_state + cross_cov_state.T
    # This is one of the two places a new Sigma is ever produced across the
    # whole T-step recursion (see _project_to_valid_covariance's docstring).
    Sigma_next = _project_to_valid_covariance(Sigma_next)
    return mu_next, Sigma_next


def _build_joint(mu_t, Sigma_t, mean_u, var_u, cov_x_u):
    """Algorithm 2, steps 2-3: the joint Gaussian over tilde_x_t=[x_t,u_t].
    The other of the two places a new Sigma is produced (see
    _project_to_valid_covariance) -- combining Sigma_t with policy-derived
    var_u/cov_x_u can reintroduce invalidity (e.g. cov_x_u inconsistent with
    var_u) even when Sigma_t itself was already valid coming in."""
    D = mu_t.shape[0]
    tilde_mu = torch.cat([mu_t, mean_u.reshape(1)])
    tilde_Sigma = torch.zeros(D + 1, D + 1, dtype=DTYPE, device=mu_t.device)
    tilde_Sigma[:D, :D] = Sigma_t
    tilde_Sigma[:D, D] = cov_x_u
    tilde_Sigma[D, :D] = cov_x_u
    tilde_Sigma[D, D] = var_u
    tilde_Sigma = _project_to_valid_covariance(tilde_Sigma)
    return tilde_mu, tilde_Sigma


# ============================================================
# Policy evaluation (Sec 3.2) + analytic policy improvement (Sec 3.3)
# ============================================================

def rollout_expected_cost(policy, gps, mu0, Sigma0, target, W, horizon):
    """Eq (2): J^pi(theta) = sum_{t=0}^{T} E_xt[c(xt)], propagated analytically
    via Algorithm 2 at every step -- no particles, no sampling: a single
    deterministic (given the current policy parameters) trajectory of Gaussian
    moments, fully differentiable end-to-end via autograd."""
    mu_t, Sigma_t = mu0, Sigma0
    total = expected_saturating_cost(mu_t, Sigma_t, target, W)
    def transition_and_cost(mu, Sigma):
        mean_u, var_u, cov_x_u = policy.moment_match(mu, Sigma)
        tilde_mu, tilde_Sigma = _build_joint(
            mu, Sigma, mean_u, var_u, cov_x_u)
        mu_delta, Sigma_delta, cross_cov_full = gp_moment_match(
            gps, tilde_mu, tilde_Sigma)
        next_mu, next_Sigma = _successor_distribution(
            mu, Sigma, cross_cov_full[:STATE_DIM, :], mu_delta, Sigma_delta)
        return next_mu, next_Sigma, expected_saturating_cost(
            next_mu, next_Sigma, target, W)

    for _ in range(horizon):
        if torch.is_grad_enabled() and any(
                parameter.requires_grad for parameter in policy.parameters()):
            mu_t, Sigma_t, immediate = gradient_checkpoint(
                transition_and_cost, mu_t, Sigma_t, use_reentrant=False)
        else:
            mu_t, Sigma_t, immediate = transition_and_cost(mu_t, Sigma_t)
        total = total + immediate
    return total


def improve_policy(policy, gps, mu0, Sigma0, target, W, horizon,
                   steps=POLICY_GRAD_STEPS, lr=POLICY_LR):
    if steps < 1:
        raise ValueError("Policy improvement steps must be positive")
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    dynamics_parameters = [parameter for gp in gps for parameter in gp.parameters()]
    previous_requires_grad = [parameter.requires_grad for parameter in dynamics_parameters]
    for parameter in dynamics_parameters:
        parameter.requires_grad_(False)
        parameter.grad = None
    best_loss = float("inf")
    best_policy = None
    try:
        for iteration in range(steps):
            opt.zero_grad(set_to_none=True)
            cost = rollout_expected_cost(
                policy, gps, mu0, Sigma0, target, W, horizon)
            if not torch.isfinite(cost):
                raise FloatingPointError(f"Nonfinite policy cost at improvement step {iteration}")
            value = float(cost.detach().item())
            if value < best_loss:
                best_loss = value
                best_policy = {
                    key: tensor.detach().clone()
                    for key, tensor in policy.state_dict().items()
                }
            cost.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0, error_if_nonfinite=True)
            opt.step()
            if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
                raise FloatingPointError(f"Nonfinite policy parameter at improvement step {iteration}")
        with torch.no_grad():
            final_cost = rollout_expected_cost(
                policy, gps, mu0, Sigma0, target, W, horizon)
        if torch.isfinite(final_cost) and float(final_cost.item()) < best_loss:
            best_loss = float(final_cost.item())
            best_policy = {
                key: tensor.detach().clone()
                for key, tensor in policy.state_dict().items()
            }
        if best_policy is None:
            raise FloatingPointError("Policy improvement produced no finite iterate")
        policy.load_state_dict(best_policy)
    finally:
        for parameter, requires_grad in zip(dynamics_parameters, previous_requires_grad):
            parameter.requires_grad_(requires_grad)
    return best_loss


# ============================================================
# Real-episode execution: policy.forward_point() only, no planning at test time.
# ============================================================

def run_policy_episode(env, policy, device=DEVICE, *, render_speed=1.0):
    obs, info = env.reset()
    transitions = []
    ep_return = 0.0
    done = truncated = False
    render_period = env.ctrl_dt / render_speed if env.render_mode == "human" else None
    with torch.no_grad():
        while not (done or truncated):
            step_started = time.perf_counter() if render_period is not None else None
            state = env.get_state()
            x_t = torch.as_tensor(state, dtype=DTYPE, device=device)
            tau = float(policy.forward_point(x_t).item())
            a_norm = np.array(
                [_to_action_norm(tau, env.p.tau_min, env.p.tau_max)], dtype=np.float32
            )
            obs, reward, done, truncated, info = env.step(a_norm)
            next_state = env.get_state()
            transitions.append((state.copy(), np.array([tau], dtype=np.float32), next_state.copy()))
            ep_return += reward
            if render_period is not None:
                time.sleep(max(0.0, render_period - (time.perf_counter() - step_started)))
    return transitions, ep_return, info


def run_exploration_episode(env, rng):
    """Collect a temporally coherent random forward-torque experiment.

    The goal direction is part of the task definition. One torque sampled
    uniformly over the positive action range is held for the episode. Across
    trials this covers stopped, collision, crash, and crossing trajectories;
    unlike 50 ms white noise, it also exposes the GP to sustained acceleration.
    No successful torque value is hardcoded.
    """
    env.reset()
    state = env.get_state()
    transitions = []
    episode_return = 0.0
    done = truncated = False
    tau = float(rng.uniform(0.0, env.p.tau_max))
    action = np.array(
        [_to_action_norm(tau, env.p.tau_min, env.p.tau_max)], dtype=np.float32)
    while not (done or truncated):
        _, reward, done, truncated, info = env.step(action)
        next_state = env.get_state()
        transitions.append(
            (state.copy(), np.array([tau], dtype=np.float32), next_state.copy()))
        episode_return += reward
        state = next_state
    return transitions, episode_return, info


def _inverse_policy_squash(normalized_action):
    """Invert PILCO's monotone sine squash on its principal interval."""
    target = normalized_action.clamp(-1.0, 1.0)
    lower = torch.full_like(target, -math.pi / 2)
    upper = torch.full_like(target, math.pi / 2)
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        squashed = (9.0 / 8.0 * torch.sin(middle)
                    + 1.0 / 8.0 * torch.sin(3.0 * middle))
        lower = torch.where(squashed < target, middle, lower)
        upper = torch.where(squashed >= target, middle, upper)
    return 0.5 * (lower + upper)


def initialize_policy_from_replay(policy, buffer, excluded_episode_ids=()):
    """Initialize from the real episode with the greatest forward coverage.

    The RBF centers follow diverse states from that episode, and virtual targets
    reproduce its measured actions through the inverse policy squash. This is a
    data-driven starting point for PILCO's gradient search, not a hardcoded
    controller; the selected episode and torques change with the replay.
    """
    if buffer.size == 0:
        raise ValueError("Cannot initialize a policy without real replay data")
    identifiers = np.unique(buffer.episode_id[:buffer.size])
    identifiers = identifiers[identifiers >= 0]
    identifiers = identifiers[
        ~np.isin(identifiers, list(excluded_episode_ids))]
    if len(identifiers) == 0:
        raise ValueError("Replay has no episode identities for policy initialization")
    best_episode = max(
        map(int, identifiers),
        key=lambda episode: np.max(buffer.s_next[:buffer.size][
            buffer.episode_id[:buffer.size] == episode, 0]),
    )
    episode_indices = np.flatnonzero(
        buffer.episode_id[:buffer.size] == best_episode)
    values = buffer.s[episode_indices].astype(np.float64)
    standardized = _robust_standardize(values)
    indices = _farthest_indices(
        standardized, np.arange(len(values)), min(len(values), policy.M.shape[0]))
    if not indices:
        raise ValueError("Cannot initialize a policy without training states")
    while len(indices) < policy.M.shape[0]:
        indices.extend(indices[:policy.M.shape[0] - len(indices)])
    chosen_replay = episode_indices[np.asarray(indices, dtype=np.int64)]
    states = torch.as_tensor(
        buffer.s[chosen_replay], dtype=DTYPE, device=policy.M.device)
    actions = torch.as_tensor(
        buffer.a[chosen_replay, 0], dtype=DTYPE, device=policy.M.device)
    all_episode_states = torch.as_tensor(
        values, dtype=DTYPE, device=policy.M.device)
    with torch.no_grad():
        policy.M.copy_(states)
        policy.log_lengthscale.copy_(
            all_episode_states.std(0, unbiased=False).clamp_min(0.2).log())
        policy.t.copy_(_inverse_policy_squash(actions / policy.tau_max))
    coverage = float(np.max(buffer.s_next[episode_indices, 0]))
    print(f"[pilco] initialized policy from replay episode {best_episode} "
          f"(coverage x={coverage:.2f}, mean torque={float(actions.mean()):.2f})",
          flush=True)


def write_episode_log(path, rows):
    """Write new and legacy rows without losing either schema during resume."""
    preferred = [
        "episode", "mode", "x_final", "reached", "crashed", "return", "steps",
        "sim_time", "effort_rms", "buffer_size", "train_points",
        "coverage_x",
        "validation_points", "validation_active_points", "validation_passed",
        "validation_skill", "validation_active_skill", "validation_obstacle_skill",
        "validation_reason", "policy_updated", "policy_update_error",
        "gp_loss_x", "gp_loss_v",
        "gp_loss_theta", "gp_loss_omega", "policy_loss", "wall_seconds",
    ]
    extras = []
    for row in rows:
        for key in row:
            if key not in preferred and key not in extras:
                extras.append(key)
    fieldnames = preferred + extras
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


# ============================================================
# Main loop (Algorithm 1): collect one real trial, refit the GPs, improve the
# policy, repeat.
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-only", action="store_true", default=not TRAIN)
    parser.add_argument("--resume", action="store_true", default=LOAD_MODEL)
    parser.add_argument(
        "--eval-best", action=argparse.BooleanOptionalAction, default=True,
        help="evaluate the best measured real-trial policy when one exists "
             "(use --no-eval-best for the latest untested update)")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=RENDER_EVAL)
    parser.add_argument("--render-speed", type=float, default=1.0,
                        help="rendered evaluation speed: 1=real time, 0.5=half speed, 0.25=quarter speed")
    parser.add_argument("--device", default="cuda",
                        help="torch device for GP fitting and policy optimization (default: cuda)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help="number of additional real trials, including warmup when starting fresh")
    parser.add_argument("--warmup-trials", type=int, default=WARMUP_TRIALS)
    parser.add_argument("--gp-iters", type=int, default=GP_TRAIN_ITERS)
    parser.add_argument("--gp-restarts", type=int, default=GP_RESTARTS)
    parser.add_argument("--policy-steps", type=int, default=POLICY_GRAD_STEPS)
    parser.add_argument("--horizon", type=int, default=PLAN_HORIZON)
    parser.add_argument("--max-gp-points", type=int, default=MAX_GP_POINTS)
    parser.add_argument("--basis", type=int, default=POLICY_N_BASIS)
    for name, default in (("x", SIGMA_X), ("v", SIGMA_V),
                          ("theta", SIGMA_THETA), ("omega", SIGMA_OMEGA)):
        parser.add_argument(f"--sigma-{name}", type=float, default=default,
                            help="width of the PILCO saturating target cost in physical units")
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if not math.isfinite(args.render_speed) or args.render_speed <= 0:
        parser.error("--render-speed must be finite and positive")
    for name in ("threads", "gp_iters", "gp_restarts", "policy_steps", "horizon",
                 "max_gp_points", "basis"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if min(args.trials, args.warmup_trials, args.eval_episodes) < 0:
        parser.error("trial and episode counts must be nonnegative")
    if any(not math.isfinite(getattr(args, f"sigma_{name}")) or
           getattr(args, f"sigma_{name}") <= 0 for name in ("x", "v", "theta", "omega")):
        parser.error("cost widths must be finite and positive")
    return args


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    explicit_options = {
        token.split("=", 1)[0] for token in raw_argv if token.startswith("--")
    }
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or pass --device cpu explicitly."
        )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        print(f"[pilco] device={device} ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print(f"[pilco] device={device}", flush=True)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / MODEL_PATH.name
    buffer_path = args.output_dir / BUFFER_PATH.name
    log_path = args.output_dir / EPISODE_LOG_PATH.name
    best_path = args.output_dir / "pilco_best_policy.pt"
    checkpoint = None
    if args.resume or args.eval_only:
        if not model_path.exists():
            raise FileNotFoundError(f"No PILCO checkpoint at {model_path}; train first.")
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        # Checkpoints know their architecture; evaluation needs no matching CLI flags.
        args.basis = int(checkpoint["policy"]["M"].shape[0])
        if args.resume and not args.eval_only and \
                int(checkpoint.get("format_version", 2)) >= 3:
            # Continue the same experiment unless the user explicitly overrides a
            # scientific setting on this command line. Runtime choices such as
            # device, rendering, and the number of additional trials stay local
            # to the invocation.
            option_by_setting = {
                "warmup_trials": "--warmup-trials",
                "gp_iters": "--gp-iters",
                "gp_restarts": "--gp-restarts",
                "policy_steps": "--policy-steps",
                "horizon": "--horizon",
                "max_gp_points": "--max-gp-points",
                "sigma_x": "--sigma-x",
                "sigma_v": "--sigma-v",
                "sigma_theta": "--sigma-theta",
                "sigma_omega": "--sigma-omega",
            }
            restored = []
            saved_settings = checkpoint.get("settings", {})
            for setting, option in option_by_setting.items():
                if option in explicit_options or setting not in saved_settings:
                    continue
                current = getattr(args, setting)
                setattr(args, setting, type(current)(saved_settings[setting]))
                restored.append(setting)
            if restored:
                print("[pilco] restored saved training settings: "
                      + ", ".join(restored), flush=True)
        if args.resume and not args.eval_only and not buffer_path.exists():
            raise FileNotFoundError(f"Cannot resume without real replay data: {buffer_path}")
    else:
        # These paths belong to this PILCO experiment. A fresh invocation starts
        # one coherent generation rather than leaving files from an older run
        # available if the first new trial is interrupted.
        for stale_path in (model_path, buffer_path, log_path, best_path):
            stale_path.unlink(missing_ok=True)

    env = ObstacleWheelieEnv(reward_cfg=RewardConfig())
    eval_env = None
    try:
        p = env.p
        if not np.isclose(p.tau_min, -p.tau_max):
            raise ValueError("PILCO's sine policy currently requires symmetric torque bounds.")
        policy = RBFPolicy(STATE_DIM, args.basis, p.tau_max).to(device)
        gps = [ExactGP(GP_INPUT_DIM).to(device) for _ in range(STATE_DIM)]
        buffer = PILCOReplayBuffer()
        start_trial = 0
        best_return = float("-inf")
        checkpoint_format = int(checkpoint.get("format_version", 2)) \
            if checkpoint is not None else 0
        policy_initialized = bool(checkpoint.get(
            "policy_initialized", checkpoint_format >= 3)) \
            if checkpoint is not None else False
        policy_ready = False
        validation_episode_ids = []
        episode_rows = []
        if checkpoint is not None:
            policy.load_state_dict(checkpoint["policy"])
            saved_dynamics = checkpoint.get("dynamics")
            if saved_dynamics is not None:
                if len(saved_dynamics) != STATE_DIM:
                    raise ValueError("PILCO checkpoint does not contain four dynamics GPs")
                for gp, state in zip(gps, saved_dynamics):
                    gp.load_state_dict(state)
                policy_ready = bool(checkpoint.get("policy_ready", False))
            elif args.resume and not args.eval_only:
                print("[pilco] legacy two-GP checkpoint: refitting four normalized "
                      "GPs and reinitializing policy targets from real replay", flush=True)
            policy_ready = bool(checkpoint.get("policy_ready", policy_ready))
            validation_episode_ids = list(map(
                int, checkpoint.get("validation_episode_ids", [])))
            start_trial = int(checkpoint.get("next_trial", 0))
            best_return = float(checkpoint.get("best_return", float("-inf")))
            if best_path.exists():
                saved_best = torch.load(
                    best_path, map_location=device, weights_only=True)
                if saved_best.get("validation_approved", False):
                    # The measured-best file is published as soon as a real
                    # rollout finishes. If the process stopped before the main
                    # generation commit, retain that newer measurement.
                    best_return = max(
                        best_return, float(saved_best["episode_return"]))
            if not args.eval_only:
                buffer.load(buffer_path)
                if "rng_state" in checkpoint:
                    rng.bit_generator.state = checkpoint["rng_state"]
                if "torch_rng_state" in checkpoint:
                    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
                if "numpy_rng_state" in checkpoint:
                    state = checkpoint["numpy_rng_state"]
                    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32),
                                         state[2], state[3], state[4]))
                if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
                    torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])
                if log_path.exists():
                    with log_path.open(newline="") as stream:
                        episode_rows = list(csv.DictReader(stream))
                    if "next_trial" not in checkpoint and episode_rows:
                        start_trial = int(episode_rows[-1]["episode"]) + 1
                buffer.restore_legacy_episode_ids(episode_rows)
                if int(checkpoint.get("format_version", 2)) >= 3:
                    expected_replay_size = int(
                        checkpoint.get("replay_size", buffer.size))
                    if buffer.size < expected_replay_size:
                        raise ValueError(
                            "Replay is older than the committed PILCO checkpoint: "
                            f"{buffer.size} < {expected_replay_size} transitions")
                    recovered = False
                    if buffer.size > expected_replay_size:
                        discarded = buffer.size - expected_replay_size
                        buffer.truncate(expected_replay_size)
                        recovered = True
                        print(f"[pilco] discarded {discarded} uncommitted replay "
                              "transitions", flush=True)
                    committed_rows = [
                        row for row in episode_rows
                        if int(float(row["episode"])) < start_trial
                    ]
                    if len(committed_rows) != len(episode_rows):
                        episode_rows = committed_rows
                        recovered = True
                        print("[pilco] discarded uncommitted episode-log rows", flush=True)
                    if recovered:
                        temporary_buffer = buffer_path.with_name(buffer_path.name + ".tmp")
                        temporary_log = log_path.with_name(log_path.name + ".tmp")
                        buffer.save(temporary_buffer)
                        write_episode_log(temporary_log, episode_rows)
                        temporary_buffer.replace(buffer_path)
                        temporary_log.replace(log_path)
            print(f"[pilco] loaded {model_path}; real transitions={buffer.size}", flush=True)

        if args.eval_only and checkpoint is not None and \
                int(checkpoint.get("format_version", 2)) >= 3 and not policy_ready:
            raise RuntimeError(
                "This checkpoint has no validation-approved policy to evaluate. "
                "Resume training or choose a checkpoint with policy_ready=true."
            )

        env.reset(seed=args.seed)
        mu0 = torch.as_tensor(env.get_state(), dtype=DTYPE, device=device)
        Sigma0 = INIT_STATE_VAR * torch.eye(STATE_DIM, dtype=DTYPE, device=device)
        target = torch.tensor([env.goal_x, 0.0, 0.0, 0.0], dtype=DTYPE, device=device)
        W = torch.diag(torch.tensor(
            [args.sigma_x ** -2, args.sigma_v ** -2,
             args.sigma_theta ** -2, args.sigma_omega ** -2],
            dtype=DTYPE, device=device))
        if not args.eval_only:
            for trial in range(start_trial, start_trial + args.trials):
                t0 = time.perf_counter()
                coverage_before_trial = _maximum_replay_x(
                    buffer, validation_episode_ids)
                if trial < args.warmup_trials or buffer.size == 0:
                    mode = "random-warmup"
                    transitions, ep_return, info = run_exploration_episode(env, rng)
                elif _is_scheduled_validation_episode(
                        trial, args.warmup_trials):
                    mode = "random-validation"
                    transitions, ep_return, info = run_exploration_episode(env, rng)
                elif coverage_before_trial < env.goal_x:
                    mode = "random-coverage"
                    transitions, ep_return, info = run_exploration_episode(env, rng)
                elif not policy_ready:
                    mode = "random-model-gate"
                    transitions, ep_return, info = run_exploration_episode(env, rng)
                else:
                    mode = "policy"
                    transitions, ep_return, info = run_policy_episode(env, policy, device=device)
                    if ep_return > best_return:
                        best_return = float(ep_return)
                        temporary_best = best_path.with_suffix(".tmp")
                        torch.save(dict(policy=policy.state_dict(), trial=trial,
                                        episode_return=best_return, info=info,
                                        format_version=3, validation_approved=True), temporary_best)
                        temporary_best.replace(best_path)
                for state, action, next_state in transitions:
                    buffer.add(state, action, next_state, episode_id=trial)
                if best_path.exists():
                    measured_best = torch.load(
                        best_path, map_location=device, weights_only=True)
                    if measured_best.get("validation_approved", False):
                        # A candidate policy becomes the optimization base only
                        # after a real policy episode proves it is at least the
                        # best seen so far. Random collection and failed
                        # candidates therefore cannot displace a working policy.
                        policy.load_state_dict(measured_best["policy"])
                        policy_initialized = True
                        policy_ready = True
                validation_episode_ids = update_validation_episode_ids(
                    buffer, validation_episode_ids, args.warmup_trials)
                coverage_x = _maximum_replay_x(
                    buffer, validation_episode_ids)
                datasets = build_gp_datasets(
                    buffer, validation_episode_ids, args.max_gp_points,
                    p.tau_max, device)

                gp_losses = [float("nan")] * STATE_DIM
                validation = {
                    "passed": False,
                    "reason": "waiting for a complete training/validation split",
                    "points": 0,
                    "active_points": 0,
                    "overall_skill": float("nan"),
                    "active_skill": float("nan"),
                    "obstacle_skill": float("nan"),
                }
                policy_updated = False
                policy_update_error = ""
                policy_loss = float("nan")
                train_points = 0
                if datasets is not None:
                    X_train, Y_train, X_validation, Y_validation, _chosen = datasets
                    train_points = len(X_train)
                    print(
                        f"[pilco trial {trial}] fitting 4 GPs on {train_points} diverse "
                        f"points ({args.gp_restarts} restarts each); validating on "
                        f"{len(X_validation)} held-out episode points",
                        flush=True,
                    )
                    input_mean, input_scale = ExactGP.input_statistics(X_train)
                    previous_dynamics = [
                        {key: value.detach().clone() for key, value in gp.state_dict().items()}
                        for gp in gps
                    ]
                    try:
                        for dimension, gp in enumerate(gps):
                            gp_losses[dimension] = gp.fit(
                                X_train, Y_train[:, dimension],
                                input_mean=input_mean, input_scale=input_scale,
                                iters=args.gp_iters, lr=GP_LR,
                                restarts=args.gp_restarts)
                        validation = validate_dynamics(
                            gps, X_validation, Y_validation, p.tau_max)
                    except (RuntimeError, FloatingPointError, ValueError) as error:
                        for gp, state in zip(gps, previous_dynamics):
                            gp.load_state_dict(state)
                        validation["reason"] = f"GP fitting failed: {error}"

                    if coverage_x < env.goal_x:
                        coverage_reason = (
                            f"GP training replay reaches only x={coverage_x:.2f}; "
                            f"goal coverage requires x>={env.goal_x:.2f}")
                        validation["passed"] = False
                        validation["reason"] = (
                            coverage_reason if validation["reason"] == "passed"
                            else validation["reason"] + "; " + coverage_reason)

                    if validation["passed"]:
                        previous_policy = {
                            key: value.detach().clone()
                            for key, value in policy.state_dict().items()
                        }
                        previous_policy_initialized = policy_initialized
                        previous_policy_ready = policy_ready
                        try:
                            if not policy_initialized:
                                initialize_policy_from_replay(
                                    policy, buffer, validation_episode_ids)
                                policy_initialized = True
                            policy_loss = improve_policy(
                                policy, gps, mu0, Sigma0, target, W,
                                args.horizon, steps=args.policy_steps)
                            policy_ready = True
                            policy_updated = any(
                                not torch.equal(
                                    policy.state_dict()[key], previous_value)
                                for key, previous_value in previous_policy.items()
                            )
                        except (RuntimeError, FloatingPointError, ValueError) as error:
                            policy.load_state_dict(previous_policy)
                            policy_initialized = previous_policy_initialized
                            policy_ready = previous_policy_ready
                            policy_update_error = str(error)
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                            print(
                                f"[pilco trial {trial}] policy improvement failed; "
                                f"previous policy restored: {error}", flush=True)
                    else:
                        print(f"[pilco trial {trial}] policy update skipped: "
                              f"{validation['reason']}", flush=True)
                else:
                    print(f"[pilco trial {trial}] policy update skipped: "
                          f"{validation['reason']}", flush=True)
                wall = time.perf_counter() - t0
                row = dict(
                    episode=trial, mode=mode, x_final=info["x"],
                    reached=int(info["reached"]), crashed=int(info["crashed"]),
                    **{"return": ep_return}, steps=len(transitions),
                    sim_time=info["sim_time"], effort_rms=info["effort_rms"],
                    buffer_size=buffer.size, train_points=train_points,
                    coverage_x=coverage_x,
                    validation_points=validation["points"],
                    validation_active_points=validation["active_points"],
                    validation_passed=int(validation["passed"]),
                    validation_skill=validation["overall_skill"],
                    validation_active_skill=validation["active_skill"],
                    validation_obstacle_skill=validation["obstacle_skill"],
                    validation_reason=validation["reason"],
                    policy_updated=int(policy_updated),
                    policy_update_error=policy_update_error,
                    gp_loss_x=gp_losses[0], gp_loss_v=gp_losses[1],
                    gp_loss_theta=gp_losses[2], gp_loss_omega=gp_losses[3],
                    policy_loss=policy_loss, wall_seconds=wall)
                episode_rows.append(row)
                temporary_model = model_path.with_name(model_path.name + ".tmp")
                temporary_buffer = buffer_path.with_name(buffer_path.name + ".tmp")
                temporary_log = log_path.with_name(log_path.name + ".tmp")
                numpy_state = np.random.get_state()
                torch.save(dict(
                    format_version=3, policy=policy.state_dict(),
                    dynamics=[gp.state_dict() for gp in gps],
                    next_trial=trial + 1, replay_size=buffer.size,
                    best_return=best_return,
                    policy_initialized=policy_initialized, policy_ready=policy_ready,
                    validation_episode_ids=validation_episode_ids,
                    last_validation=validation,
                    rng_state=rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                    numpy_rng_state=(numpy_state[0], numpy_state[1].tolist(),
                                     numpy_state[2], numpy_state[3], numpy_state[4]),
                    cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    settings={key: str(value) if isinstance(value, Path) else value
                              for key, value in vars(args).items()}), temporary_model)
                buffer.save(temporary_buffer)
                write_episode_log(temporary_log, episode_rows)
                # The checkpoint is the commit marker, so publish it only after
                # replay and log are durable. Resume trims either file back to
                # the last committed generation if an earlier replace was the
                # point of interruption.
                temporary_buffer.replace(buffer_path)
                temporary_log.replace(log_path)
                temporary_model.replace(model_path)
                print(f"[trial {trial:02d}] mode={mode} x_final={info['x']:.2f} "
                      f"reached={info['reached']} crashed={info['crashed']} "
                      f"return={ep_return:.2f} model_valid={validation['passed']} "
                      f"policy_updated={policy_updated} policy_loss={policy_loss:.6f} "
                      f"wall={wall:.1f}s",
                      flush=True)

        legacy_policy_only = (checkpoint is not None
                              and int(checkpoint.get("format_version", 2)) < 3
                              and args.eval_only)
        if args.eval_episodes and not (policy_ready or legacy_policy_only):
            print("[pilco] evaluation skipped: no validation-approved policy exists yet",
                  flush=True)
        elif args.eval_episodes:
            if args.eval_best:
                if best_path.exists():
                    best = torch.load(best_path, map_location=device, weights_only=True)
                    policy.load_state_dict(best["policy"])
                    print(f"[pilco] evaluating best real-trial policy: trial={best['trial']} "
                          f"return={best['episode_return']:.2f}", flush=True)
                else:
                    print("[pilco] no measured best policy yet; evaluating the latest "
                          "validation-approved candidate", flush=True)
            eval_env = ObstacleWheelieEnv(
                render_mode="human" if args.render else None, reward_cfg=RewardConfig())
            n_reached = 0
            for ep in range(args.eval_episodes):
                transitions, ep_return, info = run_policy_episode(
                    eval_env, policy, device=device, render_speed=args.render_speed)
                n_reached += int(info["reached"])
                print(f"[eval ep {ep}] x_final={info['x']:.2f} reached={info['reached']} "
                      f"crashed={info['crashed']} return={ep_return:.2f} "
                      f"steps={len(transitions)}", flush=True)
            print(f"[eval] reached goal in {n_reached}/{args.eval_episodes} episodes")
    finally:
        if eval_env is not None:
            eval_env.close()
        env.close()


if __name__ == "__main__":
    main()
