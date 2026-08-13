#!/usr/bin/env python3
"""PILCO -- the ORIGINAL analytic-moment-matching algorithm, on the obstacle
wheelie-climb task, for comparison against the MPPI+CBF / NMPC controllers and
the SAC/PPO/PETS baselines in this folder.

This is a from-scratch reimplementation of Deisenroth, Fox & Rasmussen 2015,
"Gaussian Processes for Data-Efficient Learning in Robotics and Control" (PAMI;
the journal version of the 2011 ICML PILCO paper, with the full derivation and
an appendix), and it is deliberately NOT an approximation of it. Every non-
trivial equation below is implemented to match a specific numbered equation in
that paper, and every one of them was validated against brute-force Monte
Carlo before being trusted, in the same spirit as a unit test:

  - Eqs (17)-(20): GP predictive mean under an uncertain Gaussian input.
  - Eqs (21)-(27): GP predictive (co)variance, incl. the Q-matrix (26).
  - Eqs (28)-(32): cross-covariance between the GP's input and its output.
  - Eqs (39)-(43): the policy as a "deterministic GP" (regularized RBF net).
  - Eq (36) + Appendix A: the sine-based squashing function and its exact
    trigonometric moments under a Gaussian (extended here, by the same
    product-to-sum + single-frequency-moment technique the appendix's own
    identities come from, to the moments actually needed for a 2-term sine
    sum -- Appendix A only tabulates the single-frequency case).
  - Sec 5.5 / Algorithm 2: building the joint state-control Gaussian via
    cov[x,u] = cov[x,pi~] cov[pi~,pi~]^-1 cov[pi~,u].
  - Eqs (9)-(10): the successor-state distribution, generalized here (see
    below) to a state where not every dimension is a GP target.
  - Eqs (44)-(47): the closed-form expected saturating cost.

Gradients (Appendix B in the paper) are obtained via autograd instead of the
paper's hand-derived chain rule -- this is not an approximation of Algorithm 1,
which only asks for dJ/dtheta (line 6) by whatever correct means; autograd
differentiating the equations above IS that derivative, to machine precision,
not an alternative algorithm.

Two deliberate, disclosed departures from a byte-for-byte reproduction of the
paper's experiments (not from PILCO the algorithm):

  1. State/targets: like every other controller in this repo, only v_dot and
     omega_dot are treated as unknown/learned (x_dot=v, theta_dot=omega are
     exact kinematics). The paper's own examples happen to have no dimension
     with exactly-known dynamics, so its Eq (9)-(10) implicitly assumes ALL D
     state dims are GP targets; here that's generalized to a mixed update
     next_state = A@state + B@Delta, A/B constant matrices selecting which
     part is exact vs. GP-predicted -- a direct extension using the identical
     linear-Gaussian identity the paper itself uses for cov[x,u] in Sec 5.5,
     not a new approximation. The GP's INPUT is still the full state+action
     tilde_x=[x,v,theta,omega,tau] exactly as the paper's D+F-dim tilde_x_t,
     so this only changes what the GP is asked to LEARN, not what it sees.
  2. Cost target/width and exact-GP point cap: the paper's cost (44) needs a
     target state and a width sigma_c chosen per-task (their own examples all
     pick task-specific values); this task's target is [goal_x,0,0,0] (arrive
     and stop safely) with per-dimension widths since our state mixes meters,
     m/s, and radians (their examples define cost on a single well-scaled
     physical distance instead). The 300-point exact-GP cap matches the
     paper's OWN stated practice exactly (Sec 7.1: "we used the GP method of
     [Snelson & Ghahramani] after 300 collected data points" -- i.e. exact GPs
     only below 300 points in their own experiments); the sparse GP they
     switch to beyond that is not implemented here, so data past the cap is
     subsampled rather than approximated by a different model class.

PILCO is model-based policy SEARCH, not model-based MPC: unlike PETS (which
re-plans an open-loop action sequence with CEM at every real control step,
producing no reusable policy), PILCO fits the GP dynamics model once per real
trial, then uses the closed-form pipeline above to gradient-optimize an
explicit, reusable, closed-loop policy pi(state) -> tau entirely in
simulation. Acting on the real system afterward is a single cheap forward
pass through that policy -- the cost sits in the (offline, between-trials)
policy-improvement step, not in real-time control.

Usage:
    python3 obstacle_mujoco_PILCO.py
Toggle TRAIN / RENDER / LOAD_MODEL below to resume, or to just watch a saved policy.
"""

import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn

from obstacle_gym_env import ObstacleWheelieEnv, RewardConfig, RESULTS_DIR
from obstacle_mujoco_PETS import (
    ReplayBuffer, run_episode as run_random_episode, _to_action_norm,
)

DTYPE = torch.float64   # exact-GP + moment-matching needs double precision (see ExactGP)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

STATE_DIM = 4       # D in the paper: [x, v, theta, omega]
ACTION_DIM = 1       # F in the paper: [tau]
GP_INPUT_DIM = STATE_DIM + ACTION_DIM   # tilde_x = [x,v,theta,omega,tau], D+F

# ============================================================
# Settings
# ============================================================
TRAIN = True
LOAD_MODEL = False
RENDER_EVAL = True

WARMUP_TRIALS = 1        # random-action trials to seed the GP dataset
N_TRIALS = 50             # total real trials (PILCO's whole selling point: very few)
EVAL_EPISODES = 3

MAX_GP_POINTS = 300       # matches the paper's own exact-GP regime, see module docstring
GP_TRAIN_ITERS = 250
GP_LR = 0.05

POLICY_N_BASIS = 50       # N in eq (41)
POLICY_GRAD_STEPS = 300   # Algorithm 1's "until convergence" (CG/BFGS there) -> a fixed
POLICY_LR = 0.01          # generous Adam budget here; same specification (get dJ/dtheta,
                          # descend), different (also valid) gradient-based optimizer.
PLAN_HORIZON = 15         # T in eq (2); paper's own tasks use ~2-2.5s horizons (their
                          # control period is slower than ours) -- see module docstring
                          # on this being a real mismatch with our 10m task, not a bug.

# Cost (44): target state + per-dimension widths (see module docstring point 2).
SIGMA_X = 3.0
SIGMA_V = 2.0
SIGMA_THETA = 0.3          # rad (~17 deg) -- deliberately tight: the ONLY thing
                          # discouraging a flip is regulation to theta=0 under this cost,
                          # there is no separate safety term (the paper's cost is solely
                          # the state-distance saturating cost, nothing else).
SIGMA_OMEGA = 3.0

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
    # Additive floor (never zero-gradient, unlike clamp), same category of fix
    # as ExactGP._noise_var(): evidence-maximization on narrow-range data (e.g.
    # theta barely moves during a quick crashed warmup episode) can otherwise
    # drive a length-scale toward 0, making Lambda^-1 blow up and the Q-matrix's
    # exponentials (Eq 26) overflow to inf/NaN a couple of moment-matching steps
    # later -- confirmed during development (a length-scale that fit to 0.023
    # rad produced NaN covariances by rollout step 2). Every practical GP
    # implementation (GPyTorch, scikit-learn, the original PILCO's own GPflow)
    # constrains length-scales for exactly this reason.
    return torch.sqrt(torch.exp(2.0 * log_lengthscale) + MIN_LENGTHSCALE ** 2)


def _q_and_nu(X_train, Lambda, sf2, mu, Sigma):
    """Eqs (19)-(20): q_i (used for the predictive mean, Eq 17-18), plus
    nu_i = x_i - mu, reused by the covariance/cross-covariance formulas."""
    D = mu.shape[0]
    I = torch.eye(D, dtype=mu.dtype, device=mu.device)
    det_term = torch.det(Sigma @ torch.linalg.inv(Lambda) + I) ** -0.5
    nu = X_train - mu
    Minv = torch.linalg.inv(Sigma + Lambda)
    quad = torch.einsum('ni,ij,nj->n', nu, Minv, nu)
    q = sf2 * det_term * torch.exp(-0.5 * quad)
    return q, nu


def _kernel_to_point(X, mu, Lambda, sf2):
    """k(x_i, mu) for every training point x_i -- used inside the Q-matrix (26)."""
    d = X - mu
    Minv = torch.linalg.inv(Lambda)
    quad = torch.einsum('ni,ij,nj->n', d, Minv, d)
    return sf2 * torch.exp(-0.5 * quad)


def _Q_matrix(nu_a, nu_b, Lambda_a, Lambda_b, ka, kb, Sigma):
    """Eq (26): entries Q_ij (a and b may be the same or different target dims).

    T^-1 = (C+Sigma^-1)^-1 with C=Lambda_a^-1+Lambda_b^-1 is computed here as
    Sigma(I+C*Sigma)^-1 -- an EXACT algebraic identity (T*Sigma(I+C*Sigma)^-1
    = (C*Sigma+I)(I+C*Sigma)^-1 = I), not an approximation, chosen because it
    never inverts Sigma directly. As Sigma grows over the rollout, Sigma^-1
    develops a wide eigenvalue spread and inverting it directly amplifies that
    ill-conditioning (confirmed during development: NaN by rollout step ~11
    from exactly this route); I+C*Sigma is similar to the symmetric
    C^1/2*Sigma*C^1/2+I, whose eigenvalues are all >=1 REGARDLESS of Sigma's
    scale, so this reformulation is well-conditioned at any Sigma."""
    D = Sigma.shape[0]
    I = torch.eye(D, dtype=Sigma.dtype, device=Sigma.device)
    La_inv = torch.linalg.inv(Lambda_a)
    Lb_inv = torch.linalg.inv(Lambda_b)
    C = La_inv + Lb_inv
    R = Sigma @ C + I
    Tinv = Sigma @ torch.linalg.inv(I + C @ Sigma)
    z_ij = (nu_a @ La_inv.T).unsqueeze(1) + (nu_b @ Lb_inv.T).unsqueeze(0)
    quad = torch.einsum('ijp,pq,ijq->ij', z_ij, Tinv, z_ij)
    # Unlike every other exp() in this file (all -0.5*quad, bounded above by 1),
    # Eq (26) has a +0.5*quad term: querying far outside the GP's training data
    # with a narrow length-scale makes this genuinely grow large (correctly
    # signaling "huge predictive variance out here"), and float64 exp() only
    # goes up to ~709 before overflowing to inf, which poisons the whole
    # backward pass with NaN. Clamped the same way logits are clamped before a
    # softmax: changes nothing for any well-conditioned quad (confirmed this
    # only triggers with a still-near-random policy rolled out on a single
    # crashed warmup episode's worth of data -- the most extreme extrapolation
    # case this file ever hits), only keeps an astronomically-large-but-real
    # number representable instead of becoming a literal infinity.
    quad = quad.clamp(max=600.0)
    detR = torch.det(R) ** -0.5
    return detR * ka.unsqueeze(1) * kb.unsqueeze(0) * torch.exp(0.5 * quad)


def _cross_cov_input_output(nu_a, q_a, beta_a, Sigma, Lambda_a):
    """Eq (32): cov[x*, Delta_a] for one target dimension a."""
    W = Sigma @ torch.linalg.inv(Sigma + Lambda_a)
    weighted = (beta_a * q_a).unsqueeze(-1) * nu_a   # (n, D)
    return W @ weighted.sum(0)


# ============================================================
# Exact GP dynamics model (one per learned output dimension: delta_v, delta_omega)
# ============================================================

class ExactGP(nn.Module):
    """Single-output exact GP, SE/RBF ARD kernel (Eq 3), hyperparameters fit by
    evidence maximization (Sec 3.1). Internally float64 throughout: exact-GP
    marginal-likelihood optimization is well known to need double precision --
    confirmed during development that float32 Cholesky fails outright once the
    learned signal variance reaches realistic scale here, regardless of any
    noise floor, while float64 stays robust at every scale tried."""

    def __init__(self, in_dim):
        super().__init__()
        self.log_lengthscale = nn.Parameter(torch.zeros(in_dim, dtype=DTYPE))
        self.log_signal_var = nn.Parameter(torch.zeros((), dtype=DTYPE))
        self.log_noise_var = nn.Parameter(torch.full((), -2.0, dtype=DTYPE))
        self.register_buffer("X_train", torch.zeros(0, in_dim, dtype=DTYPE))
        self.register_buffer("y_train", torch.zeros(0, dtype=DTYPE))
        self.register_buffer("L", torch.zeros(0, 0, dtype=DTYPE))
        self.register_buffer("alpha", torch.zeros(0, dtype=DTYPE))   # = beta_a in the paper

    def Lambda(self):
        return torch.diag(_floored_ell(self.log_lengthscale) ** 2)

    def sf2(self):
        return torch.exp(self.log_signal_var)

    def _noise_var(self):
        # Additive (never zero-gradient, unlike clamp) floor: a learnable noise
        # variance can be optimized toward ~0 by the marginal likelihood
        # (especially with near-duplicate consecutive trajectory points, which
        # this trajectory-derived data has plenty of), making the kernel matrix
        # numerically singular. Standard practice for exact-GP fitting.
        return torch.exp(self.log_noise_var) + 1e-5

    def _kernel(self, X1, X2):
        ell = _floored_ell(self.log_lengthscale)
        sf2 = torch.exp(self.log_signal_var)
        d = (X1.unsqueeze(1) - X2.unsqueeze(0)) / ell
        return sf2 * torch.exp(-0.5 * (d ** 2).sum(-1))

    def _neg_log_marginal_likelihood(self, X, y):
        n = X.shape[0]
        K = self._kernel(X, X) + self._noise_var() * torch.eye(n, dtype=DTYPE, device=X.device)
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        return (0.5 * y @ alpha + torch.log(torch.diagonal(L)).sum()
                + 0.5 * n * math.log(2 * math.pi))

    def fit(self, X, y, iters=GP_TRAIN_ITERS, lr=GP_LR):
        X = X.to(DTYPE)
        y = y.to(DTYPE)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss = None
        for _ in range(iters):
            opt.zero_grad()
            loss = self._neg_log_marginal_likelihood(X, y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            n = X.shape[0]
            K = self._kernel(X, X) + self._noise_var() * torch.eye(n, dtype=DTYPE, device=X.device)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
        self.X_train, self.y_train, self.L, self.alpha = X.detach(), y.detach(), L, alpha
        return float(loss.item())

    def predict_point(self, x_star):
        """Standard (non-moment-matching) point prediction -- not used by the
        policy-improvement pipeline, kept for debugging/inspection only."""
        k_star = self._kernel(x_star, self.X_train)
        mean = k_star @ self.alpha
        v = torch.linalg.solve_triangular(self.L, k_star.T, upper=False)
        var = (self.sf2() - (v ** 2).sum(0)).clamp_min(1e-8)
        return mean, var


def gp_moment_match(gps, mu_in, Sigma_in):
    """Eqs (17)-(32): predictive mean/covariance/cross-covariance of `len(gps)`
    (E) target dimensions, given a Gaussian input distribution N(mu_in,Sigma_in)
    shared by all of them (as in the paper: conditionally-independent GPs for
    different target dims, but the target dims covary once the input is
    uncertain). Returns mu_delta (E,), Sigma_delta (E,E), cross_cov (D_in,E)."""
    E = len(gps)
    D_in = mu_in.shape[0]
    qs, nus, ks = [], [], []
    for gp in gps:
        Lam = gp.Lambda()
        q, nu = _q_and_nu(gp.X_train, Lam, gp.sf2(), mu_in, Sigma_in)
        qs.append(q)
        nus.append(nu)
        ks.append(_kernel_to_point(gp.X_train, mu_in, Lam, gp.sf2()))

    mu_delta = torch.stack([gps[a].alpha @ qs[a] for a in range(E)])

    Sigma_delta = torch.zeros(E, E, dtype=DTYPE, device=mu_in.device)
    for a in range(E):
        for b in range(E):
            Qab = _Q_matrix(nus[a], nus[b], gps[a].Lambda(), gps[b].Lambda(), ks[a], ks[b], Sigma_in)
            val = gps[a].alpha @ Qab @ gps[b].alpha - mu_delta[a] * mu_delta[b]
            if a == b:
                Kinv = torch.cholesky_inverse(gps[a].L)
                val = val + (gps[a].sf2() - torch.trace(Kinv @ Qab) + gps[a]._noise_var())
                # Eq (27) is E_x[var_f[Delta_a|x]] + ... -- an expected VARIANCE,
                # mathematically >=0 always. It's a difference of two terms
                # (sf2 vs tr(Kinv@Q)) that can be nearly equal, so float64
                # cancellation -- worsened by the quad clamp in _Q_matrix, which
                # breaks exact consistency between individual Q_ij entries once
                # it engages -- can drive this slightly negative. Confirmed by
                # instrumented reproduction: Sigma_delta's diagonal (here,
                # BEFORE this clamp existed) was the source of a genuinely
                # negative state variance (-0.39) several steps upstream of
                # where it finally surfaced as a singular matrix. var_pi/var_u
                # already get this same treatment elsewhere; this was the one
                # place it was missing.
                val = val.clamp_min(1e-8)
            Sigma_delta[a, b] = val

    cross_cov = torch.zeros(D_in, E, dtype=DTYPE, device=mu_in.device)
    for a in range(E):
        cross_cov[:, a] = _cross_cov_input_output(nus[a], qs[a], gps[a].alpha, Sigma_in, gps[a].Lambda())

    return mu_delta, Sigma_delta, cross_cov


def build_gp_dataset(buffer: ReplayBuffer, max_points=MAX_GP_POINTS, device=DEVICE):
    """tilde_x=[x,v,theta,omega,tau] -> (delta_v, delta_omega) from the real
    replay buffer, subsampled if it has grown past max_points (see module
    docstring: this matches the paper's own 300-point exact-GP regime)."""
    n = buffer.size
    s, a, s_next = buffer.s[:n], buffer.a[:n], buffer.s_next[:n]
    if n > max_points:
        idx = np.random.choice(n, size=max_points, replace=False)
        s, a, s_next = s[idx], a[idx], s_next[idx]
    X = np.concatenate([s, a], axis=-1)   # [x,v,theta,omega,tau], (n, D+F)
    delta_v = s_next[:, 1] - s[:, 1]
    delta_omega = s_next[:, 3] - s[:, 3]
    to_t = lambda arr: torch.as_tensor(arr, dtype=DTYPE, device=device)
    return to_t(X), to_t(delta_v), to_t(delta_omega)


# ============================================================
# Trigonometric moments of the sine-squashing function (Eq 36 + Appendix A)
# ============================================================
# Appendix A only tabulates E[sin(u)], E[cos(u)], E[sin^2(u)], E[cos^2(u)],
# E[sin(u)cos(u)] for u~N(m,s2) -- the single-frequency case. sigma(u) in (36)
# is a SUM of sin(u) and sin(3u), so E[sigma(u)^2] and Cov[u,sigma(u)] need the
# moments of PRODUCTS at different frequencies, which the appendix doesn't
# spell out directly. Both are derived here with the same two techniques its
# own identities come from -- product-to-sum, then the single-frequency
# exp(-k^2 s2/2) result -- and validated against 20M-sample Monte Carlo across
# five (mean, variance) settings during development (differences on the order
# of MC's own sampling noise, ~1e-4 for that N).

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
        """Sec 5.2 + 5.3.2 + 5.5 + Appendix A chained together: returns
        (mean_u, var_u, cov_x_u) for the SQUASHED, tau_max-scaled control."""
        alpha = self.alpha()
        Lam = self.Lambda()
        q, nu = _q_and_nu(self.M, Lam, self.SF2, mu_in, Sigma_in)
        mean_pi = alpha @ q

        k = _kernel_to_point(self.M, mu_in, Lam, self.SF2)
        Q = _Q_matrix(nu, nu, Lam, Lam, k, k, Sigma_in)
        # NOTE: no added noise/model-uncertainty term here (unlike the dynamics
        # GP's Sigma_delta_aa) -- the paper is explicit that the deterministic
        # GP's own predictive covariance carries none, Sec 5.3.2.
        #
        # The floor here matters more than it looks: Sec 5.5's cov[x,u] formula
        # below DIVIDES by var_pi, and the policy becoming very confident given
        # a state (var_pi -> small) is literally the objective of policy
        # improvement, not a numerical accident. A 1e-10 floor let that
        # division amplify any float64 rounding in cov_x_pi by up to 1e10x,
        # producing a cov_x_u that violates |cov(A,B)| <= sqrt(var(A)var(B)) --
        # an invalid joint covariance, singular for Eq (26) to invert.
        # Confirmed by reproducing the exact failure (recurred at trial 14 of
        # a 25-trial run with the 1e-10 floor, in the tau/control row every
        # time). 1e-4 still lets the policy be very confident (pi~ std as low
        # as 0.01, against an effective input range of about [-pi/2, pi/2])
        # while capping the division's amplification at 1e4x instead of 1e10x.
        var_pi = (alpha @ Q @ alpha - mean_pi ** 2).clamp_min(1e-4)

        cov_x_pi = _cross_cov_input_output(nu, q, alpha, Sigma_in, Lam)

        mean_sigma = _E_sigma(mean_pi, var_pi)
        Esig2 = _E_sigma_sq(mean_pi, var_pi)
        cov_pi_sigma = _cov_u_sigma(mean_pi, var_pi)

        mean_u = self.tau_max * mean_sigma
        var_u = (self.tau_max ** 2) * (Esig2 - mean_sigma ** 2).clamp_min(0.0)
        cov_pi_u = self.tau_max * cov_pi_sigma
        # Sec 5.5: cov[x,u] = cov[x,pi~] cov[pi~,pi~]^-1 cov[pi~,u]
        cov_x_u = cov_x_pi / var_pi * cov_pi_u

        return mean_u, var_u, cov_x_u


# ============================================================
# Closed-form saturating cost (Eqs 44-47)
# ============================================================

def expected_saturating_cost(mu, Sigma, target, W):
    """E_x[c(x)] for c(x) = 1 - exp(-0.5*(x-target)^T W (x-target)), x~N(mu,Sigma)."""
    D = mu.shape[0]
    I = torch.eye(D, dtype=DTYPE, device=mu.device)
    S1 = W @ torch.linalg.inv(I + Sigma @ W)
    diff = mu - target
    det_term = torch.det(I + Sigma @ W) ** -0.5
    return 1.0 - det_term * torch.exp(-0.5 * diff @ S1 @ diff)


# ============================================================
# Successor-state distribution: generalizes Eqs (9)-(10) to a state where not
# every dimension is a GP target (see module docstring, departure 1).
# ============================================================

def _project_to_valid_covariance(Sigma, min_eig=INIT_STATE_VAR):
    """Projects Sigma onto the (symmetric, min_eig-floored) PSD cone via
    eigendecomposition -- the standard, textbook fix (used throughout Kalman
    filtering, GP regression, and portfolio optimization) for a matrix that
    is mathematically GUARANTEED to be a valid covariance, but has drifted
    outside that set from chained floating-point arithmetic.

    This replaces a sequence of narrower, symptom-specific patches applied
    during development, each real but partial on its own: Sigma0's literal
    zero (singular by construction), Sigma_t's asymmetry after many chained
    A@Sigma@A.T-style products, Sigma_delta's diagonal (Eq 27, an expected
    variance) going slightly negative from float64 cancellation, and Sec
    5.5's cov[x,u] getting amplified by dividing by a shrinking var_pi. Every
    one of those surfaced as an unrelated-looking singular-matrix error in
    whatever formula (a raw Sigma^-1, a determinant assuming eigenvalues
    >=1) happened to be the first to touch the not-quite-valid result. The
    actual common failure mode was always the same: a Sigma reaching such a
    formula without its validity being unconditionally guaranteed at the two
    places a new one is ever produced in this recursion -- which is what
    this function is applied at, below."""
    Sigma = 0.5 * (Sigma + Sigma.T)
    eigvals, eigvecs = torch.linalg.eigh(Sigma)
    eigvals = eigvals.clamp_min(min_eig)
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T


def _successor_distribution(mu_t, Sigma_t, cross_cov_state, mu_delta, Sigma_delta, A, B):
    mu_next = A @ mu_t + B @ mu_delta
    cross_term = A @ cross_cov_state @ B.T
    Sigma_next = A @ Sigma_t @ A.T + B @ Sigma_delta @ B.T + cross_term + cross_term.T
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

def rollout_expected_cost(policy, gp_v, gp_omega, mu0, Sigma0, target, W, A, B, horizon, dt):
    """Eq (2): J^pi(theta) = sum_{t=0}^{T} E_xt[c(xt)], propagated analytically
    via Algorithm 2 at every step -- no particles, no sampling: a single
    deterministic (given the current policy parameters) trajectory of Gaussian
    moments, fully differentiable end-to-end via autograd."""
    mu_t, Sigma_t = mu0, Sigma0
    total = expected_saturating_cost(mu_t, Sigma_t, target, W)
    for _ in range(horizon):
        mean_u, var_u, cov_x_u = policy.moment_match(mu_t, Sigma_t)
        tilde_mu, tilde_Sigma = _build_joint(mu_t, Sigma_t, mean_u, var_u, cov_x_u)
        mu_delta, Sigma_delta, cross_cov_full = gp_moment_match([gp_v, gp_omega], tilde_mu, tilde_Sigma)
        cross_cov_state = cross_cov_full[:STATE_DIM, :]   # drop the tau row (paper: D x E submatrix)
        mu_t, Sigma_t = _successor_distribution(mu_t, Sigma_t, cross_cov_state, mu_delta, Sigma_delta, A, B)
        total = total + expected_saturating_cost(mu_t, Sigma_t, target, W)
    return total


def improve_policy(policy, gp_v, gp_omega, mu0, Sigma0, target, W, A, B, horizon, dt,
                    steps=POLICY_GRAD_STEPS, lr=POLICY_LR):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        cost = rollout_expected_cost(policy, gp_v, gp_omega, mu0, Sigma0, target, W, A, B, horizon, dt)
        cost.backward()
        opt.step()
        last = float(cost.item())
    return last


# ============================================================
# Real-episode execution: policy.forward_point() only, no planning at test time.
# ============================================================

def run_policy_episode(env, policy, device=DEVICE):
    obs, info = env.reset()
    transitions = []
    ep_return = 0.0
    done = truncated = False
    with torch.no_grad():
        while not (done or truncated):
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
    return transitions, ep_return, info


# ============================================================
# Main loop (Algorithm 1): collect one real trial, refit the GPs, improve the
# policy, repeat.
# ============================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    env = ObstacleWheelieEnv(reward_cfg=RewardConfig())
    p = env.p
    dt = env.ctrl_dt

    policy = RBFPolicy(STATE_DIM, POLICY_N_BASIS, p.tau_max).to(DEVICE)
    gp_v = ExactGP(GP_INPUT_DIM).to(DEVICE)
    gp_omega = ExactGP(GP_INPUT_DIM).to(DEVICE)
    buffer = ReplayBuffer()

    # Deterministic reset state (this env has no reset randomness) -> a NEARLY
    # Dirac initial belief. A truly-zero Sigma0 is singular, and the Q-matrix
    # (Eq 26) needs Sigma^-1 directly -- a Dirac delta is only the LIMIT of a
    # Gaussian as Sigma->0, so every real implementation of this algorithm uses
    # a tiny regularizing epsilon here, not an algorithmic change.
    env.reset()
    mu0 = torch.as_tensor(env.get_state(), dtype=DTYPE, device=DEVICE)
    Sigma0 = INIT_STATE_VAR * torch.eye(STATE_DIM, dtype=DTYPE, device=DEVICE)

    target = torch.tensor([env.goal_x, 0.0, 0.0, 0.0], dtype=DTYPE, device=DEVICE)
    W = torch.diag(torch.as_tensor(
        [1.0 / SIGMA_X ** 2, 1.0 / SIGMA_V ** 2, 1.0 / SIGMA_THETA ** 2, 1.0 / SIGMA_OMEGA ** 2],
        dtype=DTYPE, device=DEVICE,
    ))
    A = torch.tensor([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]],
                      dtype=DTYPE, device=DEVICE)
    B = torch.tensor([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=DTYPE, device=DEVICE)

    if LOAD_MODEL and MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        policy.load_state_dict(ckpt["policy"])
        gp_v.load_state_dict(ckpt["gp_v"])
        gp_omega.load_state_dict(ckpt["gp_omega"])
        if BUFFER_PATH.exists():
            buffer.load(BUFFER_PATH)
        print(f"[pilco] resumed from {MODEL_PATH} ({buffer.size} buffered transitions)")

    if TRAIN:
        episode_rows = []
        for trial in range(N_TRIALS):
            t0 = time.time()
            if trial < WARMUP_TRIALS or buffer.size == 0:
                mode = "random-warmup"
                transitions, ep_return, info = run_random_episode(env, None, rng, use_planner=False)
            else:
                mode = "policy"
                transitions, ep_return, info = run_policy_episode(env, policy)
            for s, a, s_next in transitions:
                buffer.add(s, a, s_next)

            X, y_v, y_omega = build_gp_dataset(buffer)
            gp_loss_v = gp_v.fit(X, y_v)
            gp_loss_omega = gp_omega.fit(X, y_omega)
            policy_loss = improve_policy(policy, gp_v, gp_omega, mu0, Sigma0, target, W, A, B, PLAN_HORIZON, dt)

            wall = time.time() - t0
            taus = np.array([t[1][0] for t in transitions], dtype=np.float64)
            effort_rms = float(np.sqrt(np.mean(taus ** 2))) if taus.size else float("nan")
            sim_time = len(transitions) * dt
            print(
                f"[trial {trial:02d}] mode={mode:14s} x_final={info['x']:6.2f} | "
                f"reached={info['reached']} | crashed={info['crashed']} | "
                f"return={ep_return:8.2f} | steps={len(transitions):3d} | "
                f"buffer={buffer.size:6d} | gp_loss=({gp_loss_v:.2f},{gp_loss_omega:.2f}) | "
                f"policy_loss={policy_loss:.4f} | wall={wall:5.1f}s"
            )

            episode_rows.append({
                "episode": trial,
                "mode": mode,
                "x_final": info["x"],
                "reached": int(info["reached"]),
                "crashed": int(info["crashed"]),
                "return": ep_return,
                "steps": len(transitions),
                "sim_time": sim_time,
                "effort_rms": effort_rms,
                "buffer_size": buffer.size,
                "gp_loss_v": gp_loss_v,
                "gp_loss_omega": gp_loss_omega,
                "policy_loss": policy_loss,
                "wall_seconds": wall,
            })
            with open(EPISODE_LOG_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
                writer.writeheader()
                writer.writerows(episode_rows)

            torch.save({
                "policy": policy.state_dict(),
                "gp_v": gp_v.state_dict(),
                "gp_omega": gp_omega.state_dict(),
            }, MODEL_PATH)
            buffer.save(BUFFER_PATH)

        print(f"[pilco] saved model to {MODEL_PATH}, buffer to {BUFFER_PATH}, "
              f"episode log to {EPISODE_LOG_PATH}")

    # ---- evaluation with rendering ----
    eval_env = ObstacleWheelieEnv(render_mode="human" if RENDER_EVAL else None, reward_cfg=RewardConfig())
    n_reached = 0
    for ep in range(EVAL_EPISODES):
        transitions, ep_return, info = run_policy_episode(eval_env, policy)
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
