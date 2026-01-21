#!/usr/bin/env python3
"""
mppi_residual_all_in_one.py

Modes:
  1) --mode run
     Runs the ROS2 MPPI controller and applies a learned residual (TorchScript) near target.

  2) --mode train_residual
     Trains a residual policy using SB3 SAC in a fast GP-simulated gymnasium env and exports:
       - models/residual_sac.zip
       - models/residual_policy.pt  (TorchScript, loadable by the ROS node)

Notes:
  - The residual policy output is in [-1, 1] and is scaled by cfg.residual_max_delta to produce Δu.
  - The residual is gated on near-target conditions (|err| and |rate|) to avoid interfering with flip-up.

SB3 export pattern is aligned with SB3 docs (actor.latent_pi -> actor.mu -> Tanh)
and Torch JIT trace/freeze/optimize workflow. See SB3 "Exporting models".  (docs)
"""

import os
import math
import time
import csv
import argparse
import threading
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# ROS imports (only needed in --mode run)
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

from collections import deque

from gp_dynamics import GPManager
from re_train_dynamics_gp import train_dynamics_gp_from_arrays


# =========================
# Small utilities
# =========================
def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def smooth_gate(val: float, on_thresh: float, k: float) -> float:
    """
    gate ~ 1 when val < on_thresh, ~0 when val > on_thresh
    """
    return 1.0 / (1.0 + math.exp(k * (val - on_thresh)))


# Optional fallback architecture (only used if residual_path is a state_dict, not TorchScript)
class ResidualMLP(nn.Module):
    def __init__(self, in_dim=4, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
            nn.Linear(hid, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Config
# =========================
@dataclass
class MPPIConfig:
    # Timing
    ctrl_dt: float = 0.1
    horizon: int = 20
    num_rollouts: int = 2000

    # MPPI hyper-parameters
    lambda_: float = 1.0
    sigma: float = 1.6

    # Action bounds
    u_min: float = -1.0
    u_max: float = 1.0

    # Target / stop conditions
    pitch_target: float = math.pi/2.0 #math.pi
    flip_stop_abs: float = 3.1

    # Paths to trained GP models
    gp_flip_path: str = "models/gp_dynamics_0.pt"
    gp_rate_path: str = "models/gp_dynamics_1.pt"

    # ---- logging ----
    log_dir: str = "logs"
    max_log_points: int = 200_000

    # ---- retrain ----
    min_points_to_train: int = 2_000
    N_target_train: int = 1_000
    train_kernel: str = "RQ"
    train_iters: int = 300
    max_points_for_train: int = 50_000
    min_new_points_between_trains: int = 500
    retrain_every_episodes: int = 10

    # ---- episode control ----
    episode_timeout_sec: float = 20.0

    # ---- entropy exploration (optional) ----
    entropy_beta: float = 0.0
    entropy_use_log: bool = True
    entropy_var_floor: float = 1e-6
    entropy_var_cap: float = 1e2
    entropy_dt_scale: bool = True

    # ---- seed dataset ----
    seed_npz_path: str = "data/mujoco_random_run_dt0p1.npz"
    seed_episode_id: int = -1
    keep_seed: bool = True

    # ---- residual RL overlay ----
    residual_enabled: bool = True
    residual_path: str = "models/residual_policy.pt"   # TorchScript preferred
    residual_max_delta: float = 0.30                   # |Δu| max (scaled by policy output)
    residual_err_on: float = 0.60                      # rad: turn on near target
    residual_rate_on: float = 3.0                      # rad/s
    residual_gate_k: float = 8.0
    residual_dudt_max: float = 10.0                    # max Δu rate (per sec)

    # Baseline feature for residual input (in ROS + training env)
    residual_kp_base: float = 0.0
    residual_kd_base: float = 0.0


# ============================================================
# Training: SB3 SAC on GP-simulated environment (fast)
# ============================================================
def export_sac_actor_to_torchscript(model, out_path: str):
    """
    Export only the actor as a TorchScript module:
      actor.latent_pi -> actor.mu -> Tanh
    This matches the SB3 export doc pattern for SAC actor networks. (SB3 docs)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    actor = model.policy.actor
    # Wrap like SB3 doc suggests for export: latent_pi -> mu -> tanh
    # (SB3 docs show this for ONNX; same module is traceable for TorchScript.)
    onnxable_actor = torch.nn.Sequential(
        actor.latent_pi,
        actor.mu,
        torch.nn.Tanh(),
    ).eval()

    obs_shape = model.observation_space.shape
    dummy = torch.randn(1, *obs_shape)

    traced = torch.jit.trace(onnxable_actor, dummy)
    frozen = torch.jit.freeze(traced)
    frozen = torch.jit.optimize_for_inference(frozen)
    torch.jit.save(frozen, out_path)


class ResidualBalanceEnv:
    """
    Gymnasium-like API (wrapped later by SB3 DummyVecEnv):
      state = [pitch, rate]
      obs   = [sin(err), cos(err), rate, u_base]

    Dynamics are simulated using your GP models:
      next_pitch = pitch + dt * GP_flip(pitch, rate, u)
      next_rate  = rate  + dt * GP_rate(pitch, rate, u)

    The agent action is residual a in [-1,1], mapped to Δu = a * residual_max_delta.
    The residual is gated near target (same idea as in ROS).
    """

    def __init__(self, cfg: MPPIConfig, device: str = "cpu",
                 max_steps: int = 200, init_err_range: float = 0.8, init_rate_range: float = 3.0):
        import gymnasium as gym
        from gymnasium import spaces

        self.cfg = cfg
        self.dt = float(cfg.ctrl_dt)
        self.max_steps = int(max_steps)
        self.init_err_range = float(init_err_range)
        self.init_rate_range = float(init_rate_range)

        self.device = torch.device(device)

        self.gp_flip = GPManager.load(cfg.gp_flip_path)
        self.gp_rate = GPManager.load(cfg.gp_rate_path)
        self.gp_flip.device = self.device
        self.gp_rate.device = self.device

        # spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._step = 0
        self.pitch = 0.0
        self.rate = 0.0

        # success criteria
        self.err_tol = 0.12   # rad
        self.rate_tol = 0.8   # rad/s
        self.hold_needed = 10
        self.hold_count = 0

    def _u_base(self, err: float) -> float:
        # optional baseline term (default 0,0)
        u = self.cfg.residual_kp_base * err + self.cfg.residual_kd_base * self.rate
        return float(np.clip(u, self.cfg.u_min, self.cfg.u_max))

    def _obs(self) -> np.ndarray:
        err = wrap_to_pi(self.pitch - self.cfg.pitch_target)
        u_base = self._u_base(err)
        return np.array([math.sin(err), math.cos(err), self.rate, u_base], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # initialize near target (stabilization-focused)
        err0 = np.random.uniform(-self.init_err_range, self.init_err_range)
        self.pitch = wrap_to_pi(self.cfg.pitch_target + float(err0))
        self.rate = float(np.random.uniform(-self.init_rate_range, self.init_rate_range))

        self._step = 0
        self.hold_count = 0
        return self._obs(), {}

    @torch.no_grad()
    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))

        err = wrap_to_pi(self.pitch - self.cfg.pitch_target)
        u_base = self._u_base(err)

        # gate residual near target (same concept as ROS)
        g_err = smooth_gate(abs(err), self.cfg.residual_err_on, self.cfg.residual_gate_k)
        g_rate = smooth_gate(abs(self.rate), self.cfg.residual_rate_on, self.cfg.residual_gate_k)
        g = g_err * g_rate

        du = a * float(self.cfg.residual_max_delta) * float(g)
        u = float(np.clip(u_base + du, self.cfg.u_min, self.cfg.u_max))

        # GP dynamics
        X = torch.tensor([[self.pitch, self.rate, u]], dtype=torch.float32, device=self.device)
        d_pitch, _ = self.gp_flip.predict_torch(X)
        d_rate, _ = self.gp_rate.predict_torch(X)

        self.pitch = wrap_to_pi(self.pitch + float(d_pitch.item()) * self.dt)
        self.rate = float(np.clip(self.rate + float(d_rate.item()) * self.dt, -20.0, 20.0))

        self._step += 1

        # reward: encourage holding at target
        err2 = wrap_to_pi(self.pitch - self.cfg.pitch_target)
        # smooth near pi via (1 - cos(err))
        r_theta = -2.0 * (1.0 - math.cos(err2))
        r_rate = -0.05 * (self.rate ** 2)
        r_u = -0.02 * (u ** 2)
        r_du = -0.10 * (du ** 2)
        reward = float(r_theta + r_rate + r_u + r_du)

        # success bonus if inside tolerance
        inside = (abs(err2) < self.err_tol) and (abs(self.rate) < self.rate_tol)
        if inside:
            self.hold_count += 1
            reward += 0.5
        else:
            self.hold_count = 0

        terminated = (self.hold_count >= self.hold_needed)
        truncated = (self._step >= self.max_steps)

        return self._obs(), reward, terminated, truncated, {"err": err2, "g": g, "u": u, "du": du}


def train_residual_sac(cfg: MPPIConfig, timesteps: int, device: str = "cuda"):
    """
    SB3 SAC training that exports:
      - models/residual_sac.zip
      - cfg.residual_path  (TorchScript actor)
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

    os.makedirs("models", exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    def make_env():
        return ResidualBalanceEnv(cfg, device="cpu", max_steps=220, init_err_range=1.0, init_rate_range=4.0)

    vec_env = DummyVecEnv([make_env])
    vec_env = VecMonitor(vec_env, filename=os.path.join(cfg.log_dir, "residual_monitor.csv"))

    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=nn.Tanh,
    )

    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(cfg.log_dir, "tb_residual_sac"),
    )

    model.learn(total_timesteps=int(timesteps), progress_bar=True)

    zip_path = os.path.join("models", "residual_sac.zip")
    model.save(zip_path)

    # Export TorchScript actor (loadable by ROS node)
    export_sac_actor_to_torchscript(model, cfg.residual_path)

    print(f"[OK] Saved SB3 model: {zip_path}")
    print(f"[OK] Saved TorchScript residual actor: {cfg.residual_path}")


# ============================================================
# ROS2 MPPI Controller Node (with residual overlay)
# ============================================================
class MPPICarControllerNode(Node):
    def __init__(self, cfg: MPPIConfig):
        super().__init__("mppi_car_controller")
        self.cfg = cfg

        # device robust
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using torch device: {self.device}")

        # Load GP models
        self.gp_flip: GPManager = GPManager.load(self.cfg.gp_flip_path)
        self.gp_rate: GPManager = GPManager.load(self.cfg.gp_rate_path)
        self.gp_flip.device = self.device
        self.gp_rate.device = self.device

        self.pitch_target_t = torch.tensor(self.cfg.pitch_target, dtype=torch.float32, device=self.device)

        # ROS interfaces
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)
        self.imu_sub = self.create_subscription(Imu, "car_imu", self.imu_cb, 10)

        self.reset_client = self.create_client(Trigger, "reset_car")
        self.resetting = False
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for reset_car service...")

        # State from IMU
        self.last_flip_rel: float = 0.0
        self.last_rate: float = 0.0
        self.last_state_valid: bool = False

        self.prev_theta: Optional[float] = None
        self.prev_theta_unwrapped: float = 0.0
        self.theta0: Optional[float] = None

        self.plan: Optional[torch.Tensor] = None
        self.last_u: float = 0.0

        # Reset logic
        self.waiting_post_reset = False
        self.post_reset_start_time = None
        self.warned_no_imu = False
        self.watchdog_fired = False

        # dataset buffer
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        self.episode_id: int = 0
        self.log_flip = deque(maxlen=self.cfg.max_log_points)
        self.log_rate = deque(maxlen=self.cfg.max_log_points)
        self.log_u    = deque(maxlen=self.cfg.max_log_points)
        self.log_ep   = deque(maxlen=self.cfg.max_log_points)

        # retrain state
        self.training: bool = False
        self.reload_pending: bool = False
        self.model_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.train_thread: Optional[threading.Thread] = None
        self.reset_after_retrain = False
        self.last_train_size = 0

        # episode timing
        self.episode_start_time = None
        self.episode_started = False
        self.metrics_path = os.path.join(self.cfg.log_dir, "episode_metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["episode", "flip_time_sec", "retrain_started"])

        # residual
        self.residual_policy = None
        self.last_du = 0.0

        if self.cfg.residual_enabled and os.path.exists(self.cfg.residual_path):
            try:
                self.residual_policy = torch.jit.load(self.cfg.residual_path, map_location=self.device)
                self.residual_policy.eval()
                self.get_logger().info(f"Loaded residual policy (TorchScript): {self.cfg.residual_path}")
            except Exception:
                # fallback state_dict
                self.residual_policy = ResidualMLP(in_dim=4, hid=64).to(self.device)
                self.residual_policy.load_state_dict(torch.load(self.cfg.residual_path, map_location=self.device))
                self.residual_policy.eval()
                self.get_logger().info(f"Loaded residual policy state_dict: {self.cfg.residual_path}")
        else:
            self.get_logger().info("Residual policy disabled or file not found.")

        # timer
        self.timer = self.create_timer(self.cfg.ctrl_dt, self.control_timer_cb)
        self.get_logger().info("MPPI Car Controller node initialized.")

    # ------------ live utils ------------
    def _mark_episode_started(self):
        if not self.episode_started:
            self.episode_start_time = self.get_clock().now()
            self.episode_started = True

    # ------------ quaternion helpers ------------
    @staticmethod
    def quat_to_R_and_pitch(qw, qx, qy, qz):
        R00 = 1 - 2 * (qy * qy + qz * qz)
        R01 = 2 * (qx * qy - qw * qz)
        R02 = 2 * (qx * qz + qw * qy)
        R10 = 2 * (qx * qy + qw * qz)
        R11 = 1 - 2 * (qx * qx + qz * qz)
        R12 = 2 * (qy * qz - qw * qx)
        R20 = 2 * (qx * qz - qw * qy)
        R21 = 2 * (qy * qz + qw * qx)
        R22 = 1 - 2 * (qx * qx + qy * qy)
        pitch = -math.asin(max(-1.0, min(1.0, R20)))
        R = np.array([[R00, R01, R02],
                      [R10, R11, R12],
                      [R20, R21, R22]], dtype=float)
        return R, pitch

    @staticmethod
    def unwrap_angle(prev_angle, prev_unwrapped, angle):
        if prev_angle is None:
            return angle, angle
        d = angle - prev_angle
        if d > math.pi:
            angle_unwrapped = prev_unwrapped + (d - 2 * math.pi)
        elif d < -math.pi:
            angle_unwrapped = prev_unwrapped + (d + 2 * math.pi)
        else:
            angle_unwrapped = prev_unwrapped + d
        return angle, angle_unwrapped

    def imu_cb(self, msg: Imu):
        qw = float(msg.orientation.w)
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)

        R, _ = self.quat_to_R_and_pitch(qw, qx, qy, qz)
        up_z = R[2, 2]
        theta = math.atan2(R[0, 2], R[2, 2])
        pitch_rate = float(msg.angular_velocity.y)

        if self.waiting_post_reset:
            if self.resetting:
                self.last_state_valid = False
                return

            if up_z > -0.8:
                self.last_state_valid = False
                return

            self.prev_theta = theta
            self.prev_theta_unwrapped = theta
            self.theta0 = theta

            self.last_flip_rel = 0.0
            self.last_rate = pitch_rate
            self.last_state_valid = True

            self.waiting_post_reset = False
            self.watchdog_fired = False
            return

        self.prev_theta, theta_unwrapped = self.unwrap_angle(self.prev_theta, self.prev_theta_unwrapped, theta)
        self.prev_theta_unwrapped = theta_unwrapped

        if self.theta0 is None:
            self.theta0 = theta_unwrapped

        flip_rel = theta_unwrapped - self.theta0
        flip_rel = max(-math.pi, min(math.pi, flip_rel))

        self.last_flip_rel = flip_rel
        self.last_rate = pitch_rate
        self.last_state_valid = True

    # ------------ torch helpers ------------
    @staticmethod
    def angdiff_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.remainder(a - b + torch.pi, 2 * torch.pi) - torch.pi

    def stage_cost_torch(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pitch = states[:, 0]
        rate = states[:, 1]
        u = actions
        err = self.angdiff_torch(pitch, self.pitch_target_t)

        cost_pitch = 100.0 * err ** 2
        cost_u = 0.01 * u ** 2
        cost_rate = 2.0 * rate ** 2
        return cost_pitch + cost_rate + cost_u

    def gp_step_batch_torch(self, states: torch.Tensor, actions: torch.Tensor):
        X = torch.stack([states[:, 0], states[:, 1], actions], dim=-1)

        with self.model_lock:
            if self.cfg.entropy_beta <= 0.0:
                d_flip_mean, _ = self.gp_flip.predict_torch(X)
                d_rate_mean, _ = self.gp_rate.predict_torch(X)
                d_flip_var = None
                d_rate_var = None
            else:
                d_flip_mean, d_flip_var = self.gp_flip.predict_torch(X)
                d_rate_mean, d_rate_var = self.gp_rate.predict_torch(X)

        dt = float(self.cfg.ctrl_dt)
        next_states = torch.empty_like(states)
        next_states[:, 0] = states[:, 0] + d_flip_mean * dt
        next_states[:, 1] = states[:, 1] + d_rate_mean * dt
        next_states[:, 0].clamp_(-math.pi, math.pi)
        next_states[:, 1].clamp_(-20.0, 20.0)

        if self.cfg.entropy_beta <= 0.0:
            return next_states, None

        d_flip_var = torch.clamp(d_flip_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)
        d_rate_var = torch.clamp(d_rate_var, min=self.cfg.entropy_var_floor, max=self.cfg.entropy_var_cap)

        if self.cfg.entropy_dt_scale:
            var_next = torch.stack([d_flip_var, d_rate_var], dim=-1) * (dt * dt)
        else:
            var_next = torch.stack([d_flip_var, d_rate_var], dim=-1)

        if self.cfg.entropy_use_log:
            entropy = 0.5 * torch.log(var_next).sum(dim=-1)
        else:
            entropy = var_next.sum(dim=-1)

        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        return next_states, entropy

    # ------------ MPPI ------------
    @torch.no_grad()
    def mppi_action(self, x0_np):
        cfg = self.cfg
        H = cfg.horizon
        K = cfg.num_rollouts

        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device)
        u_init = torch.zeros(H, dtype=torch.float32, device=self.device) if self.plan is None else self.plan

        eps = torch.randn(K, H, device=self.device) * float(cfg.sigma)
        U = torch.clamp(u_init.unsqueeze(0) + eps, float(cfg.u_min), float(cfg.u_max))

        states = x0.unsqueeze(0).repeat(K, 1)
        costs = torch.zeros(K, dtype=torch.float32, device=self.device)

        beta = float(cfg.entropy_beta)
        for t in range(H):
            u_t = U[:, t]
            stage = self.stage_cost_torch(states, u_t)
            states, ent = self.gp_step_batch_torch(states, u_t)
            if ent is not None:
                stage = stage - beta * ent
            costs = costs + stage

        J_min = costs.min()
        weights = torch.exp(-(costs - J_min) / float(cfg.lambda_))
        weights_sum = weights.sum() + 1e-8

        du = (weights.unsqueeze(1) * eps).sum(dim=0) / weights_sum
        u_new = torch.clamp(u_init + du, float(cfg.u_min), float(cfg.u_max))

        # receding horizon shift
        self.plan = torch.cat([u_new[1:], u_new[-1:]], dim=0).detach()
        return float(u_new[0].detach().cpu())

    # ------------ control loop ------------
    def control_timer_cb(self):
        if self.training or self.reload_pending:
            self.plan = None
            self.publish_u(0.0)
            self._reload_models_if_ready()
            return

        if self.waiting_post_reset and (not self.resetting) and (self.post_reset_start_time is not None):
            elapsed = (self.get_clock().now() - self.post_reset_start_time).nanoseconds * 1e-9
            if (elapsed > 3.0) and (not self.watchdog_fired):
                self.watchdog_fired = True
                self.get_logger().warn("Stuck in waiting_post_reset. Forcing reset retry.")
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        if self.resetting or self.waiting_post_reset:
            self.publish_u(0.0)
            return

        if not self.last_state_valid:
            if not self.warned_no_imu:
                self.get_logger().warn("Waiting for first IMU message...")
                self.warned_no_imu = True
            self.publish_u(0.0)
            return
        self.warned_no_imu = False

        flip_rel = self.last_flip_rel
        rate = self.last_rate

        if self.episode_start_time is not None:
            elapsed_ep = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
            if elapsed_ep >= float(self.cfg.episode_timeout_sec):
                self.get_logger().warn(
                    f"Episode {int(self.episode_id)} TIMEOUT after {elapsed_ep:.2f}s "
                    f"(limit={self.cfg.episode_timeout_sec:.2f}s). Forcing reset."
                )
                self._record_episode_metric(retrain_started=False)
                self.publish_u(0.0)
                self.request_reset(force=True)
                return

        if abs(flip_rel) >= self.cfg.flip_stop_abs:
            self.publish_u(0.0)

            ep_num = self.episode_id + 1
            do_retrain_now = (self.cfg.retrain_every_episodes > 0) and (ep_num % self.cfg.retrain_every_episodes == 0)

            started = False
            if do_retrain_now:
                started = self._start_retrain_async(force=True)
            self._record_episode_metric(retrain_started=started)

            if started:
                self.reset_after_retrain = True
                return
            self.reset_after_retrain = False
            self.request_reset()
            return

        # base MPPI
        x0 = np.array([flip_rel, rate], dtype=np.float32)
        try:
            u_base = self.mppi_action(x0)
        except Exception as e:
            self.get_logger().error(f"MPPI error: {e}")
            u_base = 0.0

        # residual correction
        du = 0.0
        if self.residual_policy is not None:
            err = wrap_to_pi(float(flip_rel) - float(self.cfg.pitch_target))  # FIXED
            g_err = smooth_gate(abs(err), self.cfg.residual_err_on, self.cfg.residual_gate_k)
            g_rate = smooth_gate(abs(float(rate)), self.cfg.residual_rate_on, self.cfg.residual_gate_k)
            g = g_err * g_rate

            if g > 1e-3:
                obs = torch.tensor([math.sin(err), math.cos(err), float(rate), float(u_base)],
                                   dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    du_raw = float(self.residual_policy(obs).squeeze().clamp(-1.0, 1.0).item())

                du = du_raw * float(self.cfg.residual_max_delta) * float(g)

                max_step = float(self.cfg.residual_dudt_max) * float(self.cfg.ctrl_dt)
                du = float(np.clip(du, self.last_du - max_step, self.last_du + max_step))
                self.last_du = du
            else:
                self.last_du = 0.0

        u_cmd = float(np.clip(float(u_base) + float(du), self.cfg.u_min, self.cfg.u_max))

        # mark episode start exactly when we send first action
        self._mark_episode_started()

        # PUBLISH ONCE (FIXED)
        self.publish_u(u_cmd)
        self._log_step(flip_rel, rate, u_cmd)

    def publish_u(self, u: float):
        msg = Float32()
        msg.data = float(u)
        self.cmd_pub.publish(msg)
        self.last_u = float(u)

    # ------------ logging / dataset ------------
    def _log_step(self, flip_rel: float, rate: float, u: float):
        with self.log_lock:
            self.log_flip.append(float(flip_rel))
            self.log_rate.append(float(rate))
            self.log_u.append(float(u))
            self.log_ep.append(int(self.episode_id))

    def _snapshot_dataset(self):
        with self.log_lock:
            flip = np.asarray(list(self.log_flip), dtype=np.float32)
            rate = np.asarray(list(self.log_rate), dtype=np.float32)
            u    = np.asarray(list(self.log_u), dtype=np.float32)
            ep   = np.asarray(list(self.log_ep), dtype=np.int64)
        return flip, rate, u, ep

    def _save_dataset_npz(self, flip, rate, u, ep):
        out = os.path.join(self.cfg.log_dir, f"dataset_ep{self.episode_id:04d}.npz")
        np.savez_compressed(out, flip=flip, rate=rate, u=u, episode_id=ep,
                            dt=np.array(self.cfg.ctrl_dt, dtype=np.float32))
        self.get_logger().info(f"Saved dataset snapshot: {out}")

    # ------------ retraining ------------
    def _start_retrain_async(self, force: bool = False) -> bool:
        if self.training:
            self.get_logger().info("Retrain requested but training is already running; skipping.")
            return False

        with self.log_lock:
            n = len(self.log_flip)

        if not force:
            if n < self.cfg.min_points_to_train:
                self.get_logger().info(f"Not enough data to retrain yet: {n} < {self.cfg.min_points_to_train}")
                return False
            if (n - self.last_train_size) < self.cfg.min_new_points_between_trains:
                self.get_logger().info("Not enough new data since last train; skipping.")
                return False

        flip, rate, u, ep = self._snapshot_dataset()
        self._save_dataset_npz(flip, rate, u, ep)

        M = self.cfg.max_points_for_train
        if len(flip) > M:
            flip = flip[-M:]
            rate = rate[-M:]
            u    = u[-M:]
            ep   = ep[-M:]

        self.training = True
        n_at_start = n

        self.train_thread = threading.Thread(
            target=self._train_worker,
            args=(flip, rate, u, ep, n_at_start),
            daemon=True,
        )
        self.train_thread.start()
        self.get_logger().info("Started GP retraining thread.")
        return True

    def _train_worker(self, flip, rate, u, ep, n_at_start: int):
        t0 = time.perf_counter()
        try:
            gps, _, _ = train_dynamics_gp_from_arrays(
                flip_arr=flip,
                rate_arr=rate,
                u_arr=u,
                dt=self.cfg.ctrl_dt,
                episode_id=ep,
                N_target=self.cfg.N_target_train,
                kernel=self.cfg.train_kernel,
                iters=self.cfg.train_iters,
                seed_npz_path=self.cfg.seed_npz_path,
                seed_episode_id=self.cfg.seed_episode_id,
                keep_seed=self.cfg.keep_seed,
            )

            os.makedirs("models", exist_ok=True)
            for d, gp in enumerate(gps):
                tmp_path = f"models/gp_dynamics_{d}.pt.tmp"
                out_path = f"models/gp_dynamics_{d}.pt"
                gp.save(tmp_path)
                os.replace(tmp_path, out_path)

            self.last_train_size = n_at_start
            elapsed = time.perf_counter() - t0
            self.get_logger().info(f"GP retraining finished in {elapsed:.2f}s")
            self.reload_pending = True

        except Exception as e:
            elapsed = time.perf_counter() - t0
            self.get_logger().error(f"Retraining failed after {elapsed:.2f}s: {e}")
            self.get_logger().error(traceback.format_exc())
        finally:
            self.training = False

    def _reload_models_if_ready(self):
        if not self.reload_pending or self.training:
            return

        with self.model_lock:
            self.gp_flip = GPManager.load(self.cfg.gp_flip_path)
            self.gp_rate = GPManager.load(self.cfg.gp_rate_path)
            self.gp_flip.device = self.device
            self.gp_rate.device = self.device

        self.plan = None
        self.reload_pending = False
        self.get_logger().info("Reloaded GP models (hot swap).")

        if self.reset_after_retrain:
            self.reset_after_retrain = False
            self.request_reset()

    # ------------ episode metric recording ------------
    def _record_episode_metric(self, retrain_started: bool):
        if self.episode_start_time is None:
            self.episode_started = False
            return

        dt = (self.get_clock().now() - self.episode_start_time).nanoseconds * 1e-9
        ep = int(self.episode_id)

        self.get_logger().info(f"Episode {ep} flip time: {dt:.3f} s | retrain_started={int(retrain_started)}")

        with open(self.metrics_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, float(dt), int(retrain_started)])

        self.episode_start_time = None
        self.episode_started = False

    # ------------ reset logic ------------
    def _local_reset_state(self):
        self.prev_theta = None
        self.prev_theta_unwrapped = 0.0
        self.theta0 = None
        self.last_state_valid = False
        self.plan = None
        self.last_u = 0.0
        self.last_du = 0.0  # reset residual rate limiter
        self.episode_start_time = None
        self.episode_started = False

    def request_reset(self, force: bool = False):
        if self.resetting:
            return
        if self.waiting_post_reset and not force:
            return

        self.resetting = True
        self.waiting_post_reset = True
        self.post_reset_start_time = None
        self.watchdog_fired = False

        self.episode_id += 1
        self._local_reset_state()

        req = Trigger.Request()
        future = self.reset_client.call_async(req)

        def done_callback(f):
            try:
                resp = f.result()
                self.get_logger().info(f"Reset response: {resp.message}")
            except Exception as e:
                self.get_logger().warn(f"Reset service call failed: {e}")

            self.resetting = False
            self.post_reset_start_time = self.get_clock().now()
            self.watchdog_fired = False

        future.add_done_callback(done_callback)


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "train_residual"], default="run")
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--train_device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = MPPIConfig()

    if args.mode == "train_residual":
        train_residual_sac(cfg, timesteps=args.timesteps, device=args.train_device)
        return

    # run ROS node
    rclpy.init()
    node = MPPICarControllerNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down MPPI controller, sending u=0.0")
        node.publish_u(0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
