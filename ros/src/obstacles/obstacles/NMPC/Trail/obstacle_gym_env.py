#!/usr/bin/env python3
"""Gymnasium wrapper around the same MuJoCo wheelie-truck plant used by
obstacle_mujoco_simulation*.py, so RL baselines (SAC, PPO, PETS) can be
benchmarked against the MPPI/NMPC controllers on the identical task: drive
forward over the box obstacle (x in [1.7, 2.3]) and reach x = GOAL_X without
flipping.

Shared by obstacle_mujoco_sac.py, obstacle_mujoco_ppo.py and
obstacle_mujoco_PETS.py so the MuJoCo plumbing (joint/sensor addresses,
episode reset, reward shaping) lives in one place instead of being
duplicated three times.

Observation (Box(5,), float32) -- everything a real IMU + odometry stack
would give you, no privileged obstacle position:
    [ (goal_x - x) / goal_x,   # remaining distance, signed fraction of goal_x
      v / v_scale,             # forward speed
      theta / theta_scale,     # pitch (rad); negative = wheelie, positive = nose-down
      omega / omega_scale,     # pitch rate
      tau_prev / tau_max ]     # last applied torque (makes the smoothness cost Markovian)

Action (Box(1,), float32) in [-1, 1], linearly mapped to [tau_min, tau_max].
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_ROOT = Path(__file__).resolve().parent
_MPPI = str(_ROOT / "mppi")
if _MPPI not in sys.path:
    sys.path.insert(0, _MPPI)

from params_mppi import WheelieParams  # noqa: E402  (physical params: tau_min/max, etc.)

XML_PATH = _ROOT / "monster_truck_flip_2d.xml"
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CTRL_DT = 0.05          # control period [s], matches the MPPI/NMPC scripts
SIM_TIME = 20.0         # per-episode time cap [s]
INIT_Z = 0.1512         # spawn height
GOAL_X = 10.0           # forward-x goal
INITIAL_X = 0.0
INITIAL_ROOT_PITCH_DEG = 0.0
OBSTACLE_SPAN = (1.7, 2.3)   # x-span of the box obstacle (for plotting/inspection only)


# ------------------------------------------------------------------
# Reward shaping -- written with (+abs) so the SAME code works whether
# called with python floats, numpy arrays, or torch tensors (no numpy/torch
# branching): this lets obstacle_mujoco_PETS.py reuse it directly inside a
# batched, GPU-vectorized imagined rollout.
# ------------------------------------------------------------------

def _relu(z):
    return (z + abs(z)) * 0.5


@dataclass
class RewardConfig:
    w_progress: float = 8.0          # reward per meter of forward progress this control step
    w_pitch: float = 0.02            # penalty weight on pitch-excess^2 (degrees^2)
    wheelie_soft_limit_deg: float = 75.0   # same knee as the MPPI stage cost
    w_tau: float = 0.02              # torque-effort penalty (normalized)
    w_dtau: float = 0.05             # torque-rate penalty (normalized)
    w_time: float = 0.3              # per-second time penalty (encourages speed)
    goal_bonus: float = 100.0        # one-off bonus on reaching goal_x
    crash_penalty: float = -35.0    # one-off penalty on flip/crash
    crash_wheelie_deg: float = -105.0   # beyond the MPPI's soft limit -> unrecoverable backflip
    crash_nosedive_deg: float = 30.0    # nose pitched forward past level -> crashed


def compute_step_reward(dx, v, theta, omega, tau, tau_prev, dt, cfg, tau_min, tau_max):
    """Dense per-control-step reward. dx is the forward progress made THIS step
    (x_new - x_old); v, theta, omega are the state AFTER the step; tau/tau_prev
    are the torque applied this step / the step before. Works elementwise on
    python floats, numpy arrays or torch tensors."""
    theta_deg = theta * (180.0 / math.pi)
    too_much_wheelie = _relu((-theta_deg) - cfg.wheelie_soft_limit_deg)
    nose_down = _relu(theta_deg)

    tau_mid = 0.5 * (tau_max + tau_min)
    tau_half = 0.5 * (tau_max - tau_min)
    tau_n = (tau - tau_mid) / tau_half
    dtau_n = (tau - tau_prev) / tau_half

    return (
        cfg.w_progress * dx
        - cfg.w_pitch * (too_much_wheelie ** 2 + nose_down ** 2)
        #- cfg.w_tau * tau_n ** 2
        - cfg.w_dtau * dtau_n ** 2
        #- cfg.w_time * dt
        #+ 0.0 * omega 
    )


def is_crashed(theta_deg, cfg):
    """True once the pitch leaves the recoverable envelope. Elementwise-safe
    (python bool / numpy bool array / torch bool tensor all support `|`)."""
    return (theta_deg <= cfg.crash_wheelie_deg) | (theta_deg >= cfg.crash_nosedive_deg)


# ============================================================
# ObstacleWheelieEnv
# ============================================================

class ObstacleWheelieEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": round(1.0 / CTRL_DT)}

    def __init__(self, render_mode=None, goal_x=GOAL_X, sim_time=SIM_TIME,
                 ctrl_dt=CTRL_DT, reward_cfg: RewardConfig = None):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.x_q, self.x_v = self._joint("root_x")
        self.z_q, self.z_v = self._joint("root_z")
        self.pitch_q, self.pitch_v = self._joint("root_pitch")
        self.quat_adr = self._sensor("imu_quat")
        self.gyro_adr = self._sensor("imu_gyro")
        self.drive = self._actuator("drive_motor")

        self.sim_dt = float(self.model.opt.timestep)
        self.steps_per_ctrl = max(1, round(ctrl_dt / self.sim_dt))
        self.ctrl_dt = self.steps_per_ctrl * self.sim_dt

        self.p = WheelieParams()
        self.reward_cfg = reward_cfg if reward_cfg is not None else RewardConfig()
        self.goal_x = float(goal_x)
        self.sim_time = float(sim_time)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_high = np.array([3.0, 3.0, 3.0, 3.0, 1.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._viewer = None
        self._tau_prev = 0.0
        self._t = 0.0
        self._x_prev = 0.0

    # --- MuJoCo address helpers (same pattern as ObstacleNode) ---
    def _joint(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {name}")
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise RuntimeError(f"Sensor not found: {name}")
        return int(self.model.sensor_adr[sid])

    def _actuator(self, name):
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"Actuator not found: {name}")
        return int(aid)

    # --- state readout ---
    def _read_state(self):
        """Raw physical state [x, v, theta, omega] -- theta/omega from the IMU,
        same convention as the MPPI/NMPC controllers' state vector."""
        w = float(self.data.sensordata[self.quat_adr + 0])
        x_ = float(self.data.sensordata[self.quat_adr + 1])
        y_ = float(self.data.sensordata[self.quat_adr + 2])
        z_ = float(self.data.sensordata[self.quat_adr + 3])
        theta = math.atan2(2.0 * (x_ * z_ + w * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
        omega = float(self.data.sensordata[self.gyro_adr + 1])
        x = float(self.data.qpos[self.x_q])
        v = float(self.data.qvel[self.x_v])
        return x, v, theta, omega

    def get_state(self) -> np.ndarray:
        """Public accessor for the raw [x, v, theta, omega] state -- used by
        model-based planners (PETS) that need the physical state directly,
        bypassing the normalized policy observation below."""
        return np.array(self._read_state(), dtype=np.float64)

    def _obs(self, state=None):
        x, v, theta, omega = state if state is not None else self._read_state()
        obs = np.array([
            (self.goal_x - x) / self.goal_x,
            v / 5.0,
            theta / math.radians(100.0),
            omega / 5.0,
            self._tau_prev / self.p.tau_max,
        ], dtype=np.float32)
        return obs

    # --- Gym API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.x_q] = INITIAL_X
        self.data.qpos[self.z_q] = INIT_Z
        self.data.qpos[self.pitch_q] = math.radians(INITIAL_ROOT_PITCH_DEG)
        self.data.ctrl[self.drive] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._tau_prev = 0.0
        self._t = 0.0
        self._tau_sq_sum = 0.0     # for effort_rms: RMS torque over the episode so far
        self._n_ctrl_steps = 0
        x, v, theta, omega = self._read_state()
        self._x_prev = x

        if self.render_mode == "human":
            self._ensure_viewer()

        obs = self._obs((x, v, theta, omega))
        info = {"x": x, "v": v, "theta_deg": math.degrees(theta), "omega": omega}
        return obs, info

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        tau_min, tau_max = self.p.tau_min, self.p.tau_max
        tau = tau_min + (a + 1.0) * 0.5 * (tau_max - tau_min)

        for _ in range(self.steps_per_ctrl):
            self.data.ctrl[self.drive] = tau
            mujoco.mj_step(self.model, self.data)

        self._t += self.ctrl_dt
        self._tau_sq_sum += tau * tau
        self._n_ctrl_steps += 1
        effort_rms = math.sqrt(self._tau_sq_sum / self._n_ctrl_steps)
        x, v, theta, omega = self._read_state()
        dx = x - self._x_prev
        theta_deg = math.degrees(theta)

        reward = compute_step_reward(
            dx, v, theta, omega, tau, self._tau_prev, self.ctrl_dt,
            self.reward_cfg, tau_min, tau_max,
        )
        crashed = bool(is_crashed(theta_deg, self.reward_cfg))
        reached = bool(x >= self.goal_x)
        if reached:
            reward += self.reward_cfg.goal_bonus
        if crashed:
            reward += self.reward_cfg.crash_penalty

        terminated = bool(crashed or reached)
        truncated = bool(self._t >= self.sim_time)

        self._x_prev = x
        self._tau_prev = tau

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        obs = self._obs((x, v, theta, omega))
        info = {
            "x": x, "v": v, "theta_deg": theta_deg, "omega": omega, "tau": tau,
            "reached": reached, "crashed": crashed, "t": self._t,
            # Named "sim_time" (not "t") so it doesn't collide with SB3 Monitor's own
            # "t" column (wall-clock training time) when logged via info_keywords.
            "sim_time": self._t, "effort_rms": effort_rms,
        }
        return obs, float(reward), terminated, truncated, info

    def _ensure_viewer(self):
        if self._viewer is None:
            import mujoco.viewer as mj_viewer
            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    def render(self):
        if self.render_mode == "human":
            self._ensure_viewer()
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ------------------------------------------------------------------
# Self-test: random rollout + Gym API compliance check.
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = ObstacleWheelieEnv()
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True)
        print("[obstacle_gym_env] check_env passed")
    except Exception as exc:
        print(f"[obstacle_gym_env] check_env skipped/failed: {exc}")

    obs, info = env.reset()
    ep_ret = 0.0
    for k in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += reward
        if terminated or truncated:
            print(f"[random rollout] done at k={k} x={info['x']:.2f} "
                  f"reached={info['reached']} crashed={info['crashed']} return={ep_ret:.2f}")
            break
    else:
        print(f"[random rollout] ran 200 steps without terminating, x={info['x']:.2f} return={ep_ret:.2f}")
    env.close()
