#!/usr/bin/env python3
"""PPO baseline on the obstacle wheelie-climb task (obstacle_gym_env.ObstacleWheelieEnv),
for comparison against the MPPI+CBF / NMPC controllers in this folder.

On-policy algorithms live and die by how fast they collect rollouts, so training runs
N_ENVS independent MuJoCo instances in separate OS processes (SubprocVecEnv) and
gathers N_STEPS transitions per env before each update -- N_ENVS * N_STEPS transitions
per PPO iteration. Evaluation afterward drops back to a single process so the viewer
can render.

Usage:
    python3 obstacle_mujoco_ppo.py            # train (headless, parallel) then render eval episodes
Toggle TRAIN / RENDER / LOAD_MODEL below to resume training, or to just watch a saved policy.
"""

from pathlib import Path
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from obstacle_gym_env import ObstacleWheelieEnv, RewardConfig, RESULTS_DIR

# ============================================================
# Settings
# ============================================================
TRAIN = True
LOAD_MODEL = False
RENDER_EVAL = True

N_ENVS = 50                    # parallel MuJoCo worker processes during training
N_STEPS = 1024                # rollout length per env per PPO update (N_ENVS*N_STEPS/update)
TOTAL_TIMESTEPS = 100_000_000
CHECKPOINT_FREQ = 25_000      # in total env steps across all workers
EVAL_EPISODES = 100
SEED = 0

MODEL_PATH = RESULTS_DIR / "ppo_obstacle.zip"
CHECKPOINT_DIR = RESULTS_DIR / "ppo_checkpoints"
TB_LOG = RESULTS_DIR / "tb_ppo"
# Per-episode training log (read by results/plot_results.py): one *.monitor.csv per
# worker, written by make_vec_env's monitor_dir below. load_results() there globs and
# time-sorts all of them into a single combined training curve.
MONITOR_DIR = RESULTS_DIR / "ppo_monitor"


def _env_factory():
    # NOT wrapped in Monitor here -- make_vec_env(monitor_dir=...) below does that
    # wrapping itself (with the filename/info_keywords it needs); double-wrapping
    # would nest two Monitors around the same env.
    return ObstacleWheelieEnv(reward_cfg=RewardConfig())


def train():
    vec_env = make_vec_env(
        _env_factory,
        n_envs=N_ENVS,
        seed=SEED,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=str(MONITOR_DIR),
        monitor_kwargs=dict(info_keywords=("x", "reached", "crashed", "sim_time", "effort_rms")),
    )

    if LOAD_MODEL and MODEL_PATH.exists():
        print(f"[ppo] resuming from {MODEL_PATH}")
        model = PPO.load(str(MODEL_PATH), env=vec_env)
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=N_STEPS,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            learning_rate=3e-4,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=SEED,
            verbose=1,
            tensorboard_log=str(TB_LOG),
        )

    # save_freq is per-worker steps in SB3's CheckpointCallback accounting.
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, CHECKPOINT_FREQ // N_ENVS),
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_obstacle",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, progress_bar=True)
    model.save(str(MODEL_PATH))
    print(f"[ppo] saved model to {MODEL_PATH}")
    vec_env.close()
    return model


def evaluate(model, episodes=EVAL_EPISODES, render=RENDER_EVAL):
    env = Monitor(ObstacleWheelieEnv(render_mode="human" if render else None, reward_cfg=RewardConfig()))
    n_reached = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        ep_return = 0.0
        steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward
            steps += 1
            time.sleep(0.05)
        n_reached += int(info.get("reached", False))
        print(
            f"[eval ep {ep}] x_final={info['x']:.2f} | reached={info['reached']} | "
            f"crashed={info['crashed']} | return={ep_return:.2f} | steps={steps}"
        )
    print(f"[eval] reached goal in {n_reached}/{episodes} episodes")
    env.close()


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    if TRAIN:
        model = train()
    else:
        print(f"[ppo] loading {MODEL_PATH} (TRAIN=False)")
        model = PPO.load(str(MODEL_PATH))
    evaluate(model)


if __name__ == "__main__":
    # SubprocVecEnv spawns worker processes that re-import this module; guarding the
    # entry point is required on every platform that uses "spawn" (macOS/Windows) and
    # is good hygiene on "fork" platforms (Linux) too.
    main()
