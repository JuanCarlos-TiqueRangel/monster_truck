#!/usr/bin/env python3
"""SAC baseline on the obstacle wheelie-climb task (obstacle_gym_env.ObstacleWheelieEnv),
for comparison against the MPPI+CBF / NMPC controllers in this folder.

Off-policy, so a single MuJoCo instance is fine -- SAC's sample efficiency comes from
the replay buffer, not from parallel rollouts (unlike PPO, see obstacle_mujoco_ppo.py).

Usage:
    python3 obstacle_mujoco_sac.py            # train (headless) then render a few eval episodes
Toggle TRAIN / RENDER / LOAD_MODEL below to resume training, or to just watch a saved policy.
"""

from pathlib import Path
import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from obstacle_gym_env import ObstacleWheelieEnv, RewardConfig, RESULTS_DIR

# ============================================================
# Settings
# ============================================================
TRAIN = True                 # False = skip training, just load MODEL_PATH and evaluate
LOAD_MODEL = False            # True = resume training from MODEL_PATH instead of from scratch
RENDER_EVAL = True            # watch the truck during the post-training evaluation episodes

TOTAL_TIMESTEPS = 3_000_000
CHECKPOINT_FREQ = 25_000
EVAL_EPISODES = 100
SEED = 0

MODEL_PATH = RESULTS_DIR / "sac_obstacle.zip"
CHECKPOINT_DIR = RESULTS_DIR / "sac_checkpoints"
TB_LOG = RESULTS_DIR / "tb_sac"
# Per-episode training log (read by results/plot_results.py). A dedicated directory
# rather than a bare filename so stable_baselines3's load_results() -- which globs
# *monitor.csv in a directory -- only ever picks up SAC's own runs.
MONITOR_DIR = RESULTS_DIR / "sac_monitor"


def make_env(render_mode=None, monitor_path=None):
    env = ObstacleWheelieEnv(render_mode=render_mode, reward_cfg=RewardConfig())
    return Monitor(env, filename=monitor_path,
                    info_keywords=("x", "reached", "crashed", "sim_time", "effort_rms"))


def train():
    MONITOR_DIR.mkdir(exist_ok=True)
    env = DummyVecEnv([lambda: make_env(monitor_path=str(MONITOR_DIR))])

    if LOAD_MODEL and MODEL_PATH.exists():
        print(f"[sac] resuming from {MODEL_PATH}")
        model = SAC.load(str(MODEL_PATH), env=env)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=2_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            seed=SEED,
            verbose=1,
            tensorboard_log=str(TB_LOG),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="sac_obstacle",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb, progress_bar=True)
    model.save(str(MODEL_PATH))
    print(f"[sac] saved model to {MODEL_PATH}")
    return model


def evaluate(model, episodes=EVAL_EPISODES, render=RENDER_EVAL):
    env = make_env(render_mode="human" if render else None)
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
        print(f"[sac] loading {MODEL_PATH} (TRAIN=False)")
        model = SAC.load(str(MODEL_PATH))
    evaluate(model)


if __name__ == "__main__":
    main()
