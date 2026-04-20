#!/usr/bin/env python3
"""Fast headless in-process training loop for SVGP + MPPI.

This script keeps your MuJoCo dynamics in-process and removes ROS2 timers,
so episodes run as fast as the machine allows. Use this for learning/testing,
and keep the ROS2 nodes for debugging or hardware-in-the-loop runs.
"""
import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GP_DIR = BASE_DIR / "gp"

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params
from gp.svgp_dynamics import SVGPManager
from gp.svgp_retrain_manager import GPRetrainManager
from utils.mppi_core import MPPICore
from utils.dataset_buffer import DatasetBuffer
from utils.live_plot import LivePlotter
from utils.episode_metrics import EpisodeMetricsWriter
from mujoco_model.planar_headless_env import PlanarMonsterTruckEnv


class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")
    def warn(self, msg):
        print(f"[WARN] {msg}")
    def warning(self, msg):
        self.warn(msg)
    def error(self, msg):
        print(f"[ERROR] {msg}")


@dataclass
class MPPIConfig:
    ctrl_dt: float = cfg_params.gp.sample_time_dt
    horizon: int = 20
    num_rollouts: int = 512
    lambda_: float = 100.0
    sigma: float = 0.6
    u_min: float = 0.0
    u_max: float = 1.0
    goal_x: float = 5.0
    pitch_target: float = 1.0
    pitch_stop_abs: float = 1.5
    roll_stop_abs: float = 1.2

    gp_xpos_path: str = str(GP_DIR / "models" / cfg_params.models.xpos)
    gp_xpos_dot_path: str = str(GP_DIR / "models" / cfg_params.models.xpos_dot)
    gp_pitch_path: str = str(GP_DIR / "models" / cfg_params.models.pitch)
    gp_pitch_dot_path: str = str(GP_DIR / "models" / cfg_params.models.pitch_dot)

    log_dir: str = str(BASE_DIR / "logs_headless")
    max_log_points: int = 200_000

    min_points_to_train: int = 20
    N_target_train: int = 10000
    train_kernel: str = cfg_params.gp.kernel
    train_iters: int = cfg_params.gp.iterations
    train_lr: float = cfg_params.gp.learning_rate
    train_num_inducing: int = cfg_params.gp.num_inducing
    train_batch_size: int | None = cfg_params.gp.batch_size
    gp_target_mode: str = cfg_params.gp.type_of_data
    min_new_points_between_trains: int = 20

    live_plot: bool = False
    live_plot_save_png: bool = False
    live_plot_mode: str = "both"

    episode_timeout_sec: float = 20.0
    seed_npz_path: str = str(DATA_DIR / cfg_params.files.ini_data_file)
    seed_episode_id: int = -1
    keep_seed: bool = True

    w_u: float = 7.1
    w_du: float = 15.0
    w_pitch: float = 10.0
    w_pitch_dot: float = 32.0
    w_goal: float = 10.0
    w_xpos_dot: float = 20.0
    w_uncertainty: float = 450.0
    beta_safety: float = 10.0

    x_min_terminate: float = -3.0
    just_gp_model: bool = False
    stop_re_training_mode: bool = False

    online_update_steps: int = 50
    online_replay_size: int = 1024
    online_max_keep_points: int = 20000
    full_retrain_every_episodes: int = 10
    retrain_every_episodes: int = 1
    max_points_for_train: int = 20000

    max_episodes: int = 50
    render_eval: bool = False


class HeadlessSVGPTrainer:
    def __init__(self, cfg: MPPIConfig):
        self.cfg = cfg
        self.logger = SimpleLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using torch device: {self.device}")

        self.env = PlanarMonsterTruckEnv(ctrl_dt=self.cfg.ctrl_dt, enable_viewer=self.cfg.render_eval)

        self.gp_pitch: SVGPManager = SVGPManager.load(self.cfg.gp_pitch_path, device=self.device)
        self.gp_pitch_dot: SVGPManager = SVGPManager.load(self.cfg.gp_pitch_dot_path, device=self.device)
        self.gp_xpos: SVGPManager = SVGPManager.load(self.cfg.gp_xpos_path, device=self.device)
        self.gp_xpos_dot: SVGPManager = SVGPManager.load(self.cfg.gp_xpos_dot_path, device=self.device)

        self.model_lock = threading.Lock()
        self.mppi = MPPICore(
            cfg=self.cfg,
            device=self.device,
            gp_xpos=self.gp_xpos,
            gp_xpos_dot=self.gp_xpos_dot,
            gp_pitch=self.gp_pitch,
            gp_pitch_dot=self.gp_pitch_dot,
            model_lock=self.model_lock,
            logger=self.logger,
        )

        self.dataset = DatasetBuffer(
            maxlen=self.cfg.max_log_points,
            log_dir=self.cfg.log_dir,
            ctrl_dt=self.cfg.ctrl_dt,
            logger=self.logger,
        )
        self.retrain = GPRetrainManager(
            cfg=self.cfg,
            device=self.device,
            model_lock=self.model_lock,
            logger=self.logger,
        )
        self.plotter = LivePlotter(
            enabled=self.cfg.live_plot,
            save_png=self.cfg.live_plot_save_png,
            out_dir=self.cfg.log_dir,
            mode=self.cfg.live_plot_mode,
            logger=self.logger,
        )
        self.metrics = EpisodeMetricsWriter(
            log_dir=self.cfg.log_dir,
            plotter=self.plotter,
            logger=self.logger,
        )
        self.episode_id = 0

    def _accumulate_executed_cost(self, x0_np, u_cmd):
        x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=self.device).view(1, 4)
        u = torch.as_tensor([u_cmd], dtype=torch.float32, device=self.device)
        c = self.mppi.stage_cost_torch(x0, u)
        return float(c.item())

    def _log_step(self, obs, u):
        self.dataset.append_step(
            pitch=float(obs["pitch"]),
            pitch_dot=float(obs["pitch_dot"]),
            xpos=float(obs["xpos"]),
            xpos_dot=float(obs["xpos_dot"]),
            u=float(u),
            episode_id=int(self.episode_id),
        )

    def _maybe_retrain_and_reload(self):
        ep_num = self.episode_id + 1
        do_retrain_now = (self.cfg.retrain_every_episodes > 0) and (ep_num % self.cfg.retrain_every_episodes == 0)
        if not do_retrain_now:
            return False

        started = self.retrain.maybe_start_retrain_async(self.dataset, episode_id=self.episode_id, force=True)
        if not started:
            return False

        while self.retrain.training:
            time.sleep(0.05)

        loaded = self.retrain.reload_models_if_ready()
        if loaded is None:
            return True

        gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot = loaded
        self.gp_xpos, self.gp_xpos_dot, self.gp_pitch, self.gp_pitch_dot = loaded
        self.mppi.set_models(gp_xpos, gp_xpos_dot, gp_pitch, gp_pitch_dot)
        self.mppi.reset_plan()
        return True

    def run(self):
        for ep in range(self.cfg.max_episodes):
            self.episode_id = ep
            obs, _ = self.env.reset()
            self.mppi.reset_plan()
            ep_cost_sum = 0.0
            ep_cost_steps = 0
            t0 = time.perf_counter()
            success = 0
            retrain_started = False

            max_steps = max(1, int(math.ceil(self.cfg.episode_timeout_sec / self.cfg.ctrl_dt)))
            for step_idx in range(max_steps):
                pitch = float(obs["pitch"])
                pitch_dot = float(obs["pitch_dot"])
                roll = float(obs["roll"])
                xpos = float(obs["xpos"])
                xpos_dot = float(obs["xpos_dot"])

                if xpos <= float(self.cfg.x_min_terminate):
                    self.logger.warn(f"Episode {ep}: x boundary fail at x={xpos:.3f}")
                    self.dataset.drop_episode(ep)
                    break

                if xpos >= float(self.cfg.goal_x):
                    success = 1
                    break

                if abs(pitch) >= float(self.cfg.pitch_stop_abs) or abs(roll) >= float(self.cfg.roll_stop_abs):
                    self.logger.warn(f"Episode {ep}: flip detected roll={roll:.3f}, pitch={pitch:.3f}")
                    self.dataset.drop_episode(ep)
                    break

                x0 = np.array([xpos, xpos_dot, pitch, pitch_dot], dtype=np.float32)
                try:
                    u_cmd = self.mppi.action(x0)
                    if not math.isfinite(u_cmd):
                        self.logger.error("u_cmd became NaN/Inf. Forcing 0.")
                        u_cmd = 0.0
                except Exception as e:
                    self.logger.error(f"MPPI action failed: {e}")
                    u_cmd = 0.0

                self._log_step(obs, u_cmd)
                ep_cost_sum += self._accumulate_executed_cost(x0, u_cmd)
                ep_cost_steps += 1
                obs, _, _, _, _ = self.env.step(u_cmd)

            dt = time.perf_counter() - t0
            avg_cost = ep_cost_sum / max(1, ep_cost_steps)

            if self.dataset.n_points() < self.cfg.N_target_train and not self.cfg.just_gp_model:
                retrain_started = self._maybe_retrain_and_reload()

            self.metrics.write(
                episode=ep,
                time_to_goal_sec=float(dt),
                retrain_started=bool(retrain_started),
                cost=float(avg_cost),
                success=int(success),
            )
            self.logger.info(
                f"Episode {ep} finished | success={success} | steps={ep_cost_steps} | wall_time={dt:.3f}s | avg_cost={avg_cost:.3f}"
            )

        self.env.close()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50, help="Number of episodes to run.")
    ap.add_argument("--viewer", action="store_true", help="Enable MuJoCo viewer during headless trainer.")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=20.0)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = MPPIConfig(
        max_episodes=int(args.episodes),
        render_eval=bool(args.viewer),
        horizon=int(args.horizon),
        num_rollouts=int(args.rollouts),
        episode_timeout_sec=float(args.timeout),
    )
    trainer = HeadlessSVGPTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
