#!/usr/bin/env python3

import sys
import math
import time
from pathlib import Path
import csv

import numpy as np
import mujoco
import mujoco.viewer as mj_viewer

# The controller sub-packages and the XML live in the Trail/ root (next to this file).
# nmpc/ is added LAST so it sits FIRST on sys.path -> its params (the superset) wins.
_ROOT = Path(__file__).resolve().parent
for _sub in ("mppi", "gp", "rls", "nmpc"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---- Controller selection -------------------------------------------------------
CONTROLLER = "mppi"          # "nmpc" or "mppi"

if CONTROLLER == "mppi":
    from params_mppi import WheelieParams, MPPIConfig as ControllerConfig

    # from mppi_dynamics import MPPITorch as Controller
    # from mppi_rls import MPPITorch as Controller
    from mppi_gp import MPPITorch as Controller
    # from mppi_rls_gp import MPPITorch as Controller
elif CONTROLLER == "nmpc":
    from params_nmpc import WheelieParams, MPCConfig as ControllerConfig
    from nmpc import NMPC as Controller
else:
    raise ValueError(f"CONTROLLER must be 'mppi' or 'nmpc', got {CONTROLLER!r}")

from GP import GPConfig as SSGPConfig   # GPyTorch sparse GP residual model (shared by both)
from nominal_model import nominal_accel


# ============================================================
# Settings (the node's "parameters")
# ============================================================

XML_PATH = _ROOT / "monster_truck_flip_2d.xml"
# Generated outputs (CSV log + model checkpoint) go in results/ to keep the source tree clean.
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "obstacle_mujoco.csv"
MODEL_PATH = RESULTS_DIR / "obstacle_model.npz"

RENDER   = False       # True = watch the truck; False = fast headless run
CTRL_DT  = 0.05       # control / MPPI period [s]
INIT_Z   = 0.1512     # spawn height
SIM_TIME = 20.0       # per-episode time cap [s]
PRINT_EVERY_N_CONTROLS = 1

# Episodic learning: the GP (and RLS) persist across episodes, the MuJoCo state is
# reset each episode. Episode 1 learns where the obstacle is; episode 2 reuses that.
N_EPISODES = 4000

# GP settings
LOAD_MODEL = False    # True = RESUME from a saved model (GP posterior + RLS state)
GP_ENABLED = True     # False = ZERO the GP's contribution (RLS-only control; GP still learns/logs)

LIVE_PLOT = True          # True = live X-vs-theta plot of the MPPI plan while the sim runs

# RLS settings. RLS_FREEZE=True holds a_rls at the nominal physics values
RLS_FREEZE = True
FORGETTING_FACTOR = 0.9995   # < 1 keeps the fit adaptive
INITIAL_COVARIANCE = 3.0     # prior uncertainty on the weights
SIGMA_V_DOT = 2.0            # measurement-noise std, v_dot channel
SIGMA_OMEGA_DOT = 5.0        # measurement-noise std, omega_dot channel

# Outlier gate: skip a step if the measured accel is bigger than this (contact impact).
ACCEL_CAP_V = 15.0           # m/s^2
ACCEL_CAP_W = 200.0           # rad/s^2

# Controller reference
GOAL_X = 10.0
GOAL_TOL = 0.15          # an episode ends early once |x - GOAL_X| < GOAL_TOL

INITIAL_X = 0.0
INITIAL_Z = INIT_Z
INITIAL_ROOT_PITCH_DEG = 0.0

# The 10 weight names, in the order rls.py uses them.
# WEIGHT_NAMES = ["b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_tan", "b_w2_cos", "b_0",
#                 "a_g", "a_tau", "a_omega", "a_v", 
#                 "a_abs_omega", "a_v_omega", "a_absV_omega",
#                 "a_cos_omega", "a_sin_v", "a_sin", "a_cos_tau", "a_tau_v",
#                 "a_0"]

# WEIGHT_NAMES = ["b_tau", 
#                 "b_v", 
#                 "b_tau_cos", 
#                 "b_0",

#                 "a_g", 
#                 "a_tau", 
#                 "a_omega", 
#                 "a_v", 
#                 "a_0"]

WEIGHT_NAMES = ["b_tau", 
                "b_v", 
                "b_abs_v", 
                "b_tau_cos", 
                "b_tanh", 
                "b_omega_cos", 
                "b_0",

                "a_g", 
                "a_tau", 
                "a_omega", 
                "a_v", 
                "a_absw_w", 
                "a_v_omega", 
                "a_absv_w", 
                "a_cos_omega", 
                "a_sin_v",
                "a_sin",
                "a_cos_v",
                "a_tau_v",
                "a_0"]

# ============================================================
# Model persistence (GP obstacle knowledge + RLS dynamics)
# ============================================================

def save_model(path: Path, gp) -> None:
    """Checkpoint the learned model so a later run can resume from it. Saves the GP
    posterior (state_dict) plus the RLS state to a single .npz. No-op if the GP has not
    warmed up yet (nothing learned to save)."""
    sd = gp.state_dict()
    if not sd:
        return
    np.savez(path, **{f"gp_{k}": v for k, v in sd.items()})


def load_model(path: Path, gp) -> tuple[np.ndarray, np.ndarray]:
    """Restore a model saved by save_model() into `gp`; returns (a_rls, P_rls)."""
    d = np.load(path, allow_pickle=False)
    gp_sd = {k[len("gp_"):]: d[k] for k in d.files if k.startswith("gp_")}
    gp.load_state_dict(gp_sd)


# ============================================================
# EpisodeLogger -- collects the per-step rows, then saves the CSV
# ============================================================

class EpisodeLogger:
    def __init__(self):
        self.history: list[dict] = []

    def record(self, row: dict):
        self.history.append(row)

    def save_csv(self):
        if not self.history:
            print("No data to save.")
            return
        fieldnames = list(self.history[0].keys())
        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        print(f"Saved CSV log: {CSV_PATH}")


# ============================================================
# ObstacleNode -- the node. Sets up the MuJoCo plant directly: the sensor reads and the
# actuator command are its own methods (the "topics"). run() loops the episodes, stepping
# the plant and running one control update every CTRL_DT.
# ============================================================

class ObstacleNode:
    def __init__(self):
        # --- MuJoCo plant ---
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.x_q, self.x_v = self._joint("root_x")        # odometry: qpos / qvel address
        self.z_q, self.z_v = self._joint("root_z")
        self.pitch_q, self.pitch_v = self._joint("root_pitch")   # for the raw-pitch log columns
        self.quat_adr = self._sensor("imu_quat")          # IMU orientation (framequat: w,x,y,z)
        self.gyro_adr = self._sensor("imu_gyro")          # IMU gyro (body rates); y-axis = pitch rate
        self.acc_adr = self._sensor("imu_acc")            # IMU accelerometer (specific force, body frame)
        self.drive = self._actuator("drive_motor")
        self.ctrl_min = float(self.model.actuator_ctrlrange[self.drive, 0])
        self.ctrl_max = float(self.model.actuator_ctrlrange[self.drive, 1])

        self.sim_dt = float(self.model.opt.timestep)
        self.steps_per_ctrl = max(1, round(CTRL_DT / self.sim_dt))
        self.ctrl_dt = self.steps_per_ctrl * self.sim_dt

        # --- estimator, logger, controller (PERSIST across episodes) ---
        self.p = WheelieParams()
        self.cfg = ControllerConfig()        # MPPIConfig or MPCConfig, per CONTROLLER

        self.logger = EpisodeLogger()

        # GPyTorch sparse-variational GP residual learner (GP.py). Feature
        # z = [x, v, theta, omega, tau]: the FULL system input. Built ONCE and kept alive
        self.gp_cfg = SSGPConfig()
        self.gp = self.gp_cfg.build()

        # Build the selected controller. The MPPI holds the BUILT gp object (and takes an
        # integrator); the NMPC takes the gp CONFIG and gets the GP via gp_params each solve.
        if CONTROLLER == "mppi":
            # self.controller = Controller(self.p, self.cfg, self.gp, integrator="rk4",
            #                              live_plot=(LIVE_PLOT and RENDER))
            
            self.controller = Controller(self.p, self.cfg, self.gp,
                             live_plot=(LIVE_PLOT and RENDER))
            
            self.controller.plot_obstacle_span = (1.7, 2.3)   # obstacle x-span shaded in the plan plot
        else:  # nmpc
            self.controller = Controller(self.p, self.cfg, self.gp_cfg)
        print(f"[controller] {CONTROLLER}")

        # GP_ENABLED=False -> RLS-only rollout. Capture the GP's not-ready params (alpha=0 so
        # the contribution is zero, x_std=1 so the standardization stays finite) as the
        # neutral vector handed to the controller.
        self.gp_params_zero = self.gp.mpc_params().copy()
        if not GP_ENABLED:
            print("[GP DISABLED] controller rollout uses RLS only (the GP still learns + logs).")

        # The unique MPPI reference: reach the goal (v_ref=0 so it stops there; the
        # theta/omega entries are ignored because their cost weights are 0).
        # references = [xpos, velocity, theta, theta_dot]
        self.ref = np.array([GOAL_X, 0.0, 0.0, 0.0], dtype=float)

        # Optionally RESUME from a previously saved model (GP posterior + RLS state).
        if LOAD_MODEL and MODEL_PATH.exists():
            load_model(MODEL_PATH, self.gp)
            print(f"Loaded model from {MODEL_PATH.name} "
                  f"(gp_ready={self.gp.ready}, n_active={self.gp.n_active})")
        elif LOAD_MODEL:
            print(f"[warn] LOAD_MODEL=True but {MODEL_PATH.name} not found -- learning from scratch")

        # --- run-level diagnostics (accumulated across episodes) ---
        self.n_used = 0                   # RLS/GP updates fed
        self.n_held = 0                   # steps where a contact-impact outlier was replaced

        # --- per-episode bookkeeping (reset by reset_episode each episode) ---
        self._prev_omega = None           # previous gyro pitch rate, for the omega_dot difference
        self._prev_tau = None             # command applied last step (matches the measured accel)
        self._last_vdot = 0.0             # last accepted accel measurements (held over an outlier)
        self._last_wdot = 0.0
        self.tau_prev = 0.0               # MPPI warm-start anchor
        self.tau_cmd = 0.0
        self.ctrl_cmd = 0.0
        self.solve_success = False
        self.control_count = 0
        self.res_v_dot = 0.0
        self.res_omega_dot = 0.0
        self.gp_pred_v_dot_pre = 0.0      # GP prediction at z_prev BEFORE the update
        self.gp_pred_omega_dot_pre = 0.0
        self.last_state = np.zeros(4)     # most recent IMU controller state [x,v,theta,omega]

    # --- MuJoCo address helpers ---
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

    # --- subscriptions (sensor reads) ---

    def read_imu(self) -> tuple[float, float, float, float]:

        w = float(self.data.sensordata[self.quat_adr + 0])
        x = float(self.data.sensordata[self.quat_adr + 1])
        y = float(self.data.sensordata[self.quat_adr + 2])
        z = float(self.data.sensordata[self.quat_adr + 3])
        pitch = math.atan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (y * y + z * z))
        rate = float(self.data.sensordata[self.gyro_adr + 1])      # gyro y-axis -> pitch rate

        acc_body = np.array(self.data.sensordata[self.acc_adr:self.acc_adr + 3])
        quat = np.array(self.data.sensordata[self.quat_adr:self.quat_adr + 4])
        acc_world = np.zeros(3)
        mujoco.mju_rotVecQuat(acc_world, acc_body, quat)
        v_dot = float(acc_world[0])

        omega_dot = 0.0 if self._prev_omega is None else (rate - self._prev_omega) / self.ctrl_dt
        self._prev_omega = rate

        if abs(v_dot) <= ACCEL_CAP_V and abs(omega_dot) <= ACCEL_CAP_W:
            self._last_vdot, self._last_wdot = v_dot, omega_dot
        else:
            v_dot, omega_dot = self._last_vdot, self._last_wdot
            self.n_held += 1

        return pitch, rate, v_dot, omega_dot

    def read_odometry(self) -> tuple[float, float]:
        """Longitudinal position x [m] and velocity v [m/s]."""
        return float(self.data.qpos[self.x_q]), float(self.data.qvel[self.x_v])

    # --- command publisher ---
    def publish_command(self, tau):
        self.data.ctrl[self.drive] = tau

    # --- plant stepping ---
    def reset_episode(self):
        """Reset the MuJoCo state and per-episode bookkeeping. KEEPS the RLS weights and
        the GP posterior -- that learned knowledge is what carries to the next episode."""
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.x_q] = INITIAL_X
        self.data.qpos[self.z_q] = INITIAL_Z
        self.data.qpos[self.pitch_q] = math.radians(INITIAL_ROOT_PITCH_DEG)
        self.data.ctrl[self.drive] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._prev_omega = None         # don't difference across the reset
        self._prev_tau = None
        self._last_vdot = 0.0
        self._last_wdot = 0.0
        self.tau_prev = 0.0
        self.tau_cmd = 0.0
        self.ctrl_cmd = 0.0
        self.solve_success = False
        self.control_count = 0
        self.res_v_dot = 0.0
        self.res_omega_dot = 0.0
        self.gp_pred_v_dot_pre = 0.0
        self.gp_pred_omega_dot_pre = 0.0
        self.last_state = np.zeros(4)
        self.controller.last_solution = None      # fresh warm start each episode

    def step(self):
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self):
        return float(self.data.time)

    # --- control / run ---
    def _solve(self, state):
        """Call the active controller with the right signature. The MPPI reads the GP
        through its held gp object; the NMPC needs the flat gp_params each solve."""
        if CONTROLLER == "mppi":
            tau_opt, info = self.controller.solve(state, self.ref, self.tau_prev)
            return tau_opt, info
        
        if GP_ENABLED:
            gp_params = self.gp.mpc_params() 
        else:
            gp_params = self.gp_params_zero
        
        tau_opt, info = self.controller.solve(state, self.ref, self.tau_prev, gp_params)
        return tau_opt, info

    def control_update(self):
            x, v = self.read_odometry()
            pitch, rate, vdot, wdot = self.read_imu()
            state_now = np.array([x, v, pitch, rate], dtype=float)
            self.last_state = state_now      # cached so log_row needn't re-read the IMU

            if self._prev_tau is not None:
                # GP target: residual against the SAME nominal the rollout uses.
                # nominal_accel = the lever model + ground clamp from nominal_model.py,
                # which mppi_gp._deriv must also import (one function, two callers).
                nom_v, nom_w = nominal_accel(state_now, self._prev_tau)
                self.res_v_dot = float(vdot - nom_v)
                self.res_omega_dot = float(wdot - nom_w)
                self.n_used += 1

                z = np.array([x, v, pitch, rate, self._prev_tau], dtype=float)
                # predict at the SAME z BEFORE observing -> honest one-step error
                self.gp_pred_v_dot_pre, self.gp_pred_omega_dot_pre, _ = self.gp.predict(z)

                # keep crash physics out of the map: observe only in the driving regime
                if abs(pitch) < 1.0 and abs(rate) < 8.0:
                    self.gp.observe(z, self.res_v_dot, self.res_omega_dot)

            tau, info = self._solve(state_now)
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))
            ctrl = float(np.clip(tau, self.ctrl_min, self.ctrl_max))
            self.publish_command(ctrl)

            self.tau_prev = tau
            self._prev_tau = tau
            self.tau_cmd = tau
            self.ctrl_cmd = ctrl
            self.solve_success = bool(info["success"])
            self.cost = info["cost"]

            if self.control_count % PRINT_EVERY_N_CONTROLS == 0:
                print(
                    f"t={self.time:6.3f} | "
                    f"x={state_now[0]:7.3f} | v={state_now[1]:7.3f} | "
                    f"pitch={math.degrees(state_now[2]):8.2f} deg | "
                    f"omega={state_now[3]:8.3f} | tau={tau:8.3f} | "
                    f"ctrl={ctrl:8.3f} | cost={info['cost']} | "
                    f"res[v={self.res_v_dot:6.3f}, w={self.res_omega_dot:6.3f}]"
                )
            self.control_count += 1



    def log_row(self, episode: int) -> dict:
        x, v = self.read_odometry()
        z = float(self.data.qpos[self.z_q])
        z_dot = float(self.data.qvel[self.z_v])
        raw_pitch = float(self.data.qpos[self.pitch_q])
        raw_pitch_dot = float(self.data.qvel[self.pitch_v])
        pitch, rate = float(self.last_state[2]), float(self.last_state[3])   # IMU pitch/rate from the last control read

        # GP residual prediction at the current state (diagnostics).
        z_now = np.array([x, v, pitch, rate, self.tau_cmd], dtype=float)
        gp_v_dot_pred, gp_omega_dot_pred, _ = self.gp.predict(z_now)

        return {
            "episode": int(episode),
            "time": float(self.data.time),
            "x": x,
            "z": z,
            "x_dot": v,
            "z_dot": z_dot,
            "raw_pitch_rad": raw_pitch,
            "raw_pitch_deg": math.degrees(raw_pitch),
            "raw_pitch_dot": raw_pitch_dot,
            "pitch_rad": pitch,
            "pitch_deg": math.degrees(pitch),
            "pitch_dot": rate,
            "goal_x": GOAL_X,
            "tau_cmd": self.tau_cmd,
            "ctrl_cmd": self.ctrl_cmd,
            "solve_success": int(self.solve_success),

            # GP residual: measured target fed to the GP (r_*) and the GP's own prediction at
            # the current state. Plot these vs x to see the obstacle the GP has localised.
            "r_v_dot": float(self.res_v_dot),
            "r_omega_dot": float(self.res_omega_dot),
            "gp_v_dot_pred": float(gp_v_dot_pred),
            "gp_omega_dot_pred": float(gp_omega_dot_pred),
            # one-step prediction at z BEFORE the GP saw it; the one-step error is
            # r_* - gp_*_pred_pre (clean target/pred pair).
            "gp_v_dot_pred_pre": float(self.gp_pred_v_dot_pre),
            "gp_omega_dot_pred_pre": float(self.gp_pred_omega_dot_pre),
            "gp_ready": int(self.gp.ready),
        }

    def run_episode(self, episode: int, viewer=None):
        self.reset_episode()
        print(f"\n==================== EPISODE {episode} "
              f"(gp_ready={self.gp.ready}, n_active={self.gp.n_active}) ====================")
        ep_history = []
        reached = False
        k = 0
        # The episode ends either at SIM_TIME (while condition) or as soon as the car
        # reaches the goal (break below).
        while self.time < SIM_TIME:
            if viewer is not None and not viewer.is_running():
                break
            start = time.time()

            if k % self.steps_per_ctrl == 0:
                self.control_update()

            self.step()
            row = self.log_row(episode)
            self.logger.record(row)
            ep_history.append(row)

            if viewer is not None:
                viewer.sync()
                sleep_time = self.sim_dt - (time.time() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            k += 1

            if abs(float(self.data.qpos[self.x_q]) - GOAL_X) < GOAL_TOL:
                reached = True
                break

        # Absorb this episode's buffered residuals into the GP (MBRL boundary): the FIRST
        # full episode FITS the GPyTorch model (inducing points, ARD lengthscales, noise) on
        # a rich batch; later episodes WARM-REFINE it. The GP was frozen during the episode.
        self.gp.end_episode()

        x_final = float(self.data.qpos[self.x_q])
        print(f"[episode {episode}] x_final={x_final:.2f} m | goal={GOAL_X:.2f} m | "
              f"reached={reached} | t={self.time:.2f}s | "
              f"gp_ready={self.gp.ready} n_active={self.gp.n_active}")

        # Per-episode GP one-step RMSE over the steps where the GP was active.
        ready = [r for r in ep_history if r["gp_ready"] == 1]
        if ready:
            ew = np.array([r["r_omega_dot"] - r["gp_omega_dot_pred_pre"] for r in ready])
            ev = np.array([r["r_v_dot"] - r["gp_v_dot_pred_pre"] for r in ready])
            print(f"           GP one-step RMSE  omega={np.sqrt(np.mean(ew**2)):.4f}  "
                  f"v={np.sqrt(np.mean(ev**2)):.4f}  ({len(ready)} ready steps)")

        save_model(MODEL_PATH, self.gp)   # checkpoint after each episode

    def run(self, viewer=None):
        """Loop the episodes (GP + RLS persist; MuJoCo state resets each episode)."""
        for ep in range(N_EPISODES):
            if viewer is not None and not viewer.is_running():
                break
            self.run_episode(ep, viewer)

    def finish(self):
        """Report the final (persisted) RLS coefficients and save the CSV log."""
        # print(f"\nRLS/GP updates used: {self.n_used}   |   outliers held: {self.n_held}")
        # print("Final RLS coefficients (persisted across episodes)")
        # print("v_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*cos(theta) + b_0")
        # print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
        # for i, name in enumerate(WEIGHT_NAMES):
        #     print(f"{name:10s} learned: {self.rls.a[i]: .4f} | nominal: {self.rls.a_nom[i]: .4f}")

        self.logger.save_csv()
        if MODEL_PATH.exists():
            print(f"Model checkpoint: {MODEL_PATH}  (set LOAD_MODEL=True to resume from it)")


# ============================================================
# Main
# ============================================================

def main():
    node = ObstacleNode()
    if RENDER:
        with mj_viewer.launch_passive(node.model, node.data) as viewer:
            node.run(viewer)
    else:
        node.run()
    node.finish()


if __name__ == "__main__":
    main()
