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
    from mppi_dynamics import MPPITorch as Controller
elif CONTROLLER == "nmpc":
    from params_nmpc import WheelieParams, MPCConfig as ControllerConfig
    from nmpc_dynamics import NMPC as Controller
else:
    raise ValueError(f"CONTROLLER must be 'mppi' or 'nmpc', got {CONTROLLER!r}")


# ============================================================
# Settings (the node's "parameters")
# ============================================================

XML_PATH = _ROOT / "monster_truck_flip_2d.xml"
# Generated outputs (CSV log + model checkpoint) go in results/ to keep the source tree clean.
RESULTS_DIR = _ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_PATH = RESULTS_DIR / "obstacle_mujoco.csv"
MODEL_PATH = RESULTS_DIR / "obstacle_model.npz"

RENDER   = True       # True = watch the truck; False = fast headless run
CTRL_DT  = 0.05       # control / MPPI period [s]
INIT_Z   = 0.1512     # spawn height
SIM_TIME = 20.0       # per-episode time cap [s]
PRINT_EVERY_N_CONTROLS = 1

# Episodic learning: the GP (and RLS) persist across episodes, the MuJoCo state is
# reset each episode. Episode 1 learns where the obstacle is; episode 2 reuses that.
N_EPISODES = 20

LIVE_PLOT = True          # True = live X-vs-theta plot of the MPPI plan while the sim runs

# Outlier gate: skip a step if the measured accel is bigger than this (contact impact).
ACCEL_CAP_V = 15.0           # m/s^2
ACCEL_CAP_W = 200.0           # rad/s^2

# Controller reference
GOAL_X = 10.0
GOAL_TOL = 0.15          # an episode ends early once |x - GOAL_X| < GOAL_TOL

INITIAL_X = 0.0
INITIAL_Z = INIT_Z
INITIAL_ROOT_PITCH_DEG = 0.0

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

        # Build the selected controller. 
        if CONTROLLER == "mppi":          
            self.controller = Controller(
                                p=self.p, 
                                cfg=self.cfg, 
                                integrator="rk4",
                                live_plot=(LIVE_PLOT and RENDER)
                                )
            self.controller.plot_obstacle_span = (1.7, 2.3)   # obstacle x-span shaded in the plan plot
        else:  # nmpc
            self.controller = Controller(
                                self.p,
                                self.cfg,
                                live_plot=(LIVE_PLOT and RENDER),
                            )
            self.controller.plot_obstacle_span = (1.7, 2.3)


        print(f"[controller] {CONTROLLER}")

        # The unique MPPI reference: reach the goal (v_ref=0 so it stops there; the
        # theta/omega entries are ignored because their cost weights are 0).
        # references = [xpos, velocity, theta, theta_dot]
        self.ref = np.array([GOAL_X, 0.0, 0.0, 0.0], dtype=float)

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
            #self.n_held += 1

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
        self.last_state = np.zeros(4)
        self.controller.last_solution = None      # fresh warm start each episode

    def step(self):
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self):
        return float(self.data.time)

    # --- control / run ---
    def _solve(self, state):

        if CONTROLLER == "mppi":
            tau_opt, info = self.controller.solve(state, self.ref, self.tau_prev)
            return tau_opt, info
        if CONTROLLER == "nmpc":
            tau_opt, info = self.controller.solve(state, self.ref, self.tau_prev)
            return tau_opt, info
        else:
            print("WARNING: USE MPPI CONTROLLER")
        
        tau_opt, info = self.controller.solve(state, self.ref, self.tau_prev)
        return tau_opt, info

    def control_update(self):
        x, v = self.read_odometry()
        pitch, rate, vdot, wdot = self.read_imu()
        state_now = np.array([x, v, pitch, rate], dtype=float)
        self.last_state = state_now      # cached so log_row needn't re-read the IMU

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
                f"ctrl={ctrl:8.3f} | cost={info['cost']} | solve={self.solve_success}"
            )
        self.control_count += 1

    def log_row(self, episode: int) -> dict:
        x, v = self.read_odometry()
        z = float(self.data.qpos[self.z_q])
        z_dot = float(self.data.qvel[self.z_v])
        raw_pitch = float(self.data.qpos[self.pitch_q])
        raw_pitch_dot = float(self.data.qvel[self.pitch_v])
        pitch, rate = float(self.last_state[2]), float(self.last_state[3])   # IMU pitch/rate from the last control read

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

        }

    def run_episode(self, episode: int, viewer=None):
        self.reset_episode()
        print(f"\n==================== EPISODE {episode} ====================")
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

        x_final = float(self.data.qpos[self.x_q])
        print(f"[episode {episode}] x_final={x_final:.2f} m | goal={GOAL_X:.2f} m | "
              f"reached={reached} | t={self.time:.2f}s | ")

    def run(self, viewer=None):
        for ep in range(N_EPISODES):
            if viewer is not None and not viewer.is_running():
                break
            self.run_episode(ep, viewer)

    def finish(self):
        self.logger.save_csv()
        print("[INFO]: data saved from the experiment")



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
