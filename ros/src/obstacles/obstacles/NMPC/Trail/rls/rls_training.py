#!/usr/bin/env python3

import sys
import time
import math
from pathlib import Path
from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as mj_viewer

# The controller sub-packages and the XML live in the Trail/ root, one level up.
_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("mppi", "gp", "rls", "nmpc"):
    sys.path.insert(0, str(_ROOT / _sub))

from params_mppi import WheelieParams
from rls import rls_update


# ============================================================
# Settings (the node's "parameters")
# ============================================================

XML_PATH = _ROOT / "monster_truck_flip_2d.xml"
IMG_PATH = Path(__file__).with_name("images") / "rls_training.png"
NPZ_PATH = Path(__file__).with_name("rls_trained.npz")
CSV_PATH = Path(__file__).with_name("rls_accel.csv")   # measured vs fit, for correlation_plot.py

RENDER   = False      # True = watch the truck; False = fast headless run
CTRL_DT  = 0.05      # control / RLS period [s]
INIT_Z   = 0.1512    # spawn height

# GP settings
LOAD_MODEL = False
GP_ENABLED = True

# RLS settings
RLS_FREEZE = True
FORGETTING_FACTOR = 0.999    # < 1 keeps the fit adaptive
INITIAL_COVARIANCE = 5.0     # prior uncertainty on the weights
SIGMA_V_DOT = 2.0            # measurement-noise std, v_dot channel
SIGMA_OMEGA_DOT = 2.0        # measurement-noise std, omega_dot channel

# Outlier gate: skip a step if the measured accel is bigger than this (contact impact).
ACCEL_CAP_V = 15.0           # m/s^2
ACCEL_CAP_W = 30.0           # rad/s^2

# Held-wheelie operating points [deg]. Several distinct angles give several cos(theta)
# levels, which is what makes the gravity term a_g identifiable.
HELD_ANGLES = (25.0, 45.0, 60.0, 72.0, -25.0, -45.0, -60.0, -72.0)

# The 10 weight names, in the order rls.py uses them.
WEIGHT_NAMES = ["b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
                "a_g", "a_tau", "a_omega", "a_v", "a_0"]


# ============================================================
# Excitation signals  ->  each is a function tau(t_local, state)
# (open-loop signals ignore `state`; the closed-loop ones read pitch/speed)
# ============================================================

def constant(c):
    return lambda t, s: c


def sine(bias, amp, freq):
    return lambda t, s: bias + amp * math.sin(2 * math.pi * freq * t)


def chirp(amp, f0, f1, T):
    """Sine whose frequency sweeps f0 -> f1 over the segment (rich excitation)."""
    return lambda t, s: amp * math.sin(2 * math.pi * (f0 + 0.5 * (f1 - f0) * t / T) * t)


def prbs(amp, dwell, seed=0):
    """Random piecewise-constant torque: a fresh random level every `dwell` seconds."""
    rng = np.random.default_rng(seed)
    box = {"t_next": -1.0, "value": 0.0}

    def signal(t, s):
        if t >= box["t_next"]:
            box["value"] = amp * float(rng.uniform(-1.0, 1.0))
            box["t_next"] = t + dwell
        return box["value"]

    return signal


def hold_wheelie(target_deg):
    """PI + gravity-feedforward balancer that holds a wheelie at `target_deg`, with a
    small multi-frequency dither. Holding several distinct angles is what excites the
    gravity term. (A plain PD only ever settles at its one natural balance angle.)"""
    target = math.radians(target_deg)
    box = {"integral": 0.0}

    def signal(t, s):
        theta, omega = s[2], s[3]
        error = theta - target
        box["integral"] = float(np.clip(box["integral"] + error * CTRL_DT, -4.0, 4.0))
        dither = 1.2 * math.sin(2 * math.pi * 1.0 * t) + 0.8 * math.sin(2 * math.pi * 2.3 * t)
        # neg tau rears up, so theta below target (error<0) -> negative torque
        return -2.0 - 3.5 * math.cos(theta) + 16.0 * error + 12.0 * box["integral"] + 2.8 * omega + dither

    return signal


def track_speed(setpoints, dwell=1.2):
    """PD speed tracker that chases a sweep of speed setpoints (a new one every
    `dwell` s) so |v| spans a wide range both directions, while a pitch term keeps
    the truck flat. This is the phase that identifies the v_dot (drag) terms."""
    box = {"t_next": -1.0, "idx": -1, "v_set": 0.0}

    def signal(t, s):
        if t >= box["t_next"]:
            box["idx"] = (box["idx"] + 1) % len(setpoints)
            box["v_set"] = setpoints[box["idx"]]
            box["t_next"] = t + dwell
        v, theta, omega = s[1], s[2], s[3]
        tau_speed = float(np.clip(2.5 * (box["v_set"] - v), -4.0, 4.0))   # +tau drives forward
        tau_flat = -12.0 * theta - 3.0 * omega          # +tau rears up, so counter with -theta
        return float(np.clip(tau_speed + tau_flat, -6.0, 6.0))

    return signal


# One maneuver in the schedule. `signal` is a tau(t_local, state); `reset_after` returns
# the truck upright when the phase ends (used after flips).
Phase = namedtuple("Phase", ["name", "duration", "signal", "reset_after"])


def build_schedule(p):
    """The maneuver script that excites every regressor."""
    schedule = [
        Phase("speed sweep",    7.2, track_speed([+3.0, -3.0, +1.5, -1.5, +2.5, -2.5]), False),
        Phase("forward cruise", 2.5, sine(-3.0, 1.8, 0.8),                              False),
        Phase("coast",          0.8, constant(0.0),                                     False),
        Phase("reverse dither", 2.0, sine(+4.0, 2.0, 1.0),                              False),
        Phase("fwd/back chirp", 3.5, chirp(6.0, 0.3, 2.5, 3.5),                         False),
    ]
    for angle in HELD_ANGLES:
        schedule.append(Phase(f"hold wheelie {angle:.0f}deg", 4.5, hold_wheelie(angle), True))
    schedule += [
        Phase("PRBS",           3.5, prbs(5.0, 0.30),                                   True),
        Phase("backward flip",  1.6, constant(p.tau_min),                              True),
        Phase("settle",         0.6, constant(0.0),                                     False),
        Phase("forward flip",   1.3, constant(p.tau_max),                              True),
        Phase("settle",         0.6, constant(0.0),                                     False),
    ]
    return schedule


# ============================================================
# RLSEstimator -- recursive least squares for the two-output model
# ============================================================

class RLSEstimator:
    def __init__(self, n=len(WEIGHT_NAMES)):
        self.a = np.zeros(n)                              # learn from scratch
        self.P = INITIAL_COVARIANCE * np.eye(n)

    def update(self, state_prev, tau_prev, state_next, accel_meas, dt):
        self.a, self.P, info = rls_update(
            state_prev=state_prev, tau=tau_prev, state_next=state_next, dt=dt,
            a=self.a, P=self.P, forgetting_factor=FORGETTING_FACTOR,
            sigma_v_dot=SIGMA_V_DOT, sigma_omega_dot=SIGMA_OMEGA_DOT,
            y_dot_meas=accel_meas,
        )
        return info

    @property
    def weight_std(self):
        """1-sigma uncertainty on each learned weight (sqrt of the diagonal of P)."""
        return np.sqrt(np.diag(self.P))


# ============================================================
# TrainingLogger -- collects the run, then reports / saves / plots
# ============================================================

class TrainingLogger:
    FIELDS = ("t", "v", "theta", "tau",
              "vdot_meas", "vdot_hat", "vdot_std",
              "wdot_meas", "wdot_hat", "wdot_std")

    def __init__(self):
        self.log = {k: [] for k in self.FIELDS}
        self.phase_marks = []                             # (t_start, name) for the plot

    def mark_phase(self, t, name):
        self.phase_marks.append((t, name))

    def record(self, t, state, tau, info):
        self.log["t"].append(t)
        self.log["v"].append(state[1])
        self.log["theta"].append(state[2])
        self.log["tau"].append(tau)
        self.log["vdot_meas"].append(info["v_dot_measured"])
        self.log["vdot_hat"].append(info["v_dot_hat"])
        self.log["vdot_std"].append(info["v_dot_std"])
        self.log["wdot_meas"].append(info["omega_dot_measured"])
        self.log["wdot_hat"].append(info["omega_dot_hat"])
        self.log["wdot_std"].append(info["omega_dot_std"])

    def report(self, rls, n_used, n_held):
        vdot_err = np.array(self.log["vdot_meas"]) - np.array(self.log["vdot_hat"])
        wdot_err = np.array(self.log["wdot_meas"]) - np.array(self.log["wdot_hat"])
        rmse_v = float(np.sqrt(np.mean(vdot_err ** 2)))
        rmse_w = float(np.sqrt(np.mean(wdot_err ** 2)))
        vdot_std = np.array(self.log["vdot_std"])
        wdot_std = np.array(self.log["wdot_std"])
        weight_std = rls.weight_std

        print("\n========== RLS identification result ==========")
        print(f"updates used: {n_used}   |   outliers held: {n_held}")
        print(f"one-step RMSE        v_dot = {rmse_v:.4f} m/s^2    omega_dot = {rmse_w:.4f} rad/s^2")
        print(f"predictive 1-sigma   v_dot = {vdot_std.mean():.4f} (final {vdot_std[-1]:.4f})    "
              f"omega_dot = {wdot_std.mean():.4f} (final {wdot_std[-1]:.4f})")
        print("\n  weight        value      +/- 1 sigma")
        for i, name in enumerate(WEIGHT_NAMES):
            print(f"  {name:10s} {rls.a[i]: .6f}   +/- {weight_std[i]:.4f}")
        print("===============================================")

    def save_csv(self):
        """Write measured vs RLS-fit accelerations, with the columns correlation_plot.py expects."""
        table = np.column_stack([self.log["vdot_hat"], self.log["wdot_hat"],
                                 self.log["vdot_meas"], self.log["wdot_meas"]])
        header = "v_dot_hat,omega_dot_hat,v_dot_measured,omega_dot_measured"
        np.savetxt(CSV_PATH, table, delimiter=",", header=header, comments="", fmt="%.6f")
        print(f"Saved training data: {CSV_PATH.name}")

    def plot(self):
        t = np.array(self.log["t"])
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(13, 12))

        # 1) excitation torque
        axs[0].plot(t, self.log["tau"], color="C2")
        axs[0].set_ylabel("tau [Nm]")

        # 2) v_dot: measured vs RLS fit
        self._accel_panel(axs[1], t, self.log["vdot_meas"], self.log["vdot_hat"], "v_dot [m/s$^2$]")

        # 3) omega_dot: measured vs RLS fit
        self._accel_panel(axs[2], t, self.log["wdot_meas"], self.log["wdot_hat"], "omega_dot [rad/s$^2$]")

        # 4) weight convergence (commented out -- showing pitch instead)
        # weights = np.array(self.log["weights"])
        # for i, name in enumerate(WEIGHT_NAMES):
        #     axs[3].plot(t, weights[:, i], label=name)
        # axs[3].set_ylabel("weights")
        # axs[3].legend(fontsize=8, ncol=5)

        # 4) pitch angle
        axs[3].plot(t, np.degrees(self.log["theta"]), color="C3")
        axs[3].set_ylabel("pitch [deg]")
        axs[3].set_xlabel("time [s]")

        for t_mark, name in self.phase_marks:
            for ax in axs:
                ax.axvline(t_mark, color="0.8", ls="--", lw=0.7)
            axs[0].text(t_mark, axs[0].get_ylim()[1], name, rotation=90,
                        va="top", ha="right", fontsize=6, color="0.4")
        for ax in axs:
            ax.grid(True, alpha=0.3)

        fig.suptitle("Two-output RLS identification (fit + pitch)")
        fig.tight_layout()
        IMG_PATH.parent.mkdir(exist_ok=True)
        fig.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {IMG_PATH}")
        plt.show()

    @staticmethod
    def _accel_panel(ax, t, meas, hat, ylabel):
        meas, hat = np.array(meas), np.array(hat)
        # alternative styles (commented): measured as a solid line / fit as a dashed line
        # ax.plot(t, meas, color="0.6", lw=1.0, label="measured")
        ax.plot(t, meas, color="0.3", marker="*", ls="none", ms=4, label="measured")
        # ax.plot(t, hat, color="C0", ls="--", label="RLS fit")
        ax.plot(t, hat, color="C0", marker=".", ls="none", ms=4, label="RLS fit")
        # one-step error trace (commented): ax.plot(t, meas - hat, color="C3", ls=":", label="error")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, ncol=3)


# ============================================================
# RLSTrainingNode -- the node. Sets up the MuJoCo plant directly: the sensor reads and the
# actuator command are its own methods (the "topics"). spin() steps the plant and runs one
# control update every CTRL_DT.
# ============================================================

class RLSTrainingNode:
    def __init__(self):
        # --- MuJoCo plant ---
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.x_q, self.x_v = self._joint("root_x")        # odometry: qpos / qvel address
        self.z_q, _ = self._joint("root_z")
        self.quat_adr = self._sensor("imu_quat")          # IMU orientation (framequat: w,x,y,z)
        self.gyro_adr = self._sensor("imu_gyro")          # IMU gyro (body rates); y-axis = pitch rate
        self.acc_adr = self._sensor("imu_acc")            # IMU accelerometer (specific force, body frame)
        self.drive = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
        self.sim_dt = float(self.model.opt.timestep)
        self.steps_per_ctrl = max(1, round(CTRL_DT / self.sim_dt))
        self.ctrl_dt = self.steps_per_ctrl * self.sim_dt

        # --- estimator, logger, schedule ---
        self.p = WheelieParams()
        self.rls = RLSEstimator()
        self.logger = TrainingLogger()
        self.schedule = build_schedule(self.p)

        # per-phase / per-step state used by the control update
        self._signal = constant(0.0)
        self._phase_t0 = 0.0
        self._prev_omega = None         # previous gyro pitch rate, for the omega_dot difference
        self._prev_tau = None           # command applied last step (matches the measured accel)
        self._last_vdot = 0.0           # last accepted accel measurements (held over an outlier)
        self._last_wdot = 0.0
        self.n_used = 0
        self.n_held = 0                 # steps where a contact-impact outlier was replaced

    # --- MuJoCo address helpers ---

    def _joint(self, name):
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return int(self.model.sensor_adr[sid])

    # --- subscriptions (sensor reads) ---
    def read_imu(self):

        w = float(self.data.sensordata[self.quat_adr + 0])
        x = float(self.data.sensordata[self.quat_adr + 1])
        y = float(self.data.sensordata[self.quat_adr + 2])
        z = float(self.data.sensordata[self.quat_adr + 3])
        pitch = math.atan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (y * y + z * z))

        rate = float(self.data.sensordata[self.gyro_adr + 1])      # gyro y-axis -> pitch rate

        # accelerometer (body frame) -> world frame: the IMU measures along the truck's own
        # axes, so when it pitches we rotate the reading by the IMU quaternion to get the
        # world-x acceleration the model expects (gravity is vertical, so it leaves x alone).
        acc_body = np.array(self.data.sensordata[self.acc_adr:self.acc_adr + 3])
        quat = np.array(self.data.sensordata[self.quat_adr:self.quat_adr + 4])
        acc_world = np.zeros(3)
        mujoco.mju_rotVecQuat(acc_world, acc_body, quat)
        v_dot = float(acc_world[0])

        # omega_dot from the gyro-rate difference (no angular-accel sensor)
        omega_dot = 0.0 if self._prev_omega is None else (rate - self._prev_omega) / self.ctrl_dt
        self._prev_omega = rate

        # outlier rejection: a wheel contact spikes the accel/omega_dot for a single step, so
        # hold the last good measurement instead of feeding the spike to the estimator.
        if abs(v_dot) <= ACCEL_CAP_V and abs(omega_dot) <= ACCEL_CAP_W:
            self._last_vdot = v_dot
            self._last_wdot = omega_dot
        else:
            v_dot, omega_dot = self._last_vdot, self._last_wdot
            self.n_held += 1

        return pitch, rate, v_dot, omega_dot

    def read_odometry(self):
        """Longitudinal position x [m] and velocity v [m/s]."""
        pos_x = float(self.data.qpos[self.x_q])
        velocity = float(self.data.qvel[self.x_v])
        return pos_x, velocity

    # --- command publisher ---
    def publish_command(self, tau):
        self.data.ctrl[self.drive] = tau

    # --- plant stepping ---
    def reset(self):
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.z_q] = INIT_Z
        self.data.ctrl[self.drive] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self):
        return float(self.data.time)

    # --- control / run ---

    def control_step(self):
        """One control update at CTRL_DT: read the IMU + odometry, run one RLS update, then
        publish the next command. read_imu holds the last good accel over a contact impact."""
        x, v = self.read_odometry()
        pitch, rate, vdot, wdot = self.read_imu()                     # subscriptions (IMU)
        state = np.array([x, v, pitch, rate], dtype=float)

        # The accel we just read is the truck's response to the command already applied
        # (prev tau), so pair it with that command in the regressor.
        if self._prev_tau is not None:
            info = self.rls.update(state, self._prev_tau, state, np.array([vdot, wdot]), self.ctrl_dt)
            self.logger.record(self.time, state, self._prev_tau, info)
            self.n_used += 1

        # compute + publish the next command from the current state
        elapsed = self.time - self._phase_t0
        tau = float(np.clip(self._signal(elapsed, state), self.p.tau_min, self.p.tau_max))
        self.publish_command(tau)                                     # command publisher
        self._prev_tau = tau

    def spin(self, viewer=None):
        """Run the schedule, stepping the plant and running one control update every CTRL_DT."""
        self.reset()
        for phase in self.schedule:
            self._phase_t0 = self.time
            self._signal = phase.signal
            self.logger.mark_phase(self._phase_t0, phase.name)
            print(f"--- {phase.name}  [{phase.duration:.1f}s] ---")

            for k in range(round(phase.duration / self.sim_dt)):
                if viewer is not None and not viewer.is_running():
                    return
                if k % self.steps_per_ctrl == 0:
                    self.control_step()
                self.step()
                if viewer is not None:
                    viewer.sync()
                    time.sleep(self.sim_dt)                           # roughly real-time playback

            if phase.reset_after:
                self.reset()
                self._prev_omega = None                               # don't difference across the teleport
                self._prev_tau = None

    def finish(self):
        """Report, save the weights + CSV, and plot."""
        self.logger.report(self.rls, self.n_used, self.n_held)
        self.logger.save_csv()
        np.savez(NPZ_PATH, a_rls=self.rls.a, P_rls=self.rls.P)
        print(f"Saved learned weights: {NPZ_PATH.name}")
        print("a_rls =", np.array2string(self.rls.a, precision=6, separator=", "))
        self.logger.plot()


# ============================================================
# Main
# ============================================================

def main():
    node = RLSTrainingNode()
    if RENDER:
        with mj_viewer.launch_passive(node.model, node.data) as viewer:
            node.spin(viewer)
    else:
        node.spin()
    node.finish()


if __name__ == "__main__":
    main()
