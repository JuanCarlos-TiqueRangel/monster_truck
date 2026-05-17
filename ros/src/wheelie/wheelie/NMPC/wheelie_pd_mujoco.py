#!/usr/bin/env python3
"""
wheelie_pd_mujoco_no_ramp.py

MuJoCo implementation of a wheelie pitch controller using feedback
linearization + PD with a DIRECT step reference.

No reference ramp is used.

Controller model:
    x_dot     = v
    v_dot     = tau / (m*r) - c_v*v
    theta_dot = omega
    omega_dot = (-tau + m*g*l*cos(theta)) / I_eff

Feedback-linearizing PD law:
    tau = m*g*l*cos(theta) + I_eff*(kp*(theta - theta_ref) + kd*omega)

In the ideal simplified model this gives:
    e_ddot + kd*e_dot + kp*e = 0

Run:
    python3 wheelie_pd_mujoco_no_ramp.py

Expected files in the same folder:
    monster_truck_flip_2d.xml
"""

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer as mj_viewer
import matplotlib.pyplot as plt


# ============================================================
# Easy-to-debug settings
# ============================================================

XML_PATH = Path(__file__).with_name("monster_truck_flip_2d.xml")
CSV_PATH = Path(__file__).with_name("wheelie_pd_mujoco_no_ramp_log.csv")

RENDER = True
SIM_TIME = 20.0
CTRL_DT = 0.05
PRINT_EVERY_N_CONTROLS = 5

# In this MuJoCo XML, root_pitch is negative during a backward wheelie.
# This makes the controller see a backward wheelie as positive pitch.
PITCH_SIGN = -1.0

# This matches your NMPC convention.
# If the motor acts in the wrong direction, change this to +1.0.
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0

INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_ROOT_PITCH_DEG = 0.0

# DIRECT step reference. No ramp is used.
PITCH_REF_DEG = 80.0

# Optional command-rate limit. Disabled by default because the request is
# direct step tracking with no ramp/smoothing.
TAU_RATE_LIMIT = None  # example: 200.0 for N*m/s, or None to disable


# ============================================================
# Model and controller parameters
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1
    l: float = 0.18
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.085
    g: float = 9.81
    c_v: float = 9.0

    # Controller torque-like bounds before MuJoCo ctrl clipping.
    tau_min: float = -12.0
    tau_max: float = 12.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class PDGains:
    # Direct step reference needs more damping than the ramped version.
    # If it is too slow: increase kp a little.
    # If it overshoots: increase kd first.
    kp: float = 150.0
    kd: float = 40.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def rate_limit(value: float, previous: float, max_rate: float | None, dt: float) -> float:
    if max_rate is None:
        return value
    max_step = max_rate * dt
    return clamp(value, previous - max_step, previous + max_step)


def pd_feedback_linearizing_controller(
    theta: float,
    omega: float,
    theta_ref: float,
    p: WheelieParams,
    gains: PDGains,
) -> float:
    """
    Feedback linearization + PD.

    Original pitch dynamics:
        I_eff*theta_ddot = -tau + m*g*l*cos(theta)

    Desired virtual acceleration:
        theta_ddot = -kp*(theta - theta_ref) - kd*omega

    Solving for tau:
        tau = m*g*l*cos(theta) + I_eff*(kp*(theta - theta_ref) + kd*omega)
    """
    error_pitch = theta - theta_ref
    tau = (
        p.m * p.g * p.l * math.cos(theta)
        + p.I_eff * (gains.kp * error_pitch + gains.kd * omega)
    )
    return clamp(tau, p.tau_min, p.tau_max)


# ============================================================
# MuJoCo helpers
# ============================================================

def get_joint_addresses(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    qpos_id = int(model.jnt_qposadr[jid])
    qvel_id = int(model.jnt_dofadr[jid])
    return qpos_id, qvel_id


def get_actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        raise RuntimeError(f"Actuator not found: {actuator_name}")
    return aid


def save_history_csv(history: list[dict], csv_path: Path) -> None:
    if not history:
        print("No data to save.")
        return

    fieldnames = list(history[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved CSV log: {csv_path}")


def plot_history(history: list[dict]) -> None:
    if not history:
        return

    t = np.array([row["time"] for row in history])
    pitch = np.array([row["pitch_deg"] for row in history])
    pitch_ref = np.array([row["pitch_ref_deg"] for row in history])
    omega = np.array([row["pitch_dot"] for row in history])
    tau = np.array([row["tau_cmd"] for row in history])
    ctrl = np.array([row["ctrl_cmd"] for row in history])
    x_dot = np.array([row["x_dot"] for row in history])

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 9))

    axs[0].plot(t, pitch, label="theta")
    axs[0].plot(t, pitch_ref, linestyle="--", label="theta_ref")
    axs[0].set_ylabel("pitch [deg]")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, omega)
    axs[1].set_ylabel("omega [rad/s]")
    axs[1].grid(True)

    axs[2].plot(t, tau, label="tau_cmd")
    axs[2].plot(t, ctrl, linestyle="--", label="ctrl_cmd")
    axs[2].set_ylabel("command")
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(t, x_dot)
    axs[3].set_xlabel("time [s]")
    axs[3].set_ylabel("x velocity [m/s]")
    axs[3].grid(True)

    fig.suptitle("Wheelie MuJoCo PD Closed-Loop Response, No Reference Ramp")
    fig.tight_layout()
    plt.show()


# ============================================================
# Main simulation
# ============================================================

def main() -> None:
    p = WheelieParams()
    gains = PDGains()

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    root_x_qid, root_x_vid = get_joint_addresses(model, "root_x")
    root_z_qid, root_z_vid = get_joint_addresses(model, "root_z")
    root_pitch_qid, root_pitch_vid = get_joint_addresses(model, "root_pitch")
    drive_id = get_actuator_id(model, "drive_motor")

    ctrl_min = float(model.actuator_ctrlrange[drive_id, 0])
    ctrl_max = float(model.actuator_ctrlrange[drive_id, 1])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[root_x_qid] = INITIAL_X
    data.qpos[root_z_qid] = INITIAL_Z
    data.qpos[root_pitch_qid] = math.radians(INITIAL_ROOT_PITCH_DEG)
    data.ctrl[drive_id] = 0.0
    mujoco.mj_forward(model, data)

    theta_ref = math.radians(PITCH_REF_DEG)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    n_steps = int(round(SIM_TIME / sim_dt))

    tau_prev = 0.0
    tau_cmd = 0.0
    ctrl_cmd = 0.0
    control_count = 0
    history: list[dict] = []

    def read_controller_state() -> tuple[float, float, float, float, float, float]:
        x = float(data.qpos[root_x_qid])
        z = float(data.qpos[root_z_qid])
        v = float(data.qvel[root_x_vid])
        z_dot = float(data.qvel[root_z_vid])

        raw_pitch = float(data.qpos[root_pitch_qid])
        raw_pitch_dot = float(data.qvel[root_pitch_vid])

        theta = PITCH_SIGN * raw_pitch
        omega = PITCH_SIGN * raw_pitch_dot
        return x, z, v, z_dot, theta, omega

    def control_update() -> None:
        nonlocal tau_prev, tau_cmd, ctrl_cmd, control_count

        x, z, v, z_dot, theta, omega = read_controller_state()

        # Direct step reference every control cycle. No ramp.
        tau_raw = pd_feedback_linearizing_controller(theta, omega, theta_ref, p, gains)
        tau_raw = rate_limit(tau_raw, tau_prev, TAU_RATE_LIMIT, CTRL_DT)
        tau = float(np.clip(tau_raw, p.tau_min, p.tau_max))
        tau_prev = tau

        ctrl = ACTUATOR_SIGN * TAU_TO_CTRL * tau
        ctrl = float(np.clip(ctrl, ctrl_min, ctrl_max))
        data.ctrl[drive_id] = ctrl

        tau_cmd = tau
        ctrl_cmd = ctrl

        if control_count % PRINT_EVERY_N_CONTROLS == 0:
            print(
                f"t={data.time:6.3f} | "
                f"x={x:7.3f} | v={v:7.3f} | "
                f"pitch={math.degrees(theta):8.2f} deg | "
                f"ref={PITCH_REF_DEG:8.2f} deg | "
                f"omega={omega:8.3f} | tau={tau:8.3f} | ctrl={ctrl:8.3f}"
            )

        control_count += 1

    def log_step() -> None:
        x, z, v, z_dot, theta, omega = read_controller_state()
        raw_pitch = float(data.qpos[root_pitch_qid])
        raw_pitch_dot = float(data.qvel[root_pitch_vid])

        history.append({
            "time": float(data.time),
            "x": x,
            "z": z,
            "x_dot": v,
            "z_dot": z_dot,
            "raw_pitch_rad": raw_pitch,
            "raw_pitch_deg": math.degrees(raw_pitch),
            "raw_pitch_dot": raw_pitch_dot,
            "pitch_rad": theta,
            "pitch_deg": math.degrees(theta),
            "pitch_dot": omega,
            "pitch_ref_rad": theta_ref,
            "pitch_ref_deg": PITCH_REF_DEG,
            "tau_cmd": tau_cmd,
            "ctrl_cmd": ctrl_cmd,
        })

    if RENDER:
        k = 0
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_TIME:
                start = time.time()

                if k % ctrl_steps == 0:
                    control_update()

                mujoco.mj_step(model, data)
                log_step()
                viewer.sync()

                sleep_time = sim_dt - (time.time() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

                k += 1
    else:
        for k in range(n_steps):
            if k % ctrl_steps == 0:
                control_update()
            mujoco.mj_step(model, data)
            log_step()

    final_pitch = PITCH_SIGN * float(data.qpos[root_pitch_qid])
    final_omega = PITCH_SIGN * float(data.qvel[root_pitch_vid])

    print("\nFinal state")
    print(f"x      = {float(data.qpos[root_x_qid]):.3f} m")
    print(f"v      = {float(data.qvel[root_x_vid]):.3f} m/s")
    print(f"pitch  = {math.degrees(final_pitch):.2f} deg")
    print(f"omega  = {final_omega:.3f} rad/s")
    print(f"tau    = {tau_cmd:.3f}")
    print(f"ctrl   = {ctrl_cmd:.3f}")

    save_history_csv(history, CSV_PATH)
    plot_history(history)


if __name__ == "__main__":
    main()
