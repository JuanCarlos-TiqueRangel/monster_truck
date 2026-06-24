#!/usr/bin/env python3
"""
rls_training_short.py
---------------------
MINIMAL system-identification run for the two-output full-dynamics RLS (rls.py): drive the
MuJoCo truck with a single PRBS excitation torque -- nothing else, no scripted maneuvers --
and learn the 10 dynamics weights online. PRINTS the weights + one-step errors and PLOTS the
weight convergence together with the prediction error.

Controller-free counterpart to obstacle_mujoco_simulation.py: no NMPC, no GP -- just an
open-loop PRBS -> RLS.

Sign conventions match obstacle_mujoco_simulation.py. Measured on this model:
  * NEGATIVE controller tau -> drives FORWARD and rears up (backward wheelie/flip)
  * POSITIVE controller tau -> drives BACKWARD and noses over (forward flip)
"""

import math
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as mj_viewer

_HERE = Path(__file__).resolve().parent
# This script lives in Trail/rls/, but the controller sub-packages (mppi/ gp/ rls/ nmpc/)
# AND the XML live in the Trail/ ROOT one level up -- so anchor on the parent, not _HERE.
_ROOT = _HERE.parent
for _sub in ("mppi", "gp", "rls", "nmpc"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from params_mppi import WheelieParams
from rls import nominal_rls_parameters, rls_update
#from rls_njit import nominal_rls_parameters, rls_update


# ============================================================
# Settings
# ============================================================

XML_PATH = _ROOT / "monster_truck_flip_2d.xml"   # the XML lives in the Trail/ root
IMG_PATH = Path(__file__).with_name("images") / "rls_training_short.png"
NPZ_PATH = Path(__file__).with_name("rls_trained_short.npz")

RENDER = False           # watch the truck (set False for a fast/headless run)
DURATION = 20.0          # length of the PRBS excitation run [s]
CTRL_DT = 0.05           # RLS/control period [s]  (MuJoCo integrates at model.opt.timestep)
PRINT_EVERY_N = 5        # print one status line every N control steps
INITIAL_Z = 0.1512       # spawn height

# Use "qacc" for SIM identification (exact ground truth); "imu" is the hardware path.
ACCEL_SOURCE = "finite_diff"

# ---- PRBS excitation ----
PRBS_AMP = 4.0           # torque amplitude [Nm]
PRBS_DWELL = 0.2        # hold each random level for this long [s]
PRBS_SEED = 0            # RNG seed (reproducible excitation)

# ---- RLS settings ----
ZERO_INIT = False              # start weights at ZERO (honest from-scratch identification)
FORGETTING_FACTOR = 0.999     # <1 keeps the fit adaptive
INITIAL_COVARIANCE = 2.0      # prior uncertainty on the weights
SIGMA_V_DOT = 2.0             # measurement-noise std for the v_dot channel (RLS weighting)
SIGMA_OMEGA_DOT = 0.5         # measurement-noise std for the omega_dot channel (RLS weighting)


# ============================================================
# Excitation signal -- PRBS only (controller-tau convention)
# Returns fn(t_local, state) -> tau; open-loop, so `state` is ignored.
# ============================================================

def prbs(amp, dwell, rng):
    """Pseudo-random piecewise-constant torque: a fresh random level every `dwell` s."""
    st = {"tnext": -1.0, "val": 0.0}

    def f(tl, s):
        if tl >= st["tnext"]:
            st["val"] = amp * float(rng.uniform(-1.0, 1.0))
            st["tnext"] = tl + dwell
        return st["val"]

    return f


# ============================================================
# MuJoCo helpers
# ============================================================

def get_joint_addresses(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def get_actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        raise RuntimeError(f"Actuator not found: {name}")
    return aid


def get_sensor_adr(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        raise RuntimeError(f"Sensor not found: {name}")
    return int(model.sensor_adr[sid])


# ============================================================
# Training run
# ============================================================

# history column layout
COLS = ["t", "x", "v", "theta", "omega", "tau",
        "v_dot_meas", "v_dot_hat", "v_dot_err",
        "omega_dot_meas", "omega_dot_hat", "omega_dot_err",
        "no_update"]
NAMES = ["b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
         "a_g", "a_tau", "a_omega", "a_v", "a_0"]
NC = len(NAMES)   # number of RLS coefficients (10)


def main():
    p = WheelieParams()
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    xq, xv = get_joint_addresses(model, "root_x")
    zq, _ = get_joint_addresses(model, "root_z")
    pq, pv = get_joint_addresses(model, "root_pitch")
    drive = get_actuator_id(model, "drive_motor")
    ctrl_min = float(model.actuator_ctrlrange[drive, 0])
    ctrl_max = float(model.actuator_ctrlrange[drive, 1])

    # IMU handles (for ACCEL_SOURCE="imu") + gravity vector (for the accelerometer).
    acc_adr = get_sensor_adr(model, "imu_acc")
    gyro_adr = get_sensor_adr(model, "imu_gyro")
    imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
    g_world = np.array(model.opt.gravity, dtype=float)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    ctrl_dt_actual = ctrl_steps * sim_dt
    print(f"RLS accel target: ACCEL_SOURCE={ACCEL_SOURCE!r}")

    a_nom = nominal_rls_parameters(p)
    a_rls = np.zeros_like(a_nom) if ZERO_INIT else a_nom.copy()
    P_rls = INITIAL_COVARIANCE * np.eye(a_nom.shape[0])
    print(f"RLS init: {'ZERO' if ZERO_INIT else 'analytical nominal'}  | "
          f"forgetting={FORGETTING_FACTOR}")

    # mutable run state
    state = {
        "last_state": None,
        "tau_applied": 0.0,
        "info": None,
        "control_count": 0,
        "last_gyro": None,     # previous gyro pitch-rate (for imu omega_dot differentiation)
    }
    history = []

    def read_state():
        x = float(data.qpos[xq])
        v = float(data.qvel[xv])
        theta = float(data.qpos[pq])
        omega = float(data.qvel[pv])
        return np.array([x, v, theta, omega], dtype=float)

    def read_accel():
        """Measured [v_dot, omega_dot] for the CURRENT (state, ctrl). Call AFTER a
        mj_forward so qacc/sensordata are consistent with data.qpos/qvel/ctrl."""
        if ACCEL_SOURCE == "qacc":
            return float(data.qacc[xv]), float(data.qacc[pv])
        # "imu": accelerometer -> world-x linear accel; gyro -> pitch rate (differentiated).
        a_meas = np.asarray(data.sensordata[acc_adr:acc_adr + 3], dtype=float)
        R = np.asarray(data.site_xmat[imu_site], dtype=float).reshape(3, 3)
        v_dot = float((R @ a_meas + g_world)[0])     # remove gravity, rotate to world, take x
        omega = float(data.sensordata[gyro_adr + 1])   # gyro y-axis = pitch rate
        if state["last_gyro"] is None:
            omega_dot = 0.0
        else:
            omega_dot = (omega - state["last_gyro"]) / ctrl_dt_actual
        state["last_gyro"] = omega
        return v_dot, omega_dot

    def reset_pose():
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[zq] = INITIAL_Z
        data.ctrl[drive] = 0.0
        mujoco.mj_forward(model, data)
        state["last_state"] = None      # skip the cross-teleport derivative
        state["tau_applied"] = 0.0
        state["info"] = None            # don't log a stale error on the first post-reset row
        state["last_gyro"] = None       # restart the gyro differentiation after the teleport

    def _rls_step(state_prev, tau, state_next, y_dot_meas):
        nonlocal a_rls, P_rls
        a_rls, P_rls, info = rls_update(
            state_prev=state_prev,
            tau=tau,
            state_next=state_next,
            dt=ctrl_dt_actual,
            a=a_rls,
            P=P_rls,
            forgetting_factor=FORGETTING_FACTOR,
            sigma_v_dot=SIGMA_V_DOT,
            sigma_omega_dot=SIGMA_OMEGA_DOT,
            y_dot_meas=y_dot_meas,
        )
        state["info"] = info

    def control_update(tau_fn, t_local):
        s_now = read_state()

        # PRBS torque for this step (open-loop: ignores s_now).
        tau = float(np.clip(tau_fn(t_local, s_now), p.tau_min, p.tau_max))
        ctrl = float(np.clip(tau, ctrl_min, ctrl_max))
        data.ctrl[drive] = ctrl

        if ACCEL_SOURCE == "finite_diff":
            # legacy: interval update phi(state_prev, tau_applied) -> finite-diff(state_prev->s_now)
            if state["last_state"] is not None:
                _rls_step(state["last_state"], state["tau_applied"], s_now, None)
        else:
            # measured-accel: sample qacc/IMU consistent with THIS (s_now, tau), then do an
            # INSTANTANEOUS matched update phi(s_now, tau) -> measured accel (no finite diff).
            mujoco.mj_forward(model, data)
            vdot, wdot = read_accel()
            _rls_step(s_now, tau, s_now, np.array([vdot, wdot], dtype=float))

        info = state["info"]

        if info:
            v_dot_measured     = info["v_dot_measured"]
            v_dot_hat          = info["v_dot_hat"]
            v_dot_error        = info["v_dot_error"]
            omega_dot_measured = info["omega_dot_measured"]
            omega_dot_hat      = info["omega_dot_hat"]
            omega_dot_error    = info["omega_dot_error"]
            no_rls_update      = 0.0
        else:
            v_dot_measured     = 0.0
            v_dot_hat          = 0.0
            v_dot_error        = 0.0
            omega_dot_measured = 0.0
            omega_dot_hat      = 0.0
            omega_dot_error    = 0.0
            no_rls_update      = 1.0   # first post-reset step: no RLS update

        row = np.array([
            data.time,
            s_now[0], s_now[1], s_now[2], s_now[3],
            tau,
            v_dot_measured,     v_dot_hat,     v_dot_error,
            omega_dot_measured, omega_dot_hat, omega_dot_error,
            no_rls_update,
        ], dtype=float)

        history.append(np.concatenate([row, a_rls]))   # 13 logs + 10 current weights

        H = np.array(history)

        CSV_PATH = Path(__file__).with_name("rls_accel_short.csv")

        v_dot_hat          = H[:, 7]
        omega_dot_hat      = H[:, 10]
        v_dot_measured     = H[:, 6]
        omega_dot_measured = H[:, 9]

        table = np.column_stack([v_dot_hat, omega_dot_hat, v_dot_measured, omega_dot_measured])
        header = "v_dot_hat,omega_dot_hat,v_dot_measured,omega_dot_measured"

        np.savetxt(CSV_PATH, table, delimiter=",", header=header, comments="", fmt="%.6f")
        #print(f"Saved accel CSV: {CSV_PATH.name}")

        if state["control_count"] % PRINT_EVERY_N == 0:
            ve = info["v_dot_error"] if info else 0.0
            we = info["omega_dot_error"] if info else 0.0
            # print(f"t={data.time:6.2f} | x={s_now[0]:6.2f} | v={s_now[1]:6.2f} | "
            #       f"pitch={math.degrees(s_now[2]):8.1f} deg | omega={s_now[3]:7.2f} | "
            #       f"tau={tau:6.2f} | err[v={ve:7.3f} w={we:7.3f}]")

        #print("[state now]: ", info["omega_dot_hat"])

        state["tau_applied"] = tau
        state["last_state"] = s_now
        state["control_count"] += 1

    def run_prbs(viewer=None):
        reset_pose()
        signal = prbs(PRBS_AMP, PRBS_DWELL, np.random.default_rng(PRBS_SEED))
        t0 = data.time
        nsub = int(round(DURATION / sim_dt))
        print(f"\n--- PRBS excitation  [{DURATION:.1f}s | amp={PRBS_AMP} dwell={PRBS_DWELL}] ---")
        for s in range(nsub):
            if viewer is not None and not viewer.is_running():
                return
            if s % ctrl_steps == 0:
                control_update(signal, data.time - t0)
            start = time.time()
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                sleep = sim_dt - (time.time() - start)
                if sleep > 0:
                    time.sleep(sleep)

    if RENDER:
        with mj_viewer.launch_passive(model, data) as viewer:
            run_prbs(viewer)
    else:
        run_prbs(None)

    H = np.array(history)
    print_results(H, a_rls, a_nom)
    np.savez(NPZ_PATH, a_rls=a_rls, P_rls=P_rls, a_nom=a_nom)
    print(f"Saved learned weights: {NPZ_PATH.name}")
    print("a_rls =", np.array2string(a_rls, precision=6, separator=", "))
    plot_results(H, a_nom)


# ============================================================
# Reporting
# ============================================================

def print_results(H, a_rls, a_nom):
    used = H[H[:, 12] < 0.5]          # rows where the RLS actually updated (not the first post-reset step)
    rmse_v = float(np.sqrt(np.mean(used[:, 8] ** 2))) if len(used) else float("nan")
    rmse_w = float(np.sqrt(np.mean(used[:, 11] ** 2))) if len(used) else float("nan")

    print("\n========== RLS identification result ==========")
    print(f"control steps: {len(H)}  |  used: {len(used)}")
    print(f"one-step RMSE   v_dot={rmse_v:.4f} m/s^2   omega_dot={rmse_w:.4f} rad/s^2")
    print("\nv_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0")
    print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    for i, name in enumerate(NAMES):
        print(f"  {name:10s} learned: {a_rls[i]: .6f} | nominal: {a_nom[i]: .6f}")
    print("===============================================")


def plot_results(H, a_nom):
    t = H[:, 0]
    coeffs = H[:, 13:13 + NC]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(13, 15))

    # # 0) states: forward/backward (v) + pitch (theta)
    # ax = axs[0]
    # ax.plot(t, H[:, 2], color="C0", label="v [m/s]")
    # ax.axhline(0, color="0.85", lw=0.8)
    # ax.set_ylabel("v [m/s]", color="C0")
    # axr = ax.twinx()
    # axr.plot(t, np.degrees(H[:, 3]), color="C3", lw=1.0, label="pitch [deg]")
    # axr.set_ylabel("pitch [deg]", color="C3")
    # ax.set_title("RLS training: PRBS excitation (v forward/backward, pitch)")

    # 1) excitation torque
    axs[0].plot(t, H[:, 5], color="C2")
    axs[0].set_ylabel("tau [Nm]")

    # 2) v_dot channel: measured vs RLS vs error
    axs[1].plot(t, H[:, 6], color="0.3", marker="*", ls="none", ms=5, label="measured")
    axs[1].plot(t, H[:, 7], color="C0", marker=".", ls="none", ms=5, label="RLS fit")
    axs[1].plot(t, H[:, 8], color="C3", ls=":", label="error")
    axs[1].set_ylabel("v_dot [m/s$^2$]")
    axs[1].legend(fontsize=8, ncol=3)

    # 3) omega_dot channel: measured vs RLS vs error
    axs[2].plot(t, H[:, 9], color="0.3", marker="*", ls="none", ms=5, label="measured")
    axs[2].plot(t, H[:, 10], color="C0", marker=".", ls="none", ms=5, label="RLS fit")
    axs[2].plot(t, H[:, 11], color="C3", ls=":", label="error")
    axs[2].set_ylabel("omega_dot [rad/s$^2$]")
    axs[2].legend(fontsize=8, ncol=3)

    # # 4) v_dot weights
    # for i in range(5):
    #     line, = axs[4].plot(t, coeffs[:, i], label=NAMES[i])
    #     axs[4].axhline(a_nom[i], color=line.get_color(), ls=":", lw=0.8)
    # axs[4].set_ylabel("v_dot weights")
    # axs[4].legend(fontsize=8, ncol=5)

    # # 5) omega_dot weights
    # for i in range(5, NC):
    #     line, = axs[5].plot(t, coeffs[:, i], label=NAMES[i])
    #     axs[5].axhline(a_nom[i], color=line.get_color(), ls=":", lw=0.8)
    # axs[5].set_ylabel("omega_dot weights")
    # axs[5].set_xlabel("time [s]")
    # axs[5].legend(fontsize=8, ncol=6)

    for a in axs:
        a.grid(True, alpha=0.3)

    fig.suptitle("Two-output full-dynamics RLS -- PRBS training (weights + one-step error)")
    fig.tight_layout()
    try:
        IMG_PATH.parent.mkdir(exist_ok=True)
        fig.savefig(IMG_PATH, dpi=1000, bbox_inches="tight")
        print(f"Saved figure: {IMG_PATH}")
    except PermissionError:
        alt = Path("/tmp/rls_training_short.png")
        fig.savefig(alt, dpi=150, bbox_inches="tight")
        print(f"[warn] {IMG_PATH.parent} not writable; saved to {alt}")
    plt.show()


if __name__ == "__main__":
    main()
