#!/usr/bin/env python3
"""
wheelie_gp_climb.py
-------------------
MuJoCo driver for the obstacle-climbing wheelie task. This file only does the
"plumbing": run the simulator, read state, drive the control loop, and log.

The control stack is split across modules so each piece is easy to debug:

    rls.py          online RLS identification of v_dot and omega_dot (always on)
    gp_residual.py  streaming sparse GP for the contact residual
    nmpc.py         the NMPC predictor, cost and IPOPT solver

Control loop, each control step:
    1. measure v_dot, omega_dot over the last interval
    2. update BOTH RLS channels (no freeze) and feed the GP the residual
    3. predict the GP residual at the current state (logging / diagnostics)
    4. solve the NMPC toward a fixed goal B -- the goal weight q_x is what
       drives the truck toward B (and over obstacles); RLS + GP give the dynamics
"""

import math
import time
from pathlib import Path
import csv

import numpy as np

from nmpc import WheelieParams, MPCConfig, WheelieNMPC, nominal_rls_seeds
from rls import RLS, RLSConfig, omega_regressor, v_regressor
from gp_residual import GPResidual, GPConfig

try:
    import mujoco
    import mujoco.viewer as mj_viewer
    _HAS_MUJOCO = True
except Exception:
    _HAS_MUJOCO = False


# ============================================================
# Settings
# ============================================================

XML_PATH = Path(__file__).with_name("monster_truck_flip_2d.xml")
CSV_PATH = Path(__file__).with_name("wheelie_gp_climb_log.csv")

RENDER = True
SIM_TIME = 40.0
CTRL_DT = 0.05
PRINT_EVERY_N_CONTROLS = 5

PITCH_SIGN = -1.0
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0

INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_ROOT_PITCH_DEG = 0.0

# ---- Goal: drive from start (A) to point B at this x position. The NMPC is a
#      pure goal-reacher -- MPC.q_x below sets how hard it is pulled to the goal,
#      which is also what gives it the authority to drive over an obstacle. ----
GOAL_X = 5.0                        # target x position B [m]


# ============================================================
# CONTROLLER TUNING
# ------------------------------------------------------------
# Every knob you would sweep lives here, in one place. Edit a value and
# re-run; you should not need to open rls.py / nmpc.py / gp_residual.py.
# (Defaults shown explicitly so a sweep is just a one-line change.)
# ============================================================

# -- Physical model + actuator/state limits (full list in nmpc.py) ----------
#    v_max is capped low so the truck approaches obstacles at a controlled
#    speed -- ramming at high speed launches/flips it instead of climbing.
PARAMS = WheelieParams(v_max=1.5, v_min=-1.5)

# -- RLS online identification (rls.py) -------------------------------------
RLS_CFG = RLSConfig(
    forgetting=0.9995,  # lambda in (0, 1]; ->1.0 adapts slower / more robust.
                        #   0.9995 keeps the violent climb maneuver from
                        #   corrupting the model (0.999 -> post-climb runaway)
    p0_scale=3.0,       # initial covariance P0 = p0_scale * I
)

# -- NMPC horizon, cost weights and solver (nmpc.py) ------------------------
MPC = MPCConfig(
    dt=CTRL_DT,
    N=20,               # prediction horizon (steps)
    # --- Tuned so the truck pops a CONTROLLED wheelie to climb the 0.20 m step.
    #     Narrow operating point: q_omega is knife-edge (50 or 70 flip, 60 climbs).
    #     Change these carefully; see the note at the bottom of this block. ---
    q_x=15.0,           # GOAL weight: pull toward GOAL_X
    q_v=8.0,            # speed regulation. Low -> enough authority to climb;
                        #   the price is the truck OVERSHOOTS the goal (see notes)
    q_theta=6.0,        # pitch: low, so the truck is free to rear into a wheelie
    q_omega=60.0,       # pitch-RATE damping: catches the wheelie before it flips.
                        #   THE critical knob -- too low flips, too high won't climb
    r_tau=0.05,         # torque effort (small -> allow a strong push)
    r_dtau=1.0,         # torque-rate (smoothness)
    q_terminal_theta=6.0,
    q_terminal_omega=60.0,
    ipopt_max_iter=80,
)
# NOTE: this point CLIMBS the 0.20 m step but cannot also stop AT the goal --
# climbing needs forward authority (low q_v) while stopping needs braking
# (high q_v), and one fixed weight set can't switch between them. A clean
# climb-AND-stop needs phase-dependent behavior (see the chat summary).

# -- Streaming sparse GP residual model (gp_residual.py) --------------------
GP = GPConfig(
    max_points=20,            # dictionary size M (keep small for real-time)
    sf2=4.0,                  # signal variance (kernel output scale)
    sn2=0.25,                 # observation noise / ridge
    novelty_thresh=0.85,      # add a new point only if this novel (0..1)
    activation_thresh=0.6,    # |residual| to count as a contact event
    lengthscales=(0.30, 2.0, 1.0, 3.0),   # ARD for z = [theta, omega, v, tau]
    refit_every=1,            # recompute alpha every k observations
)


# ============================================================
# MuJoCo helpers
# ============================================================

def get_joint_addresses(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint not found: {joint_name}")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def get_actuator_id(model, actuator_name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        raise RuntimeError(f"Actuator not found: {actuator_name}")
    return aid


# ============================================================
# CSV logging
# ============================================================

def save_history_csv(history, csv_path):
    if not history:
        print("No data to save."); return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader(); writer.writerows(history)
    print(f"Saved CSV log: {csv_path}")


# ============================================================
# Obstacle XML helper
# ============================================================

def add_obstacle_xml_snippet() -> str:
    """XML snippet to paste into <worldbody> of monster_truck_flip_2d.xml:
    a static step obstacle the truck must climb."""
    return """
  <!-- ===== Obstacle: a static step the truck must climb ===== -->
  <body name="obstacle" pos="1.5 0 0.05">
    <geom name="obstacle_geom" type="box" size="0.15 0.5 0.05"
          rgba="0.7 0.3 0.2 1" friction="1.0 0.01 0.001"/>
  </body>
"""


# ============================================================
# Main simulation
# ============================================================

def main():
    # All tuning comes from the CONTROLLER TUNING block at the top of the file.
    p, cfg, gp_cfg, rls_cfg = PARAMS, MPC, GP, RLS_CFG

    nmpc = WheelieNMPC(p, cfg, gp_cfg)

    # Streaming sparse GP residual model (online)
    gp = GPResidual(n_features=4, max_points=gp_cfg.max_points,
                    lengthscales=np.asarray(gp_cfg.lengthscales),
                    sf2=gp_cfg.sf2, sn2=gp_cfg.sn2,
                    novelty_thresh=gp_cfg.novelty_thresh,
                    activation_thresh=gp_cfg.activation_thresh)

    # Two always-on RLS estimators, seeded from nominal physics.
    a0, b0 = nominal_rls_seeds(p)
    rls_w = RLS(a0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)  # omega_dot
    rls_v = RLS(b0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)  # v_dot

    # Fixed goal state B = [x_goal, stop, level, no spin]. The NMPC tracks this
    # every step; reaching it (q_x) is what drives the truck over the obstacle.
    GOAL = np.array([GOAL_X, 0.0, 0.0, 0.0], dtype=float)

    if not _HAS_MUJOCO:
        print("MuJoCo not available in this environment.")
        print("This script is meant to run in your project next to the XML.")
        print("\nObstacle XML snippet to add to <worldbody>:")
        print(add_obstacle_xml_snippet())
        return

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    root_x_qid, root_x_vid = get_joint_addresses(model, "root_x")
    root_z_qid, root_z_vid = get_joint_addresses(model, "root_z")
    root_pitch_qid, root_pitch_vid = get_joint_addresses(model, "root_pitch")
    drive_id = get_actuator_id(model, "drive_motor")
    ctrl_min = float(model.actuator_ctrlrange[drive_id, 0])
    ctrl_max = float(model.actuator_ctrlrange[drive_id, 1])

    data.qpos[:] = 0.0; data.qvel[:] = 0.0
    data.qpos[root_x_qid] = INITIAL_X
    data.qpos[root_z_qid] = INITIAL_Z
    data.qpos[root_pitch_qid] = math.radians(INITIAL_ROOT_PITCH_DEG)
    data.ctrl[drive_id] = 0.0
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    ctrl_dt_actual = ctrl_steps * sim_dt
    n_steps = int(round(SIM_TIME / sim_dt))

    # Control / logging state
    tau_prev = tau_cmd = ctrl_cmd = 0.0
    solve_success = False
    control_count = 0
    history = []

    gp_v = gp_omega = 0.0
    gp_std = float(np.sqrt(gp_cfg.sf2))
    last_state = None                  # previous NMPC state [x, v, theta, omega]
    rls_err_v = rls_err_w = 0.0

    def read_controller_state():
        x = float(data.qpos[root_x_qid]); z = float(data.qpos[root_z_qid])
        v = float(data.qvel[root_x_vid]); z_dot = float(data.qvel[root_z_vid])
        raw_pitch = float(data.qpos[root_pitch_qid])
        raw_pitch_dot = float(data.qvel[root_pitch_vid])
        theta = PITCH_SIGN * raw_pitch
        omega = PITCH_SIGN * raw_pitch_dot
        return x, z, v, z_dot, theta, omega

    def read_nmpc_state():
        x, _, v, _, theta, omega = read_controller_state()
        return np.array([x, v, theta, omega], dtype=float)

    def control_update():
        nonlocal tau_prev, tau_cmd, ctrl_cmd, solve_success, control_count
        nonlocal last_state, gp_v, gp_omega, gp_std
        nonlocal rls_err_v, rls_err_w

        state_now = read_nmpc_state()
        x_now, v_now, theta_now, omega_now = state_now

        # --- 1. Learn dynamics: RLS (always on) + GP residual ---
        if last_state is not None:
            _, v_p, theta_p, omega_p = last_state

            # measured accelerations over the last control interval
            v_dot_meas = (v_now - v_p) / ctrl_dt_actual
            omega_dot_meas = (omega_now - omega_p) / ctrl_dt_actual

            # regressors at the PREVIOUS state with the torque that was applied
            phi_v = np.array(v_regressor(theta_p, v_p, tau_cmd, sin=np.sin))
            phi_w = np.array(omega_regressor(theta_p, omega_p, v_p, tau_cmd,
                                             cos=np.cos))

            # residual the GP must explain = measurement - current RLS model
            # (computed BEFORE the RLS update so the GP is not starved)
            r_v = v_dot_meas - rls_v.predict(phi_v)
            r_w = omega_dot_meas - rls_w.predict(phi_w)
            z_prev = np.array([theta_p, omega_p, v_p, tau_cmd], dtype=float)
            gp.observe(z_prev, r_v, r_w)
            if gp.n_seen % gp_cfg.refit_every == 0:
                gp.refit()

            # always adapt the linear model (no freeze)
            info_v = rls_v.update(phi_v, v_dot_meas)
            info_w = rls_w.update(phi_w, omega_dot_meas)
            rls_err_v, rls_err_w = info_v["error"], info_w["error"]

        # --- 2. GP prediction at current state (logging / diagnostics) ---
        z_now = np.array([theta_now, omega_now, v_now, tau_cmd], dtype=float)
        gp_v, gp_omega, gp_std = gp.predict(z_now)

        # --- 3. Solve NMPC toward the fixed goal B; q_x pulls us there and
        #        over whatever is in the way (RLS + GP supply the dynamics) ---
        tau, success = nmpc.solve(state_now, GOAL, tau_prev,
                                  rls_w.theta, rls_v.theta, gp.mpc_params())
        tau = float(np.clip(tau, p.tau_min, p.tau_max))
        tau_prev = tau

        ctrl = float(np.clip(ACTUATOR_SIGN * TAU_TO_CTRL * tau, ctrl_min, ctrl_max))
        data.ctrl[drive_id] = ctrl

        tau_cmd, ctrl_cmd, solve_success = tau, ctrl, bool(success["success"])
        last_state = state_now.copy()

        if control_count % PRINT_EVERY_N_CONTROLS == 0:
            print(f"t={data.time:6.3f} | goal_d={GOAL_X - x_now:6.3f} | x={x_now:6.3f} "
                  f"v={v_now:6.3f} | pitch={math.degrees(theta_now):6.1f} | "
                  f"tau={tau:7.3f} | gp_v={gp_v:6.2f} gp_w={gp_omega:6.2f} "
                  f"std={gp_std:5.2f} | M={gp.n_active}")
        control_count += 1

    def log_step():
        x, z, v, z_dot, theta, omega = read_controller_state()
        a, b = rls_w.theta, rls_v.theta
        history.append({
            "time": float(data.time),
            "x": x, "z": z, "x_dot": v, "z_dot": z_dot,
            "pitch_deg": math.degrees(theta), "pitch_dot": omega,
            "goal_x": GOAL_X, "dist_to_goal": GOAL_X - x,
            "tau_cmd": tau_cmd, "ctrl_cmd": ctrl_cmd,
            "solve_success": int(solve_success),
            "gp_v": gp_v, "gp_omega": gp_omega, "gp_std": gp_std,
            "gp_active_points": gp.n_active,
            "rls_err_v": float(rls_err_v), "rls_err_omega": float(rls_err_w),
            # angular model a = [cos(theta), tau, omega, v, 1]
            "a_g": float(a[0]), "a_tau": float(a[1]), "a_omega": float(a[2]),
            "a_v": float(a[3]), "a_0": float(a[4]),
            # linear model b = [tau, v, sin(theta), 1]
            "b_tau": float(b[0]), "b_v": float(b[1]),
            "b_sin": float(b[2]), "b_0": float(b[3]),
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
    print("\nFinal state")
    print(f"x      = {float(data.qpos[root_x_qid]):.3f} m")
    print(f"v      = {float(data.qvel[root_x_vid]):.3f} m/s")
    print(f"pitch  = {math.degrees(final_pitch):.2f} deg")
    print(f"GP dictionary points used: {gp.n_active}")
    save_history_csv(history, CSV_PATH)


if __name__ == "__main__":
    main()
