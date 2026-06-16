#!/usr/bin/env python3
"""
wheelie_gp_climb.py
-------------------
MuJoCo driver for the obstacle-climbing wheelie task. This file only does the
"plumbing": run the simulator, read state, drive the control loop, and log.

The control stack is split across modules so each piece is easy to debug:

    rls.py          online RLS identification of v_dot and omega_dot (always on)
    SSGP.py         streaming sparse variational GP for the contact residual
                    (recursive-FITC legacy: online_sparseGP.py)
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

import mujoco
import mujoco.viewer as mj_viewer

import numpy as np

# everything is local to this folder (self-contained NMPC stack)
from nmpc import WheelieParams, MPCConfig, WheelieNMPC, nominal_rls_seeds
from rls import RLS, RLSConfig, omega_regressor, v_regressor
# streaming variational (VFE) sparse GP (SSGP.py) and the recursive-FITC legacy
# (online_sparseGP.py) are fully INDEPENDENT modules; each config builds its own
# learner via gp_cfg.build(). The NMPC's reliable default is FITC (see GP below).
from SSGP import SSGPConfig, AdaptiveSSGPConfig


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
GOAL_X = 8.0                        # target x position B [m]


# ============================================================
# CONTROLLER TUNING
# ------------------------------------------------------------
# Every knob you would sweep lives here, in one place. Edit a value and
# re-run; you should not need to open rls.py / nmpc.py / SSGP.py.
# (Defaults shown explicitly so a sweep is just a one-line change.)
# ============================================================

# -- Physical model + actuator/state limits (full list in nmpc.py) ----------
#    v_max is capped low so the truck approaches obstacles at a controlled
#    speed -- ramming at high speed launches/flips it instead of climbing.
# theta_min=-30deg: braking induces a nose-DOWN pitch, so the old theta_min=0
# bound made the brake trajectory infeasible (the truck could never stop). Allow
# some nose-down so the NMPC can brake; it can still wheelie up to theta_max.
PARAMS = WheelieParams(v_max=1.5, v_min=-1.5, theta_min=math.radians(-30.0))

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
    q_v=20.0,           # velocity-REFERENCE tracking gain (climb authority comes
                        #   from v_cruise now, so q_v can be high for a crisp stop)
    q_theta=6.0,        # pitch: low, so the truck is free to rear into a wheelie
    q_omega=60.0,       # pitch-RATE damping: catches the wheelie before it flips.
                        #   THE critical knob -- too low flips, too high won't climb
    r_tau=0.05,         # torque effort (small -> allow a strong push)
    r_dtau=1.0,         # torque-rate (smoothness)
    q_terminal_theta=6.0,
    q_terminal_omega=60.0,
    ipopt_max_iter=80,
    # --- CLEARS ALL THREE OBSTACLES AND STOPS (fixed weights, no PD/scheduling) ---
    # goal-distance velocity reference (brakes to a stop) + omega-aware flip penalty
    # (allows a tall HELD wheelie to climb but kills the tip-over rotation) + a
    # velocity barrier. ALL THREE are load-bearing (ablation: drop any one -> stall
    # or flip or runaway). The thing that makes them actually work is COLD-STARTING
    # the solver (MPCConfig.warm_start=False, in nmpc.py): with a warm start IPOPT
    # gets trapped in the cruising minimum and runs away; cold-started it finds the
    # brake/reverse trajectory. Verified deterministic+fast: climbs box(1)+box(3)+
    # cylinders(5) and stops at final_x~8.1, max|pitch|~57deg, no flip, no runaway.
    v_ref_gain=0.5, v_cruise=1.3,
    q_flip=5.0e3, theta_soft_deg=93.0,        # static cap is now just a backstop
    q_flipw=2.0e3, theta_climb_deg=55.0,      # omega-aware: the real flip guard
    q_vbar=5.0e3, v_hard=1.2,                 # tight speed barrier -> no runaway
)
# NOTE: this point CLIMBS the 0.20 m step but cannot also stop AT the goal --
# climbing needs forward authority (low q_v) while stopping needs braking
# (high q_v), and one fixed weight set can't switch between them. A clean
# climb-AND-stop needs phase-dependent behavior (see the chat summary).

# -- streaming sparse GP residual model (recursive FITC -> online_sparseGP.py) -
# The NMPC's RELIABLE default is FITC. The variational SSGP (SSGP.py) is the
# default for the MPPI, but it DESTABILISES this NMPC: the NMPC bakes its GP kernel
# at COMPILE time (it is NOT re-synced to the GP's fitted kernel each step, the way
# the MPPI rollout is), so it is sensitive to the residual's SHAPE -- and the VFE
# swap is knife-edge (sn2_frac 2 stops, 2.5 RUNS AWAY, 3 stops, >=4 runs away).
# Keep FITC here. To make SSGPConfig()/AdaptiveSSGPConfig() reliable on the NMPC,
# nmpc.py would need to take the GP kernel as a runtime parameter (out of scope).
GP = SSGPConfig(
    max_points=20,            # inducing-set size M (keep small for real-time)
    warmup=60,                # steps buffered before the SGP fits its kernel
    sn2_frac=2.0,             # noise/signal -> regularise irreducible residual ~0
    lengthscales=(0.30, 2.0, 1.0, 3.0),   # NMPC rollout kernel ARD [theta,omega,v,tau]
    sf2=4.0,                  # NMPC rollout kernel scale
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
# Main simulation
# ============================================================

def main():
    # All tuning comes from the CONTROLLER TUNING block at the top of the file.
    p, cfg, gp_cfg, rls_cfg = PARAMS, MPC, GP, RLS_CFG

    nmpc = WheelieNMPC(p, cfg, gp_cfg)

    # streaming sparse GP residual learner (online): the config builds its own
    # (here StreamingGPConfig -> recursive FITC; SSGPConfig -> variational VFE).
    # sn2_frac regularises the (proven irreducible) residual toward ~0; the NMPC's
    # compile-time kernel is harmless since alpha ~ 0 -> mean ~ 0. Drop-in.
    gp = gp_cfg.build(n_features=4)

    # Two always-on RLS estimators, seeded from nominal physics.
    a0, b0 = nominal_rls_seeds(p)
    rls_w = RLS(a0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)  # omega_dot
    rls_v = RLS(b0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)  # v_dot

    # Fixed goal state B = [x_goal, stop, level, no spin]. The NMPC tracks this
    # every step; reaching it (q_x) is what drives the truck over the obstacle.
    GOAL = np.array([GOAL_X, 0.0, 0.0, 0.0], dtype=float)

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
    gp_v_std = gp_w_std = float(np.sqrt(gp_cfg.sf2))   # per-channel predictive std (plot bands)
    res_v = res_w = 0.0                                # measured residual the GP is fed (plot)
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
        nonlocal last_state, gp_v, gp_omega, gp_std, gp_v_std, gp_w_std, res_v, res_w
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
            res_v, res_w = r_v, r_w                 # measured residual (for the residual plot)
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
        gp_v, gp_v_std, gp_omega, gp_w_std = gp.predict_channels(z_now)   # per-channel mean+std

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
            "gp_v_std": float(gp_v_std), "gp_w_std": float(gp_w_std),
            "r_v": float(res_v), "r_omega": float(res_w),   # measured residual (GP target)
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
                    log_step()            # log once per CONTROL step (0.05 s)
                mujoco.mj_step(model, data)
                viewer.sync()
                sleep_time = sim_dt - (time.time() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                k += 1
    else:
        for k in range(n_steps):
            if k % ctrl_steps == 0:
                control_update()
                log_step()                # log once per CONTROL step (0.05 s)
            mujoco.mj_step(model, data)

    final_pitch = PITCH_SIGN * float(data.qpos[root_pitch_qid])
    print("\nFinal state")
    print(f"x      = {float(data.qpos[root_x_qid]):.3f} m")
    print(f"v      = {float(data.qvel[root_x_vid]):.3f} m/s")
    print(f"pitch  = {math.degrees(final_pitch):.2f} deg")
    print(f"GP dictionary points used: {gp.n_active}")
    save_history_csv(history, CSV_PATH)
    return history


if __name__ == "__main__":
    main()
