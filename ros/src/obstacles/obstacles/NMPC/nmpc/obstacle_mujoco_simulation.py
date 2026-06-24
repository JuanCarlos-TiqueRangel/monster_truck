#!/usr/bin/env python3

import math
import time
from pathlib import Path
import csv

import numpy as np
import mujoco
import mujoco.viewer as mj_viewer

from params_mujoco import WheelieParams, MPCConfig
from rls import nominal_rls_parameters, rls_update
from nmpc import NMPC            # <- NMPC (IPOPT) controller; swapped for MPPI below
#from mppi import MPPI
#from SSGP import SSGPConfig                  # streaming sparse VARIATIONAL GP (VFE) -- default
from SGP import SGPConfig as SSGPConfig       # <- swap in: streaming sparse FITC GP (compare VFE vs FITC)
from prewheelie_learner import PreWheelieLearner   # episodic optimiser for the pre-wheelie angle

# ============================================================
# Easy-to-debug settings
# ============================================================

XML_PATH = Path(__file__).with_name("monster_truck_flip_2d.xml")
CSV_PATH = Path(__file__).with_name("obstacle_mujoco.csv")
MODEL_PATH = Path(__file__).with_name("obstacle_model.npz")

# Set LOAD_MODEL=True to RESUME from a model saved by a previous run (the GP
# obstacle knowledge + RLS dynamics). False -> learn from scratch.
LOAD_MODEL = False

# Set GP_ENABLED=False to ZERO the GP's contribution to the NMPC rollout (RLS-only
# control). The GP still learns + logs, so you can A/B it: if behaviour is still good
# with the GP off, the RLS is doing the work; if it gets worse, the GP matters.
GP_ENABLED = True

# Hard-freeze the RLS dynamics weights: hold a_rls at its INITIAL values for the whole
# run (no online adaptation -- a "fixed weights" controller). The GP still learns: the
# residual y - H@a_fixed against the FROZEN weights is fed to it, so freezing pins the
# SMOOTH dynamics while the GP keeps mapping the obstacle. Set the fixed values via
# nominal_rls_parameters() (or LOAD_MODEL=True to freeze at a previously learned set).
RLS_FREEZE = True

# Acceleration TARGET for the RLS (= the GP residual too). "qacc" = MuJoCo exact accel
# (best in sim), "imu" = on-board IMU (accelerometer->v_dot, gyro(diff)->omega_dot;
# hardware-realistic), "finite_diff" = legacy (v_next-v_prev)/dt. Measured targets remove
# the differentiation noise/half-step bias, so the RLS fit AND the GP residual get cleaner.
# Use "qacc" for SIM (exact ground truth); "imu" is the hardware-realistic path.
ACCEL_SOURCE = "qacc"

RENDER = False
SIM_TIME = 20.0          # per-episode time cap [s]
CTRL_DT = 0.05
PRINT_EVERY_N_CONTROLS = 5

# Episodic learning: the GP (and RLS) persist across episodes, the MuJoCo state is
# reset each episode. Episode 1 learns where the obstacle is; episode 2 reuses that.
N_EPISODES = 50

# The UNIQUE NMPC reference: reach the goal position GOAL_X. There is NO pitch
# reference -- a wheelie is only popped if the learned dynamics/GP make rearing the
# cheapest way to reach the goal (e.g. to climb the obstacle).
GOAL_X = 10.0
GOAL_TOL = 0.15          # an episode ends early once |x - GOAL_X| < GOAL_TOL

# GP-DISCOVERED pre-wheelie: where the GP predicts a blockage > OBS_BLOCK [m/s^2], steer the
# pitch reference to a controlled climb angle at the obstacle the GP LEARNED -- no hardcoded
# location. THETA_OBS_DEG = the FIXED angle (0 -> off, rams the box).
THETA_OBS_DEG = 0.0
OBS_BLOCK = 8.0

# PREWHEELIE_LEARN=True: an episodic learner (prewheelie_learner.py) DISCOVERS the fastest-safe
# pre-wheelie angle online, improving the obstacle-crossing time each episode (ignores
# THETA_OBS_DEG, sets the angle itself). False -> fixed THETA_OBS_DEG.
PREWHEELIE_LEARN = True

# Your MuJoCo root_pitch is negative during a backward wheelie.
# This makes the controller see backward wheelie as positive pitch.
PITCH_SIGN = -1.0

# If the motor acts in the wrong direction, change this to -1.0.
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0

INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_ROOT_PITCH_DEG = 0.0


# ============================================================
# Small MuJoCo helpers
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


def get_sensor_adr(model: mujoco.MjModel, sensor_name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    if sid < 0:
        raise RuntimeError(f"Sensor not found: {sensor_name}")
    return int(model.sensor_adr[sid])


def empty_rls_info() -> dict:
    return {
        "v_dot_raw": 0.0,
        "omega_dot_raw": 0.0,
        "v_dot_measured": 0.0,
        "omega_dot_measured": 0.0,
        "v_dot_hat": 0.0,
        "omega_dot_hat": 0.0,
        "v_dot_error": 0.0,
        "omega_dot_error": 0.0,
        "skipped": True,
    }


# ============================================================
# CSV logging
# ============================================================

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


# ============================================================
# Model persistence (GP obstacle knowledge + RLS dynamics)
# ============================================================

def save_model(path: Path, gp, a_rls: np.ndarray, P_rls: np.ndarray) -> None:
    """Checkpoint the learned model so a later run can resume from it. Saves the GP
    posterior (state_dict) plus the RLS state to a single .npz. No-op if the GP has
    not warmed up yet (nothing learned to save)."""
    sd = gp.state_dict()
    if not sd:
        return
    np.savez(path, a_rls=a_rls, P_rls=P_rls,
             **{f"gp_{k}": v for k, v in sd.items()})


def load_model(path: Path, gp) -> tuple[np.ndarray, np.ndarray]:
    """Restore a model saved by save_model() into `gp`; returns (a_rls, P_rls)."""
    d = np.load(path, allow_pickle=False)
    gp_sd = {k[len("gp_"):]: d[k] for k in d.files if k.startswith("gp_")}
    gp.load_state_dict(gp_sd)
    return d["a_rls"].copy(), d["P_rls"].copy()


def plot_prewheelie_learning(learner) -> None:
    """Plot the episodic pre-wheelie learning: crossing time (down=faster) and the discovered
    angle, per episode. Also saves the learner's history JSON next to the model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = learner.history
    eps = list(range(len(h)))
    tc = [r["t_cross"] for r in h]
    bt = [r["best_time"] for r in h]
    ang = [r["angle"] for r in h]
    safe = [r["safe"] for r in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(eps, tc, "o-", ms=4, color="0.6", label="episode crossing time")
    a1.plot(eps, bt, "-", lw=2.5, color="C0", label="best-so-far")
    a1.set_xlabel("episode"); a1.set_ylabel("obstacle-crossing time [s]")
    a1.set_title("learning curve  (DOWN = faster)"); a1.grid(True, alpha=.3); a1.legend(fontsize=9)
    a2.plot(eps, ang, "o-", ms=4, color="C3")
    for e, s in zip(eps, safe):                                  # mark unsafe (flip) episodes
        if not s:
            a2.scatter([e], [ang[e]], s=60, facecolors="none", edgecolors="k", label="_unsafe")
    a2.axhline(learner.best_angle, color="0.7", ls="--", label=f"converged {learner.best_angle:.0f}$\\degree$")
    a2.set_xlabel("episode"); a2.set_ylabel("pre-wheelie angle [deg]")
    a2.set_title("discovered angle  (o = tried, circle = flipped)"); a2.grid(True, alpha=.3); a2.legend(fontsize=9)
    fig.suptitle(f"Episodic pre-wheelie learning  ->  best {learner.best_angle:.0f}$\\degree$ at "
                 f"{learner.best_time:.2f}s crossing")
    fig.tight_layout()
    img = Path(__file__).with_name("images")
    img.mkdir(exist_ok=True)
    fig.savefig(img / "prewheelie_learning.png", dpi=140, bbox_inches="tight")
    learner.save(Path(__file__).with_name("prewheelie_learner.json"))
    print(f"Saved: {img / 'prewheelie_learning.png'}  +  prewheelie_learner.json")


# ============================================================
# Main simulation
# ============================================================

def main(seed=None, overrides=None):
    """overrides: optional dict patching MPCConfig fields (and 'theta_obs_deg') so an
    outer optimiser (cem_learn.py) can sweep cost parameters without editing the module."""
    p = WheelieParams()
    rng = np.random.default_rng(seed) if seed is not None else None   # init jitter (multi-seed eval)

    # MBRL REWARD (maximise return = minimise time). The planner optimises forward PROGRESS
    # (w_progress) against the LEARNED SSGP model -- not a hand "stay-flat" cost. Because the
    # model knows rearing reduces the obstacle blockage, the speed-maximising plan is to WHEELIE
    # at the obstacle, so the maneuver EMERGES from the reward+model (no hand-set angle/location).
    #   q_v=0       : NO slow-down penalty (the old q_v wanted v=0 -> it dawdled).
    #   w_progress  : the reward -- reward forward speed every step.
    #   q_theta SMALL: only a mild flat-preference, so it stays low on open ground (no blockage
    #                 to fight) but rears at the obstacle where the model says speed pays.
    #   q_flip/...   : the SAFETY part of the reward (don't tip over).
    cfg = MPCConfig(
        q_x=20.0,
        q_v=0.0,
        q_theta=5.0,
        q_omega=0.0,
        q_terminal_theta=0.0,
        q_terminal_omega=0.0,
        w_progress=15.0,
        q_flip=2000.0,
        theta_soft_deg=80.0,
        q_flipw=200.0,
        theta_climb_deg=55.0,
        theta_obs_deg=THETA_OBS_DEG, obs_block=OBS_BLOCK,
    )
    if overrides:                                  # CEM/optimiser candidate parameters
        for k, val in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, float(val))
            else:
                raise KeyError(f"main() override '{k}' is not an MPCConfig field")
    gp_cfg = SSGPConfig()
    nmpc = NMPC(p, cfg, gp_cfg)        # <- IPOPT NMPC (kept for reference)
    #nmpc = MPPI(p, cfg, gp_cfg)          # sampling-based MPPI, same interface (.solve / .last_solution)

    # Streaming sparse-variational GP residual learner. Feature
    # z = [x, theta, omega, v, tau] -- x first so the GP learns WHERE (in x) the
    # contact residual lives. Built ONCE and kept alive across episodes so episode 2
    # starts with episode 1's obstacle knowledge (rebuilding it would wipe the posterior).
    gp = gp_cfg.build(n_features=2)
    gp_params_zero = np.zeros_like(gp.mpc_params())   # GP_ENABLED=False -> RLS-only rollout
    if not GP_ENABLED:
        print("[GP DISABLED] NMPC rollout uses RLS only (the GP still learns + logs).")

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    root_x_qid, root_x_vid = get_joint_addresses(model, "root_x")
    root_z_qid, root_z_vid = get_joint_addresses(model, "root_z")
    root_pitch_qid, root_pitch_vid = get_joint_addresses(model, "root_pitch")
    drive_id = get_actuator_id(model, "drive_motor")

    ctrl_min = float(model.actuator_ctrlrange[drive_id, 0])
    ctrl_max = float(model.actuator_ctrlrange[drive_id, 1])

    # IMU handles (ACCEL_SOURCE="imu") + gravity vector (for the accelerometer conversion).
    acc_adr = get_sensor_adr(model, "imu_acc")
    gyro_adr = get_sensor_adr(model, "imu_gyro")
    imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_site")
    g_world = np.array(model.opt.gravity, dtype=float)

    # The unique NMPC reference: reach the goal (v_ref=0 so it stops there; the
    # theta/omega entries are ignored because their cost weights are 0).
    ref = np.array([GOAL_X, 0.0, 0.0, 0.0], dtype=float)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    ctrl_dt_actual = ctrl_steps * sim_dt

    # ---- RLS settings ----
    forgetting_factor = 0.9995
    initial_covariance = 3.0
    derivative_alpha = 0.0
    clip_parameters = False
    sigma_v_dot = 2.0
    sigma_omega_dot = 5.0

    # ---- learners that PERSIST across episodes (the knowledge carried over) ----
    a_nom = nominal_rls_parameters(p)
    a_rls = a_nom.copy()
    P_rls = initial_covariance * np.eye(10)
    # gp persists too (built above, outside the episode loop)

    # Optionally RESUME from a previously saved model (GP posterior + RLS state).
    if LOAD_MODEL and MODEL_PATH.exists():
        a_rls, P_rls = load_model(MODEL_PATH, gp)
        print(f"Loaded model from {MODEL_PATH.name} "
              f"(gp_ready={gp.ready}, n_active={gp.n_active})")
    elif LOAD_MODEL:
        print(f"[warn] LOAD_MODEL=True but {MODEL_PATH.name} not found -- learning from scratch")

    # ---- per-episode bookkeeping (reset by reset_episode each episode) ----
    tau_prev = 0.0
    tau_cmd = 0.0
    ctrl_cmd = 0.0
    solve_success = False
    control_count = 0
    filtered_y_dot = None
    last_control_state = None
    res_v_dot = 0.0
    res_omega_dot = 0.0
    gp_pred_v_dot_pre = 0.0          # (A) GP prediction at z_prev BEFORE the update
    gp_pred_omega_dot_pre = 0.0
    last_rls_info = empty_rls_info()
    last_accel = None           # measured accel matched to last_control_state (ACCEL_SOURCE!=finite_diff)
    last_gyro = None            # previous gyro pitch-rate (for imu omega_dot differentiation)

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

    def read_nmpc_state() -> np.ndarray:
        x, _, v, _, theta, omega = read_controller_state()
        return np.array([x, v, theta, omega], dtype=float)

    def read_accel() -> np.ndarray:
        """Measured [v_dot, omega_dot] for the CURRENT (state, ctrl). Call AFTER a
        mj_forward so qacc/sensordata are consistent with data.qpos/qvel/ctrl."""
        nonlocal last_gyro
        if ACCEL_SOURCE == "qacc":
            return np.array([float(data.qacc[root_x_vid]),
                             PITCH_SIGN * float(data.qacc[root_pitch_vid])], dtype=float)
        # "imu": accelerometer -> world-x linear accel; gyro -> pitch rate (differentiated).
        a_meas = np.asarray(data.sensordata[acc_adr:acc_adr + 3], dtype=float)
        R = np.asarray(data.site_xmat[imu_site], dtype=float).reshape(3, 3)
        v_dot = float((R @ a_meas + g_world)[0])     # remove gravity, rotate to world, take x
        omega = PITCH_SIGN * float(data.sensordata[gyro_adr + 1])   # gyro y-axis = pitch rate
        omega_dot = 0.0 if last_gyro is None else (omega - last_gyro) / ctrl_dt_actual
        last_gyro = omega
        return np.array([v_dot, omega_dot], dtype=float)

    def control_update():
        nonlocal tau_prev, tau_cmd, ctrl_cmd, solve_success, control_count
        nonlocal a_rls, P_rls, filtered_y_dot, last_rls_info, last_control_state
        nonlocal res_v_dot, res_omega_dot, gp_pred_v_dot_pre, gp_pred_omega_dot_pre
        nonlocal last_accel

        state_now = read_nmpc_state()

        # Update the RLS ONCE per control period (the weights are FROZEN, so this just
        # computes the residual y - H@a_fixed against the fixed model -- the GP's target).
        # The accel target is the measured accel sampled last step at (last_control_state,
        # tau_cmd) -- the matched instant for phi (see read_accel) -- or the finite diff.
        accel_ok = ACCEL_SOURCE == "finite_diff" or last_accel is not None
        if last_control_state is not None and accel_ok:
            a_saved, P_saved = (a_rls.copy(), P_rls.copy()) if RLS_FREEZE else (None, None)
            a_rls, P_rls, filtered_y_dot, last_rls_info = rls_update(
                state_prev=last_control_state,
                tau=tau_cmd,
                state_next=state_now,
                dt=ctrl_dt_actual,
                a=a_rls,
                P=P_rls,
                filtered_y_dot=filtered_y_dot,
                forgetting_factor=forgetting_factor,
                derivative_alpha=derivative_alpha,
                sigma_v_dot=sigma_v_dot,
                sigma_omega_dot=sigma_omega_dot,
                clip_parameters=clip_parameters,
                y_dot_meas=(None if ACCEL_SOURCE == "finite_diff" else last_accel),
                p=p,
            )
            if RLS_FREEZE:          # hold weights fixed (the residual in last_rls_info is the GP target)
                a_rls, P_rls = a_saved, P_saved

            # GP target = the velocity residual (the BLOCKAGE / deceleration the model
            # can't explain); it is the smooth, learnable obstacle signal (the omega
            # residual is impulsive noise -> stays ~0). Feature z = [x, theta] only:
            # the obstacle is a function of WHERE you are and HOW reared you are, so
            # samples at the same (x, theta) AVERAGE (in 5-D they scattered into noise).
            res_v_dot = float(last_rls_info["v_dot_error"])
            res_omega_dot = float(last_rls_info["omega_dot_error"])
            z_prev = np.array([
                last_control_state[0],   # x      (position -> obstacle location)
                last_control_state[2],   # theta  (pitch -> "rearing reduces blockage")
            ], dtype=float)
            # (A) predict at the SAME z that produced the residual, BEFORE the GP
            #     sees it -> a true one-step prediction error (r_* - gp_*_pred_pre).
            gp_pred_v_dot_pre, gp_pred_omega_dot_pre, _ = gp.predict(z_prev)

            gp.observe(z_prev, res_v_dot, res_omega_dot)

        gp_params = gp.mpc_params() if GP_ENABLED else gp_params_zero
        tau, info = nmpc.solve(state_now, ref, tau_prev, a_rls, gp_params)
        tau = float(np.clip(tau, p.tau_min, p.tau_max))
        tau_prev = tau

        # ACTUATOR_SIGN belongs only here, at the MuJoCo command interface.
        ctrl = ACTUATOR_SIGN * TAU_TO_CTRL * tau
        ctrl = float(np.clip(ctrl, ctrl_min, ctrl_max))
        data.ctrl[drive_id] = ctrl

        # Sample the measured accel for THIS (state_now, tau): mj_forward makes qacc/sensors
        # consistent with the just-set ctrl, so it's the matched target for phi next step.
        if ACCEL_SOURCE != "finite_diff":
            mujoco.mj_forward(model, data)
            last_accel = read_accel()

        tau_cmd = tau
        ctrl_cmd = ctrl
        solve_success = bool(info["success"])
        last_control_state = state_now.copy()

        if control_count % PRINT_EVERY_N_CONTROLS == 0:
            print(
                f"t={data.time:6.3f} | "
                f"x={state_now[0]:7.3f} | v={state_now[1]:7.3f} | "
                f"pitch={math.degrees(state_now[2]):8.2f} deg | "
                f"omega={state_now[3]:8.3f} | tau={tau:8.3f} | "
                f"ctrl={ctrl:8.3f} | success={info['success']} | "
                f"rls_err[v={last_rls_info['v_dot_error']:6.3f}, "
                f"w={last_rls_info['omega_dot_error']:6.3f}]"
            )

        control_count += 1

    def log_row(episode: int) -> dict:
        x, z, v, z_dot, theta, omega = read_controller_state()
        raw_pitch = float(data.qpos[root_pitch_qid])
        raw_pitch_dot = float(data.qvel[root_pitch_vid])

        # GP residual prediction at the current state (diagnostics).
        z_now = np.array([x, theta], dtype=float)
        gp_v_dot_pred, gp_omega_dot_pred, _ = gp.predict(z_now)

        return {
            "episode": int(episode),
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
            "goal_x": GOAL_X,
            "tau_cmd": tau_cmd,
            "ctrl_cmd": ctrl_cmd,
            "solve_success": int(solve_success),

            # RLS logs (two-output: v_dot and omega_dot).
            "v_dot_raw": float(last_rls_info["v_dot_raw"]),
            "v_dot_filtered": float(last_rls_info["v_dot_measured"]),
            "v_dot_rls": float(last_rls_info["v_dot_hat"]),
            "v_dot_error": float(last_rls_info["v_dot_error"]),
            "omega_dot_raw": float(last_rls_info["omega_dot_raw"]),
            "omega_dot_filtered": float(last_rls_info["omega_dot_measured"]),
            "omega_dot_rls": float(last_rls_info["omega_dot_hat"]),
            "omega_dot_error": float(last_rls_info["omega_dot_error"]),

            # v_dot coeffs: v_dot = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0
            "b_tau": float(a_rls[0]),
            "b_v": float(a_rls[1]),
            "b_abs_v": float(a_rls[2]),
            "b_tau_cos": float(a_rls[3]),
            "b_0": float(a_rls[4]),

            # omega_dot coeffs: omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0
            "a_g": float(a_rls[5]),
            "a_tau": float(a_rls[6]),
            "a_omega": float(a_rls[7]),
            "a_v": float(a_rls[8]),
            "a_0": float(a_rls[9]),

            # GP residual: measured target fed to the GP (r_*) and the GP's own
            # prediction at the current state. Plot these vs x to see the obstacle
            # the SSGP has localised (compare episode 0 vs episode 1).
            "r_v_dot": float(res_v_dot),
            "r_omega_dot": float(res_omega_dot),
            "gp_v_dot_pred": float(gp_v_dot_pred),
            "gp_omega_dot_pred": float(gp_omega_dot_pred),
            # (A) one-step prediction at z_prev BEFORE the GP saw that point;
            #     the one-step error is r_* - gp_*_pred_pre (clean target/pred pair).
            "gp_v_dot_pred_pre": float(gp_pred_v_dot_pre),
            "gp_omega_dot_pred_pre": float(gp_pred_omega_dot_pre),
            "gp_ready": int(gp.ready),
        }

    def reset_episode():
        """Reset the MuJoCo state and per-episode bookkeeping. KEEPS a_rls, P_rls
        and the GP posterior -- that learned knowledge is what carries to the next
        episode."""
        nonlocal tau_prev, tau_cmd, ctrl_cmd, solve_success, control_count
        nonlocal filtered_y_dot, last_control_state, res_v_dot, res_omega_dot, last_rls_info
        nonlocal gp_pred_v_dot_pre, gp_pred_omega_dot_pre, last_accel, last_gyro

        # seeded start jitter (only when main(seed=...) is given) -> each seed is a different
        # path through the chaotic climb, so a multi-seed sweep can see the learning trend.
        dx = rng.normal(0.0, 0.03) if rng is not None else 0.0
        dpitch = math.radians(rng.normal(0.0, 1.5)) if rng is not None else 0.0
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[root_x_qid] = INITIAL_X + dx
        data.qpos[root_z_qid] = INITIAL_Z
        data.qpos[root_pitch_qid] = math.radians(INITIAL_ROOT_PITCH_DEG) + dpitch
        data.ctrl[drive_id] = 0.0
        data.time = 0.0
        mujoco.mj_forward(model, data)

        tau_prev = 0.0
        tau_cmd = 0.0
        ctrl_cmd = 0.0
        solve_success = False
        control_count = 0
        filtered_y_dot = None
        last_control_state = None
        res_v_dot = 0.0
        res_omega_dot = 0.0
        gp_pred_v_dot_pre = 0.0
        gp_pred_omega_dot_pre = 0.0
        last_rls_info = empty_rls_info()
        last_accel = None
        last_gyro = None
        nmpc.last_solution = None      # fresh warm start each episode

    def run_episode(episode: int, viewer=None) -> list[dict]:
        reset_episode()
        print(f"\n==================== EPISODE {episode} "
              f"(gp_ready={gp.ready}, n_active={gp.n_active}) ====================")
        ep_history = []
        reached = False
        k = 0
        # The episode ends either at SIM_TIME (while condition) or as soon as the
        # car reaches the goal (break below).
        while data.time < SIM_TIME:
            if viewer is not None and not viewer.is_running():
                break
            start = time.time()

            if k % ctrl_steps == 0:
                control_update()

            mujoco.mj_step(model, data)
            ep_history.append(log_row(episode))

            if viewer is not None:
                viewer.sync()
                sleep_time = sim_dt - (time.time() - start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            k += 1

            if abs(float(data.qpos[root_x_qid]) - GOAL_X) < GOAL_TOL:
                reached = True
                break

        x_final = float(data.qpos[root_x_qid])
        print(f"[episode {episode}] x_final={x_final:.2f} m | goal={GOAL_X:.2f} m | "
              f"reached={reached} | t={data.time:.2f}s | "
              f"gp_ready={gp.ready} n_active={gp.n_active}")

        # (C) per-episode GP one-step RMSE over the steps where the GP was active.
        ready = [r for r in ep_history if r["gp_ready"] == 1]
        if ready:
            ew = np.array([r["r_omega_dot"] - r["gp_omega_dot_pred_pre"] for r in ready])
            ev = np.array([r["r_v_dot"] - r["gp_v_dot_pred_pre"] for r in ready])
            print(f"           GP one-step RMSE  omega={np.sqrt(np.mean(ew**2)):.4f}  "
                  f"v={np.sqrt(np.mean(ev**2)):.4f}  ({len(ready)} ready steps)")

        save_model(MODEL_PATH, gp, a_rls, P_rls)   # checkpoint after each episode
        return ep_history

    # Episodic pre-wheelie learner: discovers the fastest-safe rear angle online (the GP
    # supplies WHERE the obstacle is; this learns HOW MUCH to rear). It sets nmpc.theta_obs
    # each episode (a runtime solver param -- no rebuild) and updates from the crossing time.
    learner = PreWheelieLearner(angle0=0.0) if PREWHEELIE_LEARN else None

    def crossing_metrics(ep_hist):
        x = np.array([r["x"] for r in ep_hist]); t = np.array([r["time"] for r in ep_hist])
        pit = np.array([abs(r["pitch_deg"]) for r in ep_hist])
        reached = bool((np.abs(x - GOAL_X) < GOAL_TOL).any())
        cleared = x >= 3.0                                  # obstacle (x=2) cleared
        t_cross = float(t[cleared][0]) if cleared.any() else SIM_TIME
        return t_cross, (float(pit.max()) if len(pit) else 0.0), reached

    def do_episode(ep, viewer):
        if learner is not None:
            nmpc.theta_obs = math.radians(learner.angle)   # this episode's candidate angle
        ep_hist = run_episode(ep, viewer)
        if learner is not None:
            tc, mp, rc = crossing_metrics(ep_hist)
            learner.update(tc, mp, rc)
            print(f"[learn] ep{ep:2d}: angle={learner.history[-1]['angle']:5.1f}deg  "
                  f"t_cross={tc:5.2f}s  maxPitch={mp:3.0f}  ->  best {learner.best_angle:4.1f}deg / "
                  f"{learner.best_time:.2f}s")
        return ep_hist

    history = []
    if RENDER:
        with mj_viewer.launch_passive(model, data) as viewer:
            for ep in range(N_EPISODES):
                if not viewer.is_running():
                    break
                history.extend(do_episode(ep, viewer))
    else:
        for ep in range(N_EPISODES):
            history.extend(do_episode(ep, None))

    if learner is not None:
        plot_prewheelie_learning(learner)

    names = [
        "b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0",
        "a_g", "a_tau", "a_omega", "a_v", "a_0",
    ]
    print("\nFinal RLS coefficients (persisted across episodes)")
    print("v_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0")
    print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    for i, name in enumerate(names):
        print(f"{name:10s} learned: {a_rls[i]: .4f} | nominal: {a_nom[i]: .4f}")

    save_history_csv(history, CSV_PATH)
    if MODEL_PATH.exists():
        print(f"Model checkpoint: {MODEL_PATH}  "
              f"(set LOAD_MODEL=True to resume from it)")
    return history


if __name__ == "__main__":
    main()
