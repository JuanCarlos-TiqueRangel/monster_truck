#!/usr/bin/env python3
"""
rls_training.py
---------------
Stand-alone TRAINING / system-identification run for the two-output full-dynamics RLS
(rls.py). It drives the MuJoCo truck through a scripted EXCITATION schedule -- forward,
backward, wheelie, flip, chirp, PRBS -- so EVERY RLS regressor is excited, then learns
the 10 dynamics weights online. It PRINTS the weights + one-step errors (live and at the
end) and PLOTS the weight convergence together with the prediction error.

This is the controller-free counterpart to obstacle_mujoco_simulation.py: no NMPC, no
GP -- just open-loop excitation -> RLS. Run it once to get good weights, then either
freeze them in the controller (RLS_FREEZE=True) or paste the printed array into
nominal_rls_parameters().

Sign conventions match obstacle_mujoco_simulation.py. Measured on this model:
  * NEGATIVE controller tau -> drives FORWARD and rears up (backward wheelie/flip)
  * POSITIVE controller tau -> drives BACKWARD and noses over (forward flip)
"""

import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as mj_viewer

from params_mujoco import WheelieParams
from rls import nominal_rls_parameters, rls_update


# ============================================================
# Settings
# ============================================================

XML_PATH = Path(__file__).with_name("monster_truck_flip_2d.xml")
IMG_PATH = Path(__file__).with_name("images") / "rls_training.png"
NPZ_PATH = Path(__file__).with_name("rls_trained.npz")

RENDER = True            # watch the truck run the maneuvers (set False for a fast/headless run)
REPEATS = 1             # repeat the whole schedule N times for better convergence
CTRL_DT = 0.05           # RLS/control period [s]  (MuJoCo integrates at model.opt.timestep)
PRINT_EVERY_N = 5        # print one status line every N control steps

# Sign conventions (must match the controller's MuJoCo interface).
PITCH_SIGN = -1.0
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0
INITIAL_Z = 0.1512

# Acceleration TARGET for the RLS. The RLS regresses phi -> [v_dot, omega_dot]; the target
# can be the noisy finite-difference of the velocities (legacy) or a MEASURED acceleration
# sampled at the SAME instant as phi (no differentiation, no half-step bias):
#   "qacc"        : MuJoCo's exact generalized acceleration -- exact + perfectly matched to
#                   the model's (v, omega) coordinates. Best for sim identification.
#   "imu"         : the on-board IMU -- accelerometer (imu_acc) -> v_dot, gyro (imu_gyro)
#                   -> omega_dot. Hardware-realistic. NOTE: a gyro measures omega, not its
#                   derivative, so omega_dot is still differentiated from the gyro (an IMU
#                   has no angular-accelerometer); only the v_dot channel is truly clean here.
#   "finite_diff" : legacy (v_next - v_prev)/dt + low-pass.
# Use "qacc" for SIM identification (exact ground truth); "imu" is the hardware path.
ACCEL_SOURCE = "qacc"

# ---- RLS settings ----
# Initialise the weights from the analytical model (nominal_rls_parameters) or from ZERO.
# The analytical nominal is in a DIFFERENT tau-sign convention than this MuJoCo plant, so
# for a clean from-scratch identification ZERO_INIT=True is the honest choice (no biased
# prior on the weakly-excited directions). Set False to start at the analytical nominal.
ZERO_INIT = True
# Held-wheelie operating points [deg]. MULTIPLE distinct angles give multiple cos(theta)
# levels, which breaks the gravity-term collinearities -- cos-vs-constant (a_g vs a_0) and
# tau-vs-tau*cos (b_tau vs b_tau_cos) -- raising lambda_min of the information matrix
# toward the persistence-of-excitation bound. A wide ladder of angles + the PI balancer
# below gives clean data across the whole reachable pitch range. (2 angles already suffice
# for PE-OK; more sharpens a_g/a_0. The truck cannot sit at negative theta on flat ground.)
HELD_ANGLES = (25.0, 45.0, 60.0, 72.0)
FORGETTING_FACTOR = 0.999     # <1 keeps the fit adaptive so late (flip) data still moves the weights
INITIAL_COVARIANCE = 5.0      # prior uncertainty on the weights
DERIVATIVE_ALPHA = 0.0        # 0 = no extra filtering of the finite-difference accelerations
CLIP_PARAMETERS = False       # False = report the RAW identified weights (no projection box)
SIGMA_V_DOT = 2.0
SIGMA_OMEGA_DOT = 5.0
# Reject contact-impulse / crash outliers (normalised innovation, chi^2(2)); raise toward
# inf to keep ALL data, lower to be stricter about what corrupts the smooth-dynamics fit.
NIS_GATE = 40.0


# ============================================================
# Excitation signal builders (controller-tau convention)
# Each returns fn(t_local, state) -> tau; open-loop signals ignore `state`, the
# held-wheelie feedback law uses it.
# ============================================================

def const(c):
    return lambda tl, s: c


def sine(bias, amp, f):
    return lambda tl, s: bias + amp * math.sin(2.0 * math.pi * f * tl)


def chirp(amp, f0, f1, T):
    """Sinusoid whose frequency sweeps f0 -> f1 over the segment (persistent excitation)."""
    return lambda tl, s: amp * math.sin(2.0 * math.pi * (f0 + 0.5 * (f1 - f0) * tl / T) * tl)


def prbs(amp, dwell, rng):
    """Pseudo-random piecewise-constant torque: a fresh random level every `dwell` s."""
    st = {"tnext": -1.0, "val": 0.0}

    def f(tl, s):
        if tl >= st["tnext"]:
            st["val"] = amp * float(rng.uniform(-1.0, 1.0))
            st["tnext"] = tl + dwell
        return st["val"]

    return f


def held_wheelie(target_deg, kp=16.0, ki=12.0, kd=2.8, tau_ff=-2.0, ff_grav=-3.5,
                 i_clamp=4.0, dither=((1.2, 1.0), (0.8, 2.3))):
    """Closed-loop PI+gravity-feedforward balancer that holds a wheelie AT target_deg
    while dithering the torque -- producing CLEAN data at SUSTAINED non-zero pitch, which
    is what identifies the gravity term a_g and separates b_tau from b_tau_cos.

    Why more than a plain PD: a wheelie is an unstable equilibrium with steady-state error,
    so a PD only ever settles at its ONE natural balance point (~70 deg here) regardless of
    target. To place the truck at a SPREAD of angles (needed to span cos/sin(theta) for
    PE), we add:
      * ff_grav*cos(theta) -- gravity-compensation feedforward (less hold torque near
        tip-over, where cos->0), so the operating point is set by the target, not gravity;
      * ki*integral(theta-target) -- removes the steady-state error so theta -> target
        (anti-windup clamped to +/- i_clamp so it can't run away on the unstable plant).
    Sign convention: negative tau rears up, so theta<target (e<0) drives tau negative.

    `dither` is a list of (amp, freq) sinusoids; several distinct frequencies raise the
    persistence-of-excitation order (a sum of k sinusoids is PE for ~2k parameters).
    """
    target = math.radians(target_deg)
    st = {"i": 0.0, "t_prev": None}

    def f(tl, s):
        theta, omega = s[2], s[3]
        dt = CTRL_DT if st["t_prev"] is None else max(tl - st["t_prev"], 1e-3)
        st["t_prev"] = tl
        e = theta - target
        st["i"] = float(np.clip(st["i"] + e * dt, -i_clamp, i_clamp))   # anti-windup
        d = sum(a * math.sin(2.0 * math.pi * fr * tl) for (a, fr) in dither)
        return tau_ff + ff_grav * math.cos(theta) + kp * e + ki * st["i"] + kd * omega + d

    return f


def track_speed(setpoints, kp=2.5, speed_cap=4.0, k_theta=12.0, k_omega=3.0,
                dwell=1.2, tau_cap=6.0):
    """Closed-loop velocity tracker: a PD chases a SWEEP of speed setpoints (a new one
    every `dwell` s) so |v| spans a wide range in BOTH directions. This decorrelates v
    from |v|*v -> lifts the PE constant of the v_dot regressor (its weakest direction).

    Sign convention: negative tau drives forward, so below the target (v-v*<0) the speed
    term drives tau negative to accelerate forward. Hard acceleration rears this truck, so
    a PD pitch STABILISER (k_theta*theta + k_omega*omega, with the omega term catching the
    rotation EARLY) keeps it flat; the speed command is saturated (speed_cap) BELOW the
    torque limit so the stabiliser always has authority to override an incipient wheelie.
    """
    sp = list(setpoints)
    st = {"tnext": -1.0, "idx": -1, "vt": 0.0}

    def f(tl, s):
        if tl >= st["tnext"]:
            st["idx"] = (st["idx"] + 1) % len(sp)
            st["vt"] = sp[st["idx"]]
            st["tnext"] = tl + dwell
        v, theta, omega = s[1], s[2], s[3]
        tau_speed = float(np.clip(kp * (v - st["vt"]), -speed_cap, speed_cap))
        tau_flat = k_theta * theta + k_omega * omega        # PD stabiliser -> theta,omega ~ 0
        return float(np.clip(tau_speed + tau_flat, -tau_cap, tau_cap))

    return f


def build_schedule(p: WheelieParams):
    """The maneuver script. Each entry: (name, duration_s, tau_fn(t_local), reset_after).
    reset_after=True returns the truck upright after a flip so the next phase is clean."""
    rng = np.random.default_rng(0)
    # Clean persistent-excitation phases FIRST (they do the actual identifying), then the
    # flips as a finale (mostly gated as contact outliers, each reset to upright after).
    sched = [
        # closed-loop speed SWEEP: flat driving across a wide |v| range both directions
        # (the PE-critical phase for the v_dot channel: decorrelates v from |v|*v).
        ("speed sweep (PD)",       7.2, track_speed([+3.0, -3.0, +1.5, -1.5, +2.5, -2.5]), False),
        ("forward cruise+dither",  2.5, sine(-3.0, 1.8, 0.8),        False),
        ("coast",                  0.8, const(0.0),                  False),
        ("reverse dither",         2.0, sine(+4.0, 2.0, 1.0),        False),
        ("fwd<->back chirp",       3.5, chirp(6.0, 0.3, 2.5, 3.5),   False),
    ]
    # closed-loop held-wheelie SWEEP: one phase per operating angle, each giving a fresh
    # cos(theta) level with multi-frequency torque dither (the PE-critical phases).
    for ang in HELD_ANGLES:
        sched.append((f"held wheelie ~{ang:.0f}deg", 4.5, held_wheelie(ang), True))
    sched += [
        ("PRBS excitation",        3.5, prbs(5.0, 0.30, rng),        True),
        ("backward WHEELIE/flip",  1.6, const(p.tau_min),            True),
        ("settle",                 0.6, const(0.0),                  False),
        ("forward NOSE flip",      1.3, const(p.tau_max),            True),
        ("settle",                 0.6, const(0.0),                  False),
    ]
    return sched


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
        "skipped", "seg"]
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
          f"forgetting={FORGETTING_FACTOR}  nis_gate={NIS_GATE}")

    # mutable run state
    state = {
        "filtered": None,
        "last_state": None,
        "tau_applied": 0.0,
        "info": None,
        "n_gated": 0,
        "control_count": 0,
        "last_gyro": None,     # previous gyro pitch-rate (for imu omega_dot differentiation)
    }
    history = []
    seg_bounds = []   # (t_start, name) for the plot annotations

    def read_state():
        x = float(data.qpos[xq])
        v = float(data.qvel[xv])
        theta = PITCH_SIGN * float(data.qpos[pq])
        omega = PITCH_SIGN * float(data.qvel[pv])
        return np.array([x, v, theta, omega], dtype=float)

    def read_accel():
        """Measured [v_dot, omega_dot] for the CURRENT (state, ctrl). Call AFTER a
        mj_forward so qacc/sensordata are consistent with data.qpos/qvel/ctrl."""
        if ACCEL_SOURCE == "qacc":
            return float(data.qacc[xv]), PITCH_SIGN * float(data.qacc[pv])
        # "imu": accelerometer -> world-x linear accel; gyro -> pitch rate (differentiated).
        a_meas = np.asarray(data.sensordata[acc_adr:acc_adr + 3], dtype=float)
        R = np.asarray(data.site_xmat[imu_site], dtype=float).reshape(3, 3)
        v_dot = float((R @ a_meas + g_world)[0])     # remove gravity, rotate to world, take x
        omega = PITCH_SIGN * float(data.sensordata[gyro_adr + 1])   # gyro y-axis = pitch rate
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
        state["filtered"] = None
        state["tau_applied"] = 0.0
        state["info"] = None            # don't log a stale error on the first post-reset row
        state["last_gyro"] = None       # restart the gyro differentiation after the teleport

    def _rls_step(state_prev, tau, state_next, y_dot_meas):
        nonlocal a_rls, P_rls
        a_rls, P_rls, state["filtered"], info = rls_update(
            state_prev=state_prev,
            tau=tau,
            state_next=state_next,
            dt=ctrl_dt_actual,
            a=a_rls,
            P=P_rls,
            filtered_y_dot=state["filtered"],
            forgetting_factor=FORGETTING_FACTOR,
            derivative_alpha=DERIVATIVE_ALPHA,
            sigma_v_dot=SIGMA_V_DOT,
            sigma_omega_dot=SIGMA_OMEGA_DOT,
            nis_gate=NIS_GATE,
            clip_parameters=CLIP_PARAMETERS,
            y_dot_meas=y_dot_meas,
            p=p,
        )
        state["info"] = info
        if info["skipped"]:
            state["n_gated"] += 1

    def control_update(tau_fn, t_local, seg_idx):
        s_now = read_state()

        # torque for this step (open-loop signals ignore s_now; the held-wheelie law uses it)
        tau = float(np.clip(tau_fn(t_local, s_now), p.tau_min, p.tau_max))
        ctrl = float(np.clip(ACTUATOR_SIGN * TAU_TO_CTRL * tau, ctrl_min, ctrl_max))
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
        row = np.array([
            data.time, s_now[0], s_now[1], s_now[2], s_now[3], tau,
            info["v_dot_measured"] if info else 0.0,
            info["v_dot_hat"] if info else 0.0,
            info["v_dot_error"] if info else 0.0,
            info["omega_dot_measured"] if info else 0.0,
            info["omega_dot_hat"] if info else 0.0,
            info["omega_dot_error"] if info else 0.0,
            float(info["skipped"]) if info else 1.0,
            float(seg_idx),
        ], dtype=float)
        history.append(np.concatenate([row, a_rls]))   # 14 logs + 10 current weights

        if state["control_count"] % PRINT_EVERY_N == 0:
            ve = info["v_dot_error"] if info else 0.0
            we = info["omega_dot_error"] if info else 0.0
            print(f"t={data.time:6.2f} | x={s_now[0]:6.2f} | v={s_now[1]:6.2f} | "
                  f"pitch={math.degrees(s_now[2]):8.1f} deg | omega={s_now[3]:7.2f} | "
                  f"tau={tau:6.2f} | err[v={ve:7.3f} w={we:7.3f}]"
                  f"{'  [gated]' if (info and info['skipped']) else ''}")

        state["tau_applied"] = tau
        state["last_state"] = s_now
        state["control_count"] += 1

    def run_segments(viewer=None):
        reset_pose()
        schedule = build_schedule(p)
        for rep in range(REPEATS):
            for seg_idx, (name, dur, fn, reset_after) in enumerate(schedule):
                gidx = rep * len(schedule) + seg_idx
                t0 = data.time
                seg_bounds.append((t0, name))
                tag = f" (repeat {rep+1}/{REPEATS})" if REPEATS > 1 else ""
                print(f"\n--- phase {gidx}: {name}{tag}  [{dur:.1f}s] ---")
                nsub = int(round(dur / sim_dt))
                for s in range(nsub):
                    if viewer is not None and not viewer.is_running():
                        return
                    if s % ctrl_steps == 0:
                        control_update(fn, data.time - t0, gidx)
                    start = time.time()
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                        sleep = sim_dt - (time.time() - start)
                        if sleep > 0:
                            time.sleep(sleep)
                if reset_after:
                    print(f"    [reset to upright after '{name}']")
                    reset_pose()

    if RENDER:
        with mj_viewer.launch_passive(model, data) as viewer:
            run_segments(viewer)
    else:
        run_segments(None)

    H = np.array(history)
    print_results(H, a_rls, a_nom, state["n_gated"])
    pe_report(H)
    np.savez(NPZ_PATH, a_rls=a_rls, P_rls=P_rls, a_nom=a_nom)
    print(f"Saved learned weights: {NPZ_PATH.name}")
    print("a_rls =", np.array2string(a_rls, precision=6, separator=", "))
    plot_results(H, a_nom, seg_bounds)


# ============================================================
# Reporting
# ============================================================

def print_results(H, a_rls, a_nom, n_gated):
    used = H[H[:, 12] < 0.5]          # rows where the RLS actually updated (not gated/first)
    rmse_v = float(np.sqrt(np.mean(used[:, 8] ** 2))) if len(used) else float("nan")
    rmse_w = float(np.sqrt(np.mean(used[:, 11] ** 2))) if len(used) else float("nan")

    print("\n========== RLS identification result ==========")
    print(f"control steps: {len(H)}  |  used: {len(used)}  |  gated outliers: {n_gated}")
    print(f"one-step RMSE   v_dot={rmse_v:.4f} m/s^2   omega_dot={rmse_w:.4f} rad/s^2")
    print("\nv_dot     = b_tau*tau + b_v*v + b_abs_v*|v|v + b_tau_cos*tau*(cos(theta)-1) + b_0")
    print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    for i, name in enumerate(NAMES):
        print(f"  {name:10s} learned: {a_rls[i]: .6f} | nominal: {a_nom[i]: .6f}")
    print("===============================================")


def pe_report(H):
    """Persistence-of-excitation diagnostic. Rebuilds the two regressors over the USED
    (non-gated) steps, RMS-scales each column (so the units don't dominate and the
    constant column maps to 1), and reports the eigenstructure of the normalized
    information matrix G = (1/N) Phi_s^T Phi_s:

        lambda_min  -- the PE constant alpha (G >= alpha*I). >0 and bounded away from
                       zero  <=>  every parameter direction is excited (PE satisfied).
        cond        -- lambda_max/lambda_min; large => a near-collinear (weakly excited)
                       direction, identified below by its eigenvector.

    Rule of thumb on this normalized scale: lambda_min >~ 0.05 and cond <~ 100 means the
    direction is well excited; lambda_min ~ 0 flags an UNIDENTIFIABLE combination.
    """
    used = H[H[:, 12] < 0.5]
    tau, v, th, om = used[:, 5], used[:, 2], used[:, 3], used[:, 4]
    ct = np.cos(th)
    one = np.ones_like(tau)
    Phi_v = np.column_stack([tau, v, np.abs(v) * v, tau * (ct - 1.0), one])
    Phi_w = np.column_stack([ct, tau, om, v, one])

    def report(Phi, names, label):
        N = len(Phi)
        rms = np.sqrt(np.mean(Phi ** 2, axis=0))
        dead = [names[i] for i in range(len(names)) if rms[i] < 1e-9]
        Phi_s = Phi / np.where(rms < 1e-9, 1.0, rms)
        G = (Phi_s.T @ Phi_s) / N
        w, V = np.linalg.eigh(G)
        lam_min, lam_max = float(w[0]), float(w[-1])
        cond = lam_max / lam_min if lam_min > 1e-12 else float("inf")
        verdict = "OK (PE)" if (lam_min > 0.05 and cond < 100) else "WEAK"
        print(f"\n[PE] {label}  (N={N}):  lambda_min={lam_min:.3e}  cond={cond:8.1f}  -> {verdict}")
        vmin = V[:, 0]
        order = np.argsort(-np.abs(vmin))[:3]
        combo = " ".join(f"{vmin[i]:+.2f}*{names[i]}" for i in order)
        print(f"     least-excited direction ~ ({combo})")
        if dead:
            print(f"     NEVER excited (rms~0, unidentifiable): {dead}")

    print("\n========== Persistence-of-excitation check ==========")
    print("(regressors RMS-normalized; lambda_min = PE constant; higher = richer excitation)")
    report(Phi_v, ["b_tau", "b_v", "b_abs_v", "b_tau_cos", "b_0"], "v_dot regressor")
    report(Phi_w, ["a_g", "a_tau", "a_omega", "a_v", "a_0"], "omega_dot regressor")
    print("=====================================================")


def plot_results(H, a_nom, seg_bounds):
    t = H[:, 0]
    coeffs = H[:, 14:14 + NC]

    fig, axs = plt.subplots(6, 1, sharex=True, figsize=(13, 15))

    # 0) states: forward/backward (v) + flips (theta)
    ax = axs[0]
    ax.plot(t, H[:, 2], color="C0", label="v [m/s]")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_ylabel("v [m/s]", color="C0")
    axr = ax.twinx()
    axr.plot(t, np.degrees(H[:, 3]), color="C3", lw=1.0, label="pitch [deg]")
    axr.set_ylabel("pitch [deg]", color="C3")
    ax.set_title("RLS training: maneuvers (v forward/backward, pitch wheelie/flip)")

    # 1) excitation torque
    axs[1].plot(t, H[:, 5], color="C2")
    axs[1].set_ylabel("tau [Nm]")

    # 2) v_dot channel: measured vs RLS vs error
    axs[2].plot(t, H[:, 6], color="0.6", lw=1.0, label="measured")
    axs[2].plot(t, H[:, 7], color="C0", ls="--", label="RLS fit")
    axs[2].plot(t, H[:, 8], color="C3", ls=":", label="error")
    axs[2].set_ylabel("v_dot [m/s$^2$]")
    axs[2].legend(fontsize=8, ncol=3)

    # 3) omega_dot channel: measured vs RLS vs error
    axs[3].plot(t, H[:, 9], color="0.6", lw=1.0, label="measured")
    axs[3].plot(t, H[:, 10], color="C0", ls="--", label="RLS fit")
    axs[3].plot(t, H[:, 11], color="C3", ls=":", label="error")
    axs[3].set_ylabel("omega_dot [rad/s$^2$]")
    axs[3].legend(fontsize=8, ncol=3)

    # 4) v_dot weights
    for i in range(5):
        line, = axs[4].plot(t, coeffs[:, i], label=NAMES[i])
        axs[4].axhline(a_nom[i], color=line.get_color(), ls=":", lw=0.8)
    axs[4].set_ylabel("v_dot weights")
    axs[4].legend(fontsize=8, ncol=5)

    # 5) omega_dot weights
    for i in range(5, NC):
        line, = axs[5].plot(t, coeffs[:, i], label=NAMES[i])
        axs[5].axhline(a_nom[i], color=line.get_color(), ls=":", lw=0.8)
    axs[5].set_ylabel("omega_dot weights")
    axs[5].set_xlabel("time [s]")
    axs[5].legend(fontsize=8, ncol=6)

    # phase boundaries + labels on every panel
    for (tb, name) in seg_bounds:
        for a in axs:
            a.axvline(tb, color="0.8", ls="--", lw=0.7)
        axs[0].text(tb, axs[0].get_ylim()[1], name, rotation=90, va="top", ha="right",
                    fontsize=6, color="0.4")
    for a in axs:
        a.grid(True, alpha=0.3)

    fig.suptitle("Two-output full-dynamics RLS -- training (weights + one-step error)")
    fig.tight_layout()
    try:
        IMG_PATH.parent.mkdir(exist_ok=True)
        fig.savefig(IMG_PATH, dpi=150, bbox_inches="tight")
        print(f"Saved figure: {IMG_PATH}")
    except PermissionError:
        alt = Path("/tmp/rls_training.png")
        fig.savefig(alt, dpi=150, bbox_inches="tight")
        print(f"[warn] {IMG_PATH.parent} not writable; saved to {alt}")
    plt.show()


if __name__ == "__main__":
    main()
