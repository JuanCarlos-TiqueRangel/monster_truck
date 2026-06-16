#!/usr/bin/env python3
"""
watch_learning.py
-----------------
WATCH the controller learn to go A->B faster, lap after lap, in ONE viewer window.

Unlike run_episode (which opens a fresh viewer per lap and closes it -- so on a
flaky display you only ever see the first lap), this owns the MuJoCo sim and
keeps a SINGLE viewer open, just resetting the truck between episodes. So you see
every episode run back-to-back in the same window.

A greedy (1+1) search keeps the best cost weights so far and probes a small change
each lap; faster & safe -> ACCEPT, else revert. Early laps are slow with the big
near-flip wheelie; accepted laps get faster and cleaner (lower pitch). The GP is
persisted across laps so the plant is stable.

    python3 watch_learning.py [num_episodes]     # default 15
    RENDER=False at top -> headless (fast, just numbers)
"""
import sys
import math
import time
import json
from pathlib import Path
import numpy as np
import mujoco
import torch

from sim_harness import (XML_PATH, GOAL_X, PITCH_SIGN, ACTUATOR_SIGN, TAU_TO_CTRL,
                         INITIAL_X, INITIAL_Z, INITIAL_ROOT_PITCH_DEG, CTRL_DT,
                         SSGPConfig, _joint)
from rls import RLS, RLSConfig, omega_regressor, v_regressor
from params import WheelieParams, nominal_rls_seeds
from mppi import WheelieMPPI, MPPIConfig

NUM_EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
RENDER = False
SIM_TIME = 40.0

HERE = Path(__file__).resolve().parent
LEARNED = HERE / "learned_cost.json"

PARAMS = WheelieParams(v_max=4.0, v_min=-4.0)
GP_CFG = SSGPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

# ── build the sim ONCE (so the viewer can persist across episodes) ───────────
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
xq, xv = _joint(model, "root_x")
zq, zv = _joint(model, "root_z")
pq, pv = _joint(model, "root_pitch")
drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
cmin = float(model.actuator_ctrlrange[drive, 0])
cmax = float(model.actuator_ctrlrange[drive, 1])
sim_dt = float(model.opt.timestep)
csteps = max(1, int(round(CTRL_DT / sim_dt)))
cdt = csteps * sim_dt
goal = np.array([GOAL_X, 0.0, 0.0, 0.0])


def make_cfg(th):
    v_cruise, v_ref_gain, q_theta = [float(t) for t in th]
    return MPPIConfig(
        dt=0.05, N=20, num_samples=2048, temperature=10.0, noise_sigma=4.0,
        q_x=15.0, q_v=8.0, q_theta=q_theta, q_omega=60.0,
        r_tau=0.05, r_dtau=1.0, q_terminal_theta=q_theta, q_terminal_omega=60.0,
        flip_threshold_deg=85.0, flip_penalty=5.0e4, v_barrier=50.0,
        v_ref_gain=v_ref_gain, v_cruise=v_cruise, seed=2)


def run_lap(th, viewer, gp_state):
    """Run one lap in the (persistent) sim with cost weights th. Returns
    (t2g, reached, flipped, max_pitch, new_gp_state)."""
    # reset truck to the start line
    data.qpos[:] = 0.0; data.qvel[:] = 0.0
    data.qpos[xq] = INITIAL_X; data.qpos[zq] = INITIAL_Z
    data.qpos[pq] = math.radians(INITIAL_ROOT_PITCH_DEG)
    data.time = 0.0
    mujoco.mj_forward(model, data)

    ctrl = WheelieMPPI(PARAMS, make_cfg(th), GP_CFG)
    gp = GP_CFG.build(n_features=4)
    if gp_state is not None:                       # carry the learned GP across laps
        gp.__dict__.update(gp_state)
    a0, b0 = nominal_rls_seeds(PARAMS)
    rls_w = RLS(a0, forgetting=RLS_CFG.forgetting, p0_scale=RLS_CFG.p0_scale)
    rls_v = RLS(b0, forgetting=RLS_CFG.forgetting, p0_scale=RLS_CFG.p0_scale)

    st = {"tau_prev": 0.0, "tau_cmd": 0.0, "last": None}
    pitches = []

    def read_state():
        return np.array([float(data.qpos[xq]), float(data.qvel[xv]),
                         PITCH_SIGN * float(data.qpos[pq]),
                         PITCH_SIGN * float(data.qvel[pv])])

    def do_control():
        s = read_state(); x, v, theta, omega = s
        if st["last"] is not None:
            _, vp, thp, omp = st["last"]
            phv = np.array(v_regressor(thp, vp, st["tau_cmd"], sin=np.sin))
            phw = np.array(omega_regressor(thp, omp, vp, st["tau_cmd"], cos=np.cos))
            r_v = (v - vp) / cdt - rls_v.predict(phv)
            r_w = (omega - omp) / cdt - rls_w.predict(phw)
            z = np.array([thp, omp, vp, st["tau_cmd"]])
            gp.observe(z, r_v, r_w)
            if gp.n_seen % getattr(GP_CFG, "refit_every", 1) == 0:
                gp.refit()
            rls_v.update(phv, (v - vp) / cdt)
            rls_w.update(phw, (omega - omp) / cdt)
        if getattr(gp, "ready", False) and hasattr(ctrl, "inv_l2"):
            ctrl.inv_l2 = torch.as_tensor(1.0 / np.asarray(gp.l) ** 2,
                                          dtype=ctrl.dtype, device=ctrl.device)
            ctrl.sf2 = float(gp.sf2)
        tau, _ = ctrl.solve(s, goal, st["tau_prev"], rls_w.theta, rls_v.theta, gp.mpc_params())
        tau = float(np.clip(tau, PARAMS.tau_min, PARAMS.tau_max))
        st["tau_prev"] = st["tau_cmd"] = tau
        data.ctrl[drive] = float(np.clip(ACTUATOR_SIGN * TAU_TO_CTRL * tau, cmin, cmax))
        st["last"] = s.copy()
        pitches.append(abs(math.degrees(theta)))

    t2g = SIM_TIME; reached = False
    k = 0
    while data.time < SIM_TIME:
        if viewer is not None and not viewer.is_running():
            break
        start = time.time()
        if k % csteps == 0:
            do_control()
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            slp = sim_dt - (time.time() - start)
            if slp > 0:
                time.sleep(slp)
        k += 1
        if float(data.qpos[xq]) >= GOAL_X:
            t2g = float(data.time); reached = True
            break

    max_pitch = max(pitches) if pitches else 0.0
    flipped = max_pitch > 110.0
    # snapshot the GP to carry forward
    new_state = {kk: gp.__dict__[kk] for kk in gp.__dict__}
    return t2g, reached, flipped, max_pitch, new_state


# ── greedy (1+1) learning loop, all inside ONE viewer ────────────────────────
theta = np.array([1.2, 0.6, 6.0])              # baseline -> ep1 is the 'before'
sigma = np.array([0.25, 0.15, 6.0])
LO = np.array([1.2, 0.4, 3.0]); HI = np.array([4.0, 2.5, 60.0])
rng = np.random.default_rng(0)
best_theta, best_t2g = theta.copy(), float("inf")
gp_state = None

print("\n" + "=" * 78)
print("WATCH IT LEARN  (single viewer, one lap per episode)")
print("=" * 78)
print(f"{'ep':>3} | {'v_cruise':>8} {'v_ref_g':>8} {'q_theta':>8} | {'t2g':>6} "
      f"{'pitch':>6} {'result':>9} | {'best':>6}")
print("-" * 78)


def learn(viewer):
    global theta, sigma, best_theta, best_t2g, gp_state
    for ep in range(1, NUM_EPISODES + 1):
        cand = best_theta.copy() if ep == 1 else np.clip(
            best_theta + sigma * rng.standard_normal(3), LO, HI)
        t2g, reached, flipped, pitch, gp_state = run_lap(cand, viewer, gp_state)
        safe = reached and not flipped
        if (safe and t2g < best_t2g - 1e-3) or (ep == 1 and safe):
            best_theta, best_t2g = cand.copy(), t2g
            tag = "ACCEPT" if ep > 1 else "baseline"
        else:
            tag = "keep" if safe else ("FLIP" if flipped else "no-reach")
        sigma = np.maximum(sigma * 0.82, [0.03, 0.02, 0.6])
        print(f"{ep:>3} | {cand[0]:>8.2f} {cand[1]:>8.2f} {cand[2]:>8.1f} | {t2g:>6.2f} "
              f"{pitch:>6.0f} {tag:>9} | {best_t2g:>6.2f}")
        if viewer is not None and not viewer.is_running():
            print("(viewer closed -- stopping)"); break


if RENDER:
    import mujoco.viewer as mj_viewer
    print("(one viewer window will open and stay open for all episodes)\n")
    with mj_viewer.launch_passive(model, data) as viewer:
        learn(viewer)
else:
    learn(None)

print("-" * 78)
print(f"learned (fastest safe) weights:  v_cruise={best_theta[0]:.2f}  "
      f"v_ref_gain={best_theta[1]:.2f}  q_theta={best_theta[2]:.1f}")
print(f"best time A->B: {best_t2g:.2f} s")
print("=" * 78 + "\n")
json.dump({"v_cruise": float(best_theta[0]), "v_ref_gain": float(best_theta[1]),
           "q_theta": float(best_theta[2])}, open(LEARNED, "w"), indent=2)
print(f"✓ saved {LEARNED.name}\n")
