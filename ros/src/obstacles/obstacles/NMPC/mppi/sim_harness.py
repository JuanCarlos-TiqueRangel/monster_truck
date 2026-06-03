#!/usr/bin/env python3


import math
import sys
import time
from pathlib import Path

import numpy as np

# make the parent NMPC folder importable (rls / gp_residual / nmpc)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rls import RLS, RLSConfig, omega_regressor, v_regressor          # noqa: E402
from gp_residual import GPResidual, GPConfig                          # noqa: E402
from nmpc import WheelieParams                                        # noqa: E402

import mujoco                                                         # noqa: E402


# ---- Scenario (kept identical to wheelie_gp_climb.py) ----
XML_PATH = Path(__file__).resolve().parent.parent / "monster_truck_flip_2d.xml"
GOAL_X = 5.0
SIM_TIME = 12.0
CTRL_DT = 0.05

PITCH_SIGN = -1.0
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0
INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_ROOT_PITCH_DEG = 0.0


def _joint(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def obstacle_far_x(model, data):
    """Right-most x reached by any static obstacle geom (for a 'climbed' test).
    Ignores the ground and the high-up marker pole."""
    far = 0.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name == "ground":
            continue
        x, _, z = data.geom_xpos[g]
        if z > 1.0:           # the far green marker pole is up at z~10
            continue
        if x < 0.1 or x > 5.5:
            continue
        far = max(far, float(x) + float(model.geom_size[g, 0]))
    return far


def run_episode(controller, p, gp_cfg, rls_cfg, *, xml_path=XML_PATH,
                sim_time=SIM_TIME, ctrl_dt=CTRL_DT, goal_x=GOAL_X,
                render=False, verbose=False, print_every=10):
    """Run one closed-loop episode; return dict(history, metrics)."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    xq, xv = _joint(model, "root_x")
    zq, zv = _joint(model, "root_z")
    pq, pv = _joint(model, "root_pitch")
    drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
    cmin = float(model.actuator_ctrlrange[drive, 0])
    cmax = float(model.actuator_ctrlrange[drive, 1])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[xq] = INITIAL_X
    data.qpos[zq] = INITIAL_Z
    data.qpos[pq] = math.radians(INITIAL_ROOT_PITCH_DEG)
    mujoco.mj_forward(model, data)
    far_x = obstacle_far_x(model, data)

    sim_dt = float(model.opt.timestep)
    csteps = max(1, int(round(ctrl_dt / sim_dt)))
    cdt = csteps * sim_dt
    nsteps = int(round(sim_time / sim_dt))

    # online learners (fresh each episode)
    gp = GPResidual(n_features=4, max_points=gp_cfg.max_points,
                    lengthscales=np.asarray(gp_cfg.lengthscales),
                    sf2=gp_cfg.sf2, sn2=gp_cfg.sn2,
                    novelty_thresh=gp_cfg.novelty_thresh,
                    activation_thresh=gp_cfg.activation_thresh)
    from nmpc import nominal_rls_seeds
    a0, b0 = nominal_rls_seeds(p)
    rls_w = RLS(a0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)
    rls_v = RLS(b0, forgetting=rls_cfg.forgetting, p0_scale=rls_cfg.p0_scale)

    goal = np.array([goal_x, 0.0, 0.0, 0.0])
    tau_prev = tau_cmd = 0.0
    last_state = None
    hist = {k: [] for k in ("t", "x", "v", "pitch_deg", "tau", "solve_ms")}
    solve_times = []

    def read_state():
        x = float(data.qpos[xq]); v = float(data.qvel[xv])
        theta = PITCH_SIGN * float(data.qpos[pq])
        omega = PITCH_SIGN * float(data.qvel[pv])
        return np.array([x, v, theta, omega])

    cc = 0

    def do_control():
        nonlocal tau_prev, tau_cmd, last_state, cc
        s = read_state()
        x, v, theta, omega = s
        if last_state is not None:
            _, vp, thp, omp = last_state
            phv = np.array(v_regressor(thp, vp, tau_cmd, sin=np.sin))
            phw = np.array(omega_regressor(thp, omp, vp, tau_cmd, cos=np.cos))
            gp.observe(np.array([thp, omp, vp, tau_cmd]),
                       (v - vp) / cdt - rls_v.predict(phv),
                       (omega - omp) / cdt - rls_w.predict(phw))
            if gp.n_seen % gp_cfg.refit_every == 0:
                gp.refit()
            rls_v.update(phv, (v - vp) / cdt)
            rls_w.update(phw, (omega - omp) / cdt)

        t0 = time.perf_counter()
        tau, _ = controller.solve(s, goal, tau_prev,
                                  rls_w.theta, rls_v.theta, gp.mpc_params())
        solve_ms = (time.perf_counter() - t0) * 1e3
        solve_times.append(solve_ms)

        tau = float(np.clip(tau, p.tau_min, p.tau_max))
        tau_prev = tau_cmd = tau
        ctrl = float(np.clip(ACTUATOR_SIGN * TAU_TO_CTRL * tau, cmin, cmax))
        data.ctrl[drive] = ctrl
        last_state = s.copy()

        hist["t"].append(float(data.time)); hist["x"].append(x)
        hist["v"].append(v); hist["pitch_deg"].append(math.degrees(theta))
        hist["tau"].append(tau); hist["solve_ms"].append(solve_ms)
        if verbose and cc % print_every == 0:
            print(f"t={data.time:5.2f} x={x:6.3f} v={v:6.3f} "
                  f"pitch={math.degrees(theta):6.1f} tau={tau:7.3f} "
                  f"solve={solve_ms:5.1f}ms")
        cc += 1

    if render:
        import mujoco.viewer as mj_viewer
        with mj_viewer.launch_passive(model, data) as viewer:
            k = 0
            while viewer.is_running() and data.time < sim_time:
                start = time.time()
                if k % csteps == 0:
                    do_control()
                mujoco.mj_step(model, data)
                viewer.sync()
                slp = sim_dt - (time.time() - start)
                if slp > 0:
                    time.sleep(slp)
                k += 1
    else:
        for k in range(nsteps):
            if k % csteps == 0:
                do_control()
            mujoco.mj_step(model, data)

    x_arr = np.array(hist["x"]); pit = np.array(hist["pitch_deg"])
    tau_arr = np.array(hist["tau"])
    metrics = {
        "max_x": float(x_arr.max()),
        "final_x": float(x_arr[-1]),
        "final_v": float(hist["v"][-1]),
        "obstacle_far_x": float(far_x),
        "climbed": bool(x_arr.max() > far_x + 0.05),
        "reached_goal": bool(x_arr.max() >= goal_x - 0.2),
        "settle_err": float(abs(x_arr[-1] - goal_x)),
        "max_abs_pitch": float(np.abs(pit).max()),
        "flipped": bool(np.abs(pit).max() > 110.0),
        "mean_abs_tau": float(np.abs(tau_arr).mean()),
        "mean_abs_dtau": float(np.abs(np.diff(tau_arr)).mean()) if len(tau_arr) > 1 else 0.0,
        "solve_ms_mean": float(np.mean(solve_times)),
        "solve_ms_p95": float(np.percentile(solve_times, 95)),
    }
    return {"history": hist, "metrics": metrics}


if __name__ == "__main__":
    # quick standalone test of MPPI through the harness
    from mppi import WheelieMPPI, MPPIConfig
    p = WheelieParams(v_max=1.5, v_min=-1.5)
    gp_cfg = GPConfig()
    rls_cfg = RLSConfig(forgetting=0.9995)
    ctrl = WheelieMPPI(p, MPPIConfig(), gp_cfg)
    out = run_episode(ctrl, p, gp_cfg, rls_cfg, verbose=True)
    print("\nMPPI metrics:")
    for k, val in out["metrics"].items():
        print(f"  {k:16s}: {val}")
