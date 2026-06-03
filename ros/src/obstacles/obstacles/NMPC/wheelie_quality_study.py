#!/usr/bin/env python3
"""
wheelie_quality_study.py
------------------------
Isolated wheelie-maneuver benchmark: gradient NMPC vs sampling MPPI.

The wheelie is DECOUPLED from the obstacle traverse: flat ground, no obstacle,
the controller is simply asked to pop and hold a target body pitch (theta_ref).
We measure how well each controller executes that maneuver over many trials
(varied initial conditions; MPPI also varied sampling seed), giving the
"precision vs robustness" numbers for a paper.

Metrics per trial (steady window t >= 2 s):
    rms_err   RMS pitch tracking error to theta_ref   [deg]   (precision)
    peak      peak pitch                              [deg]   (overshoot/aggr.)
    jerk      mean |d tau| over the trial             [N.m]   (smoothness)
    solve     mean / p95 solve time                   [ms]    (compute)
    success   reached >= half of theta_ref and did not flip

Outputs (saved in THIS folder):
    wheelie_quality_trajectories.png
    wheelie_quality_metrics.png
"""

import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "mppi"))

import mujoco                                                   # noqa: E402
from nmpc import WheelieParams, MPCConfig, WheelieNMPC, nominal_rls_seeds  # noqa: E402
from rls import RLS, RLSConfig, omega_regressor, v_regressor    # noqa: E402
from gp_residual import GPResidual, GPConfig                    # noqa: E402
from mppi import WheelieMPPI, MPPIConfig                        # noqa: E402

# ── Settings ─────────────────────────────────────────────────────────────────
THETA_REF_DEG = 45.0      # wheelie target pitch
SIM_TIME = 5.0
CTRL_DT = 0.05
N_TRIALS = 15
STEADY_T = 2.0            # start of the steady-state window for tracking error

# velocity left ~unbounded so the v-limit does not confound the pitch task
PARAMS = WheelieParams(v_max=10.0, v_min=-10.0)
GP_CFG = GPConfig()
RLS_CFG = RLSConfig(forgetting=0.9995)

# pitch-tracking cost, identical for both controllers (fair comparison)
W = dict(N=20, q_x=0.0, q_v=0.5, q_theta=200.0, q_omega=10.0,
         r_tau=0.05, r_dtau=1.0, q_terminal_theta=200.0, q_terminal_omega=10.0)

PITCH_SIGN = -1.0
ACTUATOR_SIGN = -1.0


def make_flat_xml():
    """Strip every obstacle geom -> flat ground only."""
    base = (HERE / "monster_truck_flip_2d.xml").read_text()
    flat = re.sub(r'<geom name="(obs_box_\d+|pole_box_\d+|ramp_\d+)".*?/>',
                  '', base, flags=re.S)
    out = Path("/tmp/flat_wheelie.xml"); out.write_text(flat)
    return out


def make_nmpc(seed):
    return WheelieNMPC(PARAMS, MPCConfig(dt=CTRL_DT, ipopt_max_iter=80, **W), GP_CFG)


def make_mppi(seed):
    return WheelieMPPI(PARAMS, MPPIConfig(dt=CTRL_DT, num_samples=2048,
                                          temperature=10.0, noise_sigma=4.0,
                                          seed=seed, **W), GP_CFG)


def run_trial(make_ctrl, seed, init_pitch_deg, init_v, xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    def j(n):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        return int(model.jnt_qposadr[i]), int(model.jnt_dofadr[i])
    xq, xv = j("root_x"); zq, _ = j("root_z"); pq, pv = j("root_pitch")
    drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
    cmin = float(model.actuator_ctrlrange[drive, 0])
    cmax = float(model.actuator_ctrlrange[drive, 1])

    data.qpos[zq] = 0.1512
    data.qpos[pq] = math.radians(PITCH_SIGN * init_pitch_deg)
    data.qvel[xv] = init_v
    mujoco.mj_forward(model, data)

    sim_dt = float(model.opt.timestep)
    cs = max(1, int(round(CTRL_DT / sim_dt))); cdt = cs * sim_dt
    n = int(round(SIM_TIME / sim_dt))

    gp = GPResidual(n_features=4, max_points=GP_CFG.max_points,
                    lengthscales=np.asarray(GP_CFG.lengthscales), sf2=GP_CFG.sf2,
                    sn2=GP_CFG.sn2, novelty_thresh=GP_CFG.novelty_thresh,
                    activation_thresh=GP_CFG.activation_thresh)
    a0, b0 = nominal_rls_seeds(PARAMS)
    rw = RLS(a0, forgetting=RLS_CFG.forgetting, p0_scale=RLS_CFG.p0_scale)
    rv = RLS(b0, forgetting=RLS_CFG.forgetting, p0_scale=RLS_CFG.p0_scale)
    ctrl = make_ctrl(seed)

    ref = np.array([0.0, 0.0, math.radians(THETA_REF_DEG), 0.0])
    tau = tau_cmd = 0.0
    last = None
    log = {k: [] for k in ("t", "pitch", "tau", "solve_ms")}

    for k in range(n):
        if k % cs == 0:
            x = float(data.qpos[xq]); v = float(data.qvel[xv])
            theta = PITCH_SIGN * float(data.qpos[pq])
            omega = PITCH_SIGN * float(data.qvel[pv])
            s = np.array([x, v, theta, omega])
            if last is not None:
                _, vp, thp, omp = last
                phv = np.array(v_regressor(thp, vp, tau_cmd, sin=np.sin))
                phw = np.array(omega_regressor(thp, omp, vp, tau_cmd, cos=np.cos))
                gp.observe(np.array([thp, omp, vp, tau_cmd]),
                           (v - vp) / cdt - rv.predict(phv),
                           (omega - omp) / cdt - rw.predict(phw))
                gp.refit(); rv.update(phv, (v - vp) / cdt); rw.update(phw, (omega - omp) / cdt)
            t0 = time.perf_counter()
            tau, _ = ctrl.solve(s, ref, tau, rw.theta, rv.theta, gp.mpc_params())
            solve_ms = (time.perf_counter() - t0) * 1e3
            tau = float(np.clip(tau, PARAMS.tau_min, PARAMS.tau_max)); tau_cmd = tau
            data.ctrl[drive] = float(np.clip(ACTUATOR_SIGN * tau, cmin, cmax))
            last = s.copy()
            log["t"].append(float(data.time)); log["pitch"].append(math.degrees(theta))
            log["tau"].append(tau); log["solve_ms"].append(solve_ms)
        mujoco.mj_step(model, data)
    return {k: np.array(val) for k, val in log.items()}


def trial_metrics(tr):
    t = tr["t"]; pit = tr["pitch"]; tau = tr["tau"]
    steady = t >= STEADY_T
    rms = float(np.sqrt(np.mean((pit[steady] - THETA_REF_DEG) ** 2)))
    peak = float(np.max(np.abs(pit)))
    jerk = float(np.mean(np.abs(np.diff(tau)))) if len(tau) > 1 else 0.0
    success = bool(peak < 110.0 and np.mean(pit[steady]) > 0.5 * THETA_REF_DEG)
    return dict(rms=rms, peak=peak, jerk=jerk,
                solve_mean=float(np.mean(tr["solve_ms"])),
                solve_p95=float(np.percentile(tr["solve_ms"], 95)),
                steady_mean=float(np.mean(pit[steady])), success=success)


def main():
    xml = make_flat_xml()
    rng = np.random.default_rng(0)
    inits = [(s, float(rng.uniform(-2, 2)), float(rng.uniform(-0.1, 0.1)))
             for s in range(N_TRIALS)]

    results = {}
    for name, mk in [("NMPC", make_nmpc), ("MPPI", make_mppi)]:
        print(f"running {name}: {N_TRIALS} trials ...", flush=True)
        trials, mets = [], []
        for (seed, ip, iv) in inits:
            tr = run_trial(mk, seed, ip, iv, xml)
            trials.append(tr); mets.append(trial_metrics(tr))
        results[name] = {"trials": trials, "mets": mets}

    # ── summary table ──
    keys = ["rms", "jerk", "peak", "solve_mean", "solve_p95"]
    labels = {"rms": "RMS pitch err [deg]", "jerk": "mean |dtau| [N.m]",
              "peak": "peak pitch [deg]", "solve_mean": "solve mean [ms]",
              "solve_p95": "solve p95 [ms]"}
    print("\n" + "=" * 64)
    print(f"Isolated wheelie to {THETA_REF_DEG:.0f} deg, {N_TRIALS} trials each")
    print(f"{'metric':22s} | {'NMPC (mean±std)':>18s} | {'MPPI (mean±std)':>18s}")
    print("-" * 64)
    for k in keys:
        row = []
        for name in ("NMPC", "MPPI"):
            v = np.array([m[k] for m in results[name]["mets"]])
            row.append(f"{v.mean():6.2f} ± {v.std():4.2f}")
        print(f"{labels[k]:22s} | {row[0]:>18s} | {row[1]:>18s}")
    for name in ("NMPC", "MPPI"):
        sr = np.mean([m["success"] for m in results[name]["mets"]])
        print(f"{'success rate':22s} | {name}: {sr*100:.0f}%")
    print("=" * 64)

    # ── plot 1: pitch trajectories ──
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, name, color in [(axs[0], "NMPC", "C0"), (axs[1], "MPPI", "C1")]:
        for tr in results[name]["trials"]:
            ax.plot(tr["t"], tr["pitch"], color=color, lw=1.0, alpha=0.5)
        ax.axhline(THETA_REF_DEG, ls="--", c="k", lw=1.3, label="target")
        ax.axvline(STEADY_T, ls=":", c="gray", lw=1.0)
        ax.set_title(f"{name}: wheelie pitch, {N_TRIALS} trials")
        ax.set_xlabel("time [s]"); ax.grid(True, alpha=0.4); ax.legend()
    axs[0].set_ylabel("body pitch [deg]")
    fig.suptitle("Isolated wheelie maneuver — gradient NMPC vs sampling MPPI")
    fig.tight_layout()
    p1 = HERE / "wheelie_quality_trajectories.png"
    fig.savefig(p1, dpi=150); plt.close(fig)

    # ── plot 2: metric box plots ──
    box_keys = ["rms", "jerk", "peak", "solve_mean"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for ax, k in zip(axs, box_keys):
        data = [[m[k] for m in results["NMPC"]["mets"]],
                [m[k] for m in results["MPPI"]["mets"]]]
        bp = ax.boxplot(data, tick_labels=["NMPC", "MPPI"], patch_artist=True)
        for patch, c in zip(bp["boxes"], ["C0", "C1"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_title(labels[k]); ax.grid(True, alpha=0.4)
    fig.suptitle(f"Wheelie-quality metrics ({N_TRIALS} trials each) — lower is better")
    fig.tight_layout()
    p2 = HERE / "wheelie_quality_metrics.png"
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"\nsaved plots:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
