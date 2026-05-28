#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import mujoco
import mujoco.viewer as mj_viewer
import casadi as ca


# ============================================================
# Easy-to-debug settings
# ============================================================

XML_PATH = Path(__file__).with_name("monster_truck_flip_2d.xml")
CSV_PATH = Path(__file__).with_name("wheelie_mujoco_log_rls.csv")

RENDER = True
SIM_TIME = 40.0
CTRL_DT = 0.05
PRINT_EVERY_N_CONTROLS = 5

# Your MuJoCo root_pitch is negative during a backward wheelie.
# This makes the controller see backward wheelie as positive pitch.
PITCH_SIGN = -1.0

# If the motor acts in the wrong direction, change this to -1.0.
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0

INITIAL_X = 0.0
INITIAL_Z = 0.1512
INITIAL_ROOT_PITCH_DEG = 0.0

PITCH_REF_DEG = 80.0
V_REF = 0.0


# ============================================================
# Model and NMPC parameters
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1
    l: float = 0.2
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.085
    g: float = 9.81
    c_v: float = 9.0

    tau_min: float = -8.0
    tau_max: float = 12.0

    theta_min: float = math.radians(0.0)
    theta_max: float = math.radians(100.0)

    omega_min: float = -8.0
    omega_max: float = 8.0

    v_min: float = -5.0
    v_max: float = 5.0

    @property
    def I_eff(self) -> float:
        return self.I_body + self.m * self.l**2


@dataclass
class MPCConfig:
    dt: float = CTRL_DT
    N: int = 10
    q_x: float = 5.0
    q_v: float = 165.0
    q_theta: float = 1340.0
    q_omega: float = 0.1
    r_tau: float = 0.5
    r_dtau: float = 2.5
    q_terminal_theta: float = 700.0
    q_terminal_omega: float = 0.1
    ipopt_max_iter: int = 50

    # dt: float = CTRL_DT
    # N: int = 15

    # q_x: float = 5.0
    # q_v: float = 55.01
    # q_theta: float = 1400.0
    # q_omega: float = 0.1

    # r_tau: float = 0.6
    # r_dtau: float = 0.85

    # q_terminal_theta: float = 700.0
    # q_terminal_omega: float = 0.1

    # ipopt_max_iter: int = 50


# ============================================================
# RLS in one function
# ============================================================

def rls_update(
    state_prev: np.ndarray,
    tau: float,
    state_next: np.ndarray,
    dt: float,
    p: WheelieParams,
    a: np.ndarray,
    P: np.ndarray,
    filtered_omega_dot: float | None,
    forgetting_factor: float = 0.999,
    derivative_alpha: float = 0.85,
    clip_parameters: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, dict]:

    _, v_prev, theta_prev, omega_prev = state_prev
    omega_next = float(state_next[3])

    # 1) Measured angular acceleration
    omega_dot_raw = (omega_next - float(omega_prev)) / dt

    # 2) Filter measured angular acceleration FIRST
    if filtered_omega_dot is None:
        filtered_omega_dot = omega_dot_raw
    else:
        filtered_omega_dot = (
            derivative_alpha * filtered_omega_dot
            + (1.0 - derivative_alpha) * omega_dot_raw
        )

    omega_dot_measured = float(filtered_omega_dot)

    # 3) Nominal angular acceleration
    omega_dot_nominal = (
        -tau + p.m * p.g * p.l * np.cos(theta_prev)
    ) / p.I_eff

    # 4) Residual target
    residual_target = omega_dot_measured - omega_dot_nominal

    # 5) Feature vector: phi = [cos(theta), tau, omega, v, 1]
    phi = np.array(
        [np.cos(theta_prev), tau, omega_prev, v_prev, 1.0],
        dtype=float,
    )

    # 6) RLS prediction of residual
    residual_hat_before = float(phi @ a)
    error_before = residual_target - residual_hat_before

    # 7) RLS gain
    P_phi = P @ phi
    denom = forgetting_factor + float(phi @ P_phi)

    if abs(denom) < 1e-12:
        omega_dot_hat = omega_dot_nominal + residual_hat_before
        info = {
            "omega_dot_raw": float(omega_dot_raw),
            "y": float(omega_dot_measured),
            "y_hat": float(omega_dot_hat),
            "residual_target": float(residual_target),
            "residual_hat": float(residual_hat_before),
            "error": float(omega_dot_measured - omega_dot_hat),
            "skipped": True,
        }
        return a, P, float(filtered_omega_dot), info

    K = P_phi / denom

    # 8) Update residual parameters
    a = a + K * error_before

    # 9) Covariance update
    I = np.eye(5)
    P = ((I - np.outer(K, phi)) @ P) / forgetting_factor
    P = 0.5 * (P + P.T)

    # 10) Clip residual parameters around zero
    if clip_parameters:
        a_g_nom = p.m * p.g * p.l / p.I_eff
        a_tau_nom = -1.0 / p.I_eff

        a[0] = np.clip(a[0], -0.5 * abs(a_g_nom), 0.5 * abs(a_g_nom))
        a[1] = np.clip(a[1], -0.5 * abs(a_tau_nom), 0.5 * abs(a_tau_nom))
        a[2] = np.clip(a[2], -5.0, 5.0)
        a[3] = np.clip(a[3], -1.0, 1.0)
        a[4] = np.clip(a[4], -10.0, 10.0)

    residual_hat_after = float(phi @ a)
    omega_dot_hat = omega_dot_nominal + residual_hat_after

    info = {
        "omega_dot_raw": float(omega_dot_raw),
        "y": float(omega_dot_measured),
        "y_hat": float(omega_dot_hat),
        "residual_target": float(residual_target),
        "residual_hat": float(residual_hat_after),
        "error": float(omega_dot_measured - omega_dot_hat),
        "skipped": False,
    }

    return a, P, float(filtered_omega_dot), info


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


# ============================================================
# NMPC controller
# ============================================================

class WheelieNMPC:
    def __init__(self, p: WheelieParams, cfg: MPCConfig):
        self.p = p
        self.cfg = cfg
        self.nx = 4
        self.nu = 1
        self.n_rls = 5
        self.last_solution = None
        self._build_solver()

    def _omega_dot_ca(self, x, u, a_rls):
        p = self.p

        omega_dot_nominal = (-u[0] + p.m * p.g * p.l * ca.cos(x[2])) / p.I_eff

        omega_dot_rls = (
            a_rls[0] * ca.cos(x[2])
            + a_rls[1] * u[0]
            + a_rls[2] * x[3]
            + a_rls[3] * x[1]
            + a_rls[4]
        )

        return omega_dot_nominal + omega_dot_rls

    def _f_ca(self, x, u, a_rls):
        p = self.p
        # State x = [position, velocity, pitch, pitch_rate]
        x_dot = x[1]
        v_dot = u[0] / (p.m * p.r) - p.c_v * x[1]
        theta_dot = x[3]
        omega_dot = self._omega_dot_ca(x, u, a_rls)

        return ca.vertcat(x_dot, v_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u, a_rls):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u, a_rls)
        k2 = self._f_ca(x + 0.5 * dt * k1, u, a_rls)
        k3 = self._f_ca(x + 0.5 * dt * k2, u, a_rls)
        k4 = self._f_ca(x + dt * k3, u, a_rls)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _build_solver(self):
        cfg = self.cfg
        p = self.p
        N = cfg.N

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # Parameter vector:
        # state [x, v, theta, omega]
        # ref   [x_ref, v_ref, theta_ref, omega_ref]
        # tau_prev
        # RLS coefficients [a_g, a_tau, a_omega, a_v, a_0]
        P = ca.SX.sym("P", 9 + self.n_rls)

        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]
        a_rls = P[9:14]

        obj = 0
        g = []

        g.append(X[:, 0] - x0)

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        # for k in range(N):
        #     xk = X[:, k]
        #     uk = U[:, k]

        #     theta_error = xk[2] - ref[2]

        #     if k == 0:
        #         du = uk[0] - tau_prev
        #     else:
        #         du = uk[0] - U[0, k - 1]

        #     tau_eq = p.m * p.g * p.l * ca.cos(ref[2])

        #     # Angle tracking only
        #     obj += cfg.q_theta * theta_error**2

        #     # Keep torque effort
        #     obj += cfg.r_tau * (uk[0] - tau_eq)**2
        #     #obj += cfg.r_tau * uk[k]**2 #(uk[0] - tau_eq) ** 2

        #     # Keep torque-rate smoothing
        #     #obj += cfg.r_dtau * du**2

        #     x_next = self._rk4_ca(xk, uk, a_rls)
        #     g.append(X[:, k + 1] - x_next)

        # # Terminal angle tracking only
        # theta_error_N = X[2, N] - ref[2]
        # obj += cfg.q_terminal_theta * theta_error_N**2


        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            tau_eq = p.m * p.g * p.l * ca.cos(ref[2])

            obj += ca.mtimes([e.T, Q, e])
            obj += cfg.r_tau * (uk[0] - tau_eq) ** 2
            obj += cfg.r_dtau * du ** 2

            x_next = self._rk4_ca(xk, uk, a_rls)
            g.append(X[:, k + 1] - x_next)

        eN = X[:, N] - ref
        obj += ca.mtimes([eN.T, Qf, eN])



        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g)

        nlp = {"f": obj, "x": opt_vars, "g": g, "p": P}

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.tol": 1e-4,
            "print_time": 0,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        nX = self.nx * (N + 1)

        lbx = []
        ubx = []

        for _ in range(N + 1):
            lbx += [-ca.inf, p.v_min, p.theta_min, p.omega_min]
            ubx += [ ca.inf, p.v_max, p.theta_max, p.omega_max]

        for _ in range(N):
            lbx += [p.tau_min]
            ubx += [p.tau_max]

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)

        self.lbg = np.zeros(self.nx * (N + 1))
        self.ubg = np.zeros(self.nx * (N + 1))

        self.nX = nX

    def _initial_guess(self, state: np.ndarray, tau_prev: float) -> np.ndarray:
        N = self.cfg.N

        if self.last_solution is not None:
            sol = self.last_solution.copy()

            X_sol = sol[:self.nX].reshape((self.nx, N + 1), order="F")
            U_sol = sol[self.nX:].reshape((self.nu, N), order="F")

            X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])

            X_guess[:, 0] = state
        else:
            X_guess = np.tile(state.reshape(-1, 1), (1, N + 1))
            U_guess = np.full((self.nu, N), tau_prev)

        return np.concatenate([
            X_guess.reshape(-1, order="F"),
            U_guess.reshape(-1, order="F"),
        ])

    def solve(
        self,
        state: np.ndarray,
        ref: np.ndarray,
        tau_prev: float,
        a_rls: np.ndarray,) -> tuple[float, dict]:
        
        params = np.concatenate([
            state,
            ref,
            np.array([tau_prev], dtype=float),
            np.asarray(a_rls, dtype=float).reshape(-1),
        ])

        x_init = self._initial_guess(state, tau_prev)

        try:
            sol = self.solver(
                x0=x_init,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )

            w = np.array(sol["x"]).flatten()
            self.last_solution = w

            U_opt = w[self.nX:].reshape((self.nu, self.cfg.N), order="F")
            tau = float(U_opt[0, 0])
            tau = float(np.clip(tau, self.p.tau_min, self.p.tau_max))

            return tau, {"success": True, "cost": float(sol["f"])}

        except RuntimeError as exc:
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, {"success": False, "error": str(exc)}


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
# Main simulation
# ============================================================

def main():
    p = WheelieParams()
    cfg = MPCConfig()
    nmpc = WheelieNMPC(p, cfg)

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
    ref = np.array([0.0, V_REF, theta_ref, 0.0], dtype=float)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    ctrl_dt_actual = ctrl_steps * sim_dt
    n_steps = int(round(SIM_TIME / sim_dt))

    tau_prev = 0.0
    tau_cmd = 0.0
    ctrl_cmd = 0.0
    solve_success = False
    control_count = 0
    history = []

    # RLS settings.
    forgetting_factor = 0.9995
    initial_covariance = 1.0
    derivative_alpha = 0.0
    clip_parameters = False

    # Initial RLS parameters from the nominal physics model.
    a_nom = np.array(
        [
            p.m * p.g * p.l / p.I_eff,
            -1.0 / p.I_eff,
            0.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    #a_rls = a_nom.copy()
    a_rls = np.zeros(5)
    P_rls = initial_covariance * np.eye(5)
    filtered_omega_dot = None
    last_rls_info = {
        "omega_dot_raw": 0.0,
        "y": 0.0,
        "y_hat": 0.0,
        "omega_dot_nominal": 0.0,
        "residual_target": 0.0,
        "residual_hat": 0.0,
        "error": 0.0,
        "skipped": True,
    }

    # State at the previous NMPC update. Used so RLS updates once per control period,
    # not at every 0.001 s MuJoCo integration step.
    last_control_state = None

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

    def control_update():
        nonlocal tau_prev, tau_cmd, ctrl_cmd, solve_success, control_count
        nonlocal a_rls, P_rls, filtered_omega_dot, last_rls_info, last_control_state

        state_now = read_nmpc_state()

        # Update RLS ONCE per control period using the previous applied tau_cmd.
        # This avoids fitting high-frequency MuJoCo contact impulses at 1000 Hz.
        if last_control_state is not None:
            a_rls, P_rls, filtered_omega_dot, last_rls_info = rls_update(
                state_prev=last_control_state,
                tau=tau_cmd,
                state_next=state_now,
                dt=ctrl_dt_actual,
                p=p,
                a=a_rls,
                P=P_rls,
                filtered_omega_dot=filtered_omega_dot,
                forgetting_factor=forgetting_factor,
                derivative_alpha=derivative_alpha,
                clip_parameters=clip_parameters,
            )

        tau, success = nmpc.solve(state_now, ref, tau_prev, a_rls)
        tau = float(np.clip(tau, p.tau_min, p.tau_max))
        tau_prev = tau

        # ACTUATOR_SIGN belongs only here, at the MuJoCo command interface.
        ctrl = ACTUATOR_SIGN * TAU_TO_CTRL * tau
        ctrl = float(np.clip(ctrl, ctrl_min, ctrl_max))
        data.ctrl[drive_id] = ctrl

        tau_cmd = tau
        ctrl_cmd = ctrl
        solve_success = bool(success)
        last_control_state = state_now.copy()

        if control_count % PRINT_EVERY_N_CONTROLS == 0:
            print(
                f"t={data.time:6.3f} | "
                f"x={state_now[0]:7.3f} | v={state_now[1]:7.3f} | "
                f"pitch={math.degrees(state_now[2]):8.2f} deg | "
                f"omega={state_now[3]:8.3f} | tau={tau:8.3f} | "
                f"ctrl={ctrl:8.3f} | success={success} | "
                f"rls_err={last_rls_info['error']:8.3f}"
            )

        control_count += 1

    def log_step():
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
            "solve_success": int(solve_success),

            # RLS logs.
            "omega_dot_raw": float(last_rls_info["omega_dot_raw"]),
            "omega_dot_filtered": float(last_rls_info["y"]),
            "omega_dot_rls": float(last_rls_info["y_hat"]),
            "rls_error": float(last_rls_info["error"]),
            "rls_skipped": int(last_rls_info["skipped"]),
            "a_g": float(a_rls[0]),
            "a_tau": float(a_rls[1]),
            "a_omega": float(a_rls[2]),
            "a_v": float(a_rls[3]),
            "a_0": float(a_rls[4]),
            "a_g_nom": float(a_nom[0]),
            "a_tau_nom": float(a_nom[1]),
        })

    def step_sim():
        mujoco.mj_step(model, data)

    if RENDER:
        k = 0
        with mj_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_TIME:
                start = time.time()

                if k % ctrl_steps == 0:
                    control_update()

                step_sim()
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
            step_sim()
            log_step()

    final_pitch = PITCH_SIGN * float(data.qpos[root_pitch_qid])
    final_omega = PITCH_SIGN * float(data.qvel[root_pitch_vid])

    print("\nFinal state")
    print(f"x      = {float(data.qpos[root_x_qid]):.3f} m")
    print(f"v      = {float(data.qvel[root_x_vid]):.3f} m/s")
    print(f"pitch  = {math.degrees(final_pitch):.2f} deg")
    print(f"omega  = {final_omega:.3f} rad/s")

    print("\nFinal RLS coefficients")
    print("omega_dot = a_g*cos(theta) + a_tau*tau + a_omega*omega + a_v*v + a_0")
    print(f"a_g     learned: {a_rls[0]: .4f} | nominal: {a_nom[0]: .4f}")
    print(f"a_tau   learned: {a_rls[1]: .4f} | nominal: {a_nom[1]: .4f}")
    print(f"a_omega learned: {a_rls[2]: .4f} | nominal: {a_nom[2]: .4f}")
    print(f"a_v     learned: {a_rls[3]: .4f} | nominal: {a_nom[3]: .4f}")
    print(f"a_0     learned: {a_rls[4]: .4f} | nominal: {a_nom[4]: .4f}")

    save_history_csv(history, CSV_PATH)


if __name__ == "__main__":
    main()