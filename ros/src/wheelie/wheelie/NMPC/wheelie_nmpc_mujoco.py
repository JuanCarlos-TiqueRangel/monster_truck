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
CSV_PATH = Path(__file__).with_name("wheelie_mujoco_log.csv")

RENDER = True                 # True = open MuJoCo viewer, False = headless
SIM_TIME = 20.0                # seconds
CTRL_DT = 0.05                # NMPC update period [s]
PRINT_EVERY_N_CONTROLS = 5

# Your MuJoCo root_pitch is negative during a backward wheelie.
# This makes the controller see backward wheelie as positive pitch.
PITCH_SIGN = -1.0

# If the motor acts in the wrong direction, change this to -1.0.
ACTUATOR_SIGN = -1.0
TAU_TO_CTRL = 1.0             # controller torque-like command -> MuJoCo ctrl scale

INITIAL_X = 0.0
INITIAL_Z = 0.1512            # good starting height for this XML
INITIAL_ROOT_PITCH_DEG = 0.0  # MuJoCo root_pitch initial angle

PITCH_REF_DEG = 80.0
V_REF = 0.0


# ============================================================
# Model and NMPC parameters
# ============================================================

@dataclass
class WheelieParams:
    m: float = 5.1
    l: float = 0.18
    I_body: float = (1.0 / 12.0) * 5.1 * (0.53**2 + 0.30**2)
    r: float = 0.085
    g: float = 9.81
    c_v: float = 9.0

    tau_min: float = -12.0
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

    # dt: float = CTRL_DT
    # N: int = 10
    # q_x = 0.0
    # q_v = 0.01
    # q_theta = 1000.0
    # q_omega = 15.0
    # r_tau = 0.1
    # r_dtau = 0.01
    # q_terminal_theta = 400.0
    # q_terminal_omega = 100.0
    # ipopt_max_iter: int = 50

    # dt: float = CTRL_DT
    # N: int = 10
    # q_x: float = 3.5
    # q_v: float = 55.0
    # q_theta: float = 1284.0
    # q_omega: float = 0.1
    # r_tau: float = 0.34867
    # r_dtau: float = 2.5
    # q_terminal_theta: float = 700.0
    # q_terminal_omega: float = 0.1
    # ipopt_max_iter: int = 50

@dataclass
class MPCConfig:
    dt: float = CTRL_DT
    N: int = 10
    q_x: float = 5.0
    q_v: float = 55.0
    q_theta: float = 1340.0
    q_omega: float = 0.1
    r_tau: float = 0.34867
    r_dtau: float = 2.5
    q_terminal_theta: float = 700.0
    q_terminal_omega: float = 0.1
    ipopt_max_iter: int = 50


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
        self.last_solution = None
        self._build_solver()

    def _f_ca(self, x, u):
        p = self.p
        # DYNAMICS x = [position, velocity, pitch, pitch_rate]
        xpos_dot = x[1]
        velocity_dot = u[0] / (p.m * p.r) - p.c_v * x[1]
        theta_dot = x[3]
        omega_dot = (-u[0] + p.m * p.g * p.l * ca.cos(x[2])) / p.I_eff
        return ca.vertcat(xpos_dot, velocity_dot, theta_dot, omega_dot)

    def _rk4_ca(self, x, u):
        dt = self.cfg.dt
        k1 = self._f_ca(x, u)
        k2 = self._f_ca(x + 0.5 * dt * k1, u)
        k3 = self._f_ca(x + 0.5 * dt * k2, u)
        k4 = self._f_ca(x + dt * k3, u)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next


    def _build_solver(self):
        p = self.p
        cfg = self.cfg
        N = cfg.N

        X = ca.SX.sym("X", self.nx, N + 1)
        U = ca.SX.sym("U", self.nu, N)

        # P = [x, v, theta, omega, x_ref, v_ref, theta_ref, omega_ref, tau_prev]
        P = ca.SX.sym("P", 9)
        x0 = P[0:4]
        ref = P[4:8]
        tau_prev = P[8]

        Q = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_theta, cfg.q_omega))
        Qf = ca.diag(ca.vertcat(cfg.q_x, cfg.q_v, cfg.q_terminal_theta, cfg.q_terminal_omega))

        cost = 0
        constraints = [X[:, 0] - x0]

        # Stage/running cost: applied at every predicted step
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            e = xk - ref

            if k == 0:
                du = uk[0] - tau_prev
            else:
                du = uk[0] - U[0, k - 1]

            tau_eq = p.m * p.g * p.l * ca.cos(ref[2])
            # Quadratic form x^TQX
            cost += ca.mtimes([e.T, Q, e])
            cost += cfg.r_tau * (uk[0] - tau_eq) ** 2
            cost += cfg.r_dtau * du**2

            x_next = self._rk4_ca(xk, uk)
            constraints.append(X[:, k + 1] - x_next)

        # Terminal cost: applied only once at the final predicted state.
        eN = X[:, N] - ref
        # Quadratic form x^TQX
        cost += ca.mtimes([eN.T, Qf, eN])

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        constraints = ca.vertcat(*constraints)

        nlp = {"f": cost, "x": opt_vars, "g": constraints, "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.tol": 1e-4,
            "print_time": 0,
        }
        self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

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
        self.nX = self.nx * (N + 1)

    def _initial_guess(self, state: np.ndarray, tau_prev: float) -> np.ndarray:
        N = self.cfg.N

        if self.last_solution is None:
            X_guess = np.tile(state.reshape(-1, 1), (1, N + 1))
            U_guess = np.full((self.nu, N), tau_prev)
        else:
            sol = self.last_solution
            X_sol = sol[:self.nX].reshape((self.nx, N + 1), order="F")
            U_sol = sol[self.nX:].reshape((self.nu, N), order="F")

            X_guess = np.hstack([X_sol[:, 1:], X_sol[:, -1:]])
            U_guess = np.hstack([U_sol[:, 1:], U_sol[:, -1:]])
            X_guess[:, 0] = state

        return np.concatenate([
            X_guess.reshape(-1, order="F"),
            U_guess.reshape(-1, order="F"),
        ])

    def solve(self, state: np.ndarray, ref: np.ndarray, tau_prev: float) -> tuple[float, bool]:
        params = np.concatenate([state, ref, np.array([tau_prev])])
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
            return tau, True
        except RuntimeError:
            # Small fallback for debugging: keep the previous command.
            tau = float(np.clip(tau_prev, self.p.tau_min, self.p.tau_max))
            return tau, False


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
    # ref = [x_ref, v_ref, theta_ref, omega_ref]
    ref = np.array([0.0, V_REF, theta_ref, 0.0], dtype=float)

    sim_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round(CTRL_DT / sim_dt)))
    n_steps = int(round(SIM_TIME / sim_dt))

    tau_prev = 0.0
    tau_cmd = 0.0
    ctrl_cmd = 0.0
    solve_success = False
    control_count = 0
    history = []

    def read_controller_state() -> tuple[float, float, float, float, float, float]:
        x = float(data.qpos[root_x_qid])
        z = float(data.qpos[root_z_qid])
        v = float(data.qvel[root_x_vid])
        z_dot = float(data.qvel[root_z_vid])

        raw_pitch = float(data.qpos[root_pitch_qid])
        raw_pitch_dot = float(data.qvel[root_pitch_vid])

        # Convert MuJoCo pitch convention into controller convention.
        theta = PITCH_SIGN * raw_pitch
        omega = PITCH_SIGN * raw_pitch_dot

        return x, z, v, z_dot, theta, omega

    def control_update():
        nonlocal tau_prev, tau_cmd, ctrl_cmd, solve_success, control_count

        x = float(data.qpos[root_x_qid])
        v = float(data.qvel[root_x_vid])

        # Convert MuJoCo pitch convention into controller convention.
        theta = PITCH_SIGN * float(data.qpos[root_pitch_qid])
        omega = PITCH_SIGN * float(data.qvel[root_pitch_vid])

        state = np.array([x, v, theta, omega], dtype=float)
        tau, success = nmpc.solve(state, ref, tau_prev)
        tau = float(np.clip(tau, p.tau_min, p.tau_max))
        tau_prev = tau

        ctrl = ACTUATOR_SIGN * TAU_TO_CTRL * tau
        ctrl = float(np.clip(ctrl, ctrl_min, ctrl_max))
        data.ctrl[drive_id] = ctrl

        # Keep the latest command values for CSV logging.
        tau_cmd = tau
        ctrl_cmd = ctrl
        solve_success = bool(success)

        if control_count % PRINT_EVERY_N_CONTROLS == 0:
            print(
                f"t={data.time:6.3f} | "
                f"x={x:7.3f} | v={v:7.3f} | "
                f"pitch={math.degrees(theta):8.2f} deg | "
                f"omega={omega:8.3f} | tau={tau:8.3f} | "
                f"ctrl={ctrl:8.3f} | success={success}"
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

    save_history_csv(history, CSV_PATH)


if __name__ == "__main__":
    main()
