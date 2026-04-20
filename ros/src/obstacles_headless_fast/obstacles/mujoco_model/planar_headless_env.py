#!/usr/bin/env python3
"""Headless in-process MuJoCo environment for fast training/evaluation.

This keeps the same vehicle/drive dynamics as the ROS2 MuJoCo node, but removes
ROS publishers/subscribers and wall-clock timers. Training can therefore run as
fast as the machine allows, similar to a Gym environment.
"""
import os
import math
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params
from utils import geometry


START_FLIPPED = False
DRIVE_MODEL = "electrical"
ESC_MODE = "bidirectional"
USE_FULLY_CHARGED_PACK = True
ENABLE_ESC_FILTER = True
ESC_TAU = 0.035
UPRIGHT_Z0 = 0.169
FLIPPED_Z0 = 0.182
I_DRIVE_MAX = 160.0
MAX_TENDON_TORQUE_CMD = 45.0


def _load_viewer_module(enable_viewer: bool):
    if not enable_viewer:
        return None
    try:
        import mujoco.viewer as mjviewer
        return mjviewer
    except Exception:
        return None


class PlanarMonsterTruckEnv:
    def __init__(self, ctrl_dt: float = 0.1, enable_viewer: bool = False, start_flipped: bool = START_FLIPPED):
        script_dir = Path(__file__).resolve().parent
        xml_path = script_dir / cfg_params.files.mujoco_model
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.enable_viewer = bool(enable_viewer)
        self.start_flipped = bool(start_flipped)

        self.sim_dt = float(self.model.opt.timestep)
        self.ctrl_dt = float(ctrl_dt)
        self.steps_per_ctrl = max(1, int(round(self.ctrl_dt / self.sim_dt)))

        self.root_x_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_x")
        self.root_z_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_z")
        self.root_pitch_jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "root_pitch")
        if self.root_x_jid < 0 or self.root_z_jid < 0 or self.root_pitch_jid < 0:
            raise RuntimeError("Planar joints 'root_x', 'root_z', 'root_pitch' not found in model")

        self.root_x_qpos_adr = int(self.model.jnt_qposadr[self.root_x_jid])
        self.root_z_qpos_adr = int(self.model.jnt_qposadr[self.root_z_jid])
        self.root_pitch_qpos_adr = int(self.model.jnt_qposadr[self.root_pitch_jid])
        self.root_x_dof_adr = int(self.model.jnt_dofadr[self.root_x_jid])
        self.root_z_dof_adr = int(self.model.jnt_dofadr[self.root_z_jid])
        self.root_pitch_dof_adr = int(self.model.jnt_dofadr[self.root_pitch_jid])

        self.chassis_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "chassis")
        if self.chassis_body_id < 0:
            raise RuntimeError("Body 'chassis' not found in XML")

        gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_gyro")
        if gyro_id < 0:
            raise RuntimeError("imu_gyro sensor not found in XML")
        self.gyro_adr = int(self.model.sensor_adr[gyro_id])
        self.gyro_dim = int(self.model.sensor_dim[gyro_id])

        acc_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "imu_acc")
        if acc_id < 0:
            raise RuntimeError("imu_acc sensor not found in XML")
        self.acc_adr = int(self.model.sensor_adr[acc_id])
        self.acc_dim = int(self.model.sensor_dim[acc_id])

        self.obs_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "obs_box_1")
        if self.obs_geom_id < 0:
            raise RuntimeError("Geom 'obs_box_1' not found in XML")

        self.drive_act_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
        if self.drive_act_id < 0:
            raise RuntimeError("Actuator 'drive_motor' not found in XML")

        self.wheel_dof = {}
        for jname in ["j_fl", "j_fr", "j_rl", "j_rr"]:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Joint '{jname}' not found in XML")
            self.wheel_dof[jname] = int(self.model.jnt_dofadr[jid])

        self.last_action = 0.0
        self.u_esc = 0.0

        self.drive_model = DRIVE_MODEL
        self.esc_mode = ESC_MODE
        self.enable_esc_filter = ENABLE_ESC_FILTER
        self.esc_tau = float(ESC_TAU)

        self.V_bat_oc = 16.8 if USE_FULLY_CHARGED_PACK else 14.8
        self.R_bat = 0.012
        self.R_esc = 0.004
        self.R_motor = 0.028
        self.R_total = self.R_bat + self.R_esc + self.R_motor
        self.Kv_rpm_per_V = 2800.0
        self.Kt = 60.0 / (2.0 * np.pi * self.Kv_rpm_per_V)
        self.Ke = self.Kt
        self.final_drive = 18.76
        self.eta_drive = 0.85
        self.I_drive_max = I_DRIVE_MAX
        self.I_brake_max = 90.0
        self.neutral_deadband = 0.02
        self.brake_speed_thresh = 8.0
        self.wheel_drag_visc = 0.015
        self.wheel_drag_coulomb = 0.06
        self.drag_tanh_eps = 2.0
        self.max_tendon_torque_cmd = MAX_TENDON_TORQUE_CMD
        self.tau_motor_max = self.Kt * self.I_drive_max
        self.tau_drive_max = self.eta_drive * self.final_drive * self.tau_motor_max
        self.omega_wheel_nl = self.V_bat_oc / max(self.Ke * self.final_drive, 1e-9)

        self.viewer = None
        if self.enable_viewer:
            mjviewer = _load_viewer_module(True)
            if mjviewer is not None:
                try:
                    self.viewer = mjviewer.launch_passive(self.model, self.data)
                except Exception:
                    self.viewer = None

        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self._apply_start_pose_flag()
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.elapsed_time = 0.0

    def _apply_start_pose_flag(self) -> None:
        if self.start_flipped:
            pitch0 = math.pi
            z0 = FLIPPED_Z0
        else:
            pitch0 = 0.0
            z0 = UPRIGHT_Z0

        self.data.qpos[self.root_x_qpos_adr] = 0.0
        self.data.qpos[self.root_z_qpos_adr] = z0
        self.data.qpos[self.root_pitch_qpos_adr] = pitch0
        self.data.qvel[self.root_x_dof_adr] = 0.0
        self.data.qvel[self.root_z_dof_adr] = 0.0
        self.data.qvel[self.root_pitch_dof_adr] = 0.0
        for jname in ["j_fl", "j_fr", "j_rl", "j_rr"]:
            self.data.qvel[self.wheel_dof[jname]] = 0.0
        self.data.ctrl[:] = 0.0
        self.u_esc = 0.0
        mj.mj_forward(self.model, self.data)

    def _wheel_speeds(self):
        return tuple(float(self.data.qvel[self.wheel_dof[j]]) for j in ["j_fl", "j_fr", "j_rl", "j_rr"])

    def _avg_wheel_speed(self) -> float:
        w_fl, w_fr, w_rl, w_rr = self._wheel_speeds()
        return 0.25 * (w_fl + w_fr + w_rl + w_rr)

    def _wheel_side_drag_torque(self, w_avg: float) -> float:
        return self.wheel_drag_visc * w_avg + self.wheel_drag_coulomb * np.tanh(w_avg / max(self.drag_tanh_eps, 1e-9))

    def _compute_drive_torque_simple(self, u: float) -> float:
        if abs(u) < self.neutral_deadband:
            return 0.0
        w_avg = self._avg_wheel_speed()
        speed_factor = max(0.0, 1.0 - abs(w_avg) / max(self.omega_wheel_nl, 1e-9))
        tau_drive = u * self.tau_drive_max * speed_factor
        tau_drag = self._wheel_side_drag_torque(w_avg)
        tau_out = tau_drive - tau_drag
        return float(np.clip(tau_out, -self.max_tendon_torque_cmd, self.max_tendon_torque_cmd))

    def _compute_drive_torque_electrical(self, u: float) -> float:
        w_avg = self._avg_wheel_speed()
        omega_motor = self.final_drive * w_avg
        if abs(u) < self.neutral_deadband:
            I_cmd = 0.0
        else:
            if self.esc_mode == "bidirectional":
                V_cmd = u * self.V_bat_oc
                I_cmd = (V_cmd - self.Ke * omega_motor) / max(self.R_total, 1e-9)
                I_cmd = float(np.clip(I_cmd, -self.I_drive_max, self.I_drive_max))
            elif self.esc_mode == "forward_brake_reverse":
                if u < -self.neutral_deadband and w_avg > self.brake_speed_thresh:
                    I_cmd = -abs(u) * self.I_brake_max
                elif u > self.neutral_deadband and w_avg < -self.brake_speed_thresh:
                    I_cmd = abs(u) * self.I_brake_max
                else:
                    V_cmd = u * self.V_bat_oc
                    I_cmd = (V_cmd - self.Ke * omega_motor) / max(self.R_total, 1e-9)
                    I_cmd = float(np.clip(I_cmd, -self.I_drive_max, self.I_drive_max))
            else:
                raise ValueError(f"Unknown esc_mode: {self.esc_mode}")
        tau_motor = self.Kt * I_cmd
        tau_drive = self.eta_drive * self.final_drive * tau_motor
        tau_drag = self._wheel_side_drag_torque(w_avg)
        tau_out = tau_drive - tau_drag
        return float(np.clip(tau_out, -self.max_tendon_torque_cmd, self.max_tendon_torque_cmd))

    def _compute_drive_torque_from_throttle(self, u: float) -> float:
        if self.drive_model == "simple":
            return self._compute_drive_torque_simple(u)
        if self.drive_model == "electrical":
            return self._compute_drive_torque_electrical(u)
        raise ValueError(f"Unknown drive_model: {self.drive_model}")

    def get_observation(self):
        qw, qx, qy, qz = self.data.xquat[self.chassis_body_id]
        gyro = self.data.sensordata[self.gyro_adr:self.gyro_adr + self.gyro_dim]
        wx = float(gyro[0]) if self.gyro_dim >= 3 else 0.0
        wy = float(gyro[1]) if self.gyro_dim >= 3 else 0.0
        wz = float(gyro[2]) if self.gyro_dim >= 3 else 0.0
        roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot = geometry.quat_to_euler_xyz(qw, qx, qy, qz, wx, wy, wz)

        vel6 = np.zeros(6, dtype=np.float64)
        mj.mj_objectVelocity(self.model, self.data, mj.mjtObj.mjOBJ_BODY, self.chassis_body_id, vel6, 0)
        wx_b, wy_b, wz_b, vx, vy, vz = vel6
        x, y, z = self.data.xpos[self.chassis_body_id]

        p = self.data.geom_xpos[self.obs_geom_id]
        obs = {
            "xpos": float(x),
            "xpos_dot": float(vx),
            "pitch": float(pitch),
            "pitch_dot": float(pitch_dot),
            "roll": float(roll),
            "roll_dot": float(roll_dot),
            "yaw": float(yaw),
            "yaw_dot": float(yaw_dot),
            "obstacle_x": float(p[0]),
            "time": float(self.elapsed_time),
        }
        return obs

    def get_state_vector(self) -> np.ndarray:
        obs = self.get_observation()
        return np.array([obs["xpos"], obs["xpos_dot"], obs["pitch"], obs["pitch_dot"]], dtype=np.float32)

    def reset(self):
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        self._apply_start_pose_flag()
        self.elapsed_time = 0.0
        return self.get_observation(), {"reset": True}

    def step(self, action: float):
        u_cmd = float(np.clip(action, -1.0, 1.0))
        self.last_action = u_cmd

        if self.enable_esc_filter:
            alpha = 1.0 - np.exp(-self.ctrl_dt / max(self.esc_tau, 1e-9))
            self.u_esc += alpha * (u_cmd - self.u_esc)
            u_eff = self.u_esc
        else:
            u_eff = u_cmd

        tau_cmd = self._compute_drive_torque_from_throttle(u_eff)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.drive_act_id] = tau_cmd

        for _ in range(self.steps_per_ctrl):
            mj.mj_step(self.model, self.data)
        self.elapsed_time += self.ctrl_dt

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        obs = self.get_observation()
        info = {
            "tau_cmd": float(tau_cmd),
            "u_cmd": float(u_cmd),
            "u_eff": float(u_eff),
            "sim_time": float(self.elapsed_time),
        }
        reward = 0.0
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None


if __name__ == "__main__":
    enable_viewer = os.environ.get("MUJOCO_ENABLE_VIEWER", "0") == "1"
    env = PlanarMonsterTruckEnv(ctrl_dt=0.1, enable_viewer=enable_viewer)
    obs, _ = env.reset()
    for _ in range(200):
        obs, _, _, _, info = env.step(0.3)
        if _ % 20 == 0:
            print({k: round(v, 4) if isinstance(v, float) else v for k, v in obs.items()})
    env.close()
