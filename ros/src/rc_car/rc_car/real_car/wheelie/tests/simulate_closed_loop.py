"""Optimistic matched-model synthetic closed-loop checks, not car calibration."""
import sys
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mppi import MPPITorch
from params_mppi import MPPIConfig, WheelieParams
from rls_clean import rls_predict_acceleration


def physical_coefficients(balance_deg, accel_gain=140.0):
    """Planar body about rear contact, with positive pitch toward the floor.

    Unit input is the existing bounded abstract /drive command, not Nm.
    Geometry/motor gain are explicitly synthetic and never saved as car weights.
    """
    mass, gravity, com_radius, body_inertia = 1.3, 9.81, 0.12, 0.004
    angle = math.radians(balance_deg)
    com_x = com_radius * math.sin(angle)
    com_h = com_radius * math.cos(angle)
    inertia = body_inertia + mass * com_radius**2
    a = np.zeros(12)
    a[0], a[2] = accel_gain, -1.0
    a[5], a[6] = -mass * com_h / inertia, mass * com_x / inertia
    a[7], a[8] = mass * gravity * com_x / inertia, mass * gravity * com_h / inertia
    a[9] = -0.3
    return a, {'mass_kg': mass, 'com_x_m': com_x, 'com_h_m': com_h,
               'pitch_inertia_kgm2': inertia, 'accel_gain': accel_gain,
               'liftoff_acceleration_m_s2': gravity * com_x / com_h,
               'liftoff_input_at_rest': gravity * com_x / com_h / accel_gain}


def plant_step(s, u, a, dt):
    def deriv(x):
        vd, wd = rls_predict_acceleration(x, u, a)
        return np.array([x[1], vd, x[3], wd])
    for _ in range(10):
        h = dt/10
        k1 = deriv(s)
        k2 = deriv(s + .5*h*k1)
        k3 = deriv(s + .5*h*k2)
        k4 = deriv(s + h*k3)
        s = s + h*(k1+2*k2+2*k3+k4)/6
        if s[2] >= 0:
            s[2] = 0
            s[3] = min(s[3], 0)
    return s


def run(balance_deg, seed, initial_pitch=None, steps=100):
    a, details = physical_coefficients(balance_deg)
    cfg, p = MPPIConfig(), WheelieParams()
    ctrl = MPPITorch(p, cfg, a, device='cpu', integrator='rk4')
    torch.manual_seed(seed)
    ref = np.array([0, 0, -math.radians(balance_deg), 0])
    state = np.array([0., 0., math.radians(initial_pitch or 0.), 0.])
    previous = 0
    states, commands, durations = [state.copy()], [], []
    fallbacks = 0
    for _ in range(steps):
        started = time.perf_counter()
        u, info = ctrl.solve(state, ref, previous)
        durations.append(time.perf_counter()-started)
        if not info['success']:
            raise RuntimeError(info)
        fallbacks += int(info.get('used_best_sample_fallback', False))
        state = plant_step(state, u, a, cfg.dt)
        previous = u
        states.append(state.copy());commands.append(u)
    states = np.array(states);commands = np.array(commands)
    result = {'balance_deg':balance_deg,'seed':seed,'initial_pitch_deg':initial_pitch or 0,
              'seconds_simulated':steps*cfg.dt,'plant':details,'pitch_range_deg':np.rad2deg(states[:,2]).min().item(),
              'final_pitch_deg':np.rad2deg(states[-1,2]).item(),
              'last_second_pitch_rmse_deg':np.sqrt(np.mean(np.rad2deg(states[-25:,2]-ref[2])**2)).item(),
              'final_velocity':states[-1,1].item(),'final_omega':states[-1,3].item(),
              'commands_min':commands.min().item(),'commands_max':commands.max().item(),
              'max_abs_velocity':np.max(np.abs(states[:,1])).item(),
              'max_abs_omega':np.max(np.abs(states[:,3])).item(),
              'liftoff_time_s':float(np.flatnonzero(states[:,2] < -math.radians(2))[0]*cfg.dt)
                  if np.any(states[:,2] < -math.radians(2)) else None,
              'best_sample_fallbacks':fallbacks,
              'solve_ms_mean':np.mean(durations)*1000,'solve_ms_max':max(durations)*1000}
    return result


if __name__ == '__main__':
    for angle in (35,50):
        for seed in (0,1,2):
            print(json.dumps(run(angle,seed,steps=100)),flush=True)
        for seed in (0,1):
            print(json.dumps(run(angle,seed,initial_pitch=-angle+3,steps=100)),flush=True)
