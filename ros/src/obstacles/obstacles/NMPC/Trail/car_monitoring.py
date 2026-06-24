"""
Run monster_truck_flip_2d.xml in MuJoCo with a live viewer, keyboard drive
control, and three live plots (raw data, no filtering):
    Figure 1 : accelerations  -> linear (x,y,z) + angular (roll,pitch,yaw)
    Figure 2 : speeds         -> linear (x,y,z) + angular (roll,pitch,yaw)
    Figure 3 : pitch angle    -> root_pitch joint angle [deg]

Controls (focus the viewer window):
    W : +torque (forward)    S : -torque (reverse)    0 : neutral

Run:  python run_monster_truck.py     (macOS: mjpython run_monster_truck.py)
"""

import time
from collections import deque
import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

XML_PATH       = "monster_truck_flip_2d.xml"
WINDOW_SECONDS = 6.0
PLOT_HZ        = 30.0
REC_DECIM      = 10        # plot 1 sample per 10 physics steps (1000Hz -> 100Hz)
THROTTLE_STEP  = 6.0

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

dt     = model.opt.timestep
rec_dt = dt * REC_DECIM
maxlen = int(WINDOW_SECONDS / rec_dt)

act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_motor")
ctrl_lo, ctrl_hi = model.actuator_ctrlrange[act_id]

def sensor_slice(name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    return slice(adr, adr + model.sensor_dim[sid])

ACC_SL  = sensor_slice("imu_acc")    # linear accel (raw, incl. gravity), site frame
GYRO_SL = sensor_slice("imu_gyro")   # angular velocity (raw), site frame
VEL_SL  = sensor_slice("imu_vel")    # linear velocity (raw), site frame

# pitch angle = root_pitch hinge joint position (raw, radians, continuous through a flip)
PITCH_QADR = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_pitch")]

throttle = 0.0
KEY_FWD, KEY_BACK, KEY_ZERO = 87, 83, 48   # W, S, 0

def key_callback(keycode):
    global throttle
    if keycode == KEY_FWD:
        throttle = float(np.clip(throttle + THROTTLE_STEP, ctrl_lo, ctrl_hi))
        print(f"throttle = {throttle:+6.1f}")
    elif keycode == KEY_BACK:
        throttle = float(np.clip(throttle - THROTTLE_STEP, ctrl_lo, ctrl_hi))
        print(f"throttle = {throttle:+6.1f}")
    elif keycode == KEY_ZERO:
        throttle = 0.0
        print("throttle =   0.0  (neutral)")

t_buf = deque(maxlen=maxlen)
ax_buf, ay_buf, az_buf = (deque(maxlen=maxlen) for _ in range(3))   # linear accel
aR_buf, aP_buf, aY_buf = (deque(maxlen=maxlen) for _ in range(3))   # angular accel
vx_buf, vy_buf, vz_buf = (deque(maxlen=maxlen) for _ in range(3))   # linear vel
wR_buf, wP_buf, wY_buf = (deque(maxlen=maxlen) for _ in range(3))   # angular vel
pitch_buf = deque(maxlen=maxlen)                                    # pitch angle

prev_gyro = np.zeros(3)
have_prev = False

plt.ion()

def _title(fig, txt):
    try: fig.canvas.manager.set_window_title(txt)
    except Exception: pass

# Figure 1 : accelerations
fig_a, (ax_lin, ax_ang) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
_title(fig_a, "Accelerations")
ax_lin.set_title("Linear acceleration (IMU, incl. gravity) [m/s^2]")
ax_ang.set_title("Angular acceleration (roll/pitch/yaw) [rad/s^2]")
ax_ang.set_xlabel("time [s]")
l_ax, = ax_lin.plot([], [], label="x")
l_ay, = ax_lin.plot([], [], label="y")
l_az, = ax_lin.plot([], [], label="z")
ax_lin.legend(loc="upper right"); ax_lin.grid(True)
l_aR, = ax_ang.plot([], [], label="roll")
l_aP, = ax_ang.plot([], [], label="pitch")
l_aY, = ax_ang.plot([], [], label="yaw")
ax_ang.legend(loc="upper right"); ax_ang.grid(True)

# Figure 2 : speeds
fig_v, (vx_ax, w_ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
_title(fig_v, "Speeds")
vx_ax.set_title("Linear velocity (IMU) [m/s]")
w_ax.set_title("Angular velocity (roll/pitch/yaw) [rad/s]")
w_ax.set_xlabel("time [s]")
l_vx, = vx_ax.plot([], [], label="x")
l_vy, = vx_ax.plot([], [], label="y")
l_vz, = vx_ax.plot([], [], label="z")
vx_ax.legend(loc="upper right"); vx_ax.grid(True)
l_wR, = w_ax.plot([], [], label="roll")
l_wP, = w_ax.plot([], [], label="pitch")
l_wY, = w_ax.plot([], [], label="yaw")
w_ax.legend(loc="upper right"); w_ax.grid(True)

# Figure 3 : pitch angle
fig_p, p_ax = plt.subplots(1, 1, figsize=(8, 3))
_title(fig_p, "Pitch angle")
p_ax.set_title("Pitch angle [deg]")
p_ax.set_xlabel("time [s]")
l_pitch, = p_ax.plot([], [], label="pitch", color="tab:red")
p_ax.legend(loc="upper right"); p_ax.grid(True)

plt.show(block=False)

def refresh_plots():
    t = np.fromiter(t_buf, float)
    if t.size == 0: return
    l_ax.set_data(t, np.fromiter(ax_buf, float))
    l_ay.set_data(t, np.fromiter(ay_buf, float))
    l_az.set_data(t, np.fromiter(az_buf, float))
    l_aR.set_data(t, np.fromiter(aR_buf, float))
    l_aP.set_data(t, np.fromiter(aP_buf, float))
    l_aY.set_data(t, np.fromiter(aY_buf, float))
    for a in (ax_lin, ax_ang): a.relim(); a.autoscale_view()
    fig_a.canvas.draw_idle(); fig_a.canvas.flush_events()
    l_vx.set_data(t, np.fromiter(vx_buf, float))
    l_vy.set_data(t, np.fromiter(vy_buf, float))
    l_vz.set_data(t, np.fromiter(vz_buf, float))
    l_wR.set_data(t, np.fromiter(wR_buf, float))
    l_wP.set_data(t, np.fromiter(wP_buf, float))
    l_wY.set_data(t, np.fromiter(wY_buf, float))
    for a in (vx_ax, w_ax): a.relim(); a.autoscale_view()
    fig_v.canvas.draw_idle(); fig_v.canvas.flush_events()
    l_pitch.set_data(t, np.fromiter(pitch_buf, float))
    p_ax.relim(); p_ax.autoscale_view()
    fig_p.canvas.draw_idle(); fig_p.canvas.flush_events()

print(__doc__)
step_count  = 0
plot_period = 1.0 / PLOT_HZ
last_plot   = time.time()
wall_start  = time.time()
sim_time    = 0.0

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        target = time.time() - wall_start
        n = 0
        while sim_time < target and n < 50:
            data.ctrl[act_id] = throttle
            mujoco.mj_step(model, data)
            sim_time += dt
            step_count += 1
            n += 1
            if step_count % REC_DECIM == 0:
                acc  = data.sensordata[ACC_SL].copy()
                gyro = data.sensordata[GYRO_SL].copy()
                vel  = data.sensordata[VEL_SL].copy()
                if have_prev:
                    ang_acc = (gyro - prev_gyro) / rec_dt   # raw finite difference
                else:
                    ang_acc = np.zeros(3); have_prev = True
                prev_gyro = gyro
                t_buf.append(sim_time)
                ax_buf.append(acc[0]);  ay_buf.append(acc[1]);  az_buf.append(acc[2])
                aR_buf.append(ang_acc[0]); aP_buf.append(ang_acc[1]); aY_buf.append(ang_acc[2])
                vx_buf.append(vel[0]);  vy_buf.append(vel[1]);  vz_buf.append(vel[2])
                wR_buf.append(gyro[0]); wP_buf.append(gyro[1]); wY_buf.append(gyro[2])
                pitch_buf.append(np.degrees(data.qpos[PITCH_QADR]))
        if n == 0:
            time.sleep(0.0005)
        if sim_time < time.time() - wall_start - 0.5:
            wall_start = time.time() - sim_time
        viewer.sync()
        now = time.time()
        if now - last_plot >= plot_period:
            refresh_plots()
            last_plot = now

print("Viewer closed.")