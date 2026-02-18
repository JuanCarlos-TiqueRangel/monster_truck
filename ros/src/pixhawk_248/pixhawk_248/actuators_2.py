#!/usr/bin/env python3
import os, time
# Force the MAVLink dialect *before* importing pymavlink (critical)
os.environ["MAVLINK_DIALECT"] = "common"

from pymavlink import mavutil

# --- connection (pick one) ---
DEVICE = "/dev/ttyUSB0"
BAUD   = 57600
USE_SEPARATE_BAUD = True
CONN_STR = "serial:/dev/ttyUSB0:57600"  # alternative single-string style

# normalized commands in [-1, +1] (0.0 = ~1500us)
throttle_cmd = 0.1   # AUX1 (your throttle = Offboard/Peripheral Set 1)
steer_cmd    = -0.9  # AUX2 (your steering = Offboard/Peripheral Set 2)
duration_s   = 2.0   # hold time for the "bump" demo

def connect():
    if USE_SEPARATE_BAUD:
        m = mavutil.mavlink_connection(DEVICE, baud=BAUD)
    else:
        m = mavutil.mavlink_connection(CONN_STR)
    m.wait_heartbeat()
    print(f"[OK] Heartbeat from sys {m.target_system} comp {m.target_component}")
    return m

def _clip(v): return max(-1.0, min(1.0, float(v)))

def set_actuators(m, *, set1=None, set2=None, set3=None, set4=None, set5=None, set6=None):
    """Sends MAV_CMD_DO_SET_ACTUATOR (id=187). param1..param6 -> Sets 1..6."""
    # Be resilient if the symbol isn't in this dialect
    CMD_DO_SET_ACTUATOR = getattr(mavutil.mavlink, "MAV_CMD_DO_SET_ACTUATOR", 187)

    vals = [set1, set2, set3, set4, set5, set6]
    params = [float('nan') if v is None else _clip(v) for v in vals]

    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        CMD_DO_SET_ACTUATOR,   # 187 if symbol missing
        0,
        params[0], params[1], params[2],
        params[3], params[4], params[5],
        float('nan')           # param7 unused
    )
    ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=1.0)
    if ack:
        print(f"[ACK] result={ack.result}")

def bump(m, *, set1=None, set2=None, hold_s=1.0):
    set_actuators(m, set1=set1, set2=set2)
    time.sleep(max(0.0, hold_s))
    # Return touched sets to neutral (0.0)
    set_actuators(m,
        set1=0.0 if set1 is not None else None,
        set2=0.0 if set2 is not None else None
    )

if __name__ == "__main__":
    m = connect()

    # Always start neutral (maps to ~1500us if your DIS/center is configured)
    set_actuators(m, set1=0.0, set2=0.0)

    # Timed "bump" like `actuator_test -t <seconds>`
    bump(m, set1=throttle_cmd, set2=0.0, hold_s=duration_s)
    #time.sleep(2.0)
    bump(m, set1=-0.05, set2=0.0, hold_s=1)
    bump(m, set1=0.05, set2=0.0, hold_s=1)
    bump(m, set1=-0.05, set2=0.0, hold_s=1)
    bump(m, set1=0.05, set2=0.0, hold_s=1)
    bump(m, set1=-0.05, set2=0.0, hold_s=1)
    bump(m, set1=0.05, set2=0.0, hold_s=1)
    bump(m, set1=-0.05, set2=0.0, hold_s=1)
    bump(m, set1=0.05, set2=0.0, hold_s=1)

    # End neutral for safety
    set_actuators(m, set1=0.0, set2=0.0)
    print("[DONE]")
