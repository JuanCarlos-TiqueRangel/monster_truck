#!/usr/bin/env python3
from pymavlink import mavutil
import time
import sys

PORT = "/dev/ttyUSB0"
BAUD = 57600

IMU_HZ = 50
PRINT_EVERY = 1.0

def set_message_interval(master, msg_id: int, hz: float):
    interval_us = int(1_000_000 / hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        msg_id,
        interval_us,
        0, 0, 0, 0, 0
    )

def main():
    print(f"Connecting {PORT} @ {BAUD} ... (close QGC if it uses the same port)")
    m = mavutil.mavlink_connection(PORT, baud=BAUD, robust_parsing=True, autoreconnect=True)

    m.wait_heartbeat()
    print(f"HB OK: sys={m.target_system} comp={m.target_component}")

    # Request SCALED_IMU @ 50 Hz (smaller than HIGHRES_IMU, better for 3DR radios)
    set_message_interval(m, mavutil.mavlink.MAVLINK_MSG_ID_SCALED_IMU, IMU_HZ)

    imu_count = 0
    other_count = 0
    last_print = time.time()

    while True:
        msg = m.recv_match(blocking=True)
        t = time.time()

        if msg.get_type() == "SCALED_IMU":
            imu_count += 1

            # Convert to SI:
            # accel: milli-g -> m/s^2
            # gyro:  mrad/s -> rad/s
            ax = msg.xacc * 9.80665 / 1000.0
            ay = msg.yacc * 9.80665 / 1000.0
            az = msg.zacc * 9.80665 / 1000.0
            gx = msg.xgyro / 1000.0
            gy = msg.ygyro / 1000.0
            gz = msg.zgyro / 1000.0

            # Print a sample occasionally (not every packet)
            if imu_count % 10 == 0:
                print(f"SCALED_IMU sample: acc={ax:+.3f} {ay:+.3f} {az:+.3f}  gyro={gx:+.4f} {gy:+.4f} {gz:+.4f}")

        else:
            other_count += 1

        if t - last_print >= PRINT_EVERY:
            dt = t - last_print
            hz = imu_count / dt
            loss = m.packet_loss()

            print("\n=== SUMMARY ===")
            print(f"Measured SCALED_IMU: {hz:.1f} Hz (requested {IMU_HZ} Hz)")
            print(f"Other msgs: {other_count/dt:.1f} Hz")
            print(f"Packet loss (pymavlink est): {loss:.1f}%")
            print("==============\n")

            imu_count = 0
            other_count = 0
            last_print = t

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
