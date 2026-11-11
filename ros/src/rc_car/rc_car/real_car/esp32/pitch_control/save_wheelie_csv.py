
#!/usr/bin/env python3

import csv
import serial
from pathlib import Path
from datetime import datetime
from threading import Thread

# ============================================================
# SETTINGS
# ============================================================

PORT = "/dev/ttyACM0"
BAUD = 115200
SAVE_DIR = Path(__file__).resolve().parent / "wheelie_logs"

# ============================================================
# MAIN
# ============================================================

SAVE_DIR.mkdir(parents=True, exist_ok=True)
filename = SAVE_DIR / f"wheelie_{datetime.now():%Y%m%d_%H%M%S}.csv"

def send_commands(ser):
    while True:
        command = input().strip().upper()

        if command in ("S", "X"):
            ser.write(command.encode("ascii"))

with serial.Serial(PORT, BAUD, timeout=0.1) as ser:
    
    Thread(target=send_commands, args=(ser,), daemon=True).start()

    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_us", "pitch_deg", "omega_rad_s", "pwm_us"])

        print(f"Recording: {filename}")
        samples = 0

        try:
            while True:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                data = line.split(",")

                if len(data) != 4:
                    continue

                try:
                    t = int(data[0])
                    pitch = float(data[1])
                    omega = float(data[2])
                    pwm = int(data[3])
                except ValueError:
                    continue

                writer.writerow([t, pitch, omega, pwm])
                samples += 1

                if samples % 100 == 0:
                    file.flush()

        except KeyboardInterrupt:
            print(f"\nSaved {samples} samples to {filename}")
