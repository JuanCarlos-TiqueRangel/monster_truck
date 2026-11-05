"""
Saves the car's logged data (sent after X, relayed by the PC ESP32) to a .csv file.

Usage:
    pip install pyserial
    Set PORT below, then run:  python log_pitch.py   (or the Run button in your IDE)

Type S + Enter to start the PID, X + Enter to stop it.
After X the car sends everything it recorded; the rows are written to the CSV.
Several S/X runs go into the same file (column "run").
Ctrl+C sends X (safety stop), waits for the data, then closes the file.
Close the Arduino Serial Monitor first: only one program can open the port.
"""
import sys
import threading
import time

import serial

# ============================================================
# SETTINGS
# ============================================================
PORT = "/dev/ttyACM0"    # serial port of the PC ESP32. Windows: "COM5", Linux: "/dev/ttyACM0", Mac: "/dev/cu.usbmodem101"
out = time.strftime("pitch_%Y%m%d_%H%M%S.csv")   # a new file every time you run the script

ser = serial.Serial(PORT, 115200, timeout=0.1)
time.sleep(2.0)            # in case opening the port reset the board
ser.reset_input_buffer()


def keyboard():
    for line in sys.stdin:     # ends quietly if stdin closes (e.g. Ctrl+C on Windows)
        c = line.strip().upper()
        if c in ("S", "X"):
            ser.write(c.encode())


def handle(line, f):
    if line.startswith("D,"):
        f.write(line[2:] + "\n")
    elif line:
        print(line)        # status messages from the ESP32


threading.Thread(target=keyboard, daemon=True).start()

with open(out, "w", newline="", buffering=1) as f:   # line-buffered: rows reach disk as they arrive
    f.write("run,t_ms,pitch_deg,pitch_rate_dps,pwm_us\n")
    print(f"Logging to {out}. S = start, X = stop + send data, Ctrl+C = quit")
    try:
        while True:
            handle(ser.readline().decode(errors="ignore").strip(), f)
    except KeyboardInterrupt:
        ser.write(b"X")
        print("Ctrl+C: sent X, waiting for the car's data...")
        last = time.time()
        while time.time() - last < 2.0:        # give up after 2 s with nothing received
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                last = time.time()
                handle(line, f)
                if "samples received" in line:
                    break
        print(f"Saved {out}")

ser.close()