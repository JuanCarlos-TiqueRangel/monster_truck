import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

CSV_FILE = "wheelie_001.csv"
PITCH_REF = -64.0

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(CSV_FILE)

t = (data["time_us"] - data["time_us"].iloc[0]) / 1e6
pitch = data["pitch_deg"]
omega = data["omega_rad_s"]
pwm = data["pwm_us"]

# ============================================================
# PLOTTING
# ============================================================

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Pitch
ax[0].plot(t, pitch, label="Measured Pitch", linewidth=2)
ax[0].axhline(PITCH_REF, color="red", linestyle="--", label="Reference")
ax[0].set_ylabel("Pitch (deg)")
ax[0].set_title("Pitch Tracking")
ax[0].legend()
ax[0].grid(True)

# Pitch angular velocity
ax[1].plot(t, omega, label="Pitch Rate", linewidth=2)
ax[1].axhline(0, color="red", linestyle="--", label="Reference")
ax[1].set_ylabel("Pitch Rate (rad/s)")
ax[1].set_title("Pitch Angular Velocity")
ax[1].legend()
ax[1].grid(True)

# PWM
ax[2].plot(t, pwm, label="Applied PWM", linewidth=2)
ax[2].axhline(1500, color="red", linestyle="--", label="Neutral PWM")
ax[2].set_ylabel("PWM (µs)")
ax[2].set_xlabel("Time (s)")
ax[2].set_title("Motor PWM")
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.show()
