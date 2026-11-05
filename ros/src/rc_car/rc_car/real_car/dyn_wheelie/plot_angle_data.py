import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/wheelie_20260914_025546.csv"
)

t = df["time"]
pitch = df["pitch_deg"]
pitch_ref = df["pitch_ref_deg"]

plt.figure(figsize=(8, 4.5))

plt.plot(
    t,
    pitch,
    linewidth=2,
    label="Actual pitch",
)

plt.plot(
    t,
    pitch_ref,
    "--",
    linewidth=2,
    label="Reference pitch",
)

plt.xlabel("Time [s]")
plt.ylabel("Pitch [deg]")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
