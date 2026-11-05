#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

MAX_TORQUE = 0.5
N_FOLDERS = 25


# ============================================================
# CONVERT
# ============================================================

for i in range(1, N_FOLDERS + 1):

    folder = ROOT / "results"/ f"optitrack_{i:02d}"

    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in: {folder}")
        continue

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        if "drive" not in df.columns:
            print(f"'drive' column not found: {csv_file}")
            continue

        # Torque [Nm] -> PWM [%]
        #df["drive"] = (df["drive"] / MAX_TORQUE * 100.0).clip(-100.0, 100.0)

        # PWM [%] -> Torque [Nm]
        df["drive"] = df["drive"] / 100.0 * MAX_TORQUE

        df.to_csv(csv_file, index=False)

        print(f"Converted: {csv_file}")


print("\nFinished converting torque -> PWM %.")
