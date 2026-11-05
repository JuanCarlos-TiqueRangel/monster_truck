#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from params_mppi import MPPIConfig

def analyze_and_plot_csv(log_path):
    df = pd.read_csv(log_path)
    t = df["time"]
    cfg = MPPIConfig()
    t_horizon = t + cfg.N * cfg.dt

    fig, axs = plt.subplots(7, 1, figsize=(12, 12), sharex=True)

    # 1. Pitch Tracking Comparison
    axs[0].plot(t, df["pitch_deg"], "b-", lw=2, label="Pitch Measured (°)")
    axs[0].plot(t, df["ref_pitch_deg"], "r--", lw=1.5, label="Pitch Target (°)")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")
    axs[0].set_title(f"CSV Experiment Analysis: {Path(log_path).name}", fontsize=12, fontweight="bold")

    # 2. Control Input
    axs[1].plot(t, df["u"], "g-", lw=1.5, label="PWM Input (u)")
    axs[1].set_ylabel("Control u")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="upper right")

    # 3. MPPI Horizon End Prediction
    axs[2].plot(t, df["pitch_deg"], "b-", alpha=0.6, label="Pitch Actual (°)")
    axs[2].plot(t_horizon, df["pred_pitch"]*(180/np.pi), "m--", label="MPPI Horizon End θ_N (°)")
    axs[2].set_ylabel("Horizon Pitch [deg]")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper right")

    # 4. Angular Acceleration RLS Fit
    axs[3].plot(t, df["omega_dot_meas"], "k-", alpha=0.5, label=r"$\dot{\omega}$ Measured")
    axs[3].plot(t, df["omega_dot_hat"], "c--", lw=1.5, label=r"$\hat{\dot{\omega}}$ RLS Predicted")
    axs[3].set_ylabel(r"Angular Accel [rad/s$^2$]")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend(loc="upper right")

    # 5. RLS Parameters Evolution (a_0 to a_17)
    param_cols = [c for c in df.columns if c.startswith("a_")]
    for col in param_cols:
        axs[4].plot(t, df[col], label=col, alpha=0.7)
    axs[4].set_ylabel("RLS Weights")
    axs[4].set_xlabel("Time [s]")
    axs[4].grid(True, alpha=0.3)

    # 6. speed Acceleration RLS Fit
    axs[5].plot(t, df["pitch_deg"], "k-", alpha=0.5, label=r"Pitch Measured")
    axs[5].plot(t_horizon, df["pred_pitch"]*(180/np.pi), "c--", lw=1.5, label=r"pitch Predicted")
    axs[5].set_ylabel(r"Pitch pred")
    axs[5].grid(True, alpha=0.3)
    axs[5].legend(loc="upper right")



    axs[6].plot(t, df["omega"], "k-", alpha=0.5, label=r"$\omega$ Measured")
    axs[6].plot(t_horizon, df["pred_angular_speed"], "c--", lw=1.5, label=r"$\omega$ Predicted")
    axs[6].set_ylabel(r"omega pred")
    axs[6].grid(True, alpha=0.3)
    axs[6].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # Find latest CSV log in directory
        logs = sorted(Path(".").glob("mppi_rls_log_*.csv"))
        if not logs:
            print("No CSV log files found matching 'mppi_rls_log_*.csv'!")
            sys.exit(1)
        log_file = logs[-1]

    print(f"Loading CSV: {log_file}")
    analyze_and_plot_csv(log_file)
