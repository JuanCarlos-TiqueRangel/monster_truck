# %%

import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# df = pd.read_csv('rls_accel_short.csv')
df = pd.read_csv('rls_accel.csv')

v_dot_hat = np.array(df['v_dot_hat'])
omega_dot_hat = np.array(df['omega_dot_hat'])
v_dot_measured = np.array(df['v_dot_measured'])
omega_dot_measured = np.array(df['omega_dot_measured'])

print(v_dot_hat.shape)

r_v_dot = float(np.corrcoef(v_dot_measured, v_dot_hat)[0, 1])
r_omega_dot = float(np.corrcoef(omega_dot_measured, omega_dot_hat)[0, 1])

fig, axs = plt.subplots(figsize=(20, 15), nrows=2, ncols=1, sharex=True)

# LINEAR ACCELERATION
axs[0].scatter(v_dot_measured, v_dot_hat, color="green", label="data points")
# REFERENCE LINE FOR ACCELERATION
v_lo = min(v_dot_measured.min(), v_dot_hat.min())
v_max = min(v_dot_measured.max(), v_dot_hat.max())
axs[0].plot([v_lo, v_max], [v_lo, v_max], ls="--", color="red", label="perfect fit (y = x)")
axs[0].set_title(f"Correlation plot (r = {r_v_dot:.3f})")
axs[0].set_ylabel("v_dot (m/s^2)")

# OMEGA ACCELERATION
axs[1].scatter(omega_dot_measured, omega_dot_hat, label="data points")
omega_lo = min(omega_dot_measured.min(), omega_dot_hat.min())
omega_max = min(omega_dot_measured.max(), omega_dot_hat.max())
axs[1].plot([omega_lo, omega_max], [omega_lo, omega_max], color="orange", ls="--", label="perfect fit (y = x)")
# REFERENCE LINE FOR ACCELERATION
axs[1].set_title(f"Correlation plot (r = {r_omega_dot:.3f})")
axs[1].set_ylabel("\omega_dot (m/s^2)")


fig.suptitle("Correlation")
fig.savefig("rls_correlation.png", dpi=200, bbox_inches="tight")
#fig.tight_layout()

for ax in axs:
    ax.grid(True)
    ax.legend()


# %%
