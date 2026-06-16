#!/usr/bin/env python3
"""
make_paper_plot.py
------------------
Load pre-collected MPPI episode data and generate publication-quality figures.
Saves vector PDF (drop straight into a paper) and 300-dpi PNG.

First collect data with:
    python3 collect_episode_data.py

Then plot with:
    python3 make_paper_plot.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
# Use a backend that supports both saving and displaying
try:
    matplotlib.use("Qt5Agg")      # interactive with display
except:
    matplotlib.use("TkAgg")       # fallback
import matplotlib.pyplot as plt

from sim_harness import GOAL_X

# ---------------------------------------------------------------------------
# 1) Load pre-collected episode data (instant, no simulation)
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).with_name("mppi_episode_data.json")

if not DATA_FILE.exists():
    print(f"ERROR: {DATA_FILE.name} not found!")
    print(f"Run this first: python3 collect_episode_data.py")
    exit(1)

print(f"Loading episode data from {DATA_FILE.name}...")
with open(DATA_FILE) as f:
    data = json.load(f)
    h = data["history"]
t     = np.asarray(h["t"])
x     = np.asarray(h["x"])
v     = np.asarray(h["v"])
pitch = np.asarray(h["pitch_deg"])
tau   = np.asarray(h["tau"]) * -1

CONTROLLER = "MPPI"
ACCENT     = "#0072B2"           # colour-blind-safe blue
OUT        = Path(__file__).with_name("mppi_paper_plot")

# ---------------------------------------------------------------------------
# 2) the figure  (identical layout to nmpc/make_paper_plot.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,        # editable text in the PDF
    "savefig.bbox": "tight",
})

# obstacle x-zones on the course (half-extents from monster_truck_flip_2d.xml)
OBSTACLES = [(0.7, 1.3, "box 1"), (2.7, 3.3, "box 2"), (4.6, 5.5, "cylinders"), 
             (6.7, 7.3, "Squares")]

fig, ax = plt.subplots(4, 1, figsize=(7.0, 8.0), sharex=True,
                       gridspec_kw=dict(hspace=0.10))

settle = abs(x[-1] - GOAL_X)
peak_pitch = float(np.abs(pitch).max())
ax[0].set_title(f"{CONTROLLER}: Four obstacles until the goal "
                f"(settle {settle:.2f} m,  peak pitch {peak_pitch:.0f}$^\\circ$)")

# (a) position, with the obstacle zones and the goal line
for lo, hi, name in OBSTACLES:
    ax[0].axhspan(lo, hi, color="0.88", zorder=0)
    ax[0].text(t[-1] * 0.99, 0.5 * (lo + hi), name, va="center", ha="right",
               fontsize=8, color="0.45")
ax[0].axhline(GOAL_X, ls="--", color="0.35", lw=1.2)
ax[0].text(t[100], GOAL_X, "Goal ", va="bottom", ha="left", fontsize=9, color="0.35")
ax[0].plot(t, x, color=ACCENT)
ax[0].plot(t[-1], x[-1], "o", color=ACCENT, ms=6, zorder=5)   # final stop
ax[0].set_ylabel("position $x$  [m]")

# (b) velocity
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].plot(t, v, color=ACCENT)
ax[1].set_ylabel("velocity $v$  [m/s]")

# (c) pitch
ax[2].axhline(0, color="0.6", lw=0.8)
ax[2].plot(t, pitch, color=ACCENT)
ax[2].set_ylabel("pitch $\\theta$  [deg]")

# (d) torque
ax[3].axhline(0, color="0.6", lw=0.8)
ax[3].plot(t, tau, color=ACCENT)
ax[3].set_ylabel("torque $\\tau$  [N$\\cdot$m]")
ax[3].set_xlabel("time  [s]")

for i, a in enumerate(ax):
    a.grid(True, alpha=0.25)
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.margins(x=0.01)
    a.text(0.008, 0.94, f"({'abcd'[i]})", transform=a.transAxes,
           fontsize=11, fontweight="bold", va="top")

# matched y-axis ranges so MPPI and NMPC compare fairly side by side (set for the
# working runs; widen these if you plot a run that flips / overshoots).
ax[0].set_ylim(-0.5, 9.0)     # position [m]
ax[1].set_ylim(-2.5, 5.0)     # velocity [m/s]
ax[2].set_ylim(-30, 110)      # pitch [deg]
ax[3].set_ylim(-13, 15.0)      # torque [N.m]

fig.align_ylabels(ax)
fig.savefig(str(OUT) + ".pdf")
fig.savefig(str(OUT) + ".png", dpi=300)
print(f"saved  {OUT.name}.pdf  and  {OUT.name}.png")

# ---------------------------------------------------------------------------
# 3) residual figure -- the GP part: the measured dynamics residual (what the GP
#    is fed) vs the GP's learned mean +/- sigma. The GP honestly predicts ~0; the
#    sharp omega_dot spikes at contact are the irreducible part it cannot explain.
# ---------------------------------------------------------------------------
r_v = np.asarray(h["r_v"]);      r_w = np.asarray(h["r_w"])
gv  = np.asarray(h["gp_v"]);     gw  = np.asarray(h["gp_w"])
sv  = np.asarray(h["gp_v_std"]); sw  = np.asarray(h["gp_w_std"])

def residual_panel(a, meas, mean, std, ylabel):
    a.axhline(0, color="0.6", lw=0.8)
    a.fill_between(t, mean - std, mean + std, color=ACCENT, alpha=0.40, lw=0,
                   label=r"GP mean $\pm\,\sigma$ (scaled)")
    a.plot(t, meas, color="0.55", lw=0.9, label="measured residual")
    a.plot(t, mean, color=ACCENT, lw=1.4)
    a.set_ylabel(ylabel)

figr, axr = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                         gridspec_kw=dict(hspace=0.12))
axr[0].set_title(f"{CONTROLLER}: learned dynamics residual  (measured vs GP)")
# COSMETIC band (PLOT ONLY): the true GP predictive sigma is large and ~equal across
# both channels (sn2-dominated, shared kernel), so it would fill the panels. Scale each
# channel's 1-sigma band so its median width is ~BAND_FRAC of that panel -- visibility only.
BAND_FRAC = 0.4
def _band(std, yhalf):
    s = np.asarray(std, float)
    m = np.median(s[s > 0]) if np.any(s > 0) else 1.0
    return s * (BAND_FRAC * yhalf / max(m, 1e-9))
residual_panel(axr[0], r_v, gv, _band(sv, 15.0),  r"$\dot v$ residual  [m/s$^2$]")
residual_panel(axr[1], r_w, gw, _band(sw, 100.0), r"$\dot\omega$ residual  [rad/s$^2$]")
axr[0].legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
axr[1].set_xlabel("time  [s]")
for i, a in enumerate(axr):
    a.grid(True, alpha=0.25)
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.margins(x=0.01)
    a.text(0.008, 0.92, f"({'ab'[i]})", transform=a.transAxes,
           fontsize=11, fontweight="bold", va="top")
# matched y-ranges so MPPI and NMPC residuals compare fairly side by side
axr[0].set_ylim(-15, 15)        # v_dot residual    [m/s^2]
axr[1].set_ylim(-100, 100)      # omega_dot residual [rad/s^2]
# the band is the GP predictive std (1 sigma) SCALED DOWN for visibility only -- the
# true sigma is large (irreducible-noise dominated) and would otherwise fill the panel.
figr.subplots_adjust(bottom=0.17)
figr.text(0.5, 0.035, r"shaded $=$ GP predictive std (1$\sigma$), scaled per panel for "
          r"visibility (true $\sigma$ is large)", ha="center", va="bottom",
          fontsize=7.5, color="0.4", style="italic")
figr.align_ylabels(axr)
OUTR = Path(__file__).with_name("mppi_residual_paper_plot")
figr.savefig(str(OUTR) + ".pdf")
figr.savefig(str(OUTR) + ".png", dpi=300)
print(f"saved  {OUTR.name}.pdf  and  {OUTR.name}.png")

print("\nDisplaying plots...")
plt.show()
