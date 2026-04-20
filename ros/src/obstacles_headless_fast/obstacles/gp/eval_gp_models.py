import torch
import matplotlib.pyplot as plt
from svgp_dynamics import SVGPManager

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config_loader import cfg_params

device = torch.device("cuda")

points_number = 200

## Variables data set
xpos_vals = []
for i in range(0, points_number):
    xpos_vals.append(i*0.05)

xpos_dot_vals = []
for i in range(0, points_number):
    xpos_dot_vals.append(i * 0.1)

pitch_vals = []
for i in range(-15, 16):
    pitch_vals.append(i * 0.02)

pitch_dot_vals = []
for i in range(-10, 11):
    pitch_dot_vals.append(i * 0.05)

u_vals = []
for i in range(0, points_number):
    u_vals.append(1)


# xslit = [xpos, xpos_dot, pitch, pitch_dot, U]
X_list = []
#for u in u_vals:
for x_ in xpos_vals:
    X_list.append([x_, 0.0, 0.0, 0.0, 0.0])

X_test = torch.tensor(X_list, dtype=torch.float32, device=device)

gps = [
    SVGPManager.load("models/svgp_dynamics_xpos_delta.pt", device=device),
    SVGPManager.load("models/svgp_dynamics_xpos_dot_delta.pt", device=device),
    SVGPManager.load("models/svgp_dynamics_pitch_delta.pt", device=device),
    SVGPManager.load("models/svgp_dynamics_pitch_dot_delta.pt", device=device),
]

pred_cols = []
for gp in gps:
    pred_cols.append(gp.predict_mean_torch(X_test).reshape(-1, 1))

delta_pred = torch.cat(pred_cols, dim=1)
next_state = X_test[:, :4] + delta_pred



#predictions [xpos, xpos_dot, pitch, pitch_dot]
xpos      = X_test[:, 0].cpu().numpy()
xpos_dot  = X_test[:, 1].cpu().numpy()
pitch     = X_test[:, 2].cpu().numpy()
pitch_dot = X_test[:, 3].cpu().numpy()
u         = X_test[:, 4].cpu().numpy()

dxpos      = delta_pred[:, 0].cpu().numpy()
dxpos_dot  = delta_pred[:, 1].cpu().numpy()
dpitch     = delta_pred[:, 2].cpu().numpy()
dpitch_dot = delta_pred[:, 3].cpu().numpy()

next_xpos      = xpos + dxpos
next_xpos_dot  = xpos_dot + dxpos_dot
next_pitch     = pitch + dpitch
next_pitch_dot = pitch_dot + dpitch_dot

### plots
fig, (ax1, ax2) = plt.subplots(2,1)
fig.suptitle('GP Validation', fontsize=28)


## xpos
ax1.plot(xpos, dpitch, linewidth=3.0)
ax1.set_xlabel("x position", fontsize=28)
ax1.set_ylabel("$\Delta$ $\Phi$", fontsize=28)
ax1.tick_params(axis='both', labelsize=20)
ax1.grid()

## xpos_dot
ax2.plot(xpos, dxpos_dot, linewidth=3.0)
ax2.set_xlabel("x position", fontsize=28)
ax2.set_ylabel("$\Delta$ $\dot{x}$", fontsize=28)
ax2.tick_params(axis='both', labelsize=20)
ax2.grid()

plt.show()


print(xpos)

# print("Delta predictions:\n", delta_pred.cpu().numpy())
# print("\nNext-state predictions:\n", next_state.cpu().numpy())