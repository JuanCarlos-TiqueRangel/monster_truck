import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os

# --- Configuration & Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "results"

skip_experiments = [6, 12, 16, 18]
TEST_EXPERIMENT = 45
EPOCHS = 1000
SEED = 42
LEARNING_RATE = 0.01   
WINDOW_SIZE = 5

# --- 1. Sliding Window Helper Function ---
def create_sliding_windows(x_data, y_data, window_size):
    X_windowed, Y_windowed = [], []
    # Drop the very last window because we don't have a 'next' step to compute a delta against
    for i in range(len(x_data) - window_size):
        window = x_data[i : i + window_size].flatten()
        X_windowed.append(window)
        Y_windowed.append(y_data[i + window_size - 1])
    return np.array(X_windowed, dtype=np.float32), np.array(Y_windowed, dtype=np.float32)

# --- 2. Data Loading & Delta Computation ---
dfs = []
for i in range(1, 147):
    if i in skip_experiments:
        continue
    filename = os.path.join(DATA_DIR, f"rls_id_{i:02d}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        dfs.append(df)

if not dfs:
    raise FileNotFoundError(f"No training CSV data files found in {DATA_DIR}")

X_win_list, Y_win_list = [], []
for df in dfs:
    # 1. Extract raw states
    X_raw = df[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
    v_raw = df["v"].to_numpy(dtype="float32")
    omega_raw = df["omega"].to_numpy(dtype="float32")
    
    # 2. Calculate exact discrete step deltas instead of using noisy v_dot
    delta_v = np.append(np.diff(v_raw), 0) 
    delta_omega = np.append(np.diff(omega_raw), 0)
    Y_raw = np.column_stack((delta_v, delta_omega))
    
    X_w, Y_w = create_sliding_windows(X_raw, Y_raw, WINDOW_SIZE)
    X_win_list.append(X_w)
    Y_win_list.append(Y_w)

X_train_raw = np.vstack(X_win_list)
y_train_raw = np.vstack(Y_win_list)
print(f"Total structured window samples for training (Delta Model): {len(X_train_raw)}")

device = torch.device("cuda:0")
print(f"Using device: {device}")
torch.manual_seed(SEED)

X_tensor = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
Y_tensor = torch.tensor(y_train_raw, dtype=torch.float32, device=device)

X_mean = X_tensor.mean(dim=0)
X_std = X_tensor.std(dim=0) + 1e-8
Y_mean = Y_tensor.mean(dim=0)
Y_std = Y_tensor.std(dim=0) + 1e-8

x_train = (X_tensor - X_mean) / X_std
y_train = (Y_tensor - Y_mean) / Y_std

# --- 3. Network Architecture ---
class DeltaEstimator(nn.Module):
    def __init__(self, input_dim=4, window_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * window_size, 128)
        self.bn1 = nn.BatchNorm1d(128) 
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 2) 
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.fc3(x)  
        return x

model = DeltaEstimator(input_dim=4, window_size=WINDOW_SIZE).to(device)

# We use standard MSE here. Since we are predicting discrete deltas instead of 
# continuous accelerations, the continuous physics constraint (I*alpha + b*omega = 0) 
# is removed to let the network purely optimize for trajectory matching.
criterion = nn.MSELoss()

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=0.01, 
    steps_per_epoch=1, 
    epochs=EPOCHS
)

# --- 4. Training Loop ---
print("Beginning Delta-State Training Loop...")
for epoch in range(EPOCHS):
    model.train()
    
    y_pred = model(x_train) 
    loss = criterion(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if (epoch + 1) % 100 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> MSE Loss: {loss.item():.10f}")

# --- 5. Test File Evaluation Setup ---
test_file = os.path.join(DATA_DIR, f"rls_id_{TEST_EXPERIMENT:02d}.csv")
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Selected evaluation testing file does not exist: {test_file}")
    
test_data = pd.read_csv(test_file)

X_test_raw = test_data[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
v_test_raw = test_data["v"].to_numpy(dtype="float32")
omega_test_raw = test_data["omega"].to_numpy(dtype="float32")

delta_v_test = np.append(np.diff(v_test_raw), 0)
delta_omega_test = np.append(np.diff(omega_test_raw), 0)
Y_test_raw = np.column_stack((delta_v_test, delta_omega_test))

X_test_win, Y_test_win = create_sliding_windows(X_test_raw, Y_test_raw, WINDOW_SIZE)

X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32, device=device)
X_test_norm = (X_test_tensor - X_mean) / X_std

model.eval()
with torch.no_grad():
    Y_test_pred_norm = model(X_test_norm)
    Y_test_pred = Y_test_pred_norm * Y_std + Y_mean

Y_test_np = Y_test_win  
Y_test_pred_np = Y_test_pred.cpu().numpy()

# --- 6. Direct Autoregressive State Stepping (Replaces RK4) ---
t = test_data["elapsed_s"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1 : -1]
v_measured = test_data["v"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1 : -1]
omega_measured = test_data["omega"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1 : -1]
theta_measured = test_data["theta_rad"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1 : -1]

dt = np.diff(t)
num_timesteps = len(t)

v_pred = np.empty(num_timesteps, dtype=np.float64)
omega_pred = np.empty(num_timesteps, dtype=np.float64)
theta_pred = np.empty(num_timesteps, dtype=np.float64)

# Initialize starting states
v_pred[0] = v_measured[0]
omega_pred[0] = omega_measured[0]
theta_pred[0] = theta_measured[0]

# Direct delta accumulation
for k in range(num_timesteps - 1):
    dv = Y_test_pred_np[k, 0]
    domega = Y_test_pred_np[k, 1]

    v_pred[k + 1] = v_pred[k] + dv
    omega_pred[k + 1] = omega_pred[k] + domega
    # Theta integrates the current angular velocity over the timestep
    theta_pred[k + 1] = theta_pred[k] + (omega_pred[k] * dt[k])

time_axis = t - t[0]

# --- 7. Visualization Rendering ---
fig_acc, axs_acc = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axs_acc[0].plot(time_axis, Y_test_np[:num_timesteps, 0], linewidth=2, label="Measured Ground Truth Delta")
axs_acc[0].plot(time_axis, Y_test_pred_np[:num_timesteps, 0], "--", linewidth=2, label="Window NN Delta Prediction")
axs_acc[0].set_ylabel(r"$\Delta v$ [m/s per step]")
axs_acc[0].legend()
axs_acc[0].grid(True)

axs_acc[1].plot(time_axis, Y_test_np[:num_timesteps, 1], linewidth=2, label="Measured Ground Truth Delta")
axs_acc[1].plot(time_axis, Y_test_pred_np[:num_timesteps, 1], "--", linewidth=2, label="Window NN Delta Prediction")
axs_acc[1].set_xlabel("Time [s]")
axs_acc[1].set_ylabel(r"$\Delta \omega$ [rad/s per step]")
axs_acc[1].legend()
axs_acc[1].grid(True)

fig_acc.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — State Step Delta Tracking")
fig_acc.tight_layout()

fig_states, axs_states = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs_states[0].plot(time_axis, v_measured, linewidth=2, label="Measured Sensor")
axs_states[0].plot(time_axis, v_pred, "--", linewidth=2, label="Autoregressive Model Projection")
axs_states[0].set_ylabel(r"$v$ [m/s]")
axs_states[0].legend()
axs_states[0].grid(True)

axs_states[1].plot(time_axis, omega_measured, linewidth=2, label="Measured Sensor")
axs_states[1].plot(time_axis, omega_pred, "--", linewidth=2, label="Autoregressive Model Projection")
axs_states[1].set_ylabel(r"$\omega$ [rad/s]")
axs_states[1].legend()
axs_states[1].grid(True)

axs_states[2].plot(time_axis, np.rad2deg(theta_measured), linewidth=2, label="Measured Sensor")
axs_states[2].plot(time_axis, np.rad2deg(theta_pred), "--", linewidth=2, label="Autoregressive Model Projection")
axs_states[2].set_xlabel("Time [s]")
axs_states[2].set_ylabel(r"$\theta$ [deg]")
axs_states[2].legend()
axs_states[2].grid(True)

fig_states.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — State Rollouts (Discrete Delta Method)")
fig_states.tight_layout()

plt.show()


