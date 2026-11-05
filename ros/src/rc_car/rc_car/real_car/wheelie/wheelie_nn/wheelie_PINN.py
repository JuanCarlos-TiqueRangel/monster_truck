
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

skip_experiments = [6, 12, 16, 18, 17]
TEST_EXPERIMENT = 100
EPOCHS = 1000          # 50k epochs is typically plenty when utilizing temporal windows
SEED = 42
LEARNING_RATE = 0.001   # Dropped from 0.01 to 0.001 for stable convergence with windows
WINDOW_SIZE = 5         # Number of historical timesteps to give the network memory

# --- 1. Sliding Window Helper Function ---
def create_sliding_windows(x_data, y_data, window_size):
    """
    Transforms continuous time-series steps into sliding windows.
    Flattens historical steps into a single feature vector per sample.
    """
    X_windowed, Y_windowed = [], []
    for i in range(len(x_data) - window_size + 1):
        # Slice window_size rows and flatten them into a 1D array
        window = x_data[i : i + window_size].flatten()
        X_windowed.append(window)
        # Target corresponds to the acceleration values at the final step of the window
        Y_windowed.append(y_data[i + window_size - 1])
    return np.array(X_windowed, dtype=np.float32), np.array(Y_windowed, dtype=np.float32)

# --- 2. Data Loading & Preprocessing ---
dfs = []
for i in range(1, 147):
    if i in skip_experiments:
        continue
    filename = os.path.join(DATA_DIR, f"rls_id_{i:03d}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        dfs.append(df)

if not dfs:
    raise FileNotFoundError(f"No training CSV data files found in {DATA_DIR}")

# Process training data per experiment to avoid mixing boundaries across runs
X_win_list, Y_win_list = [], []
for df in dfs:
    X_raw = df[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
    Y_raw = df[["v_dot", "omega_dot"]].to_numpy(dtype="float32")
    
    X_w, Y_w = create_sliding_windows(X_raw, Y_raw, WINDOW_SIZE)
    X_win_list.append(X_w)
    Y_win_list.append(Y_w)

X_train_raw = np.vstack(X_win_list)
y_train_raw = np.vstack(Y_win_list)
print(f"Total structured window samples for training: {len(X_train_raw)}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(SEED)

# Move raw data tensors to hardware target
X_tensor = torch.tensor(X_train_raw, dtype=torch.float32, device=device)
Y_tensor = torch.tensor(y_train_raw, dtype=torch.float32, device=device)

# Save normalization factors based on the window-flattened datasets
X_mean = X_tensor.mean(dim=0)
X_std = X_tensor.std(dim=0) + 1e-8
Y_mean = Y_tensor.mean(dim=0)
Y_std = Y_tensor.std(dim=0) + 1e-8

x_train = (X_tensor - X_mean) / X_std
y_train = (Y_tensor - Y_mean) / Y_std

# --- 3. Network Architecture ---
class FunctionEstimator(nn.Module):
    def __init__(self, input_dim=4, window_size=5):
        super().__init__()
        # Input dimension scales up linearly based on window memory depth
        self.fc1 = nn.Linear(input_dim * window_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2) # Two outputs: [v_dot, omega_dot]
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  
        return x

# --- 4. Physics-Informed Loss Function ---
class PhysicsInformedLoss(nn.Module):
    def __init__(self, x_mean, x_std, y_mean, y_std, window_size):
        super(PhysicsInformedLoss, self).__init__()
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std
        self.window_size = window_size
        
        # Learnable structural dynamic physics parameters
        self.I = nn.Parameter(torch.tensor([1.0], requires_grad=True))   
        self.b = nn.Parameter(torch.tensor([0.1], requires_grad=True))   
        self.mse = nn.ModuleDict({'data': nn.MSELoss(), 'physics': nn.MSELoss()})

    def forward(self, y_pred, x_norm, y_true_norm):
        loss_data = self.mse['data'](y_pred, y_true_norm)
        
        # Extract predicted pitch acceleration (omega_dot is index 1) & convert to physical units
        omega_dot_phys = y_pred[:, 1:2] * self.y_std[1] + self.y_mean[1]
        
        # Extract pitch velocity input corresponding to the CURRENT timestep (final step in the window)
        # Features per timestep = 4. Omega is the 4th feature index (index 3).
        current_step_omega_idx = (self.window_size - 1) * 4 + 3
        
        omega_norm = x_norm[:, current_step_omega_idx : current_step_omega_idx + 1]
        omega_phys = omega_norm * self.x_std[current_step_omega_idx] + self.x_mean[current_step_omega_idx]
        
        # Rotational Equation of Motion Constraint: Residuals = I * alpha + b * omega
        net_torque = self.I * omega_dot_phys + self.b * omega_phys
        loss_physics = self.mse['physics'](net_torque, torch.zeros_like(net_torque))
        
        return loss_data, loss_physics


# Instantiate models and optimization engines
model = FunctionEstimator(input_dim=4, window_size=WINDOW_SIZE).to(device)
pinn_loss_module = PhysicsInformedLoss(X_mean, X_std, Y_mean, Y_std, WINDOW_SIZE).to(device)

optimizer = optim.Adam(
    list(model.parameters()) + list(pinn_loss_module.parameters()), 
    lr=LEARNING_RATE
)

lambda_physics = 0.01  # Modulated to preserve system tracking response speeds

# --- 5. Training Loop Execution ---
print("Beginning Physics-Informed Training Loop...")
for epoch in range(EPOCHS):
    model.train()
    
    y_pred = model(x_train) 
    loss_data, loss_physics = pinn_loss_module(y_pred, x_train, y_train)
    total_loss = loss_data + (lambda_physics * loss_physics)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] -> Total: {total_loss.item():.6f} | Data: {loss_data.item():.6f} | Phys: {loss_physics.item():.6f}")
        print(f"   Inertia Param (I): {pinn_loss_module.I.item():.4f}, Damping Param (b): {pinn_loss_module.b.item():.4f}")

# --- 6. Test File Evaluation Setup ---
test_file = os.path.join(DATA_DIR, f"rls_id_{TEST_EXPERIMENT:03d}.csv")
if not os.path.exists(test_file):
    raise FileNotFoundError(f"Selected evaluation testing file does not exist: {test_file}")
    
test_data = pd.read_csv(test_file)

X_test_raw = test_data[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
Y_test_raw = test_data[["v_dot", "omega_dot"]].to_numpy(dtype="float32")

# Create matching evaluation testing sliding window configurations
X_test_win, Y_test_win = create_sliding_windows(X_test_raw, Y_test_raw, WINDOW_SIZE)

X_test_tensor = torch.tensor(X_test_win, dtype=torch.float32, device=device)
X_test_norm = (X_test_tensor - X_mean) / X_std

model.eval()
with torch.no_grad():
    Y_test_pred_norm = model(X_test_norm)
    # Un-normalize back to engineering units using training scaling configurations
    Y_test_pred = Y_test_pred_norm * Y_std + Y_mean

# Move evaluation data states arrays out to cpu space
Y_test_np = Y_test_win  # Evaluated structural window subset comparison ground truth
Y_test_pred_np = Y_test_pred.cpu().numpy()
acceleration_pred = Y_test_pred_np

# Slice verification sensor arrays to precisely match window limits
t = test_data["elapsed_s"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1:]
v_measured = test_data["v"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1:]
omega_measured = test_data["omega"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1:]
theta_measured = test_data["theta_rad"].to_numpy(dtype=np.float64)[WINDOW_SIZE - 1:]

dt = np.diff(t)
num_timesteps = len(t)

def integrate_state_history(acceleration, dt, v0, theta0, omega0, method):
    """Integrate [v, theta, omega] using predicted [v_dot, omega_dot]."""
    num_steps = len(acceleration)
    v_pred = np.empty(num_steps, dtype=np.float64)
    theta_pred = np.empty(num_steps, dtype=np.float64)
    omega_pred = np.empty(num_steps, dtype=np.float64)

    v_pred[0] = v0
    theta_pred[0] = theta0
    omega_pred[0] = omega0

    for k in range(num_steps - 1):
        h = dt[k]
        state = np.array([v_pred[k], theta_pred[k], omega_pred[k]], dtype=np.float64)
        a_start = acceleration[k]

        if method == "euler":
            # Explicit Euler derivative layout: [v_dot, theta_dot, omega_dot]
            state_next = state + h * np.array([a_start[0], state[2], a_start[1]], dtype=np.float64)
        elif method == "trapezoidal":
            a_end = acceleration[k + 1]
            v_next = state[0] + 0.5 * h * (a_start[0] + a_end[0])
            omega_next = state[2] + 0.5 * h * (a_start[1] + a_end[1])
            theta_next = state[1] + 0.5 * h * (state[2] + omega_next)
            state_next = np.array([v_next, theta_next, omega_next], dtype=np.float64)
        elif method == "rk4":
            a_end = acceleration[k + 1]
            a_middle = 0.5 * (a_start + a_end)

            # Derivative equations array layout: [v_dot, theta_dot (omega), omega_dot]
            k1 = np.array([a_start[0], state[2], a_start[1]], dtype=np.float64)
            k2 = np.array([a_middle[0], state[2] + 0.5 * h * k1[2], a_middle[1]], dtype=np.float64)
            k3 = np.array([a_middle[0], state[2] + 0.5 * h * k2[2], a_middle[1]], dtype=np.float64)
            k4 = np.array([a_end[0], state[2] + h * k3[2], a_end[1]], dtype=np.float64)
            state_next = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unknown integration method: {method}")

        v_pred[k + 1] = state_next[0]
        theta_pred[k + 1] = state_next[1]
        omega_pred[k + 1] = state_next[2]

    return v_pred, theta_pred, omega_pred


def rmse(predicted, measured):
    return np.sqrt(np.mean((predicted - measured) ** 2))


# --- 7. Operational Numerical Integration Engines ---
v_pred_rk4, theta_pred_rk4, omega_pred_rk4 = integrate_state_history(
    acceleration_pred,
    dt,
    v_measured[0],
    theta_measured[0],
    omega_measured[0],
    method="rk4",
)
v_pred_euler, theta_pred_euler, omega_pred_euler = integrate_state_history(
    acceleration_pred,
    dt,
    v_measured[0],
    theta_measured[0],
    omega_measured[0],
    method="euler",
)
v_pred_trap, theta_pred_trap, omega_pred_trap = integrate_state_history(
    acceleration_pred,
    dt,
    v_measured[0],
    theta_measured[0],
    omega_measured[0],
    method="trapezoidal",
)

print("\nState integration RMSE:")
print(
    f"  RK4   -> v: {rmse(v_pred_rk4, v_measured):.6f}, "
    f"omega: {rmse(omega_pred_rk4, omega_measured):.6f}, "
    f"theta: {rmse(np.rad2deg(theta_pred_rk4), np.rad2deg(theta_measured)):.6f} deg"
)
print(
    f"  Euler -> v: {rmse(v_pred_euler, v_measured):.6f}, "
    f"omega: {rmse(omega_pred_euler, omega_measured):.6f}, "
    f"theta: {rmse(np.rad2deg(theta_pred_euler), np.rad2deg(theta_measured)):.6f} deg"
)
print(
    f"  Trap  -> v: {rmse(v_pred_trap, v_measured):.6f}, "
    f"omega: {rmse(omega_pred_trap, omega_measured):.6f}, "
    f"theta: {rmse(np.rad2deg(theta_pred_trap), np.rad2deg(theta_measured)):.6f} deg"
)

# Zero out display window times matching start index points
time_axis = t - t[0]

# --- 8. Visualization Rendering Operations ---
# Plot 1: Measured vs Predicted Accelerations
fig_acc, axs_acc = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axs_acc[0].plot(time_axis, Y_test_np[:, 0], linewidth=2, label="Measured Ground Truth")
axs_acc[0].plot(time_axis, Y_test_pred_np[:, 0], "--", linewidth=2, label="Window NN Prediction")
axs_acc[0].set_ylabel(r"$\dot{v}$ [m/s²]")
axs_acc[0].legend()
axs_acc[0].grid(True)

axs_acc[1].plot(time_axis, Y_test_np[:, 1], linewidth=2, label="Measured Ground Truth")
axs_acc[1].plot(time_axis, Y_test_pred_np[:, 1], "--", linewidth=2, label="Window NN Prediction")
axs_acc[1].set_xlabel("Time [s]")
axs_acc[1].set_ylabel(r"$\dot{\omega}$ [rad/s²]")
axs_acc[1].legend()
axs_acc[1].grid(True)

fig_acc.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — PINN Acceleration Tracking Comparison")
fig_acc.tight_layout()

# Plot 2: State Tracking Accuracy via Numerical Forward Integration
fig_states, axs_states = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
axs_states[0].plot(time_axis, v_measured, linewidth=2, label="Measured Sensor")
axs_states[0].plot(time_axis, v_pred_rk4, "--", linewidth=2, label="RK4 Model Projection")
axs_states[0].plot(time_axis, v_pred_euler, ":", linewidth=2, label="Euler Model Projection")
axs_states[0].plot(time_axis, v_pred_trap, "-.", linewidth=2, label="Trapezoidal Model Projection")
axs_states[0].set_ylabel(r"$v$ [m/s]")
axs_states[0].legend()
axs_states[0].grid(True)

axs_states[1].plot(time_axis, omega_measured, linewidth=2, label="Measured Sensor")
axs_states[1].plot(time_axis, omega_pred_rk4, "--", linewidth=2, label="RK4 Model Projection")
axs_states[1].plot(time_axis, omega_pred_euler, ":", linewidth=2, label="Euler Model Projection")
axs_states[1].plot(time_axis, omega_pred_trap, "-.", linewidth=2, label="Trapezoidal Model Projection")
axs_states[1].set_ylabel(r"$\omega$ [rad/s]")
axs_states[1].legend()
axs_states[1].grid(True)

axs_states[2].plot(time_axis, np.rad2deg(theta_measured), linewidth=2, label="Measured Sensor")
axs_states[2].plot(time_axis, np.rad2deg(theta_pred_rk4), "--", linewidth=2, label="RK4 Model Projection")
axs_states[2].plot(time_axis, np.rad2deg(theta_pred_euler), ":", linewidth=2, label="Euler Model Projection")
axs_states[2].plot(time_axis, np.rad2deg(theta_pred_trap), "-.", linewidth=2, label="Trapezoidal Model Projection")
axs_states[2].set_xlabel("Time [s]")
axs_states[2].set_ylabel(r"$\theta$ [deg]")
axs_states[2].legend()
axs_states[2].grid(True)

fig_states.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — State Integrations (Memory Window Enabled)")
fig_states.tight_layout()

plt.show()
