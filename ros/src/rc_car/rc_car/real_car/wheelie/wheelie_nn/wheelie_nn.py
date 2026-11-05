import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import os


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "results"

skip_experiments = [6, 12, 16, 18, 17]


TEST_EXPERIMENT = 16
EPOCHS = 1000
NEURONS_HIDDEN = 128
SEED = 45
LEARNING_RATE = 0.001

INTEGRATION_METHOD = "RK4" # "EULER" "RK4"


dfs = []

for i in range(1, 147):
    if i in skip_experiments:
        continue
    filename = os.path.join(DATA_DIR, f"rls_id_{i:03d}.csv")
    df = pd.read_csv(filename)
    dfs.append(df)

# Combine all experiments
data = pd.concat(dfs, ignore_index=True)

print(f"Total samples: {len(data)}")


device = torch.device("cuda:0")
print(f"Using GPU: {torch.cuda.get_device_name(device)}")


# Create training inputs (x) and outputs (y) with a bit of random noise
torch.manual_seed(SEED)  # For reproducibility

X = data[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
Y = data[["v_dot", "omega_dot"]].to_numpy(dtype="float32")

X = torch.tensor(X, dtype=torch.float32, device=device)
Y = torch.tensor(Y, dtype=torch.float32, device=device)

X_mean = X.mean(dim=0)
X_std = X.std(dim=0) + 1e-8

Y_mean = Y.mean(dim=0)
Y_std = Y.std(dim=0) + 1e-8

X_norm = (X - X_mean) / X_std
Y_norm = (Y - Y_mean) / Y_std

x_train = X_norm
y_train = Y_norm

# x_train = torch.linspace(-5, 5, 200, device=device).view(-1, 1)  # 200 points between -5 and 5
# y_train = true_function(x_train) + torch.randn_like(x_train) * 0.5  # Added noise

# 2. Define the Neural Network Architecture
class FunctionEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layer to hidden layer 1
        self.fc1 = nn.Linear(4, NEURONS_HIDDEN)
        # Hidden layer 1 to hidden layer 2
        self.fc2 = nn.Linear(NEURONS_HIDDEN, NEURONS_HIDDEN)
        # Hidden layer 2 to output layer
        self.fc3 = nn.Linear(NEURONS_HIDDEN, 2)
        
        # Activation function (SiLU/Swish or ReLU work well for regression)
        self.activation = nn.SiLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # No activation on the final layer for regression
        return x

# Instantiate the model
model = FunctionEstimator().to(device)

# 3. Define Loss Function and Optimizer
criterion = nn.MSELoss()  # Mean Squared Error is standard for function estimation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer with a 0.01 learning rate

# 4. Training Loop
epochs = EPOCHS


for epoch in range(epochs):

    model.train()  # Set model to training mode
    
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_train)
    
    # Compute loss
    loss = criterion(y_pred, y_train)
    
    # Zero gradients, perform a backward pass, and update the weights
    optimizer.zero_grad()  # Clear accumulated gradients from previous steps
    loss.backward()        # Compute gradients via backpropagation
    optimizer.step()       # Update weights
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}")



# 5. Evaluate model on experiment 19

test_file = os.path.join(DATA_DIR, f"rls_id_{TEST_EXPERIMENT:03d}.csv")
test_data = pd.read_csv(test_file)

X_test = test_data[["u", "v", "theta_rad", "omega"]].to_numpy(dtype="float32")
Y_test = test_data[["v_dot", "omega_dot"]].to_numpy(dtype="float32")

X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
Y_test = torch.tensor(Y_test, dtype=torch.float32, device=device)

# Normalize using TRAINING statistics
X_test_norm = (X_test - X_mean) / X_std

model.eval()

with torch.no_grad():
    Y_test_pred_norm = model(X_test_norm)

    # Convert prediction back to physical units
    Y_test_pred = Y_test_pred_norm * Y_std + Y_mean



# 7. Copy results to CPU
Y_test_np = Y_test.cpu().numpy()
Y_test_pred_np = Y_test_pred.cpu().numpy()
samples = np.arange(len(Y_test_np))



# Use predictions AFTER converting back to physical units
acceleration_pred = Y_test_pred.detach().cpu().numpy()

# Time and measured states from the same test experiment
t = test_data["elapsed_s"].to_numpy(dtype=np.float64)
v_measured = test_data["v"].to_numpy(dtype=np.float64)
omega_measured = test_data["omega"].to_numpy(dtype=np.float64)
theta_measured = test_data["theta_rad"].to_numpy(dtype=np.float64)

dt = np.diff(t)

# Allocate integrated states
v_pred = np.empty(len(t), dtype=np.float64)
omega_pred = np.empty(len(t), dtype=np.float64)
theta_pred = np.empty(len(t), dtype=np.float64)

# Initialize from the first measured state, not from zero
v_pred[0] = v_measured[0]
omega_pred[0] = omega_measured[0]
theta_pred[0] = theta_measured[0]


if INTEGRATION_METHOD == "EULER":
    # Forward Euler integration
    for k in range(len(t) - 1):
        v_pred[k + 1] = v_pred[k] + acceleration_pred[k, 0] * dt[k]
        omega_pred[k + 1] = omega_pred[k] + acceleration_pred[k, 1] * dt[k]
        theta_pred[k + 1] = theta_pred[k] + omega_pred[k] * dt[k]


if INTEGRATION_METHOD == "RK4":
    # RK4 integration of the existing predicted acceleration samples
    # State order: [v, theta, omega]
    for k in range(len(t) - 1):
        h = dt[k]
        step_size = 0.5
        a_start = acceleration_pred[k]
        a_end = acceleration_pred[k + 1]
        a_middle = step_size * (a_start + a_end)

        state = np.array([v_pred[k], theta_pred[k], omega_pred[k]], dtype=np.float64)

        # Derivative order: [v_dot, omega, omega_dot]
        k1 = np.array([a_start[0], state[2], a_start[1]])
        k2 = np.array([a_middle[0], state[2] + step_size*h*k1[2], a_middle[1]])
        k3 = np.array([a_middle[0], state[2] + step_size*h*k2[2], a_middle[1]])
        k4 = np.array([a_end[0], state[2] + h*k3[2], a_end[1]])

        state_next = state + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)

        v_pred[k + 1], theta_pred[k + 1], omega_pred[k + 1] = state_next



time = t - t[0]

# Figure 1: Measured vs NN-predicted accelerations
fig_acc, axs_acc = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Linear acceleration
axs_acc[0].plot(time, Y_test_np[:, 0], linewidth=2, label="Measured")
axs_acc[0].plot(time, Y_test_pred_np[:, 0], "--", linewidth=2, label="NN Prediction")
axs_acc[0].set_ylabel(r"$\dot{v}$ [m/s²]")
axs_acc[0].legend()
axs_acc[0].grid()

# Angular acceleration
axs_acc[1].plot(time, Y_test_np[:, 1], linewidth=2, label="Measured")
axs_acc[1].plot(time, Y_test_pred_np[:, 1], "--", linewidth=2, label="NN Prediction")
axs_acc[1].set_xlabel("Time [s]")
axs_acc[1].set_ylabel(r"$\dot{\omega}$ [rad/s²]")
axs_acc[1].legend()
axs_acc[1].grid()

fig_acc.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — NN Acceleration predictions")
fig_acc.tight_layout()

# Figure 2: Measured vs Euler-integrated states
fig_states, axs_states = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Linear velocity
axs_states[0].plot(time, v_measured, linewidth=2, label="Measured")
axs_states[0].plot(time, v_pred, "--", linewidth=2, label=f"{INTEGRATION_METHOD} integration")
axs_states[0].set_ylabel(r"$v$ [m/s]")
axs_states[0].legend()
axs_states[0].grid()

# Angular velocity
axs_states[1].plot(time, omega_measured, linewidth=2, label="Measured")
axs_states[1].plot(time, omega_pred, "--", linewidth=2, label=f"{INTEGRATION_METHOD} integration")
axs_states[1].set_ylabel(r"$\omega$ [rad/s]")
axs_states[1].legend()
axs_states[1].grid()

# Pitch: convert radians to degrees only for visualization
axs_states[2].plot(time, np.rad2deg(theta_measured), linewidth=2, label="Measured")
axs_states[2].plot(time, np.rad2deg(theta_pred), "--", linewidth=2, label=f"{INTEGRATION_METHOD} integration")
axs_states[2].set_xlabel("Time [s]")
axs_states[2].set_ylabel(r"$\theta$ [deg]")
axs_states[2].legend()
axs_states[2].grid()

fig_states.suptitle(f"Experiment {TEST_EXPERIMENT:02d} — {INTEGRATION_METHOD}-integrated states")
fig_states.tight_layout()

# Display both figures
plt.show()


