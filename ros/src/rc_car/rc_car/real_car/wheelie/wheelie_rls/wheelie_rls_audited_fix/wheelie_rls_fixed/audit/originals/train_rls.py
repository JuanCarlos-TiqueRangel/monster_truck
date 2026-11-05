
# import warnings
# # Suppress the known matplotlib Axes3D backend import warning
# warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# import os
# from glob import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from rls import rls_update, nominal_rls_parameters, history_features, history_predict  # Strictly using rls.py

# from sklearn.preprocessing import StandardScaler
# from sklearn.kernel_approximation import Nystroem

# from pathlib import Path

# model_path = Path(__file__).resolve().with_name("rls_history_model.npz")

# # ==============================================================================
# # 1. DIRECTORY CONFIGURATION
# # ==============================================================================
# dataset_folder = "wheelie_nn/dataset"
# all_files = sorted(glob(os.path.join(dataset_folder, "*.csv")))

# test_filename = "rls_id_002.csv"
# train_files = [
#     f for f in all_files
#     if os.path.basename(f) not in {test_filename, "rls_id_011.csv"}
# ]
# validation_files = [os.path.join(dataset_folder, test_filename)]




# cols = ["u", "v", "theta_rad", "omega"]
# dfs = [pd.read_csv(f) for f in train_files]
# blocks = [history_features(d[cols], d.elapsed_s) for d in dfs]
# Xv, Xw = (np.vstack([b[j] for b in blocks]) for j in (0, 1))

# sv = StandardScaler().fit(Xv)
# sw = StandardScaler().fit(Xw)
# basis = Nystroem(
#     gamma=0.02, n_components=600, random_state=42
# ).fit(sv.transform(Xv))


# a_trained = np.zeros(622)
# P_trained = np.diag(
#     np.r_[np.full(600, 4.0 / 0.03), 4e6, np.full(20, 25.0), 25e6]
# )
# forgetting_factor = 1.0                

# # ==============================================================================
# # PHASE 2: SEQUENTIAL TRAINING (Updating weights)
# # ==============================================================================
# print(">>> Phase 2: Starting Sequential Training Loop...")
# for file_path in train_files:
#     df = pd.read_csv(file_path)
#     print(f"   Training on: {os.path.basename(file_path)} | Rows: {len(df)}")

#     df["u_eff"] = df["u"].shift(4).fillna(0.0)

#     xv, xw = history_features(df[cols], df.elapsed_s)
#     FV = np.column_stack((basis.transform(sv.transform(xv)), np.ones(len(df))))
#     FW = np.column_stack((sw.transform(xw), np.ones(len(df))))
    
#     for i in range(len(df)):
#         row = df.iloc[i]

#         update_linear = abs(row["v_dot"]) <= 12.0
#         update_angular = (
#             row["sequence_index"] != -1
#             and abs(row["omega_dot"]) <= 40.0
#         )

#         if not update_linear and not update_angular:
#             continue

#         state = np.array([row['x'], row['v'], row['theta_rad'], row['omega']], dtype=float)
        
#         # Capture updated a and updated P
#         a_trained, P_trained, _ = rls_update(
#             state=state, 
#             # tau=float(row['u']),
#             tau=float(row['u_eff']), 
#             vx_dot=float(row['v_dot']), 
#             omega_pitch_dot=float(row['omega_dot']),
#             a=a_trained, 
#             P=P_trained, 
#             forgetting_factor=forgetting_factor,
#             update_angular=(row['sequence_index'] != -1),
#             features=(FV[i], FW[i]),
#         )


# print("\n>>> Global Training Complete.")

# # ==============================================================================
# # PHASE 3: VALIDATION TESTING (Using rls.py outputs directly, weights frozen)
# # ==============================================================================
# print("\n>>> Phase 3: Evaluating Unseen Validation Flight...")

# time_log = []
# v_dot_actual, v_dot_pred = [], []
# omega_dot_actual, omega_dot_pred = [], []

# for file_path in validation_files:
#     df = pd.read_csv(file_path)
#     print(f"   Testing on: {os.path.basename(file_path)}")

#     df["u_eff"] = df["u"].shift(4).fillna(0.0)

#     xv, xw = history_features(df[cols], df.elapsed_s)
#     FV = np.column_stack((basis.transform(sv.transform(xv)), np.ones(len(df))))
#     FW = np.column_stack((sw.transform(xw), np.ones(len(df))))

#     for i in range(len(df)):
#         row = df.iloc[i]
#         state = np.array([row['x'], row['v'], row['theta_rad'], row['omega']], dtype=float)
        
#         # CRITICAL TRICK: We pass a_trained and P_trained to rls_update to get predictions,
#         # but we assign the outputs to dummy variables (_, _) so our master weights REMAIN FROZEN.
#         _, _, info = rls_update(
#             state=state, 
#             # tau=float(row['u']),
#             tau=float(row['u_eff']), 
#             vx_dot=float(row['v_dot']), 
#             omega_pitch_dot=float(row['omega_dot']),
#             a=a_trained,      # Using our final trained weights
#             P=P_trained,      # Using our final trained covariance
#             forgetting_factor=forgetting_factor,
#             update_angular=(row['sequence_index'] != -1),
#             features=(FV[i], FW[i]),
#         )

#         # Log metrics directly out of your rls.py info dictionary
#         time_log.append(float(row['elapsed_s']))
#         v_dot_actual.append(row['v_dot'])
#         v_dot_pred.append(info["v_dot_hat"])  # Pulled directly from rls.py!
        
#         if row['sequence_index'] != -1:
#             omega_dot_actual.append(row['omega_dot'])
#             omega_dot_pred.append(info["omega_dot_hat"])  # Pulled directly from rls.py!
#         else:
#             omega_dot_actual.append(np.nan)
#             omega_dot_pred.append(np.nan)


# # ==============================================================================
# # PHASE 4: SAVED MODEL
# # ==============================================================================

# np.savez_compressed(
#     model_path,
#     model_type="history_nystroem_rls",
#     format_version=1,
#     learned_parameters=a_trained,
#     covariance=P_trained,
#     v_mean=sv.mean_,
#     v_scale=sv.scale_,
#     w_mean=sw.mean_,
#     w_scale=sw.scale_,
#     kernel_centers=basis.components_,
#     kernel_normalization=basis.normalization_,
#     gamma=basis.gamma,
#     feature_columns=np.asarray(cols),
#     history_length=5,
#     history_order="newest_first",
#     history_padding="repeat_oldest",
#     history_dt_floor=0.04,
#     nominal_sample_period=0.04,
#     n_v_features=601,
#     n_w_features=21,
#     forgetting_factor=forgetting_factor,
#     sigma_v_dot=2.0,
#     sigma_omega_dot=5.0,
#     training_files=np.asarray(train_files),
#     validation_file=test_filename,
# )
# print(f"Saved complete RLS model to: {model_path}")






# # ==============================================================================
# # PHASE 4: PLOT THE GENERATED TIME SERIES
# # ==============================================================================
# plt.figure(figsize=(12, 8))

# # Subplot 1: Longitudinal Acceleration Validation
# plt.subplot(2, 1, 1)
# plt.plot(time_log, v_dot_actual, label=r'Actual Measured $\dot{v}$', color='black', alpha=0.7)
# plt.plot(time_log, v_dot_pred, label=r'RLS Prediction $\hat{\dot{v}}$', color='cyan', linestyle='--')

# plt.title(f"Validation Performance: {os.path.basename(validation_files[0])}", fontsize=11)
# plt.ylabel("Linear Acceleration ($m/s^2$)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# # Subplot 2: Angular Acceleration Validation
# plt.subplot(2, 1, 2)
# plt.plot(time_log, omega_dot_actual, label=r'Actual Measured $\dot{\omega}$', color='black', alpha=0.7)
# plt.plot(time_log, omega_dot_pred, label=r'RLS Prediction $\hat{\dot{\omega}}$', color='orange', linestyle='--')
# plt.xlabel("Elapsed Flight Time (seconds)")
# plt.ylabel("Angular Accel ($rad/s^2$)")
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()























































# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path  # Added for absolute directory path management

# # Import the functional modules cleanly from your external rls.py script
# from rls import nominal_rls_parameters, rls_update

# def run_rls_estimation(csv_file_path):
#     print(f"Loading data from: {csv_file_path}")
#     df = pd.read_csv(csv_file_path)
#     if df.empty:
#         print("Error: DataFrame is empty.")
#         return

#     # 1. ADD THIS LINE: Shift the control input to account for system latency
#     df["u_eff"] = df["u"].shift(-2).fillna(0.0)
    
#     num_rows = len(df)
    
#     # Initialize the parameters and wide covariance array structures
#     a = nominal_rls_parameters()
#     P = np.eye(18) * 1000.0
    
#     # Telemetry storage memory allocations
#     v_dot_measured = df['v_dot'].values
#     omega_dot_measured = df['omega_dot'].values
#     v_dot_estimated = np.zeros(num_rows)
#     omega_dot_estimated = np.zeros(num_rows)
#     timestamps = df['elapsed_s'].values
    
#     # Dynamic list matrix to store the weights at each step
#     weight_history = []

#     print("Processing time series steps sequentially via rls.py updates...")
#     for i in range(num_rows):
#         state = np.array([df['x'].iloc[i], df['v'].iloc[i], df['theta_rad'].iloc[i], df['omega'].iloc[i]])
#         tau = float(df['u_eff'].iloc[i])
        
#         # Call the external RLS step function module
#         # Call the external RLS step function module with explicit tuning parameters
#         a, P, info = rls_update(
#             state, 
#             tau, 
#             v_dot_measured[i], 
#             omega_dot_measured[i], 
#             a, 
#             P,
#             forgetting_factor=0.80,  # Reduce memory depth to eliminate algorithmic lag
#             sigma_v_dot=2.0,         # Restored damping
#             sigma_omega_dot=5.0      # Restored damping
#         )
        
        
#         v_dot_estimated[i] = info["v_dot_hat"]
#         omega_dot_estimated[i] = info["omega_dot_hat"]
        
#         # Save active parameter indices based strictly on running features
#         n_active = info["n_v"] + info["n_w"]
#         weight_history.append(a[0:n_active].copy())
        
#     weight_history = np.array(weight_history)
#     n_v_final = info["n_v"]
#     n_w_final = info["n_w"]

#     # ==============================================================================
#     # OUTPUT SUMMARY PRINT: Dumps raw final numeric weights into the console
#     # ==============================================================================
#     print("\n" + "="*50)
#     print("FINAL CONVERGED MODEL WEIGHT SUMMARY")
#     print("="*50)
#     print(f"Linear Model Features (Total Elements: {n_v_final})")
#     for idx in range(n_v_final):
#         print(f"  Position ID {idx} Weight Value: {a[idx]:.5f}")
        
#     print(f"\nAngular Model Features (Total Elements: {n_w_final})")
#     for idx in range(n_w_final):
#         global_idx = n_v_final + idx
#         print(f"  Position ID {idx} Weight Value: {a[global_idx]:.5f}")
#     print("="*50 + "\n")

#     # ==============================================================================
#     # VISUAL RECONSTRUCTION CHARTING
#     # ==============================================================================
#     # Window 1: Accelerations Performance Dashboard
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(timestamps, v_dot_measured, color='black', label='Measured $v_{dot}$')
#     plt.plot(timestamps, v_dot_estimated, color='crimson', linestyle='--', label='RLS Estimated $\hat{v}_{dot}$')
#     plt.ylabel('Linear Accel ($m/s^2$)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(timestamps, omega_dot_measured, color='black', label='Measured $\omega_{dot}$')
#     plt.plot(timestamps, omega_dot_estimated, color='darkcyan', linestyle='--', label='RLS Estimated $\hat{\omega}_{dot}$')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Angular Accel ($rad/s^2$)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     # # Window 2: Weights Evolution History Graph (Position ID Based)
#     # plt.figure(figsize=(12, 8))
#     # plt.subplot(2, 1, 1)
#     # plt.title('Weight Tuning Trajectory Map Over Time (Position Tracking ID)')
#     # for idx in range(n_v_final):
#     #     plt.plot(timestamps, weight_history[:, idx], label=f'Pos ID {idx}')
#     # plt.ylabel('Linear Weights Values')
#     # plt.legend(loc='upper right', ncol=3)
#     # plt.grid(True, alpha=0.3)

#     # plt.subplot(2, 1, 2)
#     # for idx in range(n_w_final):
#     #     global_idx = n_v_final + idx
#     #     plt.plot(timestamps, weight_history[:, global_idx], label=f'Pos ID {idx}')
#     # plt.xlabel('Time (seconds)')
#     # plt.ylabel('Angular Weights Values')
#     # plt.legend(loc='upper right', ncol=3)
#     # plt.grid(True, alpha=0.3)
#     # plt.tight_layout()
    
#     plt.show()

# if __name__ == "__main__":
#     # # Dynamically resolve target data file paths relative to execution directories
#     # BASE_DIR = Path(__file__).resolve().parent
#     # folder_path = BASE_DIR / "wheelie_nn" / "dataset" 
#     # target_file = folder_path / "rls_id_011.csv"
    
#     # if target_file.exists():
#     #     run_rls_estimation(target_file)
#     # else:
#     #     print(f"Error: Target data profile '{target_file}' could not be located.")
#     #     print("Please verify the presence of the wheelie_nn/dataset subdirectory.")





#     # Dynamically resolve target data file paths relative to execution directories
#     BASE_DIR = Path(__file__).resolve().parent
#     folder_path = BASE_DIR / "wheelie_nn" / "dataset"

#     for i in range(100, 110):
#         target_file = folder_path / f"rls_id_{i}.csv"

#         if target_file.exists():
#             run_rls_estimation(target_file)
#         else:
#             print(f"Error: Target data profile '{target_file}' could not be located.")













































# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# # Import the functional modules cleanly from your external rls.py script
# from rls import nominal_rls_parameters, rls_update

# def run_rls_estimation(csv_file_path, a_init=None):
#     print(f"\nLoading data from: {csv_file_path}")
#     df = pd.read_csv(csv_file_path)
#     if df.empty:
#         print("Error: DataFrame is empty.")
#         return a_init, None

#     df["u_eff"] = df["u"].shift(-2).fillna(0.0)
#     num_rows = len(df)
    
#     # =========================================================================
#     # WARM START WITH COVARIANCE RESET
#     # Preserve learned weights (a), but reset covariance matrix (P) so the 
#     # filter remains flexible and adapts quickly to new experimental dynamics.
#     # =========================================================================
#     if a_init is None:
#         print("Initializing weights from zero (first run).")
#         a = nominal_rls_parameters()
#     else:
#         print("Warm starting with weights loaded from previous run / .npz file.")
#         a = a_init.copy()

#     # Always reset covariance for maximum initial flexibility on each new dataset
#     P = np.eye(18) * 1000.0

#     v_dot_measured = df['v_dot'].values
#     omega_dot_measured = df['omega_dot'].values
#     v_dot_estimated = np.zeros(num_rows)
#     omega_dot_estimated = np.zeros(num_rows)
#     timestamps = df['elapsed_s'].values
#     weight_history = []

#     print("Processing time series steps sequentially...")
#     for i in range(num_rows):
#         state = np.array([df['x'].iloc[i], df['v'].iloc[i], df['theta_rad'].iloc[i], df['omega'].iloc[i]])
#         tau = float(df['u_eff'].iloc[i])
        
#         a, P, info = rls_update(
#             state, 
#             tau, 
#             v_dot_measured[i], 
#             omega_dot_measured[i], 
#             a, 
#             P,
#             forgetting_factor=0.95,  
#             sigma_v_dot=2.0,         
#             sigma_omega_dot=5.0      
#         )
        
#         v_dot_estimated[i] = info["v_dot_hat"]
#         omega_dot_estimated[i] = info["omega_dot_hat"]
        
#         n_active = info["n_v"] + info["n_w"]
#         weight_history.append(a[0:n_active].copy())
        
#     weight_history = np.array(weight_history)
#     n_v_final = info["n_v"]
#     n_w_final = info["n_w"]

#     print("\n" + "="*50)
#     print("FINAL CONVERGED MODEL WEIGHT SUMMARY")
#     print("="*50)
#     for idx in range(n_v_final):
#         print(f"  Linear Pos ID {idx}: {a[idx]:.5f}")
#     for idx in range(n_w_final):
#         global_idx = n_v_final + idx
#         print(f"  Angular Pos ID {idx}: {a[global_idx]:.5f}")
#     print("="*50 + "\n")

#     # =========================================================================
#     # VISUAL RECONSTRUCTION CHARTING
#     # =========================================================================
#     # Figure 1: Measured vs Estimated Accelerations
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 1, 1)
#     plt.plot(timestamps, v_dot_measured, color='black', label='Measured $v_{dot}$')
#     plt.plot(timestamps, v_dot_estimated, color='crimson', linestyle='--', label='RLS Estimated $\hat{v}_{dot}$')
#     plt.ylabel('Linear Accel ($m/s^2$)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.subplot(2, 1, 2)
#     plt.plot(timestamps, omega_dot_measured, color='black', label='Measured $\omega_{dot}$')
#     plt.plot(timestamps, omega_dot_estimated, color='darkcyan', linestyle='--', label='RLS Estimated $\hat{\omega}_{dot}$')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Angular Accel ($rad/s^2$)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()

#     # Figure 2: Weights Evolution Trajectory
#     plt.figure(figsize=(12, 8))
#     plt.subplot(2, 1, 1)
#     plt.title('Weight Tuning Trajectory Map Over Time (Position Tracking ID)')
#     for idx in range(n_v_final):
#         plt.plot(timestamps, weight_history[:, idx], label=f'Pos ID {idx}')
#     plt.ylabel('Linear Weights Values')
#     plt.legend(loc='upper right', ncol=3)
#     plt.grid(True, alpha=0.3)

#     plt.subplot(2, 1, 2)
#     for idx in range(n_w_final):
#         global_idx = n_v_final + idx
#         plt.plot(timestamps, weight_history[:, global_idx], label=f'Pos ID {idx}')
#     plt.xlabel('Time (seconds)')
#     plt.ylabel('Angular Weights Values')
#     plt.legend(loc='upper right', ncol=3)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     plt.show()

#     return a, P


# if __name__ == "__main__":
#     BASE_DIR = Path(__file__).resolve().parent
#     folder_path = BASE_DIR / "wheelie_nn" / "dataset"
    
#     # Path to save/load learned weight matrices
#     weights_file_path = BASE_DIR / "wheelie_nn" / "rls_learned_weights.npz"

#     current_a = None
#     current_P = None

#     # Load existing weights from previous runs if the file exists
#     if weights_file_path.exists():
#         print(f"Found existing weights file at: {weights_file_path}")
#         with np.load(weights_file_path) as data:
#             current_a = data['a']
#             if 'P' in data:
#                 current_P = data['P']
#             print("Successfully loaded prior 'a' weight vector.")
#     else:
#         print("No prior weights file found. Starting fresh.")

#     # Process sequential datasets (rls_id_103.csv through rls_id_109.csv)
#     for i in range(1, 110):
#         target_file = folder_path / f"rls_id_{i:03d}.csv"

#         if target_file.exists():
#             # Pass weights in; covariance automatically resets inside run_rls_estimation
#             current_a, current_P = run_rls_estimation(target_file, a_init=current_a)
            
#             # Save continuous progress after every processed file
#             np.savez(weights_file_path, a=current_a, P=current_P)
#             print(f"--> Saved updated weights to {weights_file_path}")
            
#         else:
#             print(f"Error: Target data profile '{target_file}' could not be located.")










































import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

# Import the functional modules cleanly from your external rls.py script
from rls import nominal_rls_parameters, rls_update


def numerical_integration(timestamps, accelerations, initial_velocity=0.0, method='rk4'):
    """
    Integrates acceleration time series to reconstruct linear/angular velocity.
    Supports 'euler' and 'rk4' numerical schemes.
    """
    num_steps = len(timestamps)
    velocity = np.zeros(num_steps)
    velocity[0] = initial_velocity

    for i in range(num_steps - 1):
        dt = timestamps[i + 1] - timestamps[i]
        if dt <= 0:
            velocity[i + 1] = velocity[i]
            continue

        if method.lower() == 'euler':
            velocity[i + 1] = velocity[i] + accelerations[i] * dt

        elif method.lower() == 'rk4':
            # RK4 using linear interpolation for mid-step acceleration values
            a_t = accelerations[i]
            a_next = accelerations[i + 1]
            a_mid = 0.5 * (a_t + a_next)

            k1 = a_t
            k2 = a_mid
            k3 = a_mid
            k4 = a_next

            velocity[i + 1] = velocity[i] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError("Integration method must be either 'euler' or 'rk4'.")

    return velocity


def run_rls_estimation(csv_file_path, a_init=None, integration_method='rk4'):
    print(f"\nLoading data from: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    if df.empty:
        print("Error: DataFrame is empty.")
        return a_init, None

    df["u_eff"] = df["u"].shift(-2).fillna(0.0)
    num_rows = len(df)
    timestamps = df['elapsed_s'].values
    v_measured = df['v'].values
    omega_measured = df['omega'].values

    # =========================================================================
    # RE-DERIVE SYNCHRONIZED GROUND TRUTH ACCELERATIONS
    # =========================================================================
    # 1. Compute numerical gradients directly from velocity time series
    v_dot_raw = np.gradient(v_measured, timestamps)
    omega_dot_raw = np.gradient(omega_measured, timestamps)

    # 2. Zero-phase low-pass filtering to clean derivative noise without lag
    b, a_filter = butter(N=2, Wn=0.4, btype='low')  # Adjust Wn (0.1 - 0.3) if needed
    v_dot_measured = filtfilt(b, a_filter, v_dot_raw)
    omega_dot_measured = filtfilt(b, a_filter, omega_dot_raw)
    # =========================================================================
    
    # Warm start with covariance reset
    if a_init is None:
        print("Initializing weights from zero (first run).")
        a = nominal_rls_parameters()
    else:
        print("Warm starting with weights loaded from previous run / .npz file.")
        a = a_init.copy()

    # Reset covariance matrix for maximum adaptability per dataset
    P = np.eye(18) * 1000.0

    # v_dot_measured = df['v_dot'].values
    # omega_dot_measured = df['omega_dot'].values
    # v_measured = df['v'].values
    # omega_measured = df['omega'].values

    v_dot_estimated = np.zeros(num_rows)
    omega_dot_estimated = np.zeros(num_rows)
    timestamps = df['elapsed_s'].values
    weight_history = []

    print("Processing time series steps sequentially...")
    for i in range(num_rows):
        state = np.array([df['x'].iloc[i], df['v'].iloc[i], df['theta_rad'].iloc[i], df['omega'].iloc[i]])
        tau = float(df['u_eff'].iloc[i])
        
        a, P, info = rls_update(
            state, 
            tau, 
            v_dot_measured[i], 
            omega_dot_measured[i], 
            a, 
            P,
            forgetting_factor=0.95,  
            sigma_v_dot=2.0,         
            sigma_omega_dot=5.0      
        )
        
        v_dot_estimated[i] = info["v_dot_hat"]
        omega_dot_estimated[i] = info["omega_dot_hat"]
        
        n_active = info["n_v"] + info["n_w"]
        weight_history.append(a[0:n_active].copy())
        
    weight_history = np.array(weight_history)
    n_v_final = info["n_v"]
    n_w_final = info["n_w"]

    # =========================================================================
    # RECONSTRUCT VELOCITIES VIA NUMERICAL INTEGRATION
    # =========================================================================
    print(f"Integrating predicted accelerations using scheme: {integration_method.upper()}...")
    v_estimated = numerical_integration(
        timestamps, v_dot_estimated, initial_velocity=v_measured[0], method=integration_method
    )
    omega_estimated = numerical_integration(
        timestamps, omega_dot_estimated, initial_velocity=omega_measured[0], method=integration_method
    )

    print("\n" + "="*50)
    print("FINAL CONVERGED MODEL WEIGHT SUMMARY")
    print("="*50)
    for idx in range(n_v_final):
        print(f"  Linear Pos ID {idx}: {a[idx]:.5f}")
    for idx in range(n_w_final):
        global_idx = n_v_final + idx
        print(f"  Angular Pos ID {idx}: {a[global_idx]:.5f}")
    print("="*50 + "\n")

    # =========================================================================
    # VISUAL RECONSTRUCTION CHARTING
    # =========================================================================
    # Figure 1: 4-Panel Performance Dashboard (Accelerations & Integrated Velocities)
    # plt.figure(figsize=(14, 10))
    
    # # Linear Acceleration
    # plt.subplot(2, 2, 1)
    # plt.plot(timestamps, v_dot_measured, color='black', label='Measured $v_{dot}$')
    # plt.plot(timestamps, v_dot_estimated, color='crimson', linestyle='--', label='RLS Estimated $\hat{v}_{dot}$')
    # plt.ylabel('Linear Accel ($m/s^2$)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # # Integrated Linear Velocity
    # plt.subplot(2, 2, 2)
    # plt.plot(timestamps, v_measured, color='black', label='Measured $v$')
    # plt.plot(timestamps, v_estimated, color='crimson', linestyle='--', label=f'Integrated $\hat{{v}}$ ({integration_method.upper()})')
    # plt.ylabel('Linear Velocity ($m/s$)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # # Angular Acceleration
    # plt.subplot(2, 2, 3)
    # plt.plot(timestamps, omega_dot_measured, color='black', label='Measured $\omega_{dot}$')
    # plt.plot(timestamps, omega_dot_estimated, color='darkcyan', linestyle='--', label='RLS Estimated $\hat{\omega}_{dot}$')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Angular Accel ($rad/s^2$)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)

    # # Integrated Angular Velocity
    # plt.subplot(2, 2, 4)
    # plt.plot(timestamps, omega_measured, color='black', label='Measured $\omega$')
    # plt.plot(timestamps, omega_estimated, color='darkcyan', linestyle='--', label=f'Integrated $\hat{{\omega}}$ ({integration_method.upper()})')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Angular Velocity ($rad/s$)')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    
    # plt.tight_layout()

    # # Figure 2: Weights Evolution Trajectory
    # plt.figure(figsize=(12, 8))
    # plt.subplot(2, 1, 1)
    # plt.title('Weight Tuning Trajectory Map Over Time (Position Tracking ID)')
    # for idx in range(n_v_final):
    #     plt.plot(timestamps, weight_history[:, idx], label=f'Pos ID {idx}')
    # plt.ylabel('Linear Weights Values')
    # plt.legend(loc='upper right', ncol=3)
    # plt.grid(True, alpha=0.3)

    # plt.subplot(2, 1, 2)
    # for idx in range(n_w_final):
    #     global_idx = n_v_final + idx
    #     plt.plot(timestamps, weight_history[:, global_idx], label=f'Pos ID {idx}')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Angular Weights Values')
    # plt.legend(loc='upper right', ncol=3)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    
    plt.show()

    return a, P


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    folder_path = BASE_DIR / "wheelie_nn" / "dataset"
    
    weights_file_path = BASE_DIR / "wheelie_nn" / "rls_learned_weights.npz"

    current_a = None
    current_P = None

    if weights_file_path.exists():
        print(f"Found existing weights file at: {weights_file_path}")
        with np.load(weights_file_path) as data:
            current_a = data['a']
            if 'P' in data:
                current_P = data['P']
            print("Successfully loaded prior 'a' weight vector.")
    else:
        print("No prior weights file found. Starting fresh.")

    # Execute batch processing (Change 'method' to 'euler' or 'rk4' as desired)
    INTEGRATION_SCHEME = 'rk4'  # Options: 'rk4', 'euler'

    for i in range(1, 146):
        target_file = folder_path / f"rls_id_{i:03d}.csv"

        if target_file.exists():
            current_a, current_P = run_rls_estimation(
                target_file, 
                a_init=current_a, 
                integration_method=INTEGRATION_SCHEME
            )
            
            np.savez(weights_file_path, a=current_a, P=current_P)
            print(f"--> Saved updated weights to {weights_file_path}")
            
        else:
            print(f"Error: Target data profile '{target_file}' could not be located.")