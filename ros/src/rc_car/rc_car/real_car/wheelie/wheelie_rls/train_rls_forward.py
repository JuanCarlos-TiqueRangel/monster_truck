#!/usr/bin/env python3
"""Train the existing history RLS with forward-interval velocity targets.

No ROS connection. CSV files and the old model are not modified.
Assumption: u[k] is the command applied over [t[k], t[k+1]).
The input history and saved feature-map format are unchanged.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from rls import rls_update, history_features


def forward_targets(df: pd.DataFrame, nominal_dt: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Return next-interval [dv/dt, domega/dt] and a complete-history mask.

    Construct features on the ORIGINAL rows; use the mask only to select updates.
    A selected row has four genuine past rows and five admissible intervals:
    four past intervals and the interval to its next target measurement.
    """
    required = ["elapsed_s", "u", "x", "v", "theta_rad", "omega"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if len(df) < 2 or not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("Need at least two rows with finite timestamps, controls, and states.")
    dt = np.diff(df["elapsed_s"].to_numpy(dtype=float))
    if np.any(dt <= 0.0):
        raise ValueError("Timestamps must be strictly increasing within each CSV; do not concatenate episodes.")
    targets = np.full((len(df), 2), np.nan, dtype=np.float64)
    targets[:-1] = np.diff(df[["v", "omega"]].to_numpy(dtype=float), axis=0) / dt[:, None]
    good_dt = np.r_[(dt >= 0.5 * nominal_dt) & (dt <= 1.5 * nominal_dt), False]
    idx = np.arange(len(df))[:, None] - np.arange(5)
    valid = (idx >= 0).all(axis=1) & good_dt[np.maximum(idx, 0)].all(axis=1)
    valid &= np.isfinite(targets).all(axis=1)
    return targets, valid


def main():
    # ======================================================================
    # 1. DIRECTORY CONFIGURATION AND HISTORY FEATURES
    # ======================================================================
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root.parent / "wheelie_nn" / "dataset")
    parser.add_argument("--output", type=Path, default=root / "rls_history_forward_model.npz")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing the specified output file.")
    args = parser.parse_args()
    dataset_folder, model_path = args.dataset.resolve(), args.output.resolve()
    if not dataset_folder.is_dir():
        parser.error(f"Dataset directory does not exist: {dataset_folder}")
    if model_path.suffix != ".npz" or not model_path.parent.is_dir():
        parser.error("Output must be an .npz path in an existing directory.")
    if model_path.exists() and not args.overwrite:
        parser.error(f"Output already exists: {model_path}. Choose --output or explicitly use --overwrite.")

    test_filename = "rls_id_015.csv"
    all_files = sorted(dataset_folder.glob("*.csv"))
    train_files = [f for f in all_files if f.name not in {test_filename, 
                                                          "rls_id_011.csv", 
                                                          "rls_id_003.csv",
                                                          "rls_id_062.csv",
                                                          "rls_id_074.csv",
                                                          "rls_id_098.csv",
                                                          "rls_id_006.csv",
                                                          "rls_id_004.csv",
                                                          "rls_id_012.csv",
                                                          "rls_id_038.csv",
                                                          "rls_id_123.csv",
                                                          "rls_id_135.csv",
                                                          "rls_id_070.csv",
                                                          "rls_id_018.csv",
                                                          "rls_id_049.csv",
                                                          "rls_id_009.csv",
                                                          "rls_id_050.csv",
                                                          "rls_id_079.csv",
                                                          "rls_id_025.csv",
                                                          "rls_id_128.csv",
                                                          "rls_id_034.csv",
                                                          "rls_id_060.csv",
                                                          "rls_id_083.csv"}]
    validation_files = [dataset_folder / test_filename]
    if not train_files or not validation_files[0].is_file():
        parser.error("Need training CSVs and the held-out rls_id_015.csv; rls_id_011.csv remains excluded.")

    cols = ["u", "v", "theta_rad", "omega"]
    dfs = [pd.read_csv(f) for f in train_files]
    targets_and_masks = []
    for file_path, df in zip(train_files, dfs):
        try:
            targets_and_masks.append(forward_targets(df))
        except ValueError as exc:
            raise ValueError(f"{file_path.name}: {exc}") from exc
    # Never drop invalid rows before history_features: this would join separated samples.
    blocks = [history_features(df[cols], df.elapsed_s) for df in dfs]
    Xv = np.vstack([block[0][mask] for block, (_, mask) in zip(blocks, targets_and_masks)])
    Xw = np.vstack([block[1][mask] for block, (_, mask) in zip(blocks, targets_and_masks)])
    if len(Xv) < 600:
        raise ValueError(f"Only {len(Xv)} valid training transitions; need at least 600 for the unchanged 600-component basis.")

    sv = StandardScaler().fit(Xv)
    sw = StandardScaler().fit(Xw)
    basis = Nystroem(gamma=0.02, n_components=600, random_state=42).fit(sv.transform(Xv))
    a_trained = np.zeros(622, dtype=np.float64)
    P_trained = np.diag(np.r_[np.full(600, 4.0 / 0.03), 4e6, np.full(20, 25.0), 25e6])
    forgetting_factor = 1.0

    # ======================================================================
    # PHASE 2: SEQUENTIAL TRAINING
    # ======================================================================
    print("Training forward-interval targets; assume u[k] is applied over [t[k], t[k+1]).")
    training_counts = []
    for file_path, df, block, (targets, valid) in zip(train_files, dfs, blocks, targets_and_masks):
        xv, xw = block
        FV = np.column_stack((basis.transform(sv.transform(xv)), np.ones(len(df))))
        FW = np.column_stack((sw.transform(xw), np.ones(len(df))))
        states = df[["x", "v", "theta_rad", "omega"]].to_numpy(dtype=float)
        controls = df["u"].to_numpy(dtype=float)
        used = 0
        for i in np.flatnonzero(valid):
            a_trained, P_trained, info = rls_update(
                state=states[i], tau=float(controls[i]), vx_dot=float(targets[i, 0]), omega_pitch_dot=float(targets[i, 1]),
                a=a_trained, P=P_trained, forgetting_factor=forgetting_factor,
                sigma_v_dot=2.0, sigma_omega_dot=5.0, update_angular=True, features=(FV[i], FW[i]),
            )
            used += int(not info["skipped"])
        training_counts.append(used)
        print(f"{file_path.name}: {used}/{valid.sum()} eligible updates ({len(df)} rows)")
    if sum(training_counts) == 0 or not np.isfinite(a_trained).all() or not np.isfinite(P_trained).all():
        raise RuntimeError("No successful finite RLS model was produced; the output was not saved.")

    # ======================================================================
    # PHASE 3: HELD-OUT VALIDATION, WEIGHTS FROZEN
    # ======================================================================
    time_log, v_dot_actual, v_dot_pred, omega_dot_actual, omega_dot_pred = [], [], [], [], []
    for file_path in validation_files:
        df = pd.read_csv(file_path)
        targets, valid = forward_targets(df)
        if not valid.any():
            raise ValueError(f"No complete valid history/target windows in {file_path}.")
        xv, xw = history_features(df[cols], df.elapsed_s)
        FV = np.column_stack((basis.transform(sv.transform(xv)), np.ones(len(df))))
        FW = np.column_stack((sw.transform(xw), np.ones(len(df))))
        predictions = np.column_stack((FV @ a_trained[:601], FW @ a_trained[601:]))
        rmse = np.sqrt(np.mean((predictions[valid] - targets[valid])**2, axis=0))
        print(f"{file_path.name}: measured-history forward-target RMSE [m/s^2, rad/s^2] = {rmse}")
        # NaNs prevent plot lines from connecting across inadmissible windows.
        targets[~valid] = np.nan
        predictions[~valid] = np.nan
        time_log.extend(df.elapsed_s.to_numpy(dtype=float))
        v_dot_actual.extend(targets[:, 0])
        v_dot_pred.extend(predictions[:, 0])
        omega_dot_actual.extend(targets[:, 1])
        omega_dot_pred.extend(predictions[:, 1])

    # ======================================================================
    # PHASE 4: SAVE A SEPARATE MODEL, SAME INFERENCE FORMAT
    # ======================================================================
    # Extra metadata explains the NEW targets; no adapter format change is needed.
    with model_path.open("wb" if args.overwrite else "xb") as output:
        np.savez_compressed(
            output, model_type="history_nystroem_rls", format_version=1,
            learned_parameters=a_trained, covariance=P_trained,
            v_mean=sv.mean_, v_scale=sv.scale_, w_mean=sw.mean_, w_scale=sw.scale_,
            kernel_centers=basis.components_, kernel_normalization=basis.normalization_, gamma=basis.gamma,
            feature_columns=np.asarray(cols), history_length=5, history_order="newest_first", history_padding="repeat_oldest",
            history_dt_floor=0.04, nominal_sample_period=0.04, n_v_features=601, n_w_features=21,
            forgetting_factor=forgetting_factor, sigma_v_dot=2.0, sigma_omega_dot=5.0,
            training_files=np.asarray([str(f) for f in train_files]), validation_file=test_filename,
            target_type="forward_velocity_difference", target_alignment="k_to_k_plus_1",
            input_alignment="u[k] applied over [t[k], t[k+1])", minimum_training_row=4,
            min_sample_interval=0.02, max_sample_interval=0.06, use_sequence_index_mask=False,
            target_amplitude_filter="none", training_update_counts=np.asarray(training_counts, dtype=np.int64),
        )
    print(f"Saved: {model_path}")
    print("Next: run the saved-model audit and held-out free-running pitch validation before enabling drive.")

    # ======================================================================
    # PHASE 5: FORWARD-TARGET PLOTS (NOT THE OLD CSV ACCELERATION COLUMNS)
    # ======================================================================
    if not args.no_plot:
        import matplotlib.pyplot as plt
        for actual, predicted, title, units in (
            (v_dot_actual, v_dot_pred, "Linear acceleration over the next interval", "m/s^2"),
            (omega_dot_actual, omega_dot_pred, "Angular acceleration over the next interval", "rad/s^2"),
        ):
            plt.figure(figsize=(12, 5))
            plt.plot(time_log, actual, label="Forward-difference target")
            plt.plot(time_log, predicted, "--", label="Frozen RLS prediction")
            plt.title(f"{test_filename}: {title}")
            plt.xlabel("Elapsed time [s]")
            plt.ylabel(f"Acceleration [{units}]")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
