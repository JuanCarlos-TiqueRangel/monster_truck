#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from pathlib import Path

from rls import rls_dynamics_torch, rk4_step_torch


class RLSExperimentValidator:
    """
    Validate the exact RLS model implemented in rls.py.

    The validator does NOT duplicate the RLS equations.

    It uses:
        rls_dynamics_torch()
        rk4_step_torch()

    directly from rls.py, so the model being validated is the same model
    that MPPI uses.
    """

    def __init__(self, csv_file, weights=None, weights_file=None):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file).copy()

        self._check_columns()

        if weights_file is not None:
            data = np.load(weights_file)
            self.a = np.asarray(data["learned_parameters"], dtype=float)

        elif weights is not None:
            self.a = np.asarray(weights, dtype=float)

        else:
            self.a = self._weights_from_csv()

        self.a_torch = torch.tensor(self.a, dtype=torch.float64)
        self.results = None

    def _check_columns(self):
        required = [
            "elapsed_s",
            "x",
            "v",
            "theta_rad",
            "omega",
            "u",
            "v_dot",
            "omega_dot",
        ]

        missing = [name for name in required if name not in self.df.columns]

        if missing:
            raise KeyError(f"Missing CSV columns: {missing}")

        self.df = self.df.dropna(subset=required).reset_index(drop=True)

        if len(self.df) < 2:
            raise ValueError("The experiment needs at least two valid samples.")

    def _weights_from_csv(self):
        weight_columns = [
            c for c in self.df.columns
            if c.startswith("a") and c[1:].isdigit()
        ]

        if not weight_columns:
            raise ValueError(
                "No RLS weights were passed and the CSV has no a0, a1, ... columns."
            )

        weight_columns = sorted(
            weight_columns,
            key=lambda name: int(name[1:]),
        )

        valid = self.df[weight_columns].dropna()

        if valid.empty:
            raise ValueError("No valid RLS weight vector was found in the CSV.")

        return valid.iloc[-1].to_numpy(dtype=float)

    @torch.no_grad()
    def acceleration_from_rls(self, x, v, theta, omega, tau):
        state = torch.tensor(
            [[x, v, theta, omega]],
            dtype=torch.float64,
        )

        tau_tensor = torch.tensor(
            [tau],
            dtype=torch.float64,
        )

        derivative = rls_dynamics_torch(
            state,
            tau_tensor,
            self.a_torch,
        )[0]

        v_dot = float(derivative[1])
        omega_dot = float(derivative[3])

        return v_dot, omega_dot

    @torch.no_grad()
    def run(self):
        df = self.df
        n = len(df)

        v_dot_map = np.zeros(n)
        omega_dot_map = np.zeros(n)

        # -------------------------------------------------------------
        # 1. Instantaneous acceleration mapping
        #    measured state + measured tau -> RLS acceleration
        # -------------------------------------------------------------
        for k in range(n):
            v_dot_map[k], omega_dot_map[k] = self.acceleration_from_rls(
                x=float(df.loc[k, "x"]),
                v=float(df.loc[k, "v"]),
                theta=float(df.loc[k, "theta_rad"]),
                omega=float(df.loc[k, "omega"]),
                tau=float(df.loc[k, "u"]),
            )

        # -------------------------------------------------------------
        # 2. Recursive forward rollout using the EXACT same RK4
        #    implementation from rls.py
        # -------------------------------------------------------------
        state = torch.tensor(
            [[
                float(df.loc[0, "x"]),
                float(df.loc[0, "v"]),
                float(df.loc[0, "theta_rad"]),
                float(df.loc[0, "omega"]),
            ]],
            dtype=torch.float64,
        )

        x_rollout = np.zeros(n)
        v_rollout = np.zeros(n)
        theta_rollout = np.zeros(n)
        omega_rollout = np.zeros(n)

        v_dot_rollout = np.zeros(n)
        omega_dot_rollout = np.zeros(n)

        x_rollout[0] = float(state[0, 0])
        v_rollout[0] = float(state[0, 1])
        theta_rollout[0] = float(state[0, 2])
        omega_rollout[0] = float(state[0, 3])

        for k in range(n):
            tau = torch.tensor(
                [float(df.loc[k, "u"])],
                dtype=torch.float64,
            )

            derivative = rls_dynamics_torch(
                state,
                tau,
                self.a_torch,
            )[0]

            v_dot_rollout[k] = float(derivative[1])
            omega_dot_rollout[k] = float(derivative[3])

            if k == n - 1:
                break

            dt = float(
                df.loc[k + 1, "elapsed_s"]
                - df.loc[k, "elapsed_s"]
            )

            if dt <= 0.0:
                raise ValueError(
                    f"Non-positive dt at sample {k}: {dt}"
                )

            state = rk4_step_torch(
                state,
                tau,
                dt,
                self.a_torch,
            )

            x_rollout[k + 1] = float(state[0, 0])
            v_rollout[k + 1] = float(state[0, 1])
            theta_rollout[k + 1] = float(state[0, 2])
            omega_rollout[k + 1] = float(state[0, 3])

        self.results = pd.DataFrame({
            "time": df["elapsed_s"].to_numpy(),
            "tau": df["u"].to_numpy(),

            "v_measured": df["v"].to_numpy(),
            "v_predicted": v_rollout,

            "omega_measured": df["omega"].to_numpy(),
            "omega_predicted": omega_rollout,

            "theta_measured": df["theta_rad"].to_numpy(),
            "theta_predicted": theta_rollout,

            "v_dot_measured": df["v_dot"].to_numpy(),
            "v_dot_predicted": v_dot_map,

            "omega_dot_measured": df["omega_dot"].to_numpy(),
            "omega_dot_predicted": omega_dot_map,

            "v_dot_rollout": v_dot_rollout,
            "omega_dot_rollout": omega_dot_rollout,
        })

        return self.results

    @staticmethod
    def _rmse(measured, predicted):
        measured = np.asarray(measured)
        predicted = np.asarray(predicted)

        return float(
            np.sqrt(
                np.mean(
                    (measured - predicted) ** 2
                )
            )
        )

    @staticmethod
    def _r2(measured, predicted):
        measured = np.asarray(measured)
        predicted = np.asarray(predicted)

        ss_res = np.sum( (measured - predicted) ** 2 )
        ss_tot = np.sum( (measured - np.mean(measured)) ** 2 )

        if ss_tot <= 1e-12:
            return np.nan

        return float(
            1.0 - ss_res / ss_tot
        )

    def metrics(self):
        if self.results is None:
            self.run()

        r = self.results

        return {
            "v_RMSE": self._rmse(
                r["v_measured"],
                r["v_predicted"],
            ),
            "v_R2": self._r2(
                r["v_measured"],
                r["v_predicted"],
            ),

            "omega_RMSE": self._rmse(
                r["omega_measured"],
                r["omega_predicted"],
            ),
            "omega_R2": self._r2(
                r["omega_measured"],
                r["omega_predicted"],
            ),

            "theta_RMSE_deg": self._rmse(
                np.rad2deg(r["theta_measured"]),
                np.rad2deg(r["theta_predicted"]),
            ),
            "theta_R2": self._r2(
                r["theta_measured"],
                r["theta_predicted"],
            ),

            "v_dot_RMSE": self._rmse(
                r["v_dot_measured"],
                r["v_dot_predicted"],
            ),
            "v_dot_R2": self._r2(
                r["v_dot_measured"],
                r["v_dot_predicted"],
            ),

            "omega_dot_RMSE": self._rmse(
                r["omega_dot_measured"],
                r["omega_dot_predicted"],
            ),
            "omega_dot_R2": self._r2(
                r["omega_dot_measured"],
                r["omega_dot_predicted"],
            ),
        }

    def print_metrics(self):
        m = self.metrics()

        print("\nRLS experiment validation")
        print("---------------------------------------------")

        print(
            f"v rollout:          "
            f"RMSE = {m['v_RMSE']:.4f}, "
            f"R² = {m['v_R2']:.4f}"
        )

        print(
            f"omega rollout:      "
            f"RMSE = {m['omega_RMSE']:.4f}, "
            f"R² = {m['omega_R2']:.4f}"
        )

        print(
            f"theta rollout:      "
            f"RMSE = {m['theta_RMSE_deg']:.4f} deg, "
            f"R² = {m['theta_R2']:.4f}"
        )

        print(
            f"v_dot mapping:      "
            f"RMSE = {m['v_dot_RMSE']:.4f}, "
            f"R² = {m['v_dot_R2']:.4f}"
        )

        print(
            f"omega_dot mapping:  "
            f"RMSE = {m['omega_dot_RMSE']:.4f}, "
            f"R² = {m['omega_dot_R2']:.4f}"
        )

        print("\nRLS weights")
        print("---------------------------------------------")

        for i, value in enumerate(self.a):
            print(
                f"a{i} = {value:+.8f}"
            )

    def plot(self, save_path=None, show=True):
        if self.results is None:
            self.run()

        r = self.results

        fig, ax = plt.subplots( 6, 1, figsize=(14, 19), sharex=True, )

        ax[0].plot( r["time"], r["tau"], linewidth=2, )
        ax[0].set_ylabel(r"$\tau$ [Nm]")
        ax[0].set_title("Experimental Input")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(
            r["time"],
            r["v_measured"],
            linewidth=2,
            label="Measured",
        )
        ax[1].plot(
            r["time"],
            r["v_predicted"],
            "--",
            linewidth=2,
            label="RLS rollout",
        )
        ax[1].set_ylabel(r"$v$ [m/s]")
        ax[1].set_title("Linear Velocity")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        ax[2].plot(
            r["time"],
            r["omega_measured"],
            linewidth=2,
            label="Measured",
        )
        ax[2].plot(
            r["time"],
            r["omega_predicted"],
            "--",
            linewidth=2,
            label="RLS rollout",
        )
        ax[2].set_ylabel(r"$\omega$ [rad/s]")
        ax[2].set_title("Angular Velocity")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()

        ax[3].plot(
            r["time"],
            np.rad2deg(r["theta_measured"]),
            linewidth=2,
            label="Measured",
        )
        ax[3].plot(
            r["time"],
            np.rad2deg(r["theta_predicted"]),
            "--",
            linewidth=2,
            label="Integrated RLS rollout",
        )
        ax[3].set_ylabel(r"$\theta$ [deg]")
        ax[3].set_title("Pitch")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend()

        ax[4].plot( r["time"], r["v_dot_measured"], linewidth=2, label="Measured", )
        ax[4].plot( r["time"], r["v_dot_predicted"], "--", linewidth=2, label="RLS", )
        ax[4].set_ylabel( r"$\dot v$ [m/s$^2$]" )
        ax[4].set_title( "Linear Acceleration" )
        ax[4].grid( True, alpha=0.3, )
        ax[4].legend()

        ax[5].plot( r["time"], r["omega_dot_measured"], linewidth=2, label="Measured", )
        ax[5].plot( r["time"], r["omega_dot_predicted"], "--", linewidth=2, label="RLS", )
        ax[5].set_ylabel( r"$\dot\omega$ [rad/s$^2$]" )
        ax[5].set_xlabel( "Time [s]" )
        ax[5].set_title( "Angular Acceleration" )
        ax[5].grid(True, alpha=0.3,)
        ax[5].legend()

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=250,
                bbox_inches="tight",
            )

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent

    FILE = ROOT / "data_rls_collection" / "rls_id_14.csv"
    validator = RLSExperimentValidator(FILE, weights_file="rls_trained_model.npz")


    results = validator.run()
    validator.print_metrics()
    validator.plot("rls_model_validation.png",)
    results.to_csv("rls_model_predictions.csv", index=False,)
