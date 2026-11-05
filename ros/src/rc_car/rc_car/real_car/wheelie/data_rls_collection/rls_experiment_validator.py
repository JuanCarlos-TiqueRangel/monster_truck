#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RLSExperimentValidator:
    """
    Validate the 7-parameter RLS wheelie model against an experiment.

    Model:
        v_dot = a0*tau + a1*v + a2*|v|v + a3*tau*cos(theta)
        omega_dot = a4*cos(theta) + a5*tau + a6*omega

    The class produces two different kinds of predictions:

    1) Acceleration mapping:
       Uses the MEASURED (v, theta, omega, tau) at every sample.
       This tests whether the learned RLS mapping itself is correct.

    2) Open-loop state rollout:
       Uses only the first measured state, then recursively propagates the
       RLS model using the RECORDED experimental tau sequence.
       This tests what MPPI actually cares about: model rollout quality.
    """

    def __init__(self, csv_file, weights=None, ground_contact=True):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file).copy()
        self.ground_contact = ground_contact

        self._check_columns()

        if weights is None:
            self.a = self._weights_from_csv()
        else:
            self.a = np.asarray(weights, dtype=float)

        if self.a.size != 7:
            raise ValueError(f"Expected 7 RLS parameters, received {self.a.size}")

        self.results = None

    def _check_columns(self):
        required = [
            "elapsed_s",
            "u",
            "v",
            "theta_rad",
            "omega",
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
        columns = [f"a{i}" for i in range(7)]

        if not all(name in self.df.columns for name in columns):
            raise ValueError(
                "No weights were passed and the CSV does not contain a0...a6."
            )

        valid = self.df[columns].dropna()

        if valid.empty:
            raise ValueError("The CSV does not contain a valid final RLS weight vector.")

        return valid.iloc[-1].to_numpy(dtype=float)

    def acceleration(self, v, theta, omega, tau):
        a = self.a

        v_dot = (
            a[0] * tau
            + a[1] * v
            + a[2] * abs(v) * v
            + a[3] * tau * np.cos(theta)
        )

        omega_dot = (
            a[4] * np.cos(theta)
            + a[5] * tau
            + a[6] * omega
        )

        if self.ground_contact and theta >= 0.0 and omega_dot > 0.0:
            omega_dot = 0.0

        return v_dot, omega_dot

    def _dynamics(self, state, tau):
        v, theta, omega = state
        v_dot, omega_dot = self.acceleration(v, theta, omega, tau)

        return np.array([
            v_dot,
            omega,
            omega_dot,
        ], dtype=float)

    def _rk4(self, state, tau, dt):
        k1 = self._dynamics(state, tau)
        k2 = self._dynamics(state + 0.5 * dt * k1, tau)
        k3 = self._dynamics(state + 0.5 * dt * k2, tau)
        k4 = self._dynamics(state + dt * k3, tau)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def run(self):
        df = self.df
        n = len(df)

        v_dot_map = np.zeros(n)
        omega_dot_map = np.zeros(n)

        for k in range(n):
            v_dot_map[k], omega_dot_map[k] = self.acceleration(
                v=float(df.loc[k, "v"]),
                theta=float(df.loc[k, "theta_rad"]),
                omega=float(df.loc[k, "omega"]),
                tau=float(df.loc[k, "u"]),
            )

        v_rollout = np.zeros(n)
        theta_rollout = np.zeros(n)
        omega_rollout = np.zeros(n)
        v_dot_rollout = np.zeros(n)
        omega_dot_rollout = np.zeros(n)

        v_rollout[0] = float(df.loc[0, "v"])
        theta_rollout[0] = float(df.loc[0, "theta_rad"])
        omega_rollout[0] = float(df.loc[0, "omega"])

        for k in range(n):
            v_dot_rollout[k], omega_dot_rollout[k] = self.acceleration(
                v=v_rollout[k],
                theta=theta_rollout[k],
                omega=omega_rollout[k],
                tau=float(df.loc[k, "u"]),
            )

            if k == n - 1:
                break

            dt = float(df.loc[k + 1, "elapsed_s"] - df.loc[k, "elapsed_s"])

            if dt <= 0.0:
                raise ValueError(f"Non-positive dt at sample {k}: {dt}")

            state = np.array([
                v_rollout[k],
                theta_rollout[k],
                omega_rollout[k],
            ])

            next_state = self._rk4(
                state=state,
                tau=float(df.loc[k, "u"]),
                dt=dt,
            )

            v_rollout[k + 1] = next_state[0]
            theta_rollout[k + 1] = next_state[1]
            omega_rollout[k + 1] = next_state[2]

        self.results = pd.DataFrame({
            "time": df["elapsed_s"].to_numpy(),
            "tau": df["u"].to_numpy(),

            "v_measured": df["v"].to_numpy(),
            "v_predicted": v_rollout,

            "omega_measured": df["omega"].to_numpy(),
            "omega_predicted": omega_rollout,

            "v_dot_measured": df["v_dot"].to_numpy(),
            "v_dot_predicted": v_dot_map,

            "omega_dot_measured": df["omega_dot"].to_numpy(),
            "omega_dot_predicted": omega_dot_map,

            "theta_measured": df["theta_rad"].to_numpy(),
            "theta_predicted": theta_rollout,

            "v_dot_rollout": v_dot_rollout,
            "omega_dot_rollout": omega_dot_rollout,
        })

        return self.results

    @staticmethod
    def _rmse(measured, predicted):
        measured = np.asarray(measured)
        predicted = np.asarray(predicted)
        return float(np.sqrt(np.mean((measured - predicted) ** 2)))

    @staticmethod
    def _r2(measured, predicted):
        measured = np.asarray(measured)
        predicted = np.asarray(predicted)

        ss_res = np.sum((measured - predicted) ** 2)
        ss_tot = np.sum((measured - np.mean(measured)) ** 2)

        if ss_tot <= 1e-12:
            return np.nan

        return float(1.0 - ss_res / ss_tot)

    def metrics(self):
        if self.results is None:
            self.run()

        r = self.results

        metrics = {
            "v_RMSE": self._rmse(r["v_measured"], r["v_predicted"]),
            "v_R2": self._r2(r["v_measured"], r["v_predicted"]),

            "omega_RMSE": self._rmse(
                r["omega_measured"],
                r["omega_predicted"],
            ),
            "omega_R2": self._r2(
                r["omega_measured"],
                r["omega_predicted"],
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

        return metrics

    def print_metrics(self):
        m = self.metrics()

        print("\nRLS experiment validation")
        print("---------------------------------------------")
        print(f"v rollout:          RMSE = {m['v_RMSE']:.4f}, R² = {m['v_R2']:.4f}")
        print(f"omega rollout:      RMSE = {m['omega_RMSE']:.4f}, R² = {m['omega_R2']:.4f}")
        print(f"v_dot mapping:      RMSE = {m['v_dot_RMSE']:.4f}, R² = {m['v_dot_R2']:.4f}")
        print(f"omega_dot mapping:  RMSE = {m['omega_dot_RMSE']:.4f}, R² = {m['omega_dot_R2']:.4f}")

        print("\nRLS weights")
        print("---------------------------------------------")
        for i, value in enumerate(self.a):
            print(f"a{i} = {value:+.8f}")

    def plot(self, save_path=None, show=True):
        if self.results is None:
            self.run()

        r = self.results

        fig, ax = plt.subplots(6, 1, figsize=(14, 19), sharex=True)

        ax[0].plot(r["time"], r["tau"], linewidth=2)
        ax[0].set_ylabel(r"$\tau$ [Nm]")
        ax[0].set_title("Experimental Input")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(r["time"], r["v_measured"], linewidth=2, label="Measured")
        ax[1].plot(r["time"], r["v_predicted"], "--", linewidth=2, label="RLS rollout")
        ax[1].set_ylabel(r"$v$ [m/s]")
        ax[1].set_title("Linear Velocity")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()

        ax[2].plot(r["time"], r["omega_measured"], linewidth=2, label="Measured")
        ax[2].plot(r["time"], r["omega_predicted"], "--", linewidth=2, label="RLS rollout")
        ax[2].set_ylabel(r"$\omega$ [rad/s]")
        ax[2].set_title("Angular Velocity")
        ax[2].grid(True, alpha=0.3)
        ax[2].legend()

        pitch_measured_deg = np.rad2deg(r["theta_measured"])
        pitch_predicted_deg = np.rad2deg(r["theta_predicted"])

        ax[3].plot(r["time"], pitch_measured_deg, linewidth=2, label="Measured")
        ax[3].plot(r["time"], pitch_predicted_deg, "--", linewidth=2, label="Integrated RLS rollout")
        ax[3].set_ylabel(r"$\theta$ [deg]")
        ax[3].set_title("Pitch")
        ax[3].grid(True, alpha=0.3)
        ax[3].legend()

        ax[4].plot(r["time"], r["v_dot_measured"], linewidth=2, label="Measured")
        ax[4].plot(r["time"], r["v_dot_predicted"], "--", linewidth=2, label="RLS")
        ax[4].set_ylabel(r"$\dot v$ [m/s$^2$]")
        ax[4].set_title("Linear Acceleration")
        ax[4].grid(True, alpha=0.3)
        ax[4].legend()

        ax[5].plot(r["time"], r["omega_dot_measured"], linewidth=2, label="Measured")
        ax[5].plot(r["time"], r["omega_dot_predicted"], "--", linewidth=2, label="RLS")
        ax[5].set_ylabel(r"$\dot\omega$ [rad/s$^2$]")
        ax[5].set_xlabel("Time [s]")
        ax[5].set_title("Angular Acceleration")
        ax[5].grid(True, alpha=0.3)
        ax[5].legend()

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=250, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


if __name__ == "__main__":
    FILE = "rls_id_10.csv"

    validator = RLSExperimentValidator(FILE)

    results = validator.run()
    validator.print_metrics()
    validator.plot("rls_model_validation.png")

    results.to_csv("rls_model_predictions.csv", index=False)
