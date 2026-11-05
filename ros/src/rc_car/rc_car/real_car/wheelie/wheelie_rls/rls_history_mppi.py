"""Frozen Torch inference for the history_nystroem_rls model in train_rls.py.

This is an inference adapter, not an MPPI controller and not a trained model.
It reproduces the active training feature map; it does NOT use the unrelated
10-parameter rls_dynamics_torch function.

Conventions
-----------
Physical state: [..., 4] = [x, v, theta_rad, omega].
History row:   [..., 4] = [u, v, theta_rad, omega].
Full history:  [..., 5, 4], newest first, including the current candidate u.
Past history:  [..., 4, 4], newest first, EXCLUDING the current state/action.
Lag times:     [..., 4] = current_time - each past row's timestamp, in seconds.
Use the same u units, state preprocessing and timestamp alignment as training.
No extra four-sample input shift is applied: the training feature override uses u.

predict() reproduces the acceleration regression without contact projection.
step() adds a proposed forward-Euler rollout and advances each rollout's history.
The optional pitch-floor projection is NOT part of the learned regression.
No RLS updates, covariance updates, or normalization refitting occur here.

At each real control tick, rebuild the initial state/history from measurements
and actually applied actions. Never carry a sampled trajectory's history into
the real measurement buffer. Evaluate free-running rollouts before deployment.
"""

from pathlib import Path
import numpy as np
import torch


class RLSHistoryDynamics:
    def __init__(self, model_path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use torch.float32 or torch.float64; validate float32 against float64 before deployment.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this PyTorch environment.")
        with np.load(model_path, allow_pickle=False) as saved:
            metadata = {"model_type": "history_nystroem_rls", "format_version": 1, "history_length": 5,
                        "history_order": "newest_first", "history_padding": "repeat_oldest"}
            for key, expected in metadata.items():
                if key not in saved.files or saved[key].item() != expected:
                    raise ValueError(f"Expected {key}={expected!r} in {model_path}.")
            if saved["feature_columns"].tolist() != ["u", "v", "theta_rad", "omega"]:
                raise ValueError("Unexpected feature_columns; this adapter expects [u, v, theta_rad, omega].")
            keys = ("learned_parameters", "v_mean", "v_scale", "w_mean", "w_scale", "kernel_centers", "kernel_normalization")
            arrays = {key: np.asarray(saved[key], dtype=np.float64).copy() for key in keys}
            self.gamma = float(saved["gamma"].item())
            self.dt_floor = float(saved["history_dt_floor"].item())
            self.nominal_dt = float(saved["nominal_sample_period"].item())
            nv, nw = int(saved["n_v_features"].item()), int(saved["n_w_features"].item())

        centers = arrays["kernel_centers"]
        if centers.ndim != 2 or centers.shape[1] != 32 or centers.shape[0] < 1:
            raise ValueError("kernel_centers must have shape (n_components, 32).")
        m = centers.shape[0]
        shapes = {"learned_parameters": (m + 22,), "v_mean": (32,), "v_scale": (32,),
                  "w_mean": (20,), "w_scale": (20,), "kernel_normalization": (m, m)}
        if nv != m + 1 or nw != 21:
            raise ValueError("Saved feature counts do not match kernel_centers and the five-row angular history.")
        for key, shape in shapes.items():
            if arrays[key].shape != shape:
                raise ValueError(f"{key} must have shape {shape}, got {arrays[key].shape}.")
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise ValueError("Saved model contains NaN or infinity.")
        if np.any(arrays["v_scale"] <= 0) or np.any(arrays["w_scale"] <= 0):
            raise ValueError("Saved StandardScaler scales must be positive.")
        if not np.isfinite([self.gamma, self.dt_floor, self.nominal_dt]).all() or min(self.gamma, self.dt_floor, self.nominal_dt) <= 0:
            raise ValueError("gamma, history_dt_floor, and nominal_sample_period must be finite and positive.")

        a = arrays["learned_parameters"]
        # Algebraically identical to (kernel @ normalization.T) @ a[:m].
        # Precompute in float64 before converting to the inference dtype.
        v_beta = arrays["kernel_normalization"].T @ a[:m]
        # Fold the angular StandardScaler into the linear weights and bias.
        w_weight = a[nv:nv + 20] / arrays["w_scale"]
        w_bias = a[nv + 20] - arrays["w_mean"] @ w_weight
        tensors = {"centers": centers, "v_mean": arrays["v_mean"], "v_scale": arrays["v_scale"],
                   "v_beta": v_beta, "v_bias": a[m], "w_weight": w_weight, "w_bias": w_bias}
        for key, value in tensors.items():
            setattr(self, key, torch.as_tensor(value, device=self.device, dtype=dtype))
        self.center_norm = self.centers.square().sum(dim=-1)
        self.default_lags = torch.arange(1, 5, device=self.device, dtype=dtype) * self.nominal_dt

    @torch.no_grad()
    def predict(self, history: torch.Tensor, lag_times: torch.Tensor | None = None) -> torch.Tensor:
        """Return [..., 2] = [v_dot, omega_dot] for [..., 5, 4] histories.

        Pass actual lag durations for recorded data or irregular measurement
        spacing. Omitting them assumes nominal, uniformly spaced samples.
        Tensors must already use this object's device and dtype. Supply only
        finite states and positive lag durations. This function never refits.
        """
        if history.shape[-2:] != (5, 4):
            raise ValueError("history must have shape (..., 5, 4), newest first.")
        if history.device != self.device or history.dtype != self.dtype:
            # A generic 'cuda' device resolves to its actual index on allocation.
            if history.device != self.centers.device or history.dtype != self.dtype:
                raise ValueError("history must use the same device and dtype as the model.")
        lags = self.default_lags if lag_times is None else lag_times
        if lags.shape[-1] != 4:
            raise ValueError("lag_times must end with four durations, one per past row.")
        if lags.device != history.device or lags.dtype != history.dtype:
            raise ValueError("lag_times must use the same device and dtype as history.")

        xw = history.flatten(start_dim=-2)
        rates = (history[..., :1, 1:] - history[..., 1:, 1:]) / lags.clamp_min(self.dt_floor).unsqueeze(-1)
        xv = torch.cat((xw, rates.flatten(start_dim=-2)), dim=-1)
        xv = (xv - self.v_mean) / self.v_scale
        distance2 = (xv.square().sum(dim=-1, keepdim=True) + self.center_norm - 2.0 * (xv @ self.centers.T)).clamp_min(0.0)
        v_dot = torch.exp(-self.gamma * distance2) @ self.v_beta + self.v_bias
        omega_dot = xw @ self.w_weight + self.w_bias
        return torch.stack((v_dot, omega_dot), dim=-1)

    @torch.no_grad()
    def step(self, state: torch.Tensor, u: torch.Tensor, past: torch.Tensor, lag_times: torch.Tensor,
             dt: float | None = None, project_ground: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One proposed Euler rollout step, with independent histories per sample.

        state: [..., 4]; u: [...] or [..., 1]; past: [..., 4, 4].
        lag_times: [..., 4], elapsed seconds from state time to each past row.
        Returns (next_state, next_past, next_lag_times), without mutating inputs.
        next_past begins with [u, v, theta, omega] at the OLD current state.
        At the next horizon step the next state and next candidate u are prepended.

        project_ground=True assumes wheelie pitch <= 0. It blocks positive angular
        acceleration at theta >= 0 and projects a landing onto theta=0, omega=0.
        It is inappropriate for unrestricted flips; it is not a collision model
        or a substitute for controller constraints. No speed/rate clipping is added.
        """
        dt = self.nominal_dt if dt is None else float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if state.shape[-1] != 4 or past.shape != state.shape[:-1] + (4, 4):
            raise ValueError("Expected state (..., 4) and matching past (..., 4, 4).")
        if u.shape == state.shape[:-1] + (1,):
            u = u.squeeze(-1)
        if u.shape != state.shape[:-1]:
            raise ValueError("u must contain one scalar input per state.")
        current = torch.cat((u.unsqueeze(-1), state[..., 1:]), dim=-1)
        history = torch.cat((current.unsqueeze(-2), past), dim=-2)
        acceleration = self.predict(history, lag_times)
        v_dot, omega_dot = acceleration.unbind(dim=-1)
        if project_ground:
            blocked = (state[..., 2] >= 0.0) & (omega_dot > 0.0)
            omega_dot = torch.where(blocked, torch.zeros_like(omega_dot), omega_dot)
        derivative = torch.stack((state[..., 1], v_dot, state[..., 3], omega_dot), dim=-1)
        next_state = state + dt * derivative
        if project_ground:
            landing = (next_state[..., 2] >= 0.0) & (next_state[..., 3] > 0.0)
            omega_next = torch.where(landing, torch.zeros_like(next_state[..., 3]), next_state[..., 3])
            next_state = torch.stack((next_state[..., 0], next_state[..., 1], next_state[..., 2].clamp(max=0.0), omega_next), dim=-1)
        next_past = history[..., :4, :]
        next_lags = torch.cat((torch.zeros_like(lag_times[..., :1]), lag_times[..., :3]), dim=-1) + dt
        return next_state, next_past, next_lags
