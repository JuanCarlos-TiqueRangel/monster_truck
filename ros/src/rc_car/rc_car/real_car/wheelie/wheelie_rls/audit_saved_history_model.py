#!/usr/bin/env python3
"""Offline checks for the uploaded history-model MPPI. No ROS, commands, or refitting.

Place beside rls_history_mppi.py, mppi_history.py, and params_mppi.py.
  python3 audit_saved_history_model.py rls_history_model.npz
  python3 audit_saved_history_model.py rls_history_model.npz --device cuda
  python3 audit_saved_history_model.py rls_history_model.npz --run-mppi

All rollout tests start from s=[0,0,0,0], with four all-zero past rows,
nominally spaced. These are model probes, NOT commands to execute on a car.
"""

import argparse
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--params", type=Path, default=Path("params_mppi.py"))
    parser.add_argument("--code-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-mppi", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.model.is_file() or not args.params.is_file():
        parser.error("Both model and parameter paths must identify existing files.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this environment; use --device cpu.")
    torch.set_num_threads(1)
    for folder in (args.code_dir / "rls", args.code_dir / "mppi", args.code_dir):
        sys.path.insert(0, str(folder.resolve()))
    spec = importlib.util.spec_from_file_location("params_mppi", args.params.resolve())
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load parameters: {args.params}")
    params_module = importlib.util.module_from_spec(spec)
    sys.modules["params_mppi"] = params_module
    spec.loader.exec_module(params_module)
    from rls_history_mppi import RLSHistoryDynamics
    from mppi_history import MPPITorch

    p, cfg = params_module.WheelieParams(), params_module.MPPIConfig()
    print("OFFLINE MODEL AUDIT: no ROS connection, no published commands, no model changes.")
    print(f"Model SHA256: {hashlib.sha256(args.model.read_bytes()).hexdigest()}")
    print(f"Model: {args.model.resolve()}")
    print(f"Params: {args.params.resolve()}")
    print(f"Inference module: {sys.modules['rls_history_mppi'].__file__}")
    print(f"MPPI module: {sys.modules['mppi_history'].__file__}")
    print(f"NumPy={np.__version__}, Torch={torch.__version__}, device={args.device}")
    print(f"N={cfg.N}, dt={cfg.dt}, K={cfg.K}, sigma={cfg.SIGMA}, lambda={cfg.LAM}, bounds=[{p.u_min},{p.u_max}]")

    with np.load(args.model, allow_pickle=False) as z:
        a, nv = z["learned_parameters"], int(z["n_v_features"])
        weights = a[nv:nv + 20] / z["w_scale"]
        bias = float(a[nv + 20] - z["w_mean"] @ weights)
        raw_centers = z["kernel_centers"] * z["v_scale"] + z["v_mean"]
        beta = z["kernel_normalization"].T @ a[:nv - 1]
        print(f"Saved kernel-landmark history input range: [{raw_centers[:, :20:4].min():.8f}, {raw_centers[:, :20:4].max():.8f}]")
        print("This is a range of saved landmarks, not proof of the complete dataset range.")
        print(f"Angular raw-history bias={bias:+.9f}; current-input coefficient={weights[0]:+.9f}")
        print("Angular raw-history weights (rows current through lag4; columns u, v, theta, omega):")
        print(np.array2string(weights.reshape(5, 4), precision=8))
        print(f"Largest |effective kernel coefficient|: {np.max(np.abs(beta)):.9g}")
        # Direct original training transformation, with no coefficient folding.
        normalized = -z["v_mean"] / z["v_scale"]
        k = np.exp(-float(z["gamma"]) * np.square(z["kernel_centers"] - normalized).sum(axis=1))
        reference_v = float((k @ z["kernel_normalization"].T) @ a[:nv - 1] + a[nv - 1])
        print(f"Direct NumPy float64 training-map v_dot at all-zero history: {reference_v:+.9f}")

    print("\nIdentical zero histories: precision and batch-size check [v_dot, omega_dot].")
    for dtype in (torch.float64, torch.float32):
        model = RLSHistoryDynamics(args.model, device=args.device, dtype=dtype)
        for batch in (1, 64):
            y = model.predict(torch.zeros(batch, 5, 4, device=args.device, dtype=dtype)).cpu().numpy()
            print(f"{str(dtype):14s} batch={batch:3d}: first={y[0]}, min={y.min(axis=0)}, max={y.max(axis=0)}")

    controller = MPPITorch(p, cfg, args.model, device=args.device, dtype=torch.float64, integrator="euler", live_plot=False)
    dev, dtype = controller.device, controller.dtype
    ref = torch.tensor([0.0, 0.0, math.radians(-65.0), 0.0], device=dev, dtype=dtype)
    commands = torch.tensor([-0.3, 0.0, 0.3], device=dev, dtype=dtype)
    state = torch.zeros(3, 4, device=dev, dtype=dtype)
    past = torch.zeros(3, 4, 4, device=dev, dtype=dtype)
    ages = torch.arange(1, 5, device=dev, dtype=dtype).repeat(3, 1) * cfg.dt
    previous = torch.zeros(3, device=dev, dtype=dtype)
    cost = torch.zeros(3, device=dev, dtype=dtype)
    for _ in range(cfg.N):
        cost += controller._stage_cost(state, commands, previous, ref)
        state, past, ages = controller._step(state, commands, cfg.dt, past, ages)
        previous = commands
    cost += cfg.q_terminal_theta * (state[:, 2] - ref[2]).square() + cfg.q_terminal_omega * (state[:, 3] - ref[3]).square()
    print("\nCPU/GPU FLOAT64 constant-input probes using the supplied rollout and objective.")
    print("Each probe starts flat and stationary with four zero-input, zero-state past rows.")
    print("Probes are hypothetical, including negative input even if your current bounds exclude it.")
    print("       u    theta_N_deg      omega_N          v_N       total_cost")
    for u, s, j in zip(commands.cpu().numpy(), state.cpu().numpy(), cost.cpu().numpy()):
        print(f"{u:+8.3f} {np.degrees(s[2]):+14.6f} {s[3]:+12.6f} {s[1]:+12.6f} {j:16.6f}")

    if args.run_mppi:
        torch.manual_seed(args.seed)
        controller.last_solution = None
        s0 = torch.zeros(4, device=dev, dtype=dtype)
        past = torch.zeros(4, 4, device=dev, dtype=dtype)
        ages = torch.arange(1, 5, device=dev, dtype=dtype) * cfg.dt
        u, info = controller.solve(s0, ref, 0.0, past, ages)
        print(f"\nOne float64 MPPI solve, seed={args.seed}, first command={u:+.9f}: {info}")
        if info.get("success", False):
            state, history, lag_times = s0[None, :], past[None, :, :], ages[None, :]
            for command in controller.last_solution:
                state, history, lag_times = controller._step(state, command.reshape(1), cfg.dt, history, lag_times)
            print(f"Averaged-plan final pitch={torch.rad2deg(state[0, 2]).item():+.6f} deg")
        print("Random samples and floating-point results can differ by device; this is not a reproduction of a measured run.")


if __name__ == "__main__":
    main()
