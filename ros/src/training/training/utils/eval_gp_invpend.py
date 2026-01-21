# #!/usr/bin/env python3
# import math
# import numpy as np
# import torch

# from gp_dynamics import GPManager  # your class

# # =========================
# # EDIT THESE PATHS
# # =========================
# NPZ_PATH = "data/mujoco_random_run.npz"
# DT_OVERRIDE = None  # set e.g. 0.1 if you want to force dt, else use npz["dt"]

# GP_FLIP_PATH = "models/gp_dynamics_0.pt"  # predicts d(flip)/dt
# GP_RATE_PATH = "models/gp_dynamics_1.pt"  # predicts d(rate)/dt

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # target angle (your “90 degrees” goal)
# THETA_TARGET = 1.5
# NEAR_BAND = 0.3

# # rollout eval
# H = 20
# N_ROLLOUTS = 25

# SEED = 0
# np.random.seed(SEED)

# def angdiff(a, b):
#     return (a - b + math.pi) % (2 * math.pi) - math.pi

# def load_transitions(npz_path):
#     D = np.load(npz_path)

#     # required signals
#     t    = D["t"].astype(np.float32)
#     flip = D["flip"].astype(np.float32)
#     rate = D["rate"].astype(np.float32)
#     u    = D["u"].astype(np.float32)

#     dt = float(D["dt"]) if "dt" in D.files else None
#     if DT_OVERRIDE is not None:
#         dt = float(DT_OVERRIDE)
#     if dt is None:
#         # infer from time stamps
#         dt = float(np.median(np.diff(t)))

#     # build (k -> k+1) transitions, skipping time breaks
#     idx = []
#     for k in range(len(t) - 1):
#         if abs((t[k+1] - t[k]) - dt) < 0.5 * dt:  # keep “normal” steps
#             idx.append(k)
#     idx = np.asarray(idx, dtype=np.int64)

#     flip_k = flip[idx]
#     rate_k = rate[idx]
#     u_k    = u[idx]

#     flip_kp1 = flip[idx + 1]
#     rate_kp1 = rate[idx + 1]

#     X = np.stack([flip_k, rate_k, u_k], axis=1).astype(np.float32)   # (N,3)
#     dflip_dt = (flip_kp1 - flip_k) / dt
#     drate_dt = (rate_kp1 - rate_k) / dt
#     Y = np.stack([dflip_dt, drate_dt], axis=1).astype(np.float32)    # (N,2)

#     x_next = np.stack([flip_kp1, rate_kp1], axis=1).astype(np.float32)
#     x_now  = np.stack([flip_k, rate_k], axis=1).astype(np.float32)

#     return dt, X, Y, x_now, x_next

# @torch.no_grad()
# def predict_derivs(gp_flip, gp_rate, X):
#     mean0, var0 = gp_flip.predict_torch(X)
#     mean1, var1 = gp_rate.predict_torch(X)
#     mean = torch.stack([mean0, mean1], dim=-1)
#     var  = torch.stack([var0, var1], dim=-1)
#     return mean, var

# def rmse(a, b):
#     return float(np.sqrt(np.mean((a-b)**2)))

# def mae(a, b):
#     return float(np.mean(np.abs(a-b)))

# def main():
#     dt, X, Y, x_now, x_next = load_transitions(NPZ_PATH)
#     print(f"Loaded transitions: N={len(X)}, dt={dt:.4f}")
#     print("X shape:", X.shape, "Y shape:", Y.shape, "x_next shape:", x_next.shape)

#     # split train/test (just for evaluation)
#     N = len(X)
#     perm = np.random.permutation(N)
#     n_test = max(200, int(0.25 * N))
#     test_idx = perm[:n_test]

#     Xte = X[test_idx]
#     Yte = Y[test_idx]
#     x0  = x_now[test_idx]
#     x1  = x_next[test_idx]

#     # load models
#     gp_flip = GPManager.load(GP_FLIP_PATH, device=DEVICE)
#     gp_rate = GPManager.load(GP_RATE_PATH, device=DEVICE)

#     # predict derivatives on test
#     Mean_t, Var_t = predict_derivs(gp_flip, gp_rate, Xte)
#     Mean = Mean_t.detach().cpu().numpy()
#     Var  = Var_t.detach().cpu().numpy()

#     # derivative metrics
#     print("\n=== DERIVATIVE METRICS (test) ===")
#     print(f"dflip/dt  RMSE={rmse(Yte[:,0], Mean[:,0]):.4f}  MAE={mae(Yte[:,0], Mean[:,0]):.4f}")
#     print(f"drate/dt  RMSE={rmse(Yte[:,1], Mean[:,1]):.4f}  MAE={mae(Yte[:,1], Mean[:,1]):.4f}")

#     # 1-step next-state via Euler integration
#     x_pred = np.empty_like(x0)
#     x_pred[:,0] = x0[:,0] + Mean[:,0] * dt
#     x_pred[:,1] = x0[:,1] + Mean[:,1] * dt

#     print("\n=== 1-STEP NEXT-STATE METRICS (test) ===")
#     print(f"flip_next  RMSE={rmse(x1[:,0], x_pred[:,0]):.4f}  MAE={mae(x1[:,0], x_pred[:,0]):.4f}")
#     print(f"rate_next  RMSE={rmse(x1[:,1], x_pred[:,1]):.4f}  MAE={mae(x1[:,1], x_pred[:,1]):.4f}")

#     # uncertainty calibration check (derivatives)
#     print("\n=== UNCERTAINTY CHECK (derivatives) ===")
#     z0 = (Yte[:,0] - Mean[:,0]) / (np.sqrt(np.clip(Var[:,0], 1e-9, 1e9)))
#     z1 = (Yte[:,1] - Mean[:,1]) / (np.sqrt(np.clip(Var[:,1], 1e-9, 1e9)))
#     cov0 = float(np.mean(np.abs(z0) <= 2.0))
#     cov1 = float(np.mean(np.abs(z1) <= 2.0))
#     print(f"Coverage |z|<=2 (target ~0.95): flip={cov0:.3f}, rate={cov1:.3f}")

#     # near-goal subset (goal = 1.5 rad)
#     err_to_goal = np.array([angdiff(th, THETA_TARGET) for th in x0[:,0]])
#     mask = np.abs(err_to_goal) < NEAR_BAND
#     if mask.sum() > 20:
#         print("\n=== NEAR-GOAL 1-STEP MAE (subset) ===")
#         print(f"flip_next MAE={mae(x1[mask,0], x_pred[mask,0]):.5f} rad")
#         print(f"rate_next MAE={mae(x1[mask,1], x_pred[mask,1]):.5f} rad/s")
#         bias = float(np.mean(x_pred[mask,0] - x1[mask,0]))
#         print(f"flip_next bias (mean pred-true) = {bias:.5f} rad")
#     else:
#         print("\n[WARN] Not enough near-goal samples in test set to evaluate near-goal MAE.")

#     # multi-step rollout test on random starting points (using recorded actions)
#     print("\n=== MULTI-STEP ROLLOUT TEST ===")
#     # build a mapping from chosen test samples back to original indices to fetch action sequences
#     # simplest: use contiguous segments from the original arrays by searching index positions
#     # For now, just sample random contiguous windows in X (already sequential-ish from original idx list)
#     # We’ll do rollouts directly in that Xte ordering if it’s sequential enough; if not, skip.
#     if n_test > (H + 5):
#         for r in range(N_ROLLOUTS):
#             i0 = np.random.randint(0, n_test - H)
#             # initial state from x0
#             x = x0[i0].copy()
#             errs = []
#             for k in range(H):
#                 u_k = Xte[i0 + k, 2]
#                 Xq = np.array([[x[0], x[1], u_k]], dtype=np.float32)
#                 m, _ = predict_derivs(gp_flip, gp_rate, Xq)
#                 m = m.detach().cpu().numpy().ravel()
#                 x[0] = x[0] + m[0] * dt
#                 x[1] = x[1] + m[1] * dt

#                 x_true = x1[i0 + k]  # approximate “next” truth
#                 errs.append(np.linalg.norm(x - x_true))

#             print(f"rollout {r:02d}: final_err={errs[-1]:.4f}  mean_err={float(np.mean(errs)):.4f}")
#     else:
#         print("[WARN] Not enough test samples for rollout test.")

# if __name__ == "__main__":
#     main()


# import numpy as np, math

# D = np.load("data/mujoco_random_run.npz")
# flip = D["flip"].astype(np.float32)

# target = 1.5
# band = 0.3
# def angdiff(a,b):
#     return (a-b+math.pi)%(2*math.pi)-math.pi

# err = np.array([angdiff(f, target) for f in flip])
# print("Fraction within ±0.3 rad:", np.mean(np.abs(err) < band))
# print("Fraction within ±0.1 rad:", np.mean(np.abs(err) < 0.1))


#!/usr/bin/env python3
import os
import sys
import numpy as np

NPZ_PATH = "data/mujoco_random_run_dt0p1.npz"  # <-- change this

def fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

def main():
    path = NPZ_PATH if len(sys.argv) < 2 else sys.argv[1]

    if not os.path.isfile(path):
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    size_bytes = os.path.getsize(path)
    print(f"File: {path}")
    print(f"On-disk size: {size_bytes} bytes ({fmt_bytes(size_bytes)})")

    D = np.load(path, allow_pickle=False)
    print("\nKeys:", D.files)

    total_nbytes = 0
    print("\n--- Array details (uncompressed in memory) ---")
    for k in D.files:
        arr = D[k]
        nbytes = int(arr.nbytes) if hasattr(arr, "nbytes") else 0
        total_nbytes += nbytes
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        print(f"{k:12s} shape={shape!s:14s} dtype={str(dtype):10s} nbytes={nbytes:10d} ({fmt_bytes(nbytes)})")

    print(f"\nTotal array bytes (sum of nbytes): {total_nbytes} ({fmt_bytes(total_nbytes)})")
    ratio = (size_bytes / total_nbytes) if total_nbytes > 0 else float("nan")
    print(f"Compression ratio (disk / arrays): {ratio:.3f}")

if __name__ == "__main__":
    main()
