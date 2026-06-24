#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from GP import GPConfig


OBSTACLE_XS = [1.0, 3.0, 6.8, 7.2]


def find_results_dir():
    here = Path(__file__).resolve().parent
    for p in [here / "results", here.parent / "results", Path.cwd() / "results"]:
        if p.exists():
            return p
    return here.parent / "results"


def load_gp(model_path: Path):
    data = np.load(model_path, allow_pickle=False)
    sd = {k[len("gp_"):]: data[k] for k in data.files if k.startswith("gp_")}

    if "ready" not in sd or int(np.asarray(sd["ready"])) != 1:
        raise SystemExit(f"{model_path}: no trained GP found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gp = GPConfig(device=device).build()
    gp.load_state_dict(sd)

    Z_raw = gp._cache["Z"] * gp._cache["x_std"] + gp._cache["x_mean"]

    return gp, Z_raw


def load_csv(csv_path: Path, ready_only=True):
    import pandas as pd

    df = pd.read_csv(csv_path)

    if ready_only and "gp_ready" in df.columns:
        df = df[df["gp_ready"] > 0.5].copy()

    if len(df) == 0:
        raise SystemExit("CSV has no usable rows.")

    if "pitch_rad" not in df.columns and "pitch_deg" in df.columns:
        df["pitch_rad"] = np.radians(df["pitch_deg"].to_numpy())

    if "pitch_deg" not in df.columns and "pitch_rad" in df.columns:
        df["pitch_deg"] = np.degrees(df["pitch_rad"].to_numpy())

    needed = ["x", "x_dot", "pitch_rad", "pitch_deg", "pitch_dot", "tau_cmd"]
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    return df


def query_gp(gp, X, batch_size=50000):
    out_v = []
    out_w = []

    X = np.ascontiguousarray(X, dtype=float)

    for i in range(0, X.shape[0], batch_size):
        Xi = X[i:i + batch_size]
        mv, mw = gp.predict_torch_fast(Xi, dtype=torch.float64)
        out_v.append(mv.detach().cpu().numpy())
        out_w.append(mw.detach().cpu().numpy())

    return np.concatenate(out_v), np.concatenate(out_w)


def sample_hidden_features(df, n_mc, seed=0):
    """
    Sample real logged values of [v, omega, tau].

    These are the hidden dimensions when we visualize the 5D GP over only (x, theta).
    """

    rng = np.random.default_rng(seed)

    vals = np.column_stack([
        df["x_dot"].to_numpy(float),
        df["pitch_dot"].to_numpy(float),
        df["tau_cmd"].to_numpy(float),
    ])

    idx = rng.choice(vals.shape[0], size=min(n_mc, vals.shape[0]), replace=False)

    return vals[idx]


def projected_surface(gp, xs, theta_deg_grid, hidden_samples, reducer="median"):
    """
    Build a clean 2D surface from the 5D GP.

    For each (x, theta), query:
        GP([x, v_i, theta, omega_i, tau_i])

    using many real samples of (v_i, omega_i, tau_i), then reduce with median/mean.
    """

    n_theta = len(theta_deg_grid)
    n_x = len(xs)
    n_mc = hidden_samples.shape[0]

    Gv = np.empty((n_theta, n_x))
    Gw = np.empty((n_theta, n_x))

    for i, theta_deg in enumerate(theta_deg_grid):
        theta = np.radians(theta_deg)

        Xq = np.empty((n_x * n_mc, 5), dtype=float)

        row = 0
        for x in xs:
            for j in range(n_mc):
                v, omega, tau = hidden_samples[j]

                Xq[row, 0] = x
                Xq[row, 1] = v
                Xq[row, 2] = theta
                Xq[row, 3] = omega
                Xq[row, 4] = tau
                row += 1

        mv, mw = query_gp(gp, Xq)

        mv = mv.reshape(n_x, n_mc)
        mw = mw.reshape(n_x, n_mc)

        if reducer == "mean":
            Gv[i, :] = np.mean(mv, axis=1)
            Gw[i, :] = np.mean(mw, axis=1)
        else:
            Gv[i, :] = np.median(mv, axis=1)
            Gw[i, :] = np.median(mw, axis=1)

    return Gv, Gw


def smooth_surface(G, sigma):
    if sigma <= 0.0:
        return G

    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(G, sigma=sigma)
    except Exception:
        return G


def robust_lim(G):
    lim = float(np.percentile(np.abs(G[np.isfinite(G)]), 98.5))

    if lim < 1e-9:
        lim = 1.0

    return lim


def make_heatmap(args, gp, Z_raw, df):
    xs = np.linspace(args.xmin, args.xmax, args.nx)
    theta_grid = np.linspace(args.theta_min, args.theta_max, args.ntheta)

    hidden = sample_hidden_features(df, args.mc, seed=args.seed)

    Gv, Gw = projected_surface(
        gp,
        xs,
        theta_grid,
        hidden,
        reducer=args.reducer,
    )

    Gv = smooth_surface(Gv, args.smooth)
    Gw = smooth_surface(Gw, args.smooth)

    fig, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    lim_v = robust_lim(Gv)
    lim_w = robust_lim(Gw)

    pc1 = ax_v.pcolormesh(
        xs,
        theta_grid,
        Gv,
        cmap="RdBu_r",
        vmin=-lim_v,
        vmax=lim_v,
        shading="auto",
    )

    fig.colorbar(pc1, ax=ax_v, label=r"$\Delta \dot v$ [m/s$^2$]")

    pc2 = ax_w.pcolormesh(
        xs,
        theta_grid,
        Gw,
        cmap="RdBu_r",
        vmin=-lim_w,
        vmax=lim_w,
        shading="auto",
    )

    fig.colorbar(pc2, ax=ax_w, label=r"$\Delta \dot \omega$ [rad/s$^2$]")

    Zx = Z_raw[:, 0]
    Ztheta = np.degrees(Z_raw[:, 2])

    for ax in (ax_v, ax_w):
        ax.scatter(
            Zx,
            Ztheta,
            s=14,
            c="k",
            alpha=0.45,
            label="inducing pts projected to (x, theta)",
        )

        for theta in args.thetas:
            ax.axhline(theta, color="lime", ls="--", lw=1.3)

        if args.show_guides:
            for xo in OBSTACLE_XS:
                ax.axvline(xo, color="0.25", ls=":", lw=0.9)

        ax.set_ylabel(r"pitch $\theta$ [deg]")
        ax.legend(fontsize=8, loc="upper right")

    ax_v.set_title("Projected 5D GP surface over $(x, \\theta)$")
    ax_v.set_title(r"$\Delta \dot v$ residual")
    ax_w.set_title(r"$\Delta \dot \omega$ residual")
    ax_w.set_xlabel("x [m]")

    fig.suptitle(
        f"5D GP projected over $(x,\\theta)$ using {args.reducer} over "
        f"{hidden.shape[0]} real $(v,\\omega,\\tau)$ samples"
    )

    fig.tight_layout()

    return fig, xs, theta_grid, Gv, Gw


def make_slice(args, xs, theta_grid, Gv, Gw):
    fig, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.thetas)))

    for theta, color in zip(args.thetas, colors):
        idx = int(np.argmin(np.abs(theta_grid - theta)))

        ax_v.plot(xs, Gv[idx], color=color, lw=2.0, label=f"$\\theta$={theta:.0f}$\\degree$")
        ax_w.plot(xs, Gw[idx], color=color, lw=2.0, label=f"$\\theta$={theta:.0f}$\\degree$")

        peak_v = int(np.argmax(np.abs(Gv[idx])))
        peak_w = int(np.argmax(np.abs(Gw[idx])))

        print(
            f"theta={theta:6.1f} deg | "
            f"peak |v_dot|={abs(Gv[idx, peak_v]):.3f} at x={xs[peak_v]:.2f} | "
            f"peak |omega_dot|={abs(Gw[idx, peak_w]):.3f} at x={xs[peak_w]:.2f}"
        )

    for ax in (ax_v, ax_w):
        ax.axhline(0.0, color="0.75", lw=0.8)

        if args.show_guides:
            for xo in OBSTACLE_XS:
                ax.axvline(xo, color="0.35", ls=":", lw=0.9)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ax_v.set_ylabel(r"$\Delta \dot v$ [m/s$^2$]")
    ax_w.set_ylabel(r"$\Delta \dot \omega$ [rad/s$^2$]")
    ax_w.set_xlabel("x [m]")

    fig.suptitle("Horizontal cuts of the projected GP surface")
    fig.tight_layout()

    return fig


def main():
    results = find_results_dir()

    ap = argparse.ArgumentParser(description="Good projected heatmap for 5D GP residual model.")

    ap.add_argument("thetas", nargs="*", type=float, default=[30.0, 65.0],
                    help="pitch slice lines in degrees")

    ap.add_argument("--model", default=None)
    ap.add_argument("--csv", default=None)

    ap.add_argument("--xmin", type=float, default=0.0)
    ap.add_argument("--xmax", type=float, default=11.0)
    ap.add_argument("--nx", type=int, default=350)

    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=75.0)
    ap.add_argument("--ntheta", type=int, default=160)

    ap.add_argument("--mc", type=int, default=64,
                    help="number of real (v,omega,tau) samples used to project the 5D GP")

    ap.add_argument("--reducer", choices=["median", "mean"], default="median")
    ap.add_argument("--smooth", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--show-guides", action="store_true")

    args = ap.parse_args()

    model_path = Path(args.model) if args.model else results / "obstacle_model.npz"
    csv_path = Path(args.csv) if args.csv else results / "obstacle_mujoco.csv"

    if not model_path.exists():
        raise SystemExit(f"model not found: {model_path}")

    if not csv_path.exists():
        raise SystemExit(f"csv not found: {csv_path}")

    gp, Z_raw = load_gp(model_path)
    df = load_csv(csv_path, ready_only=not args.all_data)

    print(f"Loaded model: {model_path}")
    print(f"Loaded CSV:   {csv_path}")
    print(f"Rows used:    {len(df)}")

    print("\nGP lengthscales [x, v, theta, omega, tau]:")
    print(f"  v_dot     = {np.round(gp._cache['ell_v'], 3)}")
    print(f"  omega_dot = {np.round(gp._cache['ell_w'], 3)}")

    fig_hm, xs, theta_grid, Gv, Gw = make_heatmap(args, gp, Z_raw, df)
    fig_slice = make_slice(args, xs, theta_grid, Gv, Gw)

    out_dir = results / "images"
    out_dir.mkdir(exist_ok=True)

    hm_path = out_dir / "gp_projected_surface_good.png"
    sl_path = out_dir / "gp_projected_slices_good.png"

    fig_hm.savefig(hm_path, dpi=180, bbox_inches="tight")
    fig_slice.savefig(sl_path, dpi=180, bbox_inches="tight")

    print(f"\nSaved heatmap: {hm_path}")
    print(f"Saved slices:  {sl_path}")

    plt.show()


if __name__ == "__main__":
    main()