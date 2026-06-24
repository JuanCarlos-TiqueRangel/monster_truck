#!/usr/bin/env python3
"""
gp_slice.py
-----------
Inspect the learned GP fit. The residual GP lives in z = [x, theta]; this gives you
two complementary views of it:

  SLICE view (default): fix theta (e.g. 30 deg) and sweep x, so you see the learned
  surface as a 1-D curve of residual ACCELERATION vs POSITION at that pitch. Both GP
  channels are shown, mean +/- 2 sigma vs x:
      * v_dot residual     -- forward "blockage" acceleration [m/s^2] (the main signal)
      * omega_dot residual -- pitch acceleration residual [rad/s^2]
  Overlaid are (a) the measured residual samples whose pitch falls in a band around the
  slice, and (b) -- with --online -- the gp_*_pred the CONTROLLER actually used online
  (last episode), so you can check the npz-reconstructed curve matches the live model.

  HEATMAP view (--heatmap): the full residual surface over the x-theta plane, with the
  inducing points and your slice line(s) drawn on top -- shows WHERE in (x, theta) the
  GP has localised the obstacles and how the blockage changes as the truck rears.

Usage:
    python gp_slice.py                       # theta = 30 deg slice
    python gp_slice.py 0 30 55               # several pitch slices on one figure
    python gp_slice.py 30 --heatmap          # slice + 2-D x-theta surface
    python gp_slice.py 30 --no-online        # don't overlay the online predictions
    python gp_slice.py 30 --csv obstacle_test.csv --model obstacle_test_model.npz
    python gp_slice.py 30 --band 8           # +/-8 deg band for overlaying data

The model (.npz) supplies the GP; the CSV is only for the overlays (optional --
omit it with --no-data, which also disables --online).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from SSGP import StreamingSparseVGP


OBSTACLE_XS = [1.0, 3.0, 6.8, 7.2]   # known box locations, for guide lines


def load_channels(model_path: Path):
    """Rebuild both single-output GPs (v_dot, omega_dot) directly from the saved npz."""
    md = np.load(model_path, allow_pickle=False)
    g = {k[len("gp_"):]: md[k] for k in md.files if k.startswith("gp_")}
    Z = np.asarray(g["Z"], float)
    ell = np.asarray(g["l"], float)
    sf2 = float(g["sf2"])
    sn2 = float(g["sn2"])

    def build(mkey, skey):
        gp = StreamingSparseVGP(Z, ell, sf2, sn2)
        gp.m = np.asarray(g[mkey], float)
        gp.S = np.asarray(g[skey], float)
        return gp

    # tolerate the pre-rename keys (m_v / m_w) as well as the current ones
    mv = "m_v_dot" if "m_v_dot" in g else "m_v"
    sv = "S_v_dot" if "S_v_dot" in g else "S_v"
    mw = "m_omega_dot" if "m_omega_dot" in g else "m_w"
    sw = "S_omega_dot" if "S_omega_dot" in g else "S_w"
    return build(mv, sv), build(mw, sw), Z


def gp_curve(gp, xs, theta_rad):
    """Predict mean +/- sd over the x grid at a fixed theta."""
    mu = np.empty_like(xs)
    sd = np.empty_like(xs)
    for i, xx in enumerate(xs):
        mean, var = gp.predict(np.array([xx, theta_rad]))
        mu[i] = mean
        sd[i] = np.sqrt(var)
    return mu, sd


def gp_surface(gp, xs, ths_rad):
    """Predict the mean over the full (x, theta) grid -> array [n_theta, n_x]."""
    G = np.empty((len(ths_rad), len(xs)))
    for i, th in enumerate(ths_rad):
        for j, xx in enumerate(xs):
            G[i, j], _ = gp.predict(np.array([xx, th]))
    return G


def make_slice_figure(args, gp_v, gp_w, xs, df, last_ep):
    fig, (ax_v, ax_w, ax_s) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.thetas)))

    for theta_deg, c in zip(args.thetas, colors):
        th = np.radians(theta_deg)
        mu_v, sd_v = gp_curve(gp_v, xs, th)
        mu_w, sd_w = gp_curve(gp_w, xs, th)
        lbl = f"GP mean, $\\theta$={theta_deg:.0f}$\\degree$"

        ax_v.fill_between(xs, mu_v - 2*sd_v, mu_v + 2*sd_v, color=c, alpha=0.15)
        ax_v.plot(xs, mu_v, color=c, lw=2, label=lbl)
        ax_w.fill_between(xs, mu_w - 2*sd_w, mu_w + 2*sd_w, color=c, alpha=0.15)
        ax_w.plot(xs, mu_w, color=c, lw=2, label=lbl)

        if df is not None:
            band = np.abs(np.degrees(df["pitch_rad"]) - theta_deg) < args.band
            dd = df[band]
            if len(dd):
                # (a) measured residuals in the pitch band
                ax_v.scatter(dd["x"], dd["r_v_dot"], s=6, alpha=0.12, color=c)
                ax_w.scatter(dd["x"], dd["r_omega_dot"], s=6, alpha=0.12, color=c)
                # (b) the online predictions the controller used (last episode only,
                #     so it matches the FINAL npz model rather than the warmup ones)
                if args.online and last_ep is not None:
                    on = dd[dd["episode"] == last_ep]
                    if len(on):
                        ax_v.scatter(on["x"], on["gp_v_dot_pred"], s=14, marker="x",
                                     color=c, alpha=0.8,
                                     label=f"online pred (ep {last_ep})")
                        ax_w.scatter(on["x"], on["gp_omega_dot_pred"], s=14, marker="x",
                                     color=c, alpha=0.8)
                # (c) truck SPEED (x_dot) at this pitch slice, binned by x (mean) -- context
                #     for the blockage: where it is fast/slow approaching & crossing the box.
                xbin = (dd["x"] * 4).round() / 4
                spd = dd.groupby(xbin)["x_dot"].mean()
                ax_s.plot(spd.index, spd.values, color=c, lw=2,
                          label=f"speed, $\\theta$={theta_deg:.0f}$\\degree$")
                print(f"theta={theta_deg:5.1f} deg : {len(dd)} data pts within +/-{args.band:.0f} deg; "
                      f"v_dot residual peak |mu|={np.max(np.abs(mu_v)):.3f} m/s^2 "
                      f"at x={xs[np.argmax(np.abs(mu_v))]:.2f} m")
            else:
                print(f"theta={theta_deg:5.1f} deg : no data within +/-{args.band:.0f} deg "
                      f"(GP still shown); peak |mu_v|={np.max(np.abs(mu_v)):.3f} m/s^2")

    for ax in (ax_v, ax_w, ax_s):
        for xo in OBSTACLE_XS:
            ax.axvline(xo, color="0.7", ls=":", lw=0.8)
        ax.grid(True)
        ax.legend(fontsize=8, ncol=max(1, len(args.thetas)))
    for ax in (ax_v, ax_w):
        ax.axhline(0, color="0.8", lw=0.8)
    ax_v.set_ylabel("v_dot residual  [m/s$^2$]\n(forward 'blockage' accel)")
    ax_w.set_ylabel("omega_dot residual  [rad/s$^2$]\n(pitch accel)")
    ax_s.set_ylabel("x_dot (speed) [m/s]\n(measured, binned by x)")
    ax_s.set_xlabel("x [m]")
    note = "shaded = $\\pm2\\sigma$, dots = measured data in band"
    if args.online:
        note += ", x = online pred"
    ax_v.set_title(f"GP fit: residual acceleration vs x at FIXED pitch  ({note})")
    fig.tight_layout()
    return fig


def make_heatmap_figure(args, gp_v, gp_w, xs, Z, df):
    """The full residual surface over the x-theta plane (GP v_dot, GP omega_dot) plus the
    MEASURED speed x_dot over the same plane (from data), inducing pts + slice lines on top."""
    theta_hi = max(np.degrees(Z[:, 1].max()), max(args.thetas)) + 5.0
    ths = np.radians(np.linspace(0.0, theta_hi, 140))
    Gv = gp_surface(gp_v, xs, ths)
    Gw = gp_surface(gp_w, xs, ths)
    th_deg = np.degrees(ths)

    fig, (ax_v, ax_w, ax_s) = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for ax, G, title, unit in (
        (ax_v, Gv, "v_dot residual (forward blockage)", "m/s$^2$"),
        (ax_w, Gw, "omega_dot residual (pitch accel)", "rad/s$^2$"),
    ):
        lim = float(np.percentile(np.abs(G), 99)) or 1.0          # symmetric, robust
        pc = ax.pcolormesh(xs, th_deg, G, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           shading="auto")
        fig.colorbar(pc, ax=ax, label=f"residual [{unit}]")
        # inducing points: WHERE the GP actually has support
        ax.scatter(Z[:, 0], np.degrees(Z[:, 1]), s=12, c="k", marker="o",
                   alpha=0.5, label="inducing pts Z")
        for theta_deg in args.thetas:                              # the slice line(s)
            ax.axhline(theta_deg, color="lime", ls="--", lw=1.3)
        for xo in OBSTACLE_XS:
            ax.axvline(xo, color="0.3", ls=":", lw=0.8)
        ax.set_ylabel("pitch $\\theta$ [deg]")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")

    # measured SPEED x_dot binned over (x, theta) -- not a GP output, comes from the data
    if df is not None and "x_dot" in df:
        x_d = df["x"].to_numpy()
        th_d = np.degrees(df["pitch_rad"].to_numpy())
        v_d = df["x_dot"].to_numpy()
        xe = np.linspace(xs.min(), xs.max(), 90)
        te = np.linspace(0.0, th_deg.max(), 60)
        ssum = np.histogram2d(x_d, th_d, bins=[xe, te], weights=v_d)[0]
        cnt = np.histogram2d(x_d, th_d, bins=[xe, te])[0]
        with np.errstate(invalid="ignore"):
            S = np.where(cnt > 5, ssum / cnt, np.nan).T            # [theta, x] for pcolormesh
        xc, tc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (te[:-1] + te[1:])
        pc = ax_s.pcolormesh(xc, tc, S, cmap="viridis", shading="auto")
        fig.colorbar(pc, ax=ax_s, label="speed x_dot [m/s]")
        for theta_deg in args.thetas:
            ax_s.axhline(theta_deg, color="r", ls="--", lw=1.3)
        for xo in OBSTACLE_XS:
            ax_s.axvline(xo, color="0.3", ls=":", lw=0.8)
        ax_s.set_title("measured speed x_dot over (x, $\\theta$)  (empty cells = unvisited)")
    else:
        ax_s.text(0.5, 0.5, "no data (x_dot) -- run with a CSV", ha="center",
                  va="center", transform=ax_s.transAxes)
    ax_s.set_ylabel("pitch $\\theta$ [deg]")
    ax_s.set_xlabel("x [m]")
    fig.suptitle("Surfaces over (x, $\\theta$): GP residuals + measured speed   "
                 "(dashed = your slice; the slice plot is a horizontal cut here)")
    fig.tight_layout()
    return fig


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="GP fit: fixed-pitch slice and/or x-theta heatmap.")
    ap.add_argument("thetas", nargs="*", type=float, default=[30.0],
                    help="pitch slice(s) in DEGREES (default: 30)")
    ap.add_argument("--model", default=None, help="GP checkpoint .npz")
    ap.add_argument("--csv", default=None, help="trajectory CSV for the data/online overlays")
    ap.add_argument("--band", type=float, default=10.0,
                    help="+/- pitch band [deg] for overlaying measured data (default 10)")
    ap.add_argument("--xmax", type=float, default=11.0, help="max x to sweep [m]")
    ap.add_argument("--no-data", action="store_true", help="skip the measured-data overlay")
    ap.add_argument("--no-online", dest="online", action="store_false",
                    help="don't overlay the controller's online gp_*_pred")
    ap.add_argument("--heatmap", action="store_true",
                    help="also produce the 2-D x-theta residual surface")
    args = ap.parse_args()
    if not args.thetas:
        args.thetas = [30.0]

    model = Path(args.model) if args.model else here / "obstacle_model.npz"
    if not model.exists():
        raise SystemExit(f"model not found: {model}\nRun obstacle_mujoco_simulation.py first.")
    gp_v, gp_w, Z = load_channels(model)
    print(f"Loaded GP: {model.name}  (M={Z.shape[0]} inducing pts, "
          f"x in [{Z[:,0].min():.2f},{Z[:,0].max():.2f}], "
          f"theta in [{np.degrees(Z[:,1].min()):.1f},{np.degrees(Z[:,1].max()):.1f}] deg)")

    xs = np.linspace(0.0, args.xmax, 300)

    # optional measured data (+ online predictions) for the overlays
    df = None
    last_ep = None
    if not args.no_data:
        csv = Path(args.csv) if args.csv else here / "obstacle_mujoco.csv"
        if csv.exists():
            import pandas as pd
            df = pd.read_csv(csv, usecols=["episode", "x", "x_dot", "pitch_rad", "r_v_dot",
                                           "r_omega_dot", "gp_v_dot_pred",
                                           "gp_omega_dot_pred", "gp_ready"])
            df = df[df["gp_ready"] > 0.5]
            last_ep = int(df["episode"].max()) if len(df) else None
        else:
            print(f"[note] CSV {csv.name} not found -- plotting GP only "
                  f"(use --csv or --no-data).")
            args.online = False

    fig_slice = make_slice_figure(args, gp_v, gp_w, xs, df, last_ep)

    img_dir = here / "images"
    tag = "_".join(f"{t:.0f}" for t in args.thetas)

    def save(fig, name):
        try:
            img_dir.mkdir(exist_ok=True)
            out = img_dir / name
            fig.savefig(out, dpi=180, bbox_inches="tight")
        except PermissionError:
            out = Path("/tmp") / name
            fig.savefig(out, dpi=180, bbox_inches="tight")
            print(f"[warn] {img_dir} not writable; saved to {out}")
        print(f"Saved figure: {out}")

    save(fig_slice, f"gp_slice_theta{tag}.png")

    if args.heatmap:
        fig_hm = make_heatmap_figure(args, gp_v, gp_w, xs, Z, df)
        save(fig_hm, f"gp_surface_theta{tag}.png")

    plt.show()


if __name__ == "__main__":
    main()
