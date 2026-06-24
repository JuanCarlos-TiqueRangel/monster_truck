"""
================================================================================
Online Sparse GP residual model  --  5-input / 2-output, with PERSISTENCE (MBRL)
================================================================================

INPUT  (raw state, per sample):   [x, pitch, omega, velocity, torque]      (d_in = 5)
OUTPUT (residual, per sample):     [v_dot, omega_dot]                       (d_out = 2)

The obstacle is NOT given; it is inferred from where the residual map is large.

Method (unchanged): [1] ARD Matern-3/2 kernel  [2] optional contact feature (off
by default; obstacle is inferred)  [3] heteroscedastic noise (only with a contact
signal)  [4] recursive RLS/Kalman online updates with FROZEN hyperparameters
[5] proper sparse-GP predictive variance (reverts to the prior far from data).

--------------------------------------------------------------------------------
PERSISTENCE FOR MODEL-BASED RL
--------------------------------------------------------------------------------
The episodic loop you want looks like this:

    PATH = "residual_model.pkl"
    if model_exists(PATH):
        model = load_model(PATH)          # a model already exists -> keep improving it
    else:
        model = MultiOutputResidualGP(output_names=["v_dot", "omega_dot"])
        model.fit(X_warmup, Y_warmup)     # FIRST TIME ONLY: sets hyperparameters,
                                          # inducing points, and absorbs the warmup data
    # ... run the episode, collect (X_episode, Y_episode) ...
    model.update(X_episode, Y_episode)    # recursive update; refines the model
    save_model(model, PATH)               # persist for the next episode

`fit` is called once (it does the offline hyperparameter optimization + inducing-
point placement). Every later episode only `load -> update -> save`. What is saved
is the full learned posterior (m_u, S_u) plus the frozen hyperparameters and
standardization stats, so a reloaded model continues the exact same recursive
update stream. The contact_activation callable is NOT pickled; if you train with
one, pass it again to load_model(...). With the default (None) there is nothing to
pass.

Dependencies: numpy, scipy (kmeans2, minimize). matplotlib only for the demo.
================================================================================
"""

import os
import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2

_FORMAT_VERSION = 1


# ----------------------------------------------------------------------------- #
# [1] ARD Matern kernel.  k(x,x) = sf2.  nu = 1.5 (smooth) or 0.5 (sharp/jagged). #
# ----------------------------------------------------------------------------- #
def ard_matern(A, B, ell, sf2, nu=1.5):
    Aw, Bw = A / ell, B / ell
    d2 = (Aw**2).sum(1)[:, None] + (Bw**2).sum(1)[None, :] - 2.0 * Aw @ Bw.T
    r = np.sqrt(np.maximum(d2, 1e-12))
    if nu == 0.5:
        return sf2 * np.exp(-r)
    if nu == 1.5:
        c = np.sqrt(3.0) * r
        return sf2 * (1.0 + c) * np.exp(-c)
    raise ValueError("nu must be 0.5 or 1.5")


# ============================================================================ #
# Single-output recursive sparse GP (one residual channel).
# ============================================================================ #
class OnlineSparseGP:
    def __init__(self, contact_activation=None, nu=1.5, feature_scale=2.0,
                 noise_contact_mult=8.0, forgetting=1.0, jitter=1e-6):
        self.contact_activation = contact_activation
        self.nu = nu
        self.feature_scale = feature_scale
        self.noise_contact_mult = noise_contact_mult
        self.forgetting = forgetting
        self.jitter = jitter
        self.n_obs = 0                                     # total observations absorbed (bookkeeping)
        # set by fit():
        self.x_mean = self.x_std = self.y_mean = None
        self.ell = self.sf2 = self.sn2_base = None
        self.Z = self.Kzz = self.Kzz_inv = self.m_u = self.S_u = None

    @staticmethod
    def _as2d(X):
        X = np.asarray(X, float)
        return X.reshape(1, -1) if X.ndim == 1 else X

    def _augment(self, X_raw):
        X2 = self._as2d(X_raw)
        Xs = (X2 - self.x_mean) / self.x_std
        if self.contact_activation is None:
            return Xs
        act = self.contact_activation(X2).reshape(-1, 1)
        return np.hstack([Xs, self.feature_scale * act])

    def _noise_var(self, X_raw):
        X2 = self._as2d(X_raw)
        if self.contact_activation is None:
            return np.full(X2.shape[0], self.sn2_base)
        act = self.contact_activation(X2)
        return self.sn2_base * (1.0 + self.noise_contact_mult * act)

    def _compute_kernel_cache(self):
        """Recompute Kzz and its inverse from Z + hyperparameters. Does NOT touch the
        posterior (m_u, S_u) -- this is what makes loading a trained model safe."""
        Kzz = ard_matern(self.Z, self.Z, self.ell, self.sf2, self.nu)
        Kzz = Kzz + self.jitter * np.eye(len(self.Z))
        self.Kzz = Kzz
        self.Kzz_inv = np.linalg.inv(Kzz)

    def _refresh_cache(self):
        self._compute_kernel_cache()
        self.S_u = self.Kzz.copy()                         # prior over inducing values
        self.m_u = np.zeros(len(self.Z))

    def fit(self, X_raw, y, n_inducing=80):
        X2 = self._as2d(X_raw)
        y = np.asarray(y, float).ravel()
        self.x_mean = X2.mean(0)
        self.x_std = X2.std(0); self.x_std[self.x_std < 1e-8] = 1.0
        self.y_mean = float(y.mean())
        yc = y - self.y_mean

        Xa = self._augment(X2)
        D = Xa.shape[1]
        ell0 = Xa.std(0); ell0[ell0 < 1e-2] = 1e-2
        theta0 = np.concatenate([np.log(ell0),
                                 [np.log(max(np.var(yc), 1e-6))],
                                 [np.log(max(0.1 * np.var(yc), 1e-6))]])

        def nlml(theta):
            ell = np.exp(theta[:D]); sf2 = np.exp(theta[D]); sn2 = np.exp(theta[D + 1])
            K = ard_matern(Xa, Xa, ell, sf2, self.nu) + (sn2 + 1e-8) * np.eye(len(yc))
            try:
                Lc = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e12
            alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yc))
            return 0.5 * yc @ alpha + np.log(np.diag(Lc)).sum() + 0.5 * len(yc) * np.log(2 * np.pi)

        th = minimize(nlml, theta0, method="L-BFGS-B").x
        self.ell = np.exp(th[:D]); self.sf2 = float(np.exp(th[D])); self.sn2_base = float(np.exp(th[D + 1]))

        M = min(n_inducing, len(Xa))
        Z, _ = kmeans2(Xa, M, seed=0, minit="++", missing="warn")
        self.Z = Z
        self._refresh_cache()
        self.update(X2, y, apply_forgetting=False)
        return self

    def update(self, X_raw, y, apply_forgetting=True):
        X2 = self._as2d(X_raw)
        yc = np.asarray(y, float).ravel() - self.y_mean
        Xa = self._augment(X2)
        KzX = ard_matern(self.Z, Xa, self.ell, self.sf2, self.nu)
        A = self.Kzz_inv @ KzX
        r = self._noise_var(X2)
        forget = apply_forgetting and (self.forgetting < 1.0)
        for n in range(Xa.shape[0]):
            if forget:
                self.S_u = self.forgetting * self.S_u + (1.0 - self.forgetting) * self.Kzz
            a = A[:, n]
            Sa = self.S_u @ a
            s = float(a @ Sa) + r[n]
            L = Sa / s
            self.m_u = self.m_u + L * (yc[n] - float(a @ self.m_u))
            self.S_u = self.S_u - np.outer(L, Sa)
        self.S_u = 0.5 * (self.S_u + self.S_u.T)
        self.n_obs += Xa.shape[0]
        return self

    def predict(self, X_raw, return_std=False):
        X2 = self._as2d(X_raw)
        Xa = self._augment(X2)
        KzX = ard_matern(self.Z, Xa, self.ell, self.sf2, self.nu)
        B = self.Kzz_inv @ KzX
        mean = B.T @ self.m_u + self.y_mean
        nystrom = np.sum(KzX * B, axis=0)
        post = np.sum(B * (self.S_u @ B), axis=0)
        var = np.maximum(self.sf2 - nystrom + post + self._noise_var(X2), 1e-9)
        return (mean, np.sqrt(var)) if return_std else (mean, var)

    # ---- persistence (per channel) ----------------------------------------
    def get_state(self):
        """Everything needed to reconstruct this channel and keep updating it.
        Picklable: arrays + scalars only (the contact_activation callable is NOT included)."""
        return {
            "config": {"nu": self.nu, "feature_scale": self.feature_scale,
                       "noise_contact_mult": self.noise_contact_mult,
                       "forgetting": self.forgetting, "jitter": self.jitter,
                       "had_contact_activation": self.contact_activation is not None},
            "hyper": {"ell": self.ell, "sf2": self.sf2, "sn2_base": self.sn2_base},
            "stdz": {"x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean},
            "Z": self.Z, "m_u": self.m_u, "S_u": self.S_u, "n_obs": self.n_obs,
        }

    @classmethod
    def from_state(cls, st, contact_activation=None):
        cfg = st["config"]
        if cfg["had_contact_activation"] and contact_activation is None:
            raise ValueError("This model was trained with a contact_activation; "
                             "pass the same callable to load_model(..., contact_activation=...).")
        obj = cls(contact_activation=contact_activation, nu=cfg["nu"],
                  feature_scale=cfg["feature_scale"], noise_contact_mult=cfg["noise_contact_mult"],
                  forgetting=cfg["forgetting"], jitter=cfg["jitter"])
        obj.ell = st["hyper"]["ell"]; obj.sf2 = st["hyper"]["sf2"]; obj.sn2_base = st["hyper"]["sn2_base"]
        obj.x_mean = st["stdz"]["x_mean"]; obj.x_std = st["stdz"]["x_std"]; obj.y_mean = st["stdz"]["y_mean"]
        obj.Z = st["Z"]; obj.n_obs = st.get("n_obs", 0)
        obj._compute_kernel_cache()                        # rebuild Kzz/Kzz_inv WITHOUT resetting posterior
        obj.m_u = st["m_u"]; obj.S_u = st["S_u"]           # restore the LEARNED posterior
        return obj


# ============================================================================ #
# Multi-output wrapper: one independent GP per residual channel.
# ============================================================================ #
class MultiOutputResidualGP:
    def __init__(self, output_names=("v_dot", "omega_dot"), **gp_kwargs):
        self.output_names = list(output_names)
        self._gp_kwargs = dict(gp_kwargs)
        self.gps = [OnlineSparseGP(**gp_kwargs) for _ in self.output_names]

    def fit(self, X, Y, n_inducing=80):
        Y = np.asarray(Y, float)
        for j, gp in enumerate(self.gps):
            gp.fit(X, Y[:, j], n_inducing=n_inducing)
        return self

    def update(self, X, Y):
        Y = np.asarray(Y, float)
        for j, gp in enumerate(self.gps):
            gp.update(X, Y[:, j])
        return self

    def predict(self, X, return_std=False):
        means, vars_ = [], []
        for gp in self.gps:
            m, v = gp.predict(X)
            means.append(m); vars_.append(v)
        M = np.stack(means, axis=1)
        V = np.stack(vars_, axis=1)
        return (M, np.sqrt(V)) if return_std else (M, V)

    @property
    def lengthscales(self):
        return {nm: gp.ell for nm, gp in zip(self.output_names, self.gps)}

    @property
    def n_obs(self):
        return self.gps[0].n_obs if self.gps else 0

    # ---- persistence (whole model, one file) ------------------------------
    def save(self, path):
        state = {"format_version": _FORMAT_VERSION,
                 "output_names": self.output_names,
                 "gp_kwargs": {k: v for k, v in self._gp_kwargs.items() if k != "contact_activation"},
                 "channels": [gp.get_state() for gp in self.gps]}
        tmp = path + ".tmp"                                # atomic write (safe if interrupted mid-save)
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path, contact_activation=None):
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.output_names = state["output_names"]
        obj._gp_kwargs = state.get("gp_kwargs", {})
        obj.gps = [OnlineSparseGP.from_state(ch, contact_activation=contact_activation)
                   for ch in state["channels"]]
        return obj


# ---- convenience module-level functions (what you call from the RL loop) -------
def model_exists(path):
    return os.path.exists(path)


def save_model(model, path):
    return model.save(path)


def load_model(path, contact_activation=None):
    return MultiOutputResidualGP.load(path, contact_activation=contact_activation)


def load_or_create(path, output_names=("v_dot", "omega_dot"), contact_activation=None, **gp_kwargs):
    """Return (model, is_new). If a saved model exists, load it (is_new=False);
    otherwise construct a fresh one that still needs .fit() on a warmup batch (is_new=True)."""
    if model_exists(path):
        return load_model(path, contact_activation=contact_activation), False
    return MultiOutputResidualGP(output_names=output_names,
                                 contact_activation=contact_activation, **gp_kwargs), True


# ============================================================================ #
# Demo: the episodic MBRL loop with persistence, showing the model IMPROVE over
# episodes across save/load, plus a save/load integrity check.
# State columns: [x, pitch, omega, v, torque].  Delete to use as a library.
# ============================================================================ #
# if __name__ == "__main__":
#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     rng = np.random.default_rng(0)
#     IX, IP, IW, IV, IT = 0, 1, 2, 3, 4
#     X_OBS, OBS_W, V_REF = 3.0, 0.25, 2.0

#     def sample_states(n):
#         X = np.empty((n, 5))
#         X[:, IX] = rng.uniform(0, 6, n); X[:, IP] = rng.uniform(-0.3, 0.3, n)
#         X[:, IW] = rng.uniform(-2, 2, n); X[:, IV] = rng.uniform(0.5, 4, n)
#         X[:, IT] = rng.uniform(-5, 5, n)
#         return X

#     def true_residual(X):
#         contact = np.exp(-((X[:, IX] - X_OBS) ** 2) / (2 * OBS_W ** 2))
#         speed = X[:, IV] / V_REF
#         return np.stack([-3.0 * contact * speed + 0.2 * np.sin(2 * X[:, IP]),
#                          6.0 * contact * speed], axis=1)

#     def gen(n):
#         X = sample_states(n)
#         return X, true_residual(X) + rng.normal(0, [0.03, 0.05], size=(n, 2))

#     PATH = "residual_model.pkl"
#     if model_exists(PATH):
#         os.remove(PATH)                                    # clean start for the demo

#     Xtest, Ytest = gen(800)                                # fixed held-out test set
#     def test_rmse(m):
#         P = m.predict(Xtest)[0]
#         return np.sqrt(((P - Ytest) ** 2).mean(0))

#     # ---- THE LOOP: each episode loads the existing model, updates it, saves it ----
#     N_EP = 1000
#     hist = []
#     for ep in range(1, N_EP + 1):
#         Xe, Ye = gen(400 if ep == 1 else 150)
#         model, is_new = load_or_create(PATH, output_names=["v_dot", "omega_dot"], nu=1.5)
#         if is_new:
#             model.fit(Xe, Ye, n_inducing=80)               # first episode: build structure + absorb
#             tag = "fit (new model)"
#         else:
#             model.update(Xe, Ye)                           # later episodes: recursive refine
#             tag = "loaded + updated"
#         save_model(model, PATH)
#         r = test_rmse(model); hist.append(r)
#         print(f"episode {ep}: {tag:18s} test RMSE  v_dot={r[0]:.4f}  omega_dot={r[1]:.4f}  n_obs={model.n_obs}")

#     # ---- save/load integrity check ----
#     before = model.predict(Xtest)[0]
#     reloaded = load_model(PATH)
#     after = reloaded.predict(Xtest)[0]
#     print(f"\nsave/load max prediction diff: {np.max(np.abs(before - after)):.2e}  (should be ~0)")
#     reloaded.update(*gen(150))
#     print(f"reloaded model accepted further updates, n_obs now = {reloaded.n_obs}")

#     # ---- plot improvement over episodes ----
#     hist = np.array(hist)
#     fig, ax = plt.subplots(figsize=(20.0, 12.0))
#     ax.plot(range(1, N_EP + 1), hist[:, 0], "o-", label="v_dot RMSE")
#     ax.plot(range(1, N_EP + 1), hist[:, 1], "s-", label="omega_dot RMSE")
#     ax.set_xlabel("episode"); ax.set_ylabel("held-out RMSE")
#     ax.set_title("Model improves over episodes (persisted via save/load each episode)")
#     ax.legend(); ax.grid(alpha=0.3)
#     fig.tight_layout(); fig.savefig("mbrl_persistence_demo.png", dpi=130)
#     print("saved mbrl_persistence_demo.png")