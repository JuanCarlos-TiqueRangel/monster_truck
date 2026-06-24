"""
================================================================================
gp_kernel.py  --  the ONE place the GP residual's closed-form mean lives
================================================================================

The trained GPyTorch model can't be called from inside the controllers (numba MPPI,
casadi NMPC). So they re-evaluate the GP posterior MEAN as a kernel sum over the M
inducing points:

    z_std = (z - x_mean) / x_std
    mu(z) = y_mean + sum_j  w_j * k(z_std, Z_j)
    k(a, b) = sf2 * kernel_shape( r ),   r = || (a - b) / ell ||   (ARD-scaled distance)

This module owns that math so it is NOT duplicated in mppi.py / GP.py:
  * `gp_residual_mean`  -- numba scalar version, imported by the MPPI rollout.
  * `gp_mean_np`        -- numpy vectorized version, used by GP.predict() (diagnostics).
  * `casadi_gp_mean`    -- symbolic builder for the IPOPT NMPC (casadi imported lazily).
The three are different EXECUTION BACKENDS of the same formula (a scalar can't be a
casadi graph), so each writes the kernel shape once -- but they sit side by side here
and MUST stay in sync. `_KERNEL_SHAPES` below documents all of them in one place.

SWAPPABLE KERNEL. `kid` (an int id) selects the kernel SHAPE so you can change the
GPyTorch kernel and have the controllers follow:
    0 = Matern-1/2 (exp(-r))            -- roughest, non-differentiable at r=0
    1 = Matern-3/2 ((1+c)e^-c, c=v3 r)  -- default
    2 = Matern-5/2 ((1+c+c^2/3)e^-c, c=v5 r) -- smoother (nicer NMPC gradients)
    3 = RBF        (exp(-r^2/2))        -- smoothest
`make_gpytorch_kernel(name, d)` builds the matching GPyTorch kernel; `detect_kernel_id`
reads it back from a model so the export auto-tracks whatever kernel the model uses.
================================================================================
"""

import math
import numpy as np

# numba is only present where the controllers run (docker). Make it OPTIONAL so this
# module still IMPORTS on the host (for GP.py's self-test / predict): there njit just
# becomes a no-op decorator and the functions run as plain (slow) Python.
try:
    from numba import njit
except ImportError:                                   # pragma: no cover
    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]
        def _deco(f):
            return f
        return _deco

# ---- kernel ids (the single naming authority) --------------------------------
KERNEL_IDS = {"matern12": 0, "matern32": 1, "matern52": 2, "rbf": 3}
ID_TO_NAME = {v: k for k, v in KERNEL_IDS.items()}
DEFAULT_KERNEL = "matern32"

_SQRT3 = 1.7320508075688772
_SQRT5 = 2.23606797749979

# Human-readable record of every kernel shape, k(r)/sf2, kept next to the code so the
# numba / numpy / casadi copies below can be checked against one another:
#   matern12: exp(-r)        matern32: (1+v3 r) exp(-v3 r)
#   matern52: (1+v5 r + 5r^2/3) exp(-v5 r)        rbf: exp(-r^2/2)
_KERNEL_SHAPES = ("matern12", "matern32", "matern52", "rbf")


def kernel_id_from_name(name) -> int:
    """'matern32' -> 1.  Accepts an int id straight through too."""
    if isinstance(name, (int, np.integer)):
        return int(name)
    key = str(name).lower().replace("-", "").replace("_", "").replace("/", "")
    key = {"matern0.5": "matern12", "matern1.5": "matern32",
           "matern2.5": "matern52", "se": "rbf", "squaredexp": "rbf"}.get(key, key)
    if key not in KERNEL_IDS:
        raise ValueError(f"unknown kernel {name!r}; options: {list(KERNEL_IDS)}")
    return KERNEL_IDS[key]


def detect_kernel_id(covar_module) -> int:
    """Read the kernel id back from a (GPyTorch) covar module -- so the export tracks
    whatever kernel the model was built with (the 'edit one line' story)."""
    base = getattr(covar_module, "base_kernel", covar_module)
    cls = type(base).__name__
    if cls == "RBFKernel":
        return KERNEL_IDS["rbf"]
    if cls == "MaternKernel":
        nu = float(getattr(base, "nu", 1.5))
        return {0.5: 0, 1.5: 1, 2.5: 2}.get(nu, 1)
    raise ValueError(f"unsupported kernel {cls}; add it to gp_kernel.detect_kernel_id "
                     f"and the kernel-shape branches.")


def make_gpytorch_kernel(name, d):
    """Factory for the model side. The kernel choice is literally one of these lines --
    add a branch (and a kernel id) to support a new kernel everywhere at once."""
    import gpytorch
    kid = kernel_id_from_name(name)
    if kid == KERNEL_IDS["rbf"]:
        base = gpytorch.kernels.RBFKernel(ard_num_dims=d)
    else:
        nu = {0: 0.5, 1: 1.5, 2: 2.5}[kid]
        base = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=d)
    return gpytorch.kernels.ScaleKernel(base)


# ----------------------------------------------------------------------------- #
# (1) numba scalar shape + mean  -- the controllers' hot path (MPPI rollout).
# ----------------------------------------------------------------------------- #
@njit(fastmath=True, cache=True)
def _k(r2, kid, sf2):
    """sf2 * kernel_shape(r), r = sqrt(r2). r2 = squared ARD-scaled distance."""
    if kid == 3:                                       # RBF: exp(-r^2/2), no sqrt
        return sf2 * math.exp(-0.5 * r2)
    r = math.sqrt(r2 + 1e-9)                           # +eps smooths the sqrt kink
    if kid == 0:                                       # Matern-1/2
        return sf2 * math.exp(-r)
    if kid == 1:                                       # Matern-3/2
        c = _SQRT3 * r
        return sf2 * (1.0 + c) * math.exp(-c)
    c = _SQRT5 * r                                     # Matern-5/2
    return sf2 * (1.0 + c + c * c / 3.0) * math.exp(-c)


@njit(fastmath=True, cache=True)
def gp_residual_mean(x, v, th, om, tau, Z, w, x_mean, x_std, ell, sf2, y_mean, kid):
    """GP posterior mean of ONE residual channel at z=[x,v,theta,omega,tau].
    Z (M,5) standardized inducing points, w (M,) dual weights, ell (5,) ARD lengthscales.
    This is what mppi.py imports -- the kernel math lives here, not in the controller."""
    z0 = (x - x_mean[0]) / x_std[0]
    z1 = (v - x_mean[1]) / x_std[1]
    z2 = (th - x_mean[2]) / x_std[2]
    z3 = (om - x_mean[3]) / x_std[3]
    z4 = (tau - x_mean[4]) / x_std[4]
    y = y_mean
    for j in range(Z.shape[0]):
        d0 = (z0 - Z[j, 0]) / ell[0]
        d1 = (z1 - Z[j, 1]) / ell[1]
        d2 = (z2 - Z[j, 2]) / ell[2]
        d3 = (z3 - Z[j, 3]) / ell[3]
        d4 = (z4 - Z[j, 4]) / ell[4]
        r2 = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4
        y += w[j] * _k(r2, kid, sf2)
    return y


# ----------------------------------------------------------------------------- #
# (2) numpy vectorized mean  -- diagnostics (GP.predict), any feature dim d.
# ----------------------------------------------------------------------------- #
def gp_mean_np(z, Z, w, x_mean, x_std, ell, sf2, y_mean, kid):
    zc = (np.asarray(z, float).reshape(-1) - x_mean) / x_std       # (d,)
    dd = (zc[None, :] - Z) / ell[None, :]                          # (M, d)
    r2 = np.sum(dd * dd, axis=1)                                   # (M,)
    if kid == 3:
        k = sf2 * np.exp(-0.5 * r2)
    else:
        r = np.sqrt(r2 + 1e-9)
        if kid == 0:
            k = sf2 * np.exp(-r)
        elif kid == 1:
            c = _SQRT3 * r
            k = sf2 * (1.0 + c) * np.exp(-c)
        else:
            c = _SQRT5 * r
            k = sf2 * (1.0 + c + c * c / 3.0) * np.exp(-c)
    return float(y_mean + w @ k)


# ----------------------------------------------------------------------------- #
# (3) casadi symbolic mean  -- the IPOPT NMPC. casadi imported lazily so this module
#     still imports where casadi is absent (host). `kid` is a build-time Python int.
# ----------------------------------------------------------------------------- #
def casadi_gp_mean(z, Z, alpha, x_mean, x_std, ell, sf2, y_mean, kid):
    import casadi as ca
    M = Z.shape[0]
    zc = (z - x_mean) / x_std
    out = y_mean
    for i in range(M):
        diff = (zc - Z[i, :].T) / ell
        r2 = ca.dot(diff, diff)
        if kid == 3:                                  # RBF
            k = sf2 * ca.exp(-0.5 * r2)
        else:
            r = ca.sqrt(r2 + 1e-9)
            if kid == 0:                              # Matern-1/2
                k = sf2 * ca.exp(-r)
            elif kid == 1:                            # Matern-3/2
                c = _SQRT3 * r
                k = sf2 * (1.0 + c) * ca.exp(-c)
            else:                                     # Matern-5/2
                c = _SQRT5 * r
                k = sf2 * (1.0 + c + c * c / 3.0) * ca.exp(-c)
        out += alpha[i] * k
    return out
