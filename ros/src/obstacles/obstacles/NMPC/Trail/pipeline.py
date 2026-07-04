#!/usr/bin/env python3
"""
Emergent wheelie-over-obstacle: MPPI + GP residual, corrected pipeline.
No pitch reference anywhere. The wheelie must emerge from the GP learning
(a) forward progress is blocked at the wall when flat, and
(b) pressing the wall pitches the truck up (the wall is the fulcrum).

Fixes vs the original stack:
  1. Nominal model CALIBRATED to this MuJoCo plant:
       v_dot = 2.089*tau*cos(th) - 0.215*v      (measured, was 1.494/0.0119)
       omega_dot = 0 while grounded             (measured: no flat-ground wheelie at tau<=6;
                                                 the old lever model predicted -6.7 rad/s^2)
       airborne: gravity pendulum + reaction couple, GP corrects.
  2. Residuals paired with the state at the START of the control interval.
  3. Outliers SKIPPED, not held; contact impulses gated by residual magnitude.
  4. Flip barrier in the cost (soft wall at -55 deg); goal cost unchanged otherwise.
  5. Unwrapped pitch (joint) fed to the controller; episode ends at |pitch|>=80 deg (fail).
  6. Exact GP (ARD Matern-3/2, MLL fit) instead of SVGP -- same kernel-sum rollout contract.
"""
import numpy as np, mujoco, math, time
from scipy.optimize import minimize

rng = np.random.default_rng(0)
S3 = math.sqrt(3.0)

# ----------------------------- calibrated nominal ---------------------------
KV, CV = 2.089, 0.215
M_, G_ = 5.064, 9.81
RHO   = float(np.hypot(0.1892, 0.1605 - 0.081))
ALPHA = float(np.arctan2(0.1605 - 0.081, 0.1892))
I_AIR = 0.46                      # airborne effective inertia (from earlier RLS work)
C_W   = 0.3
GROUND_EPS = 0.03                 # |theta| below this = grounded

def nominal(s, tau):
    """s: (K,4) [x,v,th,om]; returns (v_dot, om_dot) nominal. Vectorized."""
    v, th, om = s[:, 1], s[:, 2], s[:, 3]
    v_dot = KV * tau * np.cos(th) - CV * v
    beta = ALPHA - th
    om_air = (M_ * G_ * RHO * np.cos(beta) - tau) / I_AIR - C_W * om
    om_dot = np.where(th >= -GROUND_EPS, 0.0, om_air)
    return v_dot, om_dot

def nominal1(s, tau):
    v_dot, om_dot = nominal(s.reshape(1, 4), np.array([tau]))
    return float(v_dot[0]), float(om_dot[0])

# ----------------------------- exact GP residual ----------------------------
class ExactGP:
    """One-channel exact GP, ARD Matern-3/2, hyperparams by MLL (analytic grads)."""
    def __init__(self, d=5):
        self.d, self.ready = d, False

    def fit(self, X, y):
        self.xm, self.xs = X.mean(0), X.std(0); self.xs[self.xs < 1e-6] = 1.0
        Xs = (X - self.xm) / self.xs
        self.ym = 0.0; yc = y - self.ym   # calibrated nominal => zero-mean residual prior
        n, d = Xs.shape
        def unpack(p):
            return np.exp(p[:d]), math.exp(p[d]), math.exp(p[d+1])   # ell, sf2, sn2
        def K_and_parts(ell, sf2):
            Xw = Xs / ell
            r2 = np.maximum(((Xw[:, None, :] - Xw[None, :, :])**2).sum(-1), 0.0)
            r = np.sqrt(r2 + 1e-12); E = np.exp(-S3 * r)
            K = sf2 * (1.0 + S3 * r) * E
            return K, E, Xw
        def negmll(p):
            ell, sf2, sn2 = unpack(p)
            K, E, Xw = K_and_parts(ell, sf2)
            Kn = K + sn2 * np.eye(n)
            try: L = np.linalg.cholesky(Kn)
            except np.linalg.LinAlgError: return 1e10, np.zeros_like(p)
            a = np.linalg.solve(L.T, np.linalg.solve(L, yc))
            nll = 0.5 * yc @ a + np.log(np.diag(L)).sum() + 0.5 * n * math.log(2*math.pi)
            Ki = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
            W = np.outer(a, a) - Ki                                  # dMLL = 0.5 tr(W dK)
            g = np.zeros_like(p)
            for j in range(d):                                       # d/dlog ell_j
                D2 = (Xw[:, None, j] - Xw[None, :, j])**2
                g[j] = -0.5 * (W * (3.0 * sf2 * E * D2)).sum()
            g[d]   = -0.5 * (W * K).sum()                            # d/dlog sf2
            g[d+1] = -0.5 * np.trace(W) * sn2                        # d/dlog sn2
            return float(nll), g
        lv, ln = math.log(max(yc.var(), 1e-3)), math.log(max(0.5*yc.var(), 1e-3))
        inits = [np.concatenate([np.zeros(d), [lv], [ln]]),
                 np.concatenate([[math.log(0.3)], np.zeros(d-1), [lv], [ln]]),   # x-localized
                 np.concatenate([np.full(d, math.log(0.5)), [lv], [ln]])]
        ub = [8.0, 8.0, 8.0, 8.0, 1.2][:d]        # x,v,th,om get 8; tau capped at 1.2 std
        bounds = [(math.log(0.15), math.log(u)) for u in ub] + [(math.log(1e-3), math.log(500.0)),
                                                         (math.log(1e-4), math.log(100.0))]
        best = None
        for p0 in inits:
            r = minimize(negmll, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                         options={"maxiter": 120})
            if best is None or r.fun < best.fun: best = r
        res = best
        self.ell, self.sf2, self.sn2 = unpack(res.x)
        K, _, _ = K_and_parts(self.ell, self.sf2)
        self.Z = Xs
        self.alpha = np.linalg.solve(K + self.sn2 * np.eye(n), yc)
        self.Zw = (Xs / self.ell).astype(np.float32)
        self.z2 = (self.Zw**2).sum(1)
        self.al32 = self.alpha.astype(np.float32)
        self.ready = True

    def mean(self, X):
        """X: (K,d) raw -> (K,) posterior mean. float32 fast path."""
        if not self.ready: return np.zeros(X.shape[0], np.float32)
        Xw = ((X - self.xm) / self.xs / self.ell).astype(np.float32)
        r2 = np.maximum((Xw**2).sum(1)[:, None] + self.z2[None, :] - 2.0 * (Xw @ self.Zw.T), 0.0)
        r = np.sqrt(r2)
        k = self.sf2 * (1.0 + S3 * r) * np.exp(-S3 * r)
        return self.ym + k @ self.al32

class ResidualGP:
    def __init__(self, max_fit=220):
        self.X = np.empty((0, 5)); self.Y = np.empty((0, 2))
        self.bx, self.by = [], []
        self.gv, self.gw = ExactGP(), ExactGP()
        self.max_fit = max_fit
    @property
    def ready(self): return self.gv.ready
    def observe(self, z, rv, rw):
        self.bx.append(z); self.by.append([rv, rw])
    def end_episode(self):
        if self.bx:
            self.X = np.vstack([self.X, np.array(self.bx)])
            self.Y = np.vstack([self.Y, np.array(self.by)])
            self.bx, self.by = [], []
        n = self.X.shape[0]
        if n < 100: return None
        # stratified subsample: keep informative points (near wall / pitched / big residual)
        info = ((self.X[:, 0] > 1.1) & (self.X[:, 0] < 2.7)) | (self.X[:, 2] < -0.05) \
               | (np.abs(self.Y[:, 0]) > 2.0) | (np.abs(self.Y[:, 1]) > 2.0)
        idx_info = np.flatnonzero(info); idx_rest = np.flatnonzero(~info)
        n_info = min(len(idx_info), int(0.7 * self.max_fit))
        n_rest = min(len(idx_rest), self.max_fit - n_info)
        pick = np.concatenate([rng.choice(idx_info, n_info, replace=False) if n_info else [],
                               rng.choice(idx_rest, n_rest, replace=False) if n_rest else []]).astype(int)
        Xf, Yf = self.X[pick], self.Y[pick]
        self.gv.fit(Xf, Yf[:, 0]); self.gw.fit(Xf, Yf[:, 1])
        return dict(n=n, nfit=len(pick),
                    ell_v=self.gv.ell, ym_v=self.gv.ym, sf2_v=self.gv.sf2,
                    ell_w=self.gw.ell, ym_w=self.gw.ym, sf2_w=self.gw.sf2)
    def mean(self, Xq):
        return self.gv.mean(Xq), self.gw.mean(Xq)

# ----------------------------- MPPI ------------------------------------------
class MPPI:
    N, K, dt = 24, 1024, 0.05
    SIGMA, BETA, LAM = 2.0, 0.75, 60.0
    TAU_MIN, TAU_MAX = -6.0, 6.0
    Q_X, Q_TH, R_DTAU, Q_TERM = 5.0, 2.0, 5.0, 10.0
    V_CAP, Q_V = 2.5, 30.0
    TH_BAR, Q_BAR = math.radians(58.0), 6000.0
    LEAD = 0.08
    LOOKAHEAD = 5.0

    def __init__(self, gp, seed=0):
        self.gp = gp
        self.rng = np.random.default_rng(seed)
        self.U = None
        self.last_ess = 0.0

    def _deriv(self, s, tau):
        v_dot, om_dot = nominal(s, tau)
        if self.gp.ready:
            Xq = np.concatenate([s, tau[:, None]], 1)
            rv, rw = self.gp.mean(Xq)
            v_dot = v_dot + rv; om_dot = om_dot + rw
        return np.stack([s[:, 1], v_dot, s[:, 3], om_dot], 1)

    def _step(self, s, tau):
        s = s + self.dt * self._deriv(s, tau)
        landing = (s[:, 2] >= 0.0) & (s[:, 3] > 0.0)
        s[:, 3] = np.where(landing, 0.0, np.clip(s[:, 3], -15, 15))
        s[:, 1] = np.clip(s[:, 1], -8, 8)
        s[:, 2] = np.clip(s[:, 2], -1.6, 0.0)
        return s

    def solve(self, state, goal_x, tau_prev):
        N, K = self.N, self.K
        if self.U is None:
            self.U = np.full(N, tau_prev)
        else:
            self.U = np.concatenate([self.U[1:], self.U[-1:]])
        # AR(1)-correlated exploration noise
        e = self.rng.standard_normal((K, N)).astype(np.float64)
        eps = np.empty_like(e); eps[:, 0] = e[:, 0]
        for n in range(1, N):
            eps[:, n] = self.BETA * eps[:, n-1] + math.sqrt(1-self.BETA**2) * e[:, n]
        U = np.clip(self.U[None, :] + self.SIGMA * eps, self.TAU_MIN, self.TAU_MAX)
        eps = U - self.U[None, :]

        ref_x = min(goal_x, state[0] + self.LOOKAHEAD)
        s = np.repeat(state.reshape(1, 4), K, 0)
        J = np.zeros(K); prev = np.full(K, tau_prev)
        for n in range(N):
            tau = U[:, n]
            J += self.Q_X * (s[:, 0] - ref_x)**2
            J += self.Q_TH * s[:, 2]**2
            J += self.R_DTAU * (tau - prev)**2
            J += self.Q_V * np.maximum(0.0, s[:, 1] - self.V_CAP)**2
            th_eff = s[:, 2] + self.LEAD * np.minimum(s[:, 3], 0.0)
            J += self.Q_BAR * np.maximum(0.0, -th_eff - self.TH_BAR)**2
            J += 40.0 * np.maximum(0.0, -s[:, 3] - 6.0)**2
            s = self._step(s, tau); prev = tau
        th_eff = s[:, 2] + self.LEAD * np.minimum(s[:, 3], 0.0)
        J += self.Q_TERM * (s[:, 0] - ref_x)**2 + 3*self.Q_BAR * np.maximum(0.0, -th_eff - self.TH_BAR)**2
        J = np.nan_to_num(J, nan=1e12, posinf=1e12)
        w = np.exp(-(J - J.min()) / self.LAM); w /= w.sum()
        self.last_ess = 1.0 / (w**2).sum()
        self.U = np.clip(self.U + w @ eps, self.TAU_MIN, self.TAU_MAX)
        return float(self.U[0])

# ----------------------------- environment ----------------------------------
class Env:
    CTRL_DT, GOAL_X, T_MAX, FLIP_DEG = 0.05, 10.0, 20.0, 80.0
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path('/mnt/user-data/uploads/monster_truck_flip_2d.xml')
        self.data = mujoco.MjData(self.model)
        j = lambda n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.qx, self.vx = self.model.jnt_qposadr[j('root_x')], self.model.jnt_dofadr[j('root_x')]
        self.qz = self.model.jnt_qposadr[j('root_z')]
        self.qp, self.vp = self.model.jnt_qposadr[j('root_pitch')], self.model.jnt_dofadr[j('root_pitch')]
        self.sub = round(self.CTRL_DT / self.model.opt.timestep)
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qz] = 0.1512
        mujoco.mj_forward(self.model, self.data)
        for _ in range(100): mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        return self.state()
    def state(self):
        d = self.data
        return np.array([d.qpos[self.qx], d.qvel[self.vx], d.qpos[self.qp], d.qvel[self.vp]])
    def step(self, tau):
        self.data.ctrl[0] = tau
        for _ in range(self.sub): mujoco.mj_step(self.model, self.data)
        return self.state()

# ----------------------------- episode loop ----------------------------------
def run_episode(env, gp, ctrl, learn=True, log=None):
    s = env.reset(); ctrl.U = None
    tau_prev, peak = 0.0, 0.0
    t0 = time.time(); static_k = 0
    while env.data.time < env.T_MAX:
        tau = ctrl.solve(s, env.GOAL_X, tau_prev)
        s_prev = s.copy()
        s = env.step(tau)
        if log is not None:
            log.append((env.data.time, *s, tau))
        peak = max(peak, abs(math.degrees(s[2])))
        if learn:
            dv = (s[1] - s_prev[1]) / env.CTRL_DT
            dw = (s[3] - s_prev[3]) / env.CTRL_DT
            nv, nw = nominal1(s_prev, tau)
            rv, rw = dv - nv, dw - nw
            static = abs(s_prev[1]) < 0.05 and abs(s_prev[3]) < 0.05
            static_k = static_k + 1 if static else 0
            if (abs(s_prev[2]) < 1.2 and abs(s_prev[3]) < 12 and abs(rv) < 25 and abs(rw) < 50
                    and (not static or static_k % 5 == 0)):
                gp.observe(np.array([*s_prev, tau]), rv, rw)
        tau_prev = tau
        if abs(math.degrees(s[2])) >= env.FLIP_DEG:
            return dict(success=False, mode='FLIP', x=s[0], t=env.data.time, peak=peak, wall=time.time()-t0)
        if s[0] >= env.GOAL_X - 0.15:
            return dict(success=peak < env.FLIP_DEG, mode='GOAL', x=s[0], t=env.data.time, peak=peak, wall=time.time()-t0)
    return dict(success=False, mode='TIMEOUT', x=s[0], t=env.data.time, peak=peak, wall=time.time()-t0)

if __name__ == "__main__":
    env, gp = Env(), ResidualGP()
    ctrl = MPPI(gp, seed=0)
    for ep in range(12):
        r = run_episode(env, gp, ctrl)
        f = gp.end_episode()
        line = (f"ep {ep:2d}: {r['mode']:7s} x={r['x']:5.2f} t={r['t']:5.2f}s "
                f"peak={r['peak']:5.1f}deg ess={ctrl.last_ess:5.0f} [{r['wall']:4.1f}s wall]")
        if f: line += (f"\n        GP n={f['n']:4d} ym_v={f['ym_v']:+5.2f} ym_w={f['ym_w']:+5.2f}"
                       f" ell_v={np.round(f['ell_v'],2)} ell_w={np.round(f['ell_w'],2)}")
        print(line, flush=True)