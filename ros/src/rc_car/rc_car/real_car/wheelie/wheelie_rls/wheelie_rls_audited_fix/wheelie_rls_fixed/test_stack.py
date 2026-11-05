#!/usr/bin/env python3
"""Offline numerical tests. No ROS connection and no commands are published."""
import math
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from rls import rls_features_numpy, rls_dynamics_torch, rls_step_torch, rls_update
from mppi import MPPITorch
from params_mppi import WheelieParams, MPPIConfig
from train_rls import fit_rls, load_experiments


class StackTests(unittest.TestCase):
    def setUp(self):
        self.p = WheelieParams()
        self.cfg = replace(MPPIConfig(), K=128, N=12)
        self.a = np.zeros(18)
        self.a[[0, 1, 7, 8, 11]] = [2.0, -0.5, -100.0, -2.0, 20.0]

    def test_numpy_torch_feature_parity(self):
        rng = np.random.default_rng(42)
        state = rng.normal(size=(50, 4))
        u, a = rng.uniform(0, 0.3, 50), rng.normal(size=18)
        fv, fw = rls_features_numpy(state, u)
        actual = rls_dynamics_torch(torch.tensor(state), torch.tensor(u), torch.tensor(a)).numpy()
        np.testing.assert_allclose(actual[:, 1], fv @ a[:7], atol=1e-12)
        np.testing.assert_allclose(actual[:, 3], fw @ a[7:], atol=1e-12)
        column = rls_dynamics_torch(torch.tensor(state), torch.tensor(u[:, None]), torch.tensor(a))
        np.testing.assert_allclose(column.numpy(), actual, atol=1e-12)

    def test_rls_matches_batch_ridge(self):
        rng = np.random.default_rng(7)
        f1, f2 = rng.normal(size=(200, 7)), rng.normal(size=(200, 11))
        y = rng.normal(size=(200, 2))
        a, P = np.zeros(18), np.eye(18)/3.0
        for i in range(len(y)):
            a, P, _ = rls_update(np.zeros(4), 0.0, *y[i], a, P, features=(f1[i], f2[i]))
        expected = np.r_[np.linalg.solve(f1.T@f1+3*np.eye(7), f1.T@y[:, 0]), np.linalg.solve(f2.T@f2+3*np.eye(11), f2.T@y[:, 1])]
        np.testing.assert_allclose(a, expected, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(P).min(), 0.0)

    def test_prior_prediction_not_posterior(self):
        a, P, info = rls_update(np.zeros(4), 0.2, 4.0, -3.0, np.zeros(18), np.eye(18)*100)
        self.assertEqual(info['v_dot_hat'], 0.0)
        self.assertNotEqual(info['v_dot_hat_post'], 0.0)
        self.assertEqual(info['v_dot_error'], 4.0)

    def test_step_uses_requested_dt(self):
        a = torch.zeros(18, dtype=torch.float64)
        a[0], a[7] = 2.0, -10.0
        state = torch.zeros((1, 4), dtype=torch.float64)
        actual = rls_step_torch(state, torch.tensor([0.2], dtype=torch.float64), 0.04, a)
        np.testing.assert_allclose(actual.numpy()[0], [0.00032, 0.016, -0.0016, -0.08], atol=1e-12)

    def test_ground_contact(self):
        state = torch.tensor([[0.0, 0.0, -0.001, 1.0]], dtype=torch.float64)
        actual = rls_step_torch(state, torch.zeros(1), 0.04, torch.zeros(18))
        self.assertEqual(actual[0, 2].item(), 0.0)
        self.assertEqual(actual[0, 3].item(), 0.0)

    def test_invalid_rollouts_not_rescued(self):
        a = self.a.copy()
        a[6] = 1e6
        ctrl = MPPITorch(self.p, self.cfg, a, device='cpu')
        u, info = ctrl.solve(np.zeros(4), [0, 0, -0.5, 0], 0.0, [0, 0, 0])
        self.assertFalse(info['success'])
        self.assertEqual(info['reason'], 'no_feasible_rollout')
        self.assertEqual(u, 0.0)

    def test_nonfinite_measurement(self):
        ctrl = MPPITorch(self.p, self.cfg, self.a, device='cpu')
        u, info = ctrl.solve([0, 0, np.nan, 0], [0, 0, -0.5, 0], 0.0, [0, 0, 0])
        self.assertFalse(info['success'])
        self.assertEqual(u, 0.0)

    def test_delay_and_returned_plan_consistency(self):
        torch.manual_seed(4)
        ctrl = MPPITorch(self.p, self.cfg, self.a, device='cpu', fast=False)
        state, ref, history = np.array([0, 0, -0.1, 0]), np.array([0, 0, -0.5, 0]), [0.1, 0.2, 0.3]
        u, info = ctrl.solve(state, ref, 0.3, history)
        self.assertTrue(info['success'])
        self.assertAlmostEqual(u, ctrl.last_solution[0].item())
        costs, trajectories, valid = ctrl._rollout(torch.tensor(state, dtype=ctrl.dtype), ctrl.last_solution[None], torch.tensor(ref, dtype=ctrl.dtype), 0.3, torch.tensor(history, dtype=ctrl.dtype))
        self.assertTrue(valid[0].item())
        np.testing.assert_allclose(ctrl.last_prediction, trajectories[0].numpy(), atol=1e-10)
        self.assertAlmostEqual(info['cost'], costs[0].item(), places=8)
        alternatives = torch.stack((torch.zeros(self.cfg.N), torch.full((self.cfg.N,), 0.3))).to(ctrl.dtype)
        _, trajectories, _ = ctrl._rollout(torch.tensor(state, dtype=ctrl.dtype), alternatives, torch.tensor(ref, dtype=ctrl.dtype), 0.3, torch.tensor(history, dtype=ctrl.dtype))
        np.testing.assert_allclose(trajectories[0, :4].numpy(), trajectories[1, :4].numpy(), atol=1e-12)

    def test_float32_command_bound_roundoff(self):
        ctrl = MPPITorch(self.p, self.cfg, self.a, device='cpu')
        upper = float(np.float32(self.p.u_max))
        u, info = ctrl.solve(np.zeros(4), [0, 0, -0.5, 0], upper, [upper]*3)
        self.assertTrue(info['success'])
        self.assertLessEqual(u, self.p.u_max)
        self.assertLessEqual(info['cost'], info['best_cost']+1e-6)

    def test_frozen_coefficients(self):
        ctrl = MPPITorch(self.p, self.cfg, self.a, device='cpu')
        before = ctrl.rls_parameters.clone()
        for _ in range(3):
            ctrl.solve(np.zeros(4), [0, 0, -0.5, 0], 0.0, [0, 0, 0])
        self.assertTrue(torch.equal(before, ctrl.rls_parameters))

    def test_planar_pitch_yaw_invariance_and_unwrapping(self):
        from scipy.spatial.transform import Rotation
        for yaw in [0, 0.8, math.pi, -math.pi/2]:
            for pitch in [-0.1, -0.9, -1.4]:
                x, y, z, w = Rotation.from_euler('ZYX', [yaw, pitch, 0.0]).as_quat()
                result = math.atan2(2*(w*y-x*z), 1-2*(x*x+y*y))
                self.assertAlmostEqual(result, pitch, places=10)
        wrapped_old, wrapped_new = 3.13, -3.13
        delta = math.atan2(math.sin(wrapped_new-wrapped_old), math.cos(wrapped_new-wrapped_old))
        self.assertLess(abs(delta), 0.03)


if __name__ == '__main__':
    unittest.main(verbosity=2)
