"""Offline behavioral checks; does not import ROS or publish commands."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from mppi import MPPITorch
from params_mppi import WheelieParams, MPPIConfig


class MPPIChecks(unittest.TestCase):
    def make(self, **kwargs):
        cfg = MPPIConfig(K=32, N=3, **kwargs)
        return MPPITorch(WheelieParams(), cfg, np.zeros(12), device='cpu', integrator='rk4')

    def test_terminal_axes_and_weights(self):
        c = self.make()
        c.cfg.q_terminal_v = 2.0
        c.cfg.q_terminal_theta = 3.0
        c.cfg.q_terminal_omega = 7.0
        errors = torch.eye(4)
        expected = torch.tensor([0., 2., 3., 7.])
        torch.testing.assert_close(c._terminal_cost(errors, torch.zeros(4)), expected)

    def test_input_effort_is_in_running_cost(self):
        c = self.make()
        s, u = torch.zeros(1, 4), torch.tensor([0.1])
        self.assertAlmostEqual(c._stage_cost(s, u, u, torch.zeros(4)).item(),
                               c.cfg.r_tau * 0.1**2, places=8)

    def test_instance_config_and_nominal_candidate(self):
        c = self.make(SIGMA=0.012, LAM=0.7)
        c.cfg.K = 1
        _, info = c.solve(np.zeros(4), np.zeros(4), 0.05)
        self.assertTrue(info['success'])
        self.assertEqual(info['valid_rollouts'], 1)
        torch.testing.assert_close(c.last_solution, torch.full((3,), 0.05))
        self.assertEqual(torch.get_num_threads(), 1)

    def test_zero_model_does_not_fake_lift(self):
        c = self.make()
        c.solve(np.zeros(4), [0, 0, -0.8, 0], 0)
        s = torch.zeros(1, 4)
        for u in c.last_solution:
            s = c._step(s, u.reshape(1), c.cfg.dt)
        self.assertEqual(float(s[0, 2]), 0.0)

    def test_hold_samples_cover_sustained_bounded_commands(self):
        torch.manual_seed(11)
        c = self.make()
        c.cfg.K, c.cfg.N, c.cfg.hold_sample_fraction = 400, 7, .25
        nominal = torch.zeros(7)
        controls = c._sample_controls(nominal)
        self.assertEqual(tuple(controls.shape), (400, 7))
        self.assertTrue(bool((controls >= c.p.tau_min).all()))
        self.assertTrue(bool((controls <= c.p.tau_max).all()))
        torch.testing.assert_close(controls[0], nominal)
        held = controls[-100:]
        changes = (held[:, 1:] != held[:, :-1]).sum(dim=1)
        self.assertTrue(bool((changes <= 1).all()))
        self.assertTrue(bool((held[:, 0] > .08).any()))
        self.assertTrue(bool((held[:, 0] < -.08).any()))

    def test_configured_state_bounds_reject_rollouts(self):
        c = self.make()
        s = torch.tensor([[0., 5.1, -.5, 0.], [0., 0., -1.6, 0.],
                          [0., 0., -.5, 5.1], [0., 0., -.5, -5.1],
                          [0., 5., -.5, 0.]])
        result = c._step(s, torch.zeros(5), c.cfg.dt)
        self.assertTrue(bool(torch.isnan(result[:4]).all()))
        self.assertTrue(bool(torch.isfinite(result[4]).all()))

    def test_reject_bad_parameters_without_corrupting_current_model(self):
        c = self.make()
        with self.assertRaises(ValueError):
            c.set_rls_parameters(np.full(12, np.nan))
        torch.testing.assert_close(c.rls_parameters, torch.zeros(12))
        with self.assertRaises(ValueError):
            MPPITorch(WheelieParams(), MPPIConfig(), np.zeros(10), device='cpu')

    def test_invalid_state_returns_failure_and_resets_plan(self):
        c = self.make()
        c.solve(np.zeros(4), np.zeros(4), 0.0)
        self.assertIsNotNone(c.last_solution)
        u, info = c.solve([0, 0, math.nan, 0], np.zeros(4), 0.0)
        self.assertFalse(info['success'])
        self.assertEqual(u, 0.0)
        self.assertIsNone(c.last_solution)

    def test_overflow_cannot_be_hidden_by_state_clamps(self):
        c = self.make()
        a = np.zeros(12)
        a[3] = np.finfo(np.float32).max / 2.0
        c.set_rls_parameters(a)
        u, info = c.solve([0, 2, 0, 0], np.zeros(4), 0)
        self.assertFalse(info['success'])
        self.assertEqual(u, 0.0)
        self.assertEqual(info['valid_rollouts'], 0)

    def test_invalid_individual_rollouts_are_excluded(self):
        class SomeInvalid(MPPITorch):
            def _step(self, s, tau, dt):
                result = s.clone()
                result[tau > 0.0] = math.nan
                return result
        torch.manual_seed(3)
        c = SomeInvalid(WheelieParams(), MPPIConfig(K=128, N=1), np.zeros(12), device='cpu')
        u, info = c.solve(np.zeros(4), np.zeros(4), 0)
        self.assertTrue(info['success'])
        self.assertGreater(info['valid_rollouts'], 0)
        self.assertLess(info['valid_rollouts'], 128)
        self.assertLess(u, 0)

    def test_infeasible_average_falls_back_to_feasible_sample(self):
        class TwoFeasibleModes(MPPITorch):
            def _sample_controls(self, nominal):
                return torch.tensor([[-.08], [.08]], dtype=self.dtype, device=self.device)
            def _step(self, s, tau, dt):
                result = s.clone()
                result[tau.abs() < .05] = math.nan
                return result
        c = TwoFeasibleModes(WheelieParams(), MPPIConfig(K=2, N=1),
                             np.zeros(12), device='cpu')
        u, info = c.solve(np.zeros(4), np.zeros(4), 0)
        self.assertTrue(info['success'])
        self.assertTrue(info['used_best_sample_fallback'])
        self.assertEqual(info['valid_rollouts'], 2)
        self.assertAlmostEqual(abs(u), .08, places=6)
        result = c._step(torch.zeros(1, 4), c.last_solution, c.cfg.dt)
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_tiny_positive_ground_measurement_is_projected(self):
        c = self.make()
        u, info = c.solve([0, 0, math.radians(.1), 0], np.zeros(4), 0)
        self.assertTrue(info['success'])

    def test_pitch_control_uses_correct_direction_for_synthetic_model(self):
        # This fixture checks signs only; it is not a proposed car calibration.
        a = np.zeros(12)
        a[10] = -30
        c = MPPITorch(WheelieParams(), MPPIConfig(K=1024), a,
                      device='cpu', integrator='rk4')
        torch.manual_seed(7)
        u, info = c.solve(np.zeros(4), [0, 0, -0.5, 0], 0)
        self.assertTrue(info['success'])
        self.assertGreater(u, 0)
        self.assertLessEqual(u, c.p.tau_max)
        s = torch.zeros(1, 4)
        for un in c.last_solution:
            s = c._step(s, un.reshape(1), c.cfg.dt)
        self.assertLess(s[0, 2].item(), -0.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
