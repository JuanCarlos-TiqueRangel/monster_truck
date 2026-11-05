"""Offline checks. This test never imports ROS or controls hardware."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MPLBACKEND', 'Agg')
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
import rls_clean as rls


class RLSCoreTests(unittest.TestCase):
    def setUp(self):
        self.a = np.array([5.,2.,-.9,-.3,.1,-1.2,-.5,.5,1.,-.4,-5.,-.4])
        self.P = np.eye(12)
        self.state = np.array([0.,1.,-.2,.1])

    def test_converges_known_acceleration_model(self):
        rng = np.random.default_rng(18)
        a, P = np.zeros(12), 1e6*np.eye(12)
        for _ in range(2500):
            state = np.array([0.,rng.uniform(-3,3),rng.uniform(-1.4,-.04),rng.uniform(-2,2)])
            tau = rng.uniform(-.114,.114)
            vdot = rls._phi_v(state,tau) @ self.a[:5]
            wdot = rls._phi_w(state,tau,vdot) @ self.a[5:]
            a,P,info = rls.rls_update(state,tau,vdot,wdot,a,P,
                                    forgetting_factor=1.,sigma_v_dot=.05,sigma_omega_dot=.05)
            self.assertFalse(info['skipped'])
        np.testing.assert_allclose(a,self.a,rtol=0,atol=.001)
        self.assertGreater(np.linalg.eigvalsh(P).min(),0)

    def test_angular_gate_preserves_coefficients(self):
        a,P,info = rls.rls_update(self.state,.1,2.,10.,self.a,self.P,update_angular=False)
        np.testing.assert_array_equal(a[5:],self.a[5:])
        self.assertFalse(np.array_equal(a[:5],self.a[:5]))
        expected = rls.rls_predict_acceleration(self.state,.1,self.a)
        self.assertEqual(info['omega_dot_hat'],expected[1])
        self.assertNotEqual(info['omega_dot_hat'],info['omega_dot_regression_hat'])

    def test_rejects_invalid_inputs_and_covariance(self):
        for bad in [np.full(12,np.nan), np.zeros(10)]:
            with self.assertRaises(ValueError):
                rls.rls_predict_acceleration(self.state,.1,bad)
        for bad in [np.zeros((12,12)), -np.eye(12), np.full((12,12),np.inf)]:
            with self.assertRaises(ValueError):
                rls.validate_rls_model(self.a,bad)
        asymmetric=np.eye(12); asymmetric[1,0]=.1
        with self.assertRaises(ValueError):
            rls.validate_rls_model(self.a,asymmetric)
        for kwargs in [dict(tau=np.nan),dict(vx_dot=np.nan),dict(forgetting_factor=0),dict(forgetting_factor=1.1),dict(sigma_v_dot=0)]:
            params=dict(state_prev=self.state,tau=.1,vx_dot=1.,omega_pitch_dot=2.,a=self.a,P=self.P)
            params.update(kwargs)
            with self.assertRaises(ValueError): rls.rls_update(**params)
        with self.assertRaises(ValueError):
            rls.rk4_step_numpy(self.state,.1,0,self.a)

    def test_numpy_torch_prediction_and_rk4_parity(self):
        rng=np.random.default_rng(20)
        states=rng.uniform([-1,-2,-1.3,-2],[1,2,.03,2],size=(250,4))
        commands=rng.uniform(-.114,.114,size=250)
        state_t=torch.tensor(states,dtype=torch.float64)
        tau_t=torch.tensor(commands,dtype=torch.float64)
        a_t=torch.tensor(self.a,dtype=torch.float64)
        numpy_d=np.array([rls.rls_dynamics_numpy(s,u,self.a) for s,u in zip(states,commands)])
        torch_d=rls.rls_dynamics_torch(state_t,tau_t,a_t).numpy()
        np.testing.assert_allclose(numpy_d,torch_d,rtol=1e-12,atol=1e-12)
        numpy_s=np.array([rls.rk4_step_numpy(s,u,.04,self.a) for s,u in zip(states,commands)])
        torch_s=rls.rk4_step_torch(state_t,tau_t,.04,a_t).numpy()
        np.testing.assert_allclose(numpy_s,torch_s,rtol=1e-12,atol=1e-12)

    def test_ground_contact_and_landing(self):
        a=np.zeros(12);a[-1]=1.
        self.assertEqual(rls.rls_predict_acceleration(np.zeros(4),0,a)[1],0)
        state=np.array([0.,0.,-.001,1.])
        actual=rls.rk4_step_numpy(state,0,.04,a)
        self.assertEqual(actual[2],0)
        self.assertEqual(actual[3],0)

    def test_model_metadata_and_atomic_storage(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.npz'
            a,P,loaded=rls.load_rls_model(path,3.)
            self.assertFalse(loaded)
            self.assertTrue(np.all(a==0))
            rls.save_rls_model(path,self.a,self.P)
            a,P,loaded=rls.load_rls_model(path,3.)
            self.assertTrue(loaded)
            np.testing.assert_array_equal(a,self.a)
            original=path.read_bytes()
            rls.save_rls_model(path,self.a,self.P,input_delay=.16)
            with self.assertRaisesRegex(RuntimeError,'input_delay mismatch'):
                rls.load_rls_model(path,3.)
            self.assertTrue(rls.load_rls_model(path,3.,input_delay=.16)[2])
            self.assertTrue(rls.load_rls_model(path,3.,input_delay=None)[2])
            original=path.read_bytes()
            with patch.object(rls.np,'savez',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError): rls.save_rls_model(path,a,P)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(list(Path(td).glob('*.tmp')),[])
            np.savez(path,learned_parameters=self.a,covariance=self.P)
            with self.assertRaisesRegex(RuntimeError,'Archive'):
                rls.load_rls_model(path,3.)
            np.savez(path,learned_parameters=self.a,covariance=self.P,model_version=2,state_convention='body_vx')
            with self.assertRaisesRegex(RuntimeError,'Archive'):
                rls.load_rls_model(path,3.)

    def test_evaluator_shared_model_and_rejects_legacy(self):
        import pandas as pd
        spec=importlib.util.spec_from_file_location('eval_rls',SOURCE/'rls_evaluation/evaluate_rls_csv.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'new.csv'
            rows=[]
            for i in range(4):
                state=self.state.copy(); state[1]+=i*.1
                v,w=rls.rls_predict_acceleration(state,.1,self.a)
                row=dict(model_version=2,state_convention=rls.STATE_CONVENTION,time=i*.04,stamp=100+i*.04,dt=.04,tau=.1,x=state[0],v=state[1],theta=state[2],omega=state[3],v_dot_measured=v,omega_dot_measured=w,used_for_rls=0,update_angular=1,command_known=1,input_delay=.16)
                row.update({f'a{k}':val for k,val in enumerate(self.a)})
                rows.append(row)
            pd.DataFrame(rows).to_csv(path,index=False)
            with contextlib.redirect_stdout(io.StringIO()):report=module.evaluate(path)
            self.assertLess(report['fully_predicted_all_known']['omega_dot_rmse'],1e-12)
            self.assertEqual(report['accepted_update_samples'],0)
            self.assertEqual(report['fully_predicted_all_known']['samples'],4)
            self.assertFalse((Path(td)/'new_evaluation/recursive_rollout_metrics.csv').exists())
            pd.DataFrame(rows).drop(columns=['model_version']).to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'version-2'):
                module.evaluate(path)


if __name__=='__main__':
    unittest.main(verbosity=2)
