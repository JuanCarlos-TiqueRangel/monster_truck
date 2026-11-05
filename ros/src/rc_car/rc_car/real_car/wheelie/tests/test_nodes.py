"""Offline node integration tests: ROS APIs, clocks and optimizer are stubs.

No rclpy initialization, ROS network, serial port or real-car node is started.
"""
from contextlib import ExitStack
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).parents[1]))
from test_wheelie_measurements import messages, quaternion


class Clock:
    def __init__(self):
        self.wall_ns = 10**18
        self.mono = 100.0

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.wall_ns += round(seconds*1e9)
        self.mono += seconds

    def now(self):
        return NS(nanoseconds=self.wall_ns)


class StubNode:
    parameters = {}
    clock = None
    events = None
    publishers = None

    def __init__(self, name):
        self.name = name
        self.logs = []

    def declare_parameter(self, name, value):
        return NS(value=self.parameters.get(name, value))

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return NS(info=lambda text:self.logs.append(text), warning=lambda text:self.logs.append(text))

    def create_subscription(self, *args):
        return args

    def create_publisher(self, *args):
        publisher = NS(values=[])
        def publish(msg):
            publisher.values.append(msg.data)
            self.events.append(("publish",msg.data))
        publisher.publish = publish
        self.publishers.append(publisher)
        return publisher

    def create_timer(self, period, callback):
        return NS(period=period, callback=callback, cancel=lambda:self.events.append(("cancel",)))

    def destroy_node(self):
        self.events.append(("destroy",))


class StubOptimizer:
    def __init__(self, p, cfg, a, **kwargs):
        self.device = kwargs.get("device", "cpu")
        self.calls = []
        self.resets = 0
        self.command = .05
        self.info = {"success":True}
        self.duration = 0.0

    def solve(self, state, reference, previous):
        self.calls.append((state.copy(), reference.copy(), previous))
        StubNode.clock.advance(self.duration)
        return self.command, dict(self.info)

    def reset(self):
        self.resets += 1

    def set_rls_parameters(self, a):
        self.a = a.copy()

    def close_plot(self):
        StubNode.events.append(("plot.close",))


class StubCSV:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def write(self, *args, **kwargs):
        self.rows.append((args,kwargs))

    def close(self):
        StubNode.events.append(("csv.close",))


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("A test attempted to initialize or spin ROS")


with patch.dict(sys.modules):
    module("rclpy",init=forbidden,spin=forbidden,spin_once=forbidden,ok=lambda:True,shutdown=forbidden)
    module("rclpy.node",Node=StubNode)
    module("rclpy.qos",QoSProfile=lambda **kwargs:NS(**kwargs),
           ReliabilityPolicy=NS(BEST_EFFORT=1,RELIABLE=2))
    module("geometry_msgs")
    module("geometry_msgs.msg",AccelStamped=type("AccelStamped",(),{}),TwistStamped=type("TwistStamped",(),{}))
    module("nav_msgs")
    module("nav_msgs.msg",Odometry=type("Odometry",(),{}))
    module("std_msgs")
    module("std_msgs.msg",Float32=type("Float32",(),{}))
    module("mppi",MPPITorch=StubOptimizer)
    controller = importlib.import_module("mppi_controller_with_rls_logger")
    identification = importlib.import_module("model_ident")
    rls = importlib.import_module("rls_clean")


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.folder = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.clock = Clock()
        self.events = []
        StubNode.clock, StubNode.events, StubNode.publishers = self.clock, self.events, []
        StubNode.parameters = {}
        self.model = self.folder / "model.npz"
        for target in (controller,identification):
            for name,value in (("ROOT",self.folder),("RLS_FILE",self.model),("time",self.clock),
                               ("RLSCSVLogger",StubCSV)):
                self.context.enter_context(patch.object(target,name,value))

    def control_ready(self):
        a = np.zeros(12)
        a[0], a[5] = 20., -2.
        rls.save_rls_model(self.model,a,np.eye(12),input_delay=StubNode.parameters.get("input_delay",0.0))
        return a

    def feed(self,node,**kwargs):
        for kind,msg in messages(self.clock.wall_ns,**kwargs).items():
            node.sensor_callback(kind,msg)

    def new_controller(self):
        self.control_ready()
        return controller.MPPIControllerNode()

    def test_missing_legacy_invalid_and_uncontrollable_models_never_create_drive(self):
        for case in ("missing","legacy","invalid","uncontrollable"):
            self.model.unlink(missing_ok=True)
            if case == "legacy":
                np.savez(self.model,learned_parameters=np.zeros(12),covariance=np.eye(12))
            elif case == "invalid":
                np.savez(self.model,learned_parameters=np.full(12,np.nan),covariance=np.eye(12),
                         model_version=rls.MODEL_VERSION,state_convention=rls.STATE_CONVENTION,input_delay=0.0)
            elif case == "uncontrollable":
                rls.save_rls_model(self.model,np.zeros(12),np.eye(12))
            before = self.model.read_bytes() if self.model.exists() else None
            with self.subTest(case=case), self.assertRaises((ValueError,RuntimeError)):
                controller.MPPIControllerNode()
            self.assertEqual(StubNode.publishers,[])
            self.assertEqual(self.model.read_bytes() if self.model.exists() else None,before)

    def test_invalid_delay_rejected_before_publisher_or_checkpoint(self):
        for delay in (-.1,1.1,float("nan"),float("inf")):
            StubNode.parameters={"input_delay":delay}
            for node_type in (controller.MPPIControllerNode,identification.RLSExperimentNode):
                with self.subTest(delay=delay,node=node_type), self.assertRaises(ValueError):
                    node_type()
            self.assertFalse(self.model.exists())
            self.assertEqual(StubNode.publishers,[])

    def test_constant_self_lifting_bias_is_not_control_authority(self):
        a = np.zeros(12)
        a[4], a[11] = .1, -1.0
        rls.save_rls_model(self.model, a, np.eye(12))
        with self.assertRaisesRegex(RuntimeError, "cannot predict lift"):
            controller.MPPIControllerNode()
        self.assertEqual(StubNode.publishers, [])

    def test_pitch_authority_error_explains_neutral_lift_bias(self):
        a = np.zeros(12)
        a[0], a[10], a[11] = 20., 8., -1.4
        error = controller.control_model_error(a, controller.WheelieParams())
        self.assertIn("throttle does not produce more nose-up acceleration than neutral", error)
        self.assertIn("omega_dot=-1.400", error)
        self.assertIn("omega_dot=-0.488", error)

    def test_identification_saves_progress_but_reports_not_ready_on_close(self):
        node = identification.RLSExperimentNode()
        self.feed(node)
        node.timer_callback()
        self.clock.advance(.04)
        self.feed(node)
        node.timer_callback()
        self.assertGreater(node.updates, 0)
        self.assertEqual(node.angular_updates, 0)
        node.close()
        self.assertTrue(self.model.exists())
        self.assertTrue(any("0 angular updates" in line for line in node.logs))
        self.assertTrue(any("Model is NOT ready for MPPI" in line for line in node.logs))
        self.assertEqual(node.drive_pub.values[-1], 0.)

    def test_identification_retains_progress_if_shutdown_publication_fails(self):
        node = identification.RLSExperimentNode()
        node.updates = 1
        with patch.object(node.drive_pub, "publish", side_effect=RuntimeError("ROS is shut down")):
            with self.assertRaisesRegex(RuntimeError, "ROS is shut down"):
                node.close()
        self.assertTrue(self.model.exists())
        self.assertIn(("csv.close",), self.events)

    def test_control_uses_one_solve_per_current_coherent_sample(self):
        node=self.new_controller()
        self.feed(node)
        node.control_callback()
        node.control_callback()
        self.assertEqual(len(node.controller.calls),1)
        self.assertEqual(node.drive_pub.values,[.05])
        self.clock.advance(.04)
        self.feed(node,q=quaternion(pitch=-.1),velocity=np.array([.2,0,0]))
        node.control_callback()
        self.assertEqual(len(node.controller.calls),2)
        np.testing.assert_allclose(node.controller.calls[-1][0],node.latest.state)

    def test_headers_rejected_before_sync_and_origin(self):
        for node in (self.new_controller(),identification.RLSExperimentNode()):
            for offset in (-81_000_000,1):
                bad=messages(self.clock.wall_ns+offset)
                node.sensor_callback("odom",bad["odom"])
                self.assertIsNone(node.measurements._origin)
                self.assertIsNone(node.measurements._last_stamp_ns)
                self.assertIsNone(node.latest)
                self.assertEqual(node.drive_pub.values[-1],0.)

    def test_control_failure_lateness_and_nonfinite_command_reset_to_neutral(self):
        for failure in ("failed","late","nan"):
            node=self.new_controller()
            self.feed(node)
            if failure == "failed":
                node.controller.info={"success":False,"reason":"forced failure"}
            elif failure == "late":
                node.controller.duration=.05
            else:
                node.controller.command=float("nan")
            node.control_callback()
            self.assertEqual(node.drive_pub.values[-1],0.)
            self.assertEqual(node.tau_prev,0.)
            self.assertEqual(node.controller.resets,1)
            self.assertFalse(node.csv.rows[-1][0][-1])

    def test_rls_uses_current_state_acceleration_and_delayed_command(self):
        StubNode.parameters={"input_delay":.01}
        node=self.new_controller()
        node.publish(.05)
        self.clock.advance(.04)
        self.feed(node)
        node.control_callback()
        self.clock.advance(.04)
        self.feed(node,q=quaternion(pitch=-.1),velocity=np.array([.2,0,0]),
                  acceleration=np.array([1.5,0,0]),alpha=np.array([0,-2.,0]))
        calls=[]
        def update(state,tau,v_dot,w_dot,a,P,**kwargs):
            calls.append((state.copy(),tau,v_dot,w_dot,kwargs))
            return a.copy(),P.copy(),{"skipped":False}
        with patch.object(controller,"rls_update",update):
            node.control_callback()
        self.assertEqual(len(calls),1)
        np.testing.assert_allclose(calls[0][0],node.latest.state)
        self.assertEqual(calls[0][1],.05)
        np.testing.assert_allclose(calls[0][2:4],node.latest.acceleration)
        self.assertTrue(calls[0][4]["update_angular"])
        self.assertTrue(node.dirty)

    def test_delay_prediction_requires_known_history_and_uses_queued_input(self):
        StubNode.parameters={"input_delay":.01,"online_rls":False}
        node=self.new_controller()
        self.feed(node)
        node.control_callback()
        self.assertEqual(node.drive_pub.values[-1],0.)
        self.assertEqual(len(node.controller.calls),0)
        self.clock.advance(.001)
        node.publish(.05)
        self.clock.advance(.04)
        self.feed(node)
        expected=rls.rk4_step_numpy(node.latest.state,.05,.01,node.a)
        node.control_callback()
        np.testing.assert_allclose(node.controller.calls[-1][0],expected,atol=1e-12)
        self.assertGreater(node.controller.calls[-1][0][1],node.latest.state[1])

    def test_shutdown_neutral_precedes_checkpoint_even_if_save_fails(self):
        for target,node in ((controller,self.new_controller()),
                            (identification,identification.RLSExperimentNode())):
            node.dirty=True
            node.updates=1
            self.events.clear()
            def save(*args,**kwargs):
                self.events.append(("save",))
                raise OSError("forced slow/disk failure")
            with patch.object(target,"save_rls_model",save),self.assertRaises(OSError):
                node.close()
            self.assertEqual(self.events[0], ("publish", 0.))
            self.assertIn(("save",), self.events)
            self.assertIn(("csv.close",),self.events)

    def test_identification_requires_ground_initialization_and_fresh_data(self):
        node=identification.RLSExperimentNode()
        self.feed(node,velocity=np.array([.2,0,0]))
        node.timer_callback()
        self.assertIsNone(node.measurements._origin)
        self.assertEqual(node.updates,0)
        self.clock.advance(.04)
        self.feed(node)
        node.timer_callback()
        updates_before_stale=node.updates
        self.clock.advance(2.1)
        node.timer_callback()
        self.assertTrue(all(command == 0 for command in node.drive_pub.values))
        self.assertEqual(node.updates,updates_before_stale)
        self.assertFalse(self.model.exists())

    def test_identification_pulses_only_with_fresh_samples_within_shared_bounds(self):
        node=identification.RLSExperimentNode()
        with patch.object(identification.random,"choice",lambda levels:max(levels)):
            for i in range(60):
                self.feed(node)
                node.timer_callback()
                self.clock.advance(.04)
        self.assertTrue(any(command > 0 for command in node.drive_pub.values))
        self.assertTrue(all(0 <= command <= node.p.tau_max for command in node.drive_pub.values))
        self.assertEqual(max(node.levels),node.p.tau_max)

    def test_identification_late_update_cannot_publish_a_pulse(self):
        node=identification.RLSExperimentNode()
        self.feed(node)
        node.timer_callback()
        self.clock.advance(.04)
        self.feed(node)
        node.change_time=self.clock.mono
        def slow_update(state,tau,v_dot,w_dot,a,P,**kwargs):
            self.clock.advance(.10)
            return a,P,{"skipped":False}
        with patch.object(identification,"rls_update",slow_update):
            node.timer_callback()
        self.assertEqual(node.drive_pub.values[-1],0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
