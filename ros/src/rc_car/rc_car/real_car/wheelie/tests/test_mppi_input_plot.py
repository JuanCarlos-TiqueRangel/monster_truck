"""Offline subscriber/plot tests; no ROS initialization or vehicle commands."""

import ast
import importlib.util
import math
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "plot_mppi_inputs_node.py"


def forbidden(*args, **kwargs):
    raise AssertionError("Live ROS/network access or publishing is forbidden")


class StubNode:
    overrides = {}

    def __init__(self, name):
        self.name, self.subscriptions, self.destroyed = name, [], False

    def declare_parameter(self, name, default):
        return NS(value=self.overrides.get(name, default))

    def create_subscription(self, msg_type, topic, callback, qos):
        subscription = NS(msg_type=msg_type, topic=topic, callback=callback, qos=qos)
        self.subscriptions.append(subscription)
        return subscription

    create_publisher = forbidden

    def get_logger(self):
        return NS(info=lambda text: None)

    def destroy_node(self):
        self.destroyed = True


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result


with patch.dict(sys.modules):
    module("rclpy", init=forbidden, spin=forbidden, shutdown=forbidden, ok=lambda: True)
    module("rclpy.node", Node=StubNode)
    module("rclpy.executors", SingleThreadedExecutor=forbidden,
           ExternalShutdownException=type("ExternalShutdownException", (Exception,), {}))
    module("rclpy.qos", QoSProfile=lambda **kwargs: NS(**kwargs),
           HistoryPolicy=NS(KEEP_LAST=1), ReliabilityPolicy=NS(BEST_EFFORT=2))
    module("geometry_msgs")
    module("geometry_msgs.msg", TwistStamped=type("TwistStamped", (), {}),
           AccelStamped=type("AccelStamped", (), {}))
    module("nav_msgs")
    module("nav_msgs.msg", Odometry=type("Odometry", (), {}))
    spec = importlib.util.spec_from_file_location("mppi_input_plot_under_test", SOURCE)
    plotter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plotter
    spec.loader.exec_module(plotter)


def messages(stamp_ns, x=1.2, pitch=-.5, vx=.6, omega=-.7, ax=24., alpha=-360.):
    header = NS(stamp=NS(sec=stamp_ns // 10**9, nanosec=stamp_ns % 10**9))
    return {
        "odom": NS(header=header, pose=NS(pose=NS(position=NS(x=x), orientation=NS(
            x=0., y=math.sin(pitch / 2), z=0., w=math.cos(pitch / 2))))),
        "twist": NS(header=header, twist=NS(linear=NS(x=vx), angular=NS(y=omega))),
        "accel": NS(header=header, accel=NS(linear=NS(x=ax), angular=NS(y=alpha))),
    }


def feed(node, stamp_ns, **kwargs):
    for topic, msg in messages(stamp_ns, **kwargs).items():
        getattr(node, topic + "_callback")(msg)


class InputPlotTests(unittest.TestCase):
    def setUp(self):
        StubNode.overrides = {}
        self.node = plotter.MPPIInputPlotNode()

    def tearDown(self):
        plt.close("all")

    def test_mppi_input_topics_and_plotting_qos(self):
        topics = []
        for subscription in self.node.subscriptions:
            topics.append(subscription.topic)
        self.assertEqual(topics, ["/optitrack/odom_810", "/optitrack/twist_810", "/optitrack/accel_810"])
        for subscription in self.node.subscriptions:
            self.assertEqual(subscription.qos.depth, 50)
            self.assertEqual(subscription.qos.history, 1)
            self.assertEqual(subscription.qos.reliability, 2)

    def test_robot_id_override(self):
        StubNode.overrides = {"robot_id": 42}
        node = plotter.MPPIInputPlotNode()
        for subscription in node.subscriptions:
            self.assertTrue(subscription.topic.endswith("_42"))

    def test_subscription_depth_override(self):
        StubNode.overrides = {"subscription_depth": 20}
        node = plotter.MPPIInputPlotNode()
        for subscription in node.subscriptions:
            self.assertEqual(subscription.qos.depth, 20)

    def test_callbacks_select_the_six_controller_inputs(self):
        feed(self.node, 10**18)
        data, _ = self.node.snapshot()
        np.testing.assert_allclose(data["odom"], [[0., 1.2, math.degrees(-.5)]])
        np.testing.assert_allclose(data["twist"], [[0., .6, -.7]])
        np.testing.assert_allclose(data["accel"], [[0., 24., -360.]])

    def test_pitch_keeps_the_controllers_beyond_90_degree_convention(self):
        feed(self.node, 10**18, pitch=math.radians(-110.))
        data, _ = self.node.snapshot()
        self.assertAlmostEqual(data["odom"][0, 2], -110.)

    def test_time_origin_preserves_nanosecond_differences_at_large_epochs(self):
        base = 1_791_831_514_674_712_622
        feed(self.node, base)
        feed(self.node, base + 1)
        data, _ = self.node.snapshot()
        for array in data.values():
            self.assertEqual(array[0, 0], 0.)
            self.assertEqual(array[1, 0], 1e-9)

    def test_topics_keep_their_own_timestamps_and_missing_topics_stay_empty(self):
        self.node.odom_callback(messages(10**18)["odom"])
        self.node.twist_callback(messages(10**18 + 20_000_000)["twist"])
        data, age = self.node.snapshot()
        self.assertEqual(data["odom"][0, 0], 0.)
        self.assertEqual(data["twist"][0, 0], .02)
        self.assertEqual(data["accel"].shape, (0, 3))
        self.assertIsNone(age["accel"])

    def test_duplicates_and_out_of_order_messages_are_ignored(self):
        feed(self.node, 10**18, x=2.)
        feed(self.node, 10**18, x=3.)
        feed(self.node, 10**18 - 1, x=4.)
        data, _ = self.node.snapshot()
        self.assertEqual(len(data["odom"]), 1)
        self.assertEqual(data["odom"][0, 1], 2.)

    def test_message_gaps_break_lines_instead_of_interpolating(self):
        feed(self.node, 10**18)
        feed(self.node, 10**18 + 10_000_000)
        feed(self.node, 10**18 + 130_000_000)
        data, _ = self.node.snapshot()
        for array in data.values():
            self.assertEqual(len(array), 4)
            self.assertTrue(np.isnan(array[2]).all())
            self.assertAlmostEqual(array[3, 0], .13)

    def test_gap_threshold_is_strict_and_uses_integer_timestamp_precision(self):
        for stamp in (0, 20_000_000, 70_000_000, 100_000_000, 150_000_000, 200_000_000):
            feed(self.node, 10**18 + stamp)
        data, _ = self.node.snapshot()
        self.assertEqual(len(data["odom"]), 6)

    def test_memory_is_bounded(self):
        StubNode.overrides = {"max_samples": 4}
        node = plotter.MPPIInputPlotNode()
        for i in range(20):
            feed(node, 10**18 + i * 1_000_000)
        for records in node._data.values():
            self.assertEqual(len(records), 4)

    def test_old_data_is_pruned_by_window_duration(self):
        StubNode.overrides = {"window_s": 1.}
        node = plotter.MPPIInputPlotNode()
        for i in range(10):
            feed(node, 10**18 + i * 100_000_000)
        feed(node, 10**18 + 2_000_000_000)
        for records in node._data.values():
            self.assertEqual(len(records), 1)

    def test_freshness_is_time_since_local_reception_not_source_epoch(self):
        with patch.object(plotter.time, "monotonic", return_value=10.):
            feed(self.node, 10**18)
        with patch.object(plotter.time, "monotonic", return_value=10.125):
            _, age = self.node.snapshot()
        self.assertEqual(age, {"odom": .125, "twist": .125, "accel": .125})

    def test_waiting_plot_has_all_six_axes(self):
        fig, update = plotter.create_plot(self.node)
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual(len(update()), 8)
        self.assertIn("accel: waiting", fig._suptitle.get_text())
        fig.canvas.draw()

    def test_raw_and_clipped_acceleration_are_both_drawn(self):
        feed(self.node, 10**18)
        feed(self.node, 10**18 + 10_000_000, ax=-24., alpha=360.)
        fig, _ = plotter.create_plot(self.node)
        linear, angular = fig.axes[4], fig.axes[5]
        np.testing.assert_allclose(linear.lines[0].get_ydata(), [24., -24.])
        np.testing.assert_allclose(linear.lines[1].get_ydata(), [15., -15.])
        np.testing.assert_allclose(angular.lines[0].get_ydata(), [-360., 360.])
        np.testing.assert_allclose(angular.lines[1].get_ydata(), [-200., 200.])
        fig.canvas.draw()

    def test_six_plot_channels_and_rolling_time_limits(self):
        for i in range(1101):
            feed(self.node, 10**18 + i * 10_000_000)
        fig, update = plotter.create_plot(self.node)
        update()
        expected = (1.2, math.degrees(-.5), .6, -.7, 24., -360.)
        for ax, value in zip(fig.axes, expected):
            self.assertEqual(ax.get_xlim(), (1., 11.))
            self.assertAlmostEqual(ax.lines[0].get_ydata()[-1], value)

    def test_nonfinite_values_do_not_break_the_renderer(self):
        feed(self.node, 10**18, x=float("nan"), vx=float("inf"), alpha=float("nan"))
        fig, update = plotter.create_plot(self.node)
        update()
        fig.canvas.draw()
        for ax in fig.axes:
            for value in ax.get_ylim():
                self.assertTrue(math.isfinite(value))

    def test_invalid_parameters_are_rejected(self):
        for name in ("window_s", "refresh_hz", "gap_s", "linear_accel_cap", "angular_accel_cap"):
            for value in (0., -1., float("nan"), float("inf")):
                StubNode.overrides = {name: value}
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    plotter.MPPIInputPlotNode()
        StubNode.overrides = {"max_samples": 1}
        with self.assertRaises(ValueError):
            plotter.MPPIInputPlotNode()
        StubNode.overrides = {"subscription_depth": 0}
        with self.assertRaises(ValueError):
            plotter.MPPIInputPlotNode()

    def test_nodes_have_no_inline_loops_or_conditional_expressions(self):
        optitrack_source = SOURCE.parent.parent / "optitrack" / "optitrack_node.py"
        forbidden_nodes = (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.IfExp)
        for path in (SOURCE, optitrack_source, Path(__file__)):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                with self.subTest(file=str(path), node_type=type(node).__name__):
                    self.assertNotIsInstance(node, forbidden_nodes)

    def run_main(self, show_error=None):
        state = {"alive": False, "executor": None, "interval": None}

        class Executor:
            def __init__(self):
                self.started, self.stopped = threading.Event(), threading.Event()
                state["executor"] = self

            def add_node(self, node):
                self.node = node

            def spin(self):
                self.started.set()
                self.stopped.wait(2.)

            def shutdown(self, timeout_sec):
                self.stopped.set()

        def animate(fig, callback, **kwargs):
            state["interval"] = kwargs["interval"]
            callback(0)
            return NS()

        def show():
            self.assertTrue(state["executor"].started.wait(1.))
            if show_error:
                raise show_error

        with patch.object(plotter, "SingleThreadedExecutor", Executor), \
                patch.object(plotter.rclpy, "init", side_effect=lambda args: state.update(alive=True)), \
                patch.object(plotter.rclpy, "ok", side_effect=lambda: state["alive"]), \
                patch.object(plotter.rclpy, "shutdown", side_effect=lambda: state.update(alive=False)), \
                patch("matplotlib.animation.FuncAnimation", side_effect=animate), \
                patch.object(plt, "show", side_effect=show):
            if show_error:
                with self.assertRaisesRegex(RuntimeError, "display failed"):
                    plotter.main()
            else:
                plotter.main()
        self.assertFalse(state["alive"])
        self.assertTrue(state["executor"].stopped.is_set())
        self.assertTrue(state["executor"].node.destroyed)
        self.assertEqual(plt.get_fignums(), [])
        return state

    def test_main_uses_configured_redraw_rate_and_cleans_up(self):
        StubNode.overrides = {"refresh_hz": 5.}
        self.assertEqual(self.run_main()["interval"], 200.)

    def test_main_cleans_up_if_the_display_fails(self):
        self.run_main(RuntimeError("display failed"))


if __name__ == "__main__":
    with patch("socket.socket", side_effect=forbidden):
        unittest.main(verbosity=2)
