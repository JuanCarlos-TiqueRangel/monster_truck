"""Offline recorder tests: ROS is stubbed and no hardware is started."""
from contextlib import ExitStack
import csv
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))


def vector(x, y, z):
    return NS(x=x, y=y, z=z)


def messages(stamp_ns):
    """Build independent messages with all the fields saved by the recorder."""
    def header():
        seconds, nanoseconds = divmod(stamp_ns, 1_000_000_000)
        return NS(stamp=NS(sec=seconds, nanosec=nanoseconds), frame_id="world")

    def twist():
        return NS(linear=vector(.01, .02, .03), angular=vector(.04, .05, .06))

    return {
        "accel": NS(header=header(), accel=NS(
            linear=vector(1., 2., 3.), angular=vector(4., 5., 6.))),
        "odom": NS(
            header=header(), child_frame_id="base_link",
            pose=NS(pose=NS(position=vector(7., 8., 9.),
                           orientation=NS(x=.1, y=.2, z=.3, w=.9)),
                    covariance=[i * .1 for i in range(36)]),
            twist=NS(twist=twist(), covariance=[i * .2 for i in range(36)])),
        "twist": NS(header=header(), twist=twist()),
    }


class Clock:
    def __init__(self):
        self.wall_ns = 10**18
        self.mono = 100.0

    def monotonic(self):
        return self.mono

    def advance(self, seconds):
        self.wall_ns += round(seconds * 1e9)
        self.mono += seconds

    def now(self):
        return NS(nanoseconds=self.wall_ns)


class StubNode:
    clock = None
    events = None

    def __init__(self, name):
        self.name = name
        self.logs = []

    def declare_parameter(self, name, value):
        return NS(value=value)

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return NS(info=self.logs.append)

    def create_subscription(self, *args):
        return args

    def create_publisher(self, *args):
        publisher = NS(values=[])

        def publish(msg):
            publisher.values.append(msg.data)
            self.events.append(("publish", msg.data))

        publisher.publish = publish
        return publisher

    def create_timer(self, period, callback):
        return NS(period=period, callback=callback,
                  cancel=lambda: self.events.append(("cancel",)))

    def destroy_node(self):
        self.events.append(("destroy",))


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result
    return result


def forbidden(*args, **kwargs):
    raise AssertionError("A test attempted to initialize or spin ROS")


with patch.dict(sys.modules):
    module("rclpy", init=forbidden, spin=forbidden, spin_once=forbidden,
           ok=lambda: True, shutdown=forbidden)
    module("rclpy.node", Node=StubNode)
    module("rclpy.qos", QoSProfile=lambda **kwargs: NS(**kwargs),
           ReliabilityPolicy=NS(BEST_EFFORT=1))
    module("geometry_msgs")
    module("geometry_msgs.msg", AccelStamped=type("AccelStamped", (), {}),
           TwistStamped=type("TwistStamped", (), {}), Vector3=type("Vector3", (), {}))
    module("nav_msgs")
    module("nav_msgs.msg", Odometry=type("Odometry", (), {}))
    module("std_msgs")
    module("std_msgs.msg", Float32=type("Float32", (), {}))
    identification = importlib.import_module("model_ident")


class Prbs7Tests(unittest.TestCase):
    def test_sequence_is_maximal_balanced_and_repeatable(self):
        bits = identification.generate_prbs7(identification.PRBS_SEED)
        self.assertIsInstance(bits, tuple)
        self.assertEqual(len(bits), 127)
        self.assertEqual(set(bits), {0, 1})
        self.assertEqual(sum(bits), 64)
        self.assertEqual(bits[:21], tuple(map(int, "111111100000010000011")))
        self.assertEqual(identification.generate_prbs7(identification.PRBS_SEED), bits)

        # A maximal PRBS7 visits every nonzero seven-bit window exactly once.
        cyclic = bits + bits[:6]
        windows = {cyclic[index:index + 7] for index in range(127)}
        self.assertEqual(len(windows), 127)
        self.assertNotIn((0,) * 7, windows)

        bipolar = [2 * bit - 1 for bit in bits]
        for shift in range(1, 127):
            correlation = sum(bipolar[index] * bipolar[(index + shift) % 127]
                              for index in range(127))
            self.assertEqual(correlation, -1, f"Autocorrelation at shift {shift}")

        # Other nonzero seeds select a phase of the same sequence.
        rotations = {bits[index:] + bits[:index] for index in range(127)}
        for seed in (1, 53):
            alternate = identification.generate_prbs7(seed)
            self.assertIn(alternate, rotations)
            self.assertNotEqual(alternate, bits)
            self.assertEqual(identification.generate_prbs7(seed), alternate)

    def test_rejects_zero_and_out_of_range_seeds(self):
        for seed in (-1, 0, 128):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                identification.generate_prbs7(seed)


class DataCollectorTests(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.folder = Path(self.context.enter_context(tempfile.TemporaryDirectory()))
        self.clock = Clock()
        self.events = []
        StubNode.clock, StubNode.events = self.clock, self.events
        self.context.enter_context(patch.object(identification, "ROOT", self.folder))
        self.context.enter_context(patch.object(identification, "time", self.clock))

    def new_collector(self):
        node = identification.OptiTrackDataCollectorNode()
        self.addCleanup(node.close)
        return node

    def recorded_rows(self, node, kind):
        stream = node.csv_files[kind]
        if not stream.closed:
            stream.flush()
        with Path(stream.name).open(newline="") as recorded:
            return list(csv.DictReader(recorded))

    def test_records_all_message_fields_and_published_drive(self):
        node = self.new_collector()
        expected_topics = {f"/optitrack/{kind}_810" for kind in ("odom", "twist", "accel", "rpy")}
        self.assertEqual({subscription[1] for subscription in node.subscriptions_}, expected_topics)
        samples = messages(self.clock.wall_ns)
        for kind, msg in samples.items():
            getattr(node, f"on_{kind}")(msg)
        node.on_rpy(vector(.1, -.2, .3))
        node.publish_drive(.05)
        node.close()

        for kind, msg in samples.items():
            rows = self.recorded_rows(node, kind)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(int(row["received_stamp_ns"]), self.clock.wall_ns)
            self.assertEqual(int(row["stamp_ns"]), self.clock.wall_ns)
            self.assertEqual(row["frame_id"], msg.header.frame_id)
            motion = msg.twist.twist if kind == "odom" else getattr(msg, kind)
            for field in ("linear", "angular"):
                for axis in "xyz":
                    self.assertEqual(float(row[f"{field}_{axis}"]), getattr(getattr(motion, field), axis))
            self.assertEqual(len(row), 89 if kind == "odom" else 9)

        odom_row = self.recorded_rows(node, "odom")[0]
        self.assertEqual(odom_row["child_frame_id"], "base_link")
        for field, axes in (("position", "xyz"), ("orientation", "xyzw")):
            for axis in axes:
                self.assertEqual(float(odom_row[f"{field}_{axis}"]),
                                 getattr(getattr(samples["odom"].pose.pose, field), axis))
        for field in ("pose", "twist"):
            for index, value in enumerate(getattr(samples["odom"], field).covariance):
                self.assertEqual(float(odom_row[f"{field}_covariance_{index}"]), value)
        self.assertEqual(self.recorded_rows(node, "rpy"), [{
            "received_stamp_ns": str(self.clock.wall_ns), "roll": "0.1", "pitch": "-0.2", "yaw": "0.3"}])
        drive_rows = self.recorded_rows(node, "drive")
        self.assertEqual([float(row["drive"]) for row in drive_rows], [.05, 0.])
        self.assertEqual([float(row["drive"]) for row in drive_rows], node.drive_publisher.values)
        self.assertTrue(all(int(row["published_stamp_ns"]) == self.clock.wall_ns for row in drive_rows))
        self.assertEqual({path.name for path in node.output_dir.iterdir()},
                         {f"{kind}.csv" for kind in ("accel", "odom", "rpy", "twist", "drive")})

    def test_records_stale_future_duplicate_unmatched_and_headerless_messages(self):
        node = self.new_collector()
        published_before = list(node.drive_publisher.values)
        stale_stamp = self.clock.wall_ns - 1_000_000_000
        future_stamp = self.clock.wall_ns + 1_000_000_000
        stale = messages(stale_stamp)["odom"]
        node.on_odom(stale)
        node.on_odom(stale)
        node.on_accel(messages(future_stamp)["accel"])
        node.on_twist(messages(self.clock.wall_ns + 123)["twist"])
        node.on_rpy(vector(1., 2., 3.))

        self.assertEqual([int(row["stamp_ns"]) for row in self.recorded_rows(node, "odom")],
                         [stale_stamp, stale_stamp])
        self.assertEqual(int(self.recorded_rows(node, "accel")[0]["stamp_ns"]), future_stamp)
        self.assertEqual(int(self.recorded_rows(node, "twist")[0]["stamp_ns"]), self.clock.wall_ns + 123)
        self.assertEqual(len(self.recorded_rows(node, "rpy")), 1)
        for kind in ("odom", "accel", "twist", "rpy"):
            self.assertTrue(all(int(row["received_stamp_ns"]) == self.clock.wall_ns
                                for row in self.recorded_rows(node, kind)))
        self.assertEqual(node.drive_publisher.values, published_before)

    def test_prbs_starts_immediately_and_holds_bits_without_sensor_messages(self):
        node = self.new_collector()
        self.assertEqual(node.timer.period, .04)
        self.assertEqual(identification.PRBS_LOW, 0.)
        self.assertEqual(identification.PRBS_HIGH, .12)
        self.assertEqual(identification.PRBS_BIT_PERIOD_S, .20)
        started_at = self.clock.mono
        node.update_drive()
        self.assertEqual(node.experiment_started_at, started_at)
        self.assertEqual(node.drive_publisher.values[-1], .12)

        # The default seed starts with seven high bits, six low bits, then high.
        for elapsed, expected in ((.04, .12), (.199, .12), (1.399, .12),
                                  (1.4, 0.), (1.44, 0.), (2.599, 0.),
                                  (2.6, .12), (2.799, .12), (2.8, 0.)):
            with self.subTest(elapsed=elapsed):
                self.clock.advance(started_at + elapsed - self.clock.mono)
                node.update_drive()
                self.assertEqual(node.drive_publisher.values[-1], expected)
        self.assertEqual(set(node.drive_publisher.values), {0., .12})
        for kind in ("odom", "twist", "accel", "rpy"):
            self.assertEqual(self.recorded_rows(node, kind), [])

    def test_late_callbacks_select_elapsed_bit_and_sequence_repeats(self):
        with patch.object(identification, "EXPERIMENT_DURATION_S", 60.), \
             patch.object(identification, "PRBS_LOW", .03):
            node = self.new_collector()
            started_at = self.clock.mono
            node.update_drive()
            # Jump across missed ticks, then across the full 127-bit period.
            for bit_index, expected in ((7, .03), (13, .12), (14, .03),
                                        (126, .03), (127, .12), (134, .03),
                                        (140, .12)):
                with self.subTest(bit_index=bit_index):
                    elapsed = (bit_index + .5) * identification.PRBS_BIT_PERIOD_S
                    self.clock.advance(started_at + elapsed - self.clock.mono)
                    node.update_drive()
                    self.assertEqual(node.drive_publisher.values[-1], expected)
            self.assertFalse(node.finished)

    def test_duration_finishes_with_neutral_without_sensor_messages(self):
        node = self.new_collector()
        node.update_drive()
        self.clock.advance(identification.EXPERIMENT_DURATION_S - .001)
        node.update_drive()
        self.assertFalse(node.finished)
        self.clock.advance(.001)
        node.update_drive()

        self.assertTrue(node.finished)
        self.assertEqual(node.drive_publisher.values[-1], 0.)
        self.assertIn(("cancel",), self.events)
        published_before = list(node.drive_publisher.values)
        node.update_drive()
        self.assertEqual(node.drive_publisher.values, published_before)
        for kind in ("odom", "twist", "accel", "rpy"):
            self.assertEqual(self.recorded_rows(node, kind), [])

    def test_close_sends_neutral_and_closes_files_once(self):
        node = self.new_collector()
        node.publish_drive(.12)
        node.close()
        self.assertEqual(node.drive_publisher.values, [.12, 0.])
        self.assertTrue(all(stream.closed for stream in node.csv_files.values()))
        events_before = list(self.events)
        node.close()
        self.assertEqual(self.events, events_before)

    def test_close_preserves_data_if_shutdown_publication_fails(self):
        node = self.new_collector()
        node.on_rpy(vector(1., 2., 3.))
        with patch.object(node.drive_publisher, "publish", side_effect=RuntimeError("ROS is shut down")):
            with self.assertRaisesRegex(RuntimeError, "ROS is shut down"):
                node.close()
        self.assertTrue(all(stream.closed for stream in node.csv_files.values()))
        self.assertEqual(len(self.recorded_rows(node, "rpy")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
