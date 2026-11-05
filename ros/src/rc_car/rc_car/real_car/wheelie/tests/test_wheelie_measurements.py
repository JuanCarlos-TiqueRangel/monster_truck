"""Offline tests; message-shaped objects only, no ROS runtime or hardware."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wheelie_measurements import CommandHistory, WheelieMeasurements, _rotation


def vector(values):
    return NS(x=values[0], y=values[1], z=values[2])


def quaternion(yaw=0, pitch=0, roll=0):
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    return NS(x=sr*cp*cy-cr*sp*sy, y=cr*sp*cy+sr*cp*sy,
              z=cr*cp*sy-sr*sp*cy, w=cr*cp*cy+sr*sp*sy)


def messages(stamp_ns, q=None, position=(0,0,0), velocity=(0,0,0),
             acceleration=(0,0,0), omega=(0,0,0), alpha=(0,0,0)):
    q = q or quaternion()
    R = _rotation(q)
    stamp = NS(sec=stamp_ns//10**9, nanosec=stamp_ns % 10**9)
    odom = NS(header=NS(stamp=stamp, frame_id="map"), child_frame_id="base_link",
              pose=NS(pose=NS(position=vector(position), orientation=q)))
    twist = NS(header=NS(stamp=stamp, frame_id="base_link"),
               twist=NS(linear=vector(R.T@velocity), angular=vector(R.T@omega)))
    accel = NS(header=NS(stamp=stamp, frame_id="base_link"),
               accel=NS(linear=vector(R.T@acceleration), angular=vector(R.T@alpha)))
    return dict(odom=odom, twist=twist, accel=accel)


def submit(converter, triplet, order=("odom", "twist", "accel")):
    result = None
    for kind in order:
        value = converter.push(kind, triplet[kind])
        if value is not None:
            result = value
    return result


class MeasurementTests(unittest.TestCase):
    def test_heading_invariance_and_pitch_zero(self):
        for yaw in (0, .7, 2.3):
            converter = WheelieMeasurements()
            h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            left = np.array([-h[1], h[0], 0.0])
            origin = np.array([2., -1., .3])
            bias = math.radians(3)
            first = submit(converter, messages(10**18, quaternion(yaw, bias), origin))
            np.testing.assert_allclose(first.state, 0, atol=1e-12)
            sample = submit(converter, messages(
                10**18+1, quaternion(yaw, bias-math.radians(50)), origin+2*h,
                1.2*h+np.array([0.,0.,.4]), 2*h+np.array([0.,0.,3.]),
                .7*left, 1.3*left), ("accel", "odom", "twist"))
            self.assertEqual(sample.stamp_ns, 10**18+1)
            np.testing.assert_allclose(sample.state, [2,1.2,-math.radians(50),.7], atol=1e-12)
            np.testing.assert_allclose(sample.acceleration, [2,1.3], atol=1e-12)

    def test_exact_angular_derivatives_off_plane(self):
        converter = WheelieMeasurements()
        submit(converter, messages(0))
        q = quaternion(.1, -.7, .04)
        R = _rotation(q)
        w, alpha = np.array([.1,-.5,.2]), np.array([.05,.2,-.1])
        sample = submit(converter, messages(1, q, omega=w, alpha=alpha))
        f = R[:,0]
        fd = np.cross(w,f)
        fdd = np.cross(alpha,f)+np.cross(w,fd)
        def angle(t):
            ft = f + fd*t + .5*fdd*t*t
            return np.arctan2(-ft[2],ft[0])
        dt = 1e-4
        self.assertAlmostEqual(sample.state[3], (angle(dt)-angle(-dt))/(2*dt), places=7)
        self.assertAlmostEqual(sample.acceleration[1], (angle(dt)-2*angle(0)+angle(-dt))/dt**2, places=6)

    def test_sync_duplicates_old_dropped_and_bound(self):
        converter = WheelieMeasurements(queue_size=2)
        one, two = messages(1), messages(2)
        self.assertIsNone(converter.push("odom",one["odom"]))
        self.assertIsNone(converter.push("twist",two["twist"]))
        self.assertIsNone(converter.push("accel",one["accel"]))
        result = converter.push("twist",one["twist"])
        self.assertEqual(result.stamp_ns,1)
        self.assertIsNone(submit(converter,one))
        self.assertIsNone(submit(converter,messages(0)))
        self.assertIsNone(converter.push("odom",two["odom"]))
        self.assertEqual(converter.push("accel",two["accel"]).stamp_ns,2)
        for stamp in range(3,20):
            converter.push("odom",messages(stamp)["odom"])
        self.assertEqual(len(converter._buffers["odom"]),2)

    def test_invalid_initial_data_does_not_commit_origin(self):
        invalids = []
        bad_frame = messages(1)
        bad_frame["twist"].header.frame_id="map"
        invalids.append(bad_frame)
        bad_quat = messages(2)
        bad_quat["odom"].pose.pose.orientation=NS(x=0,y=0,z=0,w=0)
        invalids.append(bad_quat)
        invalids.append(messages(3, velocity=np.array([.1,0,0])))
        invalids.append(messages(4, quaternion(pitch=math.radians(20))))
        invalids.append(messages(5, quaternion(roll=math.radians(30))))
        invalids.append(messages(6, acceleration=np.array([np.nan,0,0])))
        for triplet in invalids:
            converter=WheelieMeasurements()
            with self.assertRaises(ValueError):
                submit(converter,triplet)
            self.assertIsNone(converter._origin)
            good=submit(converter,messages(100,position=(2,3,4)))
            np.testing.assert_allclose(good.state,0)

    def test_frame_change_and_plane_departure(self):
        for bad in (messages(2,quaternion(yaw=.5)), messages(2)):
            converter=WheelieMeasurements()
            submit(converter,messages(1))
            if bad["odom"].pose.pose.orientation.z == 0:
                bad["odom"].header.frame_id="other_map"
            with self.assertRaises(ValueError):
                submit(converter,bad)


class HistoryTests(unittest.TestCase):
    def test_prediction_segments_split_command_edges_and_watchdog(self):
        history = CommandHistory()
        history.record(1_000_000_000, 0.0)
        history.record(1_100_000_000, .1)
        segments = list(history.segments(1_150_000_000, 1_600_000_000, delay_s=.1))
        self.assertAlmostEqual(sum(dt*u for dt, u in segments), .03)
        self.assertAlmostEqual(sum(dt for dt, _ in segments), .45)
        with self.assertRaisesRegex(ValueError, "history"):
            list(history.segments(500_000_000, 1_050_000_000))

    def test_delayed_lookup_changes_resends_and_watchdog(self):
        history=CommandHistory()
        history.record(10**9,0)
        history.record(1100_000_000,.1)
        history.record(1140_000_000,.1)
        self.assertIsNone(history.lookup(999_999_999))
        self.assertEqual(history.lookup(1160_000_000,delay_s=.1),(0.0,.06))
        self.assertEqual(history.lookup(1200_000_000),(.1,.1))
        self.assertEqual(history.lookup(1500_000_000),(0.0,.06))
        history.record(1600_000_000,.1)
        self.assertEqual(history.lookup(1640_000_000),(.1,.04))
        history.record(1700_000_000,0)
        history.record(1740_000_000,0)
        self.assertEqual(history.lookup(1800_000_000),(0.0,.1))

    def test_bounded_history_and_invalid_inputs(self):
        history=CommandHistory(maxlen=3)
        for i in range(10):
            history.record(i*40_000_000,.1)
        self.assertEqual(len(history._events),3)
        self.assertEqual(history.lookup(360_000_000),(.1,.36))
        self.assertIsNone(history.lookup(0))
        with self.assertRaises(ValueError):
            history.record(0,0)
        with self.assertRaises(ValueError):
            history.record(400_000_000,float("nan"))
        with self.assertRaises(ValueError):
            history.lookup(400_000_000,delay_s=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
