"""Offline NatNet decoder tests: no ROS initialization or network connection."""
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch


OPTITRACK = Path(__file__).resolve().parents[2] / "optitrack"
sys.path.insert(0, str(OPTITRACK))
from NatNetClient import NatNetClient


def rigid_body_packet(major, minor, flags=1, rigid_id=810):
    """Minimal rigid body, including legacy marker fields when required."""
    payload = struct.pack("<I3f4f", rigid_id, 1., 2., 3., 0., 0., 0., 1.)
    if 0 < major < 3:
        # Distinct marker ID/position catch accidental reuse of parser locals.
        payload += struct.pack("<I3f", 1, 9., 8., 7.)
        if major >= 2:
            payload += struct.pack("<If", 1234, .01)
    if major >= 2:
        payload += struct.pack("<f", .002)
    if major > 2 or (major == 2 and minor >= 6):
        payload += struct.pack("<h", flags)
    return payload


class NatNetTrackingTests(unittest.TestCase):
    def setUp(self):
        no_socket = patch("socket.socket", side_effect=AssertionError("Network access forbidden in decoder tests"))
        no_socket.start()
        self.addCleanup(no_socket.stop)
        self.client = NatNetClient()
        self.events = []
        self.client.rigid_body_listener = lambda *args: self.events.append(args)

    def parse(self, major, minor, flags=1):
        payload = rigid_body_packet(major, minor, flags)
        consumed, body = self.client._NatNetClient__unpack_rigid_body(payload, major, minor, 0)
        self.assertEqual(consumed, len(payload))
        self.assertEqual(body.id_num, 810)
        self.assertEqual(body.pos, (1., 2., 3.))
        return body

    def test_current_versions_dispatch_only_tracking_valid_bodies(self):
        for major, minor in ((3, 0), (4, 0)):
            for flags in (0, 1, 2, 3):
                with self.subTest(version=(major, minor), flags=flags):
                    self.events.clear()
                    body = self.parse(major, minor, flags)
                    valid = bool(flags & 1)
                    self.assertEqual(body.tracking_valid, valid)
                    self.assertEqual(len(self.events), int(valid))
                    if valid:
                        self.assertEqual(self.events[0], (810, (1., 2., 3.), (0., 0., 0., 1.)))

    def test_version_26_honors_flag_without_confusing_marker_and_body(self):
        for flags in (0, 1):
            with self.subTest(flags=flags):
                self.events.clear()
                body = self.parse(2, 6, flags)
                self.assertEqual(body.rb_marker_list[0].id, 1234)
                self.assertEqual(len(self.events), flags)
                if flags:
                    self.assertEqual(self.events[0], (810, (1., 2., 3.), (0., 0., 0., 1.)))

    def test_pre_26_streams_preserve_callback_without_a_validity_flag(self):
        for major, minor in ((1, 0), (2, 5)):
            with self.subTest(version=(major, minor)):
                self.events.clear()
                body = self.parse(major, minor)
                # The data class defaults to False even though no flag exists.
                self.assertFalse(body.tracking_valid)
                self.assertEqual(self.events, [(810, (1., 2., 3.), (0., 0., 0., 1.))])

    def test_unknown_version_keeps_existing_flagless_callback_behavior(self):
        self.parse(0, 0)
        self.assertEqual(self.events, [(810, (1., 2., 3.), (0., 0., 0., 1.))])

    def test_skeleton_bodies_use_same_validity_filter_without_losing_data(self):
        payload = (struct.pack("<II", 7, 2)
                   + rigid_body_packet(3, 0, 0, 111)
                   + rigid_body_packet(3, 0, 1, 112))
        consumed, skeleton = self.client._NatNetClient__unpack_skeleton(payload, 3, 0)
        self.assertEqual(consumed, len(payload))
        self.assertEqual([body.id_num for body in skeleton.rigid_body_list], [111, 112])
        self.assertEqual([event[0] for event in self.events], [112])

    def test_incomplete_quality_fields_do_not_dispatch_a_pose(self):
        payload = rigid_body_packet(3, 0)
        for stop in (32, 36, 37):
            with self.subTest(truncated_length=stop):
                self.events.clear()
                with self.assertRaises(struct.error):
                    self.client._NatNetClient__unpack_rigid_body(payload[:stop], 3, 0, 0)
                self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
