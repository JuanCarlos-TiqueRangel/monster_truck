#!/usr/bin/env python3

import math
import time

import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import AccelStamped, Twist, TwistStamped
from std_msgs.msg import Float32


SEND_PERIOD = 0.01
COMMAND_TIMEOUT = 0.3
MAX_TORQUE = 1.0
STEERING = 0.0

MANUAL = True

class DriveNode(Node):
    def __init__(self):
        super().__init__("drive_node")

        self.declare_parameter("port", "/dev/ttyACM0")
        port = self.get_parameter("port").value

        self.ser = serial.Serial(
            port,
            115200,
            timeout=0.05,
            write_timeout=0.1,
        )
        time.sleep(0.2)
        self.ser.reset_input_buffer()

        self.throttle = 0.0
        self.steering = 0.0
        self.last_command = 0.0
        self.armed = False

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,)

        if MANUAL:
            self.create_subscription(Twist, "/drive_manual", self.on_drive, qos)
        else:
            self.create_subscription(Float32, "/drive", self.on_drive, qos)
        self.create_timer(SEND_PERIOD, self.on_timer)
        self.get_logger().info(f"{port} open, waiting for /drive")


    def arm(self):
        self.ser.write(b"ARM\n")

        deadline = time.monotonic() + 0.3

        while time.monotonic() < deadline:
            if self.ser.readline().startswith(b"OK"):
                self.armed = True
                self.get_logger().info("armed")
                return

        self.get_logger().error("arm failed")


    def on_drive(self, msg):

        if MANUAL:
            throttle = msg.linear.x
            steering = msg.angular.z
        else:
            throttle = msg.data
            steering = STEERING

        if not math.isfinite(throttle) or not math.isfinite(steering):
            self.get_logger().error("Rejected non-finite command")
            return

        self.throttle = max(-1.0, min(1.0, throttle / MAX_TORQUE))
        self.steering = max(-1.0, min(1.0, steering))
        self.last_command = time.monotonic()

        if not self.armed:
            self.arm()


    def on_timer(self):
        age = time.monotonic() - self.last_command

        if age > COMMAND_TIMEOUT:
            value = 0
        else:
            value = int(round(self.throttle * 1000))

        steering_cmd = int(round(self.steering * 1000))

        self.ser.write(f"{value} {steering_cmd}\n".encode())

        print(f"\rwire {value:5d}   pulse {1500 + value // 2:4d} us", end="", flush=True)

        pending = self.ser.in_waiting

        if pending:
            for raw in self.ser.read(pending).splitlines():
                if not raw:
                    continue

                text = raw.decode(errors="replace")

                if text.startswith("DEADMAN"):
                    self.armed = False

                self.get_logger().warn(text)

    def stop(self):
        self.ser.write(b"0 0\n")
        self.ser.write(b"DISARM\n")
        self.ser.flush()
        time.sleep(0.05)
        self.ser.close()


def main():
    rclpy.init()

    node = None

    try:
        node = DriveNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()







































# #!/usr/bin/env python3
# """ROS 2 -> USB -> ESP-NOW latency bridge (protocol v3).

# Normal input: /drive, std_msgs/Float32, normalized PWM/throttle [-1, 1], NOT torque.
# Optional input: /drive_stamped, geometry_msgs/TwistStamped; linear.x=throttle,
# angular.z=steering; header.stamp set IMMEDIATELY BEFORE publisher.publish().
# The stamped mode requires shared/synchronized publisher and bridge clocks.

# callback_to_ack_ms is a HOST round trip, not one-way latency or motor response.
# source_to_ack_ms additionally includes publisher-stamp -> subscriber time.
# Only same-clock intervals are subtracted; no ESP32/PC clock synchronization assumed.
# Hardware PWM changes on a frame boundary, AFTER the software update acknowledged here.

# Run: python3 transmitter_node.py --ros-args -p port:=/dev/ttyACM0
# Bench safely, no arming: add -p auto_arm:=false
# Stamped mode: add -p stamped_input:=true -p source_clock_shared:=true
# Both matching C files MUST be flashed before using this node.
# """
# import csv
# import json
# import math
# import queue
# import secrets
# import statistics
# import threading
# import time
# from collections import Counter, deque
# from dataclasses import dataclass
# from datetime import datetime
# from pathlib import Path
# from typing import Optional

# import rclpy
# import serial
# from rclpy.clock import Clock, ClockType
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy
# from std_msgs.msg import Float32

# SEND_PERIOD = 0.01                 # Heartbeat / watchdog only, not a fresh-command gate.
# COMMAND_TIMEOUT = 0.3
# ACK_TIMEOUT = 0.5
# MAX_COMMAND = 1.0                 # Normalized PWM input scale, NOT physical torque.
# STEERING = 0.0
# PROTOCOL_VERSION = 3
# SERIAL_READ_TIMEOUT = 0.02
# SERIAL_WRITE_TIMEOUT = 0.02
# MAX_PENDING = 2048
# STATS_WINDOW = 10000

# CSV_FIELDS = [
#     "status", "session", "command_id", "wire_seq", "command", "throttle_int", "steering_int", "expected_armed",
#     "source_stamp_ns", "callback_wall_ns", "callback_mono_ns", "serial_write_mono_ns", "ack_mono_ns",
#     "callback_to_serial_ms", "callback_to_ack_ms", "source_to_callback_ms", "source_to_ack_ms",
#     "tx_prepare_ms", "espnow_rtt_ms", "radio_half_rtt_est_ms", "rx_queue_ms", "rx_apply_ms",
#     "rx_turnaround_ms", "tx_ack_to_usb_task_ms", "receiver_status", "throttle_us", "steering_us", "receiver_flags", "tx_send_status",
# ]


# @dataclass
# class Pending:
#     command_id: int
#     command: float
#     throttle: int
#     steering: int
#     expected_armed: bool
#     source_stamp_ns: int
#     callback_wall_ns: int
#     callback_ns: int
#     write_ns: int
#     source_to_callback_ms: Optional[float]


# def percentile(values, percent):
#     """Linear-interpolated percentile; no NumPy dependency."""
#     if not values:
#         return None
#     data = sorted(values)
#     position = (len(data) - 1) * percent / 100.0
#     low, high = math.floor(position), math.ceil(position)
#     return data[low] + (data[high] - data[low]) * (position - low)


# def describe(values):
#     if not values:
#         return {}
#     return {"n": len(values), "min_ms": min(values), "mean_ms": statistics.fmean(values),
#             "median_ms": statistics.median(values), "std_ms": statistics.pstdev(values),
#             "p95_ms": percentile(values, 95), "p99_ms": percentile(values, 99), "max_ms": max(values)}


# class DriveNode(Node):
#     def __init__(self):
#         super().__init__("drive_node")
#         self.declare_parameter("port", "/dev/ttyACM0")
#         self.declare_parameter("auto_arm", True)
#         self.declare_parameter("stamped_input", False)
#         self.declare_parameter("source_clock_shared", False)
#         self.declare_parameter("csv_dir", "latency_logs")
#         self.declare_parameter("ack_timeout", ACK_TIMEOUT)
#         self.declare_parameter("command_limit", MAX_COMMAND)
#         self.auto_arm = bool(self.get_parameter("auto_arm").value)
#         self.stamped_input = bool(self.get_parameter("stamped_input").value)
#         self.source_clock_shared = bool(self.get_parameter("source_clock_shared").value)
#         self.ack_timeout = float(self.get_parameter("ack_timeout").value)
#         self.command_limit = float(self.get_parameter("command_limit").value)
#         if not math.isfinite(self.command_limit) or not 0.0 < self.command_limit <= 1.0:
#             raise ValueError("command_limit must be in (0, 1], in normalized PWM units")
#         if not math.isfinite(self.ack_timeout) or not 0.05 <= self.ack_timeout <= 10.0:
#             raise ValueError("ack_timeout must be between 0.05 and 10 seconds")
#         if self.stamped_input and not self.source_clock_shared:
#             raise ValueError("Stamped input requires -p source_clock_shared:=true and shared/synchronized clocks")
#         if self.get_parameter("use_sim_time").value:
#             raise ValueError("Use real time, not use_sim_time, for this hardware latency test")
#         default_topic = "/drive_stamped" if self.stamped_input else "/drive"
#         self.declare_parameter("drive_topic", default_topic)
#         topic = str(self.get_parameter("drive_topic").value)
#         self.session = secrets.randbelow(0xFFFFFFFF) + 1
#         self.sequence = 0
#         self.armed = False
#         self.last_command_ns = 0
#         self.last_write_ns = 0
#         self.pending = {}
#         self.pending_lock = threading.Lock()
#         self.io_lock = threading.Lock()
#         self.stop_event = threading.Event()
#         self.logger_stop = threading.Event()
#         self.fault = threading.Event()
#         self.rows = queue.Queue(maxsize=8192)
#         self.notices = queue.Queue(maxsize=128)
#         self.late_or_duplicate = 0
#         self.dropped_log_rows = 0
#         self.rejected = 0
#         self.submitted = 0
#         self.latest = None
#         self.last_published_id = 0
#         self.closed = False
#         self.fault_disarm_attempted = False
#         self.log_path = Path(str(self.get_parameter("csv_dir").value)).expanduser().resolve()
#         self.log_path.mkdir(parents=True, exist_ok=True)
#         label = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
#         self.csv_path = self.log_path / f"drive_latency_{label}.csv"
#         self.summary_path = self.csv_path.with_suffix(".summary.json")
#         self.csv_file = self.csv_path.open("x", newline="", buffering=65536)
#         self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_FIELDS, extrasaction="raise")
#         self.csv_writer.writeheader()
#         port = str(self.get_parameter("port").value)
#         try:
#             self.ser = serial.Serial(port, 115200, timeout=SERIAL_READ_TIMEOUT, write_timeout=SERIAL_WRITE_TIMEOUT)
#             self._handshake()  # Startup only. No waiting for ACKs inside on_drive().
#         except Exception:
#             if hasattr(self, "ser"):
#                 self.ser.close()
#             self.csv_file.close()
#             raise
#         self.reader = threading.Thread(target=self._read_loop, name="serial-acks", daemon=True)
#         self.logger_thread = threading.Thread(target=self._log_loop, name="latency-csv", daemon=True)
#         self.logger_thread.start()
#         self.reader.start()
#         qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
#         if self.stamped_input:
#             from geometry_msgs.msg import TwistStamped
#             self.subscription = self.create_subscription(TwistStamped, topic, self.on_drive_stamped, qos)
#         else:
#             self.subscription = self.create_subscription(Float32, topic, self.on_drive, qos)
#         self.latency_pub = self.create_publisher(Float32, "/drive_latency/callback_to_ack_ms", 10)
#         self.source_pub = self.create_publisher(Float32, "/drive_latency/source_to_ack_ms", 10)
#         self.radio_pub = self.create_publisher(Float32, "/drive_latency/espnow_rtt_ms", 10)
#         self.timer = self.create_timer(SEND_PERIOD, self.on_timer, clock=Clock(clock_type=ClockType.STEADY_TIME))
#         self.get_logger().info(f"{port} ready; input={topic}; auto_arm={self.auto_arm}; PWM limit={self.command_limit}")
#         self.get_logger().info(f"CSV: {self.csv_path}")
#         self.get_logger().info("Reported host latency includes the ACK return path; it is NOT one-way or physical PWM latency.")

#     def _handshake(self):
#         time.sleep(0.2)
#         self.ser.reset_input_buffer()
#         deadline, next_try, buffer = time.monotonic() + 3.0, 0.0, bytearray()
#         expected = f"READY {PROTOCOL_VERSION} {self.session}"
#         while time.monotonic() < deadline:
#             if time.monotonic() >= next_try:
#                 data = f"\nHELLO {self.session}\n".encode("ascii")
#                 if self.ser.write(data) != len(data):
#                     raise RuntimeError("Partial USB write during handshake")
#                 next_try = time.monotonic() + 0.3
#             buffer.extend(self.ser.read(max(1, min(self.ser.in_waiting, 4096))))
#             while b"\n" in buffer:
#                 line, _, buffer = buffer.partition(b"\n")
#                 if line.decode("ascii", errors="replace").strip() == expected:
#                     return
#             if len(buffer) > 4096:
#                 buffer.clear()
#         raise RuntimeError("No protocol-v3 transmitter response. Flash transmitter_serial.c and close other serial readers.")

#     def _notice(self, text):
#         try:
#             self.notices.put_nowait(text)
#         except queue.Full:
#             pass

#     def _emit(self, row):
#         try:
#             self.rows.put_nowait(row)
#         except queue.Full:
#             self.dropped_log_rows += 1
#             self.fault.set()  # Do not silently run an experiment with incomplete telemetry.

#     def _row(self, pending, status):
#         return {"status": status, "session": self.session, "command_id": pending.command_id,
#                 "command": pending.command, "throttle_int": pending.throttle, "steering_int": pending.steering,
#                 "expected_armed": int(pending.expected_armed), "source_stamp_ns": pending.source_stamp_ns or "",
#                 "callback_wall_ns": pending.callback_wall_ns, "callback_mono_ns": pending.callback_ns,
#                 "serial_write_mono_ns": pending.write_ns,
#                 "callback_to_serial_ms": (pending.write_ns - pending.callback_ns) / 1e6,
#                 "source_to_callback_ms": pending.source_to_callback_ms if pending.source_to_callback_ms is not None else ""}

#     def _write(self, data, pending=None):
#         try:
#             with self.io_lock:
#                 if pending is not None:
#                     pending.write_ns = time.perf_counter_ns()
#                     if pending.write_ns - pending.callback_ns > int(COMMAND_TIMEOUT * 1e9):
#                         raise serial.SerialTimeoutException("Command became stale before USB write")
#                 if self.ser.write(data) != len(data):
#                     raise serial.SerialTimeoutException("Partial USB write")
#             self.last_write_ns = time.perf_counter_ns()
#             return True
#         except (serial.SerialException, OSError) as error:
#             self.fault.set()
#             self._notice(f"USB write failed: {error}; stopping commands, hardware deadman remains active")
#             return False

#     def on_drive(self, msg):
#         start_ns = time.perf_counter_ns()  # First action inside the subscriber callback.
#         self._command(float(msg.data), STEERING, start_ns, time.time_ns(), 0, None)

#     def on_drive_stamped(self, msg):
#         start_ns = time.perf_counter_ns()
#         callback_ros_ns = self.get_clock().now().nanoseconds
#         source_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)
#         age_ms = (callback_ros_ns - source_ns) / 1e6
#         if source_ns <= 0 or age_ms < 0 or age_ms > COMMAND_TIMEOUT * 1000:
#             self.rejected += 1
#             self._notice("Rejected stale/future/zero publisher timestamp; check clock synchronization")
#             self._disarm()
#             return
#         self._command(float(msg.twist.linear.x), float(msg.twist.angular.z), start_ns, time.time_ns(), source_ns, age_ms)

#     def _command(self, value, steering, start_ns, wall_ns, source_ns, source_delay):
#         if self.fault.is_set() or self.closed:
#             return
#         if not math.isfinite(value) or not math.isfinite(steering):
#             self.rejected += 1
#             self._notice("Rejected non-finite command; disarming")
#             self._disarm()
#             return
#         value = max(-self.command_limit, min(self.command_limit, value))
#         throttle = int(round(value * 1000))
#         steering_int = int(round(max(-1.0, min(1.0, steering)) * 1000))
#         self.sequence += 1
#         if self.sequence > 0xFFFFFFFF:
#             self.fault.set()
#             self._notice("Command-ID range exhausted; restart the node to create a new session")
#             self._disarm()
#             return
#         prefix = ""
#         if self.auto_arm and not self.armed:
#             prefix = f"ARM {self.session}\n"
#             self.armed = True  # Request state only; the receiver ACK confirms actual packet flags.
#         line = f"{prefix}CMD {self.session} {self.sequence} {throttle} {steering_int}\n".encode("ascii")
#         write_ns = time.perf_counter_ns()
#         pending = Pending(self.sequence, value, throttle, steering_int, self.armed, source_ns, wall_ns, start_ns, write_ns, source_delay)
#         with self.pending_lock:
#             if len(self.pending) >= MAX_PENDING:
#                 self.fault.set()
#                 self._notice("Too many unacknowledged commands; stopping")
#                 return
#             self.pending[self.sequence] = pending
#         self.submitted += 1
#         self.last_command_ns = start_ns
#         # Fresh commands go straight to USB; no 10-ms timer gate and no arm-readline wait.
#         if not self._write(line, pending):
#             with self.pending_lock:
#                 failed = self.pending.pop(pending.command_id, None)
#             if failed is not None:
#                 self._emit(self._row(failed, "serial_error"))
#             self.armed = False

#     def _disarm(self):
#         self.last_command_ns = 0
#         self.armed = False
#         self._write(f"\nDISARM {self.session}\n".encode("ascii"))

#     def on_timer(self):
#         now = time.perf_counter_ns()
#         if self.fault.is_set():
#             if not self.fault_disarm_attempted:
#                 self.fault_disarm_attempted = True
#                 self._disarm()
#             return
#         if self.last_command_ns and now - self.last_command_ns > int(COMMAND_TIMEOUT * 1e9):
#             self._disarm()
#             self._notice("/drive timeout: neutral + DISARM; a fresh command may auto-arm again")
#         elif self.last_command_ns and now - self.last_write_ns >= int(SEND_PERIOD * 1e9):
#             self._write(f"KEEP {self.session}\n".encode("ascii"))
#         with self.pending_lock:
#             latest = self.latest
#         if latest is not None and latest[0] != self.last_published_id:
#             cid, callback_ms, source_ms, radio_ms = latest
#             self.latency_pub.publish(Float32(data=callback_ms))
#             self.radio_pub.publish(Float32(data=radio_ms))
#             if source_ms is not None:
#                 self.source_pub.publish(Float32(data=source_ms))
#             self.last_published_id = cid

#     def _handle_line(self, text, arrival_ns):
#         parts = text.split()
#         if not parts:
#             return
#         if parts[0] == "ACK":
#             if len(parts) != 14:
#                 self._notice("Malformed ACK ignored")
#                 return
#             try:
#                 sid, cid, wire, prep, rtt, rxq, apply, turn, status, tp, sp, flags, bridge = map(int, parts[1:])
#             except ValueError:
#                 self._notice("Non-integer ACK ignored")
#                 return
#             if sid != self.session:
#                 return
#             if min(prep, rtt, rxq, apply, turn, bridge) < 0 or not 1000 <= tp <= 2000 or not 1000 <= sp <= 2000:
#                 self._notice("Invalid ACK fields ignored")
#                 return
#             with self.pending_lock:
#                 pending = self.pending.pop(cid, None)
#             if pending is None:
#                 self.late_or_duplicate += 1
#                 return
#             callback_ms = (arrival_ns - pending.callback_ns) / 1e6
#             if callback_ms < 0:
#                 self._emit(self._row(pending, "invalid_host_time"))
#                 return
#             expected_tp = 1500 + int(pending.throttle / 2) if pending.expected_armed else 1500
#             expected_sp = 1500 + int(pending.steering / 2) if pending.expected_armed else 1500
#             success = status == 0 and bool(flags & 2) and bool(flags & 1) == pending.expected_armed and tp == expected_tp and sp == expected_sp
#             outcome = "ack" if success else "receiver_error"
#             if callback_ms > self.ack_timeout * 1000:
#                 outcome = "ack_late" if success else "receiver_error_late"
#             source_ms = pending.source_to_callback_ms + callback_ms if pending.source_to_callback_ms is not None else None
#             row = self._row(pending, outcome)
#             row.update({"wire_seq": wire, "ack_mono_ns": arrival_ns, "callback_to_ack_ms": callback_ms,
#                         "source_to_ack_ms": source_ms if source_ms is not None else "", "tx_prepare_ms": prep / 1000.0,
#                         "espnow_rtt_ms": rtt / 1000.0, "rx_queue_ms": rxq / 1000.0, "rx_apply_ms": apply / 1000.0,
#                         "rx_turnaround_ms": turn / 1000.0, "tx_ack_to_usb_task_ms": bridge / 1000.0,
#                         "radio_half_rtt_est_ms": (rtt - turn) / 2000.0 if rtt >= turn else "",
#                         "receiver_status": status, "throttle_us": tp, "steering_us": sp, "receiver_flags": flags})
#             # Half adjusted radio RTT assumes symmetric directions; it is NOT measured one-way delay.
#             self._emit(row)
#             if success:
#                 with self.pending_lock:
#                     self.latest = (cid, callback_ms, source_ms, rtt / 1000.0)
#         elif parts[0] == "TXERR" and len(parts) == 4:
#             try:
#                 sid, cid, code = map(int, parts[1:])
#             except ValueError:
#                 return
#             if sid == self.session:
#                 with self.pending_lock:
#                     pending = self.pending.pop(cid, None)
#                 if pending is not None:
#                     row = self._row(pending, "tx_send_error")
#                     row["tx_send_status"] = code
#                     self._emit(row)
#         elif parts[0] in ("DEADMAN", "BOOT", "ERR"):
#             self.fault.set()  # Don't silently re-arm after transmitter reset / protocol desynchronization.
#             self._notice(f"Transmitter: {text}; restart node after inspecting the problem")
#         elif parts[0] not in ("OK", "READY"):
#             self._notice(text)

#     def _expire_pending(self, now_ns, status="ack_timeout", expire_all=False):
#         with self.pending_lock:
#             ids = [cid for cid, p in self.pending.items() if expire_all or now_ns - p.callback_ns > int(self.ack_timeout * 1e9)]
#             expired = [self.pending.pop(cid) for cid in ids]
#         for pending in expired:
#             self._emit(self._row(pending, status))

#     def _read_loop(self):
#         buffer = bytearray()
#         discard = False
#         try:
#             while not self.stop_event.is_set():
#                 # Read available bytes, or block for ONE byte, not a fixed 4096-byte read.
#                 data = self.ser.read(max(1, min(self.ser.in_waiting, 4096)))
#                 arrival_ns = time.perf_counter_ns()
#                 for byte in data:
#                     if byte == 10:
#                         if not discard and buffer:
#                             self._handle_line(buffer.decode("ascii", errors="replace").strip(), arrival_ns)
#                         buffer.clear()
#                         discard = False
#                     elif not discard:
#                         if len(buffer) < 512:
#                             buffer.append(byte)
#                         else:
#                             buffer.clear()
#                             discard = True
#                 self._expire_pending(time.perf_counter_ns())
#         except (serial.SerialException, OSError) as error:
#             self.fault.set()
#             self._notice(f"USB reader failed: {error}")
#         except Exception as error:
#             self.fault.set()
#             self._notice(f"ACK parser stopped: {error!r}")

#     def _log_loop(self):
#         counts = Counter()
#         samples = {key: deque(maxlen=STATS_WINDOW) for key in ("callback_to_ack_ms", "source_to_ack_ms", "espnow_rtt_ms")}
#         next_report = time.monotonic() + 1.0
#         try:
#             while not self.logger_stop.is_set() or not self.rows.empty():
#                 try:
#                     row = self.rows.get(timeout=0.05)
#                     self.csv_writer.writerow(row)
#                     counts[row["status"]] += 1
#                     if row["status"] in ("ack", "ack_late"):
#                         for key in samples:
#                             if row.get(key, "") != "":
#                                 samples[key].append(row[key])
#                 except queue.Empty:
#                     pass
#                 if time.monotonic() >= next_report:
#                     self.csv_file.flush()
#                     while not self.notices.empty():
#                         try:
#                             self.get_logger().warning(self.notices.get_nowait())
#                         except queue.Empty:
#                             break
#                     stats = describe(samples["callback_to_ack_ms"])
#                     if stats:
#                         source_stats = describe(samples["source_to_ack_ms"])
#                         source_text = f" | source->ACK mean={source_stats['mean_ms']:.3f}ms" if source_stats else ""
#                         self.get_logger().info(
#                             f"callback->ACK window n={stats['n']} mean={stats['mean_ms']:.3f} median={stats['median_ms']:.3f} "
#                             f"std={stats['std_ms']:.3f} p95={stats['p95_ms']:.3f} p99={stats['p99_ms']:.3f} max={stats['max_ms']:.3f} ms "
#                             f"| timeouts={counts['ack_timeout']} txerr={counts['tx_send_error']}{source_text}")
#                     elif self.submitted:
#                         self.get_logger().warning(f"No successful command ACKs yet; check both MACs/channel/firmware. timeouts={counts['ack_timeout']}")
#                     next_report = time.monotonic() + 1.0
#             with self.pending_lock:
#                 remaining = len(self.pending)
#             completed = sum(count for status, count in counts.items() if status != "stopped_pending")
#             missing_or_late = counts["ack_timeout"] + counts["ack_late"] + counts["receiver_error_late"]
#             summary = {"session": self.session, "csv": str(self.csv_path), "submitted": self.submitted,
#                        "outcomes": dict(counts), "pending": remaining, "rejected": self.rejected,
#                        "late_or_duplicate_acks": self.late_or_duplicate, "dropped_log_rows": self.dropped_log_rows,
#                        "source_clock_shared": self.source_clock_shared, "stamped_input": self.stamped_input,
#                        "ack_timeout_seconds": self.ack_timeout, "statistics_window_limit": STATS_WINDOW,
#                        "no_ack_by_deadline_fraction_of_completed": missing_or_late / completed if completed else None,
#                        "statistics": {key: describe(value) for key, value in samples.items()},
#                        "notes": ["Statistics cover at most the latest 10000 successful acknowledged commands, including matched late ACKs.",
#                                  "Timeouts are not radio packet-loss measurements; commands/ACKs can be superseded, dropped or delayed.",
#                                  "RTT includes ACK return path; receiver software update is not physical PWM or vehicle motion.",
#                                  "radio_half_rtt_est_ms assumes symmetric radio directions; no clock synchronization is performed."]}
#             self.csv_file.flush()
#             self.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
#         except Exception as error:
#             self.fault.set()
#             self.get_logger().error(f"CSV/statistics failed: {error!r}; stopping commands")
#         finally:
#             self.csv_file.close()

#     def stop(self):
#         if self.closed:
#             return
#         self.closed = True
#         if hasattr(self, "timer"):
#             self.timer.cancel()
#         self._disarm()
#         time.sleep(0.03)
#         self._disarm()  # Best effort; TX deadman / RX timeout remain independent.
#         self.stop_event.set()
#         self.reader.join(timeout=0.5)
#         self.ser.close()
#         self.reader.join(timeout=0.5)
#         self._expire_pending(time.perf_counter_ns(), status="stopped_pending", expire_all=True)
#         self.logger_stop.set()
#         self.logger_thread.join(timeout=3.0)
#         self.get_logger().info(f"Saved {self.csv_path}; summary: {self.summary_path}")


# def main():
#     rclpy.init()
#     node = None
#     try:
#         node = DriveNode()
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         if node is not None:
#             node.stop()
#             node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()