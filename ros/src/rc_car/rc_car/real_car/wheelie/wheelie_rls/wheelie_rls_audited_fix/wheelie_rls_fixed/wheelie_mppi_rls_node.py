#!/usr/bin/env python3
"""Frozen-model wheelie MPPI. Starts in DRY RUN; enable drive explicitly.

Requires approximately planar motion and the same velocity/pitch convention
as training. Pose and twist are paired by header timestamp. No acceleration
subscription, online RLS update, or solver-deadline zero-command gate.
"""
import csv
import hashlib
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float32
from params_mppi import WheelieParams, MPPIConfig
from rls import MODEL_TYPE, V_FEATURES, W_FEATURES
from mppi import MPPITorch

ROOT = Path(__file__).resolve().parent


class MPPIControllerNode(Node):
    def __init__(self):
        super().__init__('mppi_controller')
        self.declare_parameter('model_path', str(ROOT/'wheelie_nn'/'rls_learned_weights_v2.npz'))
        self.declare_parameter('drive_enabled', False)
        self.declare_parameter('reference_pitch_deg', -55.0)
        self.declare_parameter('object_id', 810)
        self.declare_parameter('state_timeout_s', 0.20)
        self.declare_parameter('stamp_tolerance_s', 0.01)
        self.declare_parameter('pitch_offset_rad', 0.0)
        self.declare_parameter('live_plot', False)
        self.declare_parameter('device', 'cuda')
        self.p, self.cfg = WheelieParams(), MPPIConfig()
        path = Path(self.get_parameter('model_path').value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'{path}: run train_rls.py first.')
        with np.load(path, allow_pickle=False) as data:
            if 'model_type' not in data or str(data['model_type'].item()) != MODEL_TYPE:
                raise ValueError('Legacy weights are incompatible. Retrain with the supplied train_rls.py.')
            if tuple(data['v_features']) != V_FEATURES or tuple(data['w_features']) != W_FEATURES:
                raise ValueError('Saved feature definitions do not match rls.py.')
            if str(data['target'].item()) != 'forward_velocity_difference_per_second':
                raise ValueError('Expected a discrete forward-target model.')
            if not math.isclose(float(data['nominal_sample_period']), self.cfg.dt, abs_tol=1e-6):
                raise ValueError('Control dt differs from the trained model sample period.')
            if self.p.u_min < float(data['observed_u_min'])-1e-6 or self.p.u_max > float(data['observed_u_max'])+1e-6:
                raise ValueError('Control bounds exceed the observed training-command range.')
            parameters = np.array(data['a'], dtype=float)
            self.cfg.input_delay_steps = int(data['input_delay_steps'])
        self.model_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.ref = np.array((0.0, 0.0, math.radians(self.get_parameter('reference_pitch_deg').value), 0.0))
        self.controller = MPPITorch(self.p, self.cfg, parameters, device=self.get_parameter('device').value, live_plot=self.get_parameter('live_plot').value)
        self.history = deque([0.0]*self.cfg.input_delay_steps, maxlen=self.cfg.input_delay_steps)
        self.u_prev = 0.0
        # Warm up before starting the control timer; these actions are never published.
        self.controller.solve(np.zeros(4), self.ref, 0.0, list(self.history))
        self.controller.reset()
        self.odom_buffer, self.twist_buffer = deque(maxlen=64), deque(maxlen=64)
        self.previous_wrapped, self.unwrapped_pitch = None, 0.0
        self.last_odom_stamp, self.last_twist_stamp = -math.inf, -math.inf
        self.log_records = []
        self.start_time, self.last_tick = time.monotonic(), None
        self.last_status, self.was_enabled = '', False
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        object_id = self.get_parameter('object_id').value
        self.odom_sub = self.create_subscription(Odometry, f'/optitrack/odom_{object_id}', self.odom_callback, qos)
        self.twist_sub = self.create_subscription(TwistStamped, f'/optitrack/twist_{object_id}', self.velocity_callback, qos)
        self.drive_pub = self.create_publisher(Float32, '/drive', 1)
        self.drive_pub.publish(Float32(data=0.0))
        self.timer = self.create_timer(self.cfg.dt, self.control_callback)
        self.get_logger().info(f'Frozen RLS MPPI: {1/self.cfg.dt:g} Hz; drive_enabled={self.get_parameter("drive_enabled").value}; model={self.model_sha256[:12]}')

    def odom_callback(self, msg):
        stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec
        q = msg.pose.pose.orientation
        quaternion = np.array((q.x, q.y, q.z, q.w), dtype=float)
        norm = np.linalg.norm(quaternion)
        xpos = float(msg.pose.pose.position.x)
        if stamp <= self.last_odom_stamp or not np.isfinite(quaternion).all() or not np.isfinite(xpos) or norm < 1e-8:
            return
        x, y, z, w = quaternion / norm
        # atan2(-R20, R22): yaw-independent planar pitch, continuous through +/-pi.
        # Body omega_y is pitch rate only for approximately planar (small-roll) motion.
        wrapped = math.atan2(2*(w*y-x*z), 1-2*(x*x+y*y))
        if self.previous_wrapped is None:
            self.unwrapped_pitch = wrapped
        else:
            self.unwrapped_pitch += math.atan2(math.sin(wrapped-self.previous_wrapped), math.cos(wrapped-self.previous_wrapped))
        self.previous_wrapped, self.last_odom_stamp = wrapped, stamp
        pitch = self.unwrapped_pitch - float(self.get_parameter('pitch_offset_rad').value)
        self.odom_buffer.append((stamp, time.monotonic(), xpos, pitch))

    def velocity_callback(self, msg):
        stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec
        v, omega = float(msg.twist.linear.x), float(msg.twist.angular.y)
        if stamp <= self.last_twist_stamp or not math.isfinite(v) or not math.isfinite(omega):
            return
        self.last_twist_stamp = stamp
        # Preserve the training CSV's velocity convention; do not silently rotate frames.
        self.twist_buffer.append((stamp, time.monotonic(), v, omega))

    def control_callback(self):
        tick = time.monotonic()
        cycle_dt = math.nan if self.last_tick is None else tick-self.last_tick
        self.last_tick = tick
        enabled = bool(self.get_parameter('drive_enabled').value)
        if enabled != self.was_enabled:
            self.controller.reset()
        self.was_enabled = enabled
        state, pair = None, None
        tolerance = float(self.get_parameter('stamp_tolerance_s').value)
        for odom in reversed(self.odom_buffer):
            if not self.twist_buffer:
                break
            twist = min(self.twist_buffer, key=lambda item: abs(item[0]-odom[0]))
            if abs(twist[0]-odom[0]) <= tolerance:
                pair = (odom, twist)
                state = np.array((odom[2], twist[2], odom[3], twist[3]))
                break
        age = math.inf if pair is None else max(tick-pair[0][1], tick-pair[1][1])
        info, planned, solve_ms = {}, 0.0, 0.0
        if state is None:
            status = 'waiting_for_timestamp_matched_state'
        elif age > float(self.get_parameter('state_timeout_s').value):
            status = 'stale_state'
        else:
            started = time.perf_counter()
            planned, info = self.controller.solve(state, self.ref, self.u_prev, list(self.history))
            solve_ms = 1000*(time.perf_counter()-started)
            status = ('driving' if enabled else 'dry_run') if info.get('success', False) else info.get('reason', 'solver_failure')
        applied = planned if enabled and info.get('success', False) else 0.0
        applied = float(np.clip(applied, self.p.u_min, self.p.u_max))
        self.drive_pub.publish(Float32(data=applied))
        # Only actual publications enter delay history; never insert a dry-run plan.
        self.history.append(applied)
        self.u_prev = applied
        if not info.get('success', False):
            self.controller.reset()
        if status != self.last_status:
            self.get_logger().info(status) if status in ('driving', 'dry_run') else self.get_logger().warning(status)
            self.last_status = status
        s = np.full(4, np.nan) if state is None else state
        self.log_records.append({'time': tick-self.start_time, 'cycle_dt_s': cycle_dt, 'x': s[0], 'v': s[1], 'pitch': s[2], 'pitch_deg': math.degrees(s[2]), 'omega': s[3], 'ref_pitch_deg': math.degrees(self.ref[2]), 'u_plan': planned, 'u_applied': applied, 'status': status, 'solve_ms': solve_ms, 'state_age_at_start_ms': 1000*age, 'state_age_at_publish_ms': 1000*age+solve_ms, 'stamp_skew_ms': math.nan if pair is None else 1000*abs(pair[0][0]-pair[1][0]), 'state_stamp': math.nan if pair is None else pair[0][0], 'cost': info.get('cost', math.nan), 'best_cost': info.get('best_cost', math.nan), 'pred_theta_N_deg': info.get('pred_theta_N_deg', math.nan), 'pred_omega_N': info.get('pred_omega_N', math.nan), 'effective_samples': info.get('effective_samples', math.nan), 'valid_fraction': info.get('valid_fraction', math.nan), 'mean_plan_rejected': info.get('mean_plan_rejected', False), 'model_sha256': self.model_sha256})

    def save_experiment_log(self):
        if not self.log_records:
            return
        path = ROOT/f'mppi_rls_log_{datetime.now():%Y%m%d_%H%M%S}.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.log_records[0]))
            writer.writeheader()
            writer.writerows(self.log_records)
        self.get_logger().info(f'Saved {path}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MPPIControllerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            # Stop before potentially slow disk I/O. Independent hardware timeout is still required.
            if rclpy.ok():
                node.drive_pub.publish(Float32(data=0.0))
            node.controller.close_plot()
            node.save_experiment_log()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
