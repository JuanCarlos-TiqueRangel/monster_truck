#!/usr/bin/env python3
"""Preview or explicitly enable a manual-confirmation ROS 2 actuator test.

No ROS imports or motor commands occur unless --enable-drive is supplied.
Stop MPPI and every other /drive publisher before enabling this test.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


PERIOD_NS = 20_000_000
PULSE_NS = 200_000_000
NEUTRAL_NS = 3_000_000_000
MAX_CALLBACK_GAP_NS = 40_000_000


@dataclass(frozen=True)
class Trial:
    command: float
    repetition: int


def make_trials(direction):
    commands = []
    if direction in ("forward", "both"):
        commands.extend((0.11, 0.13))
    if direction in ("reverse", "both"):
        commands.extend((-0.11, -0.13))

    trials = []
    for command in commands:
        for repetition in range(1, 4):
            trials.append(Trial(command, repetition))
    return trials


class PulseSequence:
    """Timing only; no ROS, sensor filtering, or hardware access."""

    def __init__(self, trials):
        if not trials:
            raise ValueError("At least one trial is required.")
        self.trials = trials
        self.index = 0
        self.phase = "waiting"
        self.phase_start_ns = 0
        self.last_tick_ns = None
        self.abort_reason = ""

    @property
    def finished(self):
        return self.phase in ("done", "aborted")

    def start(self, now_ns):
        if self.phase != "waiting":
            return False
        self.phase = "baseline"
        self.phase_start_ns = now_ns
        return True

    def abort(self, reason):
        self.abort_reason = reason
        self.phase = "aborted"

    def tick(self, now_ns):
        if self.last_tick_ns is not None:
            gap_ns = now_ns - self.last_tick_ns
            if self.phase in ("baseline", "pulse"):
                if gap_ns < 0 or gap_ns > MAX_CALLBACK_GAP_NS:
                    self.abort("Timer callback gap exceeded 40 ms.")
        self.last_tick_ns = now_ns

        elapsed_ns = now_ns - self.phase_start_ns
        if self.phase == "baseline" and elapsed_ns >= NEUTRAL_NS:
            self.phase = "pulse"
            self.phase_start_ns = now_ns
        elif self.phase == "pulse" and elapsed_ns >= PULSE_NS:
            self.phase = "recovery"
            self.phase_start_ns = now_ns
        elif self.phase == "recovery" and elapsed_ns >= NEUTRAL_NS:
            self.index += 1
            self.phase = "waiting"
            if self.index == len(self.trials):
                self.phase = "done"

        if self.phase == "pulse":
            return self.trials[self.index].command
        return 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enable-drive", action="store_true",
        help="Enable actual /drive publishing. Without this, only preview the plan.",
    )
    parser.add_argument(
        "--direction", choices=("forward", "reverse", "both"), default="forward",
    )
    parser.add_argument("--robot-id", type=int, default=810)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "actuator_pulse_logs",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    trials = make_trials(args.direction)
    for index, trial in enumerate(trials, start=1):
        print(
            f"Trial {index}: u={trial.command:+.2f}, repeat {trial.repetition}/3; "
            "zero 3 s, pulse 0.20 s, zero at least 3 s, wait for Enter."
        )

    if not args.enable_drive:
        print("Preview only. No ROS node was started and no command was published.")
        return 0

    try:
        from actuator_pulse_node import run_ros_test
    except ModuleNotFoundError as error:
        print(f"ROS dependency missing: {error.name}. Run in your sourced ROS 2 environment.")
        return 1

    return run_ros_test(args, PulseSequence(trials))


if __name__ == "__main__":
    raise SystemExit(main())
