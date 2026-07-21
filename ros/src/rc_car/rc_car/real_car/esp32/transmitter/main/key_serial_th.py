#!/usr/bin/env python3
"""Host-side link to the ESP-NOW control bridge.

Wire protocol (ASCII lines, '\n' terminated):

  host -> esp   "<t> <s>"      throttle and steering, each in
                               [-1000, 1000] (per-mille of full
                               scale). No reply.
                "ARM"          zeroes both, enables output.
                               Reply: "OK ARM"
                "DISARM"       zeroes both, disables.
                               Reply: "OK DISARM"
                "STATUS"       Reply: "STATUS throttle=..
                               steering=.. armed=.. sendfail=.."

  esp -> host   "READY", "MAC xx:.."   once, at boot only.
                "DEADMAN"              esp auto-disarmed: no valid
                                       line from host for 300 ms.
                "ERR"                  unparseable line.

Pulse mapping on the receiver, both channels:
  -1000 -> 1000 us,  0 -> 1500 us (neutral),  +1000 -> 2000 us

The transmitter auto-disarms after 300 ms of host silence, so any
mode that holds a command must keep sending lines. The 100 Hz
control loop does this naturally; teleop resends at 50 Hz.
"""

import select
import sys
import termios
import time
import tty

import serial


class EspLink:
    def __init__(self, port="/dev/ttyACM0"):
        self.ser = serial.Serial(
            port,
            115200,          # ignored: native USB
            timeout=0.05,
            write_timeout=0.1,
        )
        time.sleep(0.2)
        self.poll()

    def poll(self):
        """Non-blocking: print any pending lines from the esp."""
        count = self.ser.in_waiting

        if count:
            data = self.ser.read(count)

            for raw in data.splitlines():
                if raw:
                    text = raw.decode(errors="replace")
                    print(f"esp: {text}")

    def send(self, throttle, steering):
        """throttle, steering in [-1.0, 1.0]."""
        t = int(round(max(-1.0, min(1.0, throttle)) * 1000))
        s = int(round(max(-1.0, min(1.0, steering)) * 1000))

        self.ser.write(f"{t} {s}\n".encode())

    def _command(self, text, wait=0.3):
        """Send a line and return the first reply line."""
        self.ser.write((text + "\n").encode())

        deadline = time.monotonic() + wait

        while time.monotonic() < deadline:
            line = self.ser.readline()

            if line:
                return line.decode(errors="replace").strip()

        return ""

    def arm(self):
        reply = self._command("ARM")

        if not reply.startswith("OK"):
            raise RuntimeError(f"arm failed: {reply!r}")

    def disarm(self):
        return self._command("DISARM")

    def status(self):
        return self._command("STATUS")

    def close(self):
        try:
            self.send(0.0, 0.0)
            self.disarm()
            self.ser.flush()
            time.sleep(0.05)
        finally:
            self.ser.close()


def control_loop(link, controller, period=0.01):
    """Run the controller at 1/period Hz.

    `controller.step()` must return (throttle, steering), each in
    [-1, 1]. Wire in your state source where marked. Occasional
    solver overruns are safe: the transmitter heartbeat repeats
    the last command (ZOH); only a >300 ms hang trips the deadman.
    """
    link.arm()

    next_t = time.monotonic()

    try:
        while True:
            # state = read_state()             # <-- your estimator
            throttle, steering = controller.step()

            link.send(throttle, steering)
            link.poll()

            next_t += period
            slack = next_t - time.monotonic()

            if slack > 0:
                time.sleep(slack)
            else:
                next_t = time.monotonic()      # overrun: resync
    finally:
        link.send(0.0, 0.0)
        link.disarm()


def teleop(link, step=100):
    """Keyboard drive. Resends at 50 Hz so the esp's 300 ms
    deadman never trips while a key is held or idle."""
    old_terminal = termios.tcgetattr(sys.stdin)

    throttle = 0
    steering = 0

    print(
        f"w/s: throttle +-{step}   "
        f"a/d: steering -+{step}   "
        f"space: zero both   q: quit"
    )

    try:
        tty.setcbreak(sys.stdin.fileno())

        link.arm()

        while True:
            ready, _, _ = select.select(
                [sys.stdin], [], [], 0.02
            )

            if ready:
                key = sys.stdin.read(1).lower()

                if key == "w":
                    throttle = min(throttle + step, 1000)
                elif key == "s":
                    throttle = max(throttle - step, -1000)
                elif key == "a":
                    steering = max(steering - step, -1000)
                elif key == "d":
                    steering = min(steering + step, 1000)
                elif key == " ":
                    throttle = 0
                    steering = 0
                elif key == "q":
                    break

            link.send(throttle / 1000, steering / 1000)
            link.poll()

            print(
                f"\rthrottle {throttle:5d}   "
                f"steering {steering:5d}",
                end="",
                flush=True,
            )
    finally:
        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_terminal,
        )

        print()

        link.send(0.0, 0.0)
        link.disarm()


if __name__ == "__main__":
    link = EspLink()

    print(link.status() or "no reply: check port / firmware")

    try:
        teleop(link)
    finally:
        link.close()