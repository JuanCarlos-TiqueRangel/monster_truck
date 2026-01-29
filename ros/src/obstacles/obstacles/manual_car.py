#!/usr/bin/env python3
import sys
import select
import tty
import termios

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class KeyboardCmdActionNode(Node):
    """
    Publishes Float32 to /cmd_action using keyboard input.

    Controls:
      W = forward  (+amplitude) while held
      S = backward (-amplitude) while held
      SPACE = stop (0.0)
      Q = quit
    """

    def __init__(self):
        super().__init__("keyboard_cmd_action_node")
        self.cmd_pub = self.create_publisher(Float32, "cmd_action", 10)

        # Tunables
        self.amplitude = 0.6   # command magnitude
        self.pub_hz = 50.0     # publish rate

        self.current_cmd = 0.0

        # Put terminal in raw mode (non-blocking single-key reads)
        self._old_term_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        self.timer = self.create_timer(1.0 / self.pub_hz, self.timer_cb)

        self.get_logger().info(
            "Keyboard control started.\n"
            "Hold W: forward | Hold S: backward | Space: stop | Q: quit\n"
            "(Click this terminal window to focus key input)"
        )

    def destroy_node(self):
        # Restore terminal settings on shutdown
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term_settings)
        except Exception:
            pass
        super().destroy_node()

    def _read_key_nonblocking(self):
        """Return one character if available, else None."""
        dr, _, _ = select.select([sys.stdin], [], [], 0.0)
        if dr:
            return sys.stdin.read(1)
        return None

    def timer_cb(self):
        # Read all pending keypresses (if user typed multiple quickly)
        key = self._read_key_nonblocking()
        got_w = False
        got_s = False
        got_space = False
        got_q = False

        while key is not None:
            k = key.lower()
            if k == "w":
                got_w = True
            elif k == "s":
                got_s = True
            elif k == " ":
                got_space = True
            elif k == "q":
                got_q = True
            key = self._read_key_nonblocking()

        if got_q:
            self.get_logger().info("Quit requested (Q).")
            rclpy.shutdown()
            return

        # Priority: Space (stop) > W/S > default 0 if no new key
        if got_space:
            self.current_cmd = 0.0
        elif got_w and not got_s:
            self.current_cmd = +self.amplitude
        elif got_s and not got_w:
            self.current_cmd = -self.amplitude
        elif got_w and got_s:
            # both pressed quickly -> neutral
            self.current_cmd = 0.0
        else:
            # No new key received this tick -> set to 0.0 (release behavior)
            self.current_cmd = 0.0

        msg = Float32()
        msg.data = float(self.current_cmd)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardCmdActionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == "__main__":
    main()
