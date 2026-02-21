import threading, sys, select

class KeyListener(threading.Thread):
    def __init__(self, on_key, logger=None, use_raw=True):
        super().__init__(daemon=True)   # IMPORTANT: not daemon
        self.on_key = on_key
        self.logger = logger
        self.use_raw = use_raw

        self._stop_event = threading.Event()

        # raw-mode bookkeeping
        self._fd = None
        self._old_termios = None
        self._raw_enabled = False


    def stop(self):
        self._stop_event.set()
        # Try to restore immediately from the main thread too
        self._restore_terminal()

    def _restore_terminal(self):
        if not self._raw_enabled:
            return
        try:
            import termios
            if self._fd is not None and self._old_termios is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        except Exception:
            pass
        finally:
            self._raw_enabled = False

    def run(self):
        if not self.use_raw:
            return self._run_fallback_line_input()

        try:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self._fd)

            tty.setcbreak(self._fd)
            self._raw_enabled = True

            if self.logger:
                self.logger.info("Keyboard: cbreak mode enabled (press keys without Enter).")

            while not self._stop_event.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if r:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.on_key(ch)

        except Exception:
            # If raw mode fails, fallback to line input
            if self.logger:
                self.logger.warn("Keyboard: raw mode not available; use 's'+Enter, 'x'+Enter, etc.")
            self._run_fallback_line_input()

        finally:
            self._restore_terminal()

    def _run_fallback_line_input(self):
        while not self._stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            ch = line.strip()[:1]
            if ch:
                self.on_key(ch)