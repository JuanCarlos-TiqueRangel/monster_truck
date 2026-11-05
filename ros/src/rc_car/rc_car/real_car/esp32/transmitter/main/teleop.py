from pynput import keyboard
import threading
import time


def teleop(link, throttle_value=225, steering_value=225):
    """
    Keyboard teleop with true key press/release behavior.

    Hold:
        w -> +throttle
        s -> -throttle
        a -> -steering
        d -> +steering

    Release the key -> corresponding command returns to zero.

    q -> quit
    space -> zero both axes

    Commands are resent at 50 Hz.
    """

    pressed = set()
    lock = threading.Lock()
    stop_event = threading.Event()

    def on_press(key):
        try:
            k = key.char.lower()
        except AttributeError:
            if key == keyboard.Key.space:
                k = "space"
            else:
                return

        if k == "q":
            stop_event.set()
            return False

        with lock:
            if k == "space":
                pressed.clear()
            elif k in ("w", "s", "a", "d"):
                pressed.add(k)

    def on_release(key):
        try:
            k = key.char.lower()
        except AttributeError:
            return

        with lock:
            pressed.discard(k)

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release,
    )

    print(
        f"Hold w/s: throttle +/-{throttle_value}   "
        f"Hold a/d: steering +/-{steering_value}   "
        f"space: zero   q: quit"
    )

    try:
        link.arm()
        listener.start()

        period = 0.02  # 50 Hz

        while not stop_event.is_set():

            with lock:
                keys = pressed.copy()

            # -------------------------
            # THROTTLE
            # -------------------------
            if "w" in keys and "s" not in keys:
                throttle = throttle_value
            elif "s" in keys and "w" not in keys:
                throttle = -throttle_value
            else:
                throttle = 0

            # -------------------------
            # STEERING
            # -------------------------
            if "d" in keys and "a" not in keys:
                steering = steering_value
            elif "a" in keys and "d" not in keys:
                steering = -steering_value
            else:
                steering = 0

            # Convert from [-1000, 1000] to [-1, 1]
            link.send(
                throttle / 1000.0,
                steering / 1000.0
            )

            link.poll()

            print(
                f"\rthrottle {throttle:5d}   "
                f"steering {steering:5d}",
                end="",
                flush=True,
            )

            time.sleep(period)

    finally:
        stop_event.set()
        listener.stop()

        # Explicitly send neutral before disarming
        link.send(0.0, 0.0)
        time.sleep(0.05)

        link.disarm()

        print("\nStopped.")