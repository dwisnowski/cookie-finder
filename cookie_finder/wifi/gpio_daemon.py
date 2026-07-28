"""
Standalone WiFi GPIO daemon: button toggles AP/client, LED shows mode.

Run:
  python -m cookie_finder.wifi.gpio_daemon

Installed as systemd unit cookie-finder-wifi.service (make init-wifi).
Independent of the Cookie Finder web app.
"""

from __future__ import annotations

import signal
import sys

from cookie_finder.wifi.gpio_control import WifiGpioController


def main() -> int:
    controller = WifiGpioController()

    def _handle_signal(signum: int, _frame) -> None:
        print(f"[wifi-gpio-daemon] received signal {signum}, shutting down")
        controller.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("[wifi-gpio-daemon] starting (button toggles WiFi AP/client)")
    try:
        controller.run_forever()
    except RuntimeError as e:
        print(f"[wifi-gpio-daemon] fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
