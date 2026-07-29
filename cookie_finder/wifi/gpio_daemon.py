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
import threading

from cookie_finder.wifi.gpio_control import WifiGpioController
from cookie_finder.wifi.manager import apply_boot_wifi_policy


def main() -> int:
    controller = WifiGpioController()

    def _handle_signal(signum: int, _frame) -> None:
        print(f"[wifi-gpio-daemon] received signal {signum}, shutting down")
        controller.stop()

    # Install handlers before any long work so systemctl stop is responsive.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    def _boot_restore() -> None:
        try:
            apply_boot_wifi_policy()
        except Exception as e:
            print(f"[wifi-gpio-daemon] boot WiFi restore failed: {e}", file=sys.stderr)

    # Client restore runs in the background so systemd/make return quickly.
    # Skips immediately when already associated (see apply_boot_wifi_policy).
    threading.Thread(
        target=_boot_restore,
        name="wifi-boot-policy",
        daemon=True,
    ).start()

    print("[wifi-gpio-daemon] starting (button toggles WiFi AP/client)")
    try:
        controller.run_forever()
    except RuntimeError as e:
        print(f"[wifi-gpio-daemon] fatal: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
