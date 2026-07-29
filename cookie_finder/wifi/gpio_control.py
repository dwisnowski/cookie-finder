"""
WiFi mode button + LED on Orange Pi Zero 2W GPIO.

Pins (gpiochip1) — kept clear of pan/tilt motor pins (phys 22–28, 31–37):
  LED    physical 7  / PWM3  / offset 269  (active-high)
  Button physical 11 / TXD.5 / offset 226  (active-low, internal pull-up)

LED patterns:
  client     solid ON
  ap         slow blink ~1 Hz  (500 ms on / 500 ms off)
  switching  fast blink ~5 Hz  (100 ms on / 100 ms off)
  other      OFF
"""

from __future__ import annotations

import platform
import threading
import time
from typing import Optional

from cookie_finder.wifi.manager import get_wifi_status, set_wifi_mode

try:
    import gpiod
    from gpiod.line import Bias, Direction, Value

    GPIO_AVAILABLE = True
except ImportError as e:
    GPIO_AVAILABLE = False
    gpiod = None  # type: ignore[assignment]
    Bias = Direction = Value = None  # type: ignore[misc, assignment]
    print(f"[wifi-gpio] WARNING: gpiod not available: {e}")

# gpiochip1 offsets (Orange Pi Zero 2W) — unused by Rust/Python gimbal
LED_PIN = 269  # physical 7, PWM3
BUTTON_PIN = 226  # physical 11, TXD.5
GPIO_CHIP = "/dev/gpiochip1"

POLL_INTERVAL_S = 0.025
DEBOUNCE_S = 0.05
STATUS_REFRESH_S = 2.0
SLOW_BLINK_HALF_S = 0.5  # ~1 Hz
FAST_BLINK_HALF_S = 0.1  # ~5 Hz


class WifiGpioController:
    """Background button watcher + LED indicator for WiFi AP/client mode."""

    def __init__(
        self,
        *,
        led_pin: int = LED_PIN,
        button_pin: int = BUTTON_PIN,
        chip_path: str = GPIO_CHIP,
    ) -> None:
        self.led_pin = led_pin
        self.button_pin = button_pin
        self.chip_path = chip_path

        self._chip = None
        self._lines = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._led_on = False
        self._last_button_raw = True  # pulled high when not pressed
        self._stable_button = True
        self._debounce_until = 0.0
        self._blink_phase = False
        self._blink_deadline = 0.0
        self._cached_status: dict = {}
        self._status_deadline = 0.0

    @property
    def available(self) -> bool:
        return (
            GPIO_AVAILABLE
            and platform.system() == "Linux"
            and self._lines is not None
        )

    def start(self) -> bool:
        """Initialize GPIO and start the control loop. Returns True on success."""
        if self._thread is not None and self._thread.is_alive():
            return True

        if not GPIO_AVAILABLE:
            print("[wifi-gpio] gpiod not installed; button/LED disabled")
            return False
        if platform.system() != "Linux":
            print("[wifi-gpio] not Linux; button/LED disabled")
            return False

        try:
            self._chip = gpiod.Chip(self.chip_path)
            config = {
                self.led_pin: gpiod.LineSettings(direction=Direction.OUTPUT),
                self.button_pin: gpiod.LineSettings(
                    direction=Direction.INPUT,
                    bias=Bias.PULL_UP,
                ),
            }
            self._lines = self._chip.request_lines(
                consumer="cookie-finder-wifi",
                config=config,
            )
            self._set_led(False)
            print(
                f"[wifi-gpio] ready (LED={self.led_pin} button={self.button_pin} "
                f"chip={self.chip_path})"
            )
        except Exception as e:
            print(f"[wifi-gpio] GPIO init failed: {e}")
            self._cleanup_gpio()
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="wifi-gpio",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the loop and release GPIO."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._cleanup_gpio()
        print("[wifi-gpio] stopped")

    def run_forever(self) -> None:
        """Start and block until stop() is called (or SIGTERM handled by caller)."""
        if not self.start():
            raise RuntimeError("WiFi GPIO controller failed to start")
        try:
            while not self._stop.is_set():
                time.sleep(0.25)
        finally:
            self.stop()

    def _cleanup_gpio(self) -> None:
        try:
            if self._lines is not None:
                try:
                    self._set_led(False)
                except Exception:
                    pass
                self._lines.release()
        except Exception:
            pass
        self._lines = None
        try:
            if self._chip is not None:
                self._chip.close()
        except Exception:
            pass
        self._chip = None

    def _set_led(self, on: bool) -> None:
        if self._lines is None or not GPIO_AVAILABLE:
            return
        value = Value.ACTIVE if on else Value.INACTIVE
        self._lines.set_values({self.led_pin: value})
        self._led_on = on

    def _read_button_released(self) -> bool:
        """Return True when button is not pressed (line high)."""
        if self._lines is None or not GPIO_AVAILABLE:
            return True
        # gpiod returns a list ordered like the offsets we pass — not a pin-keyed dict.
        values = self._lines.get_values([self.button_pin])
        return values[0] == Value.ACTIVE

    def _refresh_status(self, now: float) -> dict:
        if now >= self._status_deadline:
            self._cached_status = get_wifi_status()
            self._status_deadline = now + STATUS_REFRESH_S
        return self._cached_status

    def _handle_button(self, now: float) -> None:
        raw_released = self._read_button_released()
        if raw_released != self._last_button_raw:
            self._last_button_raw = raw_released
            self._debounce_until = now + DEBOUNCE_S
            return

        if now < self._debounce_until:
            return

        if raw_released == self._stable_button:
            return

        self._stable_button = raw_released
        # Falling edge: released → pressed
        if raw_released:
            return

        status = self._refresh_status(now)
        if not status.get("supported", True):
            print(f"[wifi-gpio] button ignored: {status.get('reason') or 'unsupported'}")
            return
        if status.get("switching"):
            print("[wifi-gpio] button ignored (switch already in progress)")
            return

        mode = status.get("mode")
        ssid = status.get("ssid")
        # Only enter AP when already associated as a client. Otherwise repair
        # client mode (common after a failed/partial AP switch left wlan0
        # "managed" with no SSID — toggling to AP would strand you again).
        if mode == "ap":
            target = "client"
        elif mode == "client" and ssid:
            target = "ap"
        else:
            target = "client"
            print(
                f"[wifi-gpio] button → repair client "
                f"(mode={mode!r} ssid={ssid!r})"
            )
        print(f"[wifi-gpio] button → switch to {target}")
        result = set_wifi_mode(target, delay_seconds=0.0)
        self._cached_status = result.get("wifi") or get_wifi_status()
        self._status_deadline = now + STATUS_REFRESH_S
        if result.get("status") in ("error", "busy"):
            print(f"[wifi-gpio] switch failed: {result.get('message')}")

    def _update_led(self, now: float, status: dict) -> None:
        if status.get("switching"):
            half = FAST_BLINK_HALF_S
        elif status.get("mode") == "ap":
            half = SLOW_BLINK_HALF_S
        elif status.get("mode") == "client":
            if not self._led_on:
                self._set_led(True)
            return
        else:
            if self._led_on:
                self._set_led(False)
            return

        if now >= self._blink_deadline:
            self._blink_phase = not self._blink_phase
            self._blink_deadline = now + half
            self._set_led(self._blink_phase)

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            try:
                self._handle_button(now)
                status = self._refresh_status(now)
                self._update_led(now, status)
            except Exception as e:
                print(f"[wifi-gpio] loop error: {e}")
                time.sleep(0.5)
            time.sleep(POLL_INTERVAL_S)
