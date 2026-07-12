#!/usr/bin/env python3
"""
Interactive terminal control for the Rust gimbal daemon.

Arrow keys pan/tilt while held (press Space or s to stop). Keys 1-9 set
step rate for 28BYJ-48 motors on the Orange Pi (1 = slow, 9 = fast).
"""

from __future__ import annotations

import argparse
import curses
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_finder.gimbal.rust_client import RustGimbalClient

# 28BYJ-48 @ 5V: ~340 Hz nominal max; 4076 steps/rev (~0.0883 deg/step).
# Presets stay reliable at low speeds and allow faster motion on 7-9 for short moves.
SPEED_PRESETS: dict[int, int] = {
    1: 150,
    2: 250,
    3: 350,
    4: 450,
    5: 550,
    6: 700,
    7: 900,
    8: 1200,
    9: 1500,
}

DEFAULT_PRESET = 5
STEP_SIZE = 2
REFRESH_MS = 200

Direction = Optional[str]


class MotorController:
    """Background stepping while an arrow direction is active."""

    def __init__(self, client: RustGimbalClient, preset: int) -> None:
        self.client = client
        self.preset = preset
        self._direction: Direction = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._apply_speed()

    def _apply_speed(self) -> None:
        hz = SPEED_PRESETS[self.preset]
        self.client.set_speed(pan_hz=hz, tilt_hz=hz)

    def set_preset(self, preset: int) -> None:
        self.preset = preset
        self._apply_speed()

    def start(self, direction: str) -> None:
        with self._lock:
            self._direction = direction
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._direction = None
        try:
            self.client.stop()
        except (OSError, ValueError):
            pass

    def shutdown(self) -> None:
        self._stop.set()
        with self._lock:
            self._direction = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @property
    def direction(self) -> Direction:
        with self._lock:
            return self._direction

    @property
    def error(self) -> Optional[str]:
        return self._error

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                direction = self._direction
            if direction is None:
                time.sleep(0.05)
                continue
            try:
                if direction == "left":
                    self.client.pan_step(-1, steps=STEP_SIZE)
                elif direction == "right":
                    self.client.pan_step(1, steps=STEP_SIZE)
                elif direction == "up":
                    self.client.tilt_step(1, steps=STEP_SIZE)
                elif direction == "down":
                    self.client.tilt_step(-1, steps=STEP_SIZE)
            except (OSError, ValueError, KeyError) as exc:
                self._error = str(exc)
                with self._lock:
                    self._direction = None
                break


def _deg_per_sec(hz: int) -> float:
    return hz * (360.0 / 4076.0)


def _draw(stdscr: curses.window, motor: MotorController, client: RustGimbalClient) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    try:
        pan, tilt = client.get_position()
    except (OSError, ValueError, KeyError):
        pan, tilt = 0.0, 0.0

    hz = SPEED_PRESETS[motor.preset]
    lines = [
        "Cookie Finder – Keyboard Gimbal Control",
        "",
        f"Position:  pan {pan:6.1f}°   tilt {tilt:6.1f}°",
        f"Speed:     preset {motor.preset}  ({hz} Hz, ~{_deg_per_sec(hz):.1f}°/s)",
        f"Moving:    {motor.direction or 'stopped'}",
        "",
        "Controls:",
        "  Arrow keys     Pan / tilt (hold; press Space or s to stop)",
        "  1-9            Set step rate (1=slow, 9=fast)",
        "  h              Home (0°, 0°)",
        "  s / Space      Stop motion",
        "  q              Quit",
        "",
        "Requires: cookie-finder-ctl daemon (make on-the-pi-rust-daemon)",
    ]

    if motor.error:
        lines.extend(["", f"Error: {motor.error}"])

    for row, text in enumerate(lines):
        if row >= height:
            break
        stdscr.addnstr(row, 0, text, width - 1)

    stdscr.refresh()


def _run(stdscr: curses.window, client: RustGimbalClient, preset: int) -> int:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(REFRESH_MS)

    motor = MotorController(client, preset)

    try:
        while True:
            _draw(stdscr, motor, client)

            if motor.error:
                stdscr.timeout(2000)
                stdscr.getch()
                return 1

            key = stdscr.getch()
            if key == -1:
                continue

            if key in (ord("q"), ord("Q")):
                motor.stop()
                return 0

            if key in (ord("s"), ord("S"), ord(" ")):
                motor.stop()
                continue

            if key in (ord("h"), ord("H")):
                motor.stop()
                client.home()
                continue

            if ord("1") <= key <= ord("9"):
                motor.set_preset(key - ord("0"))
                continue

            if key == curses.KEY_LEFT:
                motor.start("left")
            elif key == curses.KEY_RIGHT:
                motor.start("right")
            elif key == curses.KEY_UP:
                motor.start("up")
            elif key == curses.KEY_DOWN:
                motor.start("down")
    finally:
        motor.shutdown()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Keyboard control for the Rust gimbal daemon (arrow keys + 1-9 speed)."
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("COOKIE_FINDER_SOCKET"),
        help="Unix socket path (default: /tmp/cookie-finder.sock or COOKIE_FINDER_SOCKET)",
    )
    parser.add_argument(
        "--preset",
        type=int,
        choices=sorted(SPEED_PRESETS),
        default=DEFAULT_PRESET,
        help=f"Initial speed preset 1-9 (default: {DEFAULT_PRESET})",
    )
    args = parser.parse_args(argv)

    client = RustGimbalClient.connect(socket_path=args.socket)
    if client is None:
        print(
            "Rust gimbal daemon not reachable. Start it with:\n"
            "  make on-the-pi-rust-daemon",
            file=sys.stderr,
        )
        return 1

    try:
        client.set_input_enabled(False)
    except (OSError, ValueError):
        pass

    return curses.wrapper(lambda stdscr: _run(stdscr, client, args.preset))


if __name__ == "__main__":
    raise SystemExit(main())
