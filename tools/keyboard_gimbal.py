#!/usr/bin/env python3
"""
Interactive terminal control for the Rust gimbal daemon.

Arrow keys pan/tilt while held (press Space or s to stop and disable coils).
Keys 1-9 set step rate for 28BYJ-48 / 24BYJ motors on the Orange Pi (1 = slow, 9 = fast).
M cycles coil drive mode (wave / full-step / half-step).
P/T select motor for wiring permutation; [ / ] cycle; W writes mapping to config.
"""

from __future__ import annotations

import argparse
import curses
import itertools
import os
import sys
import threading
import time
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_finder.gimbal.config import (
    format_phase_order_snippet,
    resolve_config_path,
    save_phase_order,
)
from cookie_finder.gimbal.rust_client import RustGimbalClient

# Presets 1–2 are slow enough for start-up / drive-mode troubleshooting (20–50 ms/step).
# Higher presets match typical 28BYJ-48 @ 5V operation.
SPEED_PRESETS: dict[int, int] = {
    1: 25,    # 40 ms/step
    2: 50,    # 20 ms/step
    3: 150,
    4: 250,
    5: 350,
    6: 450,
    7: 550,
    8: 700,
    9: 900,
}

DRIVE_MODES: list[tuple[str, str]] = [
    ("wave", "wave (1-coil)"),
    ("full", "full-step (2-coil)"),
    ("half", "half-step"),
]

PERMUTATIONS: list[tuple[int, ...]] = list(itertools.permutations((0, 1, 2, 3)))

DEFAULT_PRESET = 1
STEP_SIZE = 2
REFRESH_MS = 200
STATUS_CLEAR_SEC = 4.0

Direction = Optional[str]
MotorName = Literal["pan", "tilt"]


def _perm_index(order: list[int]) -> int:
    key = tuple(order)
    try:
        return PERMUTATIONS.index(key)
    except ValueError:
        return 0


def _format_wiring_line(
    label: str,
    order: list[int],
    perm_idx: int,
    selected: bool,
) -> str:
    mapping = "  ".join(f"IN{i + 1}→ph{order[i]}" for i in range(4))
    marker = " <<" if selected else ""
    return f"{label:4} wiring: {mapping}   [{perm_idx + 1}/{len(PERMUTATIONS)}]{marker}"


class DriveModeState:
    """Tracks and applies wave / full-step / half-step drive algorithms."""

    def __init__(self, client: RustGimbalClient) -> None:
        self.client = client
        self.mode = "wave"
        self.label = "wave (1-coil)"
        self.mode_index = 0
        self.status_message: Optional[str] = None
        self._status_until = 0.0
        self._sync_from_daemon()

    def _sync_from_daemon(self) -> None:
        try:
            resp = self.client.get_drive_mode()
            if resp.get("ok"):
                mode = str(resp.get("mode", "wave"))
                for i, (key, label) in enumerate(DRIVE_MODES):
                    if key == mode:
                        self.mode_index = i
                        self.mode = key
                        self.label = str(resp.get("label", label))
                        return
        except (OSError, ValueError, KeyError):
            pass

    def next_mode(self) -> None:
        self.mode_index = (self.mode_index + 1) % len(DRIVE_MODES)
        self._apply_current()

    def prev_mode(self) -> None:
        self.mode_index = (self.mode_index - 1) % len(DRIVE_MODES)
        self._apply_current()

    def _apply_current(self) -> None:
        mode, label = DRIVE_MODES[self.mode_index]
        resp = self.client.set_drive_mode(mode)
        if not resp.get("ok"):
            self.status_message = f"Drive mode failed: {resp.get('error', 'unknown')}"
            self._status_until = time.monotonic() + STATUS_CLEAR_SEC
            return
        self.mode = str(resp.get("mode", mode))
        self.label = str(resp.get("label", label))
        self.status_message = (
            f"Drive mode → {self.label}  [{self.mode_index + 1}/{len(DRIVE_MODES)}]"
        )
        self._status_until = time.monotonic() + STATUS_CLEAR_SEC

    def active_status(self) -> Optional[str]:
        if self.status_message and time.monotonic() < self._status_until:
            return self.status_message
        self.status_message = None
        return None


class WiringState:
    """Tracks and applies phase-order permutations via the daemon."""

    def __init__(self, client: RustGimbalClient) -> None:
        self.client = client
        self.selected: MotorName = "pan"
        self.pan_order: list[int] = [0, 1, 2, 3]
        self.tilt_order: list[int] = [0, 1, 2, 3]
        self.pan_perm_idx = 0
        self.tilt_perm_idx = 0
        self.status_message: Optional[str] = None
        self._status_until = 0.0
        self._sync_from_daemon()

    def _sync_from_daemon(self) -> None:
        try:
            resp = self.client.get_phase_order()
            if resp.get("ok"):
                self.pan_order = list(resp["pan"])
                self.tilt_order = list(resp["tilt"])
                self.pan_perm_idx = _perm_index(self.pan_order)
                self.tilt_perm_idx = _perm_index(self.tilt_order)
        except (OSError, ValueError, KeyError):
            pass

    def _apply_order(self, motor: MotorName, order: list[int]) -> None:
        self.client.set_phase_order(motor, order)
        if motor == "pan":
            self.pan_order = order
            self.pan_perm_idx = _perm_index(order)
        else:
            self.tilt_order = order
            self.tilt_perm_idx = _perm_index(order)

    def select_motor(self, motor: MotorName) -> None:
        self.selected = motor

    def next_permutation(self) -> None:
        motor = self.selected
        idx = self.pan_perm_idx if motor == "pan" else self.tilt_perm_idx
        idx = (idx + 1) % len(PERMUTATIONS)
        self._apply_order(motor, list(PERMUTATIONS[idx]))

    def prev_permutation(self) -> None:
        motor = self.selected
        idx = self.pan_perm_idx if motor == "pan" else self.tilt_perm_idx
        idx = (idx - 1) % len(PERMUTATIONS)
        self._apply_order(motor, list(PERMUTATIONS[idx]))

    def preview_selected(self) -> str:
        order = self.pan_order if self.selected == "pan" else self.tilt_order
        return format_phase_order_snippet(self.selected, order)

    def write_selected(self, config_path: Path) -> None:
        order = self.pan_order if self.selected == "pan" else self.tilt_order
        saved = save_phase_order(self.selected, order, config_path)
        self.status_message = (
            f"Saved {self.selected}_phase_order = {order} → {saved}"
        )
        self._status_until = time.monotonic() + STATUS_CLEAR_SEC

    def write_selected_error(self, message: str) -> None:
        self.status_message = f"Save failed: {message}"
        self._status_until = time.monotonic() + STATUS_CLEAR_SEC

    def active_status(self) -> Optional[str]:
        if self.status_message and time.monotonic() < self._status_until:
            return self.status_message
        self.status_message = None
        return None


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
            self.client.disable_motors()
        except (OSError, ValueError):
            pass

    def disable_motors(self) -> None:
        with self._lock:
            self._direction = None
        try:
            self.client.disable_motors()
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


def _draw(
    stdscr: curses.window,
    motor: MotorController,
    client: RustGimbalClient,
    wiring: WiringState,
    drive: DriveModeState,
    config_path: Path,
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    try:
        pan, tilt = client.get_position()
    except (OSError, ValueError, KeyError):
        pan, tilt = 0.0, 0.0

    hz = SPEED_PRESETS[motor.preset]
    ms_per_step = 1000.0 / hz
    lines = [
        "Cookie Finder – Keyboard Gimbal Control",
        "",
        f"Position:  pan {pan:6.1f}°   tilt {tilt:6.1f}°",
        f"Speed:     preset {motor.preset}  ({hz} Hz, {ms_per_step:.0f} ms/step, "
        f"~{_deg_per_sec(hz):.1f}°/s)",
        f"Drive:     {drive.label}   [{drive.mode_index + 1}/{len(DRIVE_MODES)}]",
        f"Moving:    {motor.direction or 'stopped'}",
        "",
        _format_wiring_line(
            "Pan",
            wiring.pan_order,
            wiring.pan_perm_idx,
            wiring.selected == "pan",
        ),
        _format_wiring_line(
            "Tilt",
            wiring.tilt_order,
            wiring.tilt_perm_idx,
            wiring.selected == "tilt",
        ),
        f"Config:    {config_path}",
        "",
        "Controls:",
        "  Arrow keys     Pan / tilt (hold; press Space or s to stop)",
        "  1-9            Set step rate (1=slow ~40ms, 9=fast)",
        "  M / Shift+M    Next / previous drive mode (wave → full → half)",
        "  P / T          Select pan or tilt for wiring permutation",
        "  [ / ]          Previous / next wiring permutation",
        "  Y              Preview TOML snippet for selected motor",
        "  W              Write selected motor mapping to config file",
        "  h              Home (0°, 0°)",
        "  d              Disable motors (de-energize coils)",
        "  s / Space      Stop motion and disable motors",
        "  q              Quit",
        "",
        "Tip: for 24BYJ start-up tests use preset 1 + cycle M through drive modes.",
        "Requires: cookie-finder-ctl daemon (make on-the-pi-rust-daemon)",
    ]

    status = drive.active_status() or wiring.active_status()
    if status:
        lines.extend(["", status])

    if motor.error:
        lines.extend(["", f"Error: {motor.error}"])

    for row, text in enumerate(lines):
        if row >= height:
            break
        stdscr.addnstr(row, 0, text, width - 1)

    stdscr.refresh()


def _run(
    stdscr: curses.window,
    client: RustGimbalClient,
    preset: int,
    config_path: Path,
) -> int:
    curses.curs_set(0)
    stdscr.keypad(True)
    stdscr.timeout(REFRESH_MS)

    motor = MotorController(client, preset)
    wiring = WiringState(client)
    drive = DriveModeState(client)

    try:
        while True:
            _draw(stdscr, motor, client, wiring, drive, config_path)

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

            if key in (ord("d"), ord("D")):
                motor.disable_motors()
                continue

            if ord("1") <= key <= ord("9"):
                motor.set_preset(key - ord("0"))
                continue

            if key == ord("m"):
                motor.stop()
                drive.next_mode()
                continue

            if key == ord("M"):
                motor.stop()
                drive.prev_mode()
                continue

            if key in (ord("p"), ord("P")):
                wiring.select_motor("pan")
                continue

            if key in (ord("t"), ord("T")):
                wiring.select_motor("tilt")
                continue

            if key == ord("]"):
                wiring.next_permutation()
                continue

            if key == ord("["):
                wiring.prev_permutation()
                continue

            if key in (ord("y"), ord("Y")):
                print(wiring.preview_selected(), file=sys.stderr)
                wiring.status_message = f"Preview: {wiring.preview_selected()}"
                wiring._status_until = time.monotonic() + STATUS_CLEAR_SEC
                continue

            if key in (ord("w"), ord("W")):
                try:
                    wiring.write_selected(config_path)
                except (OSError, ValueError) as exc:
                    wiring.write_selected_error(str(exc))
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
        description=(
            "Keyboard control for the Rust gimbal daemon "
            "(arrows, 1-9 speed, M drive mode, wiring permutations)."
        )
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("COOKIE_FINDER_SOCKET"),
        help="Unix socket path (default: /tmp/cookie-finder.sock or COOKIE_FINDER_SOCKET)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Gimbal config path (default: COOKIE_FINDER_CONFIG or config/gimbal.toml)",
    )
    parser.add_argument(
        "--preset",
        type=int,
        choices=sorted(SPEED_PRESETS),
        default=DEFAULT_PRESET,
        help=f"Initial speed preset 1-9 (default: {DEFAULT_PRESET})",
    )
    args = parser.parse_args(argv)
    config_path = resolve_config_path(args.config)

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

    return curses.wrapper(
        lambda stdscr: _run(stdscr, client, args.preset, config_path)
    )


if __name__ == "__main__":
    raise SystemExit(main())
