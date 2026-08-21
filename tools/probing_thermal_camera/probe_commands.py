#!/usr/bin/env python3
"""
Probe Pixfra/Mileseey CDC serial for text-style control commands.

Uses a short non-blocking pyserial timeout so the terminal is not locked
(unlike attaching with `screen`). Run on Mac or Pi:

    uv run tools/probing_thermal_camera/probe_commands.py
    make probe-commands

Interpretation:
  - Readable text (e.g. "unknown command") → likely a text CLI protocol
  - Only binary / no response → likely a framed binary packet protocol
"""

from __future__ import annotations

import argparse
import glob
import sys
import time

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: uv sync")
    sys.exit(1)

# Common manufacturing / app-style text probes. Binary framed probes stay in
# probe_serial.py; this script only checks whether a text CLI is present.
PROBES: list[tuple[str, bytes]] = [
    ("blank CRLF", b"\r\n"),
    ("help", b"help\r\n"),
    ("?", b"?\r\n"),
    ("get_palette", b"get_palette\r\n"),
    ("NUC", b"NUC\r\n"),
    ("nuc", b"nuc\r\n"),
    ("shutter", b"shutter\r\n"),
    ("ffc", b"ffc\r\n"),
    ("palette", b"palette\r\n"),
    ("version", b"version\r\n"),
    ("AT", b"AT\r\n"),
]


def find_camera_ports() -> list[str]:
    """Find CDC ACM ports (Linux ttyACM*, macOS tty.usbmodem*)."""
    patterns = (
        "/dev/ttyACM*",
        "/dev/tty.usbmodem*",
    )
    ports: list[str] = []
    for pattern in patterns:
        ports.extend(sorted(glob.glob(pattern)))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for port in ports:
        if port not in seen:
            seen.add(port)
            unique.append(port)
    return unique


def choose_port(preferred: str | None) -> str | None:
    if preferred:
        return preferred

    ports = find_camera_ports()
    if not ports:
        print("✗ No camera serial ports found")
        print("  Looked for: /dev/ttyACM* and /dev/tty.usbmodem*")
        return None

    if len(ports) == 1:
        return ports[0]

    print(f"Found {len(ports)} ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port}")
    choice = input("Select port (0): ").strip() or "0"
    try:
        return ports[int(choice)]
    except (ValueError, IndexError):
        return ports[0]


def read_response(ser: serial.Serial, settle_s: float = 0.3) -> bytes:
    time.sleep(settle_s)
    chunks: list[bytes] = []
    # Drain anything that arrives within the port timeout window
    deadline = time.time() + settle_s
    while time.time() < deadline or ser.in_waiting > 0:
        waiting = ser.in_waiting
        if waiting > 0:
            chunks.append(ser.read(waiting))
            deadline = time.time() + 0.1
        else:
            time.sleep(0.05)
    return b"".join(chunks)


def format_response(data: bytes) -> None:
    print(f"  Raw (hex): {data.hex() if data else '(empty)'}")
    if not data:
        return
    text = data.decode("utf-8", errors="replace").strip()
    printable = sum(1 for c in text if c.isprintable() or c.isspace())
    ratio = printable / max(len(text), 1)
    if ratio >= 0.7 and text:
        print(f"  Decoded text: {text!r}")
        print("  Hint: looks like a text/CLI response")
    else:
        print(f"  Mostly binary ({len(data)} bytes); likely a framed packet protocol")


def probe_commands(port: str, baud: int, settle_s: float) -> int:
    print("=" * 60)
    print(f"Command probe: {port} @ {baud} baud")
    print("=" * 60)
    print("Non-blocking read timeout — safe vs. locking with `screen`.\n")

    try:
        ser = serial.Serial(port, baudrate=baud, timeout=0.5)
    except serial.SerialException as e:
        print(f"✗ Error opening port: {e}")
        return 1

    got_any = False
    got_text = False

    try:
        # Clear stale input before probing
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        for name, payload in PROBES:
            print(f"--- {name}: {payload!r} ---")
            ser.write(payload)
            ser.flush()
            data = read_response(ser, settle_s=settle_s)
            if data:
                got_any = True
                text = data.decode("utf-8", errors="replace")
                printable = sum(1 for c in text if c.isprintable() or c.isspace())
                if printable / max(len(text), 1) >= 0.7:
                    got_text = True
                format_response(data)
            else:
                print("  No response")
            print()
    finally:
        ser.close()
        print("Port closed.")

    print("=" * 60)
    if got_text:
        print("Result: at least one readable text reply — try more CLI guesses.")
    elif got_any:
        print("Result: binary replies only — prefer framed packets (see probe_serial.py).")
    else:
        print("Result: no replies. Port may be wrong, busy, or need binary init first.")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe thermal camera CDC serial for text-style commands"
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="Serial device (default: auto-detect ttyACM* / tty.usbmodem*)",
    )
    parser.add_argument("-b", "--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument(
        "--settle",
        type=float,
        default=0.3,
        help="Seconds to wait after each write before reading",
    )
    args = parser.parse_args()

    port = choose_port(args.port)
    if not port:
        return 1

    print(f"✓ Using port: {port}")
    return probe_commands(port, args.baud, args.settle)


if __name__ == "__main__":
    sys.exit(main())
