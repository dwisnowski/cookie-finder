#!/usr/bin/env python3
"""
Stepper motor smoke test via the Rust cookie-finder-ctl daemon.

Requires the daemon to be running (`make on-the-pi-rust-daemon`).

Usage:
  test_motors.py [command] [steps]

Commands:
  pan-cw [steps]     Pan motor clockwise (default: 50)
  pan-ccw [steps]    Pan motor counter-clockwise (default: 50)
  tilt-cw [steps]    Tilt motor clockwise (default: 50)
  tilt-ccw [steps]   Tilt motor counter-clockwise (default: 50)
  auto               Automated test sequence
  home               Home both motors to limit switches
"""

from __future__ import annotations

import sys
import time

from cookie_finder.gimbal.rust_client import RustGimbalClient

# Match former Python MotorDirection: CW = +1, CCW = -1
CW = 1
CCW = -1


def _connect() -> RustGimbalClient:
    client = RustGimbalClient.connect()
    if client is None:
        raise SystemExit(
            "Rust gimbal daemon not available. Start it with: "
            "make on-the-pi-rust-daemon"
        )
    client.set_speed(pan_hz=500, tilt_hz=500)
    return client


def _wait_idle(gimbal: RustGimbalClient, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not gimbal.is_moving():
            return
        time.sleep(0.05)


def run_command(command: str, steps: int = 50) -> int:
    gimbal = _connect()
    try:
        if command == "pan-cw":
            print(f"Pan CW {steps} steps...")
            gimbal.pan_step(CW, steps=steps)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"Pan: {pan:.1f}°")

        elif command == "pan-ccw":
            print(f"Pan CCW {steps} steps...")
            gimbal.pan_step(CCW, steps=steps)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"Pan: {pan:.1f}°")

        elif command == "tilt-cw":
            print(f"Tilt CW {steps} steps...")
            gimbal.tilt_step(CW, steps=steps)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"Tilt: {tilt:.1f}°")

        elif command == "tilt-ccw":
            print(f"Tilt CCW {steps} steps...")
            gimbal.tilt_step(CCW, steps=steps)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"Tilt: {tilt:.1f}°")

        elif command == "home":
            print("Homing both motors...")
            gimbal.home()
            _wait_idle(gimbal, timeout=120.0)
            pan, tilt = gimbal.get_position()
            print(f"Home complete: Pan {pan:.1f}°, Tilt {tilt:.1f}°")

        elif command == "auto":
            print("=== Automated Motor Test (via Rust daemon) ===\n")

            print("Testing PAN motor...")
            print("  CW 100 steps...")
            gimbal.pan_step(CW, steps=100)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"  Pan angle: {pan:.1f}°")
            time.sleep(0.5)

            print("  CCW 50 steps...")
            gimbal.pan_step(CCW, steps=50)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"  Pan angle: {pan:.1f}°")
            time.sleep(0.5)

            print("\nTesting TILT motor...")
            print("  CW 100 steps...")
            gimbal.tilt_step(CW, steps=100)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"  Tilt angle: {tilt:.1f}°")
            time.sleep(0.5)

            print("  CCW 50 steps...")
            gimbal.tilt_step(CCW, steps=50)
            _wait_idle(gimbal)
            pan, tilt = gimbal.get_position()
            print(f"  Tilt angle: {tilt:.1f}°")

            print("\n✅ Test complete!")
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        gimbal.cleanup()

    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    command = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    return run_command(command, steps)


if __name__ == "__main__":
    sys.exit(main())
