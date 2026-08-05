#!/usr/bin/env python3
"""
Test gimbal control via BlueZ HID gamepad.

Reads gamepad input and moves the gimbal (when GPIO is available).
Pair/connect the pad first via the web UI or bluetoothctl.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_finder.bluetooth.controller import BluetoothController
from cookie_finder.gimbal.pan_tilt import PanTiltGimbal


def main():
    print("[TEST] Gimbal + Gamepad Integration Test")
    print("=" * 60)

    controller = BluetoothController()
    system_devices = controller._get_system_connected_devices()

    if not system_devices:
        print("[TEST] ERROR: No BlueZ-connected gamepad found.")
        print("[TEST] Use the web UI (Pi Bluetooth Gamepad) or bluetoothctl to connect first.")
        return 1

    device_addr = list(system_devices)[0]
    device_name = controller._get_device_name_from_system(device_addr)
    if not device_name:
        device_name = f"Gamepad ({device_addr[-5:]})"

    print(f"[TEST] Found gamepad: {device_name} ({device_addr})")
    print("[TEST] Activating as input device...")
    if not controller.connect_device(device_addr):
        print(f"[TEST] ERROR: {controller.get_last_error() or 'Connection failed'}")
        return 1

    print("[TEST] Initializing gimbal...")
    try:
        gimbal = PanTiltGimbal(max_pan=150.0, max_tilt=60.0)
        gimbal.set_speed(pan_hz=500, tilt_hz=500)
        print("[TEST] Gimbal initialized")
        gimbal_available = True
    except Exception as e:
        print(f"[TEST] Gimbal not available (non-Linux or GPIO error): {e}")
        gimbal = None
        gimbal_available = False

    print("[TEST] " + "=" * 60)
    print("[TEST] Move your gamepad joystick. Press Ctrl+C to exit.")
    print("[TEST]   Left X  → Pan (horizontal)")
    print("[TEST]   Left Y  → Tilt (vertical)")
    print("[TEST] =" * 60 + "\n")

    last_state = None

    try:
        while True:
            input_data = controller.read_controller_input()
            pan_axis = input_data.get("pan_axis", 0.0)
            tilt_axis = input_data.get("tilt_axis", 0.0)
            buttons = input_data.get("buttons", {})

            if input_data != last_state:
                new_pan = (pan_axis + 1.0) / 2.0 * 150.0
                new_tilt = (tilt_axis + 1.0) / 2.0 * 60.0

                button_str = ""
                if any(buttons.values()):
                    pressed = [name for name, on in buttons.items() if on]
                    button_str = f" | buttons: {', '.join(pressed)}"

                print(
                    f"[INPUT] pan={new_pan:6.1f}° (axis={pan_axis:6.2f})  "
                    f"tilt={new_tilt:6.1f}° (axis={tilt_axis:6.2f}){button_str}"
                )

                if gimbal_available and gimbal:
                    try:
                        gimbal.move_to_angles(new_pan, new_tilt)
                        print(
                            f"[GIMBAL] → Moving to pan={new_pan:.1f}°, "
                            f"tilt={new_tilt:.1f}°"
                        )
                    except Exception as e:
                        print(f"[GIMBAL] ERROR: {type(e).__name__}: {e}")

                last_state = input_data

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted")

    finally:
        print("[TEST] Cleaning up...")
        controller.cleanup()

    print("[TEST] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
