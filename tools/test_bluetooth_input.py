#!/usr/bin/env python3
"""
Test gamepad input via BlueZ HID + pygame joystick.

Expects a Classic Bluetooth gamepad already connected in BlueZ, or use the
web UI (Pi Bluetooth Gamepad panel) to pair/connect first.

Usage:
    python tools/test_bluetooth_input.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_finder.bluetooth.controller import BluetoothController


def main():
    print("[TEST] Gamepad Input Logger (BlueZ HID → pygame)")
    print("[TEST] ================================================\n")

    controller = BluetoothController()

    print("[TEST] Checking for BlueZ-connected devices...")
    system_devices = controller._get_system_connected_devices()
    print(f"[TEST] Connected: {system_devices}")

    if not system_devices:
        print("[TEST] ERROR: No BlueZ-connected gamepad found.")
        print("[TEST] Put the pad in pairing mode, then either:")
        print("[TEST]   - Use the web UI: Scan → Pair → Connect")
        print("[TEST]   - Or: bluetoothctl pair <addr> && bluetoothctl connect <addr>")
        return 1

    device_addr = list(system_devices)[0]
    device_name = controller._get_device_name_from_system(device_addr)
    if not device_name:
        device_name = f"Gamepad ({device_addr[-5:]})"

    print(f"[TEST] Using {device_name} ({device_addr})")
    print("[TEST] Marking as active input device...")

    if not controller.connect_device(device_addr):
        err = controller.get_last_error()
        print(f"[TEST] ERROR: Failed to activate device: {err}")
        return 1

    print("[TEST] Reading joystick input for 60 seconds...")
    print("[TEST] Move sticks or press buttons:")
    print("[TEST] ================================================\n")

    start_time = time.time()
    last_state = None
    input_count = 0

    try:
        while time.time() - start_time < 60:
            input_data = controller.read_controller_input()

            if input_data != last_state:
                pan = input_data.get("pan_axis", 0.0)
                tilt = input_data.get("tilt_axis", 0.0)
                buttons = input_data.get("buttons", {})

                button_str = ""
                if any(buttons.values()):
                    pressed = [name for name, on in buttons.items() if on]
                    button_str = f" | buttons: {', '.join(pressed)}"

                print(f"[INPUT] pan={pan:6.2f}  tilt={tilt:6.2f}{button_str}")
                input_count += 1
                last_state = input_data

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")

    finally:
        print(f"\n[TEST] Total input events: {input_count}")
        print("[TEST] Cleaning up (leaving BlueZ connection intact)...")
        controller.cleanup()

    print("[TEST] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
