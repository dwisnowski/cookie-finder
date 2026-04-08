#!/usr/bin/env python3
"""
Simple script to test gamepad input reading via Linux joystick interface.
Logs all received input (joystick axes and buttons) to console.

Usage:
    python tools/test_bluetooth_input.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cookie_finder.bluetooth.controller import BluetoothController


def main():
    """Main test loop."""
    print("[TEST] Gamepad Input Logger (via /dev/input/jsX)")
    print("[TEST] ================================================\n")
    
    # Initialize controller
    controller = BluetoothController()
    
    # Check for already-connected devices
    print("[TEST] Checking for system-connected devices...")
    system_devices = controller._get_system_connected_devices()
    print(f"[TEST] System devices: {system_devices}")
    
    if not system_devices:
        print("[TEST] ERROR: No system-connected gamepad found!")
        print("[TEST] Please pair your gamepad first using: bluetoothctl pair <address>")
        return 1
    
    # Use the first (and likely only) system-connected device
    device_addr = list(system_devices)[0]
    print(f"[TEST] Using system-connected device: {device_addr}")
    
    # Get device name
    device_name = controller._get_device_name_from_system(device_addr)
    if not device_name:
        device_name = f"Gamepad ({device_addr[-5:]})"
    
    print(f"[TEST] Device: {device_name}")
    
    # Add device to controller's list (it may not be in BLE scan if already connected)
    from cookie_finder.bluetooth.controller import BluetoothDevice
    device = BluetoothDevice(device_addr, device_name, -100)
    device.paired = True
    device.connected = True
    controller.devices[device_addr] = device
    print(f"[TEST] Registered device in controller")
    
    print(f"[TEST] Connecting to establish Bleak session...")
    
    # Connect to device (ensures Bleak can access characteristics, but fastest path for system-connected)
    if not controller.connect_device(device_addr):
        print("[TEST] ERROR: Failed to connect to device")
        return 1
    
    print(f"[TEST] Connected! Reading joystick input for 60 seconds...")
    print("[TEST] Move joystick or press buttons to see data:")
    print("[TEST] ================================================\n")
    
    # Read input for 60 seconds or until interrupted
    start_time = time.time()
    last_state = None
    input_count = 0
    
    try:
        while time.time() - start_time < 60:
            input_data = controller.read_controller_input()
            
            # Only log if input changed
            if input_data != last_state:
                pan = input_data.get("pan_axis", 0.0)
                tilt = input_data.get("tilt_axis", 0.0)
                buttons = input_data.get("buttons", {})
                
                # Format output
                pan_str = f"pan={pan:6.2f}"
                tilt_str = f"tilt={tilt:6.2f}"
                
                button_str = ""
                if any(buttons.values()):
                    pressed = [name for name, pressed in buttons.items() if pressed]
                    button_str = f" | buttons: {', '.join(pressed)}"
                
                print(f"[INPUT] {pan_str}  {tilt_str}{button_str}")
                input_count += 1
                last_state = input_data
            
            time.sleep(0.05)  # Poll at 20Hz
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    
    finally:
        print(f"\n[TEST] Total input events: {input_count}")
        print("[TEST] Cleaning up...")
        try:
            controller.disconnect_device(device_addr)
        except Exception as e:
            print(f"[TEST] Disconnect warning: {type(e).__name__} (non-critical)")
        
        try:
            controller.cleanup()
        except Exception as e:
            print(f"[TEST] Cleanup warning: {type(e).__name__} (non-critical)")
    
    print("[TEST] Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
