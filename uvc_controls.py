#!/usr/bin/env python3
"""
UVC Camera Controls Utility for macOS
Lists and manages UVC camera controls using IOKit-like techniques
"""

import subprocess
import json
import sys

def list_uvc_devices():
    """List all UVC-capable devices using system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        data = json.loads(result.stdout)
        cameras = data.get("SPCameraDataType", [])
        
        print("=" * 60)
        print("UVC-CAPABLE DEVICES")
        print("=" * 60)
        
        if not cameras:
            print("No cameras found.")
            return
        
        for idx, camera in enumerate(cameras):
            print(f"\n[{idx}] {camera.get('_name', 'Unknown Camera')}")
            print(f"    Model: {camera.get('spcamera_model-id', 'N/A')}")
            print(f"    Unique ID: {camera.get('spcamera_unique-id', 'N/A')}")
        
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"Error listing devices: {e}")

def list_uvc_controls():
    """List available UVC controls using v4l2-ctl (if available on macOS via Homebrew)."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", "/dev/video0", "--list-ctrls"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("UVC CONTROLS (v4l2-ctl)")
            print("=" * 60)
            print(result.stdout)
        else:
            print("v4l2-ctl not available. Install via: brew install v4l-utils")
    except FileNotFoundError:
        print("v4l2-ctl not found. Install via: brew install v4l-utils")
    except Exception as e:
        print(f"Error listing controls: {e}")

def list_opencv_camera_properties():
    """List OpenCV camera properties for the thermal camera."""
    import cv2
    
    print("\n" + "=" * 60)
    print("OPENCV CAMERA PROPERTIES")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    properties = {
        "Frame Width": cv2.CAP_PROP_FRAME_WIDTH,
        "Frame Height": cv2.CAP_PROP_FRAME_HEIGHT,
        "FPS": cv2.CAP_PROP_FPS,
        "Brightness": cv2.CAP_PROP_BRIGHTNESS,
        "Contrast": cv2.CAP_PROP_CONTRAST,
        "Saturation": cv2.CAP_PROP_SATURATION,
        "Hue": cv2.CAP_PROP_HUE,
        "Gain": cv2.CAP_PROP_GAIN,
        "Exposure": cv2.CAP_PROP_EXPOSURE,
        "Auto Exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        "White Balance": cv2.CAP_PROP_AUTO_WB,
        "Backlight Compensation": cv2.CAP_PROP_BACKLIGHT,
        "Focus": cv2.CAP_PROP_FOCUS,
        "Auto Focus": cv2.CAP_PROP_AUTOFOCUS,
    }
    
    for name, prop_id in properties.items():
        try:
            value = cap.get(prop_id)
            print(f"  {name}: {value}")
        except Exception as e:
            print(f"  {name}: Error - {e}")
    
    cap.release()

def get_camera_control(control_name):
    """Get a specific camera control value."""
    import cv2
    
    control_map = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "focus": cv2.CAP_PROP_FOCUS,
    }
    
    if control_name.lower() not in control_map:
        print(f"Unknown control: {control_name}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    prop_id = control_map[control_name.lower()]
    value = cap.get(prop_id)
    print(f"{control_name}: {value}")
    cap.release()

def set_camera_control(control_name, value):
    """Set a specific camera control value."""
    import cv2
    
    control_map = {
        "brightness": cv2.CAP_PROP_BRIGHTNESS,
        "contrast": cv2.CAP_PROP_CONTRAST,
        "saturation": cv2.CAP_PROP_SATURATION,
        "hue": cv2.CAP_PROP_HUE,
        "gain": cv2.CAP_PROP_GAIN,
        "exposure": cv2.CAP_PROP_EXPOSURE,
        "focus": cv2.CAP_PROP_FOCUS,
    }
    
    if control_name.lower() not in control_map:
        print(f"Unknown control: {control_name}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    prop_id = control_map[control_name.lower()]
    try:
        success = cap.set(prop_id, float(value))
        if success:
            print(f"Set {control_name} to {value}")
        else:
            print(f"Failed to set {control_name}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()

def print_help():
    """Print help information."""
    help_text = """
UVC Camera Controls Utility

Usage:
  python uvc_controls.py [command] [options]

Commands:
  list-devices          List all UVC-capable devices
  list-controls         List available UVC controls
  get <control-name>    Get the value of a control
  set <control-name> <value>  Set the value of a control
  help                  Show this help message

Examples:
  python uvc_controls.py list-devices
  python uvc_controls.py list-controls
  python uvc_controls.py get brightness
  python uvc_controls.py set brightness 0.5

Available Controls:
  - brightness
  - contrast
  - saturation
  - hue
  - gain
  - exposure
  - focus
"""
    print(help_text)

def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "list-devices":
        list_uvc_devices()
    elif command == "list-controls":
        list_uvc_controls()
        list_opencv_camera_properties()
    elif command == "get" and len(sys.argv) > 2:
        get_camera_control(sys.argv[2])
    elif command == "set" and len(sys.argv) > 3:
        set_camera_control(sys.argv[2], sys.argv[3])
    elif command == "help":
        print_help()
    else:
        print(f"Unknown command: {command}")
        print_help()

if __name__ == "__main__":
    main()
