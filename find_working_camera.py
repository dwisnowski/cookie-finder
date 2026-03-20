#!/usr/bin/env python3
"""Find which video device is the thermal camera."""

import cv2
import sys
import os

print("Testing video devices for working thermal camera...\n")

working_cameras = []
for device_id in range(10):
    try:
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            continue
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        
        if ret and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            working_cameras.append(device_id)
            print(f"✓ /dev/video{device_id}: Readable - {w}x{h} {frame.dtype}")
        else:
            print(f"✗ /dev/video{device_id}: Opens but no frames")
        
        cap.release()
    except Exception as e:
        pass

print()
if working_cameras:
    thermal_cam = working_cameras[0]
    print(f"✓ Thermal camera found: /dev/video{thermal_cam}")
    print(f"\nUse the app with:")
    print(f"  make run-web          # Automatically detects /dev/video{thermal_cam}")
    print(f"  make run-standalone")
    print(f"\nOr manually specify:")
    print(f"  python main.py --camera {thermal_cam}")
    print(f"  python main.py --camera {thermal_cam} --web")
    sys.exit(0)
else:
    print("✗ No working thermal camera detected")
    print("\nTroubleshooting:")
    print("  1. Is camera connected? Run: lsusb | grep -i thermal")
    print("  2. Check permissions: groups")
    print("       Should include: video")
    print("       If not, run: sudo usermod -a -G video $USER")
    print("  3. Check device ownership: ls -la /dev/video*")
    print("  4. Try unplugging/replugging the camera")
    sys.exit(1)

