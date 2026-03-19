#!/usr/bin/env python3
"""Find which video device is the thermal camera."""

import cv2
import sys

print("Testing video devices for working thermal camera...\n")

working_cameras = []
for device_id in range(10):
    try:
        cap = cv2.VideoCapture(device_id)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                working_cameras.append(device_id)
                print(f"✓ /dev/video{device_id}: {w}x{h} {frame.dtype}")
            else:
                print(f"✗ /dev/video{device_id}: Opens but no frames")
            cap.release()
    except Exception as e:
        pass

print()
if working_cameras:
    thermal_cam = working_cameras[0]
    print(f"✓ Thermal camera found: /dev/video{thermal_cam}")
    print(f"\nUse it with:")
    print(f"  python main.py --camera {thermal_cam}")
    print(f"  python main.py --camera {thermal_cam} --web")
    print(f"  make run-web (auto-detects)")
    sys.exit(0)
else:
    print("✗ No working thermal camera detected")
    print("Check:")
    print("  - Camera is connected via USB")
    print("  - User is in 'video' group: groups")
    print("  - Camera permissions: ls -la /dev/video*")
    sys.exit(1)
