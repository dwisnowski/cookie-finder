"""
Thermal Camera Viewer - Dual Mode (Standalone GUI + WebServer)

Usage:
    python main.py                    # Standalone mode (OpenCV GUI)
    python main.py --web              # WebServer mode (FastAPI + MJPEG)
    python main.py --web --port 8000  # WebServer on custom port
"""

import os
import time
import argparse
import sys
import threading

# Suppress OpenCV warnings (camera not found messages)
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

# If running headless (no X11 DISPLAY), force Qt to use offscreen rendering.
if "DISPLAY" not in os.environ:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
cv2.setLogLevel(0)

import numpy as np

from thermal_processor import ThermalProcessor, THERMAL_PALETTES
from web_server import run_webserver


def find_available_cameras(max_devices=10):
    """Find all available camera devices by testing which ones actually work."""
    available = []
    for i in range(max_devices):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it's a real camera
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available.append(i)
                    print(f"  ✓ /dev/video{i}: {frame.shape} {frame.dtype}")
                cap.release()
        except:
            pass
    return available


def standalone_mode(camera_id=0):
    """Run standalone OpenCV GUI mode (original behavior)."""
    
    # Initialize processor
    processor = ThermalProcessor()
    
    # Try to find and open camera with reconnection loop
    print("Detecting available cameras...")
    available_cameras = find_available_cameras()
    cap = None
    camera_open = False
    retry_count = 0
    
    if available_cameras:
        print(f"✓ Found working camera(s): {available_cameras}")
        print(f"  Using camera device {available_cameras[0]}")
    else:
        print("⚠ No working cameras detected. Waiting for camera to be connected...")
    
    print("Waiting for camera (press 'r' to retry, 'q' to quit)...")
    
    # Keep trying to open camera
    while not camera_open:
        if available_cameras:
            cap = cv2.VideoCapture(available_cameras[0])
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera_open = True
                print("✓ Camera connected")
                break
            else:
                cap.release()
        
        retry_count += 1
        # Only log every 10 retries to reduce noise
        if retry_count % 10 == 1:
            print(f"⚠ Waiting for camera... (attempt {retry_count})")
        
        # Adaptive backoff
        if retry_count <= 5:
            time.sleep(0.5)
        elif retry_count <= 15:
            time.sleep(1.0)
        else:
            time.sleep(2.0)
        
        # Check for keyboard input while waiting
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested")
            return
        elif key == ord('r'):
            print("Retry requested")
            available_cameras = find_available_cameras()
            retry_count = 0
            if available_cameras:
                print(f"Found cameras: {available_cameras}")
            else:
                print("No cameras found")
    
    prev_frame = None
    buttons = {}
    current_camera_idx = 0
    camera_attempt_count = 0
    
    def draw_buttons(canvas, mode_states):
        """Draw button overlay on canvas."""
        nonlocal buttons
        buttons = {}
        
        button_w = 70
        button_h = 20
        button_color = (100, 100, 100)
        active_color = (0, 255, 0)
        text_color = (255, 255, 255)
        
        button_list = [
            ("H", "heat_seeker_mode"),
            ("C", "heat_cluster_mode"),
            ("M", "motion_mode"),
            ("U", "upscale_mode"),
            ("P", "palette_mode"),
        ]
        
        x = 10
        y = 10
        for label, key in button_list:
            is_active = mode_states.get(key, False)
            color = active_color if is_active else button_color
            cv2.rectangle(canvas, (x, y), (x + button_w, y + button_h), color, -1)
            cv2.putText(canvas, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            buttons[label] = (x, y, x + button_w, y + button_h, key)
            x += button_w + 5
        
        button_list2 = [
            ("D", "denoise_mode"),
            ("O", "normalize_mode"),
            ("E", "enhance_mode"),
            ("S", "stabilize_mode"),
            ("X", "stabilize_super"),
        ]
        
        x = 10
        y = 35
        for label, key in button_list2:
            is_active = mode_states.get(key, False)
            color = active_color if is_active else button_color
            cv2.rectangle(canvas, (x, y), (x + button_w, y + button_h), color, -1)
            cv2.putText(canvas, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            buttons[label] = (x, y, x + button_w, y + button_h, key)
            x += button_w + 5
        
        button_list3 = [
            ("T", "threshold_mode"),
            ("Y", "yolo_mode"),
            ("F", "optical_flow_mode"),
            ("I", "isotherm_mode"),
            ("W", "show_text"),
        ]
        
        x = canvas_width - (button_w + 5) * len(button_list3) - 10
        y = 10
        for label, key in button_list3:
            is_active = mode_states.get(key, False)
            color = active_color if is_active else button_color
            cv2.rectangle(canvas, (x, y), (x + button_w, y + button_h), color, -1)
            cv2.putText(canvas, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)
            buttons[label] = (x, y, x + button_w, y + button_h, key)
            x += button_w + 5
    
    # Get initial frame dimensions
    print("Press 'q' to quit, 'h' for Heat-Seeker, 'c' for Cluster, 'm' for Motion, etc.")
    print("Press 'tab' to cycle cameras, 'r' to reconnect, 'w' to toggle text display")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame from camera")
        return
    
    frame_height, frame_width = first_frame.shape[:2]
    canvas_width = int(frame_width * 1.5)
    canvas_height = int(frame_height * 1.5)
    offset_x = (canvas_width - frame_width) // 2
    offset_y = (canvas_height - frame_height) // 2
    
    # Check if GUI is available
    gui_enabled = True
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or "DISPLAY" not in os.environ:
        gui_enabled = False
        print("Headless mode detected: GUI output disabled (no DISPLAY).")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("⚠ Camera disconnected, attempting to reconnect...")
            cap.release()
            cap = None
            
            # Try to reconnect silently first
            reconnect_attempts = 0
            reconnect_logged = False
            while cap is None and reconnect_attempts < 10:
                try:
                    if available_cameras:
                        cap = cv2.VideoCapture(available_cameras[current_camera_idx])
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            print("✓ Camera reconnected")
                            break
                        else:
                            cap.release()
                            cap = None
                except:
                    pass
                
                reconnect_attempts += 1
                # Only log after 3 silent attempts
                if reconnect_attempts == 3 and not reconnect_logged:
                    print(f"⚠ Reconnecting to camera...")
                    reconnect_logged = True
                
                # Quick retries first, then slower
                wait_time = 0.2 if reconnect_attempts <= 3 else 0.5
                time.sleep(wait_time)
                
                # Check for quit signal while reconnecting
                if gui_enabled:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
            
            if cap is None:
                print("✗ Could not reconnect to camera. Press 'r' to retry or 'q' to quit.")
                while cap is None:
                    time.sleep(0.5)
                    available_cameras = find_available_cameras()
                    if gui_enabled:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            return
                        elif key == ord('r'):
                            print("Retry requested")
                            if available_cameras:
                                try:
                                    cap = cv2.VideoCapture(available_cameras[current_camera_idx])
                                    if cap.isOpened():
                                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                        print("✓ Camera reconnected")
                                    else:
                                        cap.release()
                                        cap = None
                                except:
                                    pass
            continue
        
        # Process frame
        display_frame, mode_text, metadata = processor.process_frame(frame, prev_frame)
        
        # Get text lines
        text_lines = metadata.get('text_lines', [])
        
        # Render text if enabled
        if processor.show_text:
            y_offset = 30
            for line in text_lines:
                cv2.putText(display_frame, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
        
        # Create canvas with buttons
        current_h, current_w = display_frame.shape[:2]
        
        if current_h != frame_height or current_w != frame_width:
            upscale_canvas_width = int(current_w * 1.5)
            upscale_canvas_height = int(current_h * 1.5)
            upscale_offset_x = (upscale_canvas_width - current_w) // 2
            upscale_offset_y = (upscale_canvas_height - current_h) // 2
            canvas = np.zeros((upscale_canvas_height, upscale_canvas_width, 3), dtype=np.uint8)
            canvas[upscale_offset_y:upscale_offset_y+current_h, upscale_offset_x:upscale_offset_x+current_w] = display_frame
        else:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            canvas[offset_y:offset_y+frame_height, offset_x:offset_x+frame_width] = display_frame
        
        mode_states = {
            'heat_seeker_mode': processor.heat_seeker_mode,
            'heat_cluster_mode': processor.heat_cluster_mode,
            'motion_mode': processor.motion_mode,
            'upscale_mode': processor.upscale_mode,
            'palette_mode': processor.palette_mode,
            'denoise_mode': processor.denoise_mode,
            'normalize_mode': processor.normalize_mode,
            'enhance_mode': processor.enhance_mode,
            'stabilize_mode': processor.stabilize_mode,
            'stabilize_super': processor.stabilize_super,
            'threshold_mode': processor.threshold_mode,
            'yolo_mode': processor.yolo_mode,
            'optical_flow_mode': processor.optical_flow_mode,
            'isotherm_mode': processor.isotherm_mode,
            'show_text': processor.show_text,
        }
        draw_buttons(canvas, mode_states)
        
        # Display
        key = 255
        if gui_enabled:
            try:
                cv2.namedWindow("Thermal Camera Feed", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Thermal Camera Feed", canvas.shape[1], canvas.shape[0])
                cv2.imshow("Thermal Camera Feed", canvas)
                key = cv2.waitKey(1) & 0xFF
            except Exception as e:
                print(f"GUI not available: {e}\nSwitching to headless mode.")
                gui_enabled = False
        
        if not gui_enabled:
            time.sleep(0.02)
        
        # Handle keyboard input
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Manual reconnect attempt
            print("Manual reconnect requested")
            available_cameras = find_available_cameras()
            if available_cameras:
                print(f"Found cameras: {available_cameras}")
            else:
                print("No cameras found")
        elif key == 9:  # Tab
            current_camera_idx = (current_camera_idx + 1) % len(available_cameras)
            cap.release()
            cap = cv2.VideoCapture(available_cameras[current_camera_idx])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, temp_frame = cap.read()
            if ret:
                frame_height, frame_width = temp_frame.shape[:2]
                canvas_width = int(frame_width * 1.5)
                canvas_height = int(frame_height * 1.5)
                offset_x = (canvas_width - frame_width) // 2
                offset_y = (canvas_height - frame_height) // 2
            print(f"Switched to camera device: {available_cameras[current_camera_idx]}")
        
        elif key == ord('w'):
            processor.show_text = not processor.show_text
        elif key == ord('h'):
            processor.set_mode('heat_seeker_mode', not processor.heat_seeker_mode)
        elif key == ord('c'):
            processor.set_mode('heat_cluster_mode', not processor.heat_cluster_mode)
        elif key == ord('m'):
            processor.set_mode('motion_mode', not processor.motion_mode)
        elif key == ord('u'):
            processor.upscale_mode = not processor.upscale_mode
        elif key == ord('p'):
            processor.set_mode('palette_mode', not processor.palette_mode)
        elif key == ord('n') and processor.palette_mode:
            processor.set_parameter('palette_idx', processor.palette_idx + 1)
        elif key == ord('d'):
            processor.denoise_mode = not processor.denoise_mode
        elif key == ord('o'):
            processor.normalize_mode = not processor.normalize_mode
        elif key == ord('e'):
            processor.enhance_mode = not processor.enhance_mode
        elif key == ord('t'):
            processor.set_mode('threshold_mode', not processor.threshold_mode)
        elif key == ord('y'):
            processor.init_yolo_model()
            processor.set_mode('yolo_mode', not processor.yolo_mode)
        elif key == ord('f'):
            processor.set_mode('optical_flow_mode', not processor.optical_flow_mode)
        elif key == ord('i'):
            processor.set_mode('isotherm_mode', not processor.isotherm_mode)
        elif key == ord('s'):
            processor.stabilize_mode = not processor.stabilize_mode
        elif key == ord('x'):
            processor.stabilize_super = not processor.stabilize_super
        elif key == ord('=') or key == ord('+'):
            if processor.threshold_mode:
                processor.set_parameter('threshold_value', processor.threshold_value + 5)
        elif key == ord('-'):
            if processor.threshold_mode:
                processor.set_parameter('threshold_value', processor.threshold_value - 5)
        
        prev_frame = frame.copy()
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Thermal Camera Viewer - Dual Mode (Standalone GUI + WebServer)"
    )
    parser.add_argument(
        '--web',
        action='store_true',
        help='Run in WebServer mode (FastAPI + MJPEG) instead of standalone GUI'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='WebServer port (default: 8000)'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='WebServer host (default: 0.0.0.0)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    args = parser.parse_args()
    
    if args.web:
        print("Starting Thermal Camera Viewer in WebServer mode...")
        print(f"Open browser: http://localhost:{args.port}")
        run_webserver(host=args.host, port=args.port, camera_id=args.camera)
    else:
        print("Starting Thermal Camera Viewer in Standalone mode...")
        standalone_mode(camera_id=args.camera)


if __name__ == "__main__":
    main()
