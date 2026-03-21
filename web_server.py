"""
FastAPI web server for thermal camera with MJPEG streaming and WebSocket control.
Optimized for 50 Hz video streaming over WiFi on embedded systems (Orange Pi).
"""

import os
import cv2
import json
import threading
import time
import numpy as np
from io import BytesIO
from queue import Queue
from contextlib import asynccontextmanager

# Suppress OpenCV warnings (camera not found messages)
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
cv2.setLogLevel(0)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from thermal_processor import ThermalProcessor


# Global state
camera_thread = None
frame_queue = Queue(maxsize=2)
processor = None
active_clients = set()
camera_connected = False
camera_id_current = 0
reconnect_lock = threading.Lock()
available_cameras = []  # List of working camera devices
camera_switch_event = threading.Event()  # Signal to switch cameras
camera_switch_id = 0  # Target camera ID to switch to


def try_open_camera(camera_id=0):
    """Try to open camera and verify it actually works."""
    try:
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Verify by reading a frame
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                return cap
            else:
                # Device opens but doesn't produce frames
                cap.release()
                return None
        cap.release()
    except:
        pass
    return None


def capture_frames(camera_id=0):
    """Capture frames from thermal camera with reconnection logic."""
    global camera_connected, camera_id_current, available_cameras, camera_switch_event, camera_switch_id
    
    cap = None
    prev_frame = None
    retry_count = 0
    last_log_retry = 0
    
    print(f"Camera thread: Detecting working cameras...")
    # Quick scan to find working cameras
    working_cameras = []
    for test_id in range(5):
        test_cap = try_open_camera(test_id)
        if test_cap is not None:
            working_cameras.append(test_id)
            print(f"  ✓ /dev/video{test_id} is working")
            test_cap.release()
    
    available_cameras = working_cameras
    
    if not working_cameras:
        print(f"  ✗ No working cameras detected")
        print(f"Camera thread started (waiting for device {camera_id})")
    else:
        print(f"  ✓ Found working cameras: {working_cameras}")
        # If the requested camera isn't in the working list, use the first working one
        if camera_id not in working_cameras:
            print(f"Requested /dev/video{camera_id} not in working list, using /dev/video{working_cameras[0]}")
            camera_id = working_cameras[0]
        print(f"Camera thread started (attempting device {camera_id})")
    
    while True:
        # Check if user requested camera switch
        if camera_switch_event.is_set():
            print(f"Switching cameras: /dev/video{camera_id} → /dev/video{camera_switch_id}")
            camera_id = camera_switch_id
            if cap is not None:
                cap.release()
            cap = None
            prev_frame = None
            camera_switch_event.clear()
        
        # Try to open camera if not connected
        if cap is None:
            with reconnect_lock:
                cap = try_open_camera(camera_id)
                if cap is not None:
                    camera_connected = True
                    camera_id_current = camera_id
                    print(f"✓ Camera connected (device {camera_id})")
                    retry_count = 0
                    last_log_retry = 0
                else:
                    camera_connected = False
                    camera_id_current = camera_id
                    retry_count += 1
                    # Only log every 5 retries to reduce noise
                    if retry_count == 1 or retry_count % 5 == 0:
                        print(f"⚠ Waiting for camera /dev/video{camera_id}... (attempt {retry_count})")
        
        if cap is None:
            # Adaptive backoff: 0.5s for first 5, then 1s, then 2s max
            if retry_count <= 5:
                wait_time = 0.5
            elif retry_count <= 15:
                wait_time = 1.0
            else:
                wait_time = 2.0
            time.sleep(wait_time)
            continue
        
        try:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame read failed, attempting to reconnect...")
                cap.release()
                cap = None
                prev_frame = None
                time.sleep(0.5)
                continue
            
            processed_frame, _, _ = processor.process_frame(frame, prev_frame)
            prev_frame = frame.copy()
            
            try:
                frame_queue.put_nowait(processed_frame)
            except:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(processed_frame)
                except:
                    pass
            
            time.sleep(0.02)
        
        except Exception as e:
            print(f"⚠ Error reading frame: {e}")
            if cap is not None:
                cap.release()
            cap = None
            prev_frame = None
            time.sleep(0.5)
    
    if cap is not None:
        cap.release()


def create_no_camera_image():
    """Create a placeholder image when camera is not connected."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, "Camera Disconnected", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(img, "Press Reconnect Button", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 1)
    return img


def mjpeg_generator(jpeg_quality=65):
    """Generate MJPEG stream frames or placeholder if camera disconnected."""
    no_camera_img = create_no_camera_image()
    frame_count = 0
    
    while True:
        try:
            if camera_connected:
                frame = frame_queue.get(timeout=0.5)
            else:
                frame = no_camera_img
        except:
            frame = no_camera_img
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-length: ' + str(len(buffer)).encode() + b'\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
        
        frame_count += 1
        if frame_count % 50 == 0 and not camera_connected:
            print(f"  (⏳ waiting for camera reconnection...)")
        
        time.sleep(0.02)


def create_app(camera_id=0):
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global camera_thread, processor
        
        # Startup
        print(f"Initializing processor...")
        processor = ThermalProcessor()
        print(f"Starting camera thread (device /dev/video{camera_id})...")
        camera_thread = threading.Thread(target=capture_frames, args=(camera_id,), daemon=True)
        camera_thread.start()
        print("✓ Web server started")
        
        yield
        
        # Shutdown
        print("Web server shutting down")
    
    app = FastAPI(title="Thermal Camera Viewer", lifespan=lifespan)
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add all the routes
    @app.get("/camera-status")
    async def camera_status():
        return {
            "connected": camera_connected,
            "camera_id": camera_id_current,
            "message": "Camera connected" if camera_connected else "Camera disconnected"
        }
    
    @app.post("/reconnect")
    async def reconnect():
        print("Manual reconnect requested...")
        return {"status": "reconnect_triggered", "message": "Attempting to reconnect..."}
    
    @app.post("/switch-camera/{new_camera_id}")
    async def switch_camera(new_camera_id: int):
        global camera_switch_id, camera_switch_event
        if new_camera_id not in available_cameras:
            return {"status": "error", "message": f"Camera /dev/video{new_camera_id} not available"}
        camera_switch_id = new_camera_id
        camera_switch_event.set()
        print(f"Camera switch requested: /dev/video{new_camera_id}")
        return {"status": "switching", "message": f"Switching to /dev/video{new_camera_id}..."}
    
    @app.get("/available-cameras")
    async def get_available_cameras():
        return {
            "available": available_cameras,
            "current": camera_id_current
        }
    
    @app.get("/video")
    async def video_feed():
        return StreamingResponse(
            mjpeg_generator(jpeg_quality=65),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    
    @app.get("/state")
    async def get_state():
        if processor is None:
            return {"error": "Processor not initialized"}
        return processor.get_state()
    
    @app.websocket("/control")
    async def websocket_control(websocket: WebSocket):
        await websocket.accept()
        active_clients.add(websocket)
        
        try:
            await websocket.send_json({"type": "state", "data": processor.get_state()})
            
            while True:
                data = await websocket.receive_text()
                command = json.loads(data)
                action = command.get("action")
                
                if action == "toggle_mode":
                    mode = command.get("mode")
                    current = getattr(processor, mode, False)
                    processor.set_mode(mode, not current)
                    state = processor.get_state()
                    for client in active_clients:
                        try:
                            await client.send_json({"type": "state", "data": state})
                        except:
                            pass
                
                elif action == "set_param":
                    param = command.get("param")
                    value = command.get("value")
                    processor.set_parameter(param, value)
                    state = processor.get_state()
                    for client in active_clients:
                        try:
                            await client.send_json({"type": "state", "data": state})
                        except:
                            pass
                
                elif action == "motor_command":
                    motor_cmd = command.get("command")
                    motor_state = command.get("state")
                    
                    if motor_cmd == "gamepad_input":
                        # Gamepad analog input: continuous pan/tilt angles
                        pan = command.get("pan", 0)
                        tilt = command.get("tilt", 0)
                        print(f"🎮 Gamepad: Pan={pan:.1f}°, Tilt={tilt:.1f}°")
                        # Here you can add actual GPIO/PWM gimbal control with angles
                        # Example: control_gimbal_angles(pan, tilt)
                    else:
                        # Button-based motor commands (discrete start/stop)
                        if motor_state == "start":
                            print(f"🎮 Motor: {motor_cmd} START")
                        else:
                            print(f"🎮 Motor: {motor_cmd} STOP")
                        # Here you can add actual GPIO/PWM control for pan/tilt gimbal
                        # Example: control_gimbal(motor_cmd, motor_state)
                
                elif action == "get_state":
                    await websocket.send_json({"type": "state", "data": processor.get_state()})
        
        except WebSocketDisconnect:
            active_clients.discard(websocket)
    
    @app.get("/")
    async def root():
        """Serve HTML UI."""
        html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thermal Camera Viewer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 10px;
                background-color: #1a1a1a;
                color: #fff;
            }
            
            .container {
                display: grid;
                grid-template-columns: 1fr 300px;
                gap: 20px;
                margin-top: 20px;
            }
            
            .video-section {
                text-align: center;
            }
            
            .video-stream {
                max-width: 100%;
                height: auto;
                border: 2px solid #444;
                background: #000;
            }
            
            .status {
                margin-top: 10px;
                padding: 10px;
                background: #2a2a2a;
                border-radius: 4px;
                font-size: 12px;
                text-align: left;
            }
            
            .controls {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 4px;
                max-height: 90vh;
                overflow-y: auto;
            }
            
            .section {
                margin-bottom: 20px;
            }
            
            .section-title {
                font-weight: bold;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #444;
                font-size: 14px;
            }
            
            .button-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 5px;
                margin-bottom: 10px;
            }
            
            .btn {
                padding: 8px;
                font-size: 12px;
                border: 1px solid #444;
                background: #333;
                color: #fff;
                cursor: pointer;
                border-radius: 3px;
                transition: all 0.2s;
            }
            
            .btn:hover {
                background: #444;
            }
            
            .btn.active {
                background: #00aa00;
                border-color: #00cc00;
            }
            
            .btn-motor {
                padding: 20px;
                font-size: 16px;
                border: 1px solid #1e5a3a;
                background: #1e5a3a;
                color: #fff;
                cursor: pointer;
                border-radius: 4px;
                transition: all 0.1s;
                font-weight: bold;
                touch-action: manipulation;
                user-select: none;
            }
            
            .btn-motor:hover {
                background: #2a7a52;
                border-color: #2a7a52;
            }
            
            .btn-motor:active {
                background: #0d3620;
                transform: scale(0.95);
            }
            
            .slider-group {
                margin-bottom: 10px;
            }
            
            .slider-label {
                font-size: 12px;
                margin-bottom: 3px;
                display: flex;
                justify-content: space-between;
            }
            
            input[type="range"] {
                width: 100%;
                height: 20px;
                cursor: pointer;
            }
            
            .info {
                font-size: 11px;
                color: #aaa;
                margin-top: 5px;
            }
            
            .modal-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }
            
            .modal-overlay.active {
                display: flex;
            }
            
            .modal-content {
                background: #2a2a2a;
                border: 2px solid #444;
                border-radius: 8px;
                padding: 30px;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                color: #fff;
                position: relative;
            }
            
            .modal-header {
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 20px;
                text-align: center;
                border-bottom: 2px solid #444;
                padding-bottom: 15px;
            }
            
            .modal-section {
                margin-bottom: 20px;
            }
            
            .modal-section-title {
                font-weight: bold;
                color: #0f0;
                margin-bottom: 10px;
                padding-bottom: 5px;
                border-bottom: 1px solid #333;
            }
            
            .shortcut-item {
                display: grid;
                grid-template-columns: 120px 1fr;
                gap: 15px;
                margin-bottom: 8px;
                padding: 8px;
                background: #1a1a1a;
                border-radius: 3px;
                font-size: 13px;
            }
            
            .shortcut-key {
                font-family: monospace;
                font-weight: bold;
                color: #0f0;
                text-align: right;
            }
            
            .shortcut-desc {
                color: #aaa;
            }
            
            .modal-close {
                position: absolute;
                top: 15px;
                right: 20px;
                font-size: 32px;
                font-weight: bold;
                color: #aaa;
                cursor: pointer;
                background: none;
                border: none;
                padding: 0;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .modal-close:hover {
                color: #fff;
            }
            
            .gamepad-status {
                background: #1a1a1a;
                padding: 10px;
                border-radius: 3px;
                margin-bottom: 10px;
                font-size: 12px;
            }
            
            .gamepad-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #ff4444;
                margin-right: 8px;
                vertical-align: middle;
            }
            
            .gamepad-indicator.connected {
                background: #0f0;
                box-shadow: 0 0 8px #0f0;
            }
            
            .gamepad-name {
                color: #0f0;
                font-weight: bold;
                margin: 5px 0;
                font-size: 11px;
            }
            
            .gamepad-device-count {
                color: #aaa;
                font-size: 10px;
                margin-top: 5px;
            }
            
            .btn-axis {
                background: #333;
                color: #ccc;
                border: 2px solid #444;
                padding: 8px 12px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .btn-axis:hover {
                background: #444;
                border-color: #666;
            }
            
            .btn-axis.active {
                background: #0f0;
                color: #000;
                border-color: #0f0;
                font-weight: bold;
            }
            
            .btn-toggle {
                background: #1e3a1e;
                color: #ccc;
                border: 2px solid #2a4d2a;
                padding: 6px 12px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .btn-toggle:hover {
                background: #2a4d2a;
                border-color: #3a7d3a;
            }
            
            .btn-toggle.active {
                background: #0f0;
                color: #000;
                border-color: #0f0;
                font-weight: bold;
            }
            
            .gamepad-button {
                transition: all 0.1s;
                color: #aaa;
            }
            
            .gamepad-button.pressed {
                background: #0f0 !important;
                color: #000 !important;
                box-shadow: 0 0 12px #0f0;
                transform: scale(1.1);
            }
            
            .btn-preset {
                background: #2a3a2a;
                color: #ccc;
                border: 2px solid #3a5a3a;
                padding: 8px 12px;
                border-radius: 3px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s;
            }
            
            .btn-preset:hover {
                background: #3a4a3a;
                border-color: #5a7a5a;
            }
            
            .btn-preset.active {
                background: #0d8;
                color: #000;
                border-color: #0d8;
                font-weight: bold;
                box-shadow: 0 0 12px #0d8;
            }
        </style>
    </head>
    <body>
        <h1>Thermal Camera Viewer</h1>
        
        <div class="container">
            <div class="video-section">
                <img id="videoStream" src="/video" alt="Thermal camera feed" class="video-stream">
                <div class="status">
                    <div id="statusText">Connecting...</div>
                </div>
            </div>
            
            <div class="controls">
                <!-- Detection Modes -->
                <div class="section">
                    <div class="section-title">Detection Modes</div>
                    <div class="button-grid">
                        <button class="btn" id="btn_h" title="Heat Seeker Mode">H: Heat Seeker</button>
                        <button class="btn" id="btn_c" title="Heat Cluster Mode">C: Cluster</button>
                        <button class="btn" id="btn_m" title="Motion Detection">M: Motion</button>
                        <button class="btn" id="btn_p" title="Palette/Thermal Colors">P: Palette</button>
                        <button class="btn" id="btn_t" title="Threshold">T: Threshold</button>
                        <button class="btn" id="btn_y" title="YOLO AI">Y: YOLO</button>
                        <button class="btn" id="btn_f" title="Optical Flow">F: Flow</button>
                        <button class="btn" id="btn_i" title="Isotherm">I: Isotherm</button>
                    </div>
                </div>
                
                <!-- Enhancements -->
                <div class="section">
                    <div class="section-title">Enhancements</div>
                    <div class="button-grid">
                        <button class="btn" id="btn_d" title="Denoise">D: Denoise</button>
                        <button class="btn" id="btn_o" title="Normalize">O: Normalize</button>
                        <button class="btn" id="btn_e" title="Enhance">E: Enhance</button>
                        <button class="btn" id="btn_u" title="Upscale">U: Upscale</button>
                    </div>
                </div>
                
                <!-- Stabilization -->
                <div class="section">
                    <div class="section-title">Stabilization</div>
                    <div class="button-grid">
                        <button class="btn" id="btn_s" title="Stabilize">S: Stabilize</button>
                        <button class="btn" id="btn_x" title="Super Stab">X: Super Stab</button>
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label">Strength: <span id="stab_strength">1.0</span></div>
                        <input type="range" min="0" max="100" value="100" id="slider_stabilize_strength">
                    </div>
                </div>
                
                <!-- Threshold / Parameters -->
                <div class="section">
                    <div class="section-title">Parameters</div>
                    
                    <div class="slider-group">
                        <div class="slider-label">Threshold: <span id="val_threshold">127</span></div>
                        <input type="range" min="0" max="255" value="127" id="slider_threshold_value">
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label">Isotherm Min: <span id="val_isotherm_min">100</span></div>
                        <input type="range" min="0" max="255" value="100" id="slider_isotherm_min">
                    </div>
                    
                    <div class="slider-group">
                        <div class="slider-label">Isotherm Max: <span id="val_isotherm_max">200</span></div>
                        <input type="range" min="0" max="255" value="200" id="slider_isotherm_max">
                    </div>
                </div>
                
                <!-- Color Palette -->
                <div class="section">
                    <div class="section-title">Color Palette</div>
                    <div style="display: flex; gap: 5px; margin-bottom: 10px;">
                        <button class="btn" id="btn_palette_prev" style="flex: 1; font-size: 11px;">← Prev</button>
                        <button class="btn" id="btn_palette_next" style="flex: 1; font-size: 11px;">Next →</button>
                    </div>
                    <div id="currentPaletteInfo" style="font-size: 11px; color: #aaa; text-align: center; padding: 8px; background: #1a1a1a; border-radius: 3px;">
                        <span id="currentPaletteName">Ironbow</span>
                    </div>
                </div>
                
                <!-- Motor Control -->
                <div class="section">
                    <div class="section-title">Motor Control</div>
                    
                    <!-- Gamepad/Bluetooth Control Status -->
                    <div class="gamepad-status">
                        <div>
                            <span class="gamepad-indicator" id="gamepadIndicator"></span>
                            <span id="gamepadStatusText">No Controller</span>
                        </div>
                        <div class="gamepad-name" id="gamepadNameDisplay">—</div>
                        <div class="gamepad-device-count" id="gamepadDeviceCount">0 devices</div>
                        <button class="btn" id="btn_cycle_gamepad" style="width: 100%; margin-top: 8px; font-size: 11px;">🎮 Cycle Device</button>
                    </div>
                    
                    <!-- Gamepad Presets -->
                    <div style="background: #1a1a1a; padding: 10px; border-radius: 3px; margin-bottom: 10px; text-align: center;">
                        <div style="color: #aaa; margin-bottom: 6px; font-size: 11px; font-weight: bold;">Joystick Preset</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                            <button class="btn-preset" id="btn_preset_normal" style="font-size: 10px; padding: 8px;">Normal</button>
                            <button class="btn-preset" id="btn_preset_vertical" style="font-size: 10px; padding: 8px;">Vertical</button>
                        </div>
                        <div id="currentPresetDisplay" style="margin-top: 6px; color: #0f0; font-weight: bold; font-size: 11px;">Current: Normal</div>
                    </div>
                    
                    <!-- Gamepad Axis Configuration -->
                    <div style="background: #1a1a1a; padding: 10px; border-radius: 3px; margin-bottom: 10px; font-size: 11px;">
                        <div style="color: #aaa; margin-bottom: 8px; font-weight: bold;">Joystick Axis Mapping</div>
                        <div style="margin-bottom: 8px;">
                            <label style="display: block; color: #ccc; margin-bottom: 3px;">Pan (Horizontal)</label>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                <button class="btn-axis" id="btn_pan_axis_0" style="font-size: 10px; padding: 6px;">Left X</button>
                                <button class="btn-axis" id="btn_pan_axis_1" style="font-size: 10px; padding: 6px;">Left Y</button>
                                <button class="btn-axis" id="btn_pan_axis_2" style="font-size: 10px; padding: 6px;">Right X</button>
                                <button class="btn-axis" id="btn_pan_axis_3" style="font-size: 10px; padding: 6px;">Right Y</button>
                            </div>
                            <button class="btn-toggle" id="btn_invert_pan" style="font-size: 10px; padding: 4px; margin-top: 4px; width: 100%;">
                                Invert Pan: OFF
                            </button>
                        </div>
                        <div>
                            <label style="display: block; color: #ccc; margin-bottom: 3px;">Tilt (Vertical)</label>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px;">
                                <button class="btn-axis" id="btn_tilt_axis_0" style="font-size: 10px; padding: 6px;">Left X</button>
                                <button class="btn-axis" id="btn_tilt_axis_1" style="font-size: 10px; padding: 6px;">Left Y</button>
                                <button class="btn-axis" id="btn_tilt_axis_2" style="font-size: 10px; padding: 6px;">Right X</button>
                                <button class="btn-axis" id="btn_tilt_axis_3" style="font-size: 10px; padding: 6px;">Right Y</button>
                            </div>
                            <button class="btn-toggle" id="btn_invert_tilt" style="font-size: 10px; padding: 4px; margin-top: 4px; width: 100%;">
                                Invert Tilt: OFF
                            </button>
                        </div>
                    </div>
                    
                    <!-- Gamepad Buttons Display -->
                    <div style="background: #1a1a1a; padding: 10px; border-radius: 3px; margin-bottom: 10px; text-align: center;">
                        <div style="color: #aaa; margin-bottom: 8px; font-size: 11px; font-weight: bold;">Button Press Display</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 6px;">
                            <div class="gamepad-button" id="gamepadButton0" style="padding: 12px; background: #333; border-radius: 3px; font-weight: bold; font-size: 12px; cursor: default;">
                                A
                            </div>
                            <div class="gamepad-button" id="gamepadButton1" style="padding: 12px; background: #333; border-radius: 3px; font-weight: bold; font-size: 12px; cursor: default;">
                                B
                            </div>
                            <div class="gamepad-button" id="gamepadButton2" style="padding: 12px; background: #333; border-radius: 3px; font-weight: bold; font-size: 12px; cursor: default;">
                                X
                            </div>
                            <div class="gamepad-button" id="gamepadButton3" style="padding: 12px; background: #333; border-radius: 3px; font-weight: bold; font-size: 12px; cursor: default;">
                                Y
                            </div>
                        </div>
                    </div>
                    <div style="text-align: center; margin-bottom: 15px;">
                        <svg id="panTiltIndicator" width="200" height="200" viewBox="0 0 200 200" style="border: 2px solid #444; border-radius: 50%; background: #1a1a1a;">
                            <!-- Outer circle (range limit) -->
                            <circle cx="100" cy="100" r="95" fill="none" stroke="#333" stroke-width="1"/>
                            <!-- Grid lines -->
                            <line x1="100" y1="5" x2="100" y2="195" stroke="#222" stroke-width="1"/>
                            <line x1="5" y1="100" x2="195" y2="100" stroke="#222" stroke-width="1"/>
                            <!-- Pan range indicators (120° each side) -->
                            <line x1="40" y1="100" x2="160" y2="100" stroke="#333" stroke-width="2" stroke-dasharray="3,3"/>
                            <!-- Tilt range (60° each direction shown as arcs) -->
                            <path d="M 100 40 A 60 60 0 0 0 100 160" fill="none" stroke="#333" stroke-width="1" stroke-dasharray="2,2"/>
                            <!-- Current position marker -->
                            <circle id="positionMarker" cx="100" cy="100" r="8" fill="#0f0" stroke="#0f0" stroke-width="2"/>
                            <!-- Crosshair lines from position -->
                            <line id="markerLineH" x1="100" y1="100" x2="100" y2="100" stroke="#0f0" stroke-width="1" opacity="0.5"/>
                            <line id="markerLineV" x1="100" y1="100" x2="100" y2="100" stroke="#0f0" stroke-width="1" opacity="0.5"/>
                        </svg>
                    </div>
                    
                    <!-- Pan/Tilt Angle Display -->
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px; text-align: center;">
                        <div style="background: #1a1a1a; padding: 8px; border-radius: 3px; font-size: 11px;">
                            <div style="color: #aaa;">Pan</div>
                            <div style="font-size: 16px; color: #0f0; font-weight: bold;" id="panAngle">0°</div>
                        </div>
                        <div style="background: #1a1a1a; padding: 8px; border-radius: 3px; font-size: 11px;">
                            <div style="color: #aaa;">Tilt</div>
                            <div style="font-size: 16px; color: #0f0; font-weight: bold;" id="tiltAngle">0°</div>
                        </div>
                    </div>
                    
                    <!-- D-Pad -->
                    <div style="text-align: center; width: 100%;">
                        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 5px; margin-bottom: 8px; align-items: center;">
                            <div></div>
                            <button class="btn-motor" id="btn_motor_up" style="grid-column: 2; padding: 20px 0; font-size: 18px; background: #1e5a3a;">↑</button>
                            <div></div>
                            
                            <button class="btn-motor" id="btn_motor_left" style="padding: 20px 0; font-size: 18px; background: #1e5a3a;">←</button>
                            <button class="btn-motor" id="btn_motor_home" style="padding: 12px 0; font-size: 12px; background: #2a4d2a;">HOME</button>
                            <button class="btn-motor" id="btn_motor_right" style="padding: 20px 0; font-size: 18px; background: #1e5a3a;">→</button>
                            
                            <div></div>
                            <button class="btn-motor" id="btn_motor_down" style="grid-column: 2; padding: 20px 0; font-size: 18px; background: #1e5a3a;">↓</button>
                            <div></div>
                        </div>
                        <div style="font-size: 10px; color: #aaa;">Pan / Tilt Control</div>
                    </div>
                </div>
                
                <!-- Camera Control -->
                <div class="section">
                    <div class="section-title">Camera Feed</div>
                    <div id="cameraSelector" style="margin-bottom: 10px;">
                        <!-- Camera buttons will be inserted here -->
                    </div>
                    <div id="currentCameraInfo" style="font-size: 11px; color: #aaa; margin-bottom: 8px;">
                        Current: /dev/video<span id="currentCameraId">--</span>
                    </div>
                </div>
                
                <!-- Camera Status -->
                <div class="section">
                    <div class="section-title">Status</div>
                    <div id="cameraStatus" style="padding: 8px; background: #1a1a1a; border-radius: 3px; font-size: 12px; text-align: center;">
                        <div id="statusIndicator" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ff4444; margin-right: 8px; vertical-align: middle;"></div>
                        <span id="statusMessage">Connecting...</span>
                    </div>
                    <button class="btn" id="btn_reconnect" style="width: 100%; margin-top: 10px; background: #aa6600;">Reconnect Camera</button>
                </div>
                
                <!-- Display -->
                <div class="section">
                    <div class="section-title">Display</div>
                    <button class="btn" id="btn_w" style="width: 100%; margin-bottom: 8px;">W: Show Text</button>
                    <button class="btn" id="btn_help" style="width: 100%; background: #333388;">?  Help</button>
                    <div class="info">FPS: 50 Hz MJPEG</div>
                </div>
            </div>
        </div>
        
        <!-- Help Overlay Modal -->
        <div id="helpOverlay" class="modal-overlay">
            <div class="modal-content">
                <button class="modal-close" id="closeHelpBtn">&times;</button>
                <div class="modal-header">⌨️ Keyboard Shortcuts</div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Detection Modes</div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">H</div>
                        <div class="shortcut-desc">Heat Seeker Mode</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">C</div>
                        <div class="shortcut-desc">Heat Cluster Mode</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">M</div>
                        <div class="shortcut-desc">Motion Detection</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">P</div>
                        <div class="shortcut-desc">Palette / Thermal Colors</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">T</div>
                        <div class="shortcut-desc">Threshold Mode</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">Y</div>
                        <div class="shortcut-desc">YOLO AI Detection</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">F</div>
                        <div class="shortcut-desc">Optical Flow</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">I</div>
                        <div class="shortcut-desc">Isotherm Mode</div>
                    </div>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Enhancements</div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">D</div>
                        <div class="shortcut-desc">Denoise</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">O</div>
                        <div class="shortcut-desc">Normalize</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">E</div>
                        <div class="shortcut-desc">Enhance</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">U</div>
                        <div class="shortcut-desc">Upscale</div>
                    </div>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Stabilization & Display</div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">S</div>
                        <div class="shortcut-desc">Stabilize</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">X</div>
                        <div class="shortcut-desc">Super Stabilize</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">W</div>
                        <div class="shortcut-desc">Show Text Overlay</div>
                    </div>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Color Palette</div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">[</div>
                        <div class="shortcut-desc">Previous Palette</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">]</div>
                        <div class="shortcut-desc">Next Palette</div>
                    </div>
                </div>
                
                <div class="modal-section">
                    <div class="modal-section-title">Motor / Pan-Tilt Control</div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">↑</div>
                        <div class="shortcut-desc">Tilt Up (max +60°)</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">↓</div>
                        <div class="shortcut-desc">Tilt Down (max -60°)</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">←</div>
                        <div class="shortcut-desc">Pan Left (max -150°)</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">→</div>
                        <div class="shortcut-desc">Pan Right (max +150°)</div>
                    </div>
                    <div class="shortcut-item">
                        <div class="shortcut-key">HOME</div>
                        <div class="shortcut-desc">Return to Center Position</div>
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: #1a1a1a; border-radius: 3px; font-size: 11px; color: #888;">
                        Visual crosshair indicator shows current pan/tilt position. Camera resets to center when powered on.
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 20px; font-size: 12px; color: #888;">
                    <p>Press <strong>?</strong> or click "Help" to close this overlay</p>
                </div>
            </div>
        </div>
        <script>
            const buttons = {
                'h': 'heat_seeker_mode',
                'c': 'heat_cluster_mode',
                'm': 'motion_mode',
                'p': 'palette_mode',
                't': 'threshold_mode',
                'y': 'yolo_mode',
                'f': 'optical_flow_mode',
                'i': 'isotherm_mode',
                'd': 'denoise_mode',
                'o': 'normalize_mode',
                'e': 'enhance_mode',
                'u': 'upscale_mode',
                's': 'stabilize_mode',
                'x': 'stabilize_super',
                'w': 'show_text'
            };
            
            const sliders = {
                'threshold_value': 'slider_threshold_value',
                'stabilize_strength': 'slider_stabilize_strength',
                'isotherm_min': 'slider_isotherm_min',
                'isotherm_max': 'slider_isotherm_max'
            };
            
            const palettes = [
                'Ironbow',
                'Rainbow',
                'Lava',
                'Ocean',
                'Magma',
                'WhiteHot',
                'BlackHot'
            ];
            
            let ws = null;
            let state = {};
            let availableCameras = [];
            let currentCamera = null;
            let currentPaletteIdx = 0;
            
            // Pan/Tilt tracking
            const PAN_MAX = 150;      // ±150° for ~300° total
            const TILT_MAX = 60;      // ±60° for ~120° total
            const PAN_STEP = 5;       // Degrees per button press
            const TILT_STEP = 5;      // Degrees per button press
            const GAMEPAD_DEADZONE = 0.15;  // Deadzone for stick drift
            const GAMEPAD_SENSITIVITY = 100; // Max degrees per second from stick
            
            let currentPan = 0;
            let currentTilt = 0;
            let motorActive = {};     // Track which motor is active
            let activeGamepadIndex = -1;  // -1 means no gamepad active
            let connectedGamepads = [];    // List of connected gamepads
            let lastGamepadPoll = Date.now();
            
            // Gamepad axis configuration
            let gamepadPanAxis = 0;      // 0=leftX, 1=leftY, 2=rightX, 3=rightY
            let gamepadTiltAxis = 1;     // 0=leftX, 1=leftY, 2=rightX, 3=rightY
            let gamepadInvertPan = false;
            let gamepadInvertTilt = false;
            const AXIS_NAMES = ['Left X', 'Left Y', 'Right X', 'Right Y'];
            
            // Gamepad presets
            let currentPreset = 'normal';
            const gamepadPresets = {
                'normal': {
                    panAxis: 0,          // Left X
                    tiltAxis: 1,         // Left Y
                    invertPan: false,
                    invertTilt: false,
                    label: 'Normal'
                },
                'vertical': {
                    panAxis: 1,          // Left Y
                    tiltAxis: 0,         // Left X
                    invertPan: true,     // Invert pan
                    invertTilt: false,   // Don't invert tilt
                    label: 'Vertical'
                }
            };
            
            function updateCameraSelector() {
                fetch('/available-cameras')
                    .then(r => r.json())
                    .then(data => {
                        availableCameras = data.available;
                        currentCamera = data.current;
                        
                        // Update the selector
                        const selector = document.getElementById('cameraSelector');
                        selector.innerHTML = '';
                        
                        if (availableCameras.length === 0) {
                            selector.innerHTML = '<p style="font-size: 11px; color: #aaa;">No cameras available</p>';
                        } else {
                            availableCameras.forEach(cameraId => {
                                const btn = document.createElement('button');
                                btn.className = 'btn';
                                btn.style.width = '100%';
                                btn.style.marginBottom = '5px';
                                btn.textContent = `/dev/video${cameraId}`;
                                
                                if (cameraId === currentCamera) {
                                    btn.classList.add('active');
                                }
                                
                                btn.addEventListener('click', () => {
                                    switchCamera(cameraId);
                                });
                                
                                selector.appendChild(btn);
                            });
                        }
                        
                        // Update current camera display
                        document.getElementById('currentCameraId').textContent = currentCamera !== null ? currentCamera : '--';
                    })
                    .catch(e => console.error('Failed to fetch cameras:', e));
            }
            
            function updatePanTiltIndicator() {
                // Map pan/tilt angles to SVG position
                // SVG is 200x200, center at 100,100, radius ~85 for safe zone
                const svgRadius = 75;
                const panPercent = currentPan / PAN_MAX;    // -1 to +1
                const tiltPercent = currentTilt / TILT_MAX;  // -1 to +1
                
                const x = 100 + (panPercent * svgRadius);
                const y = 100 - (tiltPercent * svgRadius);   // Y inverted (up is positive)
                
                // Update position marker
                const marker = document.getElementById('positionMarker');
                marker.setAttribute('cx', x);
                marker.setAttribute('cy', y);
                
                // Update crosshair lines
                document.getElementById('markerLineH').setAttribute('x2', x);
                document.getElementById('markerLineH').setAttribute('y2', y);
                document.getElementById('markerLineV').setAttribute('x2', x);
                document.getElementById('markerLineV').setAttribute('y2', y);
                
                // Update angle displays
                document.getElementById('panAngle').textContent = currentPan + '°';
                document.getElementById('tiltAngle').textContent = currentTilt + '°';
            }
            
            function updateGamepadStatus() {
                // Get all connected gamepads
                const gamepads = navigator.getGamepads?.() || [];
                connectedGamepads = Array.from(gamepads).filter(gp => gp !== null);
                
                const indicator = document.getElementById('gamepadIndicator');
                const statusText = document.getElementById('gamepadStatusText');
                const nameDisplay = document.getElementById('gamepadNameDisplay');
                const countDisplay = document.getElementById('gamepadDeviceCount');
                
                countDisplay.textContent = connectedGamepads.length + ' device' + (connectedGamepads.length !== 1 ? 's' : '');
                
                // Check if active gamepad is still connected
                if (activeGamepadIndex >= 0 && activeGamepadIndex < connectedGamepads.length) {
                    const activeGpad = connectedGamepads[activeGamepadIndex];
                    indicator.classList.add('connected');
                    statusText.textContent = '✓ Connected';
                    nameDisplay.textContent = activeGpad.id;
                } else if (connectedGamepads.length > 0) {
                    // Auto-select first available if active is missing
                    activeGamepadIndex = 0;
                    const activeGpad = connectedGamepads[0];
                    indicator.classList.add('connected');
                    statusText.textContent = '✓ Connected';
                    nameDisplay.textContent = activeGpad.id;
                } else {
                    activeGamepadIndex = -1;
                    indicator.classList.remove('connected');
                    statusText.textContent = '✗ Disconnected';
                    nameDisplay.textContent = '—';
                }
            }
            
            function cycleGamepad() {
                if (connectedGamepads.length === 0) {
                    return;
                }
                activeGamepadIndex = (activeGamepadIndex + 1) % connectedGamepads.length;
                updateGamepadStatus();
                console.log(`Switched to gamepad: ${connectedGamepads[activeGamepadIndex].id}`);
            }
            
            function pollGamepadInput() {
                if (activeGamepadIndex < 0 || activeGamepadIndex >= connectedGamepads.length) {
                    return;
                }
                
                const gamepad = connectedGamepads[activeGamepadIndex];
                
                // Read from configurable axes
                let panInput = gamepad.axes[gamepadPanAxis] || 0;
                let tiltInput = gamepad.axes[gamepadTiltAxis] || 0;
                
                // Apply inversion
                if (gamepadInvertPan) panInput *= -1;
                if (gamepadInvertTilt) tiltInput *= -1;
                
                // Invert tilt by default (stick up = tilt up on camera)
                tiltInput *= -1;
                
                // Apply deadzone
                panInput = Math.abs(panInput) > GAMEPAD_DEADZONE ? panInput : 0;
                tiltInput = Math.abs(tiltInput) > GAMEPAD_DEADZONE ? tiltInput : 0;
                
                // Convert stick input to angle change
                const timeDelta = (Date.now() - lastGamepadPoll) / 1000;
                lastGamepadPoll = Date.now();
                
                if (Math.abs(panInput) > 0.01 || Math.abs(tiltInput) > 0.01) {
                    const panChange = panInput * GAMEPAD_SENSITIVITY * timeDelta;
                    const tiltChange = tiltInput * GAMEPAD_SENSITIVITY * timeDelta;
                    
                    // Update angles with constraints
                    currentPan = Math.max(-PAN_MAX, Math.min(PAN_MAX, currentPan + panChange));
                    currentTilt = Math.max(-TILT_MAX, Math.min(TILT_MAX, currentTilt + tiltChange));
                    
                    updatePanTiltIndicator();
                    
                    // Send to server if changed
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            action: 'motor_command',
                            command: 'gamepad_input',
                            pan: Math.round(currentPan),
                            tilt: Math.round(currentTilt)
                        }));
                    }
                }
                
                // Update button display
                updateGamepadButtonDisplay(gamepad);
            }
            
            function updateGamepadButtonDisplay(gamepad) {
                // Update visual indicators for buttons 0-3 (ABXY)
                for (let i = 0; i < 4; i++) {
                    const button = gamepad.buttons[i];
                    const buttonElement = document.getElementById('gamepadButton' + i);
                    
                    if (button && button.pressed) {
                        buttonElement.classList.add('pressed');
                    } else {
                        buttonElement.classList.remove('pressed');
                    }
                }
            }
            
            function applyPreset(presetName) {
                if (!gamepadPresets[presetName]) {
                    console.error('Unknown preset:', presetName);
                    return;
                }
                
                const preset = gamepadPresets[presetName];
                currentPreset = presetName;
                gamepadPanAxis = preset.panAxis;
                gamepadTiltAxis = preset.tiltAxis;
                gamepadInvertPan = preset.invertPan;
                gamepadInvertTilt = preset.invertTilt;
                
                console.log('Applied preset:', presetName);
                updateGamepadAxisDisplay();
            }
            
            function updateGamepadAxisDisplay() {
                // Update visual indicator of which axis is selected
                for (let axisType of ['pan', 'tilt']) {
                    const selectedAxis = axisType === 'pan' ? gamepadPanAxis : gamepadTiltAxis;
                    for (let i = 0; i < 4; i++) {
                        const btn = document.getElementById('btn_' + axisType + '_axis_' + i);
                        if (i === selectedAxis) {
                            btn.classList.add('active');
                        } else {
                            btn.classList.remove('active');
                        }
                    }
                }
                
                // Update invert toggle displays
                const invertPanBtn = document.getElementById('btn_invert_pan');
                const invertTiltBtn = document.getElementById('btn_invert_tilt');
                
                if (gamepadInvertPan) {
                    invertPanBtn.classList.add('active');
                    invertPanBtn.textContent = 'Invert Pan: ON';
                } else {
                    invertPanBtn.classList.remove('active');
                    invertPanBtn.textContent = 'Invert Pan: OFF';
                }
                
                if (gamepadInvertTilt) {
                    invertTiltBtn.classList.add('active');
                    invertTiltBtn.textContent = 'Invert Tilt: ON';
                } else {
                    invertTiltBtn.classList.remove('active');
                    invertTiltBtn.textContent = 'Invert Tilt: OFF';
                }
                
                // Update preset button displays
                const normalBtn = document.getElementById('btn_preset_normal');
                const verticalBtn = document.getElementById('btn_preset_vertical');
                
                if (currentPreset === 'normal') {
                    normalBtn.classList.add('active');
                    verticalBtn.classList.remove('active');
                } else if (currentPreset === 'vertical') {
                    verticalBtn.classList.add('active');
                    normalBtn.classList.remove('active');
                } else {
                    normalBtn.classList.remove('active');
                    verticalBtn.classList.remove('active');
                }
                
                // Update preset display text
                const presetDisplay = document.getElementById('currentPresetDisplay');
                let presetLabel = 'Custom';
                for (let key in gamepadPresets) {
                    if (currentPreset === key) {
                        presetLabel = gamepadPresets[key].label;
                        break;
                    }
                }
                presetDisplay.textContent = 'Current: ' + presetLabel;
            }
            
            function updateMotorAngle(command) {
                const maxStepsPerSecond = 20; // Max increment steps per second
                const increment = PAN_STEP;
                
                switch(command) {
                    case 'motor_left':
                        currentPan = Math.max(-PAN_MAX, currentPan - increment);
                        break;
                    case 'motor_right':
                        currentPan = Math.min(PAN_MAX, currentPan + increment);
                        break;
                    case 'motor_up':
                        currentTilt = Math.min(TILT_MAX, currentTilt + increment);
                        break;
                    case 'motor_down':
                        currentTilt = Math.max(-TILT_MAX, currentTilt - increment);
                        break;
                    case 'motor_home':
                        currentPan = 0;
                        currentTilt = 0;
                        break;
                }
                
                updatePanTiltIndicator();
            }
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                ws = new WebSocket(protocol + '://' + window.location.host + '/control');
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    document.getElementById('statusText').innerHTML = 'Connected';
                    updateCameraSelector();
                };
                
                ws.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'state') {
                        state = msg.data;
                        updateUI();
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    document.getElementById('statusText').innerHTML = 'Connection error';
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected, reconnecting...');
                    setTimeout(connectWebSocket, 2000);
                };
            }
            
            function updateUI() {
                // Update buttons
                for (const [key, mode] of Object.entries(buttons)) {
                    const btn = document.getElementById('btn_' + key);
                    if (state[mode]) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                }
                
                // Update sliders
                if (state.threshold_value !== undefined) {
                    document.getElementById('slider_threshold_value').value = state.threshold_value;
                    document.getElementById('val_threshold').textContent = state.threshold_value;
                }
                if (state.stabilize_strength !== undefined) {
                    document.getElementById('slider_stabilize_strength').value = state.stabilize_strength * 100;
                    document.getElementById('stab_strength').textContent = state.stabilize_strength.toFixed(1);
                }
                if (state.isotherm_min !== undefined) {
                    document.getElementById('slider_isotherm_min').value = state.isotherm_min;
                    document.getElementById('val_isotherm_min').textContent = state.isotherm_min;
                }
                if (state.isotherm_max !== undefined) {
                    document.getElementById('slider_isotherm_max').value = state.isotherm_max;
                    document.getElementById('val_isotherm_max').textContent = state.isotherm_max;
                }
                
                // Update palette display
                if (state.palette_idx !== undefined) {
                    currentPaletteIdx = state.palette_idx;
                    const paletteName = palettes[currentPaletteIdx] || 'Unknown';
                    document.getElementById('currentPaletteName').textContent = paletteName;
                }
                
                // Update camera selector highlighting
                const buttons_list = document.querySelectorAll('#cameraSelector button');
                buttons_list.forEach(btn => {
                    const btnId = parseInt(btn.textContent.match(/\d+/)[0]);
                    if (btnId === currentCamera) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                });
                
                // Update status
                let statusLines = [];
                if (state.palette_name) statusLines.push('Palette: ' + state.palette_name);
                statusLines.push('Connected');
                document.getElementById('statusText').innerHTML = statusLines.join('<br>');
            }
            
            function switchCamera(newCameraId) {
                console.log('Switching to camera /dev/video' + newCameraId);
                fetch(`/switch-camera/${newCameraId}`, { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        console.log('Switch response:', data);
                        currentCamera = newCameraId;
                        updateCameraSelector();
                        // Highlight the newly selected camera
                        setTimeout(updateUI, 100);
                    })
                    .catch(e => console.error('Switch error:', e));
            }
            
            // Button click handlers
            for (const [key, mode] of Object.entries(buttons)) {
                document.getElementById('btn_' + key).addEventListener('click', () => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            action: 'toggle_mode',
                            mode: mode
                        }));
                    }
                });
            }
            
            // Slider handlers
            document.getElementById('slider_threshold_value').addEventListener('change', (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'threshold_value',
                        value: parseInt(e.target.value)
                    }));
                }
            });
            
            document.getElementById('slider_stabilize_strength').addEventListener('change', (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'stabilize_strength',
                        value: parseFloat(e.target.value) / 100
                    }));
                }
            });
            
            document.getElementById('slider_isotherm_min').addEventListener('change', (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'isotherm_min',
                        value: parseInt(e.target.value)
                    }));
                }
            });
            
            document.getElementById('slider_isotherm_max').addEventListener('change', (e) => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'isotherm_max',
                        value: parseInt(e.target.value)
                    }));
                }
            });
            
            // Palette cycling handlers
            document.getElementById('btn_palette_prev').addEventListener('click', () => {
                const newIdx = (currentPaletteIdx - 1 + palettes.length) % palettes.length;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'palette_idx',
                        value: newIdx
                    }));
                }
            });
            
            document.getElementById('btn_palette_next').addEventListener('click', () => {
                const newIdx = (currentPaletteIdx + 1) % palettes.length;
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'palette_idx',
                        value: newIdx
                    }));
                }
            });
            
            // Motor control handlers
            const motorCommands = {
                'btn_motor_up': 'motor_up',
                'btn_motor_down': 'motor_down',
                'btn_motor_left': 'motor_left',
                'btn_motor_right': 'motor_right',
                'btn_motor_home': 'motor_home'
            };
            
            let motorIntervals = {}; // Track active motor intervals
            
            for (const [btnId, command] of Object.entries(motorCommands)) {
                const btn = document.getElementById(btnId);
                if (btn) {
                    const startMotor = () => {
                        motorActive[command] = true;
                        updateMotorAngle(command);
                        
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                action: 'motor_command',
                                command: command,
                                state: 'start'
                            }));
                        }
                        
                        // Continuous movement while held
                        motorIntervals[command] = setInterval(() => {
                            if (motorActive[command]) {
                                updateMotorAngle(command);
                            }
                        }, 100);  // Update every 100ms = 10 steps per second
                    };
                    
                    const stopMotor = () => {
                        motorActive[command] = false;
                        if (motorIntervals[command]) {
                            clearInterval(motorIntervals[command]);
                            delete motorIntervals[command];
                        }
                        
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({
                                action: 'motor_command',
                                command: command,
                                state: 'stop'
                            }));
                        }
                    };
                    
                    btn.addEventListener('mousedown', startMotor);
                    btn.addEventListener('mouseup', stopMotor);
                    btn.addEventListener('mouseleave', stopMotor);
                    
                    // Touch support for mobile
                    btn.addEventListener('touchstart', (e) => {
                        e.preventDefault();
                        startMotor();
                    });
                    
                    btn.addEventListener('touchend', (e) => {
                        e.preventDefault();
                        stopMotor();
                    });
                }
            }
            
            // Keyboard shortcuts
            const keyPressState = {};
            
            document.addEventListener('keydown', (e) => {
                const key = e.key.toLowerCase();
                
                // Mode toggle shortcuts
                if (buttons[key] && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'toggle_mode',
                        mode: buttons[key]
                    }));
                }
                
                // Palette cycling with [ and ]
                if (key === '[' && ws && ws.readyState === WebSocket.OPEN) {
                    const newIdx = (currentPaletteIdx - 1 + palettes.length) % palettes.length;
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'palette_idx',
                        value: newIdx
                    }));
                    e.preventDefault();
                }
                if (key === ']' && ws && ws.readyState === WebSocket.OPEN) {
                    const newIdx = (currentPaletteIdx + 1) % palettes.length;
                    ws.send(JSON.stringify({
                        action: 'set_param',
                        param: 'palette_idx',
                        value: newIdx
                    }));
                    e.preventDefault();
                }
                
                // Arrow key motor control
                if ((key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') && ws && ws.readyState === WebSocket.OPEN) {
                    const motorMap = {
                        'arrowup': 'motor_up',
                        'arrowdown': 'motor_down',
                        'arrowleft': 'motor_left',
                        'arrowright': 'motor_right'
                    };
                    
                    if (!keyPressState[key]) {
                        keyPressState[key] = true;
                        const command = motorMap[key];
                        motorActive[command] = true;
                        updateMotorAngle(command);
                        
                        ws.send(JSON.stringify({
                            action: 'motor_command',
                            command: command,
                            state: 'start'
                        }));
                        
                        // Continuous movement
                        motorIntervals[command] = setInterval(() => {
                            if (motorActive[command]) {
                                updateMotorAngle(command);
                            }
                        }, 100);
                    }
                    e.preventDefault();
                }
            });
            
            document.addEventListener('keyup', (e) => {
                const key = e.key.toLowerCase();
                
                // Arrow key motor stop
                if ((key === 'arrowup' || key === 'arrowdown' || key === 'arrowleft' || key === 'arrowright') && ws && ws.readyState === WebSocket.OPEN) {
                    const motorMap = {
                        'arrowup': 'motor_up',
                        'arrowdown': 'motor_down',
                        'arrowleft': 'motor_left',
                        'arrowright': 'motor_right'
                    };
                    
                    const command = motorMap[key];
                    motorActive[command] = false;
                    if (motorIntervals[command]) {
                        clearInterval(motorIntervals[command]);
                        delete motorIntervals[command];
                    }
                    keyPressState[key] = false;
                    
                    ws.send(JSON.stringify({
                        action: 'motor_command',
                        command: command,
                        state: 'stop'
                    }));
                    e.preventDefault();
                }
                
                // Help toggle with ?
                if ((key === '?' || e.key === '?') || (e.shiftKey && key === '/')) {
                    const isOpen = helpOverlay.classList.contains('active');
                    if (isOpen) {
                        closeHelp();
                    } else {
                        openHelp();
                    }
                    e.preventDefault();
                }
            });
            
            // Camera status polling
            function updateCameraStatus() {
                fetch('/camera-status')
                    .then(r => r.json())
                    .then(data => {
                        const indicator = document.getElementById('statusIndicator');
                        const message = document.getElementById('statusMessage');
                        
                        if (data.connected) {
                            indicator.style.background = '#00aa00';
                            message.textContent = '✓ Camera Connected';
                        } else {
                            indicator.style.background = '#ff4444';
                            message.textContent = '✗ Camera Disconnected';
                        }
                    })
                    .catch(e => console.error('Status fetch error:', e));
            }
            
            // Reconnect button handler
            document.getElementById('btn_reconnect').addEventListener('click', () => {
                console.log('Attempting camera reconnection...');
                fetch('/reconnect', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        console.log('Reconnect response:', data);
                        // Poll status more frequently after reconnect request
                        for (let i = 0; i < 10; i++) {
                            setTimeout(updateCameraStatus, i * 500);
                        }
                    })
                    .catch(e => console.error('Reconnect error:', e));
            });
            
            // Help modal handlers
            const helpOverlay = document.getElementById('helpOverlay');
            const helpBtn = document.getElementById('btn_help');
            const closeHelpBtn = document.getElementById('closeHelpBtn');
            
            function openHelp() {
                helpOverlay.classList.add('active');
            }
            
            function closeHelp() {
                helpOverlay.classList.remove('active');
            }
            
            helpBtn.addEventListener('click', openHelp);
            closeHelpBtn.addEventListener('click', closeHelp);
            
            // Close help when clicking outside the modal
            helpOverlay.addEventListener('click', (e) => {
                if (e.target === helpOverlay) {
                    closeHelp();
                }
            });
            
            // Poll camera status every 1 second
            setInterval(updateCameraStatus, 1000);
            setInterval(updateCameraSelector, 3000);  // Update available cameras every 3 seconds
            
            // Initialize pan/tilt indicator
            updatePanTiltIndicator();
            
            // Gamepad button handler
            const cycleBtn = document.getElementById('btn_cycle_gamepad');
            if (cycleBtn) {
                cycleBtn.addEventListener('click', cycleGamepad);
            }
            
            // Gamepad connection/disconnection events
            window.addEventListener('gamepadconnected', (e) => {
                console.log('Gamepad connected:', e.gamepad.id);
                updateGamepadStatus();
            });
            
            window.addEventListener('gamepaddisconnected', (e) => {
                console.log('Gamepad disconnected');
                updateGamepadStatus();
            });
            
            // Gamepad polling loop (50ms = 20 Hz)
            setInterval(() => {
                updateGamepadStatus();
                pollGamepadInput();
            }, 50);
            
            // Initial gamepad detection
            updateGamepadStatus();
            updateGamepadAxisDisplay();
            
            // Gamepad axis configuration buttons
            // Pan axis buttons
            for (let i = 0; i < 4; i++) {
                const btn = document.getElementById('btn_pan_axis_' + i);
                if (btn) {
                    btn.addEventListener('click', () => {
                        gamepadPanAxis = i;
                        updateGamepadAxisDisplay();
                        console.log('Pan axis set to: ' + AXIS_NAMES[i]);
                    });
                }
            }
            
            // Tilt axis buttons
            for (let i = 0; i < 4; i++) {
                const btn = document.getElementById('btn_tilt_axis_' + i);
                if (btn) {
                    btn.addEventListener('click', () => {
                        gamepadTiltAxis = i;
                        updateGamepadAxisDisplay();
                        console.log('Tilt axis set to: ' + AXIS_NAMES[i]);
                    });
                }
            }
            
            // Invert buttons
            const invertPanBtn = document.getElementById('btn_invert_pan');
            if (invertPanBtn) {
                invertPanBtn.addEventListener('click', () => {
                    gamepadInvertPan = !gamepadInvertPan;
                    updateGamepadAxisDisplay();
                    console.log('Pan inversion: ' + (gamepadInvertPan ? 'ON' : 'OFF'));
                });
            }
            
            const invertTiltBtn = document.getElementById('btn_invert_tilt');
            if (invertTiltBtn) {
                invertTiltBtn.addEventListener('click', () => {
                    gamepadInvertTilt = !gamepadInvertTilt;
                    updateGamepadAxisDisplay();
                    console.log('Tilt inversion: ' + (gamepadInvertTilt ? 'ON' : 'OFF'));
                });
            }
            
            // Gamepad preset buttons
            const normalPresetBtn = document.getElementById('btn_preset_normal');
            if (normalPresetBtn) {
                normalPresetBtn.addEventListener('click', () => {
                    applyPreset('normal');
                    console.log('Switched to Normal preset');
                });
            }
            
            const verticalPresetBtn = document.getElementById('btn_preset_vertical');
            if (verticalPresetBtn) {
                verticalPresetBtn.addEventListener('click', () => {
                    applyPreset('vertical');
                    console.log('Switched to Vertical preset');
                });
            }
            
            // Connect on load
            connectWebSocket();
        </script>
    </body>
    </html>
        """
        return HTMLResponse(content=html)
    
    return app


def run_webserver(host="0.0.0.0", port=8000, camera_id=0):
    """Launch FastAPI web server with specified camera."""
    import uvicorn
    
    print(f"Creating FastAPI app (camera: /dev/video{camera_id})...")
    app = create_app(camera_id)
    
    print(f"Starting web server on {host}:{port}")
    print(f"Open browser: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
