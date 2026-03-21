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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

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


def capture_frames(camera_id=None):
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
        if camera_id is None:
            print(f"Camera thread started (waiting for any device to appear)")
        else:
            print(f"Camera thread started (waiting for device {camera_id})")
    else:
        print(f"  ✓ Found working cameras: {working_cameras}")
        # If the requested camera isn't in the working list, use the first working one
        if camera_id is None:
            # Auto-select first working camera
            camera_id = working_cameras[0]
            print(f"Auto-selecting first working camera: /dev/video{camera_id}")
        elif camera_id not in working_cameras:
            print(f"Requested /dev/video{camera_id} not in working list, using /dev/video{working_cameras[0]}")
            camera_id = working_cameras[0]
        print(f"Camera thread started (attempting device {camera_id})")
    
    # If no cameras found and none specified, wait for one to appear
    if camera_id is None and not working_cameras:
        while camera_id is None:
            time.sleep(0.5)
            for test_id in range(5):
                test_cap = try_open_camera(test_id)
                if test_cap is not None:
                    camera_id = test_id
                    print(f"✓ Detected camera at /dev/video{test_id}")
                    test_cap.release()
                    break
    
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


def create_app(camera_id=None):
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global camera_thread, processor
        
        # Startup
        print(f"Initializing processor...")
        processor = ThermalProcessor()
        camera_desc = f"/dev/video{camera_id}" if camera_id is not None else "auto-detect (none found)"
        print(f"Starting camera thread (device {camera_desc})...")
        camera_thread = threading.Thread(target=capture_frames, args=(camera_id,), daemon=True)
        camera_thread.start()
        print("✓ Web server started")
        
        yield
        
        # Shutdown
        print("Web server shutting down")
    
    app = FastAPI(title="Thermal Camera Viewer", lifespan=lifespan)
    
    # Setup templates and static files
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
    
    # Add CORS middleware
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
    async def root(request: Request):
        """Serve HTML UI."""
        return templates.TemplateResponse("index.html", {"request": request})
    
    return app


def run_webserver(host="0.0.0.0", port=8000, camera_id=None):
    """Launch FastAPI web server with specified camera."""
    import uvicorn
    
    camera_desc = f"/dev/video{camera_id}" if camera_id is not None else "auto-detect (none found)"
    print(f"Creating FastAPI app (camera: {camera_desc})...")
    app = create_app(camera_id)
    
    print(f"Starting web server on {host}:{port}")
    print(f"Open browser: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
