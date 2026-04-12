"""
FastAPI web server for thermal camera with MJPEG streaming and WebSocket control.
Optimized for 50 Hz video streaming over WiFi on embedded systems (Orange Pi).
"""

import os
import cv2
import json
import asyncio
import threading
import time
import numpy as np
from io import BytesIO
from queue import Queue
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress OpenCV warnings (camera not found messages)
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
cv2.setLogLevel(0)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

from cookie_finder.camera.processor import ThermalProcessor
from cookie_finder.gimbal.pan_tilt import PanTiltGimbal
from cookie_finder.bluetooth.controller import BluetoothController


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
gimbal = None  # PanTiltGimbal instance
motor_moving = {}  # Track which motors are moving: {command: True/False}
bluetooth_controller = None  # BluetoothController instance
bt_active_clients = set()  # WebSocket clients listening to BT updates
gimbal_position = {"pan": 0.0, "tilt": 0.0}  # Current gimbal angles
gimbal_lock = threading.Lock()  # Thread-safe access to gimbal_position
bt_device_connected = False  # Track if BT device is connected for input
control_loop = None  # Event loop used for cross-thread WebSocket broadcasts


def sync_gimbal_position() -> dict:
    """Copy the current hardware gimbal position into shared server state."""
    if gimbal is None:
        return {"pan": 0.0, "tilt": 0.0}

    pan, tilt = gimbal.get_position()
    with gimbal_lock:
        gimbal_position["pan"] = pan
        gimbal_position["tilt"] = tilt
        return gimbal_position.copy()


async def broadcast_gimbal_position(pos_data: dict | None = None) -> None:
    """Broadcast the latest gimbal position to all connected WebSocket clients."""
    if pos_data is None:
        with gimbal_lock:
            pos_data = gimbal_position.copy()

    disconnected_clients = []
    for client in list(bt_active_clients):
        try:
            await client.send_json({"type": "gimbal_position", "data": pos_data})
        except Exception:
            disconnected_clients.append(client)

    for client in disconnected_clients:
        bt_active_clients.discard(client)
        active_clients.discard(client)


def broadcast_gimbal_position_threadsafe(pos_data: dict | None = None) -> None:
    """Broadcast gimbal position from worker threads on the main event loop."""
    if pos_data is None:
        pos_data = sync_gimbal_position()

    if control_loop is None:
        return

    try:
        asyncio.run_coroutine_threadsafe(broadcast_gimbal_position(pos_data), control_loop)
    except Exception:
        pass


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


def poll_bluetooth_controller():
    """
    Background thread: Poll connected Bluetooth device for input and control gimbal.
    Converts joystick input to gimbal angles and broadcasts to all connected WebSocket clients.
    """
    global gimbal, bluetooth_controller, gimbal_position, gimbal_lock, bt_device_connected, bt_active_clients
    
    print("[BT] Input polling thread started")
    last_pan = 0.0
    last_tilt = 0.0
    last_update_time = time.time()
    deadzone = 0.15
    sensitivity = 100.0
    
    while True:
        try:
            # Check if we have a connected device
            if not bluetooth_controller or not bluetooth_controller.get_connected_device():
                bt_device_connected = False
                time.sleep(0.5)
                continue
            
            bt_device_connected = True
            
            # Read input from connected device
            input_data = bluetooth_controller.read_controller_input()
            pan_axis = input_data.get("pan_axis", 0.0)
            tilt_axis = input_data.get("tilt_axis", 0.0)
            now = time.time()
            time_delta = max(0.01, now - last_update_time)
            last_update_time = now

            pan_axis = pan_axis if abs(pan_axis) > deadzone else 0.0
            tilt_axis = tilt_axis if abs(tilt_axis) > deadzone else 0.0
            
            # Skip if input hasn't changed much (deadzone)
            if abs(pan_axis - last_pan) < 0.05 and abs(tilt_axis - last_tilt) < 0.05:
                time.sleep(0.05)
                continue
            
            last_pan = pan_axis
            last_tilt = tilt_axis
            
            if gimbal is None:
                time.sleep(0.05)
                continue

            # Apply joystick input as incremental movement so neutral does not force center.
            current_pan, current_tilt = gimbal.get_position()
            new_pan = max(0.0, min(gimbal.max_pan, current_pan + (pan_axis * sensitivity * time_delta)))
            new_tilt = max(0.0, min(gimbal.max_tilt, current_tilt + (-tilt_axis * sensitivity * time_delta)))

            if abs(new_pan - current_pan) < 0.01 and abs(new_tilt - current_tilt) < 0.01:
                time.sleep(0.05)
                continue

            # Move gimbal to new angles
            gimbal.move_to_angles(new_pan, new_tilt)
            
            # Update global position and broadcast to connected clients
            with gimbal_lock:
                gimbal_position["pan"] = new_pan
                gimbal_position["tilt"] = new_tilt
                current_position = gimbal_position.copy()

            broadcast_gimbal_position_threadsafe(current_position)
            
            print(f"[BT] Gimbal moved via BT device: pan={new_pan:.1f}°, tilt={new_tilt:.1f}°")
            
            time.sleep(0.05)  # Poll at 20Hz
        
        except Exception as e:
            print(f"[BT] Polling error: {e}")
            time.sleep(0.5)


def create_app(camera_id=None):
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global camera_thread, processor, gimbal, bluetooth_controller, control_loop
        
        # Startup
        print(f"Initializing processor...")
        control_loop = asyncio.get_running_loop()
        processor = ThermalProcessor()
        
        # Initialize gimbal for motor control
        try:
            print(f"Initializing gimbal (pan/tilt motors)...")
            gimbal = PanTiltGimbal(max_pan=150.0, max_tilt=60.0)
            gimbal.set_speed(pan_hz=500, tilt_hz=500)
            print(f"✓ Gimbal initialized")
        except Exception as e:
            print(f"⚠ Gimbal initialization failed (GPIO may require root): {e}")
            gimbal = None
        
        # Initialize Bluetooth controller
        try:
            print(f"Initializing Bluetooth controller...")
            bluetooth_controller = BluetoothController()
            
            def bt_status_callback(update):
                """Broadcast Bluetooth status to all connected WebSocket clients."""
                for client in bt_active_clients:
                    try:
                        import asyncio
                        asyncio.create_task(client.send_json({"type": "bluetooth", "data": update}))
                    except:
                        pass
            
            bluetooth_controller.set_status_callback(bt_status_callback)
            print(f"✓ Bluetooth controller initialized")
        except Exception as e:
            print(f"⚠ Bluetooth initialization failed: {e}")
            bluetooth_controller = None
        
        camera_desc = f"/dev/video{camera_id}" if camera_id is not None else "auto-detect (none found)"
        print(f"Starting camera thread (device {camera_desc})...")
        camera_thread = threading.Thread(target=capture_frames, args=(camera_id,), daemon=True)
        camera_thread.start()
        
        # Start Bluetooth input polling thread
        print(f"Starting Bluetooth input polling thread...")
        bt_polling_thread = threading.Thread(target=poll_bluetooth_controller, daemon=True)
        bt_polling_thread.start()
        
        print("✓ Web server started")
        
        yield
        
        # Shutdown
        if gimbal is not None:
            gimbal.cleanup()
        if bluetooth_controller is not None and bluetooth_controller.scanning:
            bluetooth_controller.stop_scan()
        print("Web server shutting down")
    
    app = FastAPI(title="Thermal Camera Viewer", lifespan=lifespan)
    
    # Setup templates and static files
    web_dir = Path(__file__).parent
    app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
    
    # Initialize Jinja2 environment
    jinja_env = Environment(loader=FileSystemLoader(str(web_dir / "templates")))
    
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
    
    @app.post("/bluetooth/scan")
    async def bluetooth_scan():
        """Start Bluetooth device scan."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        if bluetooth_controller.scanning:
            return {"status": "already_scanning", "message": "Scan already in progress"}
        
        try:
            bluetooth_controller.start_scan()
            return {"status": "scan_started", "message": "Bluetooth scan started..."}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/bluetooth/stop-scan")
    async def bluetooth_stop_scan():
        """Stop Bluetooth device scan."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        try:
            bluetooth_controller.stop_scan()
            return {"status": "scan_stopped", "message": "Bluetooth scan stopped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.get("/bluetooth/devices")
    async def bluetooth_get_devices():
        """Get list of discovered Bluetooth devices."""
        if bluetooth_controller is None:
            return {"devices": [], "scanning": False}
        
        return {
            "devices": bluetooth_controller.get_devices_list(),
            "scanning": bluetooth_controller.scanning
        }
    
    @app.get("/bluetooth/connected")
    async def bluetooth_get_connected():
        """Get list of connected Bluetooth devices."""
        if bluetooth_controller is None:
            return {"connected_devices": []}
        
        # Get all devices and filter for connected ones
        all_devices = bluetooth_controller.get_devices_list()
        print(f"[API] Total devices from get_devices_list: {len(all_devices)}")
        
        # Log all devices for debugging
        for i, d in enumerate(all_devices):
            print(f"[API] Device {i}: addr={d.get('address')} connected={d.get('connected')} rssi={d.get('rssi')}")
        
        connected = [d for d in all_devices if d.get("connected", False)]
        print(f"[API] After filtering for 'connected': {len(connected)} devices")
        print(f"[API] Filtered device addresses: {[d.get('address') for d in connected]}")
        
        active_addr = bluetooth_controller.get_connected_device()
        print(f"[API] Active device address: {active_addr}")
        
        # Mark which device is active
        for device in connected:
            is_active = device.get("address", "").upper() == (active_addr.upper() if active_addr else "")
            device["is_active"] = is_active
            print(f"[API]   Setting is_active for {device.get('address')}: {is_active}")
        
        # Debug logging
        print(f"[API] Connected devices endpoint - Found {len(connected)} connected devices out of {len(all_devices)} total")
        for d in connected:
            print(f"[API]   - {d.get('address')}: {d.get('name')} (connected={d.get('connected')}, active={d.get('is_active')})")
        
        return {
            "connected_devices": connected,
            "active_device": active_addr,
            "debug": {
                "total_devices": len(all_devices),
                "connected_count": len(connected),
                "all_device_addrs": [d.get('address') for d in all_devices]
            }
        }
    
    @app.post("/bluetooth/connect/{device_address}")
    async def bluetooth_connect(device_address: str):
        """Connect to a Bluetooth device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        try:
            success = bluetooth_controller.connect_device(device_address)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": f"Device {'connected' if success else 'connection failed'}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/bluetooth/set-active/{device_address}")
    async def bluetooth_set_active(device_address: str):
        """Set a device as the active input device (connects if needed)."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        try:
            device = bluetooth_controller.get_device(device_address)
            if not device:
                return {"status": "error", "message": f"Device {device_address} not found"}
            
            # Ensure device is connected (will skip for system-connected devices)
            print(f"[BT] Connecting to device for input: {device_address} ({device.name})")
            success = bluetooth_controller.connect_device(device_address)
            
            if not success:
                return {"status": "error", "message": "Failed to connect device"}
            
            # Device is now active
            print(f"[BT] Set active input device: {device_address} ({device.name})")
            
            return {
                "status": "success",
                "address": device_address,
                "name": device.name,
                "message": f"Connected and set as active input device"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/bluetooth/disconnect/{device_address}")
    async def bluetooth_disconnect(device_address: str):
        """Disconnect from a Bluetooth device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        try:
            success = bluetooth_controller.disconnect_device(device_address)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": f"Device {'disconnected' if success else 'disconnection failed'}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.post("/bluetooth/remove/{device_address}")
    async def bluetooth_remove(device_address: str):
        """Remove/forget a Bluetooth device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}
        
        try:
            success = bluetooth_controller.remove_device(device_address)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": f"Device {'removed' if success else 'removal failed'}"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.websocket("/control")
    async def websocket_control(websocket: WebSocket):
        global gimbal_position, gimbal_lock
        
        await websocket.accept()
        active_clients.add(websocket)
        bt_active_clients.add(websocket)
        
        try:
            await websocket.send_json({"type": "state", "data": processor.get_state()})
            
            # Send initial gimbal position
            with gimbal_lock:
                await websocket.send_json({"type": "gimbal_position", "data": gimbal_position.copy()})
            
            if bluetooth_controller is not None:
                await websocket.send_json({
                    "type": "bluetooth_state",
                    "data": {
                        "devices": bluetooth_controller.get_devices_list(),
                        "scanning": bluetooth_controller.scanning
                    }
                })
            
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
                    
                    if gimbal is None:
                        await websocket.send_json({"type": "error", "message": "Gimbal not initialized"})
                    elif motor_cmd == "gamepad_input":
                        # Gamepad analog input: continuous pan/tilt angles
                        pan = command.get("pan", 0)
                        tilt = command.get("tilt", 0)
                        print(f"🎮 Gamepad: Pan={pan:.1f}°, Tilt={tilt:.1f}°")
                        gimbal.move_to_angles(pan, tilt)

                        # Update global position and broadcast
                        with gimbal_lock:
                            gimbal_position["pan"] = pan
                            gimbal_position["tilt"] = tilt
                            pos_data = gimbal_position.copy()

                        await broadcast_gimbal_position(pos_data)
                    else:
                        # Button-based motor commands (discrete start/stop)
                        if motor_state == "start":
                            motor_moving[motor_cmd] = True
                            print(f"🎮 Motor: {motor_cmd} START")
                            
                            # Start continuous stepping in a background thread
                            def step_motor():
                                while motor_moving.get(motor_cmd, False):
                                    if motor_cmd == "motor_up":
                                        gimbal.tilt_step(1, steps=2)  # Small increments
                                    elif motor_cmd == "motor_down":
                                        gimbal.tilt_step(-1, steps=2)
                                    elif motor_cmd == "motor_left":
                                        gimbal.pan_step(-1, steps=2)
                                    elif motor_cmd == "motor_right":
                                        gimbal.pan_step(1, steps=2)

                                    broadcast_gimbal_position_threadsafe()
                                    time.sleep(0.1)  # Small delay between steps
                            
                            motor_thread = threading.Thread(target=step_motor, daemon=True)
                            motor_thread.start()
                        elif motor_state == "stop":
                            motor_moving[motor_cmd] = False
                            print(f"🎮 Motor: {motor_cmd} STOP")
                        elif motor_cmd == "motor_home":
                            print(f"🎮 Motor: HOMING")
                            gimbal.home()

                            await broadcast_gimbal_position(sync_gimbal_position())
                
                elif action == "bluetooth_start_scan":
                    if bluetooth_controller is None:
                        await websocket.send_json({"type": "error", "message": "Bluetooth not available"})
                    else:
                        try:
                            bluetooth_controller.start_scan()
                            await websocket.send_json({
                                "type": "bluetooth_scan_started",
                                "message": "Bluetooth scan started"
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to start Bluetooth scan: {str(e)}"
                            })
                
                elif action == "bluetooth_stop_scan":
                    if bluetooth_controller is not None:
                        bluetooth_controller.stop_scan()
                        await websocket.send_json({
                            "type": "bluetooth_scan_stopped",
                            "message": "Bluetooth scan stopped"
                        })
                
                elif action == "bluetooth_connect":
                    address = command.get("address")
                    if bluetooth_controller is None:
                        await websocket.send_json({"type": "error", "message": "Bluetooth not available"})
                    else:
                        try:
                            success = bluetooth_controller.connect_device(address)
                            await websocket.send_json({
                                "type": "bluetooth_connect_result",
                                "address": address,
                                "success": success
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to connect: {str(e)}"
                            })
                
                elif action == "bluetooth_disconnect":
                    address = command.get("address")
                    if bluetooth_controller is not None:
                        try:
                            bluetooth_controller.disconnect_device(address)
                            await websocket.send_json({
                                "type": "bluetooth_disconnect_result",
                                "address": address,
                                "success": True
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to disconnect: {str(e)}"
                            })
                
                elif action == "get_state":
                    await websocket.send_json({"type": "state", "data": processor.get_state()})
        
        except WebSocketDisconnect:
            active_clients.discard(websocket)
            bt_active_clients.discard(websocket)
    
    @app.get("/")
    async def root(request: Request):
        """Serve HTML UI."""
        template = jinja_env.get_template("index.html")
        html_content = template.render(request=request)
        return HTMLResponse(content=html_content)
    
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
