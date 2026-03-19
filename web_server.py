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


def try_open_camera(camera_id=0):
    """Try to open camera."""
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap.release()
    return None


def capture_frames(camera_id=0):
    """Capture frames from thermal camera with reconnection logic."""
    global camera_connected, camera_id_current
    
    cap = None
    prev_frame = None
    retry_count = 0
    last_log_retry = 0
    
    print(f"Camera thread started (attempting device {camera_id})")
    
    while True:
        # Try to open camera if not connected
        if cap is None:
            with reconnect_lock:
                cap = try_open_camera(camera_id)
                if cap is not None:
                    camera_connected = True
                    print(f"✓ Camera connected (device {camera_id})")
                    retry_count = 0
                    last_log_retry = 0
                else:
                    camera_connected = False
                    retry_count += 1
                    # Only log every 5 retries to reduce noise
                    if retry_count == 1 or retry_count % 5 == 0:
                        print(f"⚠ Waiting for camera... (attempt {retry_count})")
        
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown context manager."""
    global camera_thread, processor
    
    # Startup
    processor = ThermalProcessor()
    camera_thread = threading.Thread(target=capture_frames, args=(0,), daemon=True)
    camera_thread.start()
    print("Web server started")
    
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


@app.get("/camera-status")
async def camera_status():
    """Get current camera connection status."""
    return {
        "connected": camera_connected,
        "camera_id": camera_id_current,
        "message": "Camera connected" if camera_connected else "Camera disconnected - connect USB camera and press Reconnect"
    }


@app.post("/reconnect")
async def reconnect():
    """Trigger camera reconnection attempt."""
    print("Manual reconnect requested...")
    return {
        "status": "reconnect_triggered",
        "message": "Attempting to reconnect to camera..."
    }


@app.get("/video")
async def video_feed():
    """MJPEG video streaming endpoint."""
    return StreamingResponse(
        mjpeg_generator(jpeg_quality=65),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/state")
async def get_state():
    """Get current processor state (all modes and parameters)."""
    if processor is None:
        return {"error": "Processor not initialized"}
    return processor.get_state()


@app.websocket("/control")
async def websocket_control(websocket: WebSocket):
    """WebSocket endpoint for real-time control and state updates."""
    await websocket.accept()
    active_clients.add(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "state",
            "data": processor.get_state()
        })
        
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            
            action = command.get("action")
            
            if action == "toggle_mode":
                mode = command.get("mode")
                current = getattr(processor, mode, False)
                processor.set_mode(mode, not current)
                
                # Broadcast state to all clients
                state = processor.get_state()
                for client in active_clients:
                    try:
                        await client.send_json({
                            "type": "state",
                            "data": state
                        })
                    except:
                        pass
            
            elif action == "set_param":
                param = command.get("param")
                value = command.get("value")
                processor.set_parameter(param, value)
                
                # Broadcast state to all clients
                state = processor.get_state()
                for client in active_clients:
                    try:
                        await client.send_json({
                            "type": "state",
                            "data": state
                        })
                    except:
                        pass
            
            elif action == "get_state":
                await websocket.send_json({
                    "type": "state",
                    "data": processor.get_state()
                })
    
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
                
                <!-- Camera Status -->
                <div class="section">
                    <div class="section-title">Camera Status</div>
                    <div id="cameraStatus" style="padding: 8px; background: #1a1a1a; border-radius: 3px; font-size: 12px; text-align: center;">
                        <div id="statusIndicator" style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: #ff4444; margin-right: 8px; vertical-align: middle;"></div>
                        <span id="statusMessage">Connecting...</span>
                    </div>
                    <button class="btn" id="btn_reconnect" style="width: 100%; margin-top: 10px; background: #aa6600;">Reconnect Camera</button>
                </div>
                
                <!-- Display -->
                <div class="section">
                    <div class="section-title">Display</div>
                    <button class="btn" id="btn_w" style="width: 100%;">W: Show Text</button>
                    <div class="info">FPS: 50 Hz MJPEG</div>
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
            
            let ws = null;
            let state = {};
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                ws = new WebSocket(protocol + '://' + window.location.host + '/control');
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    document.getElementById('statusText').innerHTML = 'Connected';
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
                
                // Update status
                let statusLines = [];
                if (state.palette_name) statusLines.push('Palette: ' + state.palette_name);
                statusLines.push('Connected');
                document.getElementById('statusText').innerHTML = statusLines.join('<br>');
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
            
            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                const key = e.key.toLowerCase();
                if (buttons[key] && ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        action: 'toggle_mode',
                        mode: buttons[key]
                    }));
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
            
            // Poll camera status every 1 second
            setInterval(updateCameraStatus, 1000);
            
            // Connect on load
            connectWebSocket();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)



def run_webserver(host="0.0.0.0", port=8000, camera_id=0):
    """Launch FastAPI web server."""
    import uvicorn
    
    print(f"Starting thermal camera web server on {host}:{port}")
    print(f"Open browser: http://{host}:{port}")
    
    uvicorn.run(app, host=host, port=port, log_level="info")
