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
from queue import Queue
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress OpenCV warnings (camera not found messages)
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
cv2.setLogLevel(0)

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import RedirectResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, FileSystemLoader

from cookie_finder.camera.processor import ThermalProcessor
from cookie_finder.gimbal.rust_client import RustGimbalClient
from cookie_finder.bluetooth.controller import BluetoothController
from cookie_finder.wifi import AP_GATEWAY, get_switch_instructions, get_wifi_status, set_wifi_mode
from cookie_finder.poweroff import request_poweroff
from cookie_finder import cloudflare_tunnel, software_update

MDNS_HOST = "cookie-finder.local"


def _ipv4_addresses() -> list[dict[str, str]]:
    """Return non-loopback IPv4 addresses as {interface, ip} dicts."""
    import re
    import socket
    import subprocess

    addresses: list[dict[str, str]] = []
    try:
        result = subprocess.run(
            ["ip", "-4", "-o", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        for line in (result.stdout or "").splitlines():
            # "2: wlan0    inet 192.168.1.5/24 brd ..."
            match = re.match(
                r"^\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+)",
                line,
            )
            if not match:
                continue
            iface, ip = match.group(1), match.group(2)
            if iface == "lo" or ip.startswith("127.") or ip.startswith("169.254."):
                continue
            addresses.append({"interface": iface.split("@", 1)[0], "ip": ip})
        if addresses:
            return addresses
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    # Fallback for macOS / environments without `ip`
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127."):
                addresses.append({"interface": "primary", "ip": ip})
    except OSError:
        pass
    return addresses


def get_network_info() -> dict:
    """LAN IP addresses, mDNS name, and preferred URL for QR / connect UI."""
    wifi = get_wifi_status()
    addresses = _ipv4_addresses()
    mdns_url = f"http://{MDNS_HOST}/"

    preferred_ip = None
    if wifi.get("mode") == "ap":
        preferred_ip = wifi.get("ap_gateway") or AP_GATEWAY
    else:
        # Prefer WiFi, then Ethernet, then first listed address.
        for prefer in ("wlan", "eth", "en"):
            for entry in addresses:
                if entry["interface"].startswith(prefer):
                    preferred_ip = entry["ip"]
                    break
            if preferred_ip:
                break
        if not preferred_ip and addresses:
            preferred_ip = addresses[0]["ip"]

    if wifi.get("mode") == "ap":
        url = wifi.get("ap_url") or f"http://{AP_GATEWAY}/"
    elif preferred_ip:
        url = f"http://{preferred_ip}/"
    else:
        url = mdns_url

    cf = get_cloudflare_tunnel_status_payload()
    return {
        "ip": preferred_ip,
        "addresses": addresses,
        "mdns": MDNS_HOST,
        "mdns_url": mdns_url,
        "url": url,
        "wifi_mode": wifi.get("mode"),
        "cloudflare_running": bool(cf.get("running")),
        "cloudflare_url": cf.get("url"),
    }


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


# Captive-portal probe paths used by Android / Apple / Windows / Firefox.
# DNS hijack in AP mode sends these to us; we redirect to the web app home.
_CAPTIVE_PROBE_PATHS = frozenset(
    {
        "/generate_204",
        "/gen_204",
        "/hotspot-detect.html",
        "/library/test/success.html",
        "/connecttest.txt",
        "/ncsi.txt",
        "/success.txt",
        "/canonical.html",
        "/redirect",
        "/kindle-wifi/wifiredirect.html",
        "/kindle-wifi/wifistub.html",
    }
)

_CAPTIVE_HOME = os.environ.get(
    "COOKIE_FINDER_CAPTIVE_URL", f"http://{AP_GATEWAY}/"
)
_TLS_DIR = Path(
    os.environ.get("COOKIE_FINDER_TLS_DIR", "/var/lib/cookie-finder/tls")
)


# Verbose console chatter (MJPEG wait loops, etc.)
_VERBOSE = _env_flag("COOKIE_FINDER_VERBOSE")

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
gimbal = None  # RustGimbalClient (requires cookie-finder-ctl daemon)
gimbal_poller_started = False
motor_moving = {}  # Track which motors are moving: {command: True/False}
bluetooth_controller = None  # BluetoothController instance
gimbal_position = {"pan": 0.0, "tilt": 0.0}  # Current gimbal angles
gimbal_lock = threading.Lock()  # Thread-safe access to gimbal_position
bt_device_connected = False  # Track if BT device is connected for input
control_loop = None  # Event loop used for cross-thread WebSocket broadcasts
_last_camera_broadcast = {"connected": None, "camera_id": None}
_last_cameras_broadcast = None  # (tuple(available), current_id)
_last_bt_connected_broadcast = None  # hashable snapshot for dedupe


def _systemd_unit_active(unit: str) -> bool | None:
    """Return True/False if systemctl reports the unit; None if unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["systemctl", "is-active", "--quiet", unit],
            capture_output=True,
            timeout=1.5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _systemd_unit_exists(unit: str) -> bool | None:
    """Return True if the unit file is loadable; None if systemctl is unavailable."""
    import subprocess

    try:
        result = subprocess.run(
            ["systemctl", "cat", unit],
            capture_output=True,
            timeout=1.5,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def get_cloudflare_tunnel_status_payload() -> dict:
    """Status of the Cloudflare Tunnel connector (cloudflared systemd unit)."""
    return cloudflare_tunnel.status()


def ensure_gimbal_connected() -> bool:
    """Ping existing client or reconnect to the Rust daemon. Returns True if usable."""
    global gimbal, gimbal_poller_started

    if gimbal is not None:
        if gimbal.ping():
            return True
        print("⚠ Rust gimbal daemon lost — clearing client")
        gimbal = None

    client = RustGimbalClient.connect(
        max_pan=150.0, max_tilt=60.0, timeout=0.5, quiet=True
    )
    if client is None:
        return False

    try:
        client.set_speed(pan_hz=500, tilt_hz=500)
    except Exception as e:
        print(f"⚠ Rust gimbal connected but set_speed failed: {e}")
        return False

    gimbal = client
    print("✓ Gimbal reconnected via Rust daemon")
    if not gimbal_poller_started:
        threading.Thread(target=poll_gimbal_position, daemon=True).start()
        gimbal_poller_started = True
    return True


def get_gimbal_status_payload() -> dict:
    running = ensure_gimbal_connected()
    socket_path = os.environ.get(
        "COOKIE_FINDER_SOCKET",
        getattr(gimbal, "_socket_path", None) or "/tmp/cookie-finder.sock",
    )
    return {
        "running": running,
        "socket": socket_path,
        "service_active": _systemd_unit_active("cookie-finder.service"),
    }


def get_camera_status_payload() -> dict:
    return {
        "connected": camera_connected,
        "camera_id": camera_id_current,
        "message": "Camera connected" if camera_connected else "Camera disconnected",
    }


def get_available_cameras_payload() -> dict:
    return {
        "available": list(available_cameras),
        "current": camera_id_current,
    }


def get_bluetooth_connected_payload() -> dict | None:
    if bluetooth_controller is None:
        return {"connected_devices": [], "active_device": None}

    all_devices = bluetooth_controller.get_devices_list()
    connected = [d for d in all_devices if d.get("connected", False)]
    active_addr = bluetooth_controller.get_connected_device()
    for device in connected:
        device["is_active"] = device.get("address", "").upper() == (
            active_addr.upper() if active_addr else ""
        )
    return {
        "connected_devices": connected,
        "active_device": active_addr,
    }


async def broadcast_to_clients(message: dict) -> None:
    """Broadcast a JSON message to all connected WebSocket clients."""
    disconnected_clients = []
    for client in list(active_clients):
        try:
            await client.send_json(message)
        except Exception:
            disconnected_clients.append(client)

    for client in disconnected_clients:
        active_clients.discard(client)


def broadcast_to_clients_threadsafe(message: dict) -> None:
    if control_loop is None:
        return
    try:
        asyncio.run_coroutine_threadsafe(broadcast_to_clients(message), control_loop)
    except Exception:
        pass


def broadcast_camera_status_threadsafe(force: bool = False) -> None:
    global _last_camera_broadcast
    payload = get_camera_status_payload()
    if (
        not force
        and payload["connected"] == _last_camera_broadcast["connected"]
        and payload["camera_id"] == _last_camera_broadcast["camera_id"]
    ):
        return
    _last_camera_broadcast = {
        "connected": payload["connected"],
        "camera_id": payload["camera_id"],
    }
    broadcast_to_clients_threadsafe({"type": "camera_status", "data": payload})


def broadcast_available_cameras_threadsafe(force: bool = False) -> None:
    global _last_cameras_broadcast
    payload = get_available_cameras_payload()
    key = (tuple(payload["available"]), payload["current"])
    if not force and key == _last_cameras_broadcast:
        return
    _last_cameras_broadcast = key
    broadcast_to_clients_threadsafe({"type": "available_cameras", "data": payload})


def broadcast_bluetooth_connected_threadsafe(force: bool = False) -> None:
    global _last_bt_connected_broadcast
    payload = get_bluetooth_connected_payload()
    if payload is None:
        return
    key = (
        tuple(d.get("address") for d in payload["connected_devices"]),
        payload.get("active_device"),
        tuple(
            d.get("is_active") for d in payload["connected_devices"]
        ),
    )
    if not force and key == _last_bt_connected_broadcast:
        return
    _last_bt_connected_broadcast = key
    broadcast_to_clients_threadsafe({"type": "bluetooth_connected", "data": payload})


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

    await broadcast_to_clients({"type": "gimbal_position", "data": pos_data})


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
    broadcast_available_cameras_threadsafe(force=True)

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
                    working_cameras.append(test_id)
                    available_cameras = working_cameras
                    broadcast_available_cameras_threadsafe(force=True)
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
                    broadcast_camera_status_threadsafe()
                    broadcast_available_cameras_threadsafe()
                else:
                    camera_connected = False
                    camera_id_current = camera_id
                    broadcast_camera_status_threadsafe()
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
                camera_connected = False
                broadcast_camera_status_threadsafe()
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
            camera_connected = False
            broadcast_camera_status_threadsafe()
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
        if _VERBOSE and frame_count % 50 == 0 and not camera_connected:
            print("  (⏳ waiting for camera reconnection...)")
        
        time.sleep(0.02)


def poll_gimbal_position():
    """Poll hardware position from the Rust daemon and broadcast to clients."""
    global gimbal, gimbal_position, gimbal_lock
    while True:
        try:
            if gimbal is not None:
                pan, tilt = gimbal.get_position()
                with gimbal_lock:
                    gimbal_position["pan"] = pan
                    gimbal_position["tilt"] = tilt
                    pos = gimbal_position.copy()
                broadcast_gimbal_position_threadsafe(pos)
        except Exception as e:
            print(f"[Gimbal] Position poll error: {e}")
        time.sleep(0.05)


def poll_bluetooth_controller():
    """
    Background thread: push the UI's active BlueZ pad to the Rust daemon via
    set_active_input (hot-swappable; no daemon restart). The daemon owns
    /dev/input/event* and drives the motors.
    """
    global gimbal, bluetooth_controller
    global bt_device_connected

    print("[BT] Input polling thread started")
    last_rust_input_key = None

    while True:
        try:
            active_addr = (
                bluetooth_controller.get_connected_device()
                if bluetooth_controller
                else None
            )
            connected = bool(active_addr)
            bt_device_connected = connected

            if gimbal is None:
                time.sleep(0.5)
                continue

            active_name = None
            if connected and bluetooth_controller is not None:
                # Use in-memory cache — do not call bluetoothctl every tick.
                cached = bluetooth_controller.devices.get(active_addr) or (
                    bluetooth_controller.devices.get(active_addr.upper())
                    if active_addr
                    else None
                )
                if cached:
                    active_name = cached.name
            rust_key = (active_addr, active_name) if connected else (None, None)
            if rust_key != last_rust_input_key:
                if connected:
                    print(
                        f"[BT] Rust active input → {active_name or 'pad'} ({active_addr})"
                    )
                    gimbal.set_active_input(
                        True, address=active_addr, name=active_name
                    )
                else:
                    print("[BT] Rust active input cleared")
                    gimbal.set_active_input(False)
                last_rust_input_key = rust_key
            time.sleep(0.05)

        except Exception as e:
            print(f"[BT] Polling error: {e}")
            time.sleep(0.5)


def create_app(camera_id=None):
    """Create and configure the FastAPI application."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global camera_thread, processor, gimbal, bluetooth_controller, control_loop
        global gimbal_poller_started
        
        # Startup
        print(f"Initializing processor...")
        control_loop = asyncio.get_running_loop()
        processor = ThermalProcessor()
        
        # Gimbal requires the Rust cookie-finder-ctl daemon (Unix socket IPC).
        try:
            print(f"Initializing gimbal...")
            rust_gimbal = RustGimbalClient.connect(max_pan=150.0, max_tilt=60.0)
            if rust_gimbal is not None:
                rust_gimbal.set_speed(pan_hz=500, tilt_hz=500)
                gimbal = rust_gimbal
                print(f"✓ Gimbal via Rust daemon")
            else:
                print(
                    "⚠ Gimbal unavailable — start the Rust daemon "
                    "(make on-the-pi-rust-daemon)"
                )
                gimbal = None
        except Exception as e:
            print(f"⚠ Gimbal initialization failed: {e}")
            gimbal = None
        
        # Initialize Bluetooth controller (BlueZ pair/connect; input is Rust)
        try:
            print(f"Initializing Bluetooth controller...")
            bluetooth_controller = BluetoothController()
            
            def bt_status_callback(update):
                """Broadcast Bluetooth status to all connected WebSocket clients."""
                broadcast_to_clients_threadsafe({"type": "bluetooth", "data": update})
                status = update.get("status")
                if status in (
                    "scan_complete",
                    "scan_stopped",
                    "device_connected",
                    "device_disconnected",
                    "device_removed",
                    "device_paired",
                ):
                    broadcast_bluetooth_connected_threadsafe()

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

        if gimbal is not None:
            threading.Thread(target=poll_gimbal_position, daemon=True).start()
            gimbal_poller_started = True
        
        print("✓ Web server started")
        
        yield
        
        # Shutdown
        if gimbal is not None:
            gimbal.cleanup()
        if bluetooth_controller is not None and bluetooth_controller.scanning:
            bluetooth_controller.stop_scan()
        print("Web server shutting down")
    
    app = FastAPI(title="Cookie Finder", lifespan=lifespan)
    
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

    @app.middleware("http")
    async def captive_portal_middleware(request: Request, call_next):
        """Redirect OS captive-portal probes to the web app home page."""
        if request.method in ("GET", "HEAD"):
            path = request.url.path.rstrip("/") or "/"
            # Match with and without trailing slash (probe set has no slash).
            probe = path if path != "/" else "/"
            if probe in _CAPTIVE_PROBE_PATHS or request.url.path in _CAPTIVE_PROBE_PATHS:
                return RedirectResponse(url=_CAPTIVE_HOME, status_code=302)
        return await call_next(request)
    
    # Add all the routes
    @app.get("/camera-status")
    def camera_status():
        return {
            "connected": camera_connected,
            "camera_id": camera_id_current,
            "message": "Camera connected" if camera_connected else "Camera disconnected"
        }

    @app.get("/gimbal/status")
    def gimbal_status():
        """Rust cookie-finder-ctl daemon reachability (socket ping + optional systemd)."""
        return get_gimbal_status_payload()

    @app.get("/cloudflare/status")
    def cloudflare_status():
        """Cloudflare Tunnel (cloudflared.service) install + running state."""
        return get_cloudflare_tunnel_status_payload()

    @app.post("/cloudflare/start")
    def cloudflare_start():
        """Enable + start cloudflared.service (passwordless sudo via init target)."""
        result = cloudflare_tunnel.start()
        if result.get("status") != "ok":
            return JSONResponse(result, status_code=500)
        return result

    @app.post("/cloudflare/stop")
    def cloudflare_stop():
        """Disable + stop cloudflared.service (stays off across reboot)."""
        result = cloudflare_tunnel.stop()
        if result.get("status") != "ok":
            return JSONResponse(result, status_code=500)
        return result

    @app.post("/reconnect")
    def reconnect():
        """Force the capture thread to release and reopen the current camera."""
        global camera_switch_id, camera_switch_event
        print("Manual reconnect requested...")
        camera_switch_id = camera_id_current
        camera_switch_event.set()
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
    def get_available_cameras():
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

    @app.get("/wifi/status")
    def wifi_status():
        """Return current WiFi client/AP mode details."""
        return get_wifi_status()

    @app.get("/network/info")
    def network_info():
        """Return LAN IP, mDNS hostname, and preferred connect URL."""
        return get_network_info()

    @app.get("/wifi/instructions/{mode}")
    def wifi_instructions(mode: str):
        """Return confirmation-dialog copy for switching to ap or client."""
        return get_switch_instructions(mode)

    @app.post("/wifi/mode/{mode}")
    def wifi_set_mode(mode: str):
        """
        Switch WiFi to ap or client mode.

        Returns immediately, then performs the radio change in the background
        so the browser can show instructions before the link drops.
        """
        return set_wifi_mode(mode)

    @app.post("/system/poweroff")
    def system_poweroff():
        """
        Graceful power-off: LED chirp, then halt.

        Returns immediately so the browser can show a shutting-down message
        before the board goes down.
        """
        return request_poweroff()

    @app.get("/system/software")
    def system_software_status():
        """Compare local checkout to origin/main (git fetch + status)."""
        return software_update.status(fetch=True)

    @app.post("/system/software/update", status_code=202)
    def system_software_update():
        """
        Schedule a oneshot unit that fast-forwards origin/main, runs uv sync,
        and restarts cookie-finder-web. Returns immediately so the UI can poll.
        """
        try:
            return software_update.request_update()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    def bluetooth_get_devices():
        """Get list of discovered Bluetooth devices."""
        if bluetooth_controller is None:
            return {"devices": [], "scanning": False}
        
        return {
            "devices": bluetooth_controller.get_devices_list(),
            "scanning": bluetooth_controller.scanning
        }
    
    @app.get("/bluetooth/connected")
    def bluetooth_get_connected():
        """Get list of connected Bluetooth devices."""
        if bluetooth_controller is None:
            return {"connected_devices": []}
        
        # Get all devices and filter for connected ones
        all_devices = bluetooth_controller.get_devices_list()
        # print(f"[API] Total devices from get_devices_list: {len(all_devices)}")
        
        # Log all devices for debugging
        # for i, d in enumerate(all_devices):
        #     print(f"[API] Device {i}: addr={d.get('address')} connected={d.get('connected')} rssi={d.get('rssi')}")
        
        connected = [d for d in all_devices if d.get("connected", False)]
        # print(f"[API] After filtering for 'connected': {len(connected)} devices")
        # print(f"[API] Filtered device addresses: {[d.get('address') for d in connected]}")
        
        active_addr = bluetooth_controller.get_connected_device()
        # print(f"[API] Active device address: {active_addr}")
        
        # Mark which device is active
        for device in connected:
            is_active = device.get("address", "").upper() == (active_addr.upper() if active_addr else "")
            device["is_active"] = is_active
            # print(f"[API]   Setting is_active for {device.get('address')}: {is_active}")
        
        # Debug logging
        # print(f"[API] Connected devices endpoint - Found {len(connected)} connected devices out of {len(all_devices)} total")
        # for d in connected:
        #     print(f"[API]   - {d.get('address')}: {d.get('name')} (connected={d.get('connected')}, active={d.get('is_active')})")
        
        return {
            "connected_devices": connected,
            "active_device": active_addr,
            "debug": {
                "total_devices": len(all_devices),
                "connected_count": len(connected),
                "all_device_addrs": [d.get('address') for d in all_devices]
            }
        }
    
    @app.post("/bluetooth/pair/{device_address}")
    def bluetooth_pair(device_address: str):
        """Pair and trust a BlueZ HID device (does not require connect)."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}

        try:
            success = bluetooth_controller.pair_device(device_address)
            err = bluetooth_controller.get_last_error()
            if success:
                broadcast_bluetooth_connected_threadsafe(force=True)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": "Device paired" if success else (err or "Pair failed"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/bluetooth/connect/{device_address}")
    def bluetooth_connect(device_address: str):
        """Pair if needed, connect via BlueZ, and set as active input device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}

        try:
            success = bluetooth_controller.connect_device(device_address)
            err = bluetooth_controller.get_last_error()
            if success:
                broadcast_bluetooth_connected_threadsafe(force=True)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": (
                    "Device connected"
                    if success
                    else (err or "Connection failed")
                ),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/bluetooth/set-active/{device_address}")
    def bluetooth_set_active(device_address: str):
        """Set a device as the active input device (connects if needed)."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}

        try:
            device = bluetooth_controller.get_device(device_address)
            if not device:
                return {
                    "status": "error",
                    "message": f"Device {device_address} not found",
                }

            print(
                f"[BT] Setting active input device: "
                f"{device_address} ({device.name})"
            )
            success = bluetooth_controller.set_active_device(device_address)
            err = bluetooth_controller.get_last_error()

            if not success:
                return {
                    "status": "error",
                    "message": err or "Failed to set active device",
                }

            broadcast_bluetooth_connected_threadsafe(force=True)
            return {
                "status": "success",
                "address": device_address,
                "name": device.name,
                "message": "Set as active input device",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/bluetooth/disconnect/{device_address}")
    def bluetooth_disconnect(device_address: str):
        """Disconnect a BlueZ device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}

        try:
            success = bluetooth_controller.disconnect_device(device_address)
            err = bluetooth_controller.get_last_error()
            if success:
                broadcast_bluetooth_connected_threadsafe(force=True)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": (
                    "Device disconnected"
                    if success
                    else (err or "Disconnection failed")
                ),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    @app.post("/bluetooth/remove/{device_address}")
    def bluetooth_remove(device_address: str):
        """Disconnect and forget a BlueZ device."""
        if bluetooth_controller is None:
            return {"status": "error", "message": "Bluetooth not available"}

        try:
            success = bluetooth_controller.remove_device(device_address)
            err = bluetooth_controller.get_last_error()
            if success:
                broadcast_bluetooth_connected_threadsafe(force=True)
            return {
                "status": "success" if success else "failed",
                "address": device_address,
                "message": (
                    "Device removed" if success else (err or "Removal failed")
                ),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    @app.websocket("/control")
    async def websocket_control(websocket: WebSocket):
        global gimbal_position, gimbal_lock
        
        await websocket.accept()
        active_clients.add(websocket)
        
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
                bt_payload = get_bluetooth_connected_payload()
                if bt_payload is not None:
                    await websocket.send_json({
                        "type": "bluetooth_connected",
                        "data": bt_payload,
                    })

            await websocket.send_json({
                "type": "camera_status",
                "data": get_camera_status_payload(),
            })
            await websocket.send_json({
                "type": "available_cameras",
                "data": get_available_cameras_payload(),
            })
            await websocket.send_json({
                "type": "wifi_status",
                "data": get_wifi_status(),
            })
            await websocket.send_json({
                "type": "cloudflare_status",
                "data": get_cloudflare_tunnel_status_payload(),
            })

            while True:
                data = await websocket.receive_text()
                try:
                    command = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON command",
                    })
                    continue
                if not isinstance(command, dict):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Command must be a JSON object",
                    })
                    continue
                action = command.get("action")
                
                if action == "toggle_mode":
                    mode = command.get("mode")
                    try:
                        current = getattr(processor, mode, False)
                        processor.set_mode(mode, not current)
                    except (TypeError, ValueError) as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        continue
                    state = processor.get_state()
                    for client in active_clients:
                        try:
                            await client.send_json({"type": "state", "data": state})
                        except Exception:
                            pass
                
                elif action == "set_param":
                    param = command.get("param")
                    value = command.get("value")
                    try:
                        processor.set_parameter(param, value)
                    except (TypeError, ValueError) as e:
                        await websocket.send_json({"type": "error", "message": str(e)})
                        continue
                    state = processor.get_state()
                    for client in active_clients:
                        try:
                            await client.send_json({"type": "state", "data": state})
                        except Exception:
                            pass

                elif action == "set_motor_speed":
                    if gimbal is None:
                        await websocket.send_json({"type": "error", "message": "Gimbal not initialized"})
                    else:
                        pan_hz = float(command.get("pan_hz", 500))
                        tilt_hz = float(command.get("tilt_hz", 500))
                        gimbal.set_speed(pan_hz=pan_hz, tilt_hz=tilt_hz)
                
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
                    elif motor_cmd == "motor_home":
                        # Soft home to UI zero (pan/tilt). Falls back to limit-switch home.
                        # Ignores start/stop state (one-shot).
                        pan = command.get("pan")
                        tilt = command.get("tilt")
                        if pan is not None and tilt is not None:
                            pan = float(pan)
                            tilt = float(tilt)
                            print(f"🎮 Motor: SOFT HOME → Pan={pan:.1f}°, Tilt={tilt:.1f}°")
                            gimbal.move_to_angles(pan, tilt)
                            with gimbal_lock:
                                gimbal_position["pan"] = pan
                                gimbal_position["tilt"] = tilt
                                pos_data = gimbal_position.copy()
                            await broadcast_gimbal_position(pos_data)
                        else:
                            print(f"🎮 Motor: HOMING (limit switches)")
                            gimbal.home()
                            await broadcast_gimbal_position(sync_gimbal_position())
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
                            
                            motor_thread = threading.Thread(target=step_motor, daemon=True)
                            motor_thread.start()
                        elif motor_state == "stop":
                            motor_moving[motor_cmd] = False
                            print(f"🎮 Motor: {motor_cmd} STOP")
                
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
                
                elif action == "bluetooth_pair":
                    address = command.get("address")
                    if bluetooth_controller is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Bluetooth not available",
                        })
                    else:
                        try:
                            success = await asyncio.to_thread(
                                bluetooth_controller.pair_device, address
                            )
                            err = bluetooth_controller.get_last_error()
                            await websocket.send_json({
                                "type": "bluetooth_pair_result",
                                "address": address,
                                "success": success,
                                "message": (
                                    "Paired"
                                    if success
                                    else (err or "Pair failed")
                                ),
                            })
                            await websocket.send_json({
                                "type": "bluetooth_state",
                                "data": {
                                    "devices": bluetooth_controller.get_devices_list(),
                                    "scanning": bluetooth_controller.scanning,
                                },
                            })
                            if success:
                                broadcast_bluetooth_connected_threadsafe(force=True)
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to pair: {str(e)}",
                            })

                elif action == "bluetooth_connect":
                    address = command.get("address")
                    if bluetooth_controller is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Bluetooth not available",
                        })
                    else:
                        try:
                            success = await asyncio.to_thread(
                                bluetooth_controller.connect_device, address
                            )
                            err = bluetooth_controller.get_last_error()
                            await websocket.send_json({
                                "type": "bluetooth_connect_result",
                                "address": address,
                                "success": success,
                                "message": (
                                    "Connected"
                                    if success
                                    else (err or "Connection failed")
                                ),
                            })
                            await websocket.send_json({
                                "type": "bluetooth_state",
                                "data": {
                                    "devices": bluetooth_controller.get_devices_list(),
                                    "scanning": bluetooth_controller.scanning,
                                },
                            })
                            if success:
                                broadcast_bluetooth_connected_threadsafe(force=True)
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to connect: {str(e)}",
                            })

                elif action == "bluetooth_disconnect":
                    address = command.get("address")
                    if bluetooth_controller is not None:
                        try:
                            success = await asyncio.to_thread(
                                bluetooth_controller.disconnect_device, address
                            )
                            err = bluetooth_controller.get_last_error()
                            await websocket.send_json({
                                "type": "bluetooth_disconnect_result",
                                "address": address,
                                "success": success,
                                "message": (
                                    "Disconnected"
                                    if success
                                    else (err or "Disconnect failed")
                                ),
                            })
                            await websocket.send_json({
                                "type": "bluetooth_state",
                                "data": {
                                    "devices": bluetooth_controller.get_devices_list(),
                                    "scanning": bluetooth_controller.scanning,
                                },
                            })
                            broadcast_bluetooth_connected_threadsafe(force=True)
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to disconnect: {str(e)}",
                            })

                elif action == "bluetooth_remove":
                    address = command.get("address")
                    if bluetooth_controller is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Bluetooth not available",
                        })
                    else:
                        try:
                            success = await asyncio.to_thread(
                                bluetooth_controller.remove_device, address
                            )
                            err = bluetooth_controller.get_last_error()
                            await websocket.send_json({
                                "type": "bluetooth_remove_result",
                                "address": address,
                                "success": success,
                                "message": (
                                    "Removed"
                                    if success
                                    else (err or "Remove failed")
                                ),
                            })
                            if success:
                                broadcast_bluetooth_connected_threadsafe(force=True)
                                await websocket.send_json({
                                    "type": "bluetooth_state",
                                    "data": {
                                        "devices": bluetooth_controller.get_devices_list(),
                                        "scanning": bluetooth_controller.scanning,
                                    },
                                })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Failed to remove: {str(e)}",
                            })

                elif action == "get_state":
                    await websocket.send_json({"type": "state", "data": processor.get_state()})
        
        except WebSocketDisconnect:
            active_clients.discard(websocket)
    
    @app.get("/")
    async def root(request: Request):
        """Serve HTML UI."""
        template = jinja_env.get_template("index.html")
        html_content = template.render(request=request)
        return HTMLResponse(content=html_content)
    
    return app


def _ensure_tls_certs(certfile: Path, keyfile: Path) -> bool:
    """Create a self-signed cert for HTTPS if missing. Returns True on success."""
    if certfile.is_file() and keyfile.is_file():
        return True
    import subprocess

    try:
        certfile.parent.mkdir(parents=True, exist_ok=True)
        # openssl config for SAN (cookie-finder.local + AP gateway)
        conf = certfile.parent / "openssl-san.cnf"
        conf.write_text(
            "\n".join(
                [
                    "[req]",
                    "default_bits = 2048",
                    "prompt = no",
                    "default_md = sha256",
                    "distinguished_name = dn",
                    "x509_extensions = v3_req",
                    "[dn]",
                    "CN = cookie-finder.local",
                    "[v3_req]",
                    "subjectAltName = @alt_names",
                    "basicConstraints = CA:FALSE",
                    "keyUsage = digitalSignature, keyEncipherment",
                    "extendedKeyUsage = serverAuth",
                    "[alt_names]",
                    "DNS.1 = cookie-finder.local",
                    "DNS.2 = localhost",
                    f"IP.1 = {AP_GATEWAY}",
                    "IP.2 = 127.0.0.1",
                    "",
                ]
            )
        )
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-nodes",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(keyfile),
                "-out",
                str(certfile),
                "-days",
                "3650",
                "-config",
                str(conf),
            ],
            check=True,
            capture_output=True,
        )
        print(f"Generated self-signed TLS cert: {certfile}")
        return True
    except (OSError, subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"WARNING: could not create TLS cert ({exc}); HTTPS disabled")
        return False


def run_webserver(
    host="0.0.0.0",
    port=80,
    https_port=443,
    camera_id=None,
    ssl_certfile=None,
    ssl_keyfile=None,
):
    """Launch FastAPI on HTTP (and optionally HTTPS).

    Defaults: port 80 and HTTPS 443. Set https_port=0 (or None) to disable TLS.
    """
    import asyncio
    import uvicorn
    from uvicorn import Config, Server

    camera_desc = (
        f"/dev/video{camera_id}" if camera_id is not None else "auto-detect (none found)"
    )
    print(f"Creating FastAPI app (camera: {camera_desc})...")
    app = create_app(camera_id)

    access_log = _env_flag("COOKIE_FINDER_ACCESS_LOG")
    log_level = "info" if access_log else "warning"

    certfile = Path(ssl_certfile) if ssl_certfile else _TLS_DIR / "cert.pem"
    keyfile = Path(ssl_keyfile) if ssl_keyfile else _TLS_DIR / "key.pem"
    use_https = bool(https_port) and https_port > 0

    if use_https and not _ensure_tls_certs(certfile, keyfile):
        use_https = False

    print(f"Starting web server on http://{host}:{port}")
    if use_https:
        print(f"Starting web server on https://{host}:{https_port}")
    print(f"Open browser: http://{host}:{port}" + ("" if port == 80 else ""))

    async def _serve() -> None:
        servers: list[Server] = []
        http_cfg = Config(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=access_log,
        )
        servers.append(Server(http_cfg))

        if use_https:
            # Lifespan already runs on the HTTP server; skip on HTTPS.
            https_cfg = Config(
                app,
                host=host,
                port=https_port,
                log_level=log_level,
                access_log=access_log,
                ssl_certfile=str(certfile),
                ssl_keyfile=str(keyfile),
                lifespan="off",
            )
            servers.append(Server(https_cfg))

        await asyncio.gather(*(s.serve() for s in servers))

    try:
        asyncio.run(_serve())
    except OSError as exc:
        if getattr(exc, "errno", None) in (1, 13) or "Permission" in str(exc):
            print(
                f"ERROR: cannot bind port {port}"
                + (f"/{https_port}" if use_https else "")
                + " — need root or CAP_NET_BIND_SERVICE "
                "(use: make on-the-pi-web-daemon)"
            )
        raise
