# Web Server API

Started with `make run` (or `uv run main.py --web`). Serves on `http://0.0.0.0:8000` by default.

---

## HTTP Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Browser UI (`index.html`) |
| `GET` | `/video` | MJPEG stream (~50 FPS, JPEG quality 65) |
| `GET` | `/state` | Full processor state as JSON |
| `GET` | `/camera-status` | `{connected, camera_id, message}` |
| `GET` | `/available-cameras` | List of working camera device IDs + current |
| `POST` | `/reconnect` | Trigger camera reconnect |
| `POST` | `/switch-camera/{id}` | Switch to a different camera device by index |
| `GET` | `/wifi/status` | Current WiFi mode (`client` / `ap`) + SSID details |
| `GET` | `/wifi/instructions/{mode}` | Confirmation-dialog copy for switching to `ap` or `client` |
| `POST` | `/wifi/mode/{mode}` | Switch WiFi to `ap` or `client` (background; disconnects current link) |
| `POST` | `/bluetooth/scan` | Start BlueZ discovery scan |
| `POST` | `/bluetooth/stop-scan` | Stop scan |
| `GET` | `/bluetooth/devices` | Known / discovered BlueZ devices |
| `GET` | `/bluetooth/connected` | Connected devices + active input address |
| `POST` | `/bluetooth/pair/{address}` | Pair + trust (Classic HID) |
| `POST` | `/bluetooth/connect/{address}` | Pair if needed, connect, set active |
| `POST` | `/bluetooth/set-active/{address}` | Mark connected pad as gimbal input |
| `POST` | `/bluetooth/disconnect/{address}` | BlueZ disconnect |
| `POST` | `/bluetooth/remove/{address}` | Disconnect and forget (unpair) |

### Pi Bluetooth gamepad vs browser gamepad

There are two independent gamepad paths:

1. **Pi Bluetooth Gamepad** (controls panel) — Classic Bluetooth HID via BlueZ on the robot. Scan / Pair / Connect / Remove use `bluetoothctl`. After Connect or Set Active, the Rust daemon reads `/dev/input/event*` (or Python falls back to pygame). This is the path for a pad paired to the Orange Pi.
2. **Browser Gamepad API** (header badge / settings axis presets) — a controller plugged into or paired with the machine running the browser. Axes are sent over WebSocket as `motor_command` / `gamepad_input`. Unrelated to the Pi Bluetooth panel.

Do not expect the Pi Bluetooth panel to manage a pad that is only connected to your laptop.

---

## WebSocket — `/control`

A persistent WebSocket for real-time bidirectional control. The server pushes status updates so the UI does not need to poll HTTP endpoints.

### Server → client push messages

| `type` | `data` | When sent |
|--------|--------|-----------|
| `state` | Processor state object | On connect, mode/param changes |
| `gimbal_position` | `{pan, tilt}` | Motor/gamepad/BT input |
| `camera_status` | `{connected, camera_id, message}` | On connect/disconnect/switch |
| `available_cameras` | `{available, current}` | When camera list or selection changes |
| `bluetooth` | `{status, data}` | Scan and device events |
| `bluetooth_state` | `{devices, scanning}` | On WebSocket connect |
| `bluetooth_connected` | `{connected_devices, active_device}` | On connect/disconnect/scan/active change |
| `bluetooth_pair_result` | `{address, success, message}` | After `bluetooth_pair` |
| `bluetooth_connect_result` | `{address, success, message}` | After `bluetooth_connect` |
| `bluetooth_disconnect_result` | `{address, success, message}` | After `bluetooth_disconnect` |
| `bluetooth_remove_result` | `{address, success, message}` | After `bluetooth_remove` |
| `wifi_status` | WiFi status object | On WebSocket connect |

Client → server Bluetooth actions: `bluetooth_start_scan`, `bluetooth_stop_scan`, `bluetooth_pair`, `bluetooth_connect`, `bluetooth_disconnect`, `bluetooth_remove` (each with `address` where applicable).

HTTP `GET /camera-status`, `/available-cameras`, `/bluetooth/connected`, and `/wifi/status` remain available for debugging. The UI also polls `/wifi/status` every few seconds.

### Toggle a mode

```json
{
  "action": "toggle_mode",
  "mode": "heat_seeker_mode"
}
```

Valid mode names match the processor attributes: `heat_seeker_mode`, `heat_cluster_mode`, `motion_mode`, `palette_mode`, `threshold_mode`, `yolo_mode`, `optical_flow_mode`, `optical_flow_masked_mode`, `isotherm_mode`, `denoise_mode`, `normalize_mode`, `enhance_mode`, `stabilize_mode`, `stabilize_super`, `upscale_mode`, `show_text`.

### Set a parameter

```json
{
  "action": "set_param",
  "param": "threshold_value",
  "value": 150
}
```

Settable parameters: `threshold_value`, `optical_flow_threshold`, `isotherm_min`, `isotherm_max`, `heat_seeker_max_boxes`, `heat_seeker_min_brightness`, `stabilize_strength`, `stabilize_smooth`.

### Request full state

```json
{
  "action": "get_state"
}
```

Returns the same JSON as `GET /state`.

### Gimbal control (stub — not yet fully wired)

```json
{
  "action": "motor_command",
  "pan_angle": 45.0,
  "tilt_angle": 10.0
}
```

---

## Camera Thread Behavior

- Auto-scans `/dev/video0`–`4` on startup
- Adaptive reconnect backoff: 0.5s → 1s → 2s
- Frame queue size: 2 (drops oldest frame on overflow to prevent lag)
