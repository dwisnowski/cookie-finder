# Web Server API

Started with `make run` (or `uv run main.py --web`). Serves on **HTTP :80** and **HTTPS :443** by default (self-signed cert). Use `--https-port 0` to disable TLS, or `--port 8000` for unprivileged local dev.

---

## HTTP Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Browser UI (`index.html`) |
| `GET` | `/generate_204`, `/hotspot-detect.html`, … | Captive-portal probes → redirect to `http://192.168.12.1/` |
| `GET` | `/video` | MJPEG stream (~50 FPS, JPEG quality 65) |
| `GET` | `/state` | Full processor state as JSON |
| `GET` | `/camera-status` | `{connected, camera_id, message}` |
| `GET` | `/available-cameras` | List of working camera device IDs + current |
| `GET` | `/gimbal/status` | Rust daemon `{running, socket, service_active}` |
| `POST` | `/reconnect` | Trigger camera reconnect |
| `POST` | `/switch-camera/{id}` | Switch to a different camera device by index |
| `GET` | `/wifi/status` | Current WiFi mode (`client` / `ap`) + SSID details |
| `GET` | `/wifi/instructions/{mode}` | Confirmation-dialog copy for switching to `ap` or `client` |
| `POST` | `/wifi/mode/{mode}` | Switch WiFi to `ap` or `client` (background; disconnects current link) |
| `POST` | `/system/poweroff` | Graceful halt (LED chirp, then `systemctl poweroff`; returns immediately) |
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

1. **Pi Bluetooth Gamepad** (controls panel) — Classic Bluetooth HID via BlueZ on the robot. Scan / Pair / Connect / Remove use `bluetoothctl`. After Connect or Set Active, the Rust daemon reads `/dev/input/event*` and drives the motors. This is the path for a pad paired to the Orange Pi.
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

HTTP `GET /camera-status`, `/available-cameras`, `/bluetooth/connected`, `/gimbal/status`, and `/wifi/status` remain available for debugging. The UI polls `/wifi/status` and `/gimbal/status` every few seconds.

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

Settable parameters: `palette_idx`, `threshold_value`, `optical_flow_threshold`, `isotherm_min`, `isotherm_max`, `heat_seeker_max_boxes`, `heat_seeker_min_brightness`, `stabilize_strength`, `stabilize_smooth`, `phase_strength`, `phase_buffer_size`, `orb_buffer_size`. Unknown names are rejected.

### Request full state

```json
{
  "action": "get_state"
}
```

Returns the same JSON as `GET /state`.

### Gimbal control

Discrete button start/stop (also used by on-screen arrows and keyboard):

```json
{
  "action": "motor_command",
  "command": "motor_left",
  "state": "start"
}
```

`command` is one of `motor_up`, `motor_down`, `motor_left`, `motor_right`, or `motor_home` (home ignores `state`). Use `"state": "stop"` to end continuous stepping.

Soft home to a saved UI zero (absolute pan/tilt). Omitting `pan`/`tilt` runs limit-switch hardware home:

```json
{
  "action": "motor_command",
  "command": "motor_home",
  "pan": 75.0,
  "tilt": 30.0
}
```

Browser / host gamepad analog angles:

```json
{
  "action": "motor_command",
  "command": "gamepad_input",
  "pan": 45.0,
  "tilt": 10.0
}
```

Motor step rate (Hz):

```json
{
  "action": "set_motor_speed",
  "pan_hz": 100,
  "tilt_hz": 100
}
```

Gimbal control requires the Rust `cookie-finder-ctl` daemon (`RustGimbalClient` over a Unix socket). Start it with `make on-the-pi-rust-daemon`.

---

## Camera Thread Behavior

- Auto-scans `/dev/video0`–`4` on startup
- Adaptive reconnect backoff: 0.5s → 1s → 2s
- Frame queue size: 2 (drops oldest frame on overflow to prevent lag)
