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

---

## WebSocket — `/control`

A persistent WebSocket for real-time bidirectional control.

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
