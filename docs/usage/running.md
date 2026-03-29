# Running the Application

## Modes

Cookie Finder has two runtime modes:

| Mode | Command | Description |
|------|---------|-------------|
| **Web server** (default) | `make run` | FastAPI + MJPEG stream, browser UI, WebSocket control |
| **Standalone GUI** | `make run-standalone` | OpenCV window, keyboard controls, requires a display |

---

## make run (Web Server)

```bash
make run
```

Starts the server on `http://0.0.0.0:8000`. Open `http://<device-ip>:8000` in any browser on the same network.

## Custom Host / Port

```bash
make run-web-custom
# Prompts for host and port
```

Or directly via Python:

```bash
uv run main.py --web --host 0.0.0.0 --port 8080
```

## Standalone GUI

```bash
make run-standalone
# or
uv run main.py
```

Requires a connected display. Press `q` to quit.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--web` | off | Start in web server mode |
| `--host` | `0.0.0.0` | Bind host (web mode only) |
| `--port` | `8000` | Bind port (web mode only) |
| `--camera` | auto | Force a specific camera device ID |

---

## Camera Selection

On startup, the app auto-scans `/dev/video0`–`4` and picks the first working device.

To force a specific device:

```bash
uv run main.py --camera 1
```

To manually reconnect or cycle cameras at runtime:
- In standalone mode: press `r` to reconnect, `Tab` to cycle to the next device
- In web mode: use the `/reconnect` endpoint or the browser UI

---

## YOLO

YOLO is loaded lazily on first use. The `yolo` extra must be installed:

```bash
make install-yolo
```

Then press `y` in the UI or keyboard to activate.
