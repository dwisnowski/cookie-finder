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

## Web Server as systemd Daemon (Orange Pi)

To install and start the web app on boot:

```bash
make on-the-pi-web-daemon
```

That installs `cookie-finder-web.service`, enables it on boot, and starts it. Useful commands:

```bash
make on-the-pi-web-daemon-status
make on-the-pi-web-daemon-stop
sudo journalctl -u cookie-finder-web -f
```

Override bind address/port when installing:

```bash
make on-the-pi-web-daemon WEB_HOST=0.0.0.0 WEB_PORT=8080
```

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

## WiFi Access Point Mode

On the Orange Pi, run once (after `make install`):

```bash
make init-wifi
```

This installs AP tooling, passwordless sudo for `scripts/wifi-mode.sh`, and the always-on **WiFi button + LED** systemd service (`cookie-finder-wifi`). The physical button works even when the web app is not running.

The home screen badge shows **Client · …** or **AP · cookie-finder**.

In **Settings → WiFi Mode**, toggle between client WiFi and the onboard hotspot. A confirmation dialog lists reconnect steps before the radio switches. The GPIO button does the same toggle.

**Boot policy:** AP mode does **not** survive a full reboot (power cycle). After reboot, `cookie-finder-wifi` restores client (home/office) WiFi. A plain `systemctl restart` while already in AP leaves AP alone. If the radio looks like client but has no SSID, the button repairs client mode instead of entering AP.

- AP SSID: `cookie-finder`
- AP password: none (open SoftAP — WPA SoftAP is unreliable on this WiFi chip)
- AP URL: `http://192.168.12.1:8000`
- LED: solid = client, slow blink (~1 Hz) = AP, fast blink (~5 Hz) = switching

Wiring details: [Hardware — Wiring](../hardware/wiring.md#wifi-mode-button--led). Networking prerequisites (NM-only Wi‑Fi, no netplan Wi‑Fi, optional fast-boot mask): [Orange Pi setup §4](../setup/orange-pi.md#4-configure-wifi-networkmanager-only).

---

## YOLO

YOLO is loaded lazily on first use. The `yolo` extra must be installed:

```bash
make install-yolo
```

Then press `y` in the UI or keyboard to activate.
