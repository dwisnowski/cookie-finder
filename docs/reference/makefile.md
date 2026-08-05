# Makefile Reference

All available `make` targets.

---

## Install

| Target | Description |
|--------|-------------|
| `make install` | `uv sync` — install core dependencies |
| `make install-yolo` | `uv sync --extra yolo` — add YOLO/PyTorch deps |
| `make install-docs` | `uv sync --extra docs` — add MkDocs + Material theme |
| `make install-ffmpeg` | `brew install ffmpeg` (macOS) |
| `make install-libusb` | `brew install libusb` (macOS, needed for USB probing) |
| `make init` | Add user to Bluetooth group (Orange Pi) |
| `make init-wifi` | Install WiFi AP tools, captive DNS drop-in, sudoers, and `cookie-finder-wifi` button/LED service |
| `make on-the-pi-mdns` / `make mdns` | Set hostname + Avahi for `http://cookie-finder.local/` |
| `make on-the-pi-wifi-gpio-daemon` | Install/start WiFi button+LED systemd service |
| `make on-the-pi-wifi-gpio-daemon-status` | Show WiFi button+LED service status |
| `make on-the-pi-wifi-gpio-daemon-stop` | Stop WiFi button+LED service |

---

## Run

| Target | Description |
|--------|-------------|
| `make run` | Alias for `run-web` — start web server on :80 + :443 (foreground) |
| `make run-web` | Start FastAPI on HTTP :80 and HTTPS :443 (foreground) |
| `make run-standalone` | Start OpenCV GUI mode (requires display) |
| `make run-web-custom` | Prompt for host + ports, then start web server |
| `make on-the-pi-web-daemon` | Install/start web server systemd service (`cookie-finder-web`) on :80 + :443 |
| `make on-the-pi-web-daemon-status` | Show web server service status |
| `make on-the-pi-web-daemon-stop` | Stop web server service |
| `make on-the-pi-web-url` / `make web-url` | Print device IPv4 address(es) + web URL (+ terminal QR if `qrencode` is installed) |

---

## Motor Tests

| Target | Description |
|--------|-------------|
| `make test-motors` | Print motor test help and run auto sequence |
| `make test-motors-pan-cw` | Step pan motor clockwise 50 steps |
| `make test-motors-pan-ccw` | Step pan motor counter-clockwise 50 steps (`sudo`) |
| `make test-motors-tilt-cw` | Step tilt motor clockwise 50 steps |
| `make test-motors-tilt-ccw` | Step tilt motor counter-clockwise 50 steps |
| `make test-motors-home` | Home both motors to limit switches |
| `make test-pan-step` | Direct `gpioset` shell loop on gpiochip1 — confirms hardware pin offsets |
| `make test-all-gpio` | Scan and blink every GPIO pin on all chips |

---

## Camera

| Target | Description |
|--------|-------------|
| `make find-camera` | `ls /dev/video*` + `v4l2-ctl --list-devices` + python scan |
| `make list-devices` | List UVC devices via `uvc_controls.py` |
| `make list-controls` | List UVC camera controls |
| `make get-control` | Prompt for control name, print current value |
| `make set-control` | Prompt for control name + value, apply it |
| `make list-cameras` | `ffmpeg -f avfoundation -list_devices` (macOS only) |
| `make list-camera-formats` | Capture 1 TIFF frame via ffmpeg |

---

## Probing

| Target | Description |
|--------|-------------|
| `make probe` | Run all probe scripts |
| `make probe-install` | `brew install libusb` |
| `make probe-usb` | Probe USB descriptors |
| `make probe-cdc` | Probe CDC interface |
| `make probe-serial` | Probe serial interface |
| `make probe-resolution` | Probe supported resolutions |
| `make probe-xu` | Probe UVC extension units |

---

## Docs

| Target | Description |
|--------|-------------|
| `make docs` | Serve MkDocs locally at `http://127.0.0.1:8001` |

---

## Utilities

| Target | Description |
|--------|-------------|
| `make clean` | Remove `__pycache__` directories and `.pyc` files |

---

## Serial Console

USB-TTL UART access when WiFi is unavailable. Configure via `.serial.env` (copy from `.serial.env.example`).

| Target | Description |
|--------|-------------|
| `make serial-help` | List serial targets and current config |
| `make serial-list` | List `/dev/tty.usbserial*` devices on macOS |
| `make serial-connect` | Open `screen` on `SERIAL_DEVICE` at `SERIAL_BAUD` |
| `make serial-run SERIAL_CMD='…'` | Log in and run one shell command on the Pi |
| `make serial-deploy` | Tarball project, transfer over serial, run `uv sync` on Pi |
| `make serial-deploy-rust` | Cross-compile Rust binary and copy to Pi over serial |

Variables (defaults in Makefile, override in `.serial.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SERIAL_DEVICE` | `/dev/tty.usbserial-BG01PPKN` | USB-TTL device path |
| `SERIAL_BAUD` | `115200` | UART baud rate |
| `SERIAL_USER` | `cookie` | Pi login username |
| `SERIAL_PASSWORD` | *(required)* | Pi login password |
| `SERIAL_REMOTE_DIR` | `~/cookie-finder` | Remote project directory |
