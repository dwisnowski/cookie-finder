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

---

## Run

| Target | Description |
|--------|-------------|
| `make run` | Alias for `run-web` — start web server on port 8000 |
| `make run-web` | Start FastAPI server on `http://0.0.0.0:8000` |
| `make run-standalone` | Start OpenCV GUI mode (requires display) |
| `make run-web-custom` | Prompt for host + port, then start web server |

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
