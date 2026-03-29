# Development Setup (Desktop)

For writing and testing logic on macOS or Linux before deploying to the Orange Pi.

---

## Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) — Python environment manager
- A webcam or pre-recorded frames to substitute for the thermal camera

---

## Install

```bash
git clone git@github.com:dwisnowski/cookie-finder.git
cd cookie-finder
make install
```

To include YOLO AI detection:

```bash
make install-yolo
```

---

## Optional Tools

```bash
make install-ffmpeg     # brew install ffmpeg (macOS)
make install-libusb     # brew install libusb (macOS, needed for USB probing tools)
```

---

## Running (Desktop)

**Web server mode** (recommended — no display required):

```bash
make run
```

**Standalone OpenCV GUI mode** (requires a display):

```bash
make run-standalone
```

---

## Hardware-Specific Notes

The following tools and paths are Linux/ARM-specific and will not work on macOS/Windows:

| Feature | Linux/ARM only? |
|---------|-----------------|
| `v4l2-ctl`, `/dev/video*` paths | Yes |
| GPIO (gimbal motors) | Yes (`/dev/gpiochip*`) |
| `make test-pan-step` | Yes |
| Main Armbian setup scripts | Yes |

On macOS, use a standard webcam (`/dev/video0` equivalent, auto-detected by OpenCV) or pass a device index with `--camera 0`.

---

## Deploying to Orange Pi

```bash
# Copy changed files via SCP
scp -r cookie_finder/ main.py user@<pi-ip>:~/cookie-finder/

# Or push to git and pull on the device
git push
ssh user@<pi-ip> "cd cookie-finder && git pull"
```

Use logs for debugging — the web server prints frame stats and error traces to stdout.
