# Cookie Finder

Real-time USB thermal camera system for detecting and locating warm objects (cookies, people, heat sources) using an **[Orange Pi Zero 2W](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html)**.

> This project is configured specifically for the Orange Pi Zero 2W. GPIO pin offsets, chip assignments, and setup instructions are board-specific.

## What it does

- Captures thermal video via a UVC USB thermal camera
- Processes frames to detect heat signatures in real time
- Exposes a live MJPEG stream + WebSocket control panel via a web server
- Optionally controls a pan/tilt gimbal (Rust daemon preferred; Python GPIO fallback)

## Modes at a glance

| Mode | Key | Description |
|------|-----|-------------|
| Heat-Seeker | `h` | Highlights brightest pixel clusters |
| Heat-Cluster | `c` | Highlights largest heat regions |
| Motion | `m` | Frame-differencing motion detection |
| Palette | `p` | Thermal color maps |
| Threshold | `t` | Binary brightness range isolation |
| YOLO | `y` | YOLOv8n object detection |
| Optical Flow | `f` | Visualize heat velocity |
| Isotherm | `i` | Highlight a specific temperature band |

## Quick start

```bash
make install
make run
```

Then open `http://<device-ip>/` (or `http://cookie-finder.local/`) in a browser.

## Documentation

- [Hardware — Camera](hardware/camera.md)
- [Hardware — Wiring](hardware/wiring.md)
- [Hardware — Stepper Wiring Test](hardware/stepper-wiring-test.md)
- [Setup — Orange Pi](setup/orange-pi.md)
- [Setup — Development](setup/development.md)
- [Usage — Running](usage/running.md)
- [Usage — Keyboard Controls](usage/controls.md)
- [Usage — Modes](usage/modes.md)
- [Usage — Web Server API](usage/web-server.md)
- [Reference — Makefile](reference/makefile.md)
- [Reference — UVC Controls](reference/uvc-controls.md)
