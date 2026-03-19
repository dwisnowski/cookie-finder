# Cookie Finder – Copilot Instructions

## Project Overview
Cookie Finder is a hardware + software project that uses an Orange Pi Zero 2W with a USB thermal camera to detect and locate warm objects (e.g., cookies) in real time.

The system:
- Captures thermal image data via USB
- Processes frames to detect heat signatures
- Optionally tracks movement or identifies targets
- Outputs results via visualization, logs, or control signals (future: gimbal control)

This project is designed to be lightweight, headless-friendly, and runnable on embedded Linux systems.

> Note: The core viewer can also run on macOS/Windows (for development or demo), but some hardware access and tools (e.g., `v4l2-ctl`, `/dev/video*` paths, and the main Armbian setup scripts) are Linux/ARM-specific and may behave differently.

---

## Hardware Context

### Target Device
- Orange Pi Zero 2W
- OS: Armbian (Debian/Ubuntu-based)

### Peripherals
- USB thermal camera (e.g., Mileseey TNV30i or similar UVC-compatible device)
- Optional:
  - Pan/tilt gimbal (future integration)
  - External WiFi antenna

### Constraints
- Limited CPU/GPU resources
- No guarantee of hardware acceleration
- Must work without a GUI (SSH-only environments)

---

## Software Architecture

### Core Components
- **Capture Layer**
  - Interfaces with USB camera (UVC / V4L2)
  - Uses OpenCV or direct device access
- **Processing Layer**
  - Frame normalization (thermal range scaling)
  - Hotspot detection (thresholding / blob detection)
  - Optional tracking logic
- **Output Layer**
  - Debug visualization (if GUI available)
  - CLI logs / metrics
  - Future: control signals for actuators

---

## Key Libraries & Tools

Prefer:
- Python 3.x
- OpenCV (`cv2`)
- NumPy

Optional:
- `v4l2-ctl` (for debugging camera devices)
- `ffmpeg` (for stream inspection)

Avoid:
- Heavy frameworks (TensorFlow, PyTorch) unless explicitly required
- GPU-dependent solutions

---

## Coding Guidelines

### General
- Keep code **lightweight and efficient**
- Avoid unnecessary abstractions
- Prefer simple, readable functions over complex class hierarchies
- 🎯 Do not add tests or test files; this repo is intentionally not structured for automated test suites.
- Keep chat output to a minimum, only include code snippets when necessary to illustrate a point. Focus on clear, concise explanations and instructions.

### Performance
- Minimize memory allocations inside loops
- Use NumPy vectorization where possible
- Avoid blocking I/O in frame processing loops

### Hardware Safety
- Always handle device disconnects gracefully
- Retry camera initialization if it fails
- Do not assume consistent device paths (e.g., `/dev/video0`)

---

## Camera Handling

- Use OpenCV (`cv2.VideoCapture`) or direct V4L2 access
- Validate:
  - Device availability
  - Frame resolution
  - Pixel format

Example expectations:
- Frames may be grayscale or pseudo-color
- Thermal range may need normalization

---

## Thermal Processing Expectations

Typical pipeline:
1. Capture frame
2. Convert to grayscale (if needed)
3. Normalize intensity
4. Apply threshold to detect heat sources
5. Find contours / blobs
6. Return coordinates of hottest region

Keep algorithms:
- Deterministic
- Fast (target: real-time or near real-time)

---

## Networking

- Device is accessed via SSH over local network
- IP may change → avoid hardcoding
- No cloud dependency

---

## Development Workflow

### On Desktop (Windows/Linux)
- Write and test logic using recorded frames or webcam
- Avoid hardware-specific assumptions

### On Orange Pi
- Deploy via SSH / SCP
- Run headless
- Use logs for debugging

