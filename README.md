# Thermal Camera Viewer

A real-time USB thermal camera viewer using OpenCV with advanced thermal analysis modes for Intel Macs.

## Camera Profile
[purchase link](https://www.amazon.com/Monocular-TNV30i-Super-Resolution-1600-Yard-Detection/dp/B0FZKDKW87?ref_=ast_sto_dp)

[Quick start and Spec sheet](https://cdn-files.myshopline.com/file/store/1729084058744/TNV30i-Quick-Start-Guide.pdf)

[Pixfra firmware downloads](https://www.pixfra.com/pips2-0)

MILESEEY TNV30i Night Vision Thermal Scope, 512×384 Super-Resolution 50Hz 20mK 1600-Yard Thermal Imaging Camera

**Hardware:** Zhejiang Pixfra thermal camera  
**Performance:** 50 FPS (vs. 30 FPS standard webcams)  
**Resolution:** 512x390  
**Interface:** UVC (driverless, acts like a high-end webcam)  
**Output:** Processed video (YUV/RGB), not raw 16-bit temperature data  
**Supported Pixel Formats:** uyvy422, yuyv422, nv12, 0rgb, bgr0

============================================================
Device: PixFra THERMAL USB
Vendor ID: 0x2e03, Product ID: 0x2507
============================================================

Configuration 1:
  Interface 0: Class 14 (Video (UVC))
  Interface 1: Class 14 (Video (UVC))
      Endpoint 0x81: IN (Bulk), Max Packet: 512
  Interface 2: Class 2 (Communications)
      Endpoint 0x83: IN (Interrupt), Max Packet: 8
  Interface 3: Class 10 (CDC Data)
      Endpoint 0x2: OUT (Bulk), Max Packet: 512
      Endpoint 0x85: IN (Bulk), Max Packet: 512

Searching for all video devices (Class 14)...


============================================================
✗ No vendor-specific interfaces found.
  Camera is likely only exposing standard 8-bit UVC video.
============================================================

## Setup

```bash
make install
```

## Usage

```bash
make run
```

Press `q` to quit.

## Keyboard Controls

### Mode Toggles
- **h** - Heat-Seeker mode (detect brightest pixel clusters)
- **c** - Heat-Cluster mode (detect largest heat clusters)
- **m** - Motion Detection mode (detect moving objects)
- **p** - Palette mode (apply thermal color palettes)
- **t** - Threshold mode (isolate specific brightness range)
- **y** - YOLO AI Detection mode (object detection)
- **Shift+Y** - YOLO skip-frame optimization (every 4th frame)
- **f** - Optical Flow mode (visualize heat velocity)
- **Shift+F** - Masked Optical Flow mode (flow only in hot areas)
- **i** - Isotherm Highlight mode (highlight specific heat range)

### Enhancement Toggles (work with all modes)
- **d** - Denoise mode (temporal frame averaging)
- **o** - Normalize mode (stretch 0-255 range)
- **e** - Enhance Details (CLAHE contrast enhancement)
- **s** - Stabilization mode (ORB+RANSAC feature tracking)
- **Shift+S** - Toggle stabilization method (ORB+RANSAC vs ECC)
- **u** - Upscale mode (2x resolution with Lanczos interpolation)

### Heat-Seeker Mode Controls
- **Left Arrow** - Decrease max boxes (1-15)
- **Right Arrow** - Increase max boxes (1-15)
- **Up Arrow** - Increase min brightness threshold (0-255)
- **Down Arrow** - Decrease min brightness threshold (0-255)

### Palette Mode Controls
- **n** - Cycle to next palette
- Available palettes: Ironbow, Rainbow, Lava, Ocean, Magma, WhiteHot, BlackHot

### Threshold Mode Controls
- **=** - Increase threshold value (0-255)
- **-** - Decrease threshold value (0-255)

### Optical Flow Masked Mode Controls
- **=** - Increase threshold value (0-255)
- **-** - Decrease threshold value (0-255)

### Stabilization Mode Controls
- **=** - Increase stabilization strength (0.0-1.0)
- **-** - Decrease stabilization strength (0.0-1.0)
- **[** - Decrease temporal smoothing (0.0-0.99, lower = less smoothing)
- **]** - Increase temporal smoothing (0.0-0.99, higher = more smoothing over frames)

### Isotherm Highlight Mode Controls
- **Left Arrow** - Decrease min threshold
- **Right Arrow** - Increase min threshold
- **Down Arrow** - Decrease max threshold
- **Up Arrow** - Increase max threshold
- **b** - Toggle black/red mask color

### General Controls
- **q** - Quit application

## UVC Camera Controls

List available camera devices:
```bash
make list-devices
```

List available camera controls:
```bash
make list-controls
```

Get a specific control value:
```bash
make get-control
```

Set a specific control value:
```bash
make set-control
```

## Modes Explained

### Heat-Seeker
Detects and highlights the brightest pixel clusters. Useful for finding hot spots in engines, buildings, or people in darkness. Supports filtering by max box count and minimum brightness threshold.

### Heat-Cluster
Identifies the largest connected heat regions and labels them by area. Great for tracking large thermal objects.

### Motion Detection
Uses frame differencing to detect moving objects. Excellent for security applications.

### Palette
Applies thermal color maps to the grayscale feed. Choose from 7 different color schemes to visualize heat differently.

### Threshold
Creates a binary mask of pixels within a specific brightness range. Useful for isolating specific temperature ranges.

### YOLO AI Detection
Real-time object detection using YOLOv8 Nano. Detects people, animals, and objects. Skip-frame mode runs detection every 4th frame for Intel Mac performance.

### Optical Flow
Visualizes heat movement and velocity using the Farneback algorithm. Shows direction and magnitude of thermal changes.

### Masked Optical Flow
Shows optical flow only in hot areas (above threshold). Optimized for Intel Macs with low-resolution calculation.

### Isotherm Highlight
Highlights pixels within a specific brightness range in red (or black) while keeping everything else grayscale. Perfect for targeting specific heat signatures.

### Denoise
Applies temporal frame averaging to reduce sensor noise while preserving heat data. Zero CPU overhead.

### Normalize
Stretches the entire 0-255 brightness range to maximize contrast.

### Enhance Details (CLAHE)
Applies Contrast Limited Adaptive Histogram Equalization to pull out subtle thermal textures and details.

### Stabilization
Uses ORB feature tracking with RANSAC outlier rejection to align frames and reduce camera shake. Finds high-contrast corner points (like edges of hot objects) and uses them as anchors. Supports both ORB+RANSAC (default, more robust) and ECC methods. Works well even with low contrast thermal images and handles rotation.

**Methods:**
- **ORB+RANSAC** (default): Feature-based tracking, robust to moving objects, supports rotation
- **ECC**: Pixel-perfect alignment, faster but less robust to thermal blob changes

**Strength Control:** Adjust stabilization strength from 0.0 (no stabilization) to 1.0 (full stabilization) to balance between stability and natural motion.

**Temporal Smoothing:** Use `[` and `]` to adjust smoothing factor (0.0-0.99). Higher values smooth motion over more frames, reducing jitter from hand shake. Default 0.7 provides good balance. Increase to 0.9+ for very smooth video with hand movements.

### Upscale
Upscales the 512x390 feed to 1024x780 using Lanczos interpolation for smoother visuals.

## Performance Notes

- **Intel Mac Optimization:** YOLO uses imgsz=320 for ~70% faster inference
- **Skip-Frame Logic:** YOLO skip-frame mode runs detection every 4th frame (~12-15 FPS) while video stays at 50 FPS
- **Persistent Rendering:** Bounding boxes persist across frames for smooth visual feedback
- **Temporal Denoising:** Frame averaging accumulates across frames for effective noise reduction

## Clean Up

```bash
make clean
```

## Project Structure

- `main.py` - Main application with all thermal analysis modes
- `uvc_controls.py` - UVC camera control utility
- `pyproject.toml` - Project dependencies
- `Makefile` - Build and run commands
