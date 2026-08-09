# Thermal Processing Modes

All modes are toggled independently. Most can be combined with enhancement toggles (denoise, normalize, enhance, stabilize, upscale).

---

## Exclusive Modes

Only one exclusive mode is active at a time. Toggle off by pressing the same key again.

### Heat-Seeker (`h`)

Detects and highlights the top-N brightest pixel clusters with green bounding boxes.

- Useful for finding people in darkness, hot engines, or other point heat sources
- Controls: `←` / `→` to adjust max box count; `↑` / `↓` to adjust min brightness threshold

### Heat-Cluster (`c`)

Identifies the largest connected heat regions and labels them by area with yellow bounding boxes.

- Great for tracking large thermal objects
- Uses morphological close operations to group nearby hot pixels

### Motion Detection (`m`)

Uses frame differencing to detect moving objects with blue bounding boxes.

- Best for security/surveillance applications
- Highlights pixels that changed significantly between consecutive frames

### Palette (`p`)

Applies a thermal color map to the grayscale feed.

| Palette | OpenCV Map |
|---------|------------|
| Ironbow | `COLORMAP_INFERNO` |
| Rainbow | `COLORMAP_JET` |
| Lava | `COLORMAP_HOT` |
| Ocean | `COLORMAP_VIRIDIS` |
| Magma | `COLORMAP_MAGMA` |
| WhiteHot | `COLORMAP_BONE` |
| BlackHot | `COLORMAP_TWILIGHT` |

Press `n` to cycle through palettes while palette mode is active.

### Threshold (`t`)

Creates a binary mask of pixels above a specific brightness value.

- Use `=` / `-` to raise or lower the threshold (default 127)
- Useful for isolating a specific temperature band

### YOLO AI Detection (`y`)

Real-time object detection using YOLOv8n Nano.

- Detects people, animals, and common objects
- Loaded lazily on first activation — requires `make install-yolo` (weights download automatically if missing)
- Runs on CPU; uses `imgsz=320` for performance on single-board computers
- Bounding boxes persist across frames for smooth visual feedback

### Optical Flow (`f`)

Visualizes heat movement and velocity using the Farneback dense optical flow algorithm.

- Output is an HSV color map where hue = direction, saturation = magnitude
- Shows direction and speed of thermal changes between frames

### Isotherm Highlight (`i`)

Highlights pixels within a specific brightness range in red (or black) while keeping everything else grayscale.

- Perfect for targeting a specific heat signature / temperature band
- Controls: `←` / `→` to adjust min; `↑` / `↓` to adjust max; `b` to toggle mask color

---

## Enhancement Toggles

These stack with any exclusive mode (or with each other).

### Denoise (`d`)

Temporal frame averaging. Accumulates frames with `alpha=0.2` to reduce sensor noise while preserving heat data.

- Zero CPU overhead (weighted blend, no spatial filtering)

### Normalize (`o`)

Stretches the full 0–255 brightness range to maximize contrast.

- Useful when the scene has a narrow temperature range

### Enhance Details (`e`)

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

- `clipLimit=2.0`, `tileGridSize=8×8`
- Pulls out subtle thermal textures and local detail

### Stabilize (`s`)

ORB feature tracking with RANSAC outlier rejection to align frames and reduce camera shake.

- Finds high-contrast corner points (edges of hot objects) and uses them as anchors
- Supports rotation; handles low-contrast thermal scenes well
- Controls: `=` / `-` for strength (0.0–1.0); `[` / `]` for temporal smoothing (0.0–0.99)

### Super Stabilize (`x`)

Phase correlation stabilization. Pixel-perfect alignment using FFT-based translation estimation.

- Faster than ORB method but less robust to moving objects or scene changes

### Upscale (`u`)

2× upscale from 512×390 to 1024×780 using Lanczos interpolation.

- Purely cosmetic; no new thermal data is synthesized
