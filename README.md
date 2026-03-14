# Thermal Camera Viewer

A real-time USB thermal camera viewer using OpenCV and UVC (USB Video Class).

## Camera Profile

**Hardware:** Zhejiang Pixfra thermal camera  
**Performance:** 50 FPS (vs. 30 FPS standard webcams)  
**Resolution:** 512x390  
**Interface:** UVC (driverless, acts like a high-end webcam)  
**Output:** Processed video (YUV/RGB), not raw 16-bit temperature data

## Setup

```bash
make install
```

## Usage

```bash
make run
```

Press `q` to quit.

## Supported Pixel Formats

The camera supports:
- uyvy422
- yuyv422
- nv12
- 0rgb
- bgr0


## OPENCV CAMERA PROPERTIES
  Frame Width: 512.0
  Frame Height: 390.0
  FPS: 50.0
  Brightness: 0.0
  Contrast: 0.0
  Saturation: 0.0
  Hue: 0.0
  Gain: 0.0
  Exposure: 0.0
  Auto Exposure: 0.0
  White Balance: 0.0
  Backlight Compensation: 0.0
  Focus: 0.0
  Auto Focus: 0.0

## Clean Up

```bash
make clean
```
