# Camera — Hardware Reference

## Device

| Field | Value |
|-------|-------|
| Brand | Mileseey / Zhejiang Pixfra |
| Model | TNV30i |
| Vendor ID | `0x2e03` |
| Product ID | `0x2507` |
| USB Name | `PixFra THERMAL USB` |
| Interface | UVC (driverless — works like a webcam) |
| Resolution | 512 × 390 |
| Frame Rate | 50 FPS |
| Sensitivity | 20 mK |
| Detection Range | 1600 yards |

## USB Interface Layout

```
Configuration 1
├── Interface 0: Class 14 (Video / UVC)
├── Interface 1: Class 14 (Video / UVC)
│   └── Endpoint 0x81: IN (Bulk), Max Packet 512
├── Interface 2: Class 2 (Communications)
│   └── Endpoint 0x83: IN (Interrupt), Max Packet 8
└── Interface 3: Class 10 (CDC Data)
    ├── Endpoint 0x02: OUT (Bulk), Max Packet 512
    └── Endpoint 0x85: IN (Bulk), Max Packet 512
```

> No vendor-specific interfaces exposed — the camera outputs standard 8-bit UVC video only. Raw 16-bit temperature data is not accessible.

## Supported Pixel Formats

- `uyvy422`
- `yuyv422`
- `nv12`
- `0rgb`
- `bgr0`

## Output Characteristics

- Outputs **processed video** (YUV/RGB), not raw temperature data
- Pixel values represent processed thermal intensity (0–255 after normalization)
- Pseudo-color and grayscale palettes available in software

## Useful Links

- [Purchase (Amazon)](https://www.amazon.com/Monocular-TNV30i-Super-Resolution-1600-Yard-Detection/dp/B0FZKDKW87?ref_=ast_sto_dp)
- [Quick Start + Spec Sheet (PDF)](https://cdn-files.myshopline.com/file/store/1729084058744/TNV30i-Quick-Start-Guide.pdf)
- [Pixfra Firmware Downloads](https://www.pixfra.com/pips2-0)

## Verifying the Camera on Linux

```bash
lsusb
v4l2-ctl --list-devices
```

Expected output includes something like `/dev/video1`. Use the `make find-camera` command for a richer report.

```bash
make find-camera
```

Test the raw stream (HDMI display required):

```bash
mpv --vo=drm av://v4l2:/dev/video1
```
