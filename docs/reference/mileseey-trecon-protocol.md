# Mileseey T-Recon / PixFra CDC protocol

Reversed from **mileseey T-recon** Android app `com.mileseey.trecon` **1.1.6**
(`libSendDataLib.so` → `PacketUtils::makeCmd`).

## Device IDs the app accepts

| VID | PID | Notes |
|-----|-----|--------|
| `0x2E03` (11779) | `0x2507` (9479) | TNV30i / `PixFra THERMAL USB` (this project) |
| `0x2E03` | `0x2508` (9480) | Sibling SKU |
| `0x04B4` (1204) | `0x00F1` (241) | Alternate Cypress-based unit |

Transport: after CDC ACM line-coding setup, commands are **bulk OUT** on the CDC data interface (same channel as `/dev/ttyACM*`).

## Frame layout

```
6E 00 | flags | cmd | len_hi len_lo | hdr0 hdr1 | payload… | crc0 crc1
```

| Field | Size | Meaning |
|-------|------|---------|
| magic | 2 | `6E 00` |
| flags | 1 | `00` for app → device |
| cmd | 1 | function code |
| length | 2 | payload length, **big-endian** |
| hdr check | 2 | CCITT-table fold of `(cmd, length)` |
| payload | N | command arguments |
| CRC | 2 | CRC-16/CCITT (poly `0x1021`, init `0`) over header+payload |

Builder: `tools/probing_thermal_camera/mileseey_protocol.py`.

## Function codes (subset)

| Cmd | Name | Payload | Used by T-Recon UI? |
|-----|------|---------|---------------------|
| `0x0C` | FFC / shutter | empty | **Yes** (preview FFC button) |
| `0x0B` | FFC mode | BE u16: `0` auto, `1` manual | protocol present |
| `0x0D` | FFC interval | BE u16 seconds | protocol present |
| `0x10` | Color / palette | BE u16 palette id | **Protocol yes; UI no** — UI uses `CameraNative.setColor` (host recolor) |
| `0x55` | Brightness / light | BE u16 (`progress * 5` in UI) | **Yes** |
| `0x2D` | Contrast | BE u16 | protocol / native helpers |
| `0x88` | Image mode | BE u16 mode id | **Yes** (plain/jungle/rain/sketch) |
| `0x01` | Save config | empty | protocol |
| `0x02` | Restart | empty | protocol |

### Palette IDs (`WeicaiData`)

| ID | Name |
|----|------|
| 0 | White hot |
| 1 | Black hot |
| 6 | Iron red |
| 7 | Sepia |
| 13 | Green hot |
| 18 | Alarm |

### Image mode IDs (as sent via `setImgMode`)

| ID | Name |
|----|------|
| 0 | Plain |
| 1 | Jungle / forest (Bird also maps to 1 in UI) |
| 2 | Rain / fog |
| 3 | Sketch |

## Example frames

```text
FFC                      6e00000c0000aada0000
palette white_hot        6e0000100002bc9a00000000
palette black_hot        6e0000100002bc9a00011021
palette iron_red         6e0000100002bc9a000660c6
image_mode jungle        6e00008800026d0200011021
brightness 50            6e00005500024ac700321611
contrast 50              6e00002d00023b6e00321611
```

Regenerate / send:

```bash
# Print example frames
make on-the-pi-mileseey-examples

# Send to camera CDC port (default MILESEEY_PORT=/dev/ttyACM0)
make on-the-pi-mileseey-ffc
make on-the-pi-mileseey-palette MILESEEY_VALUE=iron_red
make on-the-pi-mileseey-ffc-mode MILESEEY_VALUE=manual
make on-the-pi-mileseey-image-mode MILESEEY_VALUE=jungle
make on-the-pi-mileseey-brightness MILESEEY_VALUE=50
make on-the-pi-mileseey-contrast MILESEEY_VALUE=50

# Or call the helper directly
uv run tools/probing_thermal_camera/mileseey_protocol.py --examples
uv run tools/probing_thermal_camera/mileseey_protocol.py --port /dev/ttyACM0 --cmd ffc
uv run tools/probing_thermal_camera/mileseey_protocol.py --port /dev/ttyACM0 --cmd palette --value iron_red
```

## Implications for Cookie Finder

1. **FFC** is the clearest device control to implement first over CDC.
2. **Palette changes in the phone app** mostly recolor locally; UVC to the Pi may stay on the camera’s current LUT unless cmd `0x10` is also accepted by firmware (worth probing).
3. **Image mode / brightness** are real device commands and may change the UVC picture Cookie Finder sees.
4. This is **not** the InfiRay Tiny1-C / `0BDA` vendor-control protocol from ThermalApp.
