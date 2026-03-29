# Keyboard Controls

Available in standalone GUI mode (`make run-standalone`).  
In web mode, most of these are also accessible via the browser UI and WebSocket API — see [Web Server API](web-server.md).

---

## Mode Toggles

Only one exclusive mode can be active at a time.

| Key | Mode |
|-----|------|
| `h` | Heat-Seeker |
| `c` | Heat-Cluster |
| `m` | Motion Detection |
| `p` | Palette |
| `t` | Threshold |
| `y` | YOLO AI Detection |
| `f` | Optical Flow |
| `i` | Isotherm Highlight |

---

## Enhancement Toggles

These stack on top of any mode.

| Key | Enhancement |
|-----|-------------|
| `d` | Denoise (temporal frame averaging) |
| `o` | Normalize (stretch 0–255) |
| `e` | Enhance Details (CLAHE) |
| `s` | Stabilize (ORB + RANSAC) |
| `x` | Super Stabilize (phase correlation) |
| `u` | Upscale (2× Lanczos) |
| `w` | Toggle text overlay |

---

## Camera Controls

| Key | Action |
|-----|--------|
| `r` | Manual reconnect / rescan cameras |
| `Tab` | Cycle to next camera device |

---

## Palette Mode Controls

| Key | Action |
|-----|--------|
| `n` | Cycle to next palette |

Available palettes: Ironbow, Rainbow, Lava, Ocean, Magma, WhiteHot, BlackHot.

---

## Threshold / Optical Flow Masked Mode Controls

| Key | Action |
|-----|--------|
| `=` | Increase threshold value (+5) |
| `-` | Decrease threshold value (−5) |

Range: 0–255. Default: 127.

---

## Heat-Seeker Mode Controls

| Key | Action |
|-----|--------|
| `←` | Decrease max boxes (min 1) |
| `→` | Increase max boxes (max 15) |
| `↑` | Increase min brightness threshold |
| `↓` | Decrease min brightness threshold |

---

## Isotherm Highlight Mode Controls

| Key | Action |
|-----|--------|
| `←` | Decrease min threshold |
| `→` | Increase min threshold |
| `↓` | Decrease max threshold |
| `↑` | Increase max threshold |
| `b` | Toggle black / red mask color |

---

## Stabilization Mode Controls

| Key | Action |
|-----|--------|
| `=` | Increase stabilization strength (+0.05, range 0.0–1.0) |
| `-` | Decrease stabilization strength (−0.05) |
| `[` | Decrease temporal smoothing |
| `]` | Increase temporal smoothing (range 0.0–0.99) |

Default strength: `1.0`. Default smoothing: `0.7`.

---

## General

| Key | Action |
|-----|--------|
| `q` | Quit |
