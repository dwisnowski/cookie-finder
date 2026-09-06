# Makefile Reference

Prefer host-prefixed targets: `on-the-mac-*` (run on a MacBook) and `on-the-pi-*` (run on the Orange Pi). Short names (`install`, `run`, `serial-*`, `rust-*`, …) remain as **legacy aliases**.

```bash
make help              # points at Mac vs Pi help
make on-the-mac-help   # full Mac target list
make on-the-pi-help    # full Pi target list
```

---

## Install

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-pi-install` / `make install` | Pi | `uv sync` — core dependencies |
| `make on-the-pi-install-yolo` / `make install-yolo` | Pi | `uv sync --extra yolo` (YOLO/PyTorch) |
| `make on-the-mac-install-docs` / `make install-docs` | Mac | `uv sync --extra docs` |
| `make on-the-mac-install-ffmpeg` / `make install-ffmpeg` | Mac | `brew install ffmpeg` |
| `make on-the-mac-install-libusb` / `make install-libusb` | Mac | `brew install libusb` |
| `make on-the-pi-init` / `make init` | Pi | Add user to Bluetooth group |
| `make on-the-pi-init-wifi` / `make init-wifi` | Pi | AP deps, captive DNS, sudoers (WiFi + poweroff), WiFi GPIO service |
| `make on-the-pi-init-software-update` / `make init-software-update` | Pi | Allow Settings → Software to pull GitHub `main` + restart web |
| `make on-the-pi-mdns` / `make mdns` | Pi | Hostname + Avahi for `http://cookie-finder.local/` |
| `make on-the-mac-tool-setup` | Mac | Rustup + cross + Docker check |
| `make on-the-pi-tool-setup` | Pi | apt build deps + Rustup |

---

## Run

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-pi-run` / `make run` | Pi | Web server foreground (`:80` + `:443`) |
| `make on-the-pi-run-web` / `make run-web` | Pi | Same as `on-the-pi-run` |
| `make on-the-pi-run-standalone` / `make run-standalone` | Pi | OpenCV GUI mode |
| `make on-the-pi-run-web-custom` / `make run-web-custom` | Pi | Prompt for host + ports |
| `make on-the-pi-web-daemon` / `make web-daemon` | Pi | Install/start `cookie-finder-web` systemd unit |
| `make on-the-pi-web-daemon-status` / `make web-daemon-status` | Pi | Service status |
| `make on-the-pi-web-daemon-stop` / `make web-daemon-stop` | Pi | Stop service |
| `make on-the-pi-web-url` / `make web-url` | Pi | Print IPv4 + URL (+ QR if `qrencode` installed) |

---

## WiFi

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-pi-wifi-status` / `make wifi-status` | Pi | Mode + IPs via `scripts/wifi-mode.sh` |
| `make on-the-pi-wifi-configure-clients` / `make wifi-configure-clients` | Pi | Save NM client profiles from `.wifi.env` |
| `make on-the-pi-wifi-fix` / `make wifi-fix` | Pi | Recover wedged client radio |
| `make on-the-pi-wifi-gpio-daemon` / `make wifi-gpio-daemon` | Pi | Install/start button+LED service |
| `make on-the-pi-wifi-gpio-daemon-status` | Pi | Button+LED service status |
| `make on-the-pi-wifi-gpio-daemon-stop` | Pi | Stop button+LED service |

Copy `.wifi.env.example` → `.wifi.env` and set `WIFI_HOME_PSK` / `WIFI_HOTSPOT_PSK` before `wifi-configure-clients`. Never commit `.wifi.env`.

---

## Motor / hardware tests (Pi)

| Target | Description |
|--------|-------------|
| `make on-the-pi-test-motors` / `make test-motors` | Help + auto sequence via Rust daemon IPC |
| `make on-the-pi-test-motors-pan-cw` … `-tilt-ccw` | Step one axis 50 steps via daemon |
| `make on-the-pi-test-motors-home` / `make test-motors-home` | Home both motors via daemon |
| `make on-the-pi-test-pan-step` / `make test-pan-step` | Direct `gpioset` wave-drive loop |
| `make on-the-pi-test-all-gpio` / `make test-all-gpio` | Blink every GPIO line |
| `make on-the-pi-rust-keyboard` / `make rust-keyboard` | Keyboard pan/tilt + drive mode / wiring |

---

## Camera

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-pi-find-camera` / `make find-camera` | Pi | `/dev/video*` + `v4l2-ctl` + Python scan |
| `make on-the-pi-list-devices` / `make list-devices` | Pi | UVC devices |
| `make on-the-pi-list-controls` / `make list-controls` | Pi | UVC controls |
| `make on-the-pi-get-control` / `make get-control` | Pi | Prompt + get control |
| `make on-the-pi-set-control` / `make set-control` | Pi | Prompt + set control |
| `make on-the-mac-list-cameras` / `make list-cameras` | Mac | `ffmpeg` avfoundation list |
| `make on-the-mac-list-camera-formats` | Mac | Capture one TIFF frame |

### Mileseey / PixFra CDC (Pi)

Send control frames over the camera CDC port (`MILESEEY_PORT`, default `/dev/ttyACM0`). See [Mileseey T-Recon protocol](mileseey-trecon-protocol.md).

| Target | Description |
|--------|-------------|
| `make on-the-pi-mileseey-examples` / `make mileseey-examples` | Print example frames |
| `make on-the-pi-mileseey-ffc` / `make mileseey-ffc` | Trigger FFC / shutter |
| `make on-the-pi-mileseey-palette MILESEEY_VALUE=iron_red` | Set palette (`white_hot`, `black_hot`, `iron_red`, `sepia`, `green_hot`, `alarm`) |
| `make on-the-pi-mileseey-ffc-mode MILESEEY_VALUE=auto` | FFC mode `auto` or `manual` |
| `make on-the-pi-mileseey-image-mode MILESEEY_VALUE=jungle` | Image mode (`plain`, `jungle`, `rain_fog`, `sketch`) |
| `make on-the-pi-mileseey-brightness MILESEEY_VALUE=50` | Brightness (default 50) |
| `make on-the-pi-mileseey-contrast MILESEEY_VALUE=50` | Contrast (default 50) |
| `make on-the-pi-mileseey-gain-auto MILESEEY_VALUE=0` | Set AGC / auto-gain (0–255); closest to a lock |
| `make on-the-pi-mileseey-get-gain-auto` | Query auto-gain |
| `make on-the-pi-mileseey-temp MILESEEY_VALUE=thermo_temp` | Radiometry/calibration query (`thermo_temp`, `base_gray`, `radiometry_options`, …) |
| `make on-the-pi-mileseey-temp-gain-mode` | Query temp gain mode (`0x0A`) |

---

## Probing (Mac)

| Target | Description |
|--------|-------------|
| `make on-the-mac-probe` / `make probe` | Run all probe scripts |
| `make on-the-mac-probe-install` / `make probe-install` | `brew install libusb` |
| `make on-the-mac-probe-usb` … `-xu` | Individual USB/CDC/serial/resolution/XU probes |

---

## Serial console (Mac → Pi)

USB-TTL UART when WiFi is down. Configure via `.serial.env` (copy from `.serial.env.example`).

| Target | Description |
|--------|-------------|
| `make on-the-mac-serial-help` / `make serial-help` | List serial targets + config |
| `make on-the-mac-serial-list` / `make serial-list` | List `/dev/tty.usbserial*` devices |
| `make on-the-mac-serial-connect` / `make serial-connect` | Open interactive `screen` session |
| `make on-the-mac-serial-run SERIAL_CMD='…'` | Run one remote command |
| `make on-the-mac-serial-deploy` / `make serial-deploy` | Tarball sync + `uv sync` on Pi |
| `make on-the-mac-serial-deploy-rust` | Cross-compile Rust binary + copy over serial |

| Variable | Default | Description |
|----------|---------|-------------|
| `SERIAL_DEVICE` | `/dev/tty.usbserial-BG01PPKN` | USB-TTL device path |
| `SERIAL_BAUD` | `115200` | UART baud rate |
| `SERIAL_USER` | `cookie` | Pi login username |
| `SERIAL_PASSWORD` | *(required)* | Pi login password |
| `SERIAL_REMOTE_DIR` | `~/cookie-finder` | Remote project directory |

---

## Rust gimbal daemon

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-mac-rust-check` | Mac | `cargo check` (cross target) |
| `make on-the-mac-rust-build` | Mac | Cross-compile for Pi (`cross` + Docker) |
| `make on-the-mac-rust-deploy` / `make rust-deploy` | Mac | Build + `scp` to `PI_HOST` |
| `make on-the-mac-rust-deploy-cookie` / `make rust-deploy-cookie` | Mac | Build + `scp` via SSH config host |
| `make on-the-mac-rust-build-remote` | Mac | Build on Pi over SSH |
| `make on-the-mac-rust-daemon` | Mac | Deploy + start **foreground** daemon over SSH |
| `make on-the-pi-rust-build` | Pi | Native `cargo build --release` |
| `make on-the-pi-rust-daemon` | Pi | Install/start **systemd** `cookie-finder.service` |
| `make on-the-pi-rust-daemon-stop` / `-status` | Pi | Stop / status systemd unit |

**Alias gotcha:** `make rust-daemon` → `on-the-mac-rust-daemon` (SSH foreground), **not** Pi systemd. Use `make on-the-pi-rust-daemon` on the board.

---

## Docs / utilities

| Target | Host | Description |
|--------|------|-------------|
| `make on-the-mac-docs` / `make docs` | Mac | Serve MkDocs at `http://127.0.0.1:8001` |
| `make on-the-pi-clean` / `make clean` | Pi | Remove `__pycache__` / `.pyc` |
| `make on-the-pi-armbian-home-screen` | Pi | Print Armbian MOTD |
