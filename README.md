# Cookie Finder

Real-time USB thermal camera system for detecting heat signatures using an Orange Pi Zero 2W.

**[Full documentation →](docs/index.md)**

## Makefile targets

Targets are prefixed by where you run them:

| Prefix | Run on | Examples |
|--------|--------|----------|
| `on-the-pi-*` | Orange Pi | `on-the-pi-install`, `on-the-pi-run`, `on-the-pi-rust-build` |
| `on-the-mac-*` | MacBook | `on-the-mac-rust-deploy`, `on-the-mac-serial-connect` |

List everything:

```bash
make on-the-pi-help    # on the Pi
make on-the-mac-help   # on your Mac
```

Legacy names (`install`, `run`, `rust-*`, `serial-*`, etc.) still work as aliases.

## Quick Start (Web Server)

On the Orange Pi:

```bash
make on-the-pi-install
make on-the-pi-run
```

Open `http://<device-ip>/` (or `http://cookie-finder.local/` after `make mdns`) in a browser.

To run as a systemd service (starts on boot, HTTP :80 + HTTPS :443):

```bash
make on-the-pi-web-daemon
```

(`make install` / `make run` / `make web-daemon` are aliases for the commands above.)

## WiFi Access Point

When home WiFi is unavailable, the Pi can host its own hotspot (single GPIO button press or Settings → WiFi Mode). Triple-click the button, or Settings → **Shut down**, to power off the board.

| | |
|--|--|
| **SSID** | `cookie-finder` |
| **Password** | none (open network) |
| **URL** | `http://192.168.12.1/` (phone SoftAP; captive portal opens this automatically) |
| **Tesla SoftAP** | Settings → SoftAP profile **Tesla** → join `cookie-finder` → open `http://3.3.3.3/` |

One-time setup on the Pi: `make on-the-pi-init-wifi` (or `make init-wifi`). Details: [WiFi AP mode](docs/usage/running.md#wifi-access-point-mode), [Orange Pi setup](docs/setup/orange-pi.md#5-enable-wifi-access-point-mode-optional).

## Rust Gimbal Daemon (Required on Pi for motors / gamepad)

The gimbal and Pi gamepad control loop run in a Rust daemon. The Python web server connects to it over a Unix socket; without the daemon, motors and pad input are unavailable.

### Mac: cross-compile and deploy

Prerequisites (one-time on your Mac):

```bash
make on-the-mac-tool-setup    # installs Rust/cargo + cross; Docker Desktop must be running
```

Build and deploy:

```bash
make on-the-mac-rust-build                 # produces aarch64 Linux binary
make on-the-mac-rust-deploy PI_HOST=orangepi@<pi-ip>
```

### Orange Pi: build natively

Prerequisites (one-time on the Pi):

```bash
make on-the-pi-tool-setup    # apt build deps + Rust/cargo via rustup
```

Build:

```bash
make on-the-pi-rust-build
```

Or build remotely from your Mac over SSH:

```bash
make on-the-mac-rust-build-remote PI_HOST=orangepi@<pi-ip>
```

### Run on the Pi

Install the systemd unit and start the daemon (requires `sudo` for GPIO):

```bash
make on-the-pi-rust-daemon
```

That installs `cookie-finder.service`, enables it on boot, and starts it. Useful commands:

```bash
sudo systemctl status cookie-finder
sudo journalctl -u cookie-finder -f
make on-the-pi-rust-daemon-stop
```

Or run the binary manually (without systemd):

```bash
sudo ./cookie-finder-ctl daemon
# or: sudo ./cookie_finder_rust/target/release/cookie-finder-ctl daemon
```

Then start the web server in another terminal:

```bash
make on-the-pi-run
```

Or do both from your Mac in one step:

```bash
make on-the-mac-run-with-rust PI_HOST=orangepi@<pi-ip>
```

### Useful Makefile targets

**On the Mac:**

| Target | Description |
|--------|-------------|
| `make on-the-mac-rust-check` | Typecheck without building a binary |
| `make on-the-mac-rust-build` | Cross-compile for Pi |
| `make on-the-mac-rust-deploy` | Build + copy binary to Pi |
| `make on-the-mac-rust-deploy-cookie` | Deploy via `~/.ssh/config` host (`cookie`) |
| `make on-the-mac-rust-daemon` | Deploy and start daemon on Pi |
| `make on-the-mac-run-with-rust` | Deploy, start daemon, and run web server |

**On the Pi:**

| Target | Description |
|--------|-------------|
| `make on-the-pi-rust-check` | Typecheck without building a binary |
| `make on-the-pi-rust-build` | Native release build |
| `make on-the-pi-rust-daemon` | Install systemd unit and start daemon |
| `make on-the-pi-rust-daemon-status` | Show daemon systemd status |

Override the Pi host when deploying from your Mac:

```bash
make on-the-mac-rust-deploy PI_HOST=orangepi@10.0.0.5
```

Or use an SSH config host alias (e.g. `Host cookie` in `~/.ssh/config`):

```bash
make on-the-mac-rust-deploy-cookie
```

`on-the-mac-rust-deploy` cross-compiles on your Mac and needs [cross](https://github.com/cross-rs/cross) plus Docker Desktop. If those are not set up, build on the Pi over SSH instead:

```bash
make on-the-mac-rust-deploy-cookie-remote
```

Socket path (default `/tmp/cookie-finder.sock`) can be overridden with the `COOKIE_FINDER_SOCKET` environment variable.

## Serial Console (USB-TTL, no WiFi)

When WiFi is unreliable, connect over UART0 with a 3.3V USB-TTL cable (Pi pin 8 TX → adapter RX, pin 10 RX → adapter TX, pin 9 GND → GND).

One-time setup on your Mac:

```bash
cp .serial.env.example .serial.env   # set SERIAL_DEVICE and SERIAL_PASSWORD
```

| Target | Description |
|--------|-------------|
| `make on-the-mac-serial-connect` | Interactive terminal (`screen`) |
| `make on-the-mac-serial-list` | List `/dev/tty.usbserial*` devices |
| `make on-the-mac-serial-run SERIAL_CMD='...'` | Run one command on the Pi |
| `make on-the-mac-serial-deploy` | Sync project over serial (no network needed) |
| `make on-the-mac-serial-deploy-rust` | Cross-compile and copy Rust binary over serial |

Example:

```bash
make on-the-mac-serial-connect
make on-the-mac-serial-run SERIAL_CMD='cd ~/cookie-finder && git status'
make on-the-mac-serial-deploy
```

Legacy `make serial-connect`, `make serial-deploy`, etc. still work.
