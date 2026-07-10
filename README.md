# Cookie Finder

Real-time USB thermal camera system for detecting heat signatures using an Orange Pi Zero 2W.

**[Full documentation →](docs/index.md)**

## Quick Start (Web Server)

```bash
make install
make run
```

Open `http://<device-ip>:8000` in a browser.

## Rust Gimbal Daemon (Optional, Recommended on Pi)

The gimbal and gamepad control loop can run in a Rust daemon for lower latency. The Python web server connects to it over a Unix socket and falls back to Python GPIO if the daemon is not running.

### Mac: cross-compile for Orange Pi

Prerequisites (one-time):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install cross
# Docker Desktop must be running
```

Build and deploy:

```bash
make rust-build-mac              # produces aarch64 Linux binary
make rust-deploy PI_HOST=orangepi@<pi-ip>
```

### Orange Pi: build natively

Prerequisites (one-time on the Pi):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build:

```bash
make rust-build-pi
```

Or build remotely from your Mac:

```bash
make rust-build-pi-remote PI_HOST=orangepi@<pi-ip>
```

### Run on the Pi

Start the daemon (requires `sudo` for GPIO):

```bash
sudo ./cookie_finder_rust/target/release/cookie-finder-ctl daemon
```

Then start the web server in another terminal:

```bash
make run
```

Or do both from your Mac in one step:

```bash
make run-with-rust PI_HOST=orangepi@<pi-ip>
```

### Useful Makefile targets

| Target | Description |
|--------|-------------|
| `make rust-help` | List all Rust targets |
| `make rust-check` | Typecheck without building a binary |
| `make rust-build-mac` | Cross-compile for Pi from Mac |
| `make rust-build-pi` | Native build on Orange Pi |
| `make rust-deploy` | Copy cross-built binary to Pi |
| `make rust-daemon` | Deploy and start daemon on Pi |
| `make run-with-rust` | Deploy, start daemon, and run web server |

Override the Pi host when deploying:

```bash
make rust-deploy PI_HOST=orangepi@10.0.0.5
```

Socket path (default `/tmp/cookie-finder.sock`) can be overridden with the `COOKIE_FINDER_SOCKET` environment variable.
