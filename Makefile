.PHONY: help on-the-mac-help on-the-pi-help

.DEFAULT_GOAL := help

# USB-TTL serial console (copy .serial.env.example to .serial.env)
-include .serial.env
export SERIAL_DEVICE SERIAL_BAUD SERIAL_USER SERIAL_PASSWORD SERIAL_REMOTE_DIR
SERIAL_DEVICE     ?= /dev/tty.usbserial-BG01PPKN
SERIAL_BAUD       ?= 115200
SERIAL_USER       ?= cookie
SERIAL_REMOTE_DIR ?= ~/cookie-finder

# Client WiFi profiles (copy .wifi.env.example to .wifi.env — never commit PSKs)
-include .wifi.env
export WIFI_HOME_SSID WIFI_HOME_PSK WIFI_HOME_PRIORITY
export WIFI_HOTSPOT_SSID WIFI_HOTSPOT_PSK WIFI_HOTSPOT_PRIORITY
WIFI_HOME_SSID       ?= HSH-5G
WIFI_HOME_PRIORITY   ?= 100
WIFI_HOTSPOT_SSID    ?= Ghostwire
WIFI_HOTSPOT_PRIORITY ?= 50

# --- Rust gimbal daemon (Orange Pi Zero 2W, aarch64) ---
export PATH := $(HOME)/.cargo/bin:$(PATH)
RUST_DIR     := cookie_finder_rust
RUST_BIN     := cookie-finder-ctl
RUST_TARGET  := aarch64-unknown-linux-gnu
RUST_CROSS   := $(RUST_DIR)/target/$(RUST_TARGET)/release/$(RUST_BIN)
RUST_NATIVE  := $(RUST_DIR)/target/release/$(RUST_BIN)
PI_HOST      ?= cookie@192.168.68.106
PI_SSH_HOST  ?= cookie
PI_DEST      ?= ~/cookie-finder/$(RUST_BIN)
RUST_SOCKET  ?= /tmp/cookie-finder.sock
SYSTEMD_UNIT_IN := systemd/cookie-finder.service.in
SYSTEMD_UNIT    := /etc/systemd/system/cookie-finder.service
WIFI_SYSTEMD_UNIT_IN := systemd/cookie-finder-wifi.service.in
WIFI_SYSTEMD_UNIT    := /etc/systemd/system/cookie-finder-wifi.service
WIFI_PYTHON     ?= $(CURDIR)/.venv/bin/python
WEB_SYSTEMD_UNIT_IN := systemd/cookie-finder-web.service.in
WEB_SYSTEMD_UNIT    := /etc/systemd/system/cookie-finder-web.service
WEB_PYTHON      ?= $(CURDIR)/.venv/bin/python
WEB_HOST        ?= 0.0.0.0
WEB_PORT        ?= 80
WEB_HTTPS_PORT  ?= 443
TLS_DIR         ?= /var/lib/cookie-finder/tls
MDNS_HOSTNAME   ?= cookie-finder
AVAHI_SERVICE_IN := systemd/cookie-finder.avahi.service
AVAHI_SERVICE    := /etc/avahi/services/cookie-finder.service

help:
	@echo "Cookie Finder – Makefile Targets"
	@echo ""
	@echo "Run targets on the machine where they belong:"
	@echo "  make on-the-mac-help    MacBook (cross-compile, deploy, serial, dev tools)"
	@echo "  make on-the-pi-help     Orange Pi (run app, hardware tests, native build)"
	@echo ""
	@echo "Legacy names (install, run, rust-*, serial-*, etc.) still work as aliases."

# =============================================================================
# Internal recipes (do not call directly — use on-the-mac-* or on-the-pi-*)
# =============================================================================

.PHONY: _install _install-yolo _install-docs _docs _clean _init _init-wifi \
        _run-standalone _run-web _run-web-custom \
        _test-motors _test-motors-pan-cw _test-motors-pan-ccw _test-motors-tilt-cw \
        _test-motors-tilt-ccw _test-motors-home _test-bluetooth-input _test-gimbal-gamepad \
        _test-pan-step _test-all-gpio _keyboard-gimbal \
        _find-camera _list-devices _list-controls _get-control _set-control \
        _install-ffmpeg _install-libusb _list-cameras _list-camera-formats \
        _probe-install _probe-usb _probe-cdc _probe-serial _probe-commands _probe-resolution _probe-xu _probe \
        _serial-help _serial-list _serial-connect _serial-run _serial-deploy _serial-deploy-rust \
        _rust-install-rustup _rust-install-cross-toolchain _rust-check

_install:
	uv sync

_install-yolo:
	uv sync --extra yolo

_install-docs:
	uv sync --extra docs

_docs: _install-docs
	@echo "Starting MkDocs dev server at http://127.0.0.1:8001..."
	uv run mkdocs serve --dev-addr 127.0.0.1:8001

_clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

_init:
	@echo "Initializing system permissions for Bluetooth..."
	sudo usermod -aG bluetooth cookie
	@echo "Bluetooth group permissions added. Please log out and log back in for changes to take effect."

_init-wifi:
	@echo "Installing WiFi AP dependencies (hostapd, dnsmasq, iw)..."
	sudo apt-get update
	sudo apt-get install -y hostapd dnsmasq iw iwconfig wireless-tools network-manager || true
	@if ! command -v create_ap >/dev/null 2>&1; then \
		echo "Installing create_ap helper script..."; \
		sudo curl -fsSL https://raw.githubusercontent.com/oblique/create_ap/master/create_ap \
			-o /usr/local/bin/create_ap; \
		sudo chmod +x /usr/local/bin/create_ap; \
	fi
	@SCRIPT_PATH="$$(cd "$(CURDIR)" && pwd)/scripts/wifi-mode.sh"; \
	echo "Configuring passwordless sudo for $$SCRIPT_PATH..."; \
	echo "cookie ALL=(root) NOPASSWD: $$SCRIPT_PATH" | sudo tee /etc/sudoers.d/cookie-finder-wifi >/dev/null; \
	sudo chmod 440 /etc/sudoers.d/cookie-finder-wifi; \
	sudo visudo -cf /etc/sudoers.d/cookie-finder-wifi
	@POWER_SCRIPT="$$(cd "$(CURDIR)" && pwd)/scripts/system-power.sh"; \
	echo "Configuring passwordless sudo for $$POWER_SCRIPT..."; \
	echo "cookie ALL=(root) NOPASSWD: $$POWER_SCRIPT" | sudo tee /etc/sudoers.d/cookie-finder-poweroff >/dev/null; \
	sudo chmod 440 /etc/sudoers.d/cookie-finder-poweroff; \
	sudo visudo -cf /etc/sudoers.d/cookie-finder-poweroff
	@echo "Installing NetworkManager captive-DNS drop-in (AP SoftAP only)..."
	@sudo mkdir -p /etc/NetworkManager/dnsmasq-shared.d
	@printf '%s\n' \
		'# Cookie Finder captive portal — sinkhole all DNS to SoftAP gateway.' \
		'# Only used by NetworkManager "shared" (AP) connections.' \
		'address=/#/192.168.12.1' \
		| sudo tee /etc/NetworkManager/dnsmasq-shared.d/cookie-finder-captive.conf >/dev/null
	@if [ -x "$(WIFI_PYTHON)" ]; then \
		$(MAKE) on-the-pi-wifi-gpio-daemon-install; \
		sudo systemctl restart cookie-finder-wifi.service; \
	else \
		echo "Note: .venv not found yet — after 'make install', run:"; \
		echo "  make on-the-pi-wifi-gpio-daemon"; \
	fi
	@echo "WiFi AP setup complete. SSID will be cookie-finder (open network, no password)."
	@echo "Captive portal: guests are redirected to http://192.168.12.1/ (web app)."
	@echo "Toggle AP/client via Settings or a single GPIO button press (LED on pin 29)."
	@echo "Triple-click the button, or Settings → Shut down, to power off."
	@echo "Button service: sudo systemctl status cookie-finder-wifi"

_run-standalone:
	@echo "Starting Thermal Camera Viewer (Standalone GUI mode)..."
	uv run main.py

_run-web:
	@echo "Starting Thermal Camera Viewer (WebServer mode on http://0.0.0.0:$(WEB_PORT) + https://0.0.0.0:$(WEB_HTTPS_PORT))..."
	uv run main.py --web --host $(WEB_HOST) --port $(WEB_PORT) --https-port $(WEB_HTTPS_PORT)

_run-web-custom:
	@read -p "Enter HTTP port (default $(WEB_PORT)): " port; \
	read -p "Enter HTTPS port (default $(WEB_HTTPS_PORT), 0=off): " https_port; \
	read -p "Enter host (default 0.0.0.0): " host; \
	port=$${port:-$(WEB_PORT)}; \
	https_port=$${https_port:-$(WEB_HTTPS_PORT)}; \
	host=$${host:-0.0.0.0}; \
	echo "Starting Thermal Camera Viewer (WebServer mode on http://$$host:$$port https://$$host:$$https_port)..."; \
	uv run main.py --web --port $$port --https-port $$https_port --host $$host

_test-motors:
	@echo "Motor control test script:"
	@echo "  sudo make on-the-pi-test-motors auto           # Automated test sequence"
	@echo "  sudo make on-the-pi-test-motors-pan-cw         # Pan clockwise 50 steps"
	@echo "  sudo make on-the-pi-test-motors-pan-ccw        # Pan counter-clockwise 50 steps"
	@echo "  sudo make on-the-pi-test-motors-tilt-cw        # Tilt clockwise 50 steps"
	@echo "  sudo make on-the-pi-test-motors-tilt-ccw       # Tilt counter-clockwise 50 steps"
	@echo "  sudo make on-the-pi-test-motors-home           # Home both motors"
	@sudo uv run tools/test_motors.py auto

_test-motors-pan-cw:
	@sudo uv run tools/test_motors.py pan-cw

_test-motors-pan-ccw:
	@sudo uv run tools/test_motors.py pan-ccw

_test-motors-tilt-cw:
	@sudo uv run tools/test_motors.py tilt-cw

_test-motors-tilt-ccw:
	@sudo uv run tools/test_motors.py tilt-ccw

_test-motors-home:
	@sudo uv run tools/test_motors.py home

_test-bluetooth-input:
	@echo "Bluetooth input test: reads and logs all gamepad input (60 seconds)..."
	@uv run tools/test_bluetooth_input.py

_test-gimbal-gamepad:
	@echo "Gimbal + Gamepad test: control gimbal with joystick input..."
	@uv run tools/test_gimbal_gamepad.py

_keyboard-gimbal:
	@echo "Keyboard gimbal: arrows pan/tilt, 1-9 speed, M drive mode, P/T [ ] wiring, W save, q quit..."
	@uv run tools/keyboard_gimbal.py

_test-pan-step:
	@echo "Freeing GPIO pins..."
	@bash -c '\
		for g in 258 268 271 272; do \
			echo $$g > /sys/class/gpio/unexport 2>/dev/null || true; \
		done; \
		echo "Stepping pan motor (custom pin set)..."; \
		for i in $$(seq 1 20); do \
			sudo gpioset gpiochip1 258=1 268=0 271=0 272=0; sleep 0.1; \
			sudo gpioset gpiochip1 258=0 268=1 271=0 272=0; sleep 0.1; \
			sudo gpioset gpiochip1 258=0 268=0 271=1 272=0; sleep 0.1; \
			sudo gpioset gpiochip1 258=0 268=0 271=0 272=1; sleep 0.1; \
		done; \
		echo "Done."; \
	'

_test-all-gpio:
	@echo "Scanning and blinking all available GPIO pins..."
	@bash -c '\
	for chip in /dev/gpiochip*; do \
		chipname=$$(basename $$chip); \
		echo "---- Testing $$chipname ----"; \
		lines=$$(gpioinfo $$chipname | grep "line" | wc -l); \
		for ((i=0; i<lines; i++)); do \
			sudo gpioset $$chipname $$i=1 2>/dev/null && \
			sleep 0.05 && \
			sudo gpioset $$chipname $$i=0 2>/dev/null && \
			echo "  Toggled $$chipname line $$i"; \
		done; \
	done; \
	echo "Done scanning all GPIO."; \
	'

_find-camera:
	@echo "Detecting available camera devices..."
	@echo "Checking /dev/video devices:"
	@ls -la /dev/video* 2>/dev/null || echo "No /dev/video devices found"
	@echo ""
	@echo "Camera details:"
	@v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
	@echo ""
	@uv run tools/find_camera.py

_list-devices:
	uv run tools/uvc_controls.py list-devices

_list-controls:
	uv run tools/uvc_controls.py list-controls

_get-control:
	@read -p "Enter control name: " control; \
	uv run tools/uvc_controls.py get $$control

_set-control:
	@read -p "Enter control name: " control; \
	read -p "Enter value: " value; \
	uv run tools/uvc_controls.py set $$control $$value

_install-ffmpeg:
	brew install ffmpeg

_install-libusb:
	brew install libusb

_list-cameras:
	ffmpeg -f avfoundation -list_devices true -i ""

_list-camera-formats:
	ffmpeg -f avfoundation -video_size 512x390 -framerate 50 -i "0" -vframes 1 thermal_capture.tiff

_probe-install:
	brew install libusb

_probe-usb: _probe-install
	uv run tools/probing_thermal_camera/probe_usb.py

_probe-cdc:
	uv run tools/probing_thermal_camera/probe_cdc.py

_probe-serial:
	uv run tools/probing_thermal_camera/probe_serial.py

_probe-commands:
	uv run tools/probing_thermal_camera/probe_commands.py

_probe-resolution:
	uv run tools/probing_thermal_camera/probe_resolution.py

_probe-xu:
	uv run tools/probing_thermal_camera/probe_uvc_xu.py

_probe: _probe-usb _probe-cdc _probe-serial _probe-commands _probe-resolution _probe-xu

_serial-help:
	@echo "Serial targets (USB-TTL UART when WiFi is down):"
	@echo "  make on-the-mac-serial-list                         list /dev/tty.usbserial* devices"
	@echo "  make on-the-mac-serial-connect                      screen session (SERIAL_DEVICE=$(SERIAL_DEVICE))"
	@echo "  make on-the-mac-serial-run SERIAL_CMD='git status'  run one remote command"
	@echo "  make on-the-mac-serial-deploy                       tarball + base64 sync + uv sync on Pi"
	@echo "  make on-the-mac-serial-deploy-rust                  cross-compile and copy Rust binary"
	@echo ""
	@echo "Configure in .serial.env (see .serial.env.example):"
	@echo "  SERIAL_DEVICE=$(SERIAL_DEVICE)"
	@echo "  SERIAL_BAUD=$(SERIAL_BAUD)"
	@echo "  SERIAL_USER=$(SERIAL_USER)"
	@echo "  SERIAL_PASSWORD=<set in .serial.env>"
	@echo "  SERIAL_REMOTE_DIR=$(SERIAL_REMOTE_DIR)"

_serial-list:
	@ls /dev/tty.usbserial* /dev/cu.usbserial* 2>/dev/null || echo "No USB serial devices found"

_serial-connect:
	@command -v screen >/dev/null || (echo "Install screen: brew install screen" && exit 1)
	@test -e $(SERIAL_DEVICE) || (echo "Serial device not found: $(SERIAL_DEVICE)" && $(MAKE) on-the-mac-serial-list && exit 1)
	@echo "Opening serial console: $(SERIAL_DEVICE) @ $(SERIAL_BAUD) (detach: Ctrl-A then D)"
	@TERM=xterm-256color screen $(SERIAL_DEVICE) $(SERIAL_BAUD)

_serial-run:
	@test -n "$(SERIAL_CMD)" || (echo "Usage: make on-the-mac-serial-run SERIAL_CMD='your command'" && exit 1)
	uv run tools/pi_serial.py run $(SERIAL_CMD)

_serial-deploy:
	uv run tools/pi_serial.py deploy

_serial-deploy-rust: on-the-mac-rust-build
	uv run tools/pi_serial.py deploy-file $(RUST_CROSS) $(PI_DEST)

_rust-install-rustup:
	@if command -v cargo >/dev/null; then \
		echo "cargo already installed: $$(cargo --version)"; \
	else \
		echo "Installing Rust via rustup..."; \
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; \
		echo ""; \
		echo "Rust installed. cargo is at ~/.cargo/bin (Makefile targets pick this up automatically)."; \
	fi

_rust-check:
	cd $(RUST_DIR) && cargo check

# cross on macOS needs a Linux host toolchain for metadata before it runs Docker.
# crates.io cross 0.2.5 omits --force-non-host; install it explicitly as a fallback.
_rust-install-cross-toolchain:
	@rustup toolchain list | grep -q 'stable-x86_64-unknown-linux-gnu' || \
		rustup toolchain install stable-x86_64-unknown-linux-gnu --force-non-host --profile minimal

# =============================================================================
# On the Mac (run from your MacBook)
# =============================================================================

.PHONY: on-the-mac-install on-the-mac-install-yolo on-the-mac-install-docs on-the-mac-docs \
        on-the-mac-install-ffmpeg on-the-mac-install-libusb on-the-mac-clean \
        on-the-mac-list-cameras on-the-mac-list-camera-formats \
        on-the-mac-probe on-the-mac-probe-install on-the-mac-probe-usb on-the-mac-probe-cdc \
        on-the-mac-probe-serial on-the-mac-probe-commands on-the-mac-probe-resolution on-the-mac-probe-xu \
        on-the-mac-serial-help on-the-mac-serial-list on-the-mac-serial-connect \
        on-the-mac-serial-run on-the-mac-serial-deploy on-the-mac-serial-deploy-rust \
        on-the-mac-tool-setup on-the-mac-tool-setup-rust \
        on-the-mac-rust-check on-the-mac-rust-build on-the-mac-rust-deploy \
        on-the-mac-rust-deploy-cookie on-the-mac-rust-build-remote \
        on-the-mac-rust-deploy-remote on-the-mac-rust-deploy-cookie-remote \
        on-the-mac-rust-daemon on-the-mac-rust-run on-the-mac-rust-home on-the-mac-run-with-rust

on-the-mac-help:
	@echo "On-the-Mac targets (run from MacBook):"
	@echo ""
	@echo "Installation:"
	@echo "  make on-the-mac-install            uv sync — core dependencies"
	@echo "  make on-the-mac-install-yolo       uv sync with YOLO/PyTorch extras"
	@echo "  make on-the-mac-install-docs       uv sync with MkDocs extras"
	@echo "  make on-the-mac-install-ffmpeg     brew install ffmpeg"
	@echo "  make on-the-mac-install-libusb     brew install libusb"
	@echo ""
	@echo "Documentation:"
	@echo "  make on-the-mac-docs               MkDocs dev server (http://127.0.0.1:8001)"
	@echo ""
	@echo "Camera (macOS / avfoundation):"
	@echo "  make on-the-mac-list-cameras       List cameras via ffmpeg"
	@echo "  make on-the-mac-list-camera-formats  Capture sample thermal TIFF"
	@echo ""
	@echo "Camera probing (debug, USB camera on Mac):"
	@echo "  make on-the-mac-probe              Run all probe scripts"
	@echo "  make on-the-mac-probe-install      brew install libusb"
	@echo "  make on-the-mac-probe-usb          Probe USB descriptors"
	@echo "  make on-the-mac-probe-cdc          Probe CDC interface"
	@echo "  make on-the-mac-probe-serial       Probe serial interface"
	@echo "  make on-the-mac-probe-commands     Probe text-style serial commands (NUC/palette)"
	@echo "  make on-the-mac-probe-resolution   Probe supported resolutions"
	@echo "  make on-the-mac-probe-xu           Probe UVC extension units"
	@echo ""
	@echo "Serial console (USB-TTL UART, no WiFi):"
	@echo "  make on-the-mac-serial-help        List serial targets and config"
	@echo "  make on-the-mac-serial-list        List /dev/tty.usbserial* devices"
	@echo "  make on-the-mac-serial-connect     Open interactive screen session"
	@echo "  make on-the-mac-serial-run         Run one command on Pi (SERIAL_CMD=...)"
	@echo "  make on-the-mac-serial-deploy      Sync project to Pi over serial"
	@echo "  make on-the-mac-serial-deploy-rust Cross-compile and deploy Rust binary"
	@echo ""
	@echo "Rust gimbal daemon:"
	@echo "  make on-the-mac-tool-setup                 Install Rust/cargo + cross (Docker required)"
	@echo "  make on-the-mac-tool-setup-rust            Install Rust/cargo only"
	@echo "  make on-the-mac-rust-check                 Typecheck without building"
	@echo "  make on-the-mac-rust-build                 Cross-compile for Pi (needs cross + Docker)"
	@echo "  make on-the-mac-rust-deploy                Build + scp to Pi (PI_HOST=$(PI_HOST))"
	@echo "  make on-the-mac-rust-deploy-cookie         <-- Normal Deployment -- Build + scp via SSH config (PI_SSH_HOST=$(PI_SSH_HOST))"
	@echo "  make on-the-mac-rust-build-remote          Build on Pi via SSH (PI_HOST=$(PI_HOST))"
	@echo "  make on-the-mac-rust-deploy-remote         Build on Pi via SSH + install binary"
	@echo "  make on-the-mac-rust-deploy-cookie-remote  Build on Pi via SSH config host"
	@echo "  make on-the-mac-rust-daemon                Deploy + start foreground daemon on Pi (via SSH)"
	@echo "  make on-the-mac-rust-run                   Deploy + run one-shot command on Pi"
	@echo "  make on-the-mac-rust-home                  Send home command to daemon on Pi"
	@echo "  make on-the-mac-run-with-rust              Deploy + daemon + web server on Pi"
	@echo "  (Pi systemd daemon: make on-the-pi-rust-daemon — not the rust-daemon alias)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make on-the-mac-clean              Remove Python cache files"

on-the-mac-install: _install
on-the-mac-install-yolo: _install-yolo
on-the-mac-install-docs: _install-docs
on-the-mac-docs: _docs
on-the-mac-clean: _clean
on-the-mac-install-ffmpeg: _install-ffmpeg
on-the-mac-install-libusb: _install-libusb
on-the-mac-list-cameras: _list-cameras
on-the-mac-list-camera-formats: _list-camera-formats
on-the-mac-probe: _probe
on-the-mac-probe-install: _probe-install
on-the-mac-probe-usb: _probe-usb
on-the-mac-probe-cdc: _probe-cdc
on-the-mac-probe-serial: _probe-serial
on-the-mac-probe-commands: _probe-commands
on-the-mac-probe-resolution: _probe-resolution
on-the-mac-probe-xu: _probe-xu
on-the-mac-serial-help: _serial-help
on-the-mac-serial-list: _serial-list
on-the-mac-serial-connect: _serial-connect
on-the-mac-serial-run: _serial-run
on-the-mac-serial-deploy: _serial-deploy
on-the-mac-serial-deploy-rust: _serial-deploy-rust

on-the-mac-tool-setup: on-the-mac-tool-setup-rust _rust-install-cross-toolchain
	@if command -v docker >/dev/null && docker info >/dev/null 2>&1; then \
		echo "Docker OK: $$(docker --version)"; \
	else \
		echo "ERROR: Docker Desktop must be installed and running for on-the-mac-rust-build."; \
		exit 1; \
	fi
	@if command -v cross >/dev/null; then \
		echo "cross already installed: $$(cross --version)"; \
	else \
		echo "Installing cross from cross-rs main (crates.io 0.2.5 breaks on macOS)..."; \
		cargo install cross --git https://github.com/cross-rs/cross; \
	fi

on-the-mac-tool-setup-rust: _rust-install-rustup

on-the-mac-rust-check: _rust-check

on-the-mac-rust-build:
	cd $(RUST_DIR) && cross build --release --target $(RUST_TARGET)

on-the-mac-rust-deploy: on-the-mac-rust-build
	scp $(RUST_CROSS) $(PI_HOST):$(PI_DEST)

on-the-mac-rust-deploy-cookie: on-the-mac-rust-build
	scp $(RUST_CROSS) $(PI_SSH_HOST):$(PI_DEST)

on-the-mac-rust-build-remote:
	ssh $(PI_HOST) "cd ~/cookie-finder/$(RUST_DIR) && cargo build --release"

on-the-mac-rust-deploy-remote:
	ssh $(PI_HOST) "cd ~/cookie-finder/$(RUST_DIR) && cargo build --release && install -m 755 target/release/$(RUST_BIN) $(PI_DEST)"

on-the-mac-rust-deploy-cookie-remote:
	ssh $(PI_SSH_HOST) "cd ~/cookie-finder/$(RUST_DIR) && cargo build --release && install -m 755 target/release/$(RUST_BIN) $(PI_DEST)"

on-the-mac-rust-daemon: on-the-mac-rust-deploy
	ssh -t $(PI_HOST) "sudo $(PI_DEST) daemon --socket $(RUST_SOCKET)"

on-the-mac-rust-run: on-the-mac-rust-deploy
	ssh -t $(PI_HOST) "sudo $(PI_DEST) run"

on-the-mac-rust-home:
	ssh -t $(PI_HOST) "sudo $(PI_DEST) home --socket $(RUST_SOCKET)"

on-the-mac-run-with-rust: on-the-mac-rust-deploy
	ssh -t $(PI_HOST) "sudo $(PI_DEST) daemon --socket $(RUST_SOCKET) & sleep 1; cd ~/cookie-finder && uv run main.py --web"

# =============================================================================
# On the Pi (run on Orange Pi)
# =============================================================================

.PHONY: on-the-pi-install on-the-pi-install-yolo on-the-pi-init on-the-pi-clean \
        on-the-pi-run on-the-pi-run-standalone on-the-pi-run-web on-the-pi-run-web-custom \
        on-the-pi-test-motors on-the-pi-test-motors-pan-cw on-the-pi-test-motors-pan-ccw \
        on-the-pi-test-motors-tilt-cw on-the-pi-test-motors-tilt-ccw on-the-pi-test-motors-home \
        on-the-pi-test-bluetooth-input on-the-pi-test-gimbal-gamepad \
        on-the-pi-test-pan-step on-the-pi-test-all-gpio \
        on-the-pi-find-camera on-the-pi-list-devices on-the-pi-list-controls \
        on-the-pi-get-control on-the-pi-set-control \
        on-the-pi-tool-setup on-the-pi-tool-setup-rust \
        on-the-pi-rust-check on-the-pi-rust-build \
        on-the-pi-rust-daemon-install on-the-pi-rust-daemon \
        on-the-pi-rust-daemon-stop on-the-pi-rust-daemon-status \
        on-the-pi-rust-keyboard on-the-pi-init-wifi \
        on-the-pi-wifi-gpio-daemon-install on-the-pi-wifi-gpio-daemon \
        on-the-pi-wifi-gpio-daemon-stop on-the-pi-wifi-gpio-daemon-status \
        on-the-pi-web-daemon-install on-the-pi-web-daemon \
        on-the-pi-web-daemon-stop on-the-pi-web-daemon-status \
        on-the-pi-web-url on-the-pi-mdns \
        on-the-pi-wifi-configure-clients on-the-pi-wifi-fix \
        on-the-pi-armbian-home-screen on-the-pi-wifi-status

on-the-pi-help:
	@echo "On-the-Pi targets (run on Orange Pi only):"
	@echo ""
	@echo "Installation:"
	@echo "  make on-the-pi-install             uv sync — core dependencies"
	@echo "  make on-the-pi-install-yolo        uv sync with YOLO/PyTorch extras"
	@echo "  make on-the-pi-init                  Bluetooth group permissions"
	@echo "  make on-the-pi-tool-setup            apt build deps + Rust/cargo (rustup)"
	@echo "  make on-the-pi-tool-setup-rust       Rust/cargo only (skip apt packages)"
	@echo "  make on-the-pi-init-wifi              Install WiFi AP deps + captive DNS + button/LED service"
	@echo "  make on-the-pi-wifi-gpio-daemon       Install/start WiFi button+LED service"
	@echo "  make on-the-pi-wifi-gpio-daemon-status  Show WiFi button+LED service status"
	@echo "  make on-the-pi-wifi-gpio-daemon-stop  Stop WiFi button+LED service"
	@echo "  make on-the-pi-wifi-status            Show WiFi mode + IP addresses"
	@echo "  make on-the-pi-wifi-configure-clients Save NM client profiles from .wifi.env"
	@echo "  make on-the-pi-wifi-fix              Recover wedged client WiFi (NM + wpa fallback)"
	@echo "  make on-the-pi-mdns                   Set hostname + Avahi for cookie-finder.local"
	@echo ""
	@echo "Running the application:"
	@echo "  make on-the-pi-run                   Start web server (foreground)"
	@echo "  make on-the-pi-run-standalone        Start standalone GUI mode"
	@echo "  make on-the-pi-run-web               Start web server (http://:80 + https://:443)"
	@echo "  make on-the-pi-run-web-custom        Prompt for host + ports"
	@echo "  make on-the-pi-web-daemon            Install/start web server systemd service"
	@echo "  make on-the-pi-web-daemon-status     Show web server service status"
	@echo "  make on-the-pi-web-daemon-stop       Stop web server service"
	@echo "  make on-the-pi-web-url               Print device IP(s) + web URL/port"
	@echo ""
	@echo "Camera:"
	@echo "  make on-the-pi-find-camera           Detect available camera devices"
	@echo "  make on-the-pi-list-devices          List UVC device controls"
	@echo "  make on-the-pi-list-controls         List camera control names"
	@echo "  make on-the-pi-get-control           Get camera control value (interactive)"
	@echo "  make on-the-pi-set-control           Set camera control value (interactive)"
	@echo ""
	@echo "Hardware tests:"
	@echo "  make on-the-pi-test-motors           Motor control test (auto sequence)"
	@echo "  make on-the-pi-test-motors-pan-cw    Pan clockwise 50 steps"
	@echo "  make on-the-pi-test-motors-pan-ccw   Pan counter-clockwise 50 steps"
	@echo "  make on-the-pi-test-motors-tilt-cw   Tilt clockwise 50 steps"
	@echo "  make on-the-pi-test-motors-tilt-ccw  Tilt counter-clockwise 50 steps"
	@echo "  make on-the-pi-test-motors-home      Home both motors"
	@echo "  make on-the-pi-test-bluetooth-input  Test Bluetooth gamepad input"
	@echo "  make on-the-pi-test-gimbal-gamepad   Control gimbal with joystick"
	@echo "  make on-the-pi-test-pan-step         Manual pan motor stepping"
	@echo "  make on-the-pi-test-all-gpio         Scan and test all GPIO pins"
	@echo ""
	@echo "Rust gimbal daemon:"
	@echo "  make on-the-pi-rust-check            Typecheck without building"
	@echo "  make on-the-pi-rust-build            Native release build"
	@echo "  make on-the-pi-rust-daemon-install   Install systemd unit (cookie-finder.service)"
	@echo "  make on-the-pi-rust-daemon           Install unit (if needed) and start via systemd"
	@echo "  make on-the-pi-rust-daemon-stop      Stop systemd service"
	@echo "  make on-the-pi-rust-daemon-status    Show systemd status"
	@echo "  make on-the-pi-rust-keyboard         Keyboard pan/tilt + drive mode (M) + wiring ([ ] / W)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make on-the-pi-armbian-home-screen   Print Armbian MOTD / home screen"
	@echo "  make on-the-pi-clean                 Remove Python cache files"

on-the-pi-install: _install
on-the-pi-install-yolo: _install-yolo
on-the-pi-init: _init
on-the-pi-init-wifi: _init-wifi
on-the-pi-clean: _clean
on-the-pi-armbian-home-screen:
	@sudo run-parts /etc/update-motd.d/
on-the-pi-wifi-status:
	@"$(CURDIR)/scripts/wifi-mode.sh" status
	@echo "---"
	@ip -4 -br addr show 2>/dev/null || hostname -I

# Save preferred client WiFi profiles for NetworkManager restore.
# Credentials come from .wifi.env (see .wifi.env.example) — never hardcode PSKs.
# Does not require both SSIDs to be in range — association is attempted after save.
on-the-pi-wifi-configure-clients:
	@command -v nmcli >/dev/null 2>&1 || { echo "nmcli not found (install network-manager)"; exit 1; }
	@if [ -z "$(WIFI_HOME_PSK)" ] || [ -z "$(WIFI_HOTSPOT_PSK)" ]; then \
		echo "error: WiFi PSKs not set."; \
		echo "Copy .wifi.env.example to .wifi.env and set WIFI_HOME_PSK / WIFI_HOTSPOT_PSK."; \
		exit 1; \
	fi
	@echo "Saving WiFi client profiles ($$WIFI_HOME_SSID pri $$WIFI_HOME_PRIORITY, $$WIFI_HOTSPOT_SSID pri $$WIFI_HOTSPOT_PRIORITY)..."
	@nmcli connection delete id "$(WIFI_HOME_SSID)" >/dev/null 2>&1 || true
	@nmcli connection add type wifi con-name "$(WIFI_HOME_SSID)" ifname "*" ssid "$(WIFI_HOME_SSID)" \
		wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$(WIFI_HOME_PSK)" \
		connection.autoconnect yes connection.autoconnect-priority $(WIFI_HOME_PRIORITY)
	@nmcli connection delete id "$(WIFI_HOTSPOT_SSID)" >/dev/null 2>&1 || true
	@nmcli connection add type wifi con-name "$(WIFI_HOTSPOT_SSID)" ifname "*" \
		ssid "$(WIFI_HOTSPOT_SSID)" \
		wifi-sec.key-mgmt wpa-psk wifi-sec.psk "$(WIFI_HOTSPOT_PSK)" \
		connection.autoconnect yes connection.autoconnect-priority $(WIFI_HOTSPOT_PRIORITY)
	@echo "Bringing up preferred network ($(WIFI_HOME_SSID), else $(WIFI_HOTSPOT_SSID))..."
	@nmcli -w 15 connection up "$(WIFI_HOME_SSID)" \
		|| nmcli -w 15 connection up "$(WIFI_HOTSPOT_SSID)" \
		|| echo "Profiles saved; neither network associated yet (out of range?)."
	@echo "---"
	@nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY connection show
	@$(MAKE) on-the-pi-wifi-status

# Recover wedged client WiFi after SoftAP / NM "unavailable" on UWE5622.
# Stops cookie-finder-wifi briefly so it cannot race the recovery.
# Requires saved NM profiles (make on-the-pi-wifi-configure-clients).
on-the-pi-wifi-fix:
	@echo "Recovering client WiFi (clears AP leftovers; NM then wpa_supplicant)..."
	@-sudo systemctl stop cookie-finder-wifi.service 2>/dev/null || true
	@sudo "$(CURDIR)/scripts/wifi-mode.sh" fix
	@-sudo systemctl start cookie-finder-wifi.service 2>/dev/null || true
	@$(MAKE) on-the-pi-wifi-status

on-the-pi-run: on-the-pi-run-web

on-the-pi-run-standalone: _run-standalone
on-the-pi-run-web: _run-web
on-the-pi-run-web-custom: _run-web-custom
on-the-pi-test-motors: _test-motors
on-the-pi-test-motors-pan-cw: _test-motors-pan-cw
on-the-pi-test-motors-pan-ccw: _test-motors-pan-ccw
on-the-pi-test-motors-tilt-cw: _test-motors-tilt-cw
on-the-pi-test-motors-tilt-ccw: _test-motors-tilt-ccw
on-the-pi-test-motors-home: _test-motors-home
on-the-pi-test-bluetooth-input: _test-bluetooth-input
on-the-pi-test-gimbal-gamepad: _test-gimbal-gamepad
on-the-pi-test-pan-step: _test-pan-step
on-the-pi-test-all-gpio: _test-all-gpio
on-the-pi-find-camera: _find-camera
on-the-pi-list-devices: _list-devices
on-the-pi-list-controls: _list-controls
on-the-pi-get-control: _get-control
on-the-pi-set-control: _set-control

on-the-pi-tool-setup: on-the-pi-tool-setup-rust
	@echo "Installing native build dependencies (build-essential, curl, pkg-config)..."
	@sudo apt update
	@sudo apt install -y build-essential curl pkg-config

on-the-pi-tool-setup-rust: _rust-install-rustup

on-the-pi-rust-check: _rust-check

on-the-pi-rust-build:
	cd $(RUST_DIR) && cargo build --release

# Install/refresh the systemd unit. Paths are baked from CURDIR (never ~ under sudo make).
on-the-pi-rust-daemon-install:
	@test -x "$(CURDIR)/$(RUST_BIN)" || { \
		echo "error: missing executable $(CURDIR)/$(RUST_BIN)"; \
		echo "hint: run 'make on-the-pi-rust-build' (or deploy a binary to that path)"; \
		exit 1; \
	}
	@sed \
		-e 's|@REPO_ROOT@|$(CURDIR)|g' \
		-e 's|@RUST_BIN@|$(RUST_BIN)|g' \
		-e 's|@RUST_SOCKET@|$(RUST_SOCKET)|g' \
		$(SYSTEMD_UNIT_IN) | sudo tee $(SYSTEMD_UNIT) >/dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable cookie-finder.service
	@echo "Installed $(SYSTEMD_UNIT)"
	@echo "  binary: $(CURDIR)/$(RUST_BIN)"
	@echo "  socket: $(RUST_SOCKET)"

# Stop any leftover nohup/manual daemon so GPIO pins are free, then start via systemd.
on-the-pi-rust-daemon: on-the-pi-rust-daemon-install
	@echo "Stopping any non-systemd cookie-finder-ctl daemon..."
	@-sudo pkill -x $(RUST_BIN) 2>/dev/null || true
	@sleep 0.5
	@sudo systemctl restart cookie-finder.service
	@sudo systemctl --no-pager --full status cookie-finder.service || true
	@echo ""
	@echo "Daemon managed by systemd (socket $(RUST_SOCKET))"
	@echo "Check status:  sudo systemctl status cookie-finder"
	@echo "Stop:          make on-the-pi-rust-daemon-stop"
	@echo "Restart:       sudo systemctl restart cookie-finder"
	@echo "Disable boot:  sudo systemctl disable cookie-finder"
	@echo "Follow logs:   sudo journalctl -u cookie-finder -f"

on-the-pi-rust-daemon-stop:
	@sudo systemctl stop cookie-finder.service
	@echo "Stopped cookie-finder.service"

on-the-pi-rust-daemon-status:
	@sudo systemctl --no-pager --full status cookie-finder.service

on-the-pi-rust-keyboard: _keyboard-gimbal

# Web server daemon (systemd) — listens on HTTP :80 and HTTPS :443
on-the-pi-web-daemon-install:
	@test -x "$(WEB_PYTHON)" || { \
		echo "error: missing $(WEB_PYTHON)"; \
		echo "hint: run 'make on-the-pi-install' first (creates .venv)"; \
		exit 1; \
	}
	@command -v qrencode >/dev/null 2>&1 || sudo apt-get install -y qrencode || true
	@command -v openssl >/dev/null 2>&1 || sudo apt-get install -y openssl || true
	@echo "Ensuring TLS certs in $(TLS_DIR)..."
	@sudo mkdir -p "$(TLS_DIR)"
	@if [ ! -f "$(TLS_DIR)/cert.pem" ] || [ ! -f "$(TLS_DIR)/key.pem" ]; then \
		printf '%s\n' \
			'[req]' 'default_bits = 2048' 'prompt = no' 'default_md = sha256' \
			'distinguished_name = dn' 'x509_extensions = v3_req' \
			'[dn]' 'CN = cookie-finder.local' \
			'[v3_req]' 'subjectAltName = @alt_names' \
			'basicConstraints = CA:FALSE' \
			'keyUsage = digitalSignature, keyEncipherment' \
			'extendedKeyUsage = serverAuth' \
			'[alt_names]' \
			'DNS.1 = cookie-finder.local' 'DNS.2 = localhost' \
			'IP.1 = 192.168.12.1' 'IP.2 = 127.0.0.1' \
			| sudo tee "$(TLS_DIR)/openssl-san.cnf" >/dev/null; \
		sudo openssl req -x509 -nodes -newkey rsa:2048 \
			-keyout "$(TLS_DIR)/key.pem" -out "$(TLS_DIR)/cert.pem" \
			-days 3650 -config "$(TLS_DIR)/openssl-san.cnf"; \
		echo "Generated $(TLS_DIR)/cert.pem"; \
	else \
		echo "Using existing $(TLS_DIR)/cert.pem"; \
	fi
	@sed \
		-e 's|@REPO_ROOT@|$(CURDIR)|g' \
		-e 's|@PYTHON@|$(WEB_PYTHON)|g' \
		-e 's|@WEB_HOST@|$(WEB_HOST)|g' \
		-e 's|@WEB_PORT@|$(WEB_PORT)|g' \
		-e 's|@WEB_HTTPS_PORT@|$(WEB_HTTPS_PORT)|g' \
		-e 's|@TLS_DIR@|$(TLS_DIR)|g' \
		$(WEB_SYSTEMD_UNIT_IN) | sudo tee $(WEB_SYSTEMD_UNIT) >/dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable cookie-finder-web.service
	@echo "Installed $(WEB_SYSTEMD_UNIT)"
	@echo "  python: $(WEB_PYTHON)"
	@echo "  listen: http://$(WEB_HOST):$(WEB_PORT)  https://$(WEB_HOST):$(WEB_HTTPS_PORT)"

on-the-pi-web-daemon: on-the-pi-web-daemon-install
	@echo "Stopping any foreground web server on ports $(WEB_PORT)/$(WEB_HTTPS_PORT)..."
	@-sudo fuser -k $(WEB_PORT)/tcp 2>/dev/null || true
	@-sudo fuser -k $(WEB_HTTPS_PORT)/tcp 2>/dev/null || true
	@sleep 0.5
	@sudo systemctl restart cookie-finder-web.service
	@sleep 1
	@sudo systemctl --no-pager --full status cookie-finder-web.service || true
	@echo ""
	@echo "Web server managed by systemd (http://$(WEB_HOST):$(WEB_PORT) + https://$(WEB_HOST):$(WEB_HTTPS_PORT))"
	@echo "Check status:  make on-the-pi-web-daemon-status"
	@echo "Stop:          make on-the-pi-web-daemon-stop"
	@echo "Follow logs:   sudo journalctl -u cookie-finder-web -f"

on-the-pi-web-daemon-stop:
	@sudo systemctl stop cookie-finder-web.service
	@echo "Stopped cookie-finder-web.service"

on-the-pi-web-daemon-status:
	@sudo systemctl --no-pager --full status cookie-finder-web.service || true
	@echo ""
	@echo "Web server managed by systemd"
	@echo "Check status:  make on-the-pi-web-daemon-status"
	@echo "Stop:          make on-the-pi-web-daemon-stop"
	@echo "Follow logs:   sudo journalctl -u cookie-finder-web -f"
	@echo "Recent logs:   sudo journalctl -u cookie-finder-web -n 30 --no-pager"

# Print reachable IPv4 addresses and the web app URL/port (Pi only).
# Also prints a terminal QR code when qrencode is installed.
on-the-pi-web-url:
	@port="$(WEB_PORT)"; \
	https_port="$(WEB_HTTPS_PORT)"; \
	if [ -f "$(WEB_SYSTEMD_UNIT)" ]; then \
		unit_port=$$(sed -n 's/.*--port[[:space:]]\+\([0-9][0-9]*\).*/\1/p' "$(WEB_SYSTEMD_UNIT)" | head -1); \
		unit_https=$$(sed -n 's/.*--https-port[[:space:]]\+\([0-9][0-9]*\).*/\1/p' "$(WEB_SYSTEMD_UNIT)" | head -1); \
		[ -n "$$unit_port" ] && port="$$unit_port"; \
		[ -n "$$unit_https" ] && https_port="$$unit_https"; \
	fi; \
	echo "port: $$port"; \
	echo "https_port: $$https_port"; \
	echo "mdns: http://$(MDNS_HOSTNAME).local/"; \
	echo "ips:"; \
	ips=$$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $$2, $$4}' | sed 's|/.*||'); \
	if [ -z "$$ips" ]; then \
		echo "  (none — no global IPv4 yet)"; \
		exit 0; \
	fi; \
	primary=""; \
	echo "$$ips" | while read -r iface addr; do \
		[ -z "$$addr" ] && continue; \
		echo "  $$iface  $$addr"; \
		if [ "$$port" = "80" ]; then \
			echo "  url: http://$$addr/"; \
		else \
			echo "  url: http://$$addr:$$port"; \
		fi; \
		if [ -n "$$https_port" ] && [ "$$https_port" != "0" ]; then \
			if [ "$$https_port" = "443" ]; then \
				echo "  url: https://$$addr/"; \
			else \
				echo "  url: https://$$addr:$$https_port"; \
			fi; \
		fi; \
	done; \
	primary=$$(echo "$$ips" | awk -v p="$$port" ' \
		$$1 ~ /^wlan/ { \
			if (p == "80") print "http://" $$2 "/"; \
			else print "http://" $$2 ":" p; \
			found=1; exit \
		} \
		!found && NF { \
			if (p == "80") first="http://" $$2 "/"; \
			else first="http://" $$2 ":" p \
		} \
		END { if (!found && first) print first }'); \
	if [ -n "$$primary" ]; then \
		echo "primary: $$primary"; \
		if command -v qrencode >/dev/null 2>&1; then \
			echo ""; \
			qrencode -t ANSIUTF8 "$$primary"; \
		else \
			echo "qr: install with: sudo apt-get install -y qrencode"; \
		fi; \
	fi; \
	if systemctl is-active --quiet cookie-finder-web.service 2>/dev/null; then \
		echo "service: cookie-finder-web (active)"; \
	else \
		echo "service: cookie-finder-web (not active — start with: make on-the-pi-web-daemon)"; \
	fi

# Hostname + Avahi so LAN clients can use http://cookie-finder.local/
on-the-pi-mdns:
	@echo "Installing Avahi (mDNS) for $(MDNS_HOSTNAME).local..."
	sudo apt-get update
	sudo apt-get install -y avahi-daemon libnss-mdns
	@echo "Setting hostname to $(MDNS_HOSTNAME)..."
	@sudo hostnamectl set-hostname "$(MDNS_HOSTNAME)"
	@if [ -f /etc/hosts ]; then \
		if grep -qE '[[:space:]]$(MDNS_HOSTNAME)([[:space:]]|$$)' /etc/hosts; then \
			echo "/etc/hosts already mentions $(MDNS_HOSTNAME)"; \
		else \
			if grep -qE '^127\.0\.1\.1[[:space:]]' /etc/hosts; then \
				sudo sed -i 's/^127\.0\.1\.1.*/127.0.1.1\t$(MDNS_HOSTNAME)/' /etc/hosts; \
			elif grep -qE '^127\.0\.0\.1[[:space:]]' /etc/hosts; then \
				sudo sed -i 's/^127\.0\.0\.1.*/127.0.0.1\tlocalhost $(MDNS_HOSTNAME)/' /etc/hosts; \
			else \
				echo "127.0.0.1 localhost $(MDNS_HOSTNAME)" | sudo tee -a /etc/hosts >/dev/null; \
			fi; \
			echo "Updated /etc/hosts for $(MDNS_HOSTNAME)"; \
		fi; \
	fi
	@sudo mkdir -p /etc/avahi/services
	@sudo cp "$(AVAHI_SERVICE_IN)" "$(AVAHI_SERVICE)"
	@sudo systemctl enable avahi-daemon.service
	@sudo systemctl restart avahi-daemon.service
	@sleep 1
	@sudo systemctl --no-pager --full status avahi-daemon.service || true
	@echo ""
	@echo "mDNS ready: http://$(MDNS_HOSTNAME).local/  (HTTPS: https://$(MDNS_HOSTNAME).local/)"
	@echo "Clients on the same LAN (and most SoftAP guests) can resolve this name via Avahi."
	@echo "AP hotspot tip: phones may still use the captive portal at http://192.168.12.1/"

# WiFi button + LED daemon (independent of web app)
on-the-pi-wifi-gpio-daemon-install:
	@test -x "$(WIFI_PYTHON)" || { \
		echo "error: missing $(WIFI_PYTHON)"; \
		echo "hint: run 'make on-the-pi-install' first (creates .venv)"; \
		exit 1; \
	}
	@sed \
		-e 's|@REPO_ROOT@|$(CURDIR)|g' \
		-e 's|@PYTHON@|$(WIFI_PYTHON)|g' \
		$(WIFI_SYSTEMD_UNIT_IN) | sudo tee $(WIFI_SYSTEMD_UNIT) >/dev/null
	@sudo systemctl daemon-reload
	@sudo systemctl enable cookie-finder-wifi.service
	@echo "Installed $(WIFI_SYSTEMD_UNIT)"
	@echo "  python: $(WIFI_PYTHON)"
	@echo "  module: cookie_finder.wifi.gpio_daemon"

on-the-pi-wifi-gpio-daemon: on-the-pi-wifi-gpio-daemon-install
	@sudo systemctl restart cookie-finder-wifi.service
	@sleep 1
	@sudo systemctl --no-pager --full status cookie-finder-wifi.service || true
	@echo ""
	@echo "WiFi button+LED managed by systemd"
	@echo "Check status:  make on-the-pi-wifi-gpio-daemon-status"
	@echo "Stop:          make on-the-pi-wifi-gpio-daemon-stop"
	@echo "Follow logs:   sudo journalctl -u cookie-finder-wifi -f"

on-the-pi-wifi-gpio-daemon-stop:
	@sudo systemctl stop cookie-finder-wifi.service
	@echo "Stopped cookie-finder-wifi.service"

on-the-pi-wifi-gpio-daemon-status:
	@sudo systemctl --no-pager --full status cookie-finder-wifi.service || true
	@echo ""
	@echo "WiFi button+LED managed by systemd"
	@echo "Check status:  make on-the-pi-wifi-gpio-daemon-status"
	@echo "Stop:          make on-the-pi-wifi-gpio-daemon-stop"
	@echo "Follow logs:   sudo journalctl -u cookie-finder-wifi -f"
	@echo "Recent logs:   sudo journalctl -u cookie-finder-wifi -n 30 --no-pager"

# =============================================================================
# Backward-compatible aliases (prefer on-the-mac-* / on-the-pi-*)
# =============================================================================

.PHONY: install install-yolo install-docs docs clean init init-wifi \
        run run-standalone run-web run-web-custom \
        test-motors test-motors-pan-cw test-motors-pan-ccw test-motors-tilt-cw \
        test-motors-tilt-ccw test-motors-home test-bluetooth-input test-gimbal-gamepad \
        test-pan-step test-all-gpio \
        find-camera list-devices list-controls get-control set-control \
        install-ffmpeg install-libusb list-cameras list-camera-formats \
        probe probe-install probe-usb probe-cdc probe-serial probe-commands probe-resolution probe-xu \
        serial-help serial-list serial-connect serial-run serial-deploy serial-deploy-rust \
        rust-help rust-check rust-build-mac rust-build-pi rust-build-pi-remote \
        rust-deploy rust-deploy-cookie rust-deploy-remote rust-deploy-cookie-remote \
        rust-daemon rust-run run-with-rust rust-home rust-keyboard \
        wifi-gpio-daemon wifi-gpio-daemon-install wifi-gpio-daemon-stop wifi-gpio-daemon-status \
        wifi-configure-clients wifi-fix \
        web-daemon web-daemon-install web-daemon-stop web-daemon-status \
        web-url wifi-status mdns mdns-install

# Installation
install: on-the-pi-install
install-yolo: on-the-pi-install-yolo
install-docs: on-the-mac-install-docs
install-ffmpeg: on-the-mac-install-ffmpeg
install-libusb: on-the-mac-install-libusb
init: on-the-pi-init
init-wifi: on-the-pi-init-wifi
docs: on-the-mac-docs
clean: on-the-pi-clean

# Run
run: on-the-pi-run
run-standalone: on-the-pi-run-standalone
run-web: on-the-pi-run-web
run-web-custom: on-the-pi-run-web-custom

# Hardware tests
test-motors: on-the-pi-test-motors
test-motors-pan-cw: on-the-pi-test-motors-pan-cw
test-motors-pan-ccw: on-the-pi-test-motors-pan-ccw
test-motors-tilt-cw: on-the-pi-test-motors-tilt-cw
test-motors-tilt-ccw: on-the-pi-test-motors-tilt-ccw
test-motors-home: on-the-pi-test-motors-home
test-bluetooth-input: on-the-pi-test-bluetooth-input
test-gimbal-gamepad: on-the-pi-test-gimbal-gamepad
test-pan-step: on-the-pi-test-pan-step
test-all-gpio: on-the-pi-test-all-gpio

# Camera (Pi)
find-camera: on-the-pi-find-camera
list-devices: on-the-pi-list-devices
list-controls: on-the-pi-list-controls
get-control: on-the-pi-get-control
set-control: on-the-pi-set-control

# Camera (Mac)
list-cameras: on-the-mac-list-cameras
list-camera-formats: on-the-mac-list-camera-formats

# Probing (Mac)
probe: on-the-mac-probe
probe-install: on-the-mac-probe-install
probe-usb: on-the-mac-probe-usb
probe-cdc: on-the-mac-probe-cdc
probe-serial: on-the-mac-probe-serial
probe-commands: on-the-mac-probe-commands
probe-resolution: on-the-mac-probe-resolution
probe-xu: on-the-mac-probe-xu

# Serial (Mac)
serial-help: on-the-mac-serial-help
serial-list: on-the-mac-serial-list
serial-connect: on-the-mac-serial-connect
serial-run: on-the-mac-serial-run
serial-deploy: on-the-mac-serial-deploy
serial-deploy-rust: on-the-mac-serial-deploy-rust

# Rust
rust-help: on-the-mac-help on-the-pi-help
rust-check: on-the-mac-rust-check
rust-build-mac: on-the-mac-rust-build
rust-build-pi: on-the-pi-rust-build
rust-build-pi-remote: on-the-mac-rust-build-remote
rust-deploy: on-the-mac-rust-deploy
rust-deploy-cookie: on-the-mac-rust-deploy-cookie
rust-deploy-remote: on-the-mac-rust-deploy-remote
rust-deploy-cookie-remote: on-the-mac-rust-deploy-cookie-remote
# Legacy alias: Mac SSH deploy + foreground daemon (NOT Pi systemd).
# Prefer: make on-the-mac-rust-daemon  or  make on-the-pi-rust-daemon
rust-daemon: on-the-mac-rust-daemon
rust-run: on-the-mac-rust-run
rust-home: on-the-mac-rust-home
run-with-rust: on-the-mac-run-with-rust
rust-keyboard: on-the-pi-rust-keyboard

# WiFi button + LED
wifi-gpio-daemon: on-the-pi-wifi-gpio-daemon
wifi-gpio-daemon-install: on-the-pi-wifi-gpio-daemon-install
wifi-gpio-daemon-stop: on-the-pi-wifi-gpio-daemon-stop
wifi-gpio-daemon-status: on-the-pi-wifi-gpio-daemon-status
wifi-status: on-the-pi-wifi-status
wifi-configure-clients: on-the-pi-wifi-configure-clients
wifi-fix: on-the-pi-wifi-fix

# Web server daemon
web-daemon: on-the-pi-web-daemon
web-daemon-install: on-the-pi-web-daemon-install
web-daemon-stop: on-the-pi-web-daemon-stop
web-daemon-status: on-the-pi-web-daemon-status
web-url: on-the-pi-web-url

# mDNS (cookie-finder.local)
mdns: on-the-pi-mdns
mdns-install: on-the-pi-mdns
