.PHONY: help init run run-standalone run-web run-web-custom test-gamepad test-motors test-motors-pan-cw test-motors-pan-ccw test-motors-tilt-cw test-motors-tilt-ccw test-motors-home test-bluetooth-input test-gimbal-gamepad install install-yolo install-docs docs clean list-devices list-controls get-control set-control install-ffmpeg install-libusb list-cameras list-camera-formats probe probe-install probe-usb probe-cdc probe-serial probe-resolution probe-xu find-camera test-pan-step test-all-gpio

.DEFAULT_GOAL := help

help:
	@echo "Cookie Finder – Makefile Targets"
	@echo ""
	@echo "Installation:"
	@echo "  make install           Install dependencies"
	@echo "  make install-yolo      Install with YOLO model support"
	@echo "  make install-docs      Install MkDocs dependencies"
	@echo "  make install-ffmpeg    Install FFmpeg (macOS)"
	@echo "  make install-libusb    Install libusb (macOS)"
	@echo "  make init              Initialize Bluetooth permissions (Orange Pi)"
	@echo ""
	@echo "Running the Application:"
	@echo "  make run               Start web server (default)"
	@echo "  make run-standalone    Start standalone GUI mode"
	@echo "  make run-web           Start web server (http://0.0.0.0:8000)"
	@echo "  make run-web-custom    Start web server with custom host/port"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs              Start MkDocs dev server (http://127.0.0.1:8001)"
	@echo ""
	@echo "Camera Management:"
	@echo "  make find-camera       Detect available camera devices"
	@echo "  make list-devices      List UVC device controls"
	@echo "  make list-controls     List camera control names"
	@echo "  make get-control       Get camera control value (interactive)"
	@echo "  make set-control       Set camera control value (interactive)"
	@echo "  make list-cameras      List available cameras (ffmpeg)"
	@echo "  make list-camera-formats  Capture sample thermal image"
	@echo ""
	@echo "Testing & Hardware:"
	@echo "  make test-gamepad      Test gamepad input (60 seconds)"
	@echo "  make test-bluetooth-input  Test Bluetooth gamepad input"
	@echo "  make test-gimbal-gamepad   Control gimbal with joystick"
	@echo "  make test-motors       Motor control test (auto sequence)"
	@echo "  make test-motors-pan-cw    Pan clockwise 50 steps"
	@echo "  make test-motors-pan-ccw   Pan counter-clockwise 50 steps"
	@echo "  make test-motors-tilt-cw   Tilt clockwise 50 steps"
	@echo "  make test-motors-tilt-ccw  Tilt counter-clockwise 50 steps"
	@echo "  make test-motors-home      Home both motors"
	@echo "  make test-pan-step     Manual pan motor stepping"
	@echo "  make test-all-gpio     Scan and test all GPIO pins"
	@echo ""
	@echo "Camera Probing (Debug):"
	@echo "  make probe             Run all probing tests"
	@echo "  make probe-install     Install libusb (macOS)"
	@echo "  make probe-usb         Probe USB camera details"
	@echo "  make probe-cdc         Probe CDC (serial) interface"
	@echo "  make probe-serial      Probe serial port data"
	@echo "  make probe-resolution  Probe camera resolution"
	@echo "  make probe-xu          Probe UVC extension units"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             Remove Python cache files"

install:
	uv sync

install-yolo:
	uv sync --extra yolo

install-docs:
	uv sync --extra docs

docs: install-docs
	@echo "Starting MkDocs dev server at http://127.0.0.1:8001..."
	uv run mkdocs serve --dev-addr 127.0.0.1:8001

install-ffmpeg:
	brew install ffmpeg

install-libusb:
	brew install libusb

init:
	@echo "Initializing system permissions for Bluetooth..."
	sudo usermod -aG bluetooth cookie
	@echo "Bluetooth group permissions added. Please log out and log back in for changes to take effect."

# Run modes
run: run-web

run-standalone:
	@echo "Starting Thermal Camera Viewer (Standalone GUI mode)..."
	uv run main.py

run-web:
	@echo "Starting Thermal Camera Viewer (WebServer mode on http://0.0.0.0:8000)..."
	uv run main.py --web

run-web-custom:
	@read -p "Enter port (default 8000): " port; \
	read -p "Enter host (default 0.0.0.0): " host; \
	port=$${port:-8000}; \
	host=$${host:-0.0.0.0}; \
	echo "Starting Thermal Camera Viewer (WebServer mode on http://$$host:$$port)..."; \
	uv run main.py --web --port $$port --host $$host

test-motors:
	@echo "Motor control test script:"
	@echo "  sudo make test-motors auto           # Automated test sequence"
	@echo "  sudo make test-motors-pan-cw         # Pan clockwise 50 steps"
	@echo "  sudo make test-motors-pan-ccw        # Pan counter-clockwise 50 steps"
	@echo "  sudo make test-motors-tilt-cw        # Tilt clockwise 50 steps"
	@echo "  sudo make test-motors-tilt-ccw       # Tilt counter-clockwise 50 steps"
	@echo "  sudo make test-motors-home           # Home both motors"
	@sudo uv run tools/test_motors.py auto

test-motors-pan-cw:
	@uv run tools/test_motors.py pan-cw

test-motors-pan-ccw:
	@sudo uv run tools/test_motors.py pan-ccw

test-motors-tilt-cw:
	@sudo uv run tools/test_motors.py tilt-cw

test-motors-tilt-ccw:
	@sudo uv run tools/test_motors.py tilt-ccw

test-motors-home:
	@sudo uv run tools/test_motors.py home

test-bluetooth-input:
	@echo "Bluetooth input test: reads and logs all gamepad input (60 seconds)..."
	@uv run tools/test_bluetooth_input.py

test-gimbal-gamepad:
	@echo "Gimbal + Gamepad test: control gimbal with joystick input..."
	@uv run tools/test_gimbal_gamepad.py

test-pan-step:
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

test-all-gpio:
	@echo "Scanning and blinking all available GPIO pins..."
	@bash -c '\
	for chip in /dev/gpiochip*; do \
		chipname=$$(basename $$chip); \
		echo "---- Testing $$chipname ----"; \
		lines=$$(gpioinfo $$chipname | grep "line" | wc -l); \
		for ((i=0; i<lines; i++)); do \
			# Try to toggle HIGH then LOW (skip if busy) \
			sudo gpioset $$chipname $$i=1 2>/dev/null && \
			sleep 0.05 && \
			sudo gpioset $$chipname $$i=0 2>/dev/null && \
			echo "  Toggled $$chipname line $$i"; \
		done; \
	done; \
	echo "Done scanning all GPIO."; \
	'

find-camera:
	@echo "Detecting available camera devices..."
	@echo "Checking /dev/video devices:"
	@ls -la /dev/video* 2>/dev/null || echo "No /dev/video devices found"
	@echo ""
	@echo "Camera details:"
	@v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
	@echo ""
	@uv run tools/find_camera.py

list-devices:
	uv run tools/uvc_controls.py list-devices

list-controls:
	uv run tools/uvc_controls.py list-controls

get-control:
	@read -p "Enter control name: " control; \
	uv run tools/uvc_controls.py get $control

set-control:
	@read -p "Enter control name: " control; \
	read -p "Enter value: " value; \
	uv run tools/uvc_controls.py set $control $value

list-cameras:
	ffmpeg -f avfoundation -list_devices true -i ""

list-camera-formats:
	ffmpeg -f avfoundation -video_size 512x390 -framerate 50 -i "0" -vframes 1 thermal_capture.tiff

probe-install:
	brew install libusb

probe-usb: probe-install
	uv run tools/probing_thermal_camera/probe_usb.py

probe-cdc:
	uv run tools/probing_thermal_camera/probe_cdc.py

probe-serial:
	uv run tools/probing_thermal_camera/probe_serial.py

probe-resolution:
	uv run tools/probing_thermal_camera/probe_resolution.py

probe-xu:
	uv run tools/probing_thermal_camera/probe_uvc_xu.py

probe: probe-usb probe-cdc probe-serial probe-resolution probe-xu

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
