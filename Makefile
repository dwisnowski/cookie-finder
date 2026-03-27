.PHONY: run run-standalone run-web run-web-custom test-gamepad test-motors test-motors-pan-cw test-motors-pan-ccw test-motors-tilt-cw test-motors-tilt-ccw test-motors-home install install-yolo clean list-devices list-controls get-control set-control install-ffmpeg install-libusb list-cameras list-camera-formats probe probe-install probe-usb probe-cdc probe-serial probe-resolution probe-xu find-camera

install:
	uv sync

install-yolo:
	uv sync --extra yolo

install-ffmpeg:
	brew install ffmpeg

install-libusb:
	brew install libusb

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
	@echo "  make test-motors auto           # Automated test sequence"
	@echo "  make test-motors-pan-cw         # Pan clockwise 50 steps"
	@echo "  make test-motors-pan-ccw        # Pan counter-clockwise 50 steps"
	@echo "  make test-motors-tilt-cw        # Tilt clockwise 50 steps"
	@echo "  make test-motors-tilt-ccw       # Tilt counter-clockwise 50 steps"
	@echo "  make test-motors-home           # Home both motors"
	uv run test_stepper_motors.py auto

test-motors-pan-cw:
	uv run test_stepper_motors.py pan-cw

test-motors-pan-ccw:
	uv run test_stepper_motors.py pan-ccw

test-motors-tilt-cw:
	uv run test_stepper_motors.py tilt-cw

test-motors-tilt-ccw:
	uv run test_stepper_motors.py tilt-ccw

test-motors-home:
	uv run test_stepper_motors.py home

test-pan-pins:
	@echo "Toggling pan motor pins (PI10, PI11, PI12, PI13) - watch logic analyzer..."
	@bash -c '\
		export GPIO_PINS="266 267 268 269"; \
		echo "GPIO offsets: $$GPIO_PINS"; \
		for gpio in $$GPIO_PINS; do \
			echo $$gpio > /sys/class/gpio/export 2>/dev/null || true; \
			echo "out" > /sys/class/gpio/gpio$$gpio/direction 2>/dev/null || true; \
		done; \
		for i in 1 2 3 4 5; do \
			echo "  Cycle $$i: All ON"; \
			for gpio in $$GPIO_PINS; do echo 1 > /sys/class/gpio/gpio$$gpio/value 2>/dev/null || true; done; \
			sleep 1; \
			echo "  Cycle $$i: All OFF"; \
			for gpio in $$GPIO_PINS; do echo 0 > /sys/class/gpio/gpio$$gpio/value 2>/dev/null || true; done; \
			sleep 1; \
		done; \
		echo "Done. 5 cycles complete."; \
		for gpio in $$GPIO_PINS; do \
			echo $$gpio > /sys/class/gpio/unexport 2>/dev/null || true; \
		done \
	'

find-camera:
	@echo "Detecting available camera devices..."
	@echo "Checking /dev/video devices:"
	@ls -la /dev/video* 2>/dev/null || echo "No /dev/video devices found"
	@echo ""
	@echo "Camera details:"
	@v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
	@echo ""
	@uv run find_working_camera.py

list-devices:
	uv run uvc_controls.py list-devices

list-controls:
	uv run uvc_controls.py list-controls

get-control:
	@read -p "Enter control name: " control; \
	uv run uvc_controls.py get $control

set-control:
	@read -p "Enter control name: " control; \
	read -p "Enter value: " value; \
	uv run uvc_controls.py set $control $value

list-cameras:
	ffmpeg -f avfoundation -list_devices true -i ""

list-camera-formats:
	ffmpeg -f avfoundation -video_size 512x390 -framerate 50 -i "0" -vframes 1 thermal_capture.tiff

probe-install:
	brew install libusb

probe-usb: probe-install
	uv run probing_thermal_camera/probe_usb.py

probe-cdc:
	uv run probing_thermal_camera/probe_cdc.py

probe-serial:
	uv run probing_thermal_camera/probe_serial.py

probe-resolution:
	uv run probing_thermal_camera/probe_resolution.py

probe-xu:
	uv run probing_thermal_camera/probe_uvc_xu.py

probe: probe-usb probe-cdc probe-serial probe-resolution probe-xu

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
