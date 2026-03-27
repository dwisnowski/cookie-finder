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
