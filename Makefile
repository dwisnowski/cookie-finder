.PHONY: run run-standalone run-web run-web-custom install install-yolo clean list-devices list-controls get-control set-control install-ffmpeg install-libusb list-cameras list-camera-formats probe probe-install probe-usb probe-cdc probe-serial probe-resolution probe-xu find-camera

install:
	uv sync

install-yolo:
	uv sync --extra yolo

install-ffmpeg:
	brew install ffmpeg

install-libusb:
	brew install libusb

# Run modes
run: run-standalone

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

find-camera:
	@echo "Detecting available camera devices..."
	@echo "Checking /dev/video devices:"
	@ls -la /dev/video* 2>/dev/null || echo "No /dev/video devices found"
	@echo ""
	@echo "Camera details:"
	@v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
	@echo ""
	@echo "Testing which device actually provides frames..."
	@python3 -c "import cv2, sys; devices = [(i, cv2.VideoCapture(i)) for i in range(10)]; [print(f'  /dev/video{i}: OK - {cv2.VideoCapture(i).get(3):.0f}x{cv2.VideoCapture(i).get(4):.0f}') for i, cap in devices if cap.isOpened() and cap.read()[0]]" 2>/dev/null || echo "Testing with uv run..."
	@uv run -c "import cv2; devices = [i for i in range(10) if cv2.VideoCapture(i).isOpened() and cv2.VideoCapture(i).read()[0]]; print('Found thermal camera on /dev/video' + str(devices[0]) if devices else 'No working camera found')"

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
