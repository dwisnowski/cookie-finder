.PHONY: run install clean list-devices list-controls get-control set-control install-ffmpeg install-libusb list-cameras list-camera-formats probe probe-install probe-usb probe-cdc probe-serial probe-resolution probe-xu

install:
	uv sync

install-yolo:
	uv sync --extra yolo

install-ffmpeg:
	brew install ffmpeg

install-libusb:
	brew install libusb

run:
	uv run main.py

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
