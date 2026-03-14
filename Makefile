.PHONY: run install clean list-devices list-controls get-control set-control

install:
	uv sync

run:
	uv run main.py

list-devices:
	uv run uvc_controls.py list-devices

list-controls:
	uv run uvc_controls.py list-controls

get-control:
	@read -p "Enter control name: " control; \
	uv run uvc_controls.py get $$control

set-control:
	@read -p "Enter control name: " control; \
	read -p "Enter value: " value; \
	uv run uvc_controls.py set $$control $$value

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
