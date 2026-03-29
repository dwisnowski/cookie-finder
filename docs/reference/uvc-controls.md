# UVC Camera Controls

Utility commands for inspecting and modifying UVC camera settings on the thermal camera.

---

## List Connected Devices

```bash
make list-devices
# or
uv run tools/uvc_controls.py list-devices
```

---

## List Available Controls

```bash
make list-controls
# or
uv run tools/uvc_controls.py list-controls
```

---

## Get a Control Value

```bash
make get-control
# Prompts: Enter control name:
```

Or directly:

```bash
uv run tools/uvc_controls.py get <control_name>
```

---

## Set a Control Value

```bash
make set-control
# Prompts: Enter control name: / Enter value:
```

Or directly:

```bash
uv run tools/uvc_controls.py set <control_name> <value>
```

---

## Notes

- The Mileseey TNV30i exposes only standard UVC controls — no vendor-specific extension units are accessible.
- Temperature data cannot be extracted via UVC XU controls on this device.
- For raw USB/CDC/serial probing, see the probe scripts under `tools/probing_thermal_camera/` or run `make probe`.
