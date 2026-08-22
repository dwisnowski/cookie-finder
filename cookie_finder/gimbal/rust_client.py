"""Unix-socket IPC client for the Rust gimbal daemon."""

from __future__ import annotations

import json
import os
import socket
from typing import Any, Optional

DEFAULT_SOCKET = "/tmp/cookie-finder.sock"


class RustGimbalClient:
    """IPC client for the Rust cookie-finder-ctl gimbal daemon."""

    def __init__(self, socket_path: str, max_pan: float, max_tilt: float):
        self._socket_path = socket_path
        self.max_pan = max_pan
        self.max_tilt = max_tilt

    @classmethod
    def connect(
        cls,
        socket_path: Optional[str] = None,
        max_pan: float = 150.0,
        max_tilt: float = 60.0,
        timeout: float = 1.0,
        quiet: bool = False,
    ) -> Optional["RustGimbalClient"]:
        path = socket_path or os.environ.get("COOKIE_FINDER_SOCKET", DEFAULT_SOCKET)
        client = cls(path, max_pan, max_tilt)
        try:
            resp = client._request({"cmd": "ping"}, timeout=timeout)
            if resp.get("ok"):
                if not quiet:
                    print(f"✓ Connected to Rust gimbal daemon ({path})")
                return client
        except (OSError, json.JSONDecodeError, KeyError) as e:
            if not quiet:
                print(f"⚠ Rust gimbal daemon not available ({path}): {e}")
        return None

    def _request(self, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(self._socket_path)
            sock.sendall((json.dumps(payload) + "\n").encode())
            data = b""
            while b"\n" not in data:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
        line = data.split(b"\n", 1)[0]
        return json.loads(line.decode())

    def ping(self, timeout: float = 0.5) -> bool:
        """Return True if the daemon responds to ping."""
        try:
            return bool(self._request({"cmd": "ping"}, timeout=timeout).get("ok"))
        except (OSError, json.JSONDecodeError, KeyError, TimeoutError):
            return False

    def set_speed(self, pan_hz: float = 500, tilt_hz: float = 500) -> None:
        self._request({"cmd": "set_speed", "pan_hz": pan_hz, "tilt_hz": tilt_hz})

    def move_to_angles(self, pan_angle: float, tilt_angle: float) -> None:
        self._request({"cmd": "move_to_angles", "pan": pan_angle, "tilt": tilt_angle})

    def pan(self, angle: float) -> None:
        pan, tilt = self.get_position()
        self.move_to_angles(angle, tilt)

    def tilt(self, angle: float) -> None:
        pan, tilt = self.get_position()
        self.move_to_angles(pan, angle)

    def pan_step(self, direction: int, steps: int = 1) -> None:
        self._request({"cmd": "pan_step", "direction": direction, "steps": steps})

    def tilt_step(self, direction: int, steps: int = 1) -> None:
        self._request({"cmd": "tilt_step", "direction": direction, "steps": steps})

    def get_position(self) -> tuple[float, float]:
        resp = self._request({"cmd": "get_position"})
        return float(resp["pan"]), float(resp["tilt"])

    def is_moving(self) -> bool:
        resp = self._request({"cmd": "get_status"})
        return bool(resp.get("is_moving", False))

    def home(self) -> None:
        self._request({"cmd": "home"})

    def is_calibrated(self) -> bool:
        return True

    def stop(self) -> None:
        self._request({"cmd": "stop"})

    def disable_motors(self) -> None:
        """De-energize stepper coils (all control pins LOW) to reduce heating."""
        self._request({"cmd": "disable_motors"})

    def set_input_enabled(self, enabled: bool) -> None:
        """Enable/disable gamepad→gimbal without changing the active device."""
        self._request({"cmd": "set_input_enabled", "enabled": enabled})

    def set_active_input(
        self,
        enabled: bool,
        address: str | None = None,
        name: str | None = None,
    ) -> None:
        """Select which BlueZ HID pad the daemon should read (hot-swappable)."""
        payload: dict[str, Any] = {"cmd": "set_active_input", "enabled": enabled}
        if address:
            payload["address"] = address
        if name:
            payload["name"] = name
        self._request(payload)

    def get_phase_order(self) -> dict:
        return self._request({"cmd": "get_phase_order"})

    def set_phase_order(self, motor: str, order: list[int]) -> None:
        self._request({"cmd": "set_phase_order", "motor": motor, "order": order})

    def get_drive_mode(self) -> dict:
        return self._request({"cmd": "get_drive_mode"})

    def set_drive_mode(self, mode: str) -> dict:
        """Set coil drive algorithm: 'wave', 'full', or 'half'."""
        return self._request({"cmd": "set_drive_mode", "mode": mode})

    def cleanup(self) -> None:
        pass
