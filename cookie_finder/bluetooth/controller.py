"""
BlueZ Classic HID Bluetooth device manager for gamepads on Linux (Orange Pi).

Uses bluetoothctl for scan/pair/trust/connect/disconnect/remove. Gamepad axis
input is owned by the Rust cookie-finder-ctl daemon (evdev); this module only
tracks which BlueZ device is the active input source for the web UI.
"""

from __future__ import annotations

import subprocess
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional, Set, Tuple


def normalize_address(address: str) -> str:
    """Normalize a Bluetooth MAC to uppercase colon-separated form."""
    return address.strip().upper()


class BluetoothDevice:
    """Represents a BlueZ-known Bluetooth device."""

    def __init__(
        self,
        address: str,
        name: str,
        rssi: int = -100,
        paired: bool = False,
        connected: bool = False,
        trusted: bool = False,
    ):
        self.address = normalize_address(address)
        self.name = name or f"Unknown ({self.address[-5:]})"
        self.rssi = rssi
        self.paired = paired
        self.connected = connected
        self.trusted = trusted

    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "name": self.name,
            "rssi": self.rssi,
            "paired": self.paired,
            "connected": self.connected,
            "trusted": self.trusted,
            "signal_strength": self._rssi_to_bars(self.rssi),
        }

    @staticmethod
    def _rssi_to_bars(rssi: int) -> str:
        if rssi >= -50:
            return "▓▓▓▓▓"
        if rssi >= -60:
            return "▓▓▓▓░"
        if rssi >= -70:
            return "▓▓▓░░"
        if rssi >= -80:
            return "▓▓░░░"
        return "▓░░░░"


class BluetoothController:
    """Manages Classic Bluetooth HID devices via BlueZ (bluetoothctl)."""

    SCAN_DURATION_S = 15.0
    SCAN_POLL_S = 1.5

    def __init__(self):
        self.scanning = False
        self.devices: Dict[str, BluetoothDevice] = {}
        self.connected_device_address: Optional[str] = None
        self.scan_thread: Optional[threading.Thread] = None
        self.status_callback: Optional[Callable] = None
        self.last_error: Optional[str] = None
        self._btctl_lock = threading.Lock()
        self._refresh_known_devices()

    # --- BlueZ helpers -----------------------------------------------------

    def _run_bt(
        self, *args: str, timeout: float = 30.0
    ) -> Tuple[bool, str, str]:
        """Run bluetoothctl with the given args. Returns (ok, stdout, stderr)."""
        try:
            with self._btctl_lock:
                result = subprocess.run(
                    ["bluetoothctl", "--", *args],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            ok = result.returncode == 0
            if not ok and not stderr and stdout:
                # bluetoothctl often puts failure text on stdout
                stderr = stdout
            return ok, stdout, stderr
        except FileNotFoundError:
            msg = "bluetoothctl not found — install bluez (sudo apt install bluez)"
            return False, "", msg
        except subprocess.TimeoutExpired:
            return False, "", f"bluetoothctl timed out: {' '.join(args)}"
        except Exception as e:
            return False, "", f"{type(e).__name__}: {e}"

    def _ensure_adapter(self) -> Tuple[bool, str]:
        ok, _, err = self._run_bt("power", "on", timeout=10)
        if not ok:
            return False, err or "Failed to power on Bluetooth adapter"
        self._run_bt("agent", "on", timeout=5)
        self._run_bt("default-agent", timeout=5)
        return True, ""

    def _parse_devices_output(self, stdout: str) -> Dict[str, str]:
        """Parse `Device <addr> <name>` lines → {ADDRESS: name}."""
        found: Dict[str, str] = {}
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("Device "):
                continue
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            address = normalize_address(parts[1])
            name = parts[2] if len(parts) >= 3 else f"Unknown ({address[-5:]})"
            found[address] = name
        return found

    def _parse_info(self, address: str, stdout: str) -> Dict:
        info = {
            "address": normalize_address(address),
            "name": None,
            "paired": False,
            "trusted": False,
            "connected": False,
            "rssi": -100,
        }
        for line in stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("Name:"):
                info["name"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Alias:") and not info["name"]:
                info["name"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Paired:"):
                info["paired"] = "yes" in stripped.lower()
            elif stripped.startswith("Trusted:"):
                info["trusted"] = "yes" in stripped.lower()
            elif stripped.startswith("Connected:"):
                info["connected"] = "yes" in stripped.lower()
            elif stripped.startswith("RSSI:"):
                try:
                    info["rssi"] = int(stripped.split(":", 1)[1].strip())
                except ValueError:
                    pass
        return info

    def _get_info(self, address: str) -> Optional[Dict]:
        address = normalize_address(address)
        ok, stdout, _ = self._run_bt("info", address, timeout=8)
        if not ok or not stdout:
            return None
        return self._parse_info(address, stdout)

    def _upsert_device_from_info(self, address: str, fallback_name: str = "") -> BluetoothDevice:
        address = normalize_address(address)
        info = self._get_info(address)
        existing = self.devices.get(address)
        if info:
            name = info["name"] or fallback_name or (existing.name if existing else None)
            device = BluetoothDevice(
                address=address,
                name=name or f"Unknown ({address[-5:]})",
                rssi=info["rssi"],
                paired=info["paired"],
                connected=info["connected"],
                trusted=info["trusted"],
            )
        else:
            device = existing or BluetoothDevice(
                address, fallback_name or f"Unknown ({address[-5:]})"
            )
        self.devices[address] = device
        return device

    def _refresh_known_devices(self) -> None:
        """Refresh cache from BlueZ known + connected devices."""
        ok, stdout, _ = self._run_bt("devices", timeout=8)
        known = self._parse_devices_output(stdout) if ok else {}

        ok_c, stdout_c, _ = self._run_bt("devices", "Connected", timeout=8)
        connected = set(self._parse_devices_output(stdout_c).keys()) if ok_c else set()

        # Merge newly known devices; keep prior entries that disappeared from
        # the brief scan window only if still paired (checked via info below).
        for address, name in known.items():
            self._upsert_device_from_info(address, fallback_name=name)

        for address in connected:
            if address not in self.devices:
                self._upsert_device_from_info(address)
            else:
                self.devices[address].connected = True

        # Drop stale non-paired entries that BlueZ no longer lists
        stale = [
            addr
            for addr, dev in self.devices.items()
            if addr not in known and addr not in connected and not dev.paired
        ]
        for addr in stale:
            del self.devices[addr]

    def _get_system_connected_devices(self) -> Set[str]:
        """Return uppercase MACs currently Connected in BlueZ."""
        ok, stdout, _ = self._run_bt("devices", "Connected", timeout=8)
        if not ok:
            return set()
        return set(self._parse_devices_output(stdout).keys())

    def _get_device_name_from_system(self, address: str) -> Optional[str]:
        info = self._get_info(address)
        if info and info.get("name"):
            return info["name"]
        return None

    def _set_error(self, message: str) -> None:
        self.last_error = message
        print(f"[BT] {message}")

    def _clear_error(self) -> None:
        self.last_error = None

    # --- Status / public list API ------------------------------------------

    def set_status_callback(self, callback: Callable):
        self.status_callback = callback

    def _emit_status(self, status: str, data: Optional[Dict] = None):
        if self.status_callback:
            self.status_callback({"status": status, "data": data or {}})

    def get_devices_list(self) -> List[Dict]:
        self._refresh_known_devices()
        return [d.to_dict() for d in self.devices.values()]

    def get_device(self, address: str) -> Optional[BluetoothDevice]:
        address = normalize_address(address)
        if address in self.devices:
            return self._upsert_device_from_info(address, self.devices[address].name)
        device = self._upsert_device_from_info(address)
        # If BlueZ knows nothing about it, still return a placeholder so UI
        # can attempt pair/connect after a scan discovery race.
        return device

    def get_connected_device(self) -> Optional[str]:
        return self.connected_device_address

    def get_last_error(self) -> Optional[str]:
        return self.last_error

    # --- Scan --------------------------------------------------------------

    def start_scan(self) -> bool:
        if self.scanning:
            return False

        ok, err = self._ensure_adapter()
        if not ok:
            self._set_error(err)
            self._emit_status("scan_error", {"error": err})
            return False

        self.scanning = True
        self._clear_error()
        # Preserve paired/connected devices; refresh merges new discoveries.
        print("[BT] Starting BlueZ scan...")
        self._emit_status("scan_started")
        self.scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self.scan_thread.start()
        return True

    def stop_scan(self):
        was_scanning = self.scanning
        self.scanning = False
        self._run_bt("scan", "off", timeout=5)
        if self.scan_thread and self.scan_thread.is_alive():
            self.scan_thread.join(timeout=3)
        if was_scanning:
            print("[BT] Scan stopped")
            self._refresh_known_devices()
            self._emit_status("scan_stopped", {"devices": self.get_devices_list()})

    def _scan_worker(self):
        try:
            # Non-interactive discovery for SCAN_DURATION_S
            scan_proc = subprocess.Popen(
                [
                    "bluetoothctl",
                    "--timeout",
                    str(int(self.SCAN_DURATION_S)),
                    "scan",
                    "on",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            deadline = time.time() + self.SCAN_DURATION_S
            while self.scanning and time.time() < deadline:
                self._refresh_known_devices()
                self._emit_status(
                    "scan_update", {"devices": [d.to_dict() for d in self.devices.values()]}
                )
                time.sleep(self.SCAN_POLL_S)

            self.scanning = False
            if scan_proc.poll() is None:
                scan_proc.terminate()
                try:
                    scan_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    scan_proc.kill()
            self._run_bt("scan", "off", timeout=5)
            self._refresh_known_devices()
            devices = [d.to_dict() for d in self.devices.values()]
            print(f"[BT] Scan complete. {len(devices)} device(s) known")
            if not devices:
                print(
                    "[BT] No devices found. Put the gamepad in pairing mode and "
                    "ensure bluez is running (systemctl status bluetooth)."
                )
            self._emit_status("scan_complete", {"devices": devices})
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"[BT] Scan error: {error_msg}\n{traceback.format_exc()}")
            self.scanning = False
            self._set_error(error_msg)
            self._emit_status("scan_error", {"error": error_msg})

    # --- Pair / connect / disconnect / remove ------------------------------

    def pair_device(self, address: str) -> bool:
        """Pair and trust a device (does not require connect)."""
        address = normalize_address(address)
        self._clear_error()
        ok, err = self._ensure_adapter()
        if not ok:
            self._set_error(err)
            self._emit_status("device_pair_failed", {"address": address, "error": err})
            return False

        print(f"[BT] Pairing {address}...")
        info = self._get_info(address)
        if info and info.get("paired"):
            if not info.get("trusted"):
                self._run_bt("trust", address, timeout=15)
            device = self._upsert_device_from_info(address)
            self._emit_status(
                "device_paired", {"address": address, "name": device.name}
            )
            return True

        ok, stdout, stderr = self._run_bt("pair", address, timeout=45)
        # Some BlueZ versions return non-zero if already paired
        if not ok and "AlreadyExists" not in (stdout + stderr):
            msg = stderr or stdout or "Pair failed"
            self._set_error(msg)
            self._emit_status(
                "device_pair_failed", {"address": address, "error": msg}
            )
            return False

        self._run_bt("trust", address, timeout=15)
        device = self._upsert_device_from_info(address)
        device.paired = True
        print(f"[BT] Paired {address} ({device.name})")
        self._emit_status("device_paired", {"address": address, "name": device.name})
        return True

    def connect_device(self, address: str, retries: int = 2) -> bool:
        """Pair (if needed), connect via BlueZ, and mark as active input device."""
        address = normalize_address(address)
        self._clear_error()

        for attempt in range(1, retries + 1):
            if attempt > 1:
                print(f"[BT] Retry connect {attempt}/{retries} for {address}")
                time.sleep(1.5)

            ok, err = self._ensure_adapter()
            if not ok:
                self._set_error(err)
                continue

            info = self._get_info(address)
            if info and info.get("connected"):
                print(f"[BT] Already connected at BlueZ level: {address}")
                device = self._upsert_device_from_info(address)
                self.connected_device_address = address
                self._emit_status(
                    "device_connected", {"address": address, "name": device.name}
                )
                return True

            if not info or not info.get("paired"):
                print(f"[BT] Not paired yet — pairing {address} first")
                if not self.pair_device(address):
                    continue

            print(f"[BT] Connecting to {address}...")
            ok, stdout, stderr = self._run_bt("connect", address, timeout=30)
            if not ok:
                msg = stderr or stdout or "Connect failed"
                # Already connected is success
                if "AlreadyConnected" not in (stdout + stderr):
                    self._set_error(msg)
                    print(f"[BT] Connect failed: {msg}")
                    if attempt < retries:
                        continue
                    self._emit_status(
                        "device_connect_failed",
                        {"address": address, "error": msg},
                    )
                    return False

            # Wait briefly for HID /dev/input node
            time.sleep(1.0)
            device = self._upsert_device_from_info(address)
            if not device.connected:
                # bluetoothctl sometimes lags; re-check Connected set
                if address not in self._get_system_connected_devices():
                    msg = "BlueZ connect returned ok but device is not Connected"
                    self._set_error(msg)
                    if attempt < retries:
                        continue
                    self._emit_status(
                        "device_connect_failed",
                        {"address": address, "error": msg},
                    )
                    return False
                device.connected = True

            self.connected_device_address = address
            # Rust cookie-finder-ctl owns /dev/input/event*; web server pushes
            # the active pad via set_active_input.
            print(f"[BT] Connected and active: {address} ({device.name})")
            self._emit_status(
                "device_connected", {"address": address, "name": device.name}
            )
            return True

        return False

    def set_active_device(self, address: str) -> bool:
        """Mark a BlueZ-connected device as the active gimbal input source."""
        address = normalize_address(address)
        device = self.get_device(address)
        if not device:
            self._set_error(f"Device {address} not found")
            return False
        if not device.connected:
            return self.connect_device(address)
        self.connected_device_address = address
        self._emit_status(
            "device_connected", {"address": address, "name": device.name}
        )
        return True

    def disconnect_device(self, address: str) -> bool:
        address = normalize_address(address)
        self._clear_error()
        print(f"[BT] Disconnecting {address}...")

        if self.connected_device_address == address:
            self.connected_device_address = None

        ok, stdout, stderr = self._run_bt("disconnect", address, timeout=15)
        if not ok and "not available" not in (stdout + stderr).lower():
            # Device may already be disconnected
            if address in self._get_system_connected_devices():
                msg = stderr or stdout or "Disconnect failed"
                self._set_error(msg)
                return False

        if address in self.devices:
            self.devices[address].connected = False

        print(f"[BT] Disconnected {address}")
        self._emit_status("device_disconnected", {"address": address})
        return True

    def remove_device(self, address: str) -> bool:
        """Disconnect and forget (unpair) a device in BlueZ."""
        address = normalize_address(address)
        self._clear_error()

        if address in self._get_system_connected_devices() or (
            address in self.devices and self.devices[address].connected
        ):
            self.disconnect_device(address)

        print(f"[BT] Removing {address}...")
        ok, stdout, stderr = self._run_bt("remove", address, timeout=15)
        if not ok:
            msg = stderr or stdout or "Remove failed"
            # Already gone is fine
            if "not available" not in msg.lower() and "Does Not Exist" not in msg:
                self._set_error(msg)
                return False

        self.devices.pop(address, None)
        if self.connected_device_address == address:
            self.connected_device_address = None

        print(f"[BT] Removed {address}")
        self._emit_status("device_removed", {"address": address})
        return True

    def cleanup(self):
        if self.scanning:
            self.stop_scan()
        # Do not forcibly disconnect BlueZ devices on app shutdown —
        # leave OS pairing/connection intact.
