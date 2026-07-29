"""
Detect and switch WiFi between client and access-point modes.

Designed for Orange Pi Zero 2W (Armbian). Switching is performed by
scripts/wifi-mode.sh via sudo so the web process does not need to be root.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

AP_SSID = "cookie-finder"
AP_PASSPHRASE = "cookie-finder"
AP_GATEWAY = "192.168.12.1"
AP_URL = f"http://{AP_GATEWAY}:8000"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "wifi-mode.sh"
_RUNTIME_DIR = Path(
    os.environ.get("COOKIE_FINDER_WIFI_RUNTIME", "/run/cookie-finder-wifi")
)
_SWITCHING_MARKER = _RUNTIME_DIR / "switching"

_switch_lock = threading.Lock()
_pending_mode: str | None = None


def _run(cmd: list[str], timeout: float = 2.0) -> subprocess.CompletedProcess[str]:
    """Run a command; keep status-probe timeouts short so GPIO loop stays responsive."""
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\ntimeout after {timeout}s",
        )


def _which(name: str) -> str | None:
    """Locate a binary, including /usr/sbin (often missing from non-root PATH)."""
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/usr/sbin", "/sbin", "/usr/bin", "/bin"):
        candidate = Path(prefix) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _find_wlan_iface() -> str | None:
    """Return the primary wireless interface name, if any."""
    iw = _which("iw")
    if iw:
        result = _run([iw, "dev"])
        match = re.search(r"Interface\s+(\S+)", result.stdout or "")
        if match:
            return match.group(1)

    # Fallback: look under /sys/class/net for wireless devices
    net = Path("/sys/class/net")
    if net.is_dir():
        for entry in sorted(net.iterdir()):
            if (entry / "wireless").exists() or (entry / "phy80211").exists():
                return entry.name
    return None


def _iface_type(iface: str) -> str | None:
    """Return iw interface type: 'AP', 'managed', etc."""
    iw = _which("iw")
    if not iw:
        return None
    result = _run([iw, "dev", iface, "info"])
    match = re.search(r"^\s*type\s+(\S+)", result.stdout or "", re.MULTILINE)
    if not match:
        return None
    return match.group(1)


def _client_ssid(iface: str) -> str | None:
    """Best-effort connected SSID when in client/managed mode."""
    iw = _which("iw")
    if iw:
        result = _run([iw, "dev", iface, "link"])
        match = re.search(r"SSID:\s*(.+)", result.stdout or "")
        if match:
            ssid = match.group(1).strip()
            if ssid:
                return ssid

    nmcli = _which("nmcli")
    if nmcli:
        result = _run([nmcli, "-t", "-f", "ACTIVE,SSID", "dev", "wifi"])
        for line in (result.stdout or "").splitlines():
            if line.startswith("yes:"):
                ssid = line.split(":", 1)[1].strip()
                if ssid:
                    return ssid
    return None


def _create_ap_running(iface: str) -> bool:
    result = _run(["pgrep", "-af", "create_ap"])
    out = result.stdout or ""
    return iface in out and "create_ap" in out


def _hostapd_running(iface: str) -> bool:
    result = _run(["pgrep", "-af", "hostapd"])
    out = result.stdout or ""
    return "hostapd" in out and (iface in out or "cookie-finder" in out)


def _nm_hotspot_active() -> bool:
    nmcli = _which("nmcli")
    if not nmcli:
        return False
    result = _run([nmcli, "-t", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"])
    for line in (result.stdout or "").splitlines():
        parts = line.split(":")
        if len(parts) >= 2 and "wireless" in parts[1].lower():
            name = parts[0].lower()
            if "hotspot" in name or AP_SSID.lower() in name:
                return True
    return False


def _iface_has_ap_gateway(iface: str) -> bool:
    """True if the iface has our AP gateway address assigned."""
    result = _run(["ip", "-4", "-o", "addr", "show", "dev", iface])
    return AP_GATEWAY in (result.stdout or "")


def _supported() -> tuple[bool, str]:
    if platform.system() != "Linux":
        return False, "WiFi AP mode is only available on Linux (Orange Pi)"
    if not _find_wlan_iface():
        return False, "No wireless interface found"
    if not _SCRIPT.is_file():
        return False, f"Missing helper script: {_SCRIPT}"
    has_tool = any(
        _which(name) for name in ("create_ap", "nmcli", "hostapd")
    )
    if not has_tool:
        return False, "Install create_ap, NetworkManager (nmcli), or hostapd"
    return True, "ok"


def _external_switch_in_progress() -> bool:
    """True when wifi-mode.sh (any process) is mid-switch."""
    try:
        return _SWITCHING_MARKER.is_file()
    except OSError:
        return False


def get_wifi_status() -> dict[str, Any]:
    """Return current WiFi mode and connection details."""
    supported, reason = _supported()
    iface = _find_wlan_iface()
    mode = "unknown"
    ssid: str | None = None
    gateway: str | None = None

    if iface:
        itype = _iface_type(iface)
        if (
            itype == "AP"
            or _create_ap_running(iface)
            or _hostapd_running(iface)
            or _nm_hotspot_active()
            or _iface_has_ap_gateway(iface)
        ):
            mode = "ap"
            ssid = AP_SSID
            gateway = AP_GATEWAY
        elif itype in ("managed", "station"):
            mode = "client"
            ssid = _client_ssid(iface)
        elif itype is None:
            # No iw available — assume client unless AP backends above matched.
            mode = "client"
            ssid = _client_ssid(iface)
        else:
            mode = itype

    switching = _pending_mode is not None or _external_switch_in_progress()

    return {
        "supported": supported,
        "reason": reason if not supported else None,
        "mode": mode,
        "pending_mode": _pending_mode,
        "interface": iface,
        "ssid": ssid,
        "ap_ssid": AP_SSID,
        "ap_passphrase": AP_PASSPHRASE,
        "ap_gateway": AP_GATEWAY,
        "ap_url": AP_URL,
        "switching": switching,
    }


def _sudo_script(mode: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["COOKIE_FINDER_AP_SSID"] = AP_SSID
    env["COOKIE_FINDER_AP_PASSPHRASE"] = AP_PASSPHRASE
    env["COOKIE_FINDER_AP_GATEWAY"] = AP_GATEWAY
    # Daemon runs as root via systemd; web app uses passwordless sudo.
    if os.geteuid() == 0:
        cmd = [str(_SCRIPT), mode]
    else:
        cmd = ["sudo", "-n", str(_SCRIPT), mode]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        combined = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
        if "password is required" in combined or "a password is required" in combined:
            result.stderr = (
                (result.stderr or "")
                + "\nPasswordless sudo is required. On the Orange Pi run: make init-wifi"
            ).strip()
    return result


def _perform_switch(mode: str) -> dict[str, Any]:
    global _pending_mode
    try:
        result = _sudo_script(mode)
        # Give the radio a moment to settle before reporting status
        time.sleep(2.0)
        status = get_wifi_status()
        ok = result.returncode == 0
        message = (result.stdout or result.stderr or "").strip()
        if not ok and not message:
            message = f"wifi-mode.sh exited with code {result.returncode}"
        if ok and mode == "ap" and status.get("mode") != "ap":
            # Script succeeded but detection lagged; still treat as switching done
            status["mode"] = "ap"
            status["ssid"] = AP_SSID
            status["ap_gateway"] = AP_GATEWAY
        return {
            "status": "ok" if ok else "error",
            "requested_mode": mode,
            "message": message or f"Switched to {mode} mode",
            "wifi": status,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "requested_mode": mode,
            "message": "Timed out while switching WiFi mode",
            "wifi": get_wifi_status(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "requested_mode": mode,
            "message": str(exc),
            "wifi": get_wifi_status(),
        }
    finally:
        _pending_mode = None


def apply_boot_wifi_policy() -> dict[str, Any]:
    """
    Restore client WiFi when the GPIO daemon starts (typically at boot).

    AP mode is runtime-only: a reboot always returns to home/office WiFi so a
    crashed or partial AP switch cannot leave the Pi unreachable.
    """
    global _pending_mode

    print("[wifi] boot policy: restoring client mode (AP does not persist across reboot)")
    supported, reason = _supported()
    if not supported:
        print(f"[wifi] boot policy skipped: {reason}")
        return {"status": "skipped", "message": reason, "wifi": get_wifi_status()}

    if not _switch_lock.acquire(blocking=True, timeout=120):
        return {
            "status": "busy",
            "message": "A WiFi mode switch is already in progress",
            "wifi": get_wifi_status(),
        }

    _pending_mode = "client"
    try:
        result = _perform_switch("client")
        print(
            f"[wifi] boot policy result: {result.get('status')} "
            f"{result.get('message')}"
        )
        return result
    finally:
        _switch_lock.release()


def set_wifi_mode(mode: str, *, delay_seconds: float = 1.5) -> dict[str, Any]:
    """
    Request a WiFi mode change.

    Returns immediately after scheduling the switch so the HTTP client can
    receive a response before the radio tears down the current connection.
    """
    global _pending_mode

    mode = (mode or "").strip().lower()
    if mode not in ("ap", "client"):
        return {
            "status": "error",
            "message": "mode must be 'ap' or 'client'",
            "wifi": get_wifi_status(),
        }

    supported, reason = _supported()
    if not supported:
        return {
            "status": "error",
            "message": reason,
            "wifi": get_wifi_status(),
        }

    current = get_wifi_status()
    # Client without an SSID looks like "already client" but is not associated —
    # force a restore so we don't noop while offline.
    already = (
        current.get("mode") == mode
        and not current.get("switching")
        and not (mode == "client" and not current.get("ssid"))
    )
    if already:
        return {
            "status": "noop",
            "message": f"Already in {mode} mode",
            "wifi": current,
        }

    if not _switch_lock.acquire(blocking=False):
        return {
            "status": "busy",
            "message": "A WiFi mode switch is already in progress",
            "wifi": get_wifi_status(),
        }

    _pending_mode = mode

    def _worker() -> None:
        try:
            time.sleep(max(0.0, delay_seconds))
            _perform_switch(mode)
        finally:
            _switch_lock.release()

    threading.Thread(target=_worker, name="wifi-mode-switch", daemon=True).start()

    instructions = _switch_instructions(mode)
    return {
        "status": "switching",
        "requested_mode": mode,
        "message": f"Switching to {mode} mode…",
        "instructions": instructions,
        "wifi": get_wifi_status(),
    }


def _switch_instructions(mode: str) -> dict[str, Any]:
    if mode == "ap":
        return {
            "title": "Switch to Access Point mode?",
            "summary": (
                "The Orange Pi will stop using your home/office WiFi and broadcast "
                f"its own network named “{AP_SSID}”."
            ),
            "steps": [
                "You will lose this browser connection as soon as the switch starts.",
                f"On your phone or laptop, join WiFi “{AP_SSID}”.",
                f"Password: {AP_PASSPHRASE}",
                f"Open {AP_URL} in your browser.",
                "Use Settings → WiFi again when you want to return to client mode.",
            ],
            "ssid": AP_SSID,
            "passphrase": AP_PASSPHRASE,
            "url": AP_URL,
        }

    return {
        "title": "Switch to Client mode?",
        "summary": (
            "The Orange Pi will shut down its “cookie-finder” hotspot and reconnect "
            "to a saved WiFi network."
        ),
        "steps": [
            "You will lose this browser connection as soon as the switch starts.",
            "Reconnect your phone/laptop to your normal WiFi network.",
            "Find the Orange Pi’s new IP address (router DHCP list or SSH).",
            "Open http://<orange-pi-ip>:8000 in your browser.",
        ],
        "ssid": None,
        "passphrase": None,
        "url": None,
    }


def get_switch_instructions(mode: str) -> dict[str, Any]:
    mode = (mode or "").strip().lower()
    if mode not in ("ap", "client"):
        return {"error": "mode must be 'ap' or 'client'"}
    return _switch_instructions(mode)
