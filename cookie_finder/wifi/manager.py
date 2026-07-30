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
# Open SoftAP by default — WPA SoftAP on Zero 2W (UWE5622) often fails phone joins.
AP_PASSPHRASE = ""
AP_GATEWAY = "192.168.12.1"
AP_URL = f"http://{AP_GATEWAY}:8000"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "wifi-mode.sh"
_RUNTIME_DIR = Path(
    os.environ.get("COOKIE_FINDER_WIFI_RUNTIME", "/run/cookie-finder-wifi")
)
_SWITCHING_MARKER = _RUNTIME_DIR / "switching"
_STATE_DIR = Path(
    os.environ.get("COOKIE_FINDER_WIFI_STATE", "/var/lib/cookie-finder")
)
_LAST_BOOT_ID_FILE = _STATE_DIR / "wifi-last-boot-id"
_DESIRED_MODE_FILE = _STATE_DIR / "wifi-desired-mode"
_BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")

_switch_lock = threading.Lock()
_pending_mode: str | None = None


def _current_boot_id() -> str | None:
    try:
        return _BOOT_ID_FILE.read_text().strip() or None
    except OSError:
        return None


def get_desired_mode() -> str:
    """Last requested mode (ap/client). Survives daemon restarts within a boot."""
    try:
        text = _DESIRED_MODE_FILE.read_text().strip().lower()
        if text in ("ap", "client"):
            return text
    except OSError:
        pass
    return "client"


def set_desired_mode(mode: str) -> None:
    mode = (mode or "").strip().lower()
    if mode not in ("ap", "client"):
        return
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _DESIRED_MODE_FILE.write_text(mode + "\n")
    except OSError as exc:
        print(f"[wifi] warning: could not persist desired mode: {exc}")


def _is_new_boot() -> bool:
    """True once per kernel boot (first wifi daemon start after reboot)."""
    boot_id = _current_boot_id()
    if not boot_id:
        # No boot_id (non-Linux) — treat as new boot so we still restore client.
        return True
    try:
        previous = _LAST_BOOT_ID_FILE.read_text().strip()
        if previous == boot_id:
            return False
    except OSError:
        pass
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        _LAST_BOOT_ID_FILE.write_text(boot_id + "\n")
    except OSError as exc:
        print(f"[wifi] warning: could not persist boot id: {exc}")
    return True


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
        if len(parts) < 2:
            continue
        name = parts[0].lower()
        # Our explicit SoftAP profile, generic Hotspot, or SSID-named conn.
        if (
            name in ("cookie-finder-ap", "hotspot")
            or "hotspot" in name
            or AP_SSID.lower() in name
        ):
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
        "ap_passphrase": AP_PASSPHRASE or None,
        "open_network": not bool(AP_PASSPHRASE),
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


def _log_script_output(result: subprocess.CompletedProcess[str]) -> None:
    out = (result.stdout or "").strip()
    err = (result.stderr or "").strip()
    if out:
        for line in out.splitlines():
            print(line if line.startswith("[wifi-mode]") else f"[wifi-mode] {line}")
    if err:
        for line in err.splitlines():
            print(f"[wifi-mode:err] {line}")


def _perform_switch(mode: str) -> dict[str, Any]:
    global _pending_mode
    try:
        print(f"[wifi] running wifi-mode.sh {mode}")
        result = _sudo_script(mode)
        _log_script_output(result)
        # Give the radio a moment to settle before reporting status
        time.sleep(2.0)
        status = get_wifi_status()
        ok = result.returncode == 0
        message = (result.stdout or result.stderr or "").strip()
        if not ok and not message:
            message = f"wifi-mode.sh exited with code {result.returncode}"
        print(
            f"[wifi] switch to {mode}: "
            f"{'OK' if ok else 'FAILED'} (exit {result.returncode}); "
            f"settled mode={status.get('mode')!r} ssid={status.get('ssid')!r}"
        )
        if ok and mode == "ap" and status.get("mode") != "ap":
            print(
                "[wifi] WARNING: AP script exited 0 but radio is not in AP mode — "
                "check /run/cookie-finder-wifi/hostapd.log"
            )
            ok = False
            message = (
                "AP backend exited successfully but SoftAP did not stay up. "
                "See hostapd.log on the Pi."
            )
        return {
            "status": "ok" if ok else "error",
            "requested_mode": mode,
            "message": message or f"Switched to {mode} mode",
            "wifi": status,
            "stdout": (result.stdout or "").strip(),
            "stderr": (result.stderr or "").strip(),
        }
    except subprocess.TimeoutExpired:
        print(f"[wifi] switch to {mode} timed out")
        return {
            "status": "error",
            "requested_mode": mode,
            "message": "Timed out while switching WiFi mode",
            "wifi": get_wifi_status(),
        }
    except Exception as exc:
        print(f"[wifi] switch to {mode} crashed: {exc}")
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
    WiFi policy when the GPIO daemon starts.

    - **New reboot:** restore client (AP does not persist across power cycle).
    - **Same boot** (``systemctl restart``): honor *desired* mode. If the user
      requested AP, re-apply AP instead of tearing a failed/partial AP down
      as an "unhealthy client" repair.
    """
    global _pending_mode

    supported, reason = _supported()
    if not supported:
        print(f"[wifi] boot policy skipped: {reason}")
        return {"status": "skipped", "message": reason, "wifi": get_wifi_status()}

    current = get_wifi_status()
    new_boot = _is_new_boot()
    desired = get_desired_mode()
    mode = current.get("mode")
    ssid = current.get("ssid")
    healthy_ap = mode == "ap"
    healthy_client = mode == "client" and bool(ssid)

    if new_boot:
        set_desired_mode("client")
        if healthy_client:
            print(f"[wifi] boot policy: new boot, already client on {ssid}; skip restore")
            return {
                "status": "noop",
                "message": f"Already connected as client to {ssid}",
                "wifi": current,
            }
        print(
            f"[wifi] boot policy: new boot — restoring client "
            f"(was mode={mode!r} ssid={ssid!r})"
        )
        target = "client"
    elif desired == "ap":
        if healthy_ap:
            print("[wifi] boot policy: same boot, desired AP already up; leave alone")
            return {
                "status": "noop",
                "message": "Already in AP mode",
                "wifi": current,
            }
        print(
            f"[wifi] boot policy: same boot, desired AP but radio is "
            f"{mode!r}/{ssid!r} — re-applying AP"
        )
        target = "ap"
    elif healthy_client or healthy_ap:
        print(
            f"[wifi] boot policy: same boot, leave alone "
            f"(desired={desired!r} mode={mode!r} ssid={ssid!r})"
        )
        return {
            "status": "noop",
            "message": f"Already healthy {mode} mode; left alone",
            "wifi": current,
        }
    else:
        print(
            f"[wifi] boot policy: same boot, desired client, radio unhealthy "
            f"({mode!r}/{ssid!r}) — repairing client"
        )
        target = "client"

    if not _switch_lock.acquire(blocking=True, timeout=120):
        return {
            "status": "busy",
            "message": "A WiFi mode switch is already in progress",
            "wifi": get_wifi_status(),
        }

    set_desired_mode(target)
    _pending_mode = target
    try:
        result = _perform_switch(target)
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
        set_desired_mode(mode)
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

    set_desired_mode(mode)
    _pending_mode = mode

    def _worker() -> None:
        try:
            time.sleep(max(0.0, delay_seconds))
            result = _perform_switch(mode)
            print(
                f"[wifi] switch worker done: {result.get('status')} "
                f"{result.get('message')}"
            )
        except Exception as exc:
            print(f"[wifi] switch worker crashed: {exc}")
            _pending_mode = None
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
                f"its own open network named “{AP_SSID}” (no password)."
            ),
            "steps": [
                "You will lose this browser connection as soon as the switch starts.",
                f"On your phone or laptop, join WiFi “{AP_SSID}” (open — no password).",
                f"Open {AP_URL} in your browser.",
                "Use Settings → WiFi again when you want to return to client mode.",
            ],
            "ssid": AP_SSID,
            "passphrase": AP_PASSPHRASE or None,
            "open_network": not bool(AP_PASSPHRASE),
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
