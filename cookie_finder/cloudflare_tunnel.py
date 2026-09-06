"""
Start / stop the Cloudflare Tunnel connector (cloudflared.service).

The web UI calls these helpers via passwordless sudo configured by
``make on-the-pi-init-cloudflare-tunnel``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

UNIT = "cloudflared.service"
SETUP_COMMAND = "make on-the-pi-init-cloudflare-tunnel"
SUDOERS_PATH = "/etc/sudoers.d/cookie-finder-cloudflare"


def _systemctl_bin() -> str:
    return shutil.which("systemctl") or "/usr/bin/systemctl"


def _run_systemctl(
    args: list[str],
    *,
    timeout: float = 30.0,
    use_sudo: bool = False,
) -> subprocess.CompletedProcess[str]:
    binary = _systemctl_bin()
    if use_sudo and os.geteuid() != 0:
        cmd = ["sudo", "-n", binary, *args]
    else:
        cmd = [binary, *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _unit_exists() -> bool | None:
    try:
        proc = _run_systemctl(["cat", UNIT], timeout=3.0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return proc.returncode == 0


def _unit_active() -> bool | None:
    try:
        proc = _run_systemctl(["is-active", "--quiet", UNIT], timeout=3.0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return proc.returncode == 0


def _can_control() -> tuple[bool, str]:
    """Probe passwordless sudo for enable/disable of cloudflared."""
    if os.geteuid() == 0:
        return True, ""
    # Fast path: init target writes this sudoers drop-in.
    if os.path.isfile(SUDOERS_PATH):
        return True, ""
    try:
        proc = _run_systemctl(["is-enabled", UNIT], timeout=3.0, use_sudo=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)

    combined = ((proc.stderr or "") + (proc.stdout or "")).lower()
    if (
        "password is required" in combined
        or "a password is required" in combined
        or "not allowed to execute" in combined
    ):
        return False, (
            "Passwordless sudo for cloudflared is not configured. "
            f"Run: {SETUP_COMMAND}"
        )
    return True, ""


def status() -> dict[str, Any]:
    """Install + running state for the UI badge / Settings toggle."""
    unit_exists = _unit_exists()
    running = _unit_active()
    binary = shutil.which("cloudflared") is not None
    installed = bool(unit_exists) or binary
    controllable = False
    control_error = ""

    if unit_exists:
        controllable, control_error = _can_control()
    elif unit_exists is False:
        control_error = (
            f"cloudflared.service is not installed. Run: {SETUP_COMMAND}"
        )

    return {
        "running": bool(running),
        "installed": installed,
        "unit": UNIT,
        "unit_exists": unit_exists,
        "binary": binary,
        "controllable": controllable,
        "control_error": control_error,
        "setup_command": SETUP_COMMAND,
    }


def start() -> dict[str, Any]:
    """Enable and start cloudflared.service."""
    if _unit_exists() is False:
        return {
            "status": "error",
            "message": f"cloudflared.service is not installed. Run: {SETUP_COMMAND}",
            "cloudflare": status(),
        }
    try:
        proc = _run_systemctl(["enable", "--now", UNIT], timeout=45.0, use_sudo=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "error", "message": str(exc), "cloudflare": status()}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "systemctl enable --now failed").strip()
        if "password is required" in err.lower():
            err = (
                "Passwordless sudo for cloudflared is not configured. "
                f"Run: {SETUP_COMMAND}"
            )
        return {"status": "error", "message": err, "cloudflare": status()}
    return {
        "status": "ok",
        "message": "Cloudflare Tunnel started",
        "cloudflare": status(),
    }


def stop() -> dict[str, Any]:
    """Disable and stop cloudflared.service (stays off across reboot)."""
    if _unit_exists() is False:
        return {
            "status": "error",
            "message": f"cloudflared.service is not installed. Run: {SETUP_COMMAND}",
            "cloudflare": status(),
        }
    try:
        proc = _run_systemctl(["disable", "--now", UNIT], timeout=45.0, use_sudo=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "error", "message": str(exc), "cloudflare": status()}
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "systemctl disable --now failed").strip()
        if "password is required" in err.lower():
            err = (
                "Passwordless sudo for cloudflared is not configured. "
                f"Run: {SETUP_COMMAND}"
            )
        return {"status": "error", "message": err, "cloudflare": status()}
    return {
        "status": "ok",
        "message": "Cloudflare Tunnel stopped",
        "cloudflare": status(),
    }
