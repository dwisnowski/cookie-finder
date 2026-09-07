"""
Start / stop the Cookie Finder Cloudflare *quick* tunnel.

Uses ``cookie-finder-cloudflared.service`` which runs::

    cloudflared tunnel --url http://127.0.0.1:80

That publishes a free ``*.trycloudflare.com`` hostname (no Cloudflare account
or custom domain). The Connect UI reads the hostname from the local metrics
endpoint ``/quicktunnel``.

Install / sudoers: ``make on-the-pi-init-cloudflare-tunnel``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

UNIT = "cookie-finder-cloudflared.service"
LEGACY_UNIT = "cloudflared.service"
SETUP_COMMAND = "make on-the-pi-init-cloudflare-tunnel"
SUDOERS_PATH = "/etc/sudoers.d/cookie-finder-cloudflare"
METRICS_HOST = "127.0.0.1"
METRICS_PORT = 20241
ENV_URL_KEY = "CLOUDFLARE_TUNNEL_URL"

_HOSTNAME_METRIC_RE = re.compile(
    r'cloudflared_tunnel_user_hostnames_counts\{[^}]*userHostname="([^"]+)"'
)


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


def _unit_exists(unit: str = UNIT) -> bool | None:
    try:
        proc = _run_systemctl(["cat", unit], timeout=3.0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return proc.returncode == 0


def _unit_active(unit: str = UNIT) -> bool | None:
    try:
        proc = _run_systemctl(["is-active", "--quiet", unit], timeout=3.0)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return proc.returncode == 0


def _can_control() -> tuple[bool, str]:
    """Probe passwordless sudo for enable/disable of the quick-tunnel unit."""
    if os.geteuid() == 0:
        return True, ""
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


def _normalize_tunnel_url(raw: str) -> str | None:
    value = (raw or "").strip().strip('"').strip("'")
    if not value:
        return None
    if "://" not in value:
        value = f"https://{value}"
    if not value.startswith(("http://", "https://")):
        return None
    return value if value.endswith("/") else f"{value}/"


def _http_get_text(url: str, *, timeout: float = 0.4) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _url_from_metrics_port(port: int) -> str | None:
    base = f"http://{METRICS_HOST}:{port}"
    quick = _http_get_text(f"{base}/quicktunnel")
    if quick is not None:
        try:
            hostname = (json.loads(quick).get("hostname") or "").strip()
        except json.JSONDecodeError:
            hostname = ""
        if hostname:
            return _normalize_tunnel_url(hostname)

    metrics = _http_get_text(f"{base}/metrics")
    if not metrics or "cloudflared_tunnel_" not in metrics:
        return None
    match = _HOSTNAME_METRIC_RE.search(metrics)
    if match:
        return _normalize_tunnel_url(match.group(1))
    return None


def discover_running_tunnel_url() -> str | None:
    """Public URL from the quick-tunnel metrics server (``/quicktunnel``)."""
    return _url_from_metrics_port(METRICS_PORT)


def wait_for_tunnel_url(*, timeout: float = 20.0, interval: float = 0.5) -> str | None:
    """Poll metrics until trycloudflare assigns a hostname (or timeout)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        url = discover_running_tunnel_url()
        if url:
            return url
        time.sleep(interval)
    return discover_running_tunnel_url()


def configured_tunnel_url() -> str | None:
    """Optional override from env (named-tunnel / fixed hostname setups)."""
    raw = os.environ.get(ENV_URL_KEY, "").strip()
    return _normalize_tunnel_url(raw)


def tunnel_url(*, running: bool | None = None) -> str | None:
    """Public tunnel URL for the Connect UI."""
    discovered = discover_running_tunnel_url()
    if discovered:
        return discovered
    if running is None:
        running = _unit_active()
    if running:
        return configured_tunnel_url()
    return None


def status() -> dict[str, Any]:
    """Install + running state for the UI badge / Settings toggle / Connect."""
    unit_exists = _unit_exists()
    running = bool(_unit_active())
    binary = shutil.which("cloudflared") is not None
    installed = bool(unit_exists) or binary
    controllable = False
    control_error = ""

    if unit_exists:
        controllable, control_error = _can_control()
    elif unit_exists is False:
        control_error = (
            f"{UNIT} is not installed. Run: {SETUP_COMMAND}"
        )

    url = tunnel_url(running=running)
    return {
        "running": running,
        "installed": installed,
        "unit": UNIT,
        "unit_exists": unit_exists,
        "binary": binary,
        "controllable": controllable,
        "control_error": control_error,
        "setup_command": SETUP_COMMAND,
        "url": url,
        "mode": "quick",
        "metrics": f"http://{METRICS_HOST}:{METRICS_PORT}/quicktunnel",
    }


def start() -> dict[str, Any]:
    """Enable and start the quick-tunnel service; wait briefly for a URL."""
    if _unit_exists() is False:
        return {
            "status": "error",
            "message": f"{UNIT} is not installed. Run: {SETUP_COMMAND}",
            "cloudflare": status(),
        }
    # Named-token connector fights the quick tunnel for the same metrics habits;
    # stop it if somehow still active (init normally uninstalls it).
    if _unit_active(LEGACY_UNIT):
        try:
            _run_systemctl(
                ["disable", "--now", LEGACY_UNIT], timeout=30.0, use_sudo=True
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
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

    url = wait_for_tunnel_url(timeout=18.0)
    payload = status()
    if url:
        payload["url"] = url
    return {
        "status": "ok",
        "message": (
            f"Quick tunnel ready: {url}" if url else "Quick tunnel started (URL pending)"
        ),
        "cloudflare": payload,
    }


def stop() -> dict[str, Any]:
    """Disable and stop the quick-tunnel service (stays off across reboot)."""
    if _unit_exists() is False:
        return {
            "status": "error",
            "message": f"{UNIT} is not installed. Run: {SETUP_COMMAND}",
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
        "message": "Cloudflare quick tunnel stopped",
        "cloudflare": status(),
    }
