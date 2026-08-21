"""
Request a graceful system power-off.

The WiFi GPIO daemon owns the LED. This module writes
``/run/cookie-finder-wifi/powering-off`` so that daemon can play the shutdown
chirp, then runs ``scripts/system-power.sh`` (root or passwordless sudo).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "system-power.sh"
_RUNTIME_DIR = Path(
    os.environ.get("COOKIE_FINDER_WIFI_RUNTIME", "/run/cookie-finder-wifi")
)
POWERING_OFF_MARKER = _RUNTIME_DIR / "powering-off"

_lock = threading.Lock()
_pending = False


def is_powering_off() -> bool:
    """True when this process or another has started a shutdown."""
    if _pending:
        return True
    try:
        return POWERING_OFF_MARKER.is_file()
    except OSError:
        return False


def write_powering_off_marker() -> None:
    """Create the LED-override marker (best-effort)."""
    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        POWERING_OFF_MARKER.write_text(f"{os.getpid()}\n")
    except OSError as exc:
        print(f"[poweroff] could not write marker: {exc}")


def _clear_powering_off_marker() -> None:
    try:
        POWERING_OFF_MARKER.unlink(missing_ok=True)
    except OSError:
        pass


def request_poweroff(*, delay_seconds: float = 1.5) -> dict[str, Any]:
    """
    Schedule ``systemctl poweroff`` after writing the LED marker.

    Returns immediately so an HTTP client can show a shutting-down message
    before the board halts. ``delay_seconds`` is extra wait *before* the
    privileged script (which itself waits so the LED chirp is visible).
    """
    global _pending

    if is_powering_off():
        return {
            "status": "busy",
            "message": "A shutdown is already in progress",
            "powering_off": True,
        }

    if not _lock.acquire(blocking=False):
        return {
            "status": "busy",
            "message": "A shutdown is already in progress",
            "powering_off": True,
        }

    if is_powering_off():
        _lock.release()
        return {
            "status": "busy",
            "message": "A shutdown is already in progress",
            "powering_off": True,
        }

    write_powering_off_marker()
    _pending = True

    def _worker() -> None:
        global _pending
        try:
            time.sleep(max(0.0, delay_seconds))
            result = _run_poweroff_script()
            _log_script_output(result)
            if result.returncode != 0:
                err = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
                if "already in progress" not in err:
                    print(
                        f"[poweroff] script failed (rc={result.returncode}); "
                        "clearing LED marker"
                    )
                    _clear_powering_off_marker()
                    _pending = False
        except Exception as exc:
            print(f"[poweroff] worker crashed: {exc}")
            _clear_powering_off_marker()
            _pending = False
        finally:
            _lock.release()

    threading.Thread(target=_worker, name="system-poweroff", daemon=True).start()
    return {
        "status": "shutting_down",
        "message": "Shutting down… watch the WiFi LED (slow → fast → slow).",
        "powering_off": True,
    }


def _run_poweroff_script() -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if os.geteuid() == 0:
        cmd = [str(_SCRIPT), "poweroff"]
    else:
        cmd = ["sudo", "-n", str(_SCRIPT), "poweroff"]
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
            print(line if line.startswith("[system-power]") else f"[system-power] {line}")
    if err:
        for line in err.splitlines():
            print(line if line.startswith("[system-power]") else f"[system-power] {line}")
