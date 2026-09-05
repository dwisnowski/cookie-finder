"""Software update from the Settings gear menu.

Fetches origin/main, reports commits behind HEAD, and schedules a oneshot
systemd unit that runs ``git merge --ff-only``, ``uv sync``, refreshes the
web unit, and restarts ``cookie-finder-web`` as the repo owner.
Installed by ``make init-software-update``.
"""

from __future__ import annotations

import json
import os
import platform
import pwd
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SETUP_COMMAND = "make init-software-update"
BRANCH = "main"
REMOTE = "origin"
UNIT_NAME = "cookie-finder-software-update.service"
UNIT_PATH = Path(f"/etc/systemd/system/{UNIT_NAME}")
INSTALLED_SCRIPT = Path("/usr/local/lib/cookie-finder/cookie-finder-software-update.sh")
STATUS_PATH = ROOT / "data" / "software-update.state"
_SCHEDULE_DELAY_S = 1.0


def _run(
    args: list[str],
    *,
    timeout: float = 120.0,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )


def _repo_owner_name() -> str | None:
    try:
        return pwd.getpwuid(ROOT.stat().st_uid).pw_name
    except (OSError, KeyError):
        return None


def _git_env_for_owner(owner: str) -> dict[str, str]:
    env = os.environ.copy()
    try:
        home = pwd.getpwnam(owner).pw_dir
    except KeyError:
        return env
    env["HOME"] = home
    env["USER"] = owner
    env["LOGNAME"] = owner
    # uv / cargo installs often live under the owner's home.
    local_bin = str(Path(home) / ".local" / "bin")
    cargo_bin = str(Path(home) / ".cargo" / "bin")
    path = env.get("PATH", "")
    env["PATH"] = f"{local_bin}:{cargo_bin}:{path}"
    return env


def _git(args: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    """Run git as the repo owner so HTTPS / gh credentials resolve."""
    base = ["git", "-C", str(ROOT), *args]
    if os.geteuid() != 0:
        return _run(base, timeout=timeout)
    owner = _repo_owner_name()
    if not owner or owner == "root":
        return _run(base, timeout=timeout)
    return _run(
        ["sudo", "-n", "-u", owner, "-H", *base],
        timeout=timeout,
        env=_git_env_for_owner(owner),
    )


def _systemctl(args: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    binary = shutil.which("systemctl") or "/usr/bin/systemctl"
    cmd = [binary, *args]
    if os.geteuid() == 0:
        return _run(cmd, timeout=timeout)
    return _run(["sudo", "-n", *cmd], timeout=timeout)


def read_apply_state() -> dict[str, Any] | None:
    if not STATUS_PATH.is_file():
        return None
    try:
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    phase = str(data.get("phase") or "")
    if not phase:
        return None
    return {
        "phase": phase,
        "message": str(data.get("message") or ""),
        "updated_at": str(data.get("updated_at") or ""),
    }


def _write_apply_state(phase: str, message: str = "") -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": phase,
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    tmp.replace(STATUS_PATH)


def capability() -> dict[str, str | bool]:
    if platform.system() != "Linux":
        return {
            "available": False,
            "reason": "Software updates are only supported on the Orange Pi (Linux).",
            "setup_command": "",
        }
    if not shutil.which("git"):
        return {
            "available": False,
            "reason": "git is not installed.",
            "setup_command": "",
        }
    if not (ROOT / ".git").is_dir():
        return {
            "available": False,
            "reason": "This install is not a git checkout.",
            "setup_command": "",
        }
    if not INSTALLED_SCRIPT.is_file():
        return {
            "available": False,
            "reason": "Software-update helper is not installed.",
            "setup_command": SETUP_COMMAND,
        }
    if not UNIT_PATH.is_file():
        return {
            "available": False,
            "reason": "Software-update systemd unit is not installed.",
            "setup_command": SETUP_COMMAND,
        }
    try:
        proc = _systemctl(["cat", UNIT_NAME])
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc), "setup_command": SETUP_COMMAND}
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": "Passwordless systemctl for the software-update unit is not configured.",
            "setup_command": SETUP_COMMAND,
        }
    return {"available": True, "reason": "", "setup_command": ""}


def _parse_commits(raw: str) -> list[dict[str, str]]:
    commits: list[dict[str, str]] = []
    if not raw.strip():
        return commits
    for line in raw.split("\0"):
        if not line.strip():
            continue
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        sha, subject, author, date = parts
        commits.append(
            {
                "sha": sha,
                "subject": subject,
                "author": author,
                "date": date,
            }
        )
    return commits


def status(*, fetch: bool = True) -> dict[str, Any]:
    """Capability + git comparison against origin/main."""
    cap = capability()
    out: dict[str, Any] = {
        **cap,
        "current_sha": None,
        "remote_sha": None,
        "behind_count": 0,
        "commits": [],
        "update_available": False,
        "dirty": False,
        "branch": None,
        "fetch_ok": False,
        "fetch_error": "",
        "apply_state": read_apply_state(),
    }

    if not (ROOT / ".git").is_dir() or not shutil.which("git"):
        return out

    try:
        head = _git(["rev-parse", "HEAD"], timeout=10)
        if head.returncode == 0:
            out["current_sha"] = head.stdout.strip()
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], timeout=10)
        if branch.returncode == 0:
            out["branch"] = branch.stdout.strip()
        dirty = _git(["status", "--porcelain"], timeout=15)
        if dirty.returncode == 0:
            out["dirty"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        out["fetch_error"] = str(exc)
        return out

    if not fetch:
        try:
            remote = _git(["rev-parse", f"{REMOTE}/{BRANCH}"], timeout=10)
            if remote.returncode == 0:
                out["remote_sha"] = remote.stdout.strip()
                out["fetch_ok"] = True
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            fetched = _git(["fetch", REMOTE, BRANCH], timeout=120)
            if fetched.returncode != 0:
                err = (fetched.stderr or fetched.stdout or "git fetch failed").strip()
                out["fetch_error"] = err
            else:
                out["fetch_ok"] = True
                remote = _git(["rev-parse", f"{REMOTE}/{BRANCH}"], timeout=10)
                if remote.returncode == 0:
                    out["remote_sha"] = remote.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            out["fetch_error"] = str(exc)

    current = out["current_sha"]
    remote_sha = out["remote_sha"]
    if current and remote_sha and current != remote_sha:
        try:
            count = _git(["rev-list", "--count", f"{current}..{remote_sha}"], timeout=15)
            if count.returncode == 0:
                out["behind_count"] = int(count.stdout.strip() or "0")
            log = _git(
                [
                    "log",
                    "--reverse",
                    "-z",
                    "--pretty=format:%h\t%s\t%an\t%cs",
                    f"{current}..{remote_sha}",
                ],
                timeout=30,
            )
            if log.returncode == 0:
                out["commits"] = _parse_commits(log.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            if not out["fetch_error"]:
                out["fetch_error"] = str(exc)

    out["update_available"] = bool(
        out["fetch_ok"]
        and out["behind_count"] > 0
        and not out["dirty"]
        and cap["available"]
    )
    if cap["available"]:
        if out["dirty"] and out["behind_count"] > 0:
            out["reason"] = (
                "Working tree has local changes; commit or discard them before updating."
            )
        elif out["fetch_error"]:
            out["reason"] = out["fetch_error"]
        elif out["behind_count"] == 0 and out["fetch_ok"]:
            out["reason"] = ""
    return out


def _guard_update_allowed(st: dict[str, Any]) -> None:
    if not st.get("available"):
        raise RuntimeError(st.get("reason") or "Software update is not available.")
    if st.get("dirty"):
        raise RuntimeError("Working tree has local changes; refuse to update.")
    if not st.get("fetch_ok"):
        raise RuntimeError(st.get("fetch_error") or "Could not fetch origin/main.")
    if not st.get("behind_count"):
        raise RuntimeError("Already up to date with origin/main.")

    apply_state = st.get("apply_state") or {}
    if apply_state.get("phase") in ("pulling", "building", "restarting", "scheduled"):
        raise RuntimeError("A software update is already in progress.")


def _start_unit() -> None:
    _systemctl(["reset-failed", UNIT_NAME])
    proc = _systemctl(["start", "--no-block", UNIT_NAME])
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "systemctl start failed").strip()
        _write_apply_state("error", err)
        raise RuntimeError(err)


def request_update() -> dict[str, Any]:
    """Validate, write scheduled state, and start the oneshot after the HTTP reply."""
    st = status(fetch=True)
    _guard_update_allowed(st)
    _write_apply_state("scheduled", "Update accepted; starting shortly…")

    def _kick() -> None:
        try:
            _start_unit()
        except Exception as exc:  # noqa: BLE001 — surface into status file for UI
            _write_apply_state("error", str(exc))

    threading.Timer(_SCHEDULE_DELAY_S, _kick).start()
    return {
        "status": "updating",
        "delay_ms": int(_SCHEDULE_DELAY_S * 1000),
        "current_sha": st.get("current_sha"),
        "remote_sha": st.get("remote_sha"),
        "behind_count": st.get("behind_count"),
    }
