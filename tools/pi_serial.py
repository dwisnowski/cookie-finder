#!/usr/bin/env python3
"""Connect and deploy to Orange Pi over USB-TTL serial (UART0)."""

from __future__ import annotations

import argparse
import base64
import os
import random
import re
import sys
import tarfile
import tempfile
import time
from pathlib import Path

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: uv sync", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_DIR = "~/cookie-finder"

PROMPT_RE = re.compile(
    rb"(?:[\r\n]|^)(?:\033\[[0-9;?]*[A-Za-z])*(?:cookie|[\w.-]+)@[\w.-]+:[^\r\n]*[$#](?:\033\[[0-9;?]*[A-Za-z])*[\s\r\n]"
)
LOGIN_RE = re.compile(rb"login:\s*$", re.MULTILINE | re.IGNORECASE)
PASSWORD_RE = re.compile(rb"[Pp]assword:\s*$", re.MULTILINE)

TAR_EXCLUDES = {
    ".git",
    ".venv",
    "__pycache__",
    "cookie_finder_rust/target",
    ".serial.env",
    ".DS_Store",
    "thermal_capture.tiff",
}


def load_serial_env() -> None:
    env_file = REPO_ROOT / ".serial.env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def read_available(port: serial.Serial, timeout: float = 0.2) -> bytes:
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        waiting = port.in_waiting
        if waiting:
            chunks.append(port.read(waiting))
            deadline = time.monotonic() + 0.05
        else:
            time.sleep(0.02)
    return b"".join(chunks)


def at_shell_prompt(buf: bytes) -> bool:
    return bool(PROMPT_RE.search(buf))


class PiSerialSession:
    def __init__(
        self,
        device: str,
        baud: int,
        user: str,
        password: str,
        echo: bool = True,
    ) -> None:
        self.device = device
        self.baud = baud
        self.user = user
        self.password = password
        self.echo = echo
        self.port = serial.Serial(device, baudrate=baud, timeout=0.1)

    def close(self) -> None:
        if self.port.is_open:
            self.port.close()

    def __enter__(self) -> PiSerialSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _log(self, data: bytes) -> None:
        if self.echo and data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()

    def _write_line(self, text: str) -> None:
        payload = (text.rstrip("\n") + "\n").encode()
        self.port.write(payload)

    def _drain(self, seconds: float = 0.4) -> bytes:
        time.sleep(seconds)
        data = read_available(self.port, timeout=seconds)
        self._log(data)
        return data

    def login(self, timeout: float = 45.0) -> None:
        self.port.reset_input_buffer()
        self._write_line("")
        self._write_line("")
        buf = self._drain(0.6)

        if at_shell_prompt(buf):
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if LOGIN_RE.search(buf):
                self._write_line(self.user)
                buf = self._drain(0.8)
                continue
            if PASSWORD_RE.search(buf):
                if not self.password:
                    raise RuntimeError(
                        "Serial password required. Set SERIAL_PASSWORD or add it to .serial.env"
                    )
                self._write_line(self.password)
                buf = self._drain(1.0)
                continue
            if at_shell_prompt(buf):
                return
            self._write_line("")
            buf += self._drain(0.5)

        raise TimeoutError("Could not log in over serial (no shell prompt detected)")

    def run(self, command: str, timeout: float = 300.0) -> tuple[int, str]:
        token = f"__CF_{random.randint(0, 999_999)}__"
        wrapped = f"{command}; printf '\\n{token}%s\\n' $?"
        self._write_line(wrapped)

        deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < deadline:
            buf += read_available(self.port, timeout=0.2)
            text = buf.decode("utf-8", errors="replace")
            match = re.search(rf"\n{re.escape(token)}(\d+)\n", text)
            if match:
                output = text[: match.start()]
                return int(match.group(1)), output.strip()
            time.sleep(0.02)

        raise TimeoutError(f"Command timed out: {command!r}")

    def run_checked(self, command: str, timeout: float = 300.0) -> str:
        code, output = self.run(command, timeout=timeout)
        if code != 0:
            raise RuntimeError(f"Remote command failed ({code}): {command}\n{output}")
        return output


def create_project_tarball() -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="cookie-finder-", suffix=".tar.gz", delete=False)
    tmp.close()
    tar_path = Path(tmp.name)

    def filter_member(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = Path(tarinfo.name).parts
        if any(part in TAR_EXCLUDES for part in parts):
            return None
        if tarinfo.name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    with tarfile.open(tar_path, "w:gz") as tar:
        for item in REPO_ROOT.iterdir():
            tar.add(item, arcname=item.name, filter=filter_member)

    return tar_path


def transfer_file_base64(session: PiSerialSession, local_path: Path, remote_path: str) -> None:
    payload = base64.b64encode(local_path.read_bytes()).decode("ascii")
    chunk_size = 900
    session.run_checked(f"rm -f {remote_path} {remote_path}.b64")
    for offset in range(0, len(payload), chunk_size):
        chunk = payload[offset : offset + chunk_size]
        session.run_checked(f"printf '%s' '{chunk}' >> {remote_path}.b64", timeout=120.0)
        print(f"  transferred {min(offset + chunk_size, len(payload))}/{len(payload)} bytes (base64)")
    session.run_checked(f"base64 -d {remote_path}.b64 > {remote_path} && rm -f {remote_path}.b64")


def deploy_project(device: str, baud: int, user: str, password: str, remote_dir: str) -> None:
    remote_dir = remote_dir.rstrip("/")
    tarball = create_project_tarball()
    remote_tar = "/tmp/cookie-finder-deploy.tar.gz"
    print(f"Created {tarball} ({tarball.stat().st_size / 1024 / 1024:.1f} MB)")
    print("Transferring project over serial (base64). This may take several minutes...")

    try:
        with PiSerialSession(device, baud, user, password) as session:
            session.login()
            transfer_file_base64(session, tarball, remote_tar)
            session.run_checked(f"mkdir -p {remote_dir}")
            session.run_checked(f"tar xzf {remote_tar} -C {remote_dir}")
            session.run_checked(f"rm -f {remote_tar}")
            session.run_checked(f"cd {remote_dir} && uv sync")
    finally:
        tarball.unlink(missing_ok=True)

    print(f"Deploy complete: {remote_dir}")


def deploy_file(device: str, baud: int, user: str, password: str, local_path: Path, remote_path: str) -> None:
    if not local_path.is_file():
        raise SystemExit(f"Local file not found: {local_path}")

    remote_tar = f"/tmp/{local_path.name}"
    print(f"Deploying {local_path} -> {remote_path}")

    with PiSerialSession(device, baud, user, password) as session:
        session.login()
        transfer_file_base64(session, local_path, remote_tar)
        session.run_checked(f"mkdir -p $(dirname {remote_path})")
        session.run_checked(f"mv {remote_tar} {remote_path} && chmod 755 {remote_path}")

    print("File deploy complete")


def build_parser() -> argparse.ArgumentParser:
    load_serial_env()
    default_device = os.environ.get("SERIAL_DEVICE", "/dev/tty.usbserial-BG01PPKN")
    default_baud = int(os.environ.get("SERIAL_BAUD", "115200"))
    default_user = os.environ.get("SERIAL_USER", "cookie")
    default_password = os.environ.get("SERIAL_PASSWORD", "")
    default_remote = os.environ.get("SERIAL_REMOTE_DIR", DEFAULT_REMOTE_DIR)

    parser = argparse.ArgumentParser(description="Orange Pi serial console and deploy helper")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--baud", type=int, default=default_baud)
    parser.add_argument("--user", default=default_user)
    parser.add_argument("--password", default=default_password)
    parser.add_argument("--remote-dir", default=default_remote)

    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Log in and run a remote shell command")
    run_parser.add_argument("remote_command", nargs="+", help="Command to run on the Pi")

    sub.add_parser("deploy", help="Sync project to Pi over serial")

    file_parser = sub.add_parser("deploy-file", help="Copy a single file to the Pi")
    file_parser.add_argument("local_path", type=Path)
    file_parser.add_argument("remote_path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        command = " ".join(args.remote_command)
        with PiSerialSession(args.device, args.baud, args.user, args.password) as session:
            session.login()
            code, output = session.run(command)
            if output:
                print(output)
            if code != 0:
                raise SystemExit(code)
        return

    if args.command == "deploy":
        deploy_project(
            args.device,
            args.baud,
            args.user,
            args.password,
            args.remote_dir,
        )
        return

    if args.command == "deploy-file":
        deploy_file(
            args.device,
            args.baud,
            args.user,
            args.password,
            args.local_path,
            args.remote_path,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
