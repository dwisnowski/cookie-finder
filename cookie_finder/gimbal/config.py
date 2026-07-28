"""Load and save gimbal.toml phase-order configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Optional

DEFAULT_PHASE_ORDER = [0, 1, 2, 3]
DEFAULT_CONFIG_RELATIVE = Path("config/gimbal.toml")
SYSTEM_CONFIG = Path("/etc/cookie-finder/gimbal.toml")
_ARRAY_RE = re.compile(r"^\s*([a-z_]+)\s*=\s*\[([^\]]*)\]\s*$")

MotorName = Literal["pan", "tilt"]


def resolve_config_path(explicit: Optional[str | Path] = None) -> Path:
    """Resolve config path using the same order as cookie-finder-ctl."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("COOKIE_FINDER_CONFIG", "").strip()
    if env:
        return Path(env)
    local = DEFAULT_CONFIG_RELATIVE
    if local.exists():
        return local
    if SYSTEM_CONFIG.exists():
        return SYSTEM_CONFIG
    return local


def _parse_gimbal_toml(text: str) -> dict[str, list[int]]:
    section: dict[str, list[int]] = {}
    in_gimbal = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[gimbal]":
            in_gimbal = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_gimbal = False
            continue
        if not in_gimbal:
            continue
        match = _ARRAY_RE.match(line)
        if not match:
            continue
        key, values = match.group(1), match.group(2)
        section[key] = [int(part.strip()) for part in values.split(",") if part.strip()]
    return section


def _validate_phase_order(order: list[int], name: str) -> list[int]:
    if len(order) != 4:
        raise ValueError(f"{name} must have exactly 4 elements, got {len(order)}")
    seen = set()
    for i, value in enumerate(order):
        if value < 0 or value > 3:
            raise ValueError(f"{name}[{i}] = {value}: each value must be 0-3")
        if value in seen:
            raise ValueError(f"{name}: duplicate phase index {value}")
        seen.add(value)
    return list(order)


def load_gimbal_config(config_path: Optional[str | Path] = None) -> dict[str, list[int]]:
    """Load pan/tilt phase orders from TOML, or return defaults."""
    path = resolve_config_path(config_path)
    if not path.exists():
        return {
            "pan_phase_order": list(DEFAULT_PHASE_ORDER),
            "tilt_phase_order": list(DEFAULT_PHASE_ORDER),
        }

    section = _parse_gimbal_toml(path.read_text(encoding="utf-8"))
    return {
        "pan_phase_order": _validate_phase_order(
            list(section.get("pan_phase_order", DEFAULT_PHASE_ORDER)),
            "pan_phase_order",
        ),
        "tilt_phase_order": _validate_phase_order(
            list(section.get("tilt_phase_order", DEFAULT_PHASE_ORDER)),
            "tilt_phase_order",
        ),
    }


def _format_toml(config: dict[str, list[int]]) -> str:
    pan = ", ".join(str(v) for v in config["pan_phase_order"])
    tilt = ", ".join(str(v) for v in config["tilt_phase_order"])
    return (
        "[gimbal]\n"
        "# Maps IN1..IN4 outputs to logical step phases 0..3.\n"
        "# Use make on-the-pi-rust-keyboard and press W to save after wiring discovery.\n"
        f"pan_phase_order = [{pan}]\n"
        f"tilt_phase_order = [{tilt}]\n"
    )


def save_phase_order(
    motor: MotorName,
    order: list[int],
    config_path: Optional[str | Path] = None,
) -> Path:
    """Update one motor's phase order in the config file (atomic write)."""
    key = f"{motor}_phase_order"
    validated = _validate_phase_order(list(order), key)
    path = resolve_config_path(config_path)

    if path.exists():
        section = _parse_gimbal_toml(path.read_text(encoding="utf-8"))
    else:
        section = {}

    section[key] = validated
    path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "pan_phase_order": _validate_phase_order(
            list(section.get("pan_phase_order", DEFAULT_PHASE_ORDER)),
            "pan_phase_order",
        ),
        "tilt_phase_order": _validate_phase_order(
            list(section.get("tilt_phase_order", DEFAULT_PHASE_ORDER)),
            "tilt_phase_order",
        ),
    }
    config[key] = validated

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(_format_toml(config), encoding="utf-8")
    tmp.replace(path)
    return path


def format_phase_order_snippet(motor: MotorName, order: list[int]) -> str:
    """Return a TOML line for preview (Y key)."""
    key = f"{motor}_phase_order"
    values = ", ".join(str(v) for v in order)
    return f"{key} = [{values}]"
