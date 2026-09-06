#!/usr/bin/env python3
"""
Mileseey T-Recon / PixFra (VID 0x2E03) CDC control framing.

Reversed from com.mileseey.trecon 1.1.6 (libSendDataLib.so PacketUtils::makeCmd).

Wire format (host → device, bulk OUT on CDC data interface):

  0x6E 0x00 | flags | cmd | len_hi len_lo | hdr_chk_hi hdr_chk_lo | payload… | crc_hi crc_lo

- Magic: 0x6E 0x00
- flags: 0x00 for outbound app commands
- cmd: FunctionCode (see below)
- length: big-endian payload size
- header check: CRC16-CCITT table fold over (cmd, length) (see _header_check)
- trailer: CRC16-CCITT over header+payload (init 0, poly 0x1021)

App USB filter matches:
  VID 0x2E03 (11779) + PID 0x2507 (9479)  # TNV30i / PixFra THERMAL USB
  VID 0x2E03 + PID 0x2508 (9480)
  VID 0x04B4 (1204) + PID 0x00F1 (241)

Note: In T-Recon UI, palette taps call CameraNative.setColor() (host-side
recolor of the UVC stream). Device palette via cmd 0x10 exists in the
protocol but is not wired from that UI path. FFC / image-mode / brightness
are sent on CDC.

Gain-auto (0x3E) and temperature/radiometry queries (0x50 + subcommand) exist
in the APK packet layer but are not used by the T-Recon preview UI in 1.1.6 —
included here for live probing on the Pi.
"""

from __future__ import annotations

import argparse
import sys
from typing import Union


def _ccitt_table() -> list[int]:
    table: list[int] = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return table


TABLE = _ccitt_table()

# FunctionCode.java (signed bytes shown as unsigned)
CMD_SAVE = 0x01
CMD_RESTART = 0x02
CMD_NOISE_2D = 0x03
CMD_NOISE_3D = 0x04
CMD_VERSION = 0x05
CMD_TEMP_GAIN_MODE = 0x0A  # top-level get (not under 0x50)
CMD_FFC_MODE = 0x0B
CMD_FFC = 0x0C
CMD_FFC_TIME = 0x0D
CMD_COLOR = 0x10
CMD_FOCAL_LEN = 0x15
CMD_SHARPNESS = 0xE3  # -29
CMD_LCE = 0x2C
CMD_CONTRAST = 0x2D
CMD_UPDATE = 0x22
CMD_UPDATE_PROCESS = 0x2B
CMD_GAIN_AUTO = 0x3E
CMD_TEMPERATURE = 0x50
CMD_LIGHT = 0x55  # also CODE_COMMON
CMD_IMAGE_MODE = 0x88  # -120
CMD_INDICATOR = 0x89  # -119
CMD_INIT = 0xFA  # -6
CMD_CAPS = 0x63
CMD_MOVEMENT_MODEL = 0x66

# Temperature / radiometry subcommands under CMD_TEMPERATURE (0x50).
# Payload is [subcmd, size_hint] as built by GetTemp*OutPacket in the APK.
TEMP_QUERY: dict[str, tuple[int, int]] = {
    # name: (subcmd, size_hint)
    "base_gray": (0x8D, 4),  # TEMP_BaseGray = -115
    "compensation_data": (0x01, 3),
    "current_tec_coef": (0x0F, 3),
    "distance_b": (0x89, 4),  # TEMP_DistanceB = -119
    "gray_diff_table": (0x85, 3),  # TEMP_GrayDiffTable = -123
    "radiometry_options": (0x87, 4),  # TEMP_RadioMetryOptions = -121
    "radiometry_set_msg": (0x02, 3),
    "tec_high": (0x08, 3),
    "thermo_temp": (0x25, 4),  # TEMP_ThermoTemp = 37
}

# WeicaiData.kt palette IDs used with CMD_COLOR
PALETTE = {
    "white_hot": 0,
    "black_hot": 1,
    "iron_red": 6,
    "sepia": 7,
    "green_hot": 13,
    "alarm": 18,
}

# USBImageModeEnum.kt → setImgMode payload mapping
IMAGE_MODE = {
    "plain": 0,
    "jungle": 1,  # forest; Bird also maps to 1 in app
    "rain_fog": 2,
    "sketch": 3,
}


def be16(n: int) -> bytes:
    return bytes([(n >> 8) & 0xFF, n & 0xFF])


def _header_check(cmd: int, length: int) -> tuple[int, int]:
    t0 = TABLE[cmd ^ 0x80]
    t1 = TABLE[(((t0 ^ length) >> 8) & 0xFF) ^ 0x6A]
    mixed = (t1 ^ ((t0 << 8) & 0xFFFF)) & 0xFFFF
    t2 = TABLE[(length ^ ((mixed >> 8) & 0xFF)) & 0xFF]
    b7 = t2 & 0xFF
    mixed2 = (t2 ^ ((t1 << 8) & 0xFFFF)) & 0xFFFF
    b6 = (mixed2 >> 8) & 0xFF
    return b6, b7


def _crc_trailer(buf: bytes) -> bytes:
    crc = 0
    for b in buf:
        crc = (TABLE[((crc >> 8) ^ b) & 0xFF] ^ ((crc << 8) & 0xFFFF)) & 0xFFFF
    return bytes([(crc >> 8) & 0xFF, crc & 0xFF])


def make_cmd(cmd: int, payload: bytes = b"") -> bytes:
    """Build an outbound CDC control frame."""
    length = len(payload)
    b6, b7 = _header_check(cmd & 0xFF, length)
    hdr = bytes(
        [
            0x6E,
            0x00,
            0x00,
            cmd & 0xFF,
            (length >> 8) & 0xFF,
            length & 0xFF,
            b6,
            b7,
        ]
    )
    body = hdr + payload
    return body + _crc_trailer(body)


def ffc() -> bytes:
    return make_cmd(CMD_FFC)


def set_palette(name_or_id: Union[str, int]) -> bytes:
    pid = PALETTE[name_or_id] if isinstance(name_or_id, str) else int(name_or_id)
    return make_cmd(CMD_COLOR, be16(pid))


def set_ffc_mode(manual: bool) -> bytes:
    return make_cmd(CMD_FFC_MODE, be16(1 if manual else 0))


def set_ffc_interval_sec(seconds: int) -> bytes:
    return make_cmd(CMD_FFC_TIME, be16(int(seconds)))


def set_image_mode(name_or_id: Union[str, int]) -> bytes:
    mid = IMAGE_MODE[name_or_id] if isinstance(name_or_id, str) else int(name_or_id)
    return make_cmd(CMD_IMAGE_MODE, be16(mid))


def set_brightness(value: int) -> bytes:
    """Brightness/light; UI uses progress*5 (0..100-ish)."""
    return make_cmd(CMD_LIGHT, be16(int(value)))


def set_contrast(value: int) -> bytes:
    return make_cmd(CMD_CONTRAST, be16(int(value)))


def set_gain_auto(value: int) -> bytes:
    """AGC / auto-gain (0–255). Closest thing to a brightness/contrast 'lock'."""
    return make_cmd(CMD_GAIN_AUTO, be16(max(0, min(255, int(value)))))


def get_gain_auto() -> bytes:
    """Query auto-gain (empty payload)."""
    return make_cmd(CMD_GAIN_AUTO)


def get_temp_gain_mode() -> bytes:
    """Query temp gain mode (top-level cmd 0x0A, empty payload)."""
    return make_cmd(CMD_TEMP_GAIN_MODE)


def get_temp_query(name: str) -> bytes:
    """Build a CMD_TEMPERATURE (0x50) radiometry/calibration query."""
    if name not in TEMP_QUERY:
        raise KeyError(
            f"unknown temp query {name!r}; choose from: {', '.join(sorted(TEMP_QUERY))}"
        )
    subcmd, size_hint = TEMP_QUERY[name]
    return make_cmd(CMD_TEMPERATURE, bytes([subcmd & 0xFF, size_hint & 0xFF]))


def _print_examples() -> None:
    examples = [
        ("FFC", ffc()),
        ("palette white_hot", set_palette("white_hot")),
        ("palette black_hot", set_palette("black_hot")),
        ("palette iron_red", set_palette("iron_red")),
        ("palette sepia", set_palette("sepia")),
        ("palette green_hot", set_palette("green_hot")),
        ("palette alarm", set_palette("alarm")),
        ("ffc_mode auto", set_ffc_mode(False)),
        ("ffc_mode manual", set_ffc_mode(True)),
        ("image_mode jungle", set_image_mode("jungle")),
        ("brightness 50", set_brightness(50)),
        ("contrast 50", set_contrast(50)),
        ("gain_auto 0", set_gain_auto(0)),
        ("gain_auto 1", set_gain_auto(1)),
        ("get_gain_auto", get_gain_auto()),
        ("temp thermo_temp", get_temp_query("thermo_temp")),
        ("temp base_gray", get_temp_query("base_gray")),
        ("temp radiometry_options", get_temp_query("radiometry_options")),
        ("temp radiometry_set_msg", get_temp_query("radiometry_set_msg")),
        ("temp gray_diff_table", get_temp_query("gray_diff_table")),
        ("temp compensation_data", get_temp_query("compensation_data")),
        ("temp tec_high", get_temp_query("tec_high")),
        ("temp current_tec_coef", get_temp_query("current_tec_coef")),
        ("temp distance_b", get_temp_query("distance_b")),
        ("temp_gain_mode", get_temp_gain_mode()),
    ]
    for name, pkt in examples:
        print(f"{name:32s} {pkt.hex()}")


def _send(port: str, payload: bytes, baud: int = 115200) -> None:
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial required: uv sync / pip install pyserial") from exc
    with serial.Serial(port, baudrate=baud, timeout=1) as ser:
        ser.write(payload)
        ser.flush()
        resp = ser.read(256)
    print(f"sent {payload.hex()}")
    print(f"recv {resp.hex() if resp else '(none)'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--examples", action="store_true", help="Print example frames")
    p.add_argument("--port", help="CDC serial port, e.g. /dev/ttyACM0")
    p.add_argument(
        "--cmd",
        choices=[
            "ffc",
            "palette",
            "ffc_mode",
            "image_mode",
            "brightness",
            "contrast",
            "gain_auto",
            "get_gain_auto",
            "temp",
            "temp_gain_mode",
            "raw",
        ],
    )
    p.add_argument(
        "--value",
        help="palette/mode name, gain 0-255, or temp query name "
        f"({', '.join(sorted(TEMP_QUERY))})",
    )
    p.add_argument("--hex", dest="hex_payload", help="raw hex for --cmd raw")
    args = p.parse_args(argv)

    if args.examples or (not args.cmd and not args.port):
        _print_examples()
        return 0

    if args.cmd == "ffc":
        pkt = ffc()
    elif args.cmd == "palette":
        if args.value is None:
            p.error("--value required for palette")
        pkt = set_palette(int(args.value) if args.value.isdigit() else args.value)
    elif args.cmd == "ffc_mode":
        pkt = set_ffc_mode(str(args.value).lower() in {"1", "manual", "true"})
    elif args.cmd == "image_mode":
        if args.value is None:
            p.error("--value required for image_mode")
        pkt = set_image_mode(int(args.value) if args.value.isdigit() else args.value)
    elif args.cmd == "brightness":
        pkt = set_brightness(int(args.value or "50"))
    elif args.cmd == "contrast":
        pkt = set_contrast(int(args.value or "50"))
    elif args.cmd == "gain_auto":
        if args.value is None:
            p.error("--value required for gain_auto (0-255)")
        pkt = set_gain_auto(int(args.value))
    elif args.cmd == "get_gain_auto":
        pkt = get_gain_auto()
    elif args.cmd == "temp":
        if args.value is None:
            p.error(
                "--value required for temp; "
                f"one of: {', '.join(sorted(TEMP_QUERY))}"
            )
        try:
            pkt = get_temp_query(args.value)
        except KeyError as exc:
            p.error(str(exc))
    elif args.cmd == "temp_gain_mode":
        pkt = get_temp_gain_mode()
    elif args.cmd == "raw":
        if not args.hex_payload:
            p.error("--hex required for raw")
        pkt = bytes.fromhex(args.hex_payload.replace(" ", ""))
    else:
        p.error("specify --cmd")
        return 2

    if args.port:
        _send(args.port, pkt)
    else:
        print(pkt.hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
