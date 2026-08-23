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
CMD_LIGHT = 0x55  # also CODE_COMMON
CMD_IMAGE_MODE = 0x88  # -120
CMD_INDICATOR = 0x89  # -119
CMD_INIT = 0xFA  # -6
CMD_CAPS = 0x63
CMD_MOVEMENT_MODEL = 0x66
CMD_TEMPERATURE = 0x50

# WeicaiData.kt palette IDs used with CMD_COLOR
PALETTE = {
    "white_hot": 0,  # BAI_RE
    "black_hot": 1,  # HEI_RE
    "iron_red": 6,  # TIE_HONG
    "sepia": 7,  # HU_PO
    "green_hot": 13,  # FEI_CUI
    "alarm": 18,  # BAO_JING
}

# USBImageModeEnum.kt → setImgMode payload mapping (a.d())
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
    # Native setImgMode packs [0x88 LE as u16?][mode]; Java getSendData uses cmd=0x88.
    # App path: SendDataPacket.setImgMode(int) → native with cmd embedded.
    # Prefer Java-style cmd=0x88 + BE mode byte padded as used by setImgMode native:
    # native stores halfword 0x0088 + mode byte (3-byte payload) with outer cmd=0.
    # Empirically prefer make_cmd(CMD_IMAGE_MODE, be16(mid)) first when probing.
    return make_cmd(CMD_IMAGE_MODE, be16(mid))


def set_brightness(value: int) -> bytes:
    """Brightness/light; UI uses progress*5 (0..100-ish)."""
    return make_cmd(CMD_LIGHT, be16(int(value)))


def set_contrast(value: int) -> bytes:
    return make_cmd(CMD_CONTRAST, be16(int(value)))


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
    ]
    for name, pkt in examples:
        print(f"{name:24s} {pkt.hex()}")


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
            "raw",
        ],
    )
    p.add_argument("--value", help="palette name/id, mode name, or integer")
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
