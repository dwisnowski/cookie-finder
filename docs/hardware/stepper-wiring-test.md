# Stepper Motor Wiring Test

Discover the correct **phase order** for 28BYJ-48 stepper motors without rewiring. The software remaps which logical step phase drives each ULN2003 output (IN1–IN4).

## When to use this

Use this procedure when:

- A motor buzzes, jitters, or barely moves
- You connected the four motor wires to the ULN2003 in an unknown order
- You replaced a motor or driver board

GPIO pins to the ULN2003 must already match [Wiring Guide](wiring.md). This test only finds the **coil phase mapping**, not which GPIO pin is which.

## How it works

The driver uses a 4-step full-step sequence. There are **24** possible mappings from logical phases (0–3) to physical IN1–IN4 outputs. The keyboard tool cycles through them; you pick the one that spins smoothly.

You do **not** need to unplug and reorder wires for each try—the daemon permutes the mapping in software.

## Prerequisites

1. Pan and/or tilt motors wired to ULN2003 boards (any wire order on IN1–IN4 is fine).
2. ULN2003 IN1–IN4 connected to the Orange Pi per [Wiring Guide](wiring.md).
3. Rust gimbal daemon running with GPIO access (`sudo`).

## Procedure

### 1. Start the daemon

```bash
make on-the-pi-rust-daemon
```

Or via systemd if configured. The daemon loads phase order from `config/gimbal.toml` on startup.

### 2. Open keyboard control

```bash
make on-the-pi-rust-keyboard
```

The screen shows the current wiring mapping for **both** pan and tilt motors.

### 3. Test one motor at a time

| Key | Action |
|-----|--------|
| `P` | Select **pan** motor for permutation cycling |
| `T` | Select **tilt** motor for permutation cycling |
| `[` / `]` | Previous / next permutation (1–24) |
| Arrow keys | Spin motors (hold to test; `Space` or `s` to stop) |
| `1`–`9` | Step speed (start with `1` or `2` for wiring tests) |

1. Press `P` (or `T`) to select the motor under test.
2. Hold an arrow key to spin the motor.
3. Press `]` to try the next permutation if motion is rough or wrong.
4. Repeat until the motor spins smoothly with consistent torque.

**Tips:**

- Use a low speed preset (`1` or `2`) while testing.
- If the motor spins in the wrong direction, try the opposite arrow—or continue cycling; reverse mappings can overlap.
- Only 12 unique behaviors exist if you ignore direction, but cycling all 24 is simplest.

### 4. Save the mapping

| Key | Action |
|-----|--------|
| `Y` | Preview the TOML line for the selected motor (stderr) |
| `W` | **Write** the selected motor's mapping to `config/gimbal.toml` |

After a good permutation:

1. Press `W` to save. The UI confirms the path written.
2. Test the other motor (`P` or `T`), find its mapping, press `W` again.
3. Restart the daemon to verify the file loads on startup:

   ```bash
   sudo systemctl restart cookie-finder
   ```

   Or stop and re-run `make on-the-pi-rust-daemon`.

## Config file

Default path: `config/gimbal.toml` (or set `COOKIE_FINDER_CONFIG`).

```toml
[gimbal]
pan_phase_order  = [0, 1, 2, 3]
tilt_phase_order = [0, 1, 2, 3]
```

Each array has four values 0–3, each used exactly once. Index 0 is IN1, index 1 is IN2, and so on. Value `2` at index 0 means IN1 is driven by logical phase 2.

Example after discovery:

```toml
[gimbal]
pan_phase_order  = [2, 0, 1, 3]
tilt_phase_order = [0, 1, 2, 3]
```

The daemon also accepts `--config /path/to/gimbal.toml`. Pass the same path to the keyboard tool with `--config` if not using the default.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| No motion on any permutation | GPIO wiring, 5V/GND to ULN2003, daemon running as root |
| Rough on all 24 | Loose connector, damaged motor, insufficient supply voltage |
| `W` fails with permission error | Run from a user that can write `config/gimbal.toml`, or use `sudo` |
| Mapping lost after reboot | Confirm `W` wrote the file and daemon loads it (`--config` path) |

## Related

- [Wiring Guide](wiring.md) — GPIO pin assignments
- [Makefile reference](../reference/makefile.md) — `on-the-pi-rust-keyboard` target
