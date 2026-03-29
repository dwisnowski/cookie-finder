# Wiring Guide

GPIO wiring for the **Orange Pi Zero 2W** and attached peripherals.

> This guide is specific to the **Orange Pi Zero 2W**. Pin offsets and GPIO chip assignments will differ on other boards.

## Board Reference

- [Orange Pi Zero 2W — Official Hardware Page](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html)

![Orange Pi Zero 2W Pinout](orangepi-zero2w-pinout.png)

---

> All GPIO access uses `/dev/gpiochip1` via `libgpiod`. Pin offsets below are the confirmed working values from hardware testing on the Orange Pi Zero 2W.

---

## Pan Motor (28BYJ-48 + ULN2003)

The pan motor is a 4-wire unipolar stepper driven by a ULN2003 driver board.

### Connections

| Signal | gpiochip1 Offset | Orange Pi GPIO Label | ULN2003 Pin | Wire Color (typical) |
|--------|-----------------|----------------------|-------------|----------------------|
| PAN IN1 | **258** | PI15 | IN1 | Orange |
| PAN IN2 | **268** | PI12 | IN2 | Yellow |
| PAN IN3 | **271** | PI02 | IN3 | Pink |
| PAN IN4 | **272** | PI16 | IN4 | Blue |

### Power

| Pin | Connect to |
|-----|------------|
| ULN2003 VCC | 5V (Orange Pi 5V header pin) |
| ULN2003 GND | GND (Orange Pi GND header pin) |

### Motor Specs

| Property | Value |
|----------|-------|
| Motor model | 28BYJ-48 |
| Driver | ULN2003 |
| Step sequence | Full-step: `(1,0,0,0)→(0,1,0,0)→(0,0,1,0)→(0,0,0,1)` |
| Steps/revolution | 4076 |
| Degrees per step | ~0.0883° |
| Default step speed | 500 Hz |
| Homing speed | 200 Hz |

### Quick Verification

Test the pan motor without running the full application:

```bash
make test-pan-step
```

This runs 20 full-step cycles directly via `gpioset` on the confirmed pin offsets.

---

## Pan Limit Switch

> **TBD** — Pin not yet confirmed from hardware testing.

| Signal | gpiochip1 Offset | Notes |
|--------|-----------------|-------|
| Pan Limit | _TBD_ | Active low (triggered = 0) |

---

## Tilt Motor (28BYJ-48 + ULN2003)

> **TBD** — Tilt motor wiring not yet mapped from hardware testing.

| Signal | gpiochip1 Offset | Orange Pi GPIO Label | ULN2003 Pin |
|--------|-----------------|----------------------|-------------|
| TILT IN1 | _TBD_ | _TBD_ | IN1 |
| TILT IN2 | _TBD_ | _TBD_ | IN2 |
| TILT IN3 | _TBD_ | _TBD_ | IN3 |
| TILT IN4 | _TBD_ | _TBD_ | IN4 |

---

## Tilt Limit Switch

> **TBD** — Pin not yet confirmed from hardware testing.

| Signal | gpiochip1 Offset | Notes |
|--------|-----------------|-------|
| Tilt Limit | _TBD_ | Active low (triggered = 0) |

---

## Thermal Camera (USB)

The thermal camera connects via **USB 2.0** — no GPIO required.

| Connection | Detail |
|------------|--------|
| Interface | USB 2.0 Type-A |
| Driver | UVC (driverless) |
| Typical device path | `/dev/video1` |

---

## Notes

- All GPIO access requires `sudo` or appropriate group permissions on Armbian.
- The logical pin numbers in `cookie_finder/gimbal/pan_tilt.py` (`PAN_CONTROL_PINS = (23, 24, 25, 26)`) are **not the hardware offsets**. The actual offsets used by the stepper driver are the values listed in this guide.
- When tilt and limit switch pins are confirmed, update both this file and the constants in `cookie_finder/gimbal/stepper.py`.
