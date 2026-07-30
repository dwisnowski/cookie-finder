# Orange Pi Zero 2W — Setup Guide

This project is configured for the **[Orange Pi Zero 2W](http://www.orangepi.org/html/hardWare/computerAndMicrocontrollers/details/Orange-Pi-Zero-2W.html)**. Tested on Armbian Bookworm (latest stable).

---

## 1. Flash Armbian

Use the Armbian OS Imager: [https://imager.armbian.com/](https://imager.armbian.com/)

Flash the latest Bookworm image for the Orange Pi Zero 2W to a microSD card.

---

## 2. First Boot — Update and Install Tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl v4l-utils ffmpeg mpv iw python3-pip vim htop
```

**Optional — X11 (only needed for OpenCV GUI / Qt windows):**

```bash
sudo apt install -y xserver-xorg xinit x11-xserver-utils libxcb-xinerama0 libx11-xcb1
```

> Skip X11 if running headless. The web server mode (`make run`) does not require a display.

---

## 3. Disable WiFi Power Saving

```bash
/usr/sbin/iw dev wlan0 set power_save off
echo '/usr/sbin/iw dev wlan0 set power_save off' | sudo tee -a /etc/rc.local
```

---

## 4. Configure WiFi

```bash
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
```

Add or modify the `network` block:

```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=US

network={
    ssid="YOUR_SSID"
    psk="YOUR_PASSWORD"
    key_mgmt=WPA-PSK
    freq_list=2412 2437 2462
    bgscan="simple:30:-70:86400"
}
```

Save with `Ctrl+X`, `Y`, `Enter`.

---

## 5. Enable WiFi Access Point Mode (optional)

When home/office WiFi is unavailable, the Orange Pi can host its own network named **`cookie-finder`**.

After cloning the repo and running `make install`:

```bash
make init-wifi
```

This installs `hostapd` / `dnsmasq` / `create_ap` (when missing), grants passwordless sudo for `scripts/wifi-mode.sh`, and enables the **`cookie-finder-wifi`** systemd service (GPIO button + LED on boot). After a full reboot that service restores **client** WiFi; AP mode is runtime-only and does not persist across power cycle.

**Physical button:** press to toggle client ↔ AP (works without the web app). See [Hardware — Wiring](../hardware/wiring.md#wifi-mode-button--led).

**Web UI:**

1. Open **Settings**
2. Under **WiFi Mode**, choose **Switch to AP Mode**
3. Confirm the popup instructions
4. Join WiFi **`cookie-finder`** (open network — no password)
5. Open `http://192.168.12.1:8000`

The home-screen badge shows **Client · …** or **AP · cookie-finder**.

> The onboard radio is typically client *or* AP, not both at once. Switching modes disconnects the current browser session.

---

## 6. Install uv (Python Environment Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
source $HOME/.local/bin/env
```

---

## 7. Configure SSH Key

```bash
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub
```

Add the public key to your GitHub account if you plan to clone via SSH.

---

## 8. Reboot

```bash
sudo reboot
```

---

## 9. Clone and Install

```bash
git clone git@github.com:dwisnowski/cookie-finder.git
cd cookie-finder
make install
# Optional WiFi AP support:
make init-wifi
```

---

## 10. Run

```bash
make run
```

Then open `http://<device-ip>:8000` in a browser from any device on the same network.

---

## Serial Console (USB-TTL UART)

If WiFi is down, use a **3.3V USB-TTL cable** on UART0 (same pins as Raspberry Pi):

| Pi pin | Signal | USB-TTL |
|--------|--------|---------|
| 8 | TXD.0 | RX |
| 10 | RXD.0 | TX |
| 9 | GND | GND |

Baud rate: **115200**. On your Mac, copy `.serial.env.example` to `.serial.env` and set `SERIAL_DEVICE` and `SERIAL_PASSWORD`.

```bash
make serial-connect              # interactive screen session
make serial-deploy               # sync project without WiFi
make serial-run SERIAL_CMD='…'   # run a single command
```

See [Makefile reference](../reference/makefile.md#serial-console) for all serial targets.

> **Note:** Pin 8 (PH0) is shared with the tilt motor IN1 wire in this project. Disconnect that wire for clean serial debugging if needed.

---

## Verify Camera

After plugging in the thermal camera:

```bash
lsusb
v4l2-ctl --list-devices
```

Expected: `/dev/video1` (or similar).

Test the raw stream with an HDMI display connected:

```bash
mpv --vo=drm av://v4l2:/dev/video1
```
