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

## 4. Configure WiFi (NetworkManager only)

Cookie Finder (`scripts/wifi-mode.sh`, `cookie-finder-wifi`) expects **NetworkManager** to own `wlan0`. Do **not** also manage Wi‑Fi via Armbian netplan / `systemd-networkd` — that fight leaves `nmcli` showing `wlan0:unavailable`.

### One-time: remove netplan Wi‑Fi (keep Ethernet on networkd)

Armbian first-login often creates `/etc/netplan/30-wifis-dhcp.yaml`, which starts a separate `wpa_supplicant -c /run/netplan/wpa-wlan0.conf`. Remove it so NM is the only Wi‑Fi manager:

```bash
sudo cp /etc/netplan/30-wifis-dhcp.yaml /etc/netplan/30-wifis-dhcp.yaml.bak 2>/dev/null || true
sudo rm -f /etc/netplan/30-wifis-dhcp.yaml
sudo netplan generate
sudo netplan apply
sudo pkill -f '/run/netplan/wpa-wlan0.conf' 2>/dev/null || true
sudo systemctl restart NetworkManager
sleep 3
sudo nmcli device set wlan0 managed yes
nmcli -t -f DEVICE,STATE,CONNECTION device status
# Expect: wlan0:disconnected:  or  wlan0:connected:…  — not unavailable
```

Leave `10-dhcp-all-interfaces.yaml` / `armbian.yaml` alone (`renderer: networkd` for Ethernet is fine; `eth0:unmanaged` in `nmcli` is expected).

### Join / save client networks

```bash
sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"
nmcli -t -f NAME,TYPE,AUTOCONNECT connection show
```

To save the project’s preferred home + phone hotspot profiles (priorities for AP→client restore), copy `.wifi.env.example` to `.wifi.env`, set the PSKs, then:

```bash
make on-the-pi-wifi-configure-clients
```

Verify only NM’s D-Bus `wpa_supplicant` is running (no `/run/netplan/wpa-*.conf`):

```bash
ps aux | grep wpa_supplicant | grep -v grep
```

### Faster serial login (optional but recommended)

With Ethernet on networkd and no cable plugged in, `systemd-networkd-wait-online` can block `network-online.target` for **~2 minutes**, which delays `rc-local` and the serial getty. Mask it if you do not need to wait for Ethernet before login:

```bash
sudo systemctl disable --now systemd-networkd-wait-online.service
sudo systemctl mask systemd-networkd-wait-online.service
# After reboot: systemd-analyze  → userspace ~10–15s instead of ~2min
```

Ethernet still works when plugged in; only the boot-time wait is removed.

### Symptoms if this regresses

| Symptom | Likely cause |
|---------|--------------|
| `wlan0:unavailable` / empty `nmcli device wifi list` | Netplan Wi‑Fi restored (`30-wifis-dhcp.yaml`) or netplan `wpa_supplicant` holding the radio |
| Login prompt ~2 minutes after reboot | `systemd-networkd-wait-online` unmasked / waiting on unplugged `eth0` |
| Two `wpa_supplicant` processes (one with `/run/netplan/…`) | Dual stack — remove netplan Wi‑Fi again |

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
5. Your phone should open the captive portal to `http://192.168.12.1/` (web app home). If not, open that URL manually.

The home-screen badge shows **Client · …** or **AP · cookie-finder**.

> The onboard radio is typically client *or* AP, not both at once. Switching modes disconnects the current browser session.

**Captive portal:** SoftAP DNS is sinkholed to `192.168.12.1` so OS connectivity checks redirect into the web UI (served on ports **80** and **443**).

**mDNS:** run `make on-the-pi-mdns` (alias `make mdns`) once to advertise `http://cookie-finder.local/` on the LAN.

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

Then open `http://<device-ip>/` (or `http://cookie-finder.local/`) in a browser from any device on the same network.

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

> **Note:** Serial uses physical pins **8/10** (TXD.0 / RXD.0). Tilt motor IN1 is physical pin **22** (RXD.2, gpiochip1 offset 262) — a different UART, not shared with the USB-TTL console.

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
