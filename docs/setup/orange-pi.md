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

## 5. Install uv (Python Environment Manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
source $HOME/.local/bin/env
```

---

## 6. Configure SSH Key

```bash
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub
```

Add the public key to your GitHub account if you plan to clone via SSH.

---

## 7. Reboot

```bash
sudo reboot
```

---

## 8. Clone and Install

```bash
git clone git@github.com:dwisnowski/cookie-finder.git
cd cookie-finder
make install
```

---

## 9. Run

```bash
make run
```

Then open `http://<device-ip>:8000` in a browser from any device on the same network.

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
