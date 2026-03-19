## ⚡ Quick Start (Orange Pi Zero 2W)

Armbian OS imager: https://imager.armbian.com/


```bash
# 1. Update + install basics
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl v4l-utils ffmpeg mpv iw python3-pip vim iw htop

# 1.a Optional: Install X11 (needed for OpenCV GUI / Qt windows)
# (Skip if running headless and only using console output)
sudo apt install -y xserver-xorg xinit x11-xserver-utils libxcb-xinerama0 libx11-xcb1

# 2. Disable WiFi power saving:
/usr/sbin/iw dev wlan0 set power_save off
echo '/usr/sbin/iw dev wlan0 set power_save off' | sudo tee -a /etc/rc.local

# 2.b Configure WiFi (example with HSH-5G network):
sudo nano /etc/wpa_supplicant/wpa_supplicant.conf
# Add or modify the network section to include:
# 
# ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
# update_config=1
# country=US
#
# network={
#     ssid="HSH-5G"
#     psk="wisnowskishome"
#     key_mgmt=WPA-PSK
#     freq_list=2412 2437 2462
#     bgscan="simple:30:-70:86400"
# }
# 
# Save with Ctrl+X, Y, Enter

# 3. Install uv (Python)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
source $HOME/.local/bin/env

# 4. Configure SSH
ssh-keygen -t ed25519 -C "dwisnows@gmail.com"
cat ~/.ssh/id_ed25519.pub

# 5. Reboot
sudo reboot

# 6. Clone + install
git clone git@github.com:dwisnowski/cookie-finder.git && cd cookie-finder
make install

# 8. Run
make run
```






## Verify Camera

Plug in thermal camera:

```bash
lsusb
v4l2-ctl --list-devices
```

Expected:

```
/dev/video1
```

Test display (HDMI required):

```bash
mpv --vo=drm av://v4l2:/dev/video1
```

Press `q` to quit.
