## ⚡ Quick Start (Orange Pi Zero 2W)

Armbian OS imager: https://imager.armbian.com/


```bash
# 1. Update + install basics
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl v4l-utils ffmpeg mpv iw python3-pip vim iw htop

# 2. Disable WiFi power saving:
/usr/sbin/iw dev wlan0 set power_save off
echo '/usr/sbin/iw dev wlan0 set power_save off' | sudo tee -a /etc/rc.local

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

# 7. Run
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
