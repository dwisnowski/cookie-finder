## ⚡ Quick Start (Orange Pi Zero 2W)

```bash
# 1. Update + install basics
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl v4l-utils ffmpeg mpv iw python3-pip

# 2. Fix WiFi lag (IMPORTANT)
/usr/sbin/iw dev wlan0 set power_save off

# 3. Install uv (Python)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 4. Clone + install
git clone https://github.com/dwisnowski/cookie-finder.git
cd cookie-finder
make install

# 5. Run
make run
```

----
----
----
----
----



Here’s a **clean, minimal Markdown you can drop into your repo** (no fluff, just what works).

---

```markdown
# Orange Pi Zero 2W Setup (Armbian)

This guide gets you from fresh image → working system → ready to run cookie-finder.

---

## 1. Flash OS

Use:
- **Armbian Bookworm (minimal/server)**

Flash to microSD and boot.

---

## 2. First Boot (login)

Default:
```

user: root
password: 1234

````

You will be prompted to:
- change password
- create a user

---

## 3. System Setup

Update and install essentials:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  build-essential \
  git \
  curl \
  htop \
  v4l-utils \
  ffmpeg \
  mpv \
  iw \
  python3-pip
````

---

## 4. Fix WiFi Stability (IMPORTANT)

Disable WiFi power saving:

```bash
sudo /usr/sbin/iw dev wlan0 set power_save off
```

Make it persistent:

```bash
echo '/usr/sbin/iw dev wlan0 set power_save off' | sudo tee -a /etc/rc.local
```

---

## 5. Verify Network

```bash
ping -c 10 8.8.8.8
```

Expected:

* 0% packet loss
* stable latency (~20–30 ms)

---

## 6. Verify Camera

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

---

## 7. Install uv (Python)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

---

## 8. Clone cookie-finder

```bash
cd ~
git clone https://github.com/dwisnowski/cookie-finder.git
cd cookie-finder
```

---

## 9. Install dependencies

```bash
make install
```

If download timeouts occur:

```bash
UV_HTTP_TIMEOUT=120 uv sync
```

---

## 10. Run app

```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://<PI_IP>:8000
```

---

## Notes

* Use `/dev/video1` (not `/dev/video0`)
* `/dev/video2` can be ignored
* `/dev/ttyACM0` is a control interface (advanced use)

