use crate::config::BT_DEADZONE;
use crate::input::GamepadSelector;
use evdev::{AbsoluteAxisType, Device, InputEventKind};
use std::fs::read_dir;
use std::path::PathBuf;

pub struct GamepadInput {
    path: PathBuf,
    device: Device,
    pan_axis: f64,
    tilt_axis: f64,
}

impl GamepadInput {
    pub fn open(selector: &GamepadSelector) -> anyhow::Result<Self> {
        let path = find_gamepad_path(selector)?;
        tracing::info!(
            "gamepad: {} (selector address={:?} name={:?})",
            path.display(),
            selector.address,
            selector.name
        );
        let device = Device::open(&path)?;
        Ok(Self {
            path,
            device,
            pan_axis: 0.0,
            tilt_axis: 0.0,
        })
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn poll(&mut self) -> anyhow::Result<()> {
        let events: Vec<_> = self.device.fetch_events()?.collect();
        for ev in events {
            if let InputEventKind::AbsAxis(axis) = ev.kind() {
                let val = ev.value() as f64;
                match axis {
                    AbsoluteAxisType::ABS_X | AbsoluteAxisType::ABS_RX => {
                        self.pan_axis = norm_axis(val, &self.device, axis);
                    }
                    AbsoluteAxisType::ABS_Y | AbsoluteAxisType::ABS_RY => {
                        self.tilt_axis = norm_axis(val, &self.device, axis);
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    pub fn axes(&self) -> (f64, f64) {
        let pan = if self.pan_axis.abs() > BT_DEADZONE {
            self.pan_axis
        } else {
            0.0
        };
        let tilt = if self.tilt_axis.abs() > BT_DEADZONE {
            self.tilt_axis
        } else {
            0.0
        };
        (pan, tilt)
    }
}

fn normalize_mac(s: &str) -> String {
    s.trim().to_uppercase()
}

fn looks_like_gamepad(name: &str) -> bool {
    // Normalize "Game-pad" / "Game Pad" → "gamepad" so hyphenated HID names match.
    let normalized: String = name
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    normalized.contains("gamepad")
        || normalized.contains("controller")
        || normalized.contains("xbox")
        || normalized.contains("playstation")
        || normalized.contains("joystick")
}

fn names_match(a: &str, b: &str) -> bool {
    let norm = |s: &str| -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect()
    };
    let na = norm(a);
    let nb = norm(b);
    !na.is_empty() && !nb.is_empty() && (na == nb || na.contains(&nb) || nb.contains(&na))
}

fn device_matches(dev: &Device, selector: &GamepadSelector) -> bool {
    let name = dev.name().unwrap_or_default();
    let uniq = normalize_mac(dev.unique_name().unwrap_or_default());
    let phys = normalize_mac(dev.physical_path().unwrap_or_default());

    if let Some(addr) = &selector.address {
        // Primary: BlueZ HID usually puts the MAC in uniq and/or phys.
        if (!uniq.is_empty() && uniq == *addr)
            || (!phys.is_empty() && phys.contains(addr.as_str()))
        {
            return true;
        }
        // Fallback when uniq is empty: match reported BlueZ name.
        if let Some(want) = &selector.name {
            return names_match(&name, want);
        }
        return false;
    }

    if let Some(want) = &selector.name {
        if names_match(&name, want) {
            return true;
        }
    }

    looks_like_gamepad(&name)
}

fn find_gamepad_path(selector: &GamepadSelector) -> anyhow::Result<PathBuf> {
    let mut candidates: Vec<(PathBuf, String)> = Vec::new();

    for entry in read_dir("/dev/input")? {
        let path = entry?.path();
        if !path.to_string_lossy().starts_with("/dev/input/event") {
            continue;
        }
        if let Ok(dev) = Device::open(&path) {
            let name = dev.name().unwrap_or_default().to_string();
            if device_matches(&dev, selector) {
                return Ok(path);
            }
            candidates.push((path, name));
        }
    }

    if selector.address.is_some() {
        anyhow::bail!(
            "no input device for Bluetooth address {:?} (name={:?}); seen: {:?}",
            selector.address,
            selector.name,
            candidates
        );
    }
    anyhow::bail!("no gamepad found in /dev/input/event*")
}

fn norm_axis(val: f64, dev: &Device, axis: AbsoluteAxisType) -> f64 {
    if let Ok(state) = dev.get_abs_state() {
        let abs = &state[axis.0 as usize];
        let mid = (abs.minimum + abs.maximum) as f64 / 2.0;
        let half = (abs.maximum - abs.minimum) as f64 / 2.0;
        if half > 0.0 {
            return ((val - mid) / half).clamp(-1.0, 1.0);
        }
    }
    (val / 32767.0).clamp(-1.0, 1.0)
}
