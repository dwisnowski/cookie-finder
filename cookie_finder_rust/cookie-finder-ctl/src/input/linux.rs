use crate::config::BT_DEADZONE;
use evdev::{AbsoluteAxisType, Device, InputEventKind};
use std::fs::read_dir;

pub struct GamepadInput {
    device: Device,
    pan_axis: f64,
    tilt_axis: f64,
}

impl GamepadInput {
    pub fn open() -> anyhow::Result<Self> {
        let path = find_gamepad_path()?;
        tracing::info!("gamepad: {}", path.display());
        let device = Device::open(path)?;
        Ok(Self {
            device,
            pan_axis: 0.0,
            tilt_axis: 0.0,
        })
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

fn find_gamepad_path() -> anyhow::Result<std::path::PathBuf> {
    for entry in read_dir("/dev/input")? {
        let path = entry?.path();
        if !path.to_string_lossy().starts_with("/dev/input/event") {
            continue;
        }
        if let Ok(dev) = Device::open(&path) {
            let name = dev.name().unwrap_or_default().to_lowercase();
            if name.contains("gamepad")
                || name.contains("controller")
                || name.contains("xbox")
                || name.contains("playstation")
                || name.contains("joystick")
            {
                return Ok(path);
            }
        }
    }
    anyhow::bail!("no gamepad found in /dev/input/event*")
}
