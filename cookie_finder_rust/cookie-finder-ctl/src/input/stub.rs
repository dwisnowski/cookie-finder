use crate::config::BT_DEADZONE;
use crate::input::GamepadSelector;
use std::path::{Path, PathBuf};

pub struct GamepadInput {
    path: PathBuf,
    pan_axis: f64,
    tilt_axis: f64,
}

impl GamepadInput {
    pub fn open(_selector: &GamepadSelector) -> anyhow::Result<Self> {
        Ok(Self {
            path: PathBuf::from("/dev/null"),
            pan_axis: 0.0,
            tilt_axis: 0.0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn poll(&mut self) -> anyhow::Result<()> {
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
