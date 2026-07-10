use crate::config::BT_DEADZONE;

pub struct GamepadInput {
    pan_axis: f64,
    tilt_axis: f64,
}

impl GamepadInput {
    pub fn open() -> anyhow::Result<Self> {
        Ok(Self { pan_axis: 0.0, tilt_axis: 0.0 })
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
