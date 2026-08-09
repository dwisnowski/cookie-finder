use super::stepper::{DriveMode, StepperMotor};
use crate::config::{
    validate_phase_order, DEFAULT_PAN_HZ, DEFAULT_TILT_HZ, GimbalConfig, MAX_PAN, MAX_TILT,
    PAN_CONTROL_PINS, PAN_LIMIT_MAX_PIN, PAN_LIMIT_MIN_PIN, TILT_CONTROL_PINS, TILT_LIMIT_MAX_PIN,
    TILT_LIMIT_MIN_PIN,
};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotorId {
    Pan,
    Tilt,
}

pub struct PanTiltGimbal {
    pub max_pan: f64,
    pub max_tilt: f64,
    inner: Mutex<Inner>,
}

struct Inner {
    pan: StepperMotor,
    tilt: StepperMotor,
    commanded_pan: f64,
    commanded_tilt: f64,
}

impl PanTiltGimbal {
    pub fn new(config: &GimbalConfig) -> Self {
        Self {
            max_pan: MAX_PAN,
            max_tilt: MAX_TILT,
            inner: Mutex::new(Inner {
                pan: StepperMotor::new(
                    "Pan",
                    PAN_CONTROL_PINS,
                    config.pan_phase_order,
                    MAX_PAN,
                    Some((PAN_LIMIT_MIN_PIN, PAN_LIMIT_MAX_PIN)),
                ),
                tilt: StepperMotor::new(
                    "Tilt",
                    TILT_CONTROL_PINS,
                    config.tilt_phase_order,
                    MAX_TILT,
                    Some((TILT_LIMIT_MIN_PIN, TILT_LIMIT_MAX_PIN)),
                ),
                commanded_pan: 0.0,
                commanded_tilt: 0.0,
            }),
        }
    }

    pub fn set_speed(&self, pan_hz: f64, tilt_hz: f64) {
        let mut g = self.inner.lock().unwrap();
        g.pan.set_speed(pan_hz);
        g.tilt.set_speed(tilt_hz);
    }

    pub fn move_to_angles(&self, pan: f64, tilt: f64) {
        let pan = pan.clamp(0.0, self.max_pan);
        let tilt = tilt.clamp(0.0, self.max_tilt);
        let mut g = self.inner.lock().unwrap();
        g.commanded_pan = pan;
        g.commanded_tilt = tilt;
        g.pan.set_target(pan);
        g.tilt.set_target(tilt);
    }

    pub fn pan_step(&self, direction: i32, steps: u32) {
        let mut g = self.inner.lock().unwrap();
        g.pan.step_fixed(direction, steps);
        g.commanded_pan = g.pan.get_angle();
    }

    pub fn tilt_step(&self, direction: i32, steps: u32) {
        let mut g = self.inner.lock().unwrap();
        g.tilt.step_fixed(direction, steps);
        g.commanded_tilt = g.tilt.get_angle();
    }

    pub fn get_position(&self) -> (f64, f64) {
        let g = self.inner.lock().unwrap();
        (g.pan.get_angle(), g.tilt.get_angle())
    }

    pub fn get_commanded(&self) -> (f64, f64) {
        let g = self.inner.lock().unwrap();
        (g.commanded_pan, g.commanded_tilt)
    }

    pub fn get_phase_orders(&self) -> ([usize; 4], [usize; 4]) {
        let g = self.inner.lock().unwrap();
        (g.pan.phase_order(), g.tilt.phase_order())
    }

    pub fn set_phase_order(&self, motor: MotorId, order: [usize; 4]) -> anyhow::Result<()> {
        let name = match motor {
            MotorId::Pan => "pan",
            MotorId::Tilt => "tilt",
        };
        validate_phase_order(&order, &format!("{name}_phase_order"))?;
        let mut g = self.inner.lock().unwrap();
        match motor {
            MotorId::Pan => g.pan.set_phase_order(order),
            MotorId::Tilt => g.tilt.set_phase_order(order),
        }
        Ok(())
    }

    /// Drive mode is shared by pan and tilt (same energization algorithm).
    pub fn get_drive_mode(&self) -> DriveMode {
        self.inner.lock().unwrap().pan.drive_mode()
    }

    pub fn set_drive_mode(&self, mode: DriveMode) {
        let mut g = self.inner.lock().unwrap();
        g.pan.set_drive_mode(mode);
        g.tilt.set_drive_mode(mode);
    }

    pub fn is_moving(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.pan.is_moving() || g.tilt.is_moving()
    }

    pub fn home(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pan.home();
        g.tilt.home();
        g.commanded_pan = 0.0;
        g.commanded_tilt = 0.0;
    }

    pub fn stop(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pan.stop();
        g.tilt.stop();
        g.commanded_pan = g.pan.get_angle();
        g.commanded_tilt = g.tilt.get_angle();
    }

    pub fn disable_motors(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pan.disable_coils();
        g.tilt.disable_coils();
        g.commanded_pan = g.pan.get_angle();
        g.commanded_tilt = g.tilt.get_angle();
    }

    pub fn tick(&self) {
        let mut g = self.inner.lock().unwrap();
        if g.pan.is_moving() {
            g.pan.tick();
        }
        if g.tilt.is_moving() {
            g.tilt.tick();
        }
    }

    pub fn pan_hz(&self) -> f64 {
        self.inner.lock().unwrap().pan.speed_hz()
    }

    pub fn tilt_hz(&self) -> f64 {
        self.inner.lock().unwrap().tilt.speed_hz()
    }

    pub fn cleanup(&self) {
        let mut g = self.inner.lock().unwrap();
        g.pan.cleanup();
        g.tilt.cleanup();
    }
}

impl Default for PanTiltGimbal {
    fn default() -> Self {
        let g = Self::new(&GimbalConfig::default());
        g.set_speed(DEFAULT_PAN_HZ, DEFAULT_TILT_HZ);
        g
    }
}
