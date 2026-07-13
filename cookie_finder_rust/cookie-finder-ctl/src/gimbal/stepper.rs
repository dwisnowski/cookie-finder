#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use crate::config::STEPS_PER_REV;
#[cfg(target_os = "linux")]
use std::sync::Mutex;

#[cfg(target_os = "linux")]
const FULL_STEP: [[u8; 4]; 4] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

pub struct StepperMotor {
    name: String,
    pins: [u32; 4],
    max_angle: f64,
    speed_hz: f64,
    current_angle: f64,
    target_angle: f64,
    current_step: i32,
    moving: bool,
    #[cfg(target_os = "linux")]
    handles: Option<Mutex<Vec<gpio_cdev::LineHandle>>>,
}

impl StepperMotor {
    pub fn new(name: &str, pins: [u32; 4], max_angle: f64) -> Self {
        let mut motor = Self {
            name: name.to_string(),
            pins,
            max_angle,
            speed_hz: 500.0,
            current_angle: 0.0,
            target_angle: 0.0,
            current_step: 0,
            moving: false,
            #[cfg(target_os = "linux")]
            handles: None,
        };
        motor.init_gpio();
        motor
    }

    #[cfg(target_os = "linux")]
    fn init_gpio(&mut self) {
        use gpio_cdev::{Chip, LineRequestFlags};
        let chip_path = crate::config::GPIO_CHIP;
        match Chip::new(chip_path) {
            Ok(mut chip) => {
                let mut handles = Vec::new();
                for &pin in &self.pins {
                    match chip.get_line(pin) {
                        Ok(line) => match line.request(LineRequestFlags::OUTPUT, 0, &self.name) {
                            Ok(h) => handles.push(h),
                            Err(e) => {
                                tracing::warn!("[{}] pin {} request failed: {}", self.name, pin, e);
                                return;
                            }
                        },
                        Err(e) => {
                            tracing::warn!("[{}] pin {} get_line failed: {}", self.name, pin, e);
                            return;
                        }
                    }
                }
                if handles.len() == 4 {
                    tracing::info!("[{}] GPIO ok pins {:?}", self.name, self.pins);
                    self.handles = Some(Mutex::new(handles));
                }
            }
            Err(e) => tracing::warn!("[{}] GPIO init failed: {}", self.name, e),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn init_gpio(&mut self) {
        tracing::warn!("[{}] GPIO unavailable (not Linux)", self.name);
    }

    #[cfg(target_os = "linux")]
    fn write_step(&self, step: i32) {
        let Some(handles) = &self.handles else { return };
        let idx = step.rem_euclid(4) as usize;
        let vals = FULL_STEP[idx];
        if let Ok(guard) = handles.lock() {
            for (h, &v) in guard.iter().zip(vals.iter()) {
                let _ = h.set_value(v);
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn write_step(&self, _step: i32) {}

    /// De-energize all coils (all pins LOW) to reduce heating when idle.
    #[cfg(target_os = "linux")]
    fn set_pins_low(&self) {
        let Some(handles) = &self.handles else { return };
        if let Ok(guard) = handles.lock() {
            for h in guard.iter() {
                let _ = h.set_value(0);
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn set_pins_low(&self) {}

    pub fn disable_coils(&mut self) {
        self.moving = false;
        self.set_pins_low();
    }

    pub fn set_speed(&mut self, hz: f64) {
        self.speed_hz = hz.clamp(10.0, 5000.0);
    }

    pub fn speed_hz(&self) -> f64 {
        self.speed_hz
    }

    pub fn set_target(&mut self, angle: f64) {
        self.target_angle = angle.clamp(0.0, self.max_angle);
        self.moving = (self.target_angle - self.current_angle).abs() >= 0.5;
    }

    pub fn get_angle(&self) -> f64 {
        self.current_angle
    }

    pub fn is_moving(&self) -> bool {
        self.moving
    }

    pub fn step_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs_f64(1.0 / self.speed_hz)
    }

    /// Step once toward target. Returns true if a step was taken.
    pub fn tick(&mut self) -> bool {
        let diff = self.target_angle - self.current_angle;
        if diff.abs() < 0.5 {
            self.moving = false;
            self.set_pins_low();
            return false;
        }
        let dir: i32 = if diff > 0.0 { 1 } else { -1 };
        self.current_step = (self.current_step + dir).rem_euclid(4);
        let deg = 360.0 / STEPS_PER_REV;
        self.current_angle = (self.current_angle + dir as f64 * deg).clamp(0.0, self.max_angle);
        self.write_step(self.current_step);
        self.moving = (self.target_angle - self.current_angle).abs() >= 0.5;
        true
    }

    pub fn step_fixed(&mut self, direction: i32, steps: u32) {
        for _ in 0..steps {
            self.current_step = (self.current_step + direction).rem_euclid(4);
            let deg = 360.0 / STEPS_PER_REV;
            self.current_angle =
                (self.current_angle + direction as f64 * deg).clamp(0.0, self.max_angle);
            self.write_step(self.current_step);
            std::thread::sleep(self.step_interval());
        }
        self.target_angle = self.current_angle;
        self.moving = false;
        self.set_pins_low();
    }

    pub fn home(&mut self) {
        tracing::warn!("[{}] home skipped (no limit switch)", self.name);
        self.current_angle = 0.0;
        self.target_angle = 0.0;
        self.current_step = 0;
        self.moving = false;
        self.set_pins_low();
    }

    pub fn stop(&mut self) {
        self.target_angle = self.current_angle;
        self.moving = false;
        self.write_step(self.current_step);
    }

    pub fn cleanup(&mut self) {
        #[cfg(target_os = "linux")]
        if let Some(handles) = self.handles.take() {
            if let Ok(guard) = handles.lock() {
                for h in guard.iter() {
                    let _ = h.set_value(0);
                }
            }
        }
    }
}
