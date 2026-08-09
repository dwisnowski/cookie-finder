#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use crate::config::STEPS_PER_REV;
#[cfg(target_os = "linux")]
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Wave drive (1-coil): lowest torque; many 24BYJ motors will not self-start.
const WAVE_DRIVE: [[u8; 4]; 4] = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
];

/// Full-step (2-coil): more starting torque than wave drive.
const FULL_STEP: [[u8; 4]; 4] = [
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [1, 0, 0, 1],
];

/// Half-step (alternating 1/2 coil): smoothest; good torque.
const HALF_STEP: [[u8; 4]; 8] = [
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
];

const HOME_SPEED_HZ: f64 = 200.0;
const HOME_TIMEOUT_SECS: u64 = 10;

/// Coil energization algorithm for ULN2003 unipolar steppers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DriveMode {
    /// Single-coil excitation (1000 → 0100 → 0010 → 0001).
    /// Historically mislabeled "full-step" in this codebase.
    #[default]
    Wave,
    /// Dual-coil excitation (1100 → 0110 → 0011 → 1001).
    FullStep,
    /// Alternating single/dual coil (8 patterns per cycle).
    HalfStep,
}

impl DriveMode {
    pub const ALL: [DriveMode; 3] = [DriveMode::Wave, DriveMode::FullStep, DriveMode::HalfStep];

    pub fn as_str(self) -> &'static str {
        match self {
            DriveMode::Wave => "wave",
            DriveMode::FullStep => "full",
            DriveMode::HalfStep => "half",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DriveMode::Wave => "wave (1-coil)",
            DriveMode::FullStep => "full-step (2-coil)",
            DriveMode::HalfStep => "half-step",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "wave" | "wave_drive" | "wave-drive" => Some(DriveMode::Wave),
            "full" | "full_step" | "full-step" => Some(DriveMode::FullStep),
            "half" | "half_step" | "half-step" => Some(DriveMode::HalfStep),
            _ => None,
        }
    }

    fn sequence(self) -> &'static [[u8; 4]] {
        match self {
            DriveMode::Wave => &WAVE_DRIVE,
            DriveMode::FullStep => &FULL_STEP,
            DriveMode::HalfStep => &HALF_STEP,
        }
    }

    pub fn sequence_len(self) -> i32 {
        self.sequence().len() as i32
    }
}

pub struct StepperMotor {
    name: String,
    pins: [u32; 4],
    /// Center-Off SPDT channels: (min/CCW/home, max/CW). None if not configured.
    limit_pins: Option<(u32, u32)>,
    phase_order: [usize; 4],
    drive_mode: DriveMode,
    max_angle: f64,
    speed_hz: f64,
    current_angle: f64,
    target_angle: f64,
    current_step: i32,
    moving: bool,
    #[cfg(target_os = "linux")]
    handles: Option<Mutex<Vec<gpio_cdev::LineHandle>>>,
    #[cfg(target_os = "linux")]
    limit_handles: Option<Mutex<(gpio_cdev::LineHandle, gpio_cdev::LineHandle)>>,
}

impl StepperMotor {
    pub fn new(
        name: &str,
        pins: [u32; 4],
        phase_order: [usize; 4],
        max_angle: f64,
        limit_pins: Option<(u32, u32)>,
    ) -> Self {
        let mut motor = Self {
            name: name.to_string(),
            pins,
            limit_pins,
            phase_order,
            drive_mode: DriveMode::default(),
            max_angle,
            speed_hz: 500.0,
            current_angle: 0.0,
            target_angle: 0.0,
            current_step: 0,
            moving: false,
            #[cfg(target_os = "linux")]
            handles: None,
            #[cfg(target_os = "linux")]
            limit_handles: None,
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

                // gpio-cdev v1 has no bias flags; lines are INPUT (active-low when
                // COM→GND). Prefer a 10k pull-up to 3.3V on each channel if floating.
                if let Some((min_pin, max_pin)) = self.limit_pins {
                    let min_h = match chip.get_line(min_pin).and_then(|l| {
                        l.request(
                            LineRequestFlags::INPUT,
                            0,
                            &format!("{}_limit_min", self.name),
                        )
                    }) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::warn!(
                                "[{}] limit min pin {} failed: {}",
                                self.name,
                                min_pin,
                                e
                            );
                            return;
                        }
                    };
                    let max_h = match chip.get_line(max_pin).and_then(|l| {
                        l.request(
                            LineRequestFlags::INPUT,
                            0,
                            &format!("{}_limit_max", self.name),
                        )
                    }) {
                        Ok(h) => h,
                        Err(e) => {
                            tracing::warn!(
                                "[{}] limit max pin {} failed: {}",
                                self.name,
                                max_pin,
                                e
                            );
                            return;
                        }
                    };
                    tracing::info!(
                        "[{}] limit switches ok min={} max={}",
                        self.name,
                        min_pin,
                        max_pin
                    );
                    self.limit_handles = Some(Mutex::new((min_h, max_h)));
                }
            }
            Err(e) => tracing::warn!("[{}] GPIO init failed: {}", self.name, e),
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn init_gpio(&mut self) {
        tracing::warn!("[{}] GPIO unavailable (not Linux)", self.name);
    }

    /// Active-low: tripped channel reads 0. Returns (min_hit, max_hit).
    pub fn limit_state(&self) -> (bool, bool) {
        #[cfg(target_os = "linux")]
        {
            let Some(handles) = &self.limit_handles else {
                return (false, false);
            };
            let Ok(guard) = handles.lock() else {
                return (false, false);
            };
            let min_hit = guard.0.get_value().unwrap_or(1) == 0;
            let max_hit = guard.1.get_value().unwrap_or(1) == 0;
            return (min_hit, max_hit);
        }
        #[cfg(not(target_os = "linux"))]
        {
            (false, false)
        }
    }

    fn direction_blocked(&self, direction: i32) -> bool {
        let (min_hit, max_hit) = self.limit_state();
        (direction < 0 && min_hit) || (direction > 0 && max_hit)
    }

    #[cfg(target_os = "linux")]
    fn write_step(&self, step: i32) {
        let Some(handles) = &self.handles else { return };
        let seq = self.drive_mode.sequence();
        let idx = step.rem_euclid(seq.len() as i32) as usize;
        let vals = seq[idx];
        if let Ok(guard) = handles.lock() {
            for (i, h) in guard.iter().enumerate() {
                let _ = h.set_value(vals[self.phase_order[i]]);
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

    pub fn phase_order(&self) -> [usize; 4] {
        self.phase_order
    }

    pub fn set_phase_order(&mut self, order: [usize; 4]) {
        self.phase_order = order;
    }

    pub fn drive_mode(&self) -> DriveMode {
        self.drive_mode
    }

    pub fn set_drive_mode(&mut self, mode: DriveMode) {
        if self.drive_mode != mode {
            tracing::info!(
                "[{}] drive mode {} → {}",
                self.name,
                self.drive_mode.as_str(),
                mode.as_str()
            );
            self.drive_mode = mode;
            self.current_step = 0;
        }
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

    fn advance_step(&mut self, direction: i32) -> bool {
        if self.direction_blocked(direction) {
            tracing::debug!(
                "[{}] step blocked by limit (dir={})",
                self.name,
                direction
            );
            return false;
        }
        let len = self.drive_mode.sequence_len();
        self.current_step = (self.current_step + direction).rem_euclid(len);
        // Half-step advances half the mechanical angle of wave/full per index.
        let steps_per_rev = match self.drive_mode {
            DriveMode::HalfStep => STEPS_PER_REV * 2.0,
            DriveMode::Wave | DriveMode::FullStep => STEPS_PER_REV,
        };
        let deg = 360.0 / steps_per_rev;
        self.current_angle =
            (self.current_angle + direction as f64 * deg).clamp(0.0, self.max_angle);
        self.write_step(self.current_step);
        true
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
        if !self.advance_step(dir) {
            self.moving = false;
            self.target_angle = self.current_angle;
            self.write_step(self.current_step);
            return false;
        }
        self.moving = (self.target_angle - self.current_angle).abs() >= 0.5;
        true
    }

    pub fn step_fixed(&mut self, direction: i32, steps: u32) {
        for _ in 0..steps {
            if !self.advance_step(direction) {
                break;
            }
            std::thread::sleep(self.step_interval());
        }
        self.target_angle = self.current_angle;
        self.moving = false;
        self.set_pins_low();
    }

    pub fn home(&mut self) {
        #[cfg(target_os = "linux")]
        let limits_ready = self.limit_handles.is_some();
        #[cfg(not(target_os = "linux"))]
        let limits_ready = false;

        if self.limit_pins.is_none() || !limits_ready {
            tracing::warn!("[{}] home skipped (no limit switch)", self.name);
            self.current_angle = 0.0;
            self.target_angle = 0.0;
            self.current_step = 0;
            self.moving = false;
            self.set_pins_low();
            return;
        }

        tracing::info!("[{}] homing toward min limit…", self.name);
        self.moving = false;
        let saved_hz = self.speed_hz;
        self.speed_hz = HOME_SPEED_HZ;
        let deadline = Instant::now() + Duration::from_secs(HOME_TIMEOUT_SECS);
        let home_interval = Duration::from_secs_f64(1.0 / HOME_SPEED_HZ);

        while Instant::now() < deadline {
            let (min_hit, _) = self.limit_state();
            if min_hit {
                self.current_angle = 0.0;
                self.target_angle = 0.0;
                self.current_step = 0;
                self.moving = false;
                self.speed_hz = saved_hz;
                self.write_step(self.current_step);
                tracing::info!("[{}] homed at 0°", self.name);
                return;
            }
            if !self.advance_step(-1) {
                // Blocked by min limit — treat as home.
                let (min_hit, _) = self.limit_state();
                if min_hit {
                    self.current_angle = 0.0;
                    self.target_angle = 0.0;
                    self.current_step = 0;
                    self.moving = false;
                    self.speed_hz = saved_hz;
                    self.write_step(self.current_step);
                    tracing::info!("[{}] homed at 0°", self.name);
                    return;
                }
                break;
            }
            std::thread::sleep(home_interval);
        }

        tracing::warn!(
            "[{}] homing timed out or blocked; zeroing angle anyway",
            self.name
        );
        self.current_angle = 0.0;
        self.target_angle = 0.0;
        self.current_step = 0;
        self.moving = false;
        self.speed_hz = saved_hz;
        self.set_pins_low();
    }

    pub fn stop(&mut self) {
        self.target_angle = self.current_angle;
        self.moving = false;
        self.write_step(self.current_step);
    }

    pub fn cleanup(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if let Some(handles) = self.handles.take() {
                if let Ok(guard) = handles.lock() {
                    for h in guard.iter() {
                        let _ = h.set_value(0);
                    }
                }
            }
            self.limit_handles.take();
        }
    }
}
