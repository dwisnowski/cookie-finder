pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const DEFAULT_SOCKET: &str = "/tmp/cookie-finder.sock";
#[cfg(target_os = "linux")]
pub const GPIO_CHIP: &str = "/dev/gpiochip1";

pub const PAN_CONTROL_PINS: [u32; 4] = [271, 268, 258, 272];
pub const TILT_CONTROL_PINS: [u32; 4] = [262, 229, 233, 265];

pub const MAX_PAN: f64 = 150.0;
pub const MAX_TILT: f64 = 60.0;
pub const DEFAULT_PAN_HZ: f64 = 500.0;
pub const DEFAULT_TILT_HZ: f64 = 500.0;
pub const STEPS_PER_REV: f64 = 4076.0;

pub const BT_DEADZONE: f64 = 0.15;
pub const BT_SENSITIVITY: f64 = 100.0;
pub const BT_CHANGE_THRESHOLD: f64 = 0.05;
pub const BT_MOTION_MIN_ANGLE: f64 = 0.01;
pub const LOOP_INTERVAL_MS: u64 = 5;
