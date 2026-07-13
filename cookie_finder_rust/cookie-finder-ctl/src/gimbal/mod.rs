mod stepper;
mod pan_tilt;

pub use pan_tilt::{MotorId, PanTiltGimbal};

#[cfg(not(target_os = "linux"))]
#[allow(dead_code)]
pub fn gimbal_available() -> bool {
    false
}

#[cfg(target_os = "linux")]
pub fn gimbal_available() -> bool {
    true
}
