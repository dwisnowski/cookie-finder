#[cfg(target_os = "linux")]
mod linux;
#[cfg(not(target_os = "linux"))]
mod stub;

#[cfg(target_os = "linux")]
pub use linux::GamepadInput;
#[cfg(not(target_os = "linux"))]
pub use stub::GamepadInput;
