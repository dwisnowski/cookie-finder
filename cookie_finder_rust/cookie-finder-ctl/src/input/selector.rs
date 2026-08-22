/// Which gamepad the control loop should use.
///
/// When `address` is set (Bluetooth MAC), only that HID device is accepted.
/// Otherwise any recognizable gamepad name is used (standalone `run` mode).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GamepadSelector {
    pub address: Option<String>,
    pub name: Option<String>,
}

impl GamepadSelector {
    pub fn any() -> Self {
        Self::default()
    }

    pub fn from_address_name(address: Option<&str>, name: Option<&str>) -> Self {
        Self {
            address: address
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty()),
            name: name
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string),
        }
    }
}
