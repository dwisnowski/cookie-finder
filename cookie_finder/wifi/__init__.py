"""WiFi client / access-point mode management for Orange Pi."""

from cookie_finder.wifi.manager import (
    AP_GATEWAY,
    AP_PASSPHRASE,
    AP_SSID,
    apply_boot_wifi_policy,
    get_switch_instructions,
    get_wifi_status,
    set_wifi_mode,
)

__all__ = [
    "AP_GATEWAY",
    "AP_PASSPHRASE",
    "AP_SSID",
    "apply_boot_wifi_policy",
    "get_switch_instructions",
    "get_wifi_status",
    "set_wifi_mode",
]
