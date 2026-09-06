"""WiFi client / access-point mode management for Orange Pi."""

from cookie_finder.wifi.manager import (
    AP_GATEWAY,
    AP_PASSPHRASE,
    AP_PROFILES,
    AP_SSID,
    DEFAULT_AP_PROFILE,
    apply_boot_wifi_policy,
    ap_gateway_for,
    ap_url_for,
    get_ap_profile,
    get_ap_profile_info,
    get_desired_mode,
    get_switch_instructions,
    get_wifi_status,
    set_ap_profile,
    set_desired_mode,
    set_wifi_mode,
)

__all__ = [
    "AP_GATEWAY",
    "AP_PASSPHRASE",
    "AP_PROFILES",
    "AP_SSID",
    "DEFAULT_AP_PROFILE",
    "apply_boot_wifi_policy",
    "ap_gateway_for",
    "ap_url_for",
    "get_ap_profile",
    "get_ap_profile_info",
    "get_desired_mode",
    "get_switch_instructions",
    "get_wifi_status",
    "set_ap_profile",
    "set_desired_mode",
    "set_wifi_mode",
]
