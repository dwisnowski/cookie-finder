#!/usr/bin/env python3
"""
Probe USB thermal camera for hidden interfaces and raw data endpoints.
Detects vendor-specific interfaces that may contain 16-bit raw thermal data.
"""

import sys

try:
    import usb.core
    import usb.util
except ImportError:
    print("Error: pyusb not installed. Install with: uv sync")
    sys.exit(1)

# Check for libusb backend
try:
    import usb.backend.libusb1
    backend = usb.backend.libusb1.get_backend()
    if backend is None:
        print("Error: libusb backend not available.")
        print("Install libusb with: brew install libusb")
        sys.exit(1)
except ImportError:
    print("Error: libusb1 backend not found.")
    print("Install with: brew install libusb")
    sys.exit(1)

# Known thermal camera vendor/product IDs
THERMAL_DEVICES = [
    (0x2e03, 0x2507, "Mileseey/Pixfra TNV30i"),
    (0x2e03, 0x2508, "Mileseey/Pixfra TNV30"),
    (0x1e4e, 0x0100, "Seek Thermal"),
]

def probe_device(dev):
    """Probe a USB device for all interfaces and endpoints."""
    print(f"\n{'='*60}")
    print(f"Device: {dev.manufacturer} {dev.product}")
    print(f"Vendor ID: {hex(dev.idVendor)}, Product ID: {hex(dev.idProduct)}")
    print(f"{'='*60}\n")
    
    found_vendor_specific = False
    
    try:
        for cfg in dev:
            print(f"Configuration {cfg.bConfigurationValue}:")
            for intf in cfg:
                class_name = get_class_name(intf.bInterfaceClass)
                print(f"  Interface {intf.bInterfaceNumber}: Class {intf.bInterfaceClass} ({class_name})")
                
                if intf.bInterfaceClass == 255:
                    found_vendor_specific = True
                    print(f"    ⚠️  VENDOR-SPECIFIC INTERFACE FOUND - May contain raw data!")
                
                for ep in intf:
                    ep_dir = "IN" if ep.bEndpointAddress & 0x80 else "OUT"
                    ep_type = get_endpoint_type(ep.bmAttributes)
                    print(f"      Endpoint {hex(ep.bEndpointAddress)}: {ep_dir} ({ep_type}), Max Packet: {ep.wMaxPacketSize}")
    except usb.core.USBError as e:
        print(f"Error accessing device: {e}")
        return False
    
    return found_vendor_specific

def get_class_name(class_code):
    """Get human-readable USB class name."""
    classes = {
        0: "Device",
        1: "Audio",
        2: "Communications",
        3: "HID",
        5: "Physical",
        6: "Image",
        7: "Printer",
        8: "Mass Storage",
        9: "Hub",
        10: "CDC Data",
        11: "Smart Card",
        13: "Content Security",
        14: "Video (UVC)",
        15: "Personal Healthcare",
        16: "Audio/Video",
        17: "Billboard",
        224: "Diagnostic Device",
        239: "Miscellaneous",
        255: "Vendor Specific",
    }
    return classes.get(class_code, "Unknown")

def get_endpoint_type(attributes):
    """Get human-readable endpoint type."""
    types = {
        0: "Control",
        1: "Isochronous",
        2: "Bulk",
        3: "Interrupt",
    }
    return types.get(attributes & 0x03, "Unknown")

def main():
    print("Thermal Camera USB Probe")
    print("Searching for thermal camera devices...\n")
    
    found_any = False
    found_vendor_specific = False
    
    # Search for known thermal devices
    for vendor_id, product_id, name in THERMAL_DEVICES:
        dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
        if dev is not None:
            found_any = True
            if probe_device(dev):
                found_vendor_specific = True
    
    # Also search for any video devices
    print("\nSearching for all video devices (Class 14)...\n")
    for dev in usb.core.find(find_all=True, bDeviceClass=0):
        try:
            for cfg in dev:
                for intf in cfg:
                    if intf.bInterfaceClass == 14:  # Video
                        found_any = True
                        if probe_device(dev):
                            found_vendor_specific = True
                        break
        except:
            pass
    
    if not found_any:
        print("No thermal camera devices found.")
        print("\nTry running: make list-cameras")
        return 1
    
    print(f"\n{'='*60}")
    if found_vendor_specific:
        print("✓ VENDOR-SPECIFIC INTERFACE DETECTED")
        print("  Raw 16-bit thermal data may be accessible via vendor commands.")
        print("  Next step: Use PyUSB to send vendor-specific control requests.")
    else:
        print("✗ No vendor-specific interfaces found.")
        print("  Camera is likely only exposing standard 8-bit UVC video.")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
