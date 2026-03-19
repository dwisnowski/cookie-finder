#!/usr/bin/env python3
"""
Probe UVC Extension Units (XU) for hidden 16-bit thermal data modes.
Also checks for alternative USB interface settings that macOS may hide.
"""

import sys
import subprocess

try:
    import usb.core
    import usb.util
except ImportError:
    print("Error: pyusb not installed. Run: uv sync")
    sys.exit(1)

def check_libuvc():
    """Check if libuvc is installed."""
    print("Checking for libuvc installation...")
    result = subprocess.run(["which", "uvc-ctrl"], capture_output=True)
    
    if result.returncode == 0:
        print("✓ libuvc tools found")
        return True
    else:
        print("✗ libuvc not installed")
        print("  Install with: brew install libuvc")
        return False

def probe_uvc_extension_units():
    """Probe for UVC Extension Units that may control 16-bit mode."""
    print("\n" + "="*60)
    print("Probing UVC Extension Units (XU)")
    print("="*60 + "\n")
    
    VENDOR_ID = 0x2e03
    PRODUCT_ID = 0x2507
    
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    
    if dev is None:
        print("✗ Camera not found")
        return False
    
    print(f"✓ Found camera: {dev.manufacturer} {dev.product}\n")
    
    try:
        dev.set_configuration()
        
        # Scan all interfaces for Extension Units
        found_xu = False
        
        for cfg in dev:
            for intf in cfg:
                print(f"Interface {intf.bInterfaceNumber}:")
                print(f"  Class: {intf.bInterfaceClass}")
                print(f"  Subclass: {intf.bInterfaceSubClass}")
                print(f"  Protocol: {intf.bInterfaceProtocol}")
                
                # Check for Video Control interface (Class 14, Subclass 1)
                if intf.bInterfaceClass == 14 and intf.bInterfaceSubClass == 1:
                    print(f"  ✓ Video Control Interface")
                    
                    # Try to read Extension Unit descriptor
                    # XU descriptors are typically in the interface descriptor
                    try:
                        # Request descriptor
                        desc = dev.ctrl_transfer(
                            0x80,  # bmRequestType: Device-to-Host, Standard, Device
                            6,     # bRequest: GET_DESCRIPTOR
                            0x2400,  # wValue: Descriptor Type (0x24) and Index (0x00)
                            intf.bInterfaceNumber,  # wIndex: Interface
                            255    # wLength
                        )
                        print(f"    Descriptor data: {bytes(desc).hex()[:100]}...")
                        found_xu = True
                    except:
                        pass
                
                # Check for alternate settings
                if intf.bInterfaceNumber == 1:  # Video Streaming interface
                    print(f"  Checking alternate settings...")
                    
                    # Try to enumerate alternate settings
                    for alt_setting in range(10):
                        try:
                            dev.set_interface_altsetting(intf.bInterfaceNumber, alt_setting)
                            print(f"    ✓ Alternate Setting {alt_setting} available")
                            
                            # Check endpoints for this setting
                            for ep in intf:
                                ep_dir = "IN" if ep.bEndpointAddress & 0x80 else "OUT"
                                print(f"      Endpoint {hex(ep.bEndpointAddress)}: {ep_dir}")
                            
                            found_xu = True
                        except:
                            if alt_setting == 0:
                                continue
                            break
        
        return found_xu
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def probe_with_libuvc():
    """Use libuvc command-line tools to probe camera."""
    print("\n" + "="*60)
    print("Probing with libuvc Tools")
    print("="*60 + "\n")
    
    # Try uvc-ctrl to list controls
    print("Attempting to list UVC controls with uvc-ctrl...")
    result = subprocess.run(
        ["uvc-ctrl", "-d", "0", "-l"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ UVC Controls:")
        print(result.stdout)
        
        # Look for Extension Unit controls
        if "Extension" in result.stdout or "XU" in result.stdout:
            print("\n✓ Extension Unit controls found!")
            return True
    else:
        print("✗ Could not list UVC controls")
        if result.stderr:
            print(f"  Error: {result.stderr}")
    
    return False

def check_alternative_formats():
    """Check for alternative video formats."""
    print("\n" + "="*60)
    print("Checking Alternative Video Formats")
    print("="*60 + "\n")
    
    import cv2
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return False
    
    print("Checking for Y16 (16-bit grayscale) format...")
    
    # Try to set Y16 format (if supported)
    try:
        # Some cameras support format selection via CAP_PROP_FOURCC
        fourcc_codes = [
            (0x36313659, "Y16"),   # Y16 - 16-bit grayscale
            (0x59555932, "YUY2"),  # YUY2 - 8-bit YUV
            (0x4D4A5047, "MJPG"),  # MJPG - Motion JPEG
        ]
        
        for code, name in fourcc_codes:
            cap.set(cv2.CAP_PROP_FOURCC, code)
            actual_code = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ {name}: {frame.shape} (dtype: {frame.dtype})")
                
                if frame.dtype == np.uint16:
                    print(f"    ✓ 16-bit data detected!")
                    return True
            else:
                print(f"  ✗ {name}: Failed to capture")
    except Exception as e:
        print(f"  Error: {e}")
    
    cap.release()
    return False

def main():
    print("="*60)
    print("Thermal Camera UVC Extension Unit Probe")
    print("="*60 + "\n")
    
    # Check for libuvc
    has_libuvc = check_libuvc()
    
    # Probe UVC Extension Units
    has_xu = probe_uvc_extension_units()
    
    # Try libuvc tools if available
    if has_libuvc:
        has_libuvc_controls = probe_with_libuvc()
    else:
        has_libuvc_controls = False
    
    # Check alternative formats
    import numpy as np
    has_y16 = check_alternative_formats()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if has_xu:
        print("✓ Extension Units or alternate settings found")
        print("  Camera may support hidden 16-bit mode")
    else:
        print("✗ No Extension Units found")
    
    if has_y16:
        print("✓ Y16 (16-bit) format available!")
    else:
        print("✗ Y16 format not available")
    
    if not has_libuvc:
        print("\nTo probe deeper, install libuvc:")
        print("  brew install libuvc")
    
    print("\n" + "="*60)
    print("Conclusion")
    print("="*60)
    
    if has_xu or has_y16:
        print("✓ Potential path to 16-bit data found")
    else:
        print("✗ Camera appears locked to 8-bit UVC stream")
        print("\nAlternatives for thermal analysis:")
        print("  1. Use 8-bit stream with manual gain control")
        print("  2. Implement color-to-temperature LUT for Ironbow images")
        print("  3. Use relative temperature differences instead of absolute")
    
    print("="*60 + "\n")
    
    return 0 if (has_xu or has_y16) else 1

if __name__ == "__main__":
    sys.exit(main())
