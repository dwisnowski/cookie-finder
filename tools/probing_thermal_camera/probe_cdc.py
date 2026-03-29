#!/usr/bin/env python3
"""
Probe CDC Data interface on thermal camera for 16-bit raw thermal data.
Sends UVC Extension Unit unlock command to activate raw data stream.
"""

import sys
import os
import glob

try:
    import usb.core
    import usb.util
except ImportError:
    print("Error: pyusb not installed. Run: uv sync")
    sys.exit(1)

def check_serial_ports():
    """Check for USB modem serial ports on macOS."""
    print("\n" + "="*60)
    print("Checking for USB Serial Ports...")
    print("="*60)
    
    # Use glob to find actual /dev/tty.usbmodem* devices
    ports = glob.glob("/dev/tty.usbmodem*")
    
    if ports:
        print(f"✓ Found {len(ports)} USB modem port(s):")
        for port in ports:
            print(f"  - {port}")
        return True
    else:
        print("✗ No USB modem ports found (/dev/tty.usbmodem*)")
        print("  Camera may not have CDC serial interface active")
        return False

def send_unlock_command():
    """Send UVC Extension Unit unlock command to activate raw data."""
    print("\n" + "="*60)
    print("Sending UVC Extension Unit Unlock Command...")
    print("="*60)
    
    # Pixfra/Mileseey IDs
    VENDOR_ID = 0x2e03
    PRODUCT_ID = 0x2507
    
    dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
    
    if dev is None:
        print("✗ Camera not found (Vendor: 0x2e03, Product: 0x2507)")
        return False
    
    print(f"✓ Found camera: {dev.manufacturer} {dev.product}")
    
    try:
        # Set configuration
        dev.set_configuration()
        print("✓ Configuration set")
        
        # Send the "Unlock Raw Data" Control Request
        # bmRequestType: 0x21 (Class, Interface, Host-to-Device)
        # bRequest: 0x01 (SET_CUR)
        # wValue: 0x0100
        # wIndex: 0x0300 (Interface 3 - CDC Data)
        print("\nSending unlock request to Control Endpoint...")
        print("  bmRequestType: 0x21 (Class, Interface, Host-to-Device)")
        print("  bRequest: 0x01 (SET_CUR)")
        print("  wValue: 0x0100")
        print("  wIndex: 0x0300 (Interface 3)")
        
        try:
            dev.ctrl_transfer(0x21, 0x01, 0x0100, 0x0300, b'\x01\x00\x00\x00')
            print("✓ Unlock request sent successfully")
        except usb.core.USBError as e:
            print(f"✗ Unlock request failed: {e}")
            return False
        
        # Try to read from Bulk IN endpoint (0x85)
        print("\nListening for raw 16-bit thermal data on Endpoint 0x85...")
        print("(Waiting up to 2 seconds for data...)")
        
        try:
            # Read a chunk of data (256*192*2 = 98304 bytes for typical thermal frame)
            raw_data = dev.read(0x85, 98304, timeout=2000)
            
            print(f"\n✓ SUCCESS! Captured {len(raw_data)} bytes of raw data")
            print(f"  First 32 bytes (hex): {bytes(raw_data[:32]).hex()}")
            
            # Analyze the data
            if len(raw_data) == 98304:
                print(f"  ✓ Data size matches 256x192x2 thermal frame (98304 bytes)")
                print(f"  ✓ 16-bit raw thermal data is accessible!")
            else:
                print(f"  ⚠ Unexpected data size: {len(raw_data)} bytes")
            
            return True
            
        except usb.core.USBTimeoutError:
            print("✗ Timeout: No data received on Endpoint 0x85")
            print("  Camera may not have raw data enabled")
            return False
        except usb.core.USBError as e:
            print(f"✗ Read error: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("Thermal Camera CDC Data Probe")
    print("Testing for hidden 16-bit raw thermal data interface\n")
    
    # Check serial ports first
    has_serial = check_serial_ports()
    
    # Send unlock command and probe CDC interface
    has_raw_data = send_unlock_command()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    if has_serial:
        print("✓ USB modem serial port detected")
        print("  Camera has CDC Data interface active")
    else:
        print("✗ No USB modem serial port")
        print("  (This is normal if camera is in UVC-only mode)")
    
    if has_raw_data:
        print("✓ Raw 16-bit thermal data is accessible!")
        print("\nNext steps:")
        print("  1. Implement CDC bulk read loop")
        print("  2. Parse 16-bit thermal frames (256x192 pixels)")
        print("  3. Convert to temperature values using calibration")
    else:
        print("✗ Could not access raw 16-bit data")
        print("\nAlternatives:")
        print("  1. Camera may only support 8-bit UVC video")
        print("  2. Try different unlock command sequences")
        print("  3. Check camera firmware version")
    
    print("="*60 + "\n")
    
    return 0 if has_raw_data else 1

if __name__ == "__main__":
    sys.exit(main())
