#!/usr/bin/env python3
"""
Probe serial port for raw 16-bit thermal data from Pixfra/Mileseey camera.
Uses pyserial to communicate with CDC Data interface via /dev/tty.usbmodem*.
"""

import sys
import glob
import time

try:
    import serial
except ImportError:
    print("Error: pyserial not installed. Run: uv sync")
    sys.exit(1)

def find_camera_port():
    """Find the USB modem serial port."""
    ports = glob.glob("/dev/tty.usbmodem*")
    
    if not ports:
        print("✗ No USB modem ports found")
        return None
    
    if len(ports) > 1:
        print(f"Found {len(ports)} ports:")
        for i, port in enumerate(ports):
            print(f"  {i}: {port}")
        choice = input("Select port (0): ").strip() or "0"
        try:
            return ports[int(choice)]
        except (ValueError, IndexError):
            return ports[0]
    
    return ports[0]

def test_serial_connection(port):
    """Test basic serial connection and send probe commands."""
    print("\n" + "="*60)
    print(f"Testing Serial Connection: {port}")
    print("="*60)
    
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=1)
        print(f"✓ Connected to {port} at 115200 baud")
        
        # Test 1: Send newline to see if camera responds with help/version
        print("\nTest 1: Sending newline (\\r\\n)...")
        ser.write(b'\r\n')
        time.sleep(0.5)
        
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            print(f"✓ Response received: {response[:100]}")
            print(f"  Full response ({len(response)} bytes): {response.hex()[:100]}...")
        else:
            print("✗ No response to newline")
        
        # Test 2: Send "Get Version/Status" command
        print("\nTest 2: Sending Get Version/Status command (0x55 0xAA 0x07...)...")
        cmd = b'\x55\xAA\x07\x00\x00\x00\x00\x00'
        ser.write(cmd)
        time.sleep(0.5)
        
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            print(f"✓ Response received: {len(response)} bytes")
            print(f"  First 32 bytes (hex): {response[:32].hex()}")
            
            # Check if this looks like thermal data (98304 bytes = 256x192x2)
            if len(response) == 98304:
                print(f"  ✓ Data size matches 256x192x2 thermal frame!")
                return True
        else:
            print("✗ No response to command")
        
        # Test 3: Try to read raw data stream
        print("\nTest 3: Listening for raw data stream (5 seconds)...")
        print("(Waiting for camera to send thermal frames...)")
        
        start_time = time.time()
        total_bytes = 0
        frame_count = 0
        
        while time.time() - start_time < 5:
            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                total_bytes += len(data)
                
                # Check for thermal frame size
                if len(data) == 98304 or total_bytes % 98304 == 0:
                    frame_count += 1
                    print(f"  Frame {frame_count}: {len(data)} bytes received")
                    print(f"    First 32 bytes: {data[:32].hex()}")
                    
                    if len(data) == 98304:
                        print(f"    ✓ Valid 256x192x2 thermal frame!")
                        return True
                else:
                    print(f"  Received {len(data)} bytes (total: {total_bytes})")
            
            time.sleep(0.1)
        
        if total_bytes > 0:
            print(f"\n✓ Received {total_bytes} bytes total")
            if total_bytes >= 98304:
                print(f"  ✓ Enough data for at least one thermal frame!")
                return True
        else:
            print("✗ No data received")
        
        ser.close()
        return False
        
    except serial.SerialException as e:
        print(f"✗ Serial error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def analyze_thermal_data(port):
    """Capture and analyze thermal frame data."""
    print("\n" + "="*60)
    print("Capturing Thermal Frame Data")
    print("="*60)
    
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=2)
        
        # Send command to request thermal data
        print("Sending thermal data request...")
        cmd = b'\x55\xAA\x07\x00\x00\x00\x00\x00'
        ser.write(cmd)
        
        # Read one complete frame (256*192*2 = 98304 bytes)
        print("Waiting for thermal frame (98304 bytes)...")
        frame_data = b''
        timeout_count = 0
        
        while len(frame_data) < 98304 and timeout_count < 10:
            if ser.in_waiting > 0:
                chunk = ser.read(min(ser.in_waiting, 98304 - len(frame_data)))
                frame_data += chunk
                print(f"  Received {len(frame_data)}/{98304} bytes", end='\r')
                timeout_count = 0
            else:
                timeout_count += 1
                time.sleep(0.1)
        
        print()
        
        if len(frame_data) >= 98304:
            print(f"✓ Captured complete thermal frame ({len(frame_data)} bytes)")
            
            # Parse as 16-bit little-endian values
            import struct
            pixels = struct.unpack('<' + 'H' * (len(frame_data) // 2), frame_data[:98304])
            
            min_val = min(pixels)
            max_val = max(pixels)
            avg_val = sum(pixels) // len(pixels)
            
            print(f"\nThermal Data Analysis:")
            print(f"  Min value: {min_val} (0x{min_val:04x})")
            print(f"  Max value: {max_val} (0x{max_val:04x})")
            print(f"  Avg value: {avg_val} (0x{avg_val:04x})")
            
            # Try to interpret as temperature
            print(f"\nAssuming Kelvin x10 format:")
            print(f"  Min temp: {(min_val / 10) - 273.15:.1f}°C")
            print(f"  Max temp: {(max_val / 10) - 273.15:.1f}°C")
            print(f"  Avg temp: {(avg_val / 10) - 273.15:.1f}°C")
            
            return True
        else:
            print(f"✗ Incomplete frame: only {len(frame_data)} bytes received")
            return False
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        try:
            ser.close()
        except:
            pass

def main():
    print("Thermal Camera Serial Port Probe")
    print("Reading raw 16-bit thermal data via CDC interface\n")
    
    # Find camera port
    port = find_camera_port()
    if not port:
        print("\nNo camera port found. Make sure camera is connected.")
        return 1
    
    print(f"✓ Using port: {port}")
    
    # Test serial connection
    success = test_serial_connection(port)
    
    if success:
        print("\n" + "="*60)
        print("✓ SUCCESS: Raw thermal data is accessible!")
        print("="*60)
        
        # Try to capture and analyze a frame
        analyze_thermal_data(port)
        
        return 0
    else:
        print("\n" + "="*60)
        print("✗ Could not access raw thermal data via serial port")
        print("="*60)
        print("\nTroubleshooting:")
        print("  1. Verify camera is connected and powered")
        print("  2. Check port permissions: ls -la /dev/tty.usbmodem*")
        print("  3. Try different baud rates (9600, 38400, 115200)")
        print("  4. Camera may require specific initialization sequence")
        return 1

if __name__ == "__main__":
    sys.exit(main())
