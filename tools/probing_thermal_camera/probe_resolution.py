#!/usr/bin/env python3
"""
Probe camera for multiplexed 16-bit thermal data in double-height frames.
Some Pixfra/Mileseey cameras hide raw 16-bit data in the bottom half of 256x384 frames.
"""

import sys
import cv2
import numpy as np

def test_resolutions():
    """Test various resolutions to find raw 16-bit data."""
    print("Thermal Camera Resolution Probe")
    print("Testing for multiplexed 16-bit data in double-height frames\n")
    
    # Try to open camera with AVFoundation backend
    print("Opening camera with AVFoundation backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return False
    
    print("✓ Camera opened\n")
    
    # Test resolutions
    test_resolutions_list = [
        (256, 192, "Standard thermal (256x192)"),
        (256, 384, "Double-height with raw data (256x384)"),
        (512, 384, "Upscaled thermal (512x384)"),
        (320, 240, "QVGA thermal"),
        (640, 480, "VGA thermal"),
        (512, 390, "Pixfra native (512x390)"),
    ]
    
    found_raw = False
    
    for width, height, description in test_resolutions_list:
        print(f"Testing {description} ({width}x{height})...")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual resolution set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  Actual resolution: {actual_width}x{actual_height}")
        
        # Try to capture frame
        ret, frame = cap.read()
        
        if ret:
            print(f"  ✓ Captured frame: {frame.shape}")
            
            # Check if this is a double-height frame with raw data
            if actual_height == 384 and actual_width == 256:
                print(f"  ⚠ Double-height frame detected!")
                print(f"    Top half (visual): {actual_width}x{actual_height//2}")
                print(f"    Bottom half (raw?): {actual_width}x{actual_height//2}")
                
                # Analyze top and bottom halves
                top_half = frame[:actual_height//2, :]
                bottom_half = frame[actual_height//2:, :]
                
                top_mean = np.mean(top_half)
                bottom_mean = np.mean(bottom_half)
                top_std = np.std(top_half)
                bottom_std = np.std(bottom_half)
                
                print(f"\n    Top half statistics:")
                print(f"      Mean: {top_mean:.1f}, Std: {top_std:.1f}")
                print(f"    Bottom half statistics:")
                print(f"      Mean: {bottom_mean:.1f}, Std: {bottom_std:.1f}")
                
                # Raw data typically has higher variance and different mean
                if bottom_std > top_std * 1.5 or abs(bottom_mean - top_mean) > 20:
                    print(f"\n    ✓ Bottom half looks like raw data!")
                    print(f"      (Higher variance/different mean suggests 16-bit data)")
                    found_raw = True
                    
                    # Try to parse as 16-bit
                    print(f"\n    Attempting to parse as 16-bit little-endian...")
                    try:
                        # Convert bottom half to 16-bit
                        bottom_flat = bottom_half.flatten()
                        
                        # Try to interpret pairs of bytes as 16-bit values
                        if len(bottom_flat) % 2 == 0:
                            import struct
                            values_16bit = struct.unpack('<' + 'H' * (len(bottom_flat) // 2), 
                                                        bytes(bottom_flat[:len(bottom_flat)//2*2]))
                            
                            min_val = min(values_16bit)
                            max_val = max(values_16bit)
                            avg_val = sum(values_16bit) // len(values_16bit)
                            
                            print(f"      Min: {min_val} (0x{min_val:04x})")
                            print(f"      Max: {max_val} (0x{max_val:04x})")
                            print(f"      Avg: {avg_val} (0x{avg_val:04x})")
                            
                            # Try temperature conversion
                            print(f"\n      Assuming Kelvin x10 format:")
                            print(f"        Min temp: {(min_val / 10) - 273.15:.1f}°C")
                            print(f"        Max temp: {(max_val / 10) - 273.15:.1f}°C")
                            print(f"        Avg temp: {(avg_val / 10) - 273.15:.1f}°C")
                    except Exception as e:
                        print(f"      Error parsing: {e}")
                else:
                    print(f"\n    ✗ Bottom half doesn't look like raw data")
                    print(f"      (Similar statistics to top half)")
            
            # Save frame for inspection
            filename = f"thermal_probe_{actual_width}x{actual_height}.png"
            cv2.imwrite(filename, frame)
            print(f"  Saved to: {filename}")
        else:
            print(f"  ✗ Failed to capture frame")
        
        print()
    
    cap.release()
    
    return found_raw

def test_uvc_modes():
    """Test different UVC modes."""
    print("\n" + "="*60)
    print("Testing UVC Modes")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    
    if not cap.isOpened():
        print("✗ Could not open camera")
        return
    
    # Try different CAP_PROP_MODE values
    print("Testing different UVC modes...")
    for mode in [0, 1, 2, 3, 4, 5]:
        try:
            cap.set(cv2.CAP_PROP_MODE, mode)
            actual_mode = int(cap.get(cv2.CAP_PROP_MODE))
            ret, frame = cap.read()
            
            if ret:
                print(f"  Mode {mode}: ✓ {frame.shape}")
            else:
                print(f"  Mode {mode}: ✗ Failed to capture")
        except:
            pass
    
    cap.release()

def main():
    print("="*60)
    print("Thermal Camera UVC Multiplexing Probe")
    print("="*60 + "\n")
    
    found_raw = test_resolutions()
    
    test_uvc_modes()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if found_raw:
        print("✓ Found potential 16-bit raw data in double-height frame!")
        print("\nNext steps:")
        print("  1. Extract bottom half of 256x384 frame")
        print("  2. Parse as 16-bit little-endian values")
        print("  3. Convert to temperature using calibration")
    else:
        print("✗ No multiplexed 16-bit data found")
        print("\nAlternatives:")
        print("  1. Camera may only support 8-bit UVC video")
        print("  2. Check camera firmware for raw mode support")
        print("  3. Use standard 8-bit thermal imaging")
    
    print("="*60 + "\n")
    
    return 0 if found_raw else 1

if __name__ == "__main__":
    sys.exit(main())
