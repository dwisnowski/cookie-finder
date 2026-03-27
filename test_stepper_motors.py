#!/usr/bin/env python3
"""
Stepper motor test script for pan/tilt gimbal.
Controls motors via command-line arguments.

Usage:
  test_stepper_motors.py [command] [steps]
  
Commands:
  pan-cw [steps]     Pan motor clockwise (default: 50)
  pan-ccw [steps]    Pan motor counter-clockwise (default: 50)
  tilt-cw [steps]    Tilt motor clockwise (default: 50)
  tilt-ccw [steps]   Tilt motor counter-clockwise (default: 50)
  home               Home both motors (hits limit switches)
  auto               Automated test sequence
"""

from stepper_motor_controller import StepperMotor, MotorDirection
import sys
import time

# GPIO offsets for Orange Pi Zero 2W (H618 SoC, gpiochip1)
# Confirmed working with logic analyzer
PAN_PINS = (258, 268, 271, 272)   # Pan motor GPIO offsets
TILT_PINS = (273, 274, 275, 276)  # Tilt motor GPIO offsets (estimated)
PAN_LIMIT = 277                    # Pan limit switch GPIO offset (estimated)
TILT_LIMIT = 278                   # Tilt limit switch GPIO offset (estimated)

def run_command(command, steps=50):
    """Execute motor control command."""
    
    # Initialize motors
    pan_motor = StepperMotor(
        control_pins=PAN_PINS,
        limit_switch_pin=PAN_LIMIT,
        max_angle=180.0,
        motor_name="Pan"
    )
    
    tilt_motor = StepperMotor(
        control_pins=TILT_PINS,
        limit_switch_pin=TILT_LIMIT,
        max_angle=90.0,
        motor_name="Tilt"
    )
    
    try:
        if command == "pan-cw":
            print(f"Pan CW {steps} steps...")
            pan_motor.step(MotorDirection.CLOCKWISE, steps=steps)
            print(f"Pan: {pan_motor.get_angle():.1f}°")
            
        elif command == "pan-ccw":
            print(f"Pan CCW {steps} steps...")
            pan_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=steps)
            print(f"Pan: {pan_motor.get_angle():.1f}°")
            
        elif command == "tilt-cw":
            print(f"Tilt CW {steps} steps...")
            tilt_motor.step(MotorDirection.CLOCKWISE, steps=steps)
            print(f"Tilt: {tilt_motor.get_angle():.1f}°")
            
        elif command == "tilt-ccw":
            print(f"Tilt CCW {steps} steps...")
            tilt_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=steps)
            print(f"Tilt: {tilt_motor.get_angle():.1f}°")
            
        elif command == "home":
            print("Homing motors...")
            pan_motor.home()
            tilt_motor.home()
            print(f"Pan: {pan_motor.get_angle():.1f}° | Tilt: {tilt_motor.get_angle():.1f}°")
            
        elif command == "auto":
            print("=== Automated Motor Test ===\n")
            
            print("Testing PAN motor...")
            print("  CW 100 steps...")
            pan_motor.step(MotorDirection.CLOCKWISE, steps=100)
            print(f"  Pan angle: {pan_motor.get_angle():.1f}°")
            time.sleep(0.5)
            
            print("  CCW 50 steps...")
            pan_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=50)
            print(f"  Pan angle: {pan_motor.get_angle():.1f}°")
            time.sleep(0.5)
            
            print("\nTesting TILT motor...")
            print("  CW 100 steps...")
            tilt_motor.step(MotorDirection.CLOCKWISE, steps=100)
            print(f"  Tilt angle: {tilt_motor.get_angle():.1f}°")
            time.sleep(0.5)
            
            print("  CCW 50 steps...")
            tilt_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=50)
            print(f"  Tilt angle: {tilt_motor.get_angle():.1f}°")
            
            print("\n✅ Test complete!")
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        pan_motor.cleanup()
        tilt_motor.cleanup()
    
    return 0

def main():
    """Parse arguments and run command."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 0
    
    command = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    return run_command(command, steps)

if __name__ == "__main__":
    sys.exit(main())
