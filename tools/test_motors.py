#!/usr/bin/env python3
"""
Stepper motor test script for pan/tilt gimbal.
Controls motors via command-line arguments.

Usage:
  test_motors.py [command] [steps]
  
Commands:
  pan-cw [steps]     Pan motor clockwise (default: 50)
  pan-ccw [steps]    Pan motor counter-clockwise (default: 50)
  tilt-cw [steps]    Tilt motor clockwise (default: 50)
  tilt-ccw [steps]   Tilt motor counter-clockwise (default: 50)
  auto               Automated test sequence
  home               Home both motors to limit switches
"""

from cookie_finder.gimbal.stepper import StepperMotor, MotorDirection
import sys
import time

# Orange Pi Zero 2W GPIO offsets (gpiochip1)
# Pan motor:  271(PI15), 268(PI12), 258(PI02), 272(PI16)
PAN_PINS = (271, 268, 258, 272)
PAN_LIMIT_MIN = 257  # physical 12
PAN_LIMIT_MAX = 227  # physical 13

# Tilt motor: 262(RXD.2), 229(CE.0), 233(CE.1), 265(SCL.2)
TILT_PINS = (262, 229, 233, 265)
TILT_LIMIT_MIN = 261  # physical 15
TILT_LIMIT_MAX = 270  # physical 16

def run_command(command, steps=50):
    """Execute motor control command."""

    # Initialize pan motor
    pan_motor = StepperMotor(
        control_pins=PAN_PINS,
        limit_min_pin=PAN_LIMIT_MIN,
        limit_max_pin=PAN_LIMIT_MAX,
        max_angle=180.0,
        motor_name="Pan"
    )

    # Initialize tilt motor
    tilt_motor = StepperMotor(
        control_pins=TILT_PINS,
        limit_min_pin=TILT_LIMIT_MIN,
        limit_max_pin=TILT_LIMIT_MAX,
        max_angle=180.0,
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
            print("Homing both motors...")
            pan_motor.home()
            tilt_motor.home()
            print(f"Home complete: Pan {pan_motor.get_angle():.1f}°, Tilt {tilt_motor.get_angle():.1f}°")
            
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
