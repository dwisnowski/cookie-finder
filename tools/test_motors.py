#!/usr/bin/env python3
"""
Stepper motor test script for pan/tilt gimbal.
Controls motors via command-line arguments.

Usage:
  test_stepper_motors.py [command] [steps]
  
Commands:
  pan-cw [steps]     Pan motor clockwise (default: 50)
  pan-ccw [steps]    Pan motor counter-clockwise (default: 50)
  tilt-cw [steps]    Tilt motor clockwise (default: 50) - NOT YET CONFIGURED
  tilt-ccw [steps]   Tilt motor counter-clockwise (default: 50) - NOT YET CONFIGURED
  auto               Automated test sequence
"""

from cookie_finder.gimbal.stepper import StepperMotor, MotorDirection
import sys
import time

# Orange Pi Zero 2W GPIO offsets (gpiochip1)
# Pan motor: 258(PI15), 268(PI12), 271(PI02), 272(PI16)
PAN_PINS = (258, 268, 271, 272)

def run_command(command, steps=50):
    """Execute motor control command."""
    
    # Initialize pan motor (no limit switch configured yet)
    pan_motor = StepperMotor(
        control_pins=PAN_PINS,
        max_angle=180.0,
        motor_name="Pan"
    )
    
    # TODO: Configure tilt motor pins and limit switch
    tilt_motor = None
    
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
            if tilt_motor is None:
                print("Tilt motor not yet configured")
                return 1
            print(f"Tilt CW {steps} steps...")
            tilt_motor.step(MotorDirection.CLOCKWISE, steps=steps)
            print(f"Tilt: {tilt_motor.get_angle():.1f}°")
            
        elif command == "tilt-ccw":
            if tilt_motor is None:
                print("Tilt motor not yet configured")
                return 1
            print(f"Tilt CCW {steps} steps...")
            tilt_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=steps)
            print(f"Tilt: {tilt_motor.get_angle():.1f}°")
            
        elif command == "home":
            print("Home command not yet available (limit switch not configured)")
            return 1
            
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
            
            if tilt_motor is not None:
                print("\nTesting TILT motor...")
                print("  CW 100 steps...")
                tilt_motor.step(MotorDirection.CLOCKWISE, steps=100)
                print(f"  Tilt angle: {tilt_motor.get_angle():.1f}°")
                time.sleep(0.5)
                
                print("  CCW 50 steps...")
                tilt_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=50)
                print(f"  Tilt angle: {tilt_motor.get_angle():.1f}°")
            else:
                print("\nTilt motor not yet configured")
            
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
        if tilt_motor is not None:
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
