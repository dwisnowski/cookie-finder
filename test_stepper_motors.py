#!/usr/bin/env python3
"""
Interactive stepper motor control using keyboard arrow keys.
Left/Right arrows: Pan motor
Up/Down arrows: Tilt motor
ESC: Exit
"""

from stepper_motor_controller import StepperMotor, MotorDirection
import time
import sys

try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("⚠️  pynput not installed. Install with: pip install pynput")
    sys.exit(1)

# Pin assignments for Orange Pi Zero 2W
PAN_PINS = (10, 11, 12, 13)      # PI10, PI11, PI12, PI13
TILT_PINS = (14, 15, 16, 17)     # PI14, PI15, PI16, PI17
PAN_LIMIT = 18                     # PI18
TILT_LIMIT = 19                    # PI19

# Track pressed keys
pressed_keys = set()

def on_press(key):
    """Handle key press."""
    try:
        pressed_keys.add(key)
    except AttributeError:
        pass

def on_release(key):
    """Handle key release."""
    try:
        pressed_keys.discard(key)
    except AttributeError:
        pass

def control_motors():
    """Main interactive control loop."""
    print("\n=== Keyboard Motor Control ===\n")
    print("Controls:")
    print("  LEFT/RIGHT arrows  → Pan motor")
    print("  UP/DOWN arrows     → Tilt motor")
    print("  ESC                → Exit\n")
    
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
    
    # Start keyboard listener
    exit_event = False
    
    def on_release_exit(key):
        nonlocal exit_event
        try:
            pressed_keys.discard(key)
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            exit_event = True
            return False
    
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release_exit
    )
    listener.start()
    
    try:
        while listener.is_alive() and not exit_event:
            moved = False
            
            # Pan motor control (left/right)
            if keyboard.Key.left in pressed_keys:
                pan_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=5)
                print(f"Pan: {pan_motor.get_angle():.1f}° | Tilt: {tilt_motor.get_angle():.1f}°", end='\r')
                moved = True
            
            if keyboard.Key.right in pressed_keys:
                pan_motor.step(MotorDirection.CLOCKWISE, steps=5)
                print(f"Pan: {pan_motor.get_angle():.1f}° | Tilt: {tilt_motor.get_angle():.1f}°", end='\r')
                moved = True
            
            # Tilt motor control (up/down)
            if keyboard.Key.up in pressed_keys:
                tilt_motor.step(MotorDirection.CLOCKWISE, steps=5)
                print(f"Pan: {pan_motor.get_angle():.1f}° | Tilt: {tilt_motor.get_angle():.1f}°", end='\r')
                moved = True
            
            if keyboard.Key.down in pressed_keys:
                tilt_motor.step(MotorDirection.COUNTERCLOCKWISE, steps=5)
                print(f"Pan: {pan_motor.get_angle():.1f}° | Tilt: {tilt_motor.get_angle():.1f}°", end='\r')
                moved = True
            
            if not moved:
                time.sleep(0.01)  # Idle polling
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    finally:
        listener.stop()
        pan_motor.cleanup()
        tilt_motor.cleanup()
        print("\n✅ Motors cleaned up")

if __name__ == "__main__":
    control_motors()
