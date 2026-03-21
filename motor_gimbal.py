"""
High-level pan/tilt gimbal control using two stepper motors.

This module provides a unified interface for controlling a pan/tilt camera gimbal
with two 28BYJ-48 stepper motors (pan and tilt), each with limit switch detection.

GPIO Pin Layout (Orange Pi Zero 2W, gpiochip0):
  Pan Motor:
    - IN1 (output): GPIO23
    - IN2 (output): GPIO24
    - IN3 (output): GPIO25
    - IN4 (output): GPIO26
    - Limit Switch (input, active low): GPIO31
  
  Tilt Motor:
    - IN1 (output): GPIO27
    - IN2 (output): GPIO28
    - IN3 (output): GPIO29
    - IN4 (output): GPIO30
    - Limit Switch (input, active low): GPIO32
"""

import threading
from typing import Optional
from stepper_motor_controller import StepperMotor, MotorDirection


class PanTiltGimbal:
    """
    Unified pan/tilt gimbal control.
    
    Manages two stepper motors for pan (horizontal) and tilt (vertical) rotation.
    Provides synchronized movement, position tracking, and limit switch safety.
    Thread-safe via internal locks.
    """
    
    # GPIO pin assignments for Orange Pi Zero 2W
    PAN_CONTROL_PINS = (23, 24, 25, 26)      # IN1-IN4 for pan motor
    PAN_LIMIT_PIN = 31                         # Limit switch for pan end
    
    TILT_CONTROL_PINS = (27, 28, 29, 30)     # IN1-IN4 for tilt motor
    TILT_LIMIT_PIN = 32                        # Limit switch for tilt end
    
    def __init__(self, max_pan: float = 180.0, max_tilt: float = 180.0):
        """
        Initialize pan/tilt gimbal.
        
        Args:
            max_pan: Maximum pan angle in degrees (0-180)
            max_tilt: Maximum tilt angle in degrees (0-180)
        """
        self.max_pan = max_pan
        self.max_tilt = max_tilt
        
        # Create motor controllers
        self.pan_motor = StepperMotor(
            control_pins=self.PAN_CONTROL_PINS,
            limit_switch_pin=self.PAN_LIMIT_PIN,
            max_angle=max_pan,
            motor_name="Pan",
        )
        
        self.tilt_motor = StepperMotor(
            control_pins=self.TILT_CONTROL_PINS,
            limit_switch_pin=self.TILT_LIMIT_PIN,
            max_angle=max_tilt,
            motor_name="Tilt",
        )
        
        # State tracking
        self._lock = threading.Lock()
        self._is_calibrated = False
    
    def set_speed(self, pan_hz: float = 500, tilt_hz: float = 500) -> None:
        """
        Set stepping frequency for both motors.
        
        Args:
            pan_hz: Pan motor stepping frequency (500-2000 Hz recommended)
            tilt_hz: Tilt motor stepping frequency
        """
        with self._lock:
            self.pan_motor.set_speed(pan_hz)
            self.tilt_motor.set_speed(tilt_hz)
    
    def move_to_angles(self, pan_angle: float, tilt_angle: float) -> None:
        """
        Move gimbal to specified pan and tilt angles.
        
        Arms are moved in parallel and independently.
        
        Args:
            pan_angle: Target pan angle in degrees (0 to max_pan)
            tilt_angle: Target tilt angle in degrees (0 to max_tilt)
        """
        with self._lock:
            pan_angle = max(0, min(pan_angle, self.max_pan))
            tilt_angle = max(0, min(tilt_angle, self.max_tilt))
            
            self.pan_motor.move_to_angle(pan_angle)
            self.tilt_motor.move_to_angle(tilt_angle)
    
    def pan(self, angle: float) -> None:
        """
        Move pan motor to angle, leave tilt unchanged.
        
        Args:
            angle: Pan angle in degrees
        """
        with self._lock:
            self.pan_motor.move_to_angle(angle)
    
    def tilt(self, angle: float) -> None:
        """
        Move tilt motor to angle, leave pan unchanged.
        
        Args:
            angle: Tilt angle in degrees
        """
        with self._lock:
            self.tilt_motor.move_to_angle(angle)
    
    def pan_step(self, direction: int, steps: int = 1) -> None:
        """
        Step pan motor incrementally.
        
        Args:
            direction: 1 for clockwise, -1 for counterclockwise
            steps: Number of half-steps
        """
        with self._lock:
            motor_dir = MotorDirection.CLOCKWISE if direction > 0 else MotorDirection.COUNTERCLOCKWISE
            self.pan_motor.step(motor_dir, steps)
    
    def tilt_step(self, direction: int, steps: int = 1) -> None:
        """
        Step tilt motor incrementally.
        
        Args:
            direction: 1 for up, -1 for down
            steps: Number of half-steps
        """
        with self._lock:
            motor_dir = MotorDirection.CLOCKWISE if direction > 0 else MotorDirection.COUNTERCLOCKWISE
            self.tilt_motor.step(motor_dir, steps)
    
    def get_position(self) -> tuple:
        """
        Get current pan and tilt angles.
        
        Returns:
            (pan_angle, tilt_angle) in degrees
        """
        with self._lock:
            return (self.pan_motor.get_angle(), self.tilt_motor.get_angle())
    
    def is_moving(self) -> bool:
        """Check if either motor is currently moving."""
        with self._lock:
            return self.pan_motor.is_moving or self.tilt_motor.is_moving
    
    def home(self) -> None:
        """
        Calibrate gimbal by moving both motors to home position (0, 0).
        Uses limit switches for accurate calibration.
        
        Blocks until both motors are homed.
        """
        print("[Gimbal] Homing...")
        with self._lock:
            self._is_calibrated = False
        
        # Home pan first
        self.pan_motor.home()
        
        # Home tilt
        self.tilt_motor.home()
        
        with self._lock:
            self._is_calibrated = True
        
        print("[Gimbal] Homing complete: (0°, 0°)")
    
    def is_calibrated(self) -> bool:
        """Check if gimbal has been calibrated."""
        with self._lock:
            return self._is_calibrated
    
    def stop(self) -> None:
        """Stop both motors and hold current position."""
        with self._lock:
            self.pan_motor.stop()
            self.tilt_motor.stop()
    
    def cleanup(self) -> None:
        """Clean up GPIO resources and stop motors."""
        print("[Gimbal] Cleaning up...")
        with self._lock:
            self.pan_motor.cleanup()
            self.tilt_motor.cleanup()
        print("[Gimbal] Cleanup complete")
