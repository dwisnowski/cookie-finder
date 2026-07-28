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

import logging
import threading
from typing import Optional
from cookie_finder.gimbal.config import load_gimbal_config
from cookie_finder.gimbal.stepper import StepperMotor, MotorDirection

logger = logging.getLogger(__name__)

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


class PanTiltGimbal:
    """
    Unified pan/tilt gimbal control.
    
    Manages two stepper motors for pan (horizontal) and tilt (vertical) rotation.
    Provides synchronized movement, position tracking, and limit switch safety.
    Thread-safe via internal locks.
    """
    
    # GPIO pin assignments for Orange Pi Zero 2W (gpiochip1 offsets)
    PAN_CONTROL_PINS = (271, 268, 258, 272)   # IN1-IN4 for pan motor (PI15, PI12, PI02, PI16)
    PAN_LIMIT_PIN = 264                         # Limit switch for pan end (TBD)
    
    TILT_CONTROL_PINS = (262, 229, 233, 265)  # IN1-IN4 for tilt motor (PH0, PH1, PI01, PI14)
    TILT_LIMIT_PIN = 263                        # Limit switch for tilt end (TBD)
    
    def __init__(self, max_pan: float = 180.0, max_tilt: float = 180.0):
        """
        Initialize pan/tilt gimbal.
        
        Args:
            max_pan: Maximum pan angle in degrees (0-180)
            max_tilt: Maximum tilt angle in degrees (0-180)
        """
        self.max_pan = max_pan
        self.max_tilt = max_tilt
        logger.info("PanTiltGimbal.__init__(max_pan=%.1f, max_tilt=%.1f)", max_pan, max_tilt)

        gimbal_config = load_gimbal_config()
        pan_phase = tuple(gimbal_config["pan_phase_order"])
        tilt_phase = tuple(gimbal_config["tilt_phase_order"])

        # Create motor controllers
        self.pan_motor = StepperMotor(
            control_pins=self.PAN_CONTROL_PINS,
            # limit_switch_pin=self.PAN_LIMIT_PIN,
            max_angle=max_pan,
            motor_name="Pan",
            phase_order=pan_phase,
        )
        
        self.tilt_motor = StepperMotor(
            control_pins=self.TILT_CONTROL_PINS,
            # limit_switch_pin=self.TILT_LIMIT_PIN,
            max_angle=max_tilt,
            motor_name="Tilt",
            phase_order=tilt_phase,
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
        logger.info("set_speed(pan_hz=%.1f, tilt_hz=%.1f)", pan_hz, tilt_hz)
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
        logger.info("move_to_angles(pan=%.1f, tilt=%.1f)", pan_angle, tilt_angle)
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
        logger.info("pan(angle=%.1f)", angle)
        with self._lock:
            self.pan_motor.move_to_angle(angle)

    def tilt(self, angle: float) -> None:
        """
        Move tilt motor to angle, leave pan unchanged.
        
        Args:
            angle: Tilt angle in degrees
        """
        logger.info("tilt(angle=%.1f)", angle)
        with self._lock:
            self.tilt_motor.move_to_angle(angle)

    def pan_step(self, direction: int, steps: int = 1) -> None:
        """
        Step pan motor incrementally.
        
        Args:
            direction: 1 for clockwise, -1 for counterclockwise
            steps: Number of half-steps
        """
        logger.info("pan_step(direction=%d, steps=%d)", direction, steps)
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
        logger.info("tilt_step(direction=%d, steps=%d)", direction, steps)
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
            position = (self.pan_motor.get_angle(), self.tilt_motor.get_angle())
        logger.debug("get_position() -> (%.1f, %.1f)", position[0], position[1])
        return position

    def is_moving(self) -> bool:
        """Check if either motor is currently moving."""
        with self._lock:
            moving = self.pan_motor.is_moving or self.tilt_motor.is_moving
        logger.debug("is_moving() -> %s", moving)
        return moving
    
    def home(self) -> None:
        """
        Calibrate gimbal by moving both motors to home position (0, 0).
        Uses limit switches for accurate calibration.
        
        Blocks until both motors are homed.
        """
        logger.info("home() starting")
        with self._lock:
            self._is_calibrated = False
        
        # Home pan first
        self.pan_motor.home()
        
        # Home tilt
        self.tilt_motor.home()
        
        with self._lock:
            self._is_calibrated = True
        
        logger.info("home() complete (0°, 0°)")

    def is_calibrated(self) -> bool:
        """Check if gimbal has been calibrated."""
        with self._lock:
            calibrated = self._is_calibrated
        logger.debug("is_calibrated() -> %s", calibrated)
        return calibrated

    def stop(self) -> None:
        """Stop both motors and hold current position."""
        logger.info("stop()")
        with self._lock:
            self.pan_motor.stop()
            self.tilt_motor.stop()
    
    def cleanup(self) -> None:
        """Clean up GPIO resources and stop motors."""
        logger.info("cleanup()")
        with self._lock:
            self.pan_motor.cleanup()
            self.tilt_motor.cleanup()
        logger.info("cleanup() complete")
