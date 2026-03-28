"""
Low-level stepper motor control for 28BYJ-48 + ULN2003 driver via OPi.GPIO.

GPIO Pin Assignments (Orange Pi Zero 2W, BOARD mode):
  Pan Motor (IN1-IN4):     31, 33, 35, 37 (physical pins: PI15, PI12, PI02, PI16)
  Tilt Motor (IN1-IN4):    (to be configured)
  Pan Limit Switch:        (to be configured)
  Tilt Limit Switch:       (to be configured)

Driver: ULN2003 with full-step sequence (4 steps per cycle).
Confirmed working with logic analyzer.

Speed Notes:
  - 28BYJ-48 is a geared motor (internal gearing ≈ 4076 steps/rev)
  - Step frequency = RPM × 4076 / 60
  - At 12V, typical max is ~10 RPM
  - At 5V, typical max is ~5 RPM
  - Recommended stepping frequency: 500-2000 Hz for smooth operation
"""

import threading
import time
from enum import Enum
from typing import Optional

try:
    import OPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError as e:
    GPIO_AVAILABLE = False
    GPIO = None
    print(f"[WARNING] Failed to import OPi.GPIO: {e}")
    print("[WARNING] GPIO control will not be available. Install with: uv pip install OPi.GPIO")


class MotorDirection(Enum):
    """Stepper motor direction."""
    CLOCKWISE = 1
    COUNTERCLOCKWISE = -1


class StepperMotor:
    """
    Control a 28BYJ-48 stepper motor via ULN2003 driver using gpiod.
    
    Operates in a background thread to allow non-blocking stepping.
    Monitors a limit switch GPIO for end-of-range detection.
    """
    
    # 28BYJ-48 full-step sequence (4 steps per full cycle)
    # Each tuple is (IN1, IN2, IN3, IN4) logic levels
    # Confirmed working with logic analyzer on Orange Pi Zero 2W
    FULL_STEP_SEQUENCE = [
        (1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, 0, 1),
    ]
    
    # Steps per revolution for 28BYJ-48 (with gearing)
    STEPS_PER_REVOLUTION = 4076
    
    def __init__(
        self,
        control_pins: tuple[int, int, int, int],
        limit_switch_pin: Optional[int] = None,
        max_angle: float = 180.0,
        motor_name: str = "Motor",
    ):
        """
        Initialize stepper motor controller.
        
        Args:
            control_pins: GPIO pin numbers in BOARD mode (physical pins)
            limit_switch_pin: GPIO pin number for limit switch input (BOARD mode), or None if not available
            max_angle: Maximum rotation angle (degrees)
            motor_name: Human-readable motor name
        """
        self.motor_name = motor_name
        self.control_pins = control_pins
        self.limit_switch_pin = limit_switch_pin
        self.max_angle = max_angle
        
        # Motor state
        self.current_angle = 0.0  # degrees
        self.current_step = 0  # position in FULL_STEP_SEQUENCE
        self.is_moving = False
        self.target_angle: Optional[float] = None
        self.speed_hz = 500  # stepping frequency in Hz
        
        # Limit switch state
        self.limit_triggered = False
        
        # Threading
        self._stop_event = threading.Event()
        self._step_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        if GPIO_AVAILABLE:
            self._init_gpio()
    
    def _init_gpio(self) -> None:
        """Initialize GPIO pins using OPi.GPIO."""
        try:
            # Set GPIO mode to BOARD (physical pin numbers on Orange Pi Zero 2W)
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BOARD)
                print(f"[{self.motor_name}] GPIO mode set to BOARD")
            
            # Setup control pins as outputs
            for pin in self.control_pins:
                try:
                    GPIO.setup(pin, GPIO.OUT)
                    GPIO.output(pin, 0)  # Initialize to LOW
                except Exception as pin_err:
                    print(f"[{self.motor_name}] Failed to setup pin {pin}: {pin_err}")
                    raise
            
            # Setup limit switch pin as input if provided
            if self.limit_switch_pin is not None:
                GPIO.setup(self.limit_switch_pin, GPIO.IN)
                print(f"[{self.motor_name}] GPIO initialized: control pins {self.control_pins}, limit pin {self.limit_switch_pin}")
            else:
                print(f"[{self.motor_name}] GPIO initialized: control pins {self.control_pins}")
        except Exception as e:
            print(f"[{self.motor_name}] GPIO initialization failed: {e}")
            print(f"[{self.motor_name}] GPIO access requires root. Try: sudo make test-motors-pan-ccw")
    
    def _set_step(self, step_index: int) -> None:
        """Set motor pins to a specific step in the sequence."""
        if not GPIO_AVAILABLE:
            return
        
        step_values = self.FULL_STEP_SEQUENCE[step_index % len(self.FULL_STEP_SEQUENCE)]
        try:
            for i, pin in enumerate(self.control_pins):
                GPIO.output(pin, 1 if step_values[i] else 0)
        except Exception as e:
            print(f"[{self.motor_name}] Failed to set step {step_index}: {e}")
    
    def _check_limit_switch(self) -> bool:
        """Check if limit switch is triggered (active low). Returns False if not configured or not available."""
        if self.limit_switch_pin is None or not GPIO_AVAILABLE:
            return False
        
        try:
            # Limit switch is active low (triggered = 0 when pressed)
            triggered = GPIO.input(self.limit_switch_pin) == 0
            if triggered and not self.limit_triggered:
                print(f"[{self.motor_name}] Limit switch triggered!")
                self.limit_triggered = True
            elif not triggered and self.limit_triggered:
                self.limit_triggered = False
            return triggered
        except Exception as e:
            print(f"[{self.motor_name}] Failed to read limit switch: {e}")
            return False
    
    def _stepping_loop(self) -> None:
        """Background thread: perform stepping based on target angle."""
        while not self._stop_event.is_set():
            with self._lock:
                if self.target_angle is None or self.is_moving is False:
                    time.sleep(0.01)  # idle
                    continue
                
                # Check limit switch
                if self._check_limit_switch():
                    # Stop immediately on limit switch
                    self.stop()
                    continue
                
                # Calculate direction to target
                angle_diff = self.target_angle - self.current_angle
                
                if abs(angle_diff) < 0.5:  # Close enough to target
                    self.is_moving = False
                    self.target_angle = None
                    self._set_step(self.current_step)  # Energize to hold position
                    continue
                
                # Step towards target
                direction = 1 if angle_diff > 0 else -1
                self.current_step += direction
                self.current_step %= len(self.FULL_STEP_SEQUENCE)
                
                # Update angle (each full-step is 2x half-step increment)
                # Half-step angle: 360 / (2 * 4076) degrees
                # Full-step angle: 2 * half-step angle
                degrees_per_full_step = 360.0 / self.STEPS_PER_REVOLUTION
                self.current_angle += direction * degrees_per_full_step
                self.current_angle = max(0, min(self.current_angle, self.max_angle))
                
                self._set_step(self.current_step)
                
                # Step delay based on speed_hz
                step_delay = 1.0 / self.speed_hz
                time.sleep(step_delay)
    
    def set_speed(self, speed_hz: float) -> None:
        """
        Set stepping frequency in Hz.
        
        Args:
            speed_hz: Stepping frequency (500-2000 Hz recommended)
        """
        with self._lock:
            self.speed_hz = max(10, min(speed_hz, 5000))  # Clamp to sane range
    
    def move_to_angle(self, angle: float) -> None:
        """
        Move motor to specified angle and step in background.
        
        Args:
            angle: Target angle in degrees (0 to max_angle)
        """
        with self._lock:
            angle = max(0, min(angle, self.max_angle))
            self.target_angle = angle
            self.is_moving = True
            
            # Start stepping thread if not running
            if self._step_thread is None or not self._step_thread.is_alive():
                self._stop_event.clear()
                self._step_thread = threading.Thread(
                    target=self._stepping_loop,
                    daemon=True,
                    name=f"{self.motor_name}_stepper",
                )
                self._step_thread.start()
    
    def home(self) -> None:
        """
        Move motor to home position using limit switch.
        Blocks until limit switch is triggered or timeout.
        """
        print(f"[{self.motor_name}] Homing...")
        with self._lock:
            self.target_angle = None
            self.is_moving = False
        
        # Slowly step backward until limit switch hits
        self._set_step(0)  # Start at step 0
        self.speed_hz = 200  # Slow speed for homing
        home_timeout = time.time() + 10.0  # 10 second timeout
        
        while time.time() < home_timeout:
            if self._check_limit_switch():
                with self._lock:
                    self.current_angle = 0.0
                    self.current_step = 0
                    self._set_step(self.current_step)
                print(f"[{self.motor_name}] Homed successfully at 0°")
                return
            
            # Step backward (CCW)
            self.current_step = (self.current_step - 1) % len(self.FULL_STEP_SEQUENCE)
            self._set_step(self.current_step)
            time.sleep(1.0 / self.speed_hz)
        
        print(f"[{self.motor_name}] Homing failed: limit switch not triggered within timeout")
        with self._lock:
            self.current_angle = 0.0
    
    def step(self, direction: MotorDirection, steps: int = 1) -> None:
        """
        Step motor by fixed number of steps.
        
        Args:
            direction: MotorDirection.CLOCKWISE or COUNTERCLOCKWISE
            steps: Number of half-steps
        """
        if not GPIO_AVAILABLE:
            print(f"[{self.motor_name}] GPIO not available. Cannot step motor.")
            return
        
        with self._lock:
            for _ in range(steps):
                if self._check_limit_switch():
                    break
                
                self.current_step += direction.value
                self.current_step %= len(self.FULL_STEP_SEQUENCE)
                
                self.current_angle += direction.value * 360.0 / self.STEPS_PER_REVOLUTION
                self.current_angle = max(0, min(self.current_angle, self.max_angle))
                
                self._set_step(self.current_step)
                time.sleep(1.0 / self.speed_hz)
    
    def stop(self) -> None:
        """Stop motor movement and energize coils to hold position."""
        with self._lock:
            self.is_moving = False
            self.target_angle = None
            self._set_step(self.current_step)
    
    def get_angle(self) -> float:
        """Get current motor angle in degrees."""
        with self._lock:
            return self.current_angle
    
    def cleanup(self) -> None:
        """Clean up GPIO resources and stop threads."""
        self._stop_event.set()
        if self._step_thread and self._step_thread.is_alive():
            self._step_thread.join(timeout=1.0)
        
        # De-energize motor (set all control pins to LOW) if GPIO is available
        if GPIO_AVAILABLE:
            try:
                for pin in self.control_pins:
                    GPIO.output(pin, 0)
            except Exception as e:
                print(f"[{self.motor_name}] Failed to de-energize: {e}")
            
            # Clean up GPIO
            try:
                GPIO.cleanup()
            except Exception as e:
                print(f"[{self.motor_name}] Failed to cleanup GPIO: {e}")
        
        print(f"[{self.motor_name}] Cleaned up")
