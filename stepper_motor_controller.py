"""
Low-level stepper motor control for 28BYJ-48 + ULN2003 driver via gpiod (libgpiod).

GPIO Pin Assignments (Orange Pi Zero 2W, gpiochip1):
  Pan Motor (IN1-IN4):     258, 268, 271, 272 (GPIO offsets)  pins (PI15, PI12, PI2, PI16)
  Tilt Motor (IN1-IN4):    273, 274, 275, 276 (GPIO offsets, estimated)
  Pan Limit Switch:        277 (GPIO offset, estimated)
  Tilt Limit Switch:       279 (GPIO offset, estimated)

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
from typing import Optional, List

try:
    import gpiod
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    gpiod = None


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
        limit_switch_pin: int,
        max_angle: float = 180.0,
        motor_name: str = "Motor",
    ):
        """
        Initialize stepper motor controller.
        
        Args:
            control_pins: GPIO pin numbers (IN1, IN2, IN3, IN4) for Orange Pi
            limit_switch_pin: GPIO pin number for limit switch input
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
        
        # GPIO setup using gpiod
        self.control_lines: List[Optional[gpiod.Line]] = []
        self.limit_line: Optional[gpiod.Line] = None
        self.chip: Optional[gpiod.Chip] = None
        
        if GPIO_AVAILABLE:
            self._init_gpio()
    
    def _init_gpio(self) -> None:
        """Initialize GPIO lines using gpiod on Linux."""
        try:
            # Use gpiochip1 for Orange Pi Zero 2W (H618 SoC)
            chip_path = "gpiochip1"
            try:
                self.chip = gpiod.Chip(chip_path)
            except Exception:
                # Fallback: try gpiochip0, then scan
                try:
                    self.chip = gpiod.Chip("gpiochip0")
                    chip_path = "gpiochip0"
                except Exception:
                    for i in range(5):
                        try:
                            self.chip = gpiod.Chip(f"gpiochip{i}")
                            chip_path = f"gpiochip{i}"
                            break
                        except Exception:
                            continue
            
            if self.chip is None:
                print(f"[{self.motor_name}] No GPIO chip found")
                return
            
            # Request control lines as outputs with bulk request
            control_line_offsets = list(self.control_pins)
            limit_line_offset = self.limit_switch_pin
            
            try:
                # Create settings for control (output) lines
                output_settings = gpiod.LineSettings(direction=gpiod.Line.Direction.OUTPUT)
                
                # Request all control lines
                control_request = self.chip.request_lines(
                    offsets=control_line_offsets,
                    config={self.chip.LINE_SEQ_ALL: output_settings}
                )
                self.control_lines = [control_request.get_line(offset) for offset in control_line_offsets]
                
                # Create settings for limit switch line (input with pull-up)
                input_settings = gpiod.LineSettings(
                    direction=gpiod.Line.Direction.INPUT,
                    bias=gpiod.Line.Bias.PULL_UP
                )
                
                # Request limit line
                limit_request = self.chip.request_lines(
                    offsets=[limit_line_offset],
                    config={self.chip.LINE_SEQ_ALL: input_settings}
                )
                self.limit_line = limit_request.get_line(limit_line_offset)
                
                print(f"[{self.motor_name}] GPIO initialized on {chip_path}: control pins {self.control_pins}, limit pin {self.limit_switch_pin}")
            except Exception as e:
                print(f"[{self.motor_name}] GPIO line request failed: {e}. Falling back to simple line access.")
                # Fallback: try to get lines individually
                try:
                    for pin_num in control_line_offsets:
                        line = self.chip.get_line(pin_num)
                        line.request(consumer=f"{self.motor_name}_ctrl", type=gpiod.LINE_REQUEST_DIRECTION_OUTPUT)
                        self.control_lines.append(line)
                    
                    limit_line = self.chip.get_line(limit_line_offset)
                    limit_line.request(consumer=f"{self.motor_name}_limit", type=gpiod.LINE_REQUEST_DIRECTION_INPUT)
                    self.limit_line = limit_line
                    
                    print(f"[{self.motor_name}] GPIO initialized (fallback method) on {chip_path}")
                except Exception as e2:
                    print(f"[{self.motor_name}] GPIO line access failed: {e2}")
                    self.chip = None
                    self.control_lines = []
                    self.limit_line = None
        except Exception as e:
            print(f"[{self.motor_name}] GPIO initialization failed: {e}")
            self.chip = None
            self.control_lines = []
            self.limit_line = None
    
    def _set_step(self, step_index: int) -> None:
        """Set motor pins to a specific step in the sequence."""
        if len(self.control_lines) < 4:
            return
        
        step_values = self.FULL_STEP_SEQUENCE[step_index % len(self.FULL_STEP_SEQUENCE)]
        try:
            for i, line in enumerate(self.control_lines):
                if line is not None:
                    # gpiod uses set_value with 0 or 1
                    line.set_value(1 if step_values[i] else 0)
        except Exception as e:
            print(f"[{self.motor_name}] Failed to set step {step_index}: {e}")
    
    def _check_limit_switch(self) -> bool:
        """Check if limit switch is triggered (active low)."""
        if self.limit_line is None:
            return False
        
        try:
            # Limit switch is active low (triggered = 0 when pressed)
            # gpiod get_value returns 0 or 1
            triggered = self.limit_line.get_value() == 0
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
        
        # De-energize motor (set all control pins to LOW)
        try:
            for line in self.control_lines:
                if line is not None:
                    line.set_value(0)
        except Exception as e:
            print(f"[{self.motor_name}] Failed to de-energize: {e}")
        
        # Close GPIO resources
        self.control_lines = []
        self.limit_line = None
        
        if self.chip is not None:
            try:
                # The chip will be garbage collected and properly closed
                self.chip.close()
            except Exception as e:
                print(f"[{self.motor_name}] Failed to close GPIO chip: {e}")
        
        self.chip = None
        print(f"[{self.motor_name}] Cleaned up")
