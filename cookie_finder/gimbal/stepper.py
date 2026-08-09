"""
Low-level stepper motor control for 28BYJ-48 + ULN2003 driver via gpiod (libgpiod).

GPIO Pin Assignments (Orange Pi Zero 2W, gpiochip1 offsets):
  Pan Motor (IN1-IN4):     271, 268, 258, 272 (PI15, PI12, PI02, PI16)
  Tilt Motor (IN1-IN4):    262, 229, 233, 265 (RXD.2, CE.0, CE.1, SCL.2)
  Pan Limit (min/max):     257 / 227 (physical 12 / 13) — Center-Off SPDT
  Tilt Limit (min/max):    261 / 270 (physical 15 / 16) — Center-Off SPDT
  UART0 reserved:          physical 8 / 10 (TXD.0 / RXD.0)

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
    import gpiod
    from gpiod.line import Bias, Direction, Value
    GPIO_AVAILABLE = True
except ImportError as e:
    GPIO_AVAILABLE = False
    gpiod = None
    Bias = Direction = Value = None  # type: ignore[misc, assignment]
    print(f"[WARNING] Failed to import gpiod: {e}")
    print("[WARNING] GPIO control will not be available. Install with: uv pip install gpiod")


class MotorDirection(Enum):
    """Stepper motor direction."""
    CLOCKWISE = 1
    COUNTERCLOCKWISE = -1


class StepperMotor:
    """
    Control a 28BYJ-48 stepper motor via ULN2003 driver using gpiod.

    Operates in a background thread to allow non-blocking stepping.
    Monitors a Center-Off SPDT limit switch (min + max channels, active-low).
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
        limit_min_pin: Optional[int] = None,
        limit_max_pin: Optional[int] = None,
        max_angle: float = 180.0,
        motor_name: str = "Motor",
        phase_order: Optional[tuple[int, int, int, int]] = None,
    ):
        """
        Initialize stepper motor controller.

        Args:
            control_pins: GPIO line offsets (gpiochip1) for IN1, IN2, IN3, IN4
            limit_min_pin: GPIO offset for SPDT min/CCW/home channel, or None
            limit_max_pin: GPIO offset for SPDT max/CW channel, or None
            max_angle: Maximum rotation angle (degrees)
            motor_name: Human-readable motor name
            phase_order: Maps IN1..IN4 outputs to logical phases 0..3
        """
        self.motor_name = motor_name
        self.control_pins = control_pins
        self.phase_order = phase_order or (0, 1, 2, 3)
        self.limit_min_pin = limit_min_pin
        self.limit_max_pin = limit_max_pin
        self.max_angle = max_angle

        # Motor state
        self.current_angle = 0.0  # degrees
        self.current_step = 0  # position in FULL_STEP_SEQUENCE
        self.is_moving = False
        self.target_angle: Optional[float] = None
        self.speed_hz = 500  # stepping frequency in Hz

        # Threading
        self._stop_event = threading.Event()
        self._step_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # GPIO resources
        self.chip = None
        self.lines = None  # Control pins (output)
        self.limit_line = None  # Limit switch pins (input)

        if GPIO_AVAILABLE:
            self._init_gpio()

    def _init_gpio(self) -> None:
        """Initialize GPIO lines using gpiod with bulk request."""
        try:
            self.chip = gpiod.Chip("/dev/gpiochip1")

            # Configure output settings for control pins
            output_config = {
                offset: gpiod.LineSettings(direction=Direction.OUTPUT)
                for offset in self.control_pins
            }

            # Request all control lines at once
            self.lines = self.chip.request_lines(
                consumer=self.motor_name,
                config=output_config
            )

            print(f"[{self.motor_name}] GPIO initialized: control pins {self.control_pins}")

            # Center-Off SPDT: COM→GND, channels→GPIO with pull-up (active-low)
            limit_pins = [
                p for p in (self.limit_min_pin, self.limit_max_pin) if p is not None
            ]
            if limit_pins:
                try:
                    limit_config = {
                        offset: gpiod.LineSettings(
                            direction=Direction.INPUT,
                            bias=Bias.PULL_UP,
                        )
                        for offset in limit_pins
                    }
                    self.limit_line = self.chip.request_lines(
                        consumer=f"{self.motor_name}_limit",
                        config=limit_config,
                    )
                    print(
                        f"[{self.motor_name}] Limit switches initialized "
                        f"min={self.limit_min_pin} max={self.limit_max_pin}"
                    )
                except Exception as e:
                    print(f"[{self.motor_name}] Limit switch initialization failed: {e}")
                    self.limit_line = None
        except Exception as e:
            print(f"[{self.motor_name}] GPIO initialization failed: {e}")
            print(f"[{self.motor_name}] GPIO access requires root. Try: sudo make test-motors-pan-ccw")

    def _set_step(self, step_index: int) -> None:
        """Set motor pins to a specific step in the sequence."""
        if not GPIO_AVAILABLE or self.lines is None:
            return

        step_values = self.FULL_STEP_SEQUENCE[step_index % len(self.FULL_STEP_SEQUENCE)]
        try:
            values = {
                offset: Value.ACTIVE
                if step_values[self.phase_order[i]]
                else Value.INACTIVE
                for i, offset in enumerate(self.control_pins)
            }
            self.lines.set_values(values)
        except Exception as e:
            print(f"[{self.motor_name}] Failed to set step {step_index}: {e}")

    def limit_state(self) -> tuple[bool, bool]:
        """Return (min_hit, max_hit). Active-low; False if not configured."""
        if not GPIO_AVAILABLE or self.limit_line is None:
            return (False, False)

        try:
            pins = []
            if self.limit_min_pin is not None:
                pins.append(self.limit_min_pin)
            if self.limit_max_pin is not None:
                pins.append(self.limit_max_pin)
            if not pins:
                return (False, False)

            values = self.limit_line.get_values(pins)
            # gpiod returns a list ordered like the offsets we pass
            hit = {
                pin: values[i] == Value.INACTIVE
                for i, pin in enumerate(pins)
            }
            min_hit = hit.get(self.limit_min_pin, False) if self.limit_min_pin is not None else False
            max_hit = hit.get(self.limit_max_pin, False) if self.limit_max_pin is not None else False
            return (min_hit, max_hit)
        except Exception as e:
            print(f"[{self.motor_name}] Failed to read limit switches: {e}")
            return (False, False)

    def _direction_blocked(self, direction: int) -> bool:
        """True if the given step direction is blocked by a tripped limit."""
        min_hit, max_hit = self.limit_state()
        return (direction < 0 and min_hit) or (direction > 0 and max_hit)

    def _stepping_loop(self) -> None:
        """Background thread: perform stepping based on target angle."""
        while not self._stop_event.is_set():
            with self._lock:
                if self.target_angle is None or self.is_moving is False:
                    time.sleep(0.01)  # idle
                    continue

                # Calculate direction to target
                angle_diff = self.target_angle - self.current_angle

                if abs(angle_diff) < 0.5:  # Close enough to target
                    self.is_moving = False
                    self.target_angle = None
                    self._set_step(self.current_step)  # Energize to hold position
                    continue

                direction = 1 if angle_diff > 0 else -1

                # Direction-aware hard-stop: block only motion into a tripped limit
                if self._direction_blocked(direction):
                    print(
                        f"[{self.motor_name}] Limit blocked dir={direction} "
                        f"(min/max={self.limit_state()})"
                    )
                    self.is_moving = False
                    self.target_angle = None
                    self._set_step(self.current_step)
                    continue

                # Step towards target
                self.current_step += direction
                self.current_step %= len(self.FULL_STEP_SEQUENCE)

                degrees_per_full_step = 360.0 / self.STEPS_PER_REVOLUTION
                self.current_angle += direction * degrees_per_full_step
                self.current_angle = max(0, min(self.current_angle, self.max_angle))

                self._set_step(self.current_step)

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
        Move motor to home (min limit) by stepping CCW until min trips or timeout.
        """
        print(f"[{self.motor_name}] Homing toward min limit...")
        with self._lock:
            self.target_angle = None
            self.is_moving = False

        if self.limit_line is None or self.limit_min_pin is None:
            print(f"[{self.motor_name}] Homing skipped (no limit switch); zeroing angle")
            with self._lock:
                self.current_angle = 0.0
                self.current_step = 0
            return

        saved_hz = self.speed_hz
        self.speed_hz = 200  # Slow speed for homing
        home_timeout = time.time() + 10.0

        while time.time() < home_timeout:
            min_hit, _ = self.limit_state()
            if min_hit:
                with self._lock:
                    self.current_angle = 0.0
                    self.current_step = 0
                    self._set_step(self.current_step)
                self.speed_hz = saved_hz
                print(f"[{self.motor_name}] Homed successfully at 0°")
                return

            # Step backward (CCW) unless already blocked
            if self._direction_blocked(-1):
                with self._lock:
                    self.current_angle = 0.0
                    self.current_step = 0
                    self._set_step(self.current_step)
                self.speed_hz = saved_hz
                print(f"[{self.motor_name}] Homed successfully at 0°")
                return

            self.current_step = (self.current_step - 1) % len(self.FULL_STEP_SEQUENCE)
            self._set_step(self.current_step)
            time.sleep(1.0 / self.speed_hz)

        print(f"[{self.motor_name}] Homing failed: min limit not triggered within timeout")
        with self._lock:
            self.current_angle = 0.0
        self.speed_hz = saved_hz

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
                if self._direction_blocked(direction.value):
                    print(
                        f"[{self.motor_name}] Limit blocked dir={direction.value} "
                        f"(min/max={self.limit_state()})"
                    )
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

        # De-energize motor (set all control pins to INACTIVE) if GPIO is available
        if GPIO_AVAILABLE and self.lines is not None:
            try:
                values = {offset: Value.INACTIVE for offset in self.control_pins}
                self.lines.set_values(values)
                self.lines.release()
            except Exception as e:
                print(f"[{self.motor_name}] Failed to de-energize: {e}")

        if self.limit_line is not None:
            try:
                self.limit_line.release()
            except Exception as e:
                print(f"[{self.motor_name}] Failed to release limit switches: {e}")

        self.lines = None
        self.limit_line = None
        self.chip = None
        print(f"[{self.motor_name}] Cleaned up")
