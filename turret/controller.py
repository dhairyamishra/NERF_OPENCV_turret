"""
Turret hardware controller v3.0 with:
- Trapezoidal velocity motion profiles for smooth servo movement
- Serial read-back thread with auto-reconnect
- Command rate limiting to match servo update rate
- Burst fire mode with configurable count and delay
- Patrol pattern generator (sweep, figure-8, random)

Expected Arduino protocol:
    Send: "P<pan>T<tilt>F<fire>\n"
    Receive: "OK\n" or "ERR:<message>\n" or "HB:<pan>,<tilt>,<armed>\n"
"""

import logging
import math
import random
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logger.warning("pyserial not installed. Turret control will run in simulation mode.")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class TrapezoidalProfile:
    """Generates trapezoidal velocity motion from current to target position."""

    def __init__(self, max_velocity: float, max_acceleration: float):
        self.max_vel = max_velocity
        self.max_acc = max_acceleration
        self._velocity = 0.0

    def step(self, current: float, target: float, dt: float) -> float:
        error = target - current
        if abs(error) < 0.1:
            self._velocity = 0.0
            return target

        direction = 1.0 if error > 0 else -1.0
        stopping_dist = (self._velocity ** 2) / (2.0 * self.max_acc) if self.max_acc > 0 else 0

        if abs(error) <= stopping_dist + 0.5:
            self._velocity -= direction * self.max_acc * dt
        else:
            self._velocity += direction * self.max_acc * dt

        self._velocity = max(-self.max_vel, min(self.max_vel, self._velocity))

        new_pos = current + self._velocity * dt
        if (error > 0 and new_pos > target) or (error < 0 and new_pos < target):
            new_pos = target
            self._velocity = 0.0

        return new_pos

    def reset(self):
        self._velocity = 0.0


class PatrolGenerator:
    """Generates patrol waypoints for different patterns."""

    def __init__(self, config: dict):
        turret_cfg = config["turret"]
        self.pattern = turret_cfg.get("patrol_pattern", "sweep")
        self.speed = turret_cfg.get("patrol_speed", 2.0)
        self.pan_limits = tuple(turret_cfg.get("pan_limits", [0, 180]))
        self.tilt_limits = tuple(turret_cfg.get("tilt_limits", [30, 150]))
        self.pan_center = turret_cfg.get("pan_offset", 90)
        self.tilt_center = turret_cfg.get("tilt_offset", 90)
        custom = turret_cfg.get("patrol_waypoints", [])

        if custom:
            self._waypoints = [tuple(w) for w in custom]
        else:
            self._waypoints = self._generate_waypoints()

        self._index = 0
        self._direction = 1
        self._t = 0.0

    def _generate_waypoints(self) -> list[tuple[float, float]]:
        pan_min, pan_max = self.pan_limits
        tilt_mid = (self.tilt_limits[0] + self.tilt_limits[1]) / 2

        if self.pattern == "sweep":
            return [
                (pan_min + 10, tilt_mid),
                (pan_max - 10, tilt_mid),
            ]
        elif self.pattern == "figure8":
            points = []
            for i in range(24):
                angle = (i / 24) * 2 * math.pi
                pan = self.pan_center + 40 * math.sin(angle)
                tilt = self.tilt_center + 20 * math.sin(2 * angle)
                pan = max(pan_min, min(pan_max, pan))
                tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], tilt))
                points.append((pan, tilt))
            return points
        elif self.pattern == "random":
            return [
                (random.uniform(pan_min + 10, pan_max - 10),
                 random.uniform(self.tilt_limits[0] + 5, self.tilt_limits[1] - 5))
                for _ in range(8)
            ]
        return [(self.pan_center, self.tilt_center)]

    def next_position(self) -> tuple[float, float]:
        if not self._waypoints:
            return (self.pan_center, self.tilt_center)

        target = self._waypoints[self._index]
        return target

    def advance(self, current_pan: float, current_tilt: float) -> bool:
        """Check if close enough to current waypoint to advance. Returns True if advanced."""
        target = self._waypoints[self._index]
        dist = math.hypot(current_pan - target[0], current_tilt - target[1])
        if dist < 3.0:
            if self.pattern == "sweep":
                self._index += self._direction
                if self._index >= len(self._waypoints) or self._index < 0:
                    self._direction *= -1
                    self._index += self._direction * 2
                    self._index = max(0, min(len(self._waypoints) - 1, self._index))
            elif self.pattern == "random":
                self._waypoints = self._generate_waypoints()
                self._index = 0
            else:
                self._index = (self._index + 1) % len(self._waypoints)
            return True
        return False


class TurretController:
    """
    Controls pan/tilt servos and firing mechanism.
    Supports serial (Arduino) and direct GPIO (RPi) modes.
    """

    def __init__(self, config: dict):
        turret_cfg = config["turret"]
        self.enabled = turret_cfg.get("enabled", False)
        self.serial_port = turret_cfg.get("serial_port", "COM3")
        self.baud_rate = turret_cfg.get("baud_rate", 115200)
        self.pan_limits = tuple(turret_cfg.get("pan_limits", [0, 180]))
        self.tilt_limits = tuple(turret_cfg.get("tilt_limits", [30, 150]))
        self.pan_offset = turret_cfg.get("pan_offset", 90)
        self.tilt_offset = turret_cfg.get("tilt_offset", 90)
        self.fire_duration_ms = turret_cfg.get("fire_duration_ms", 200)
        self.smoothing = turret_cfg.get("smoothing", 0.3)
        self.motion_profile_type = turret_cfg.get("motion_profile", "trapezoidal")
        self.command_rate_hz = turret_cfg.get("command_rate_hz", 50)
        self.burst_count = turret_cfg.get("burst_count", 1)
        self.burst_delay_ms = turret_cfg.get("burst_delay_ms", 100)

        self._serial: Optional[serial.Serial] = None if SERIAL_AVAILABLE else None
        self._lock = threading.Lock()
        self._current_pan = float(self.pan_offset)
        self._current_tilt = float(self.tilt_offset)
        self._target_pan = float(self.pan_offset)
        self._target_tilt = float(self.tilt_offset)
        self._is_firing = False
        self._armed = False
        self._connected = False
        self._simulation = not self.enabled

        # Motion profiles
        max_vel = turret_cfg.get("max_velocity", 120.0)
        max_acc = turret_cfg.get("max_acceleration", 45.0)
        self._pan_profile = TrapezoidalProfile(max_vel, max_acc)
        self._tilt_profile = TrapezoidalProfile(max_vel, max_acc)
        self._last_update_time = time.perf_counter()

        # Rate limiting
        self._min_command_interval = 1.0 / max(1, self.command_rate_hz)
        self._last_send_time = 0.0

        # Serial read-back
        self._read_thread = None
        self._hw_status: dict = {}
        self._reconnect_attempts = 0
        self._max_reconnect = 5

        # Patrol
        self._patrol = PatrolGenerator(config)

    def connect(self) -> bool:
        if not self.enabled:
            logger.info("Turret running in SIMULATION mode")
            self._simulation = True
            self._connected = True
            return True

        if not SERIAL_AVAILABLE:
            logger.warning("pyserial not available, running in simulation mode")
            self._simulation = True
            self._connected = True
            return True

        return self._open_serial()

    def _open_serial(self) -> bool:
        try:
            self._serial = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=1.0,
            )
            time.sleep(2.0)
            self._connected = True
            self._simulation = False
            self._reconnect_attempts = 0
            logger.info(f"Turret connected on {self.serial_port} @ {self.baud_rate}")

            self._read_thread = threading.Thread(target=self._serial_reader, daemon=True)
            self._read_thread.start()

            self.move_to(self.pan_offset, self.tilt_offset)
            return True

        except Exception as e:
            logger.error(f"Failed to connect turret: {e}")
            self._simulation = True
            self._connected = True
            return False

    def _serial_reader(self):
        """Background thread: reads responses from Arduino."""
        while self._connected and self._serial and self._serial.is_open:
            try:
                line = self._serial.readline().decode("ascii", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("HB:"):
                    parts = line[3:].split(",")
                    if len(parts) >= 3:
                        self._hw_status = {
                            "hw_pan": float(parts[0]),
                            "hw_tilt": float(parts[1]),
                            "hw_armed": parts[2] == "1",
                        }
                elif line.startswith("ERR:"):
                    logger.warning(f"Turret error: {line}")
                elif line == "OK":
                    pass
                elif line == "NERF_TURRET_READY":
                    logger.info("Arduino reports ready")

            except serial.SerialException:
                logger.error("Serial read error, attempting reconnect...")
                self._attempt_reconnect()
                break
            except Exception as e:
                logger.debug(f"Serial reader: {e}")

    def _attempt_reconnect(self):
        if self._reconnect_attempts >= self._max_reconnect:
            logger.error("Max reconnect attempts reached, switching to simulation")
            self._simulation = True
            return

        self._reconnect_attempts += 1
        logger.info(f"Reconnect attempt {self._reconnect_attempts}/{self._max_reconnect}")
        try:
            if self._serial:
                self._serial.close()
        except Exception:
            pass
        time.sleep(1.0)
        self._open_serial()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_simulation(self) -> bool:
        return self._simulation

    @property
    def is_armed(self) -> bool:
        return self._armed

    @property
    def current_position(self) -> tuple[float, float]:
        return (self._current_pan, self._current_tilt)

    @property
    def target_position(self) -> tuple[float, float]:
        return (self._target_pan, self._target_tilt)

    def arm(self):
        self._armed = True
        logger.warning("TURRET ARMED")

    def disarm(self):
        self._armed = False
        self._is_firing = False
        logger.info("Turret disarmed")

    def move_to(self, pan: float, tilt: float):
        self._target_pan = self._clamp(pan, *self.pan_limits)
        self._target_tilt = self._clamp(tilt, *self.tilt_limits)

    def move_relative(self, delta_pan: float, delta_tilt: float):
        self.move_to(
            self._current_pan + delta_pan,
            self._current_tilt + delta_tilt,
        )

    def center(self):
        self.move_to(self.pan_offset, self.tilt_offset)
        self._pan_profile.reset()
        self._tilt_profile.reset()

    def update(self):
        """Called each frame to apply motion profile and send to hardware."""
        now = time.perf_counter()
        dt = now - self._last_update_time
        self._last_update_time = now
        dt = max(0.001, min(0.1, dt))

        if self.motion_profile_type == "trapezoidal":
            self._current_pan = self._pan_profile.step(self._current_pan, self._target_pan, dt)
            self._current_tilt = self._tilt_profile.step(self._current_tilt, self._target_tilt, dt)
        else:
            alpha = self.smoothing
            self._current_pan += alpha * (self._target_pan - self._current_pan)
            self._current_tilt += alpha * (self._target_tilt - self._current_tilt)

        # Rate limit serial commands
        if now - self._last_send_time >= self._min_command_interval:
            self._send_position(self._current_pan, self._current_tilt, self._is_firing)
            self._last_send_time = now

    def fire(self):
        if not self._armed:
            logger.warning("Fire command rejected: turret not armed")
            return

        def _burst_fire():
            for i in range(self.burst_count):
                self._is_firing = True
                time.sleep(self.fire_duration_ms / 1000.0)
                self._is_firing = False
                if i < self.burst_count - 1:
                    time.sleep(self.burst_delay_ms / 1000.0)

        logger.info(f"FIRE! (burst={self.burst_count})")
        threading.Thread(target=_burst_fire, daemon=True).start()

    def get_patrol_position(self) -> tuple[float, float]:
        """Get next patrol target and advance waypoint if reached."""
        self._patrol.advance(self._current_pan, self._current_tilt)
        return self._patrol.next_position()

    def _send_position(self, pan: float, tilt: float, fire: bool):
        pan_int = int(round(pan))
        tilt_int = int(round(tilt))
        fire_int = 1 if fire else 0
        command = f"P{pan_int:03d}T{tilt_int:03d}F{fire_int}\n"

        if self._simulation:
            return

        with self._lock:
            try:
                if self._serial and self._serial.is_open:
                    self._serial.write(command.encode("ascii"))
                    self._serial.flush()
            except Exception as e:
                logger.error(f"Serial write error: {e}")

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))

    def disconnect(self):
        if self._serial and self._serial.is_open:
            self.disarm()
            self.center()
            self.update()
            time.sleep(0.1)
            self._serial.close()
        self._connected = False
        logger.info("Turret disconnected")

    @property
    def status(self) -> dict:
        base = {
            "connected": self._connected,
            "simulation": self._simulation,
            "armed": self._armed,
            "firing": self._is_firing,
            "pan": round(self._current_pan, 1),
            "tilt": round(self._current_tilt, 1),
            "target_pan": round(self._target_pan, 1),
            "target_tilt": round(self._target_tilt, 1),
            "motion_profile": self.motion_profile_type,
            "burst_count": self.burst_count,
        }
        base.update(self._hw_status)
        return base
