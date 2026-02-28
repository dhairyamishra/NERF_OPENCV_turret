"""
Targeting system v3.0 - converts tracked target positions to turret servo angles.
Features:
- Configurable FOV from config.yaml
- Adaptive PID gain scheduling (aggressive far, gentle close)
- Dead zone to prevent servo jitter near center
- Ballistic lead compensation based on target speed/distance
- Target priority scoring for automatic selection
"""

import logging
import math
import time
from typing import Optional

logger = logging.getLogger(__name__)


class PIDController:
    """Discrete PID controller with adaptive gain scheduling and dead zone."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limits: tuple[float, float] = (-30, 30),
                 adaptive: bool = False):
        self.kp_base = kp
        self.ki_base = ki
        self.kd_base = kd
        self.output_limits = output_limits
        self.adaptive = adaptive

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error: float, dead_zone: float = 0.0) -> float:
        if abs(error) < dead_zone:
            self._prev_error = error
            return 0.0

        now = time.perf_counter()
        if self._prev_time is None:
            dt = 0.033
        else:
            dt = max(now - self._prev_time, 0.001)
        self._prev_time = now

        # Adaptive gain scheduling: scale proportionally to error magnitude
        if self.adaptive:
            err_mag = abs(error)
            if err_mag > 15.0:
                gain_scale = 1.5
            elif err_mag > 5.0:
                gain_scale = 1.0
            else:
                gain_scale = 0.5
        else:
            gain_scale = 1.0

        kp = self.kp_base * gain_scale
        ki = self.ki_base * gain_scale
        kd = self.kd_base * gain_scale

        p = kp * error

        self._integral += error * dt
        self._integral = max(-100, min(100, self._integral))
        i = ki * self._integral

        d = kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p + i + d
        return max(self.output_limits[0], min(self.output_limits[1], output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None


class TargetingSystem:
    """
    Converts pixel-space target positions to turret servo angles.
    Uses adaptive PID control with dead zone and ballistic lead compensation.
    """

    def __init__(self, config: dict):
        turret_cfg = config["turret"]
        pid_cfg = turret_cfg.get("pid", {})
        tgt_cfg = config.get("targeting", {})

        self.frame_width = config["system"]["resolution"][0]
        self.frame_height = config["system"]["resolution"][1]
        self.pan_offset = turret_cfg.get("pan_offset", 90)
        self.tilt_offset = turret_cfg.get("tilt_offset", 90)
        self.pan_limits = tuple(turret_cfg.get("pan_limits", [0, 180]))
        self.tilt_limits = tuple(turret_cfg.get("tilt_limits", [30, 150]))

        self.fov_h = tgt_cfg.get("fov_h", 60.0)
        self.fov_v = tgt_cfg.get("fov_v", 45.0)
        self.dead_zone_px = tgt_cfg.get("dead_zone_px", 8)
        self.lead_blend = tgt_cfg.get("lead_blend", 0.7)
        self._on_target_threshold = tgt_cfg.get("on_target_threshold", 15.0)
        self.adaptive_pid = tgt_cfg.get("adaptive_pid", True)

        self.priority_weights = tgt_cfg.get("priority_weights", {
            "threat": 3.0, "proximity": 1.0, "confidence": 0.5,
        })

        self.pan_pid = PIDController(
            kp=pid_cfg.get("kp", 0.45),
            ki=pid_cfg.get("ki", 0.01),
            kd=pid_cfg.get("kd", 0.15),
            adaptive=self.adaptive_pid,
        )
        self.tilt_pid = PIDController(
            kp=pid_cfg.get("kp", 0.45),
            ki=pid_cfg.get("ki", 0.01),
            kd=pid_cfg.get("kd", 0.15),
            adaptive=self.adaptive_pid,
        )

        self._tracking = False
        self._lead_enabled = True

    @property
    def is_tracking(self) -> bool:
        return self._tracking

    def start_tracking(self):
        self._tracking = True
        self.pan_pid.reset()
        self.tilt_pid.reset()
        logger.info("Targeting system: tracking started")

    def stop_tracking(self):
        self._tracking = False
        self.pan_pid.reset()
        self.tilt_pid.reset()
        logger.info("Targeting system: tracking stopped")

    def set_lead_prediction(self, enabled: bool):
        self._lead_enabled = enabled

    def compute_angles(
        self,
        target_center: tuple[int, int],
        predicted_center: Optional[tuple[int, int]] = None,
        current_pan: float = 90,
        current_tilt: float = 90,
        target_speed: float = 0.0,
    ) -> tuple[float, float, bool]:
        """
        Compute pan/tilt servo angles to center on a target.

        Uses ballistic lead compensation that scales with target speed,
        adaptive PID gains, and dead zone suppression.
        """
        if not self._tracking:
            return current_pan, current_tilt, False

        aim_point = target_center
        if self._lead_enabled and predicted_center is not None:
            speed_factor = min(1.0, target_speed / 100.0) if target_speed > 2.0 else 0.0
            blend = self.lead_blend * speed_factor
            if blend > 0:
                aim_point = (
                    int((1 - blend) * target_center[0] + blend * predicted_center[0]),
                    int((1 - blend) * target_center[1] + blend * predicted_center[1]),
                )

        frame_cx = self.frame_width / 2
        frame_cy = self.frame_height / 2
        error_x = aim_point[0] - frame_cx
        error_y = aim_point[1] - frame_cy

        angle_error_pan = (error_x / self.frame_width) * self.fov_h
        angle_error_tilt = (error_y / self.frame_height) * self.fov_v

        dead_zone_angle = (self.dead_zone_px / self.frame_width) * self.fov_h

        pan_correction = self.pan_pid.compute(angle_error_pan, dead_zone=dead_zone_angle)
        tilt_correction = self.tilt_pid.compute(angle_error_tilt, dead_zone=dead_zone_angle)

        new_pan = max(self.pan_limits[0], min(self.pan_limits[1], current_pan + pan_correction))
        new_tilt = max(self.tilt_limits[0], min(self.tilt_limits[1], current_tilt + tilt_correction))

        pixel_dist = math.hypot(error_x, error_y)
        on_target = pixel_dist < self._on_target_threshold

        return new_pan, new_tilt, on_target

    def select_priority_target(self, targets: list, frame_w: int, frame_h: int):
        """
        Score and select the highest-priority target.
        Score = threat_weight * threat_score + proximity_weight / dist + conf_weight * confidence
        """
        cx, cy = frame_w / 2, frame_h / 2
        wt = self.priority_weights

        threat_map = {"hostile": 1.0, "unknown": 0.3, "safe": 0.0}

        best = None
        best_score = -1

        for t in targets:
            dist = max(1.0, math.hypot(t.center[0] - cx, t.center[1] - cy))
            threat_val = threat_map.get(t.threat_level, 0.3)

            score = (
                wt.get("threat", 3.0) * threat_val +
                wt.get("proximity", 1.0) * (1.0 / dist) * 100 +
                wt.get("confidence", 0.5) * t.confidence
            )

            if score > best_score:
                best_score = score
                best = t

        return best

    def pixel_to_angle(self, px: int, py: int) -> tuple[float, float]:
        norm_x = (px / self.frame_width) - 0.5
        norm_y = (py / self.frame_height) - 0.5

        pan = self.pan_offset + norm_x * self.fov_h
        tilt = self.tilt_offset + norm_y * self.fov_v

        return (
            max(self.pan_limits[0], min(self.pan_limits[1], pan)),
            max(self.tilt_limits[0], min(self.tilt_limits[1], tilt)),
        )

    @property
    def status(self) -> dict:
        return {
            "tracking": self._tracking,
            "lead_enabled": self._lead_enabled,
            "on_target_threshold": self._on_target_threshold,
            "dead_zone_px": self.dead_zone_px,
            "adaptive_pid": self.adaptive_pid,
            "fov": [self.fov_h, self.fov_v],
        }
