"""
HUD overlay renderer v3.0 with:
- Threat-level pulse animation on hostile targets
- Animated rotating bracket lock indicator on primary target
- Mini-map radar view showing all target positions
- Zoom inset on primary target
- Compass bar showing current pan angle
- Engagement zone overlay
- Transient notification toasts
- FPS sparkline graph
- All elements configurable via config.yaml
"""

import cv2
import math
import numpy as np
import time
from collections import deque
from typing import Optional


COLORS = {
    "safe": (0, 200, 0),
    "hostile": (0, 0, 255),
    "unknown": (0, 200, 255),
    "primary": (255, 100, 0),
    "trail": (255, 180, 0),
    "prediction": (180, 0, 255),
    "gesture": (255, 255, 0),
    "reticle": (0, 255, 255),
    "hud_text": (200, 200, 200),
    "hud_accent": (0, 220, 255),
    "crosshair": (0, 255, 0),
    "radar_bg": (20, 30, 20),
    "radar_ring": (0, 80, 0),
    "compass": (0, 180, 200),
    "notification": (255, 255, 255),
}


class Notification:
    def __init__(self, text: str, duration: float = 2.0):
        self.text = text
        self.created = time.time()
        self.duration = duration

    @property
    def expired(self) -> bool:
        return time.time() - self.created > self.duration

    @property
    def alpha(self) -> float:
        elapsed = time.time() - self.created
        if elapsed > self.duration * 0.7:
            return max(0.0, 1.0 - (elapsed - self.duration * 0.7) / (self.duration * 0.3))
        return 1.0


class HUDRenderer:
    """Renders a military-style HUD overlay with configurable elements."""

    def __init__(self, config: dict):
        self.resolution = tuple(config["system"]["resolution"])
        hud_cfg = config.get("hud", {})

        self.show_crosshair = hud_cfg.get("crosshair", True)
        self.show_trails = hud_cfg.get("trails", True)
        self.show_predictions = hud_cfg.get("predictions", True)
        self.show_minimap = hud_cfg.get("minimap", True)
        self.show_zoom = hud_cfg.get("zoom_inset", True)
        self.show_compass = hud_cfg.get("compass", True)
        self.show_notifications = hud_cfg.get("notifications", True)
        self.show_fps_graph = hud_cfg.get("fps_graph", True)
        self.show_engagement = hud_cfg.get("engagement_zone", True)
        self.show_lock_anim = hud_cfg.get("target_lock_animation", True)

        self._frame_count = 0
        self._start_time = time.time()
        self._notifications: list[Notification] = []
        self._fps_history: deque = deque(maxlen=60)
        self._inf_history: deque = deque(maxlen=60)

    def add_notification(self, text: str, duration: float = 2.5):
        self._notifications.append(Notification(text, duration))
        if len(self._notifications) > 5:
            self._notifications = self._notifications[-5:]

    def render(
        self,
        frame: np.ndarray,
        targets: dict = None,
        face_results: list = None,
        gesture_results: list = None,
        turret_status: dict = None,
        targeting_status: dict = None,
        system_mode: str = "idle",
        camera_fps: float = 0,
        inference_ms: float = 0,
        armed: bool = False,
        metrics=None,
    ) -> np.ndarray:
        hud = frame
        h, w = hud.shape[:2]
        self._frame_count += 1
        self._fps_history.append(camera_fps)
        self._inf_history.append(inference_ms)

        primary_target = None
        if targets:
            for t in targets.values():
                if t.is_primary_target:
                    primary_target = t
                    break

        if self.show_crosshair:
            self._draw_crosshair(hud, w, h)

        if self.show_compass and turret_status:
            self._draw_compass(hud, w, turret_status.get("pan", 90))

        if targets:
            for tid, target in targets.items():
                self._draw_target(hud, target)

        if self.show_lock_anim and primary_target:
            self._draw_lock_indicator(hud, primary_target)

        if face_results:
            for face in face_results:
                self._draw_face(hud, face)

        if gesture_results:
            for gesture in gesture_results:
                self._draw_gesture(hud, gesture)

        self._draw_status_bar(hud, w, h, system_mode, camera_fps, inference_ms, armed)

        if turret_status:
            self._draw_turret_info(hud, w, h, turret_status)

        if targets:
            self._draw_target_count(hud, w, targets)

        if self.show_minimap and targets:
            self._draw_minimap(hud, w, h, targets, turret_status)

        if self.show_zoom and primary_target:
            self._draw_zoom_inset(hud, frame, primary_target, w, h)

        if self.show_fps_graph:
            self._draw_fps_sparkline(hud, w, h)

        if self.show_notifications:
            self._draw_notifications(hud, w, h)

        if self.show_engagement and targeting_status and targeting_status.get("tracking"):
            self._draw_engagement_zone(hud, w, h)

        return hud

    # ── Crosshair ─────────────────────────────────────────────

    def _draw_crosshair(self, frame: np.ndarray, w: int, h: int):
        cx, cy = w // 2, h // 2
        size = 20
        gap = 6
        color = COLORS["crosshair"]

        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, 1)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, 1)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, 1)
        cv2.circle(frame, (cx, cy), 2, color, -1)
        cv2.circle(frame, (cx, cy), size + 5, color, 1)

    # ── Compass Bar ───────────────────────────────────────────

    def _draw_compass(self, frame: np.ndarray, w: int, pan_angle: float):
        bar_y = 32
        bar_h = 16
        margin = 40

        roi = frame[bar_y:bar_y + bar_h, margin:w - margin]
        roi[:] = (roi * 0.5).astype(np.uint8)

        bar_w = w - 2 * margin
        color = COLORS["compass"]

        for deg in range(0, 181, 15):
            x = margin + int((deg / 180.0) * bar_w)
            tick_h = 6 if deg % 45 == 0 else 3
            cv2.line(frame, (x, bar_y + bar_h), (x, bar_y + bar_h - tick_h), color, 1)
            if deg % 45 == 0:
                cv2.putText(frame, str(deg), (x - 8, bar_y + bar_h + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        needle_x = margin + int((pan_angle / 180.0) * bar_w)
        needle_x = max(margin, min(w - margin, needle_x))
        cv2.line(frame, (needle_x, bar_y), (needle_x, bar_y + bar_h), (0, 255, 255), 2)
        pts = np.array([[needle_x - 4, bar_y], [needle_x + 4, bar_y], [needle_x, bar_y + 5]], np.int32)
        cv2.fillPoly(frame, [pts], (0, 255, 255))

    # ── Target Drawing ────────────────────────────────────────

    def _draw_target(self, frame: np.ndarray, target):
        x1, y1, x2, y2 = target.bbox
        cx, cy = target.center

        if target.is_primary_target:
            color = COLORS["primary"]
        else:
            color = COLORS.get(target.threat_level, COLORS["unknown"])

        # Pulse effect for hostile targets
        if target.threat_level == "hostile":
            pulse = 0.5 + 0.5 * math.sin(self._frame_count * 0.15)
            color = tuple(int(c * (0.6 + 0.4 * pulse)) for c in color)

        bracket_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 2 if target.is_primary_target else 1

        for (bx, by), (dx, dy) in [
            ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
            ((x1, y2), (1, -1)), ((x2, y2), (-1, -1)),
        ]:
            cv2.line(frame, (bx, by), (bx + dx * bracket_len, by), color, thickness)
            cv2.line(frame, (bx, by), (bx, by + dy * bracket_len), color, thickness)

        label = f"#{target.target_id} {target.class_name}"
        if target.face_name:
            label += f" [{target.face_name}]"
        label += f" {target.confidence:.0%}"

        label_y = y1 - 8 if y1 > 25 else y2 + 16
        cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        threat_tag = target.threat_level.upper()
        tag_color = COLORS.get(target.threat_level, COLORS["unknown"])
        cv2.putText(frame, threat_tag, (x1, label_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tag_color, 1, cv2.LINE_AA)

        if hasattr(target, "kalman"):
            speed = target.kalman.speed
            if speed > 1.0:
                spd_text = f"SPD:{speed:.0f}"
                cv2.putText(frame, spd_text, (x2 - 60, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["hud_accent"], 1, cv2.LINE_AA)

        if self.show_trails and hasattr(target, "trail") and len(target.trail) > 1:
            points = list(target.trail)
            for i in range(1, len(points)):
                alpha = i / len(points)
                t_color = tuple(int(c * alpha) for c in COLORS["trail"])
                cv2.line(frame, points[i - 1], points[i], t_color, 1, cv2.LINE_AA)

        if self.show_predictions and hasattr(target, "predicted_path") and target.predicted_path:
            prev = target.center
            for i, pt in enumerate(target.predicted_path):
                alpha = 1.0 - (i / len(target.predicted_path))
                p_color = tuple(int(c * alpha) for c in COLORS["prediction"])
                cv2.line(frame, prev, pt, p_color, 1, cv2.LINE_AA)
                if i % 3 == 0:
                    cv2.circle(frame, pt, 2, p_color, -1)
                prev = pt

        cv2.circle(frame, (cx, cy), 3, color, -1)

    # ── Lock Indicator ────────────────────────────────────────

    def _draw_lock_indicator(self, frame: np.ndarray, target):
        """Animated rotating bracket corners on the primary target."""
        x1, y1, x2, y2 = target.bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_w = (x2 - x1) // 2 + 8
        half_h = (y2 - y1) // 2 + 8
        color = COLORS["primary"]

        angle = (self._frame_count * 3) % 360
        rad = math.radians(angle)

        for corner_angle_offset in [0, 90, 180, 270]:
            a = math.radians(angle + corner_angle_offset)
            dx = int(half_w * math.cos(a))
            dy = int(half_h * math.sin(a))
            px, py = cx + dx, cy + dy

            seg_len = 6
            ax = int(seg_len * math.cos(a + math.pi / 2))
            ay = int(seg_len * math.sin(a + math.pi / 2))
            cv2.line(frame, (px - ax, py - ay), (px + ax, py + ay), color, 2, cv2.LINE_AA)

        # Pulsing outer ring
        pulse = 0.6 + 0.4 * math.sin(self._frame_count * 0.1)
        ring_color = tuple(int(c * pulse) for c in color)
        radius = int(max(half_w, half_h) * 1.2)
        cv2.ellipse(frame, (cx, cy), (int(half_w * 1.2), int(half_h * 1.2)),
                     0, 0, 360, ring_color, 1, cv2.LINE_AA)

    # ── Face Drawing ──────────────────────────────────────────

    def _draw_face(self, frame: np.ndarray, face):
        x1, y1, x2, y2 = face.bbox
        color = COLORS.get(face.threat_level, COLORS["unknown"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label = f"{face.name} ({face.threat_level.upper()})"
        if face.confidence > 0:
            label += f" {face.confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # ── Gesture Drawing ───────────────────────────────────────

    def _draw_gesture(self, frame: np.ndarray, gesture):
        color = COLORS["gesture"]
        x1, y1, x2, y2 = gesture.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        if gesture.landmarks:
            for i in range(1, len(gesture.landmarks)):
                cv2.line(frame, gesture.landmarks[i - 1], gesture.landmarks[i], color, 1)
            for pt in gesture.landmarks:
                cv2.circle(frame, pt, 2, color, -1)

        label = f"{gesture.gesture.value} -> {gesture.action}"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # ── Status Bar ────────────────────────────────────────────

    def _draw_status_bar(self, frame: np.ndarray, w: int, h: int,
                         mode: str, fps: float, inference_ms: float, armed: bool):
        roi = frame[0:28, 0:w]
        roi[:] = (roi * 0.4).astype(np.uint8)

        y = 18
        color = COLORS["hud_accent"]

        cv2.putText(frame, f"MODE: {mode.upper()}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (160, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["crosshair"], 1, cv2.LINE_AA)
        cv2.putText(frame, f"INF: {inference_ms:.0f}ms", (260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        if armed:
            cv2.putText(frame, "ARMED", (w - 80, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["hostile"], 1, cv2.LINE_AA)
            if self._frame_count % 30 < 15:
                cv2.circle(frame, (w - 90, y - 5), 5, COLORS["hostile"], -1)
        else:
            cv2.putText(frame, "SAFE", (w - 60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["safe"], 1, cv2.LINE_AA)

    # ── Turret Info ───────────────────────────────────────────

    def _draw_turret_info(self, frame: np.ndarray, w: int, h: int, turret_status: dict):
        roi = frame[h - 24:h, 0:w]
        roi[:] = (roi * 0.4).astype(np.uint8)

        y = h - 7
        pan = turret_status.get("pan", 0)
        tilt = turret_status.get("tilt", 0)
        sim = " [SIM]" if turret_status.get("simulation", True) else ""
        profile = turret_status.get("motion_profile", "")

        text = f"PAN: {pan:.0f}  TILT: {tilt:.0f}  [{profile.upper()}]{sim}"
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["hud_text"], 1, cv2.LINE_AA)

        if turret_status.get("firing", False):
            cv2.putText(frame, "** FIRING **", (w - 120, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["hostile"], 2, cv2.LINE_AA)

    # ── Target Count ──────────────────────────────────────────

    def _draw_target_count(self, frame: np.ndarray, w: int, targets: dict):
        count = len(targets)
        hostile_count = sum(1 for t in targets.values() if t.threat_level == "hostile")
        text = f"TGT: {count}"
        if hostile_count:
            text += f" ({hostile_count} HOSTILE)"
        cv2.putText(frame, text, (w - 200, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                     COLORS["hud_accent"], 1, cv2.LINE_AA)

    # ── Mini-map Radar ────────────────────────────────────────

    def _draw_minimap(self, frame: np.ndarray, w: int, h: int, targets: dict,
                      turret_status: Optional[dict]):
        radius = 50
        cx = w - radius - 12
        cy = h - radius - 32
        bg = COLORS["radar_bg"]
        ring = COLORS["radar_ring"]

        y1r = max(0, cy - radius)
        y2r = min(h, cy + radius)
        x1r = max(0, cx - radius)
        x2r = min(w, cx + radius)
        roi = frame[y1r:y2r, x1r:x2r]
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (cx - x1r, cy - y1r), radius, 255, -1)
        roi[mask > 0] = (roi[mask > 0] * 0.3).astype(np.uint8)
        bg_fill = np.full_like(roi, bg, dtype=np.uint8)
        roi[mask > 0] = cv2.add(roi[mask > 0], (bg_fill[mask > 0] * 0.7).astype(np.uint8))

        cv2.circle(frame, (cx, cy), radius, ring, 1)
        cv2.circle(frame, (cx, cy), radius // 2, ring, 1)
        cv2.line(frame, (cx - radius, cy), (cx + radius, cy), ring, 1)
        cv2.line(frame, (cx, cy - radius), (cx, cy + radius), ring, 1)

        # Sweep line
        sweep_angle = (self._frame_count * 4) % 360
        sx = int(cx + radius * math.cos(math.radians(sweep_angle)))
        sy = int(cy + radius * math.sin(math.radians(sweep_angle)))
        cv2.line(frame, (cx, cy), (sx, sy), (0, 120, 0), 1, cv2.LINE_AA)

        frame_cx = w / 2
        frame_cy = h / 2
        scale = radius / max(frame_cx, frame_cy)

        for target in targets.values():
            dx = (target.center[0] - frame_cx) * scale
            dy = (target.center[1] - frame_cy) * scale
            dist = math.hypot(dx, dy)
            if dist > radius - 2:
                factor = (radius - 2) / dist
                dx *= factor
                dy *= factor
            tx = int(cx + dx)
            ty = int(cy + dy)
            dot_color = COLORS.get(target.threat_level, COLORS["unknown"])
            dot_size = 4 if target.is_primary_target else 2
            cv2.circle(frame, (tx, ty), dot_size, dot_color, -1)

        cv2.circle(frame, (cx, cy), 2, COLORS["crosshair"], -1)

        cv2.putText(frame, "RADAR", (cx - 16, cy - radius - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, ring, 1, cv2.LINE_AA)

    # ── Zoom Inset ────────────────────────────────────────────

    def _draw_zoom_inset(self, frame: np.ndarray, original: np.ndarray,
                         target, w: int, h: int):
        x1, y1, x2, y2 = target.bbox
        oh, ow = original.shape[:2]

        pad = 20
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(ow, x2 + pad)
        ry2 = min(oh, y2 + pad)

        if rx2 <= rx1 or ry2 <= ry1:
            return

        crop = original[ry1:ry2, rx1:rx2]
        inset_size = 100
        zoomed = cv2.resize(crop, (inset_size, inset_size), interpolation=cv2.INTER_LINEAR)

        ix, iy = 10, h - inset_size - 32
        if iy < 0:
            iy = 10

        frame[iy:iy + inset_size, ix:ix + inset_size] = zoomed
        cv2.rectangle(frame, (ix, iy), (ix + inset_size, iy + inset_size), COLORS["primary"], 1)
        cv2.putText(frame, f"#{target.target_id} 2x", (ix + 4, iy + inset_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["primary"], 1, cv2.LINE_AA)

    # ── FPS Sparkline ─────────────────────────────────────────

    def _draw_fps_sparkline(self, frame: np.ndarray, w: int, h: int):
        graph_w = 80
        graph_h = 24
        gx = w - graph_w - 12
        gy = h - graph_h - 32 - 56  # above minimap area

        roi = frame[gy:gy + graph_h, gx:gx + graph_w]
        roi[:] = (roi * 0.5).astype(np.uint8)

        if len(self._fps_history) < 2:
            return

        fps_list = list(self._fps_history)
        max_val = max(max(fps_list), 1)

        for i in range(1, len(fps_list)):
            x0 = gx + int((i - 1) / len(fps_list) * graph_w)
            x1_pt = gx + int(i / len(fps_list) * graph_w)
            y0 = gy + graph_h - int(fps_list[i - 1] / max_val * (graph_h - 2))
            y1_pt = gy + graph_h - int(fps_list[i] / max_val * (graph_h - 2))
            cv2.line(frame, (x0, y0), (x1_pt, y1_pt), COLORS["crosshair"], 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS", (gx + 2, gy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, COLORS["hud_text"], 1, cv2.LINE_AA)

    # ── Notifications ─────────────────────────────────────────

    def _draw_notifications(self, frame: np.ndarray, w: int, h: int):
        self._notifications = [n for n in self._notifications if not n.expired]

        y_offset = 60
        for notif in reversed(self._notifications[-3:]):
            alpha = notif.alpha
            color = tuple(int(c * alpha) for c in COLORS["notification"])

            text_size = cv2.getTextSize(notif.text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            tx = (w - text_size[0]) // 2
            ty = y_offset

            ry1 = max(0, ty - 16)
            ry2 = min(h, ty + 6)
            rx1 = max(0, tx - 8)
            rx2 = min(w, tx + text_size[0] + 8)
            roi = frame[ry1:ry2, rx1:rx2]
            blend = 0.5 * alpha
            roi[:] = (roi * (1.0 - blend)).astype(np.uint8)

            cv2.putText(frame, notif.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 28

    # ── Engagement Zone ───────────────────────────────────────

    def _draw_engagement_zone(self, frame: np.ndarray, w: int, h: int):
        cx, cy = w // 2, h // 2
        zone_w = 60
        zone_h = 45
        cv2.rectangle(frame, (cx - zone_w, cy - zone_h), (cx + zone_w, cy + zone_h),
                       COLORS["crosshair"], 1)
