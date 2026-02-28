"""
Target tracking engine v3.0 with:
- 6-state Kalman filter (position + velocity + acceleration)
- Hungarian algorithm for optimal detection-to-track assignment
- IoU + centroid hybrid cost matrix
- Track maturity (tentative -> confirmed) to reduce phantom tracks
- Deque-based trails for O(1) updates
- Appearance histogram re-identification after occlusion
"""

import cv2
import numpy as np
import logging
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class KalmanTracker:
    """
    Per-target 6-state Kalman filter: [x, y, vx, vy, ax, ay].
    Measurement: [x, y].
    """

    def __init__(self, initial_pos: tuple[int, int], process_noise: float = 0.03,
                 measurement_noise: float = 0.1):
        self.dt = 1.0

        # State: [x, y, vx, vy, ax, ay]
        self.x = np.array([
            initial_pos[0], initial_pos[1],
            0.0, 0.0,
            0.0, 0.0,
        ], dtype=np.float64)

        dt = self.dt
        dt2 = 0.5 * dt * dt
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1,  0, dt,  0],
            [0, 0, 0,  1, 0,  dt],
            [0, 0, 0,  0, 1,   0],
            [0, 0, 0,  0, 0,   1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ], dtype=np.float64)

        self.P = np.eye(6, dtype=np.float64) * 500.0

        q = process_noise
        self.Q = np.diag([q, q, q * 2, q * 2, q * 4, q * 4]).astype(np.float64)

        self.R = np.eye(2, dtype=np.float64) * measurement_noise

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: tuple[int, int]):
        z = np.array([measurement[0], measurement[1]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    def adapt_noise(self, confidence: float, speed: float):
        """Scale noise based on detection quality and target motion."""
        conf_factor = max(0.5, 2.0 - confidence)
        self.R = np.eye(2, dtype=np.float64) * (self.R[0, 0] * 0.9 + 0.1 * conf_factor * 0.1)

        speed_factor = max(1.0, speed / 50.0)
        base_q = self.Q[0, 0]
        self.Q = np.diag([
            base_q * speed_factor,
            base_q * speed_factor,
            base_q * speed_factor * 2,
            base_q * speed_factor * 2,
            base_q * speed_factor * 4,
            base_q * speed_factor * 4,
        ]).astype(np.float64)

    @property
    def position(self) -> tuple[int, int]:
        return (int(self.x[0]), int(self.x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self.x[2]), float(self.x[3]))

    @property
    def acceleration(self) -> tuple[float, float]:
        return (float(self.x[4]), float(self.x[5]))

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    def predict_future(self, steps: int) -> list[tuple[int, int]]:
        state = self.x.copy()
        positions = []
        for _ in range(steps):
            state = self.F @ state
            positions.append((int(state[0]), int(state[1])))
        return positions


@dataclass
class TrackedTarget:
    """A tracked target with persistent ID and state."""
    target_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    class_name: str
    kalman: KalmanTracker
    disappeared: int = 0
    age: int = 0
    hits: int = 0
    confirmed: bool = False
    threat_level: str = "unknown"
    face_name: Optional[str] = None
    trail: deque = field(default_factory=lambda: deque(maxlen=60))
    predicted_path: list = field(default_factory=list)
    is_primary_target: bool = False
    appearance_hist: Optional[np.ndarray] = None


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of (x1,y1,x2,y2) boxes.
    boxes_a: (N, 4), boxes_b: (M, 4) -> returns (N, M) IoU matrix."""
    xa = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
    ya = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
    xb = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
    yb = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    union = area_a + area_b - inter
    union = np.maximum(union, 1e-6)
    return inter / union


def _compute_appearance_hist(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """Compute a color histogram for re-identification."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(48, dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 3], [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


class TargetTracker:
    """
    Multi-object tracker with Hungarian assignment, IoU+centroid hybrid cost,
    track maturity, and appearance re-identification.
    """

    def __init__(self, config: dict):
        track_cfg = config["tracking"]
        self.max_disappeared = track_cfg["max_disappeared"]
        self.max_distance = track_cfg["max_distance"]
        self.prediction_steps = track_cfg["prediction_steps"]
        self.process_noise = track_cfg.get("kalman_process_noise", 0.03)
        self.measurement_noise = track_cfg.get("kalman_measurement_noise", 0.1)
        self.iou_weight = track_cfg.get("iou_weight", 0.4)
        self.min_hits = track_cfg.get("min_hits", 3)
        self._trail_length = track_cfg.get("trail_length", 60)
        self.reid_enabled = track_cfg.get("reid_enabled", True)

        self._next_id = 0
        self._targets: OrderedDict[int, TrackedTarget] = OrderedDict()
        self._primary_target_id: Optional[int] = None
        self._lost_tracks: list[TrackedTarget] = []
        self._frame_ref: Optional[np.ndarray] = None

    @property
    def targets(self) -> dict[int, TrackedTarget]:
        return {tid: t for tid, t in self._targets.items() if t.confirmed}

    @property
    def all_targets(self) -> dict[int, TrackedTarget]:
        return dict(self._targets)

    @property
    def primary_target(self) -> Optional[TrackedTarget]:
        if self._primary_target_id is not None and self._primary_target_id in self._targets:
            t = self._targets[self._primary_target_id]
            if t.confirmed:
                return t
        return None

    def set_primary_target(self, target_id: int):
        for tid, target in self._targets.items():
            target.is_primary_target = (tid == target_id)
        self._primary_target_id = target_id
        logger.info(f"Primary target set to ID {target_id}")

    def clear_primary_target(self):
        for target in self._targets.values():
            target.is_primary_target = False
        self._primary_target_id = None

    def update(self, detections: list, frame: np.ndarray = None) -> dict[int, TrackedTarget]:
        """
        Update tracker with new detections.
        Uses Hungarian algorithm with IoU+centroid hybrid cost.
        Returns dict of confirmed target_id -> TrackedTarget.
        """
        if frame is not None:
            self._frame_ref = frame

        # Age all tracks
        for target in self._targets.values():
            target.age += 1

        if len(detections) == 0:
            for target_id in list(self._targets.keys()):
                t = self._targets[target_id]
                t.disappeared += 1
                t.kalman.predict()
                t.center = t.kalman.position
                if t.disappeared > self.max_disappeared:
                    self._deregister(target_id)
            self._update_predictions()
            return self.targets

        det_centers = np.array([d.center for d in detections])
        det_bboxes = [d.bbox for d in detections]
        det_confs = [d.confidence for d in detections]
        det_classes = [d.class_name for d in detections]

        if len(self._targets) == 0:
            for i in range(len(detections)):
                self._register(det_centers[i], det_bboxes[i], det_confs[i], det_classes[i])
            self._update_predictions()
            return self.targets

        # Build hybrid cost matrix
        target_ids = list(self._targets.keys())
        target_centers = np.array([self._targets[tid].center for tid in target_ids])
        target_bboxes = [self._targets[tid].bbox for tid in target_ids]

        n_targets = len(target_ids)
        n_dets = len(detections)

        centroid_dists = cdist(target_centers, det_centers, metric="euclidean")
        centroid_cost = centroid_dists / max(self.max_distance, 1.0)

        target_boxes_arr = np.array(target_bboxes, dtype=np.float64)
        det_boxes_arr = np.array(det_bboxes, dtype=np.float64)
        iou_matrix = _compute_iou_matrix(target_boxes_arr, det_boxes_arr)
        iou_cost = 1.0 - iou_matrix

        w = self.iou_weight
        cost_matrix = (1 - w) * centroid_cost + w * iou_cost

        # Gate: set high cost for impossible associations
        gate_value = 1e5
        cost_matrix[centroid_dists > self.max_distance] = gate_value

        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        used_rows = set()
        used_cols = set()

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] >= gate_value:
                continue

            tid = target_ids[row]
            target = self._targets[tid]
            target.kalman.predict()
            target.kalman.update(tuple(det_centers[col]))
            target.kalman.adapt_noise(det_confs[col], target.kalman.speed)
            target.center = target.kalman.position
            target.bbox = det_bboxes[col]
            target.confidence = det_confs[col]
            target.class_name = det_classes[col]
            target.disappeared = 0
            target.hits += 1

            if not target.confirmed and target.hits >= self.min_hits:
                target.confirmed = True
                logger.debug(f"Target #{tid} confirmed after {target.hits} hits")

            target.trail.append(target.center)

            if self.reid_enabled and self._frame_ref is not None:
                target.appearance_hist = _compute_appearance_hist(self._frame_ref, target.bbox)

            used_rows.add(row)
            used_cols.add(col)

        # Unmatched targets
        for row in range(n_targets):
            if row not in used_rows:
                tid = target_ids[row]
                self._targets[tid].disappeared += 1
                self._targets[tid].kalman.predict()
                self._targets[tid].center = self._targets[tid].kalman.position
                if self._targets[tid].disappeared > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched detections - try re-identification first
        for col in range(n_dets):
            if col not in used_cols:
                reidentified = False
                if self.reid_enabled and self._frame_ref is not None and self._lost_tracks:
                    det_hist = _compute_appearance_hist(self._frame_ref, det_bboxes[col])
                    best_score = 0.0
                    best_idx = -1
                    for li, lost in enumerate(self._lost_tracks):
                        if lost.appearance_hist is not None:
                            score = cv2.compareHist(
                                det_hist, lost.appearance_hist, cv2.HISTCMP_CORREL
                            )
                            if score > best_score:
                                best_score = score
                                best_idx = li
                    if best_idx >= 0 and best_score > 0.6:
                        recovered = self._lost_tracks.pop(best_idx)
                        recovered.kalman.update(tuple(det_centers[col]))
                        recovered.center = recovered.kalman.position
                        recovered.bbox = det_bboxes[col]
                        recovered.confidence = det_confs[col]
                        recovered.disappeared = 0
                        recovered.hits += 1
                        recovered.trail.append(recovered.center)
                        self._targets[recovered.target_id] = recovered
                        reidentified = True
                        logger.debug(f"Re-identified target #{recovered.target_id} (score={best_score:.2f})")

                if not reidentified:
                    self._register(det_centers[col], det_bboxes[col], det_confs[col], det_classes[col])

        self._update_predictions()
        return self.targets

    def _register(self, center, bbox, confidence, class_name):
        kalman = KalmanTracker(
            initial_pos=tuple(center),
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        target = TrackedTarget(
            target_id=self._next_id,
            bbox=bbox,
            center=tuple(center),
            confidence=confidence,
            class_name=class_name,
            kalman=kalman,
            trail=deque([tuple(center)], maxlen=self._trail_length),
            hits=1,
            confirmed=(self.min_hits <= 1),
        )
        if self.reid_enabled and self._frame_ref is not None:
            target.appearance_hist = _compute_appearance_hist(self._frame_ref, bbox)

        self._targets[self._next_id] = target
        logger.debug(f"Registered new target ID {self._next_id} at {center}")
        self._next_id += 1

    def _deregister(self, target_id: int):
        if target_id == self._primary_target_id:
            self._primary_target_id = None

        target = self._targets[target_id]
        if self.reid_enabled and target.appearance_hist is not None:
            self._lost_tracks.append(target)
            if len(self._lost_tracks) > 20:
                self._lost_tracks.pop(0)

        del self._targets[target_id]
        logger.debug(f"Deregistered target ID {target_id}")

    def _update_predictions(self):
        for target in self._targets.values():
            target.predicted_path = target.kalman.predict_future(self.prediction_steps)

    def get_closest_to_center(self, frame_w: int, frame_h: int) -> Optional[TrackedTarget]:
        confirmed = self.targets
        if not confirmed:
            return None
        center = (frame_w // 2, frame_h // 2)
        return min(
            confirmed.values(),
            key=lambda t: (t.center[0] - center[0]) ** 2 + (t.center[1] - center[1]) ** 2,
        )
