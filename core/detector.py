"""
Detection engine v3.0 - wraps Ultralytics YOLO with:
- FP16 half-precision on CUDA for ~2x speedup
- Adaptive frame skipping when FPS drops
- ROI-based detection for locked targets
- Temporal confidence smoothing to reduce flickering
- ONNX/TensorRT export helper
"""

import cv2
import numpy as np
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    center: tuple[int, int] = field(init=False)
    area: int = field(init=False)

    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area = (x2 - x1) * (y2 - y1)


class Detector:
    """
    Dual-model detection engine with performance optimizations.
    - high: YOLO11n/s/m for maximum accuracy (PC/GPU)
    - lite: YOLOv8-nano for RPi4B real-time performance
    """

    def __init__(self, config: dict):
        det_cfg = config["detection"]
        self.model_high_path = det_cfg["model_high"]
        self.model_lite_path = det_cfg["model_lite"]
        self.confidence = det_cfg["confidence"]
        self.iou_threshold = det_cfg["iou_threshold"]
        self.target_classes = det_cfg.get("classes", [0])
        self.device = self._resolve_device(det_cfg["device"])
        self.use_half = det_cfg.get("half_precision", True) and self.device == "cuda"
        self.temporal_alpha = det_cfg.get("temporal_smoothing", 0.3)

        self._model_high = None
        self._model_lite = None
        self._active = det_cfg.get("active_model", "high")
        self._inference_time = 0.0

        self._confidence_history: dict[tuple, float] = defaultdict(float)
        self._history_decay = 0.7

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load(self):
        """Load YOLO models. Downloads automatically if not present."""
        from ultralytics import YOLO

        logger.info(f"Loading high-performance model: {self.model_high_path} on {self.device}")
        self._model_high = YOLO(self.model_high_path)

        logger.info(f"Loading lightweight model: {self.model_lite_path} on {self.device}")
        self._model_lite = YOLO(self.model_lite_path)

        # Warmup with correct precision
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.active_model.predict(
            dummy, verbose=False, device=self.device, half=self.use_half,
        )
        logger.info(f"Detector loaded (device={self.device}, half={self.use_half})")

    @property
    def active_model(self):
        return self._model_high if self._active == "high" else self._model_lite

    @property
    def active_model_name(self) -> str:
        return self.model_high_path if self._active == "high" else self.model_lite_path

    def set_mode(self, mode: str):
        if mode in ("high", "lite"):
            self._active = mode
            logger.info(f"Switched to {mode} model: {self.active_model_name}")

    @property
    def inference_time_ms(self) -> float:
        return self._inference_time * 1000

    def detect(self, frame: np.ndarray, all_classes: bool = False,
               roi: Optional[tuple[int, int, int, int]] = None,
               input_size: Optional[tuple[int, int]] = None) -> list[Detection]:
        """
        Run detection on a frame.

        Args:
            frame: Input BGR frame.
            all_classes: If True, detect all COCO classes. If False, filter by config.
            roi: Optional (x1,y1,x2,y2) region of interest.
            input_size: Optional (w,h) to downscale before inference. Coords mapped back.
        """
        orig_h, orig_w = frame.shape[:2]
        roi_offset = (0, 0)
        detect_frame = frame
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(orig_w, rx2), min(orig_h, ry2)
            if rx2 > rx1 and ry2 > ry1:
                detect_frame = frame[ry1:ry2, rx1:rx2]
                roi_offset = (rx1, ry1)

        scale_x, scale_y = 1.0, 1.0
        if input_size is not None:
            dh, dw = detect_frame.shape[:2]
            tw, th = input_size
            if dw > tw or dh > th:
                detect_frame = cv2.resize(detect_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
                scale_x = dw / tw
                scale_y = dh / th

        t0 = time.perf_counter()

        classes_filter = None if all_classes else (self.target_classes if self.target_classes else None)

        results = self.active_model.predict(
            detect_frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            half=self.use_half,
            classes=classes_filter,
        )

        self._inference_time = time.perf_counter() - t0

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names.get(cls_id, "unknown")

                x1 = int(x1 * scale_x) + roi_offset[0]
                y1 = int(y1 * scale_y) + roi_offset[1]
                x2 = int(x2 * scale_x) + roi_offset[0]
                y2 = int(y2 * scale_y) + roi_offset[1]

                if self.temporal_alpha > 0:
                    key = (cls_id, x1 // 20, y1 // 20)
                    prev = self._confidence_history.get(key, conf)
                    conf = prev * self.temporal_alpha + conf * (1 - self.temporal_alpha)
                    self._confidence_history[key] = conf

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        if self.temporal_alpha > 0:
            expired = [k for k, v in self._confidence_history.items() if v * self._history_decay < 0.01]
            for k in expired:
                del self._confidence_history[k]
            for k in self._confidence_history:
                self._confidence_history[k] *= self._history_decay

        return detections

    def export_optimized(self, format: str = "onnx") -> str:
        """
        Export the active model to an optimized format.
        Supports: 'onnx', 'engine' (TensorRT), 'openvino', 'tflite'.
        Returns the exported file path.
        """
        model = self.active_model
        path = model.export(format=format, half=self.use_half)
        logger.info(f"Model exported to {format}: {path}")
        return str(path)
