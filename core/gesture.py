"""
Hand gesture recognition v3.0 using MediaPipe Tasks API (HandLandmarker).
Compatible with mediapipe >= 0.10.30 (Python 3.13+).

Classifies gestures and maps them to system actions including
track toggle and fire trigger.
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

MP_AVAILABLE = False
try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
    )
    from mediapipe.tasks.python import BaseOptions
    from mediapipe import Image as MpImage, ImageFormat
    MP_AVAILABLE = True
except ImportError:
    logger.warning("mediapipe not installed. Gesture recognition disabled.")
except Exception as e:
    logger.warning(f"mediapipe import error: {e}. Gesture recognition disabled.")


class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    POINTING = "pointing"


@dataclass
class GestureResult:
    gesture: Gesture
    action: str
    confidence: float
    hand_label: str
    landmarks: list
    bbox: tuple[int, int, int, int]


class GestureRecognizer:
    """
    Real-time hand gesture recognition using MediaPipe Tasks API (HandLandmarker).
    Classifies finger positions to determine gesture type and maps them to actions
    like toggling tracking mode and firing.
    """

    THUMB_TIP = 4
    THUMB_IP = 3
    THUMB_MCP = 2
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    WRIST = 0

    MODEL_PATH = "data/models/hand_landmarker.task"

    def __init__(self, config: dict):
        gesture_cfg = config["gestures"]
        self.enabled = gesture_cfg.get("enabled", True) and MP_AVAILABLE
        self.min_detection_conf = gesture_cfg.get("min_detection_confidence", 0.7)
        self.min_tracking_conf = gesture_cfg.get("min_tracking_confidence", 0.5)
        self.action_map = gesture_cfg.get("actions", {})

        self._landmarker = None
        self._last_results: list[GestureResult] = []
        self._latest_mp_result: HandLandmarkerResult = None

    def start(self):
        if not self.enabled:
            logger.info("Gesture recognition disabled")
            return

        model_path = Path(self.MODEL_PATH)
        if not model_path.exists():
            logger.warning(
                f"Hand landmarker model not found at {self.MODEL_PATH}. "
                "Download from: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            )
            self.enabled = False
            return

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                num_hands=2,
                min_hand_detection_confidence=self.min_detection_conf,
                min_hand_presence_confidence=self.min_tracking_conf,
                min_tracking_confidence=self.min_tracking_conf,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            logger.info("Gesture recognizer started (MediaPipe Tasks API)")
        except Exception as e:
            logger.warning(f"Failed to initialize gesture recognizer: {e}")
            self.enabled = False
            self._landmarker = None

    def process(self, frame: np.ndarray) -> list[GestureResult]:
        if not self.enabled or self._landmarker is None:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = MpImage(image_format=ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        gesture_results = []

        if result.hand_landmarks and result.handedness:
            for hand_landmarks, handedness_list in zip(
                result.hand_landmarks, result.handedness
            ):
                landmarks_px = []
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    landmarks_px.append((px, py))

                xs = [p[0] for p in landmarks_px]
                ys = [p[1] for p in landmarks_px]
                bbox = (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)

                hand_label = handedness_list[0].category_name
                conf = handedness_list[0].score
                gesture = self._classify_gesture(landmarks_px, hand_label)
                action = self.action_map.get(gesture.value, "none")

                gesture_results.append(GestureResult(
                    gesture=gesture,
                    action=action,
                    confidence=conf,
                    hand_label=hand_label,
                    landmarks=landmarks_px,
                    bbox=bbox,
                ))

        self._last_results = gesture_results
        return gesture_results

    def _classify_gesture(self, landmarks: list[tuple[int, int]], hand_label: str) -> Gesture:
        fingers = self._get_finger_states(landmarks, hand_label)
        thumb, index, middle, ring, pinky = fingers

        if all(fingers):
            return Gesture.OPEN_PALM

        if not any(fingers):
            return Gesture.FIST

        if thumb and not index and not middle and not ring and not pinky:
            wrist_y = landmarks[self.WRIST][1]
            thumb_tip_y = landmarks[self.THUMB_TIP][1]
            if thumb_tip_y < wrist_y:
                return Gesture.THUMBS_UP
            else:
                return Gesture.THUMBS_DOWN

        if not thumb and index and middle and not ring and not pinky:
            return Gesture.PEACE

        if not thumb and index and not middle and not ring and not pinky:
            return Gesture.POINTING

        return Gesture.NONE

    def _get_finger_states(self, landmarks: list[tuple[int, int]], hand_label: str) -> list[bool]:
        if hand_label == "Right":
            thumb_extended = landmarks[self.THUMB_TIP][0] < landmarks[self.THUMB_IP][0]
        else:
            thumb_extended = landmarks[self.THUMB_TIP][0] > landmarks[self.THUMB_IP][0]

        index_extended = landmarks[self.INDEX_TIP][1] < landmarks[self.INDEX_PIP][1]
        middle_extended = landmarks[self.MIDDLE_TIP][1] < landmarks[self.MIDDLE_PIP][1]
        ring_extended = landmarks[self.RING_TIP][1] < landmarks[self.RING_PIP][1]
        pinky_extended = landmarks[self.PINKY_TIP][1] < landmarks[self.PINKY_PIP][1]

        return [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

    @property
    def last_results(self) -> list[GestureResult]:
        return self._last_results

    def stop(self):
        if self._landmarker is not None:
            self._landmarker.close()
            logger.info("Gesture recognizer stopped")
